"""
Phase 2a constraint producers. Each maps a logged network state -> ConstraintOutput with an executable
`urllc_min_prb` that plugs straight into the env's `_project_to_min_urllc` shield.

  static        : the fixed Phase-0 floor (no LLM)
  oracle_margin : load/SLA-aware oracle reservation (the safety upper bound / gold; no LLM)
  llm_no_rag    : Qwen3-14B emits the reservation from the state summary alone
  rag_llm       : Qwen3-14B emits it from the state summary + retrieved O-RAN/3GPP SLA snippets

The state summary gives the LOAD side (demand/backlog/per-PRB URLLC capacity); RETRIEVAL supplies the
SLA/reliability side (how aggressively to protect URLLC). Heavy deps (openai, RealRetriever) are
lazy-imported so the static/oracle/parse paths run in any venv.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(RAG_DIR), "env"))
sys.path.insert(0, RAG_DIR)
ARGO = "/home/huangxiaolin/ARGO2-main/ARGO"
if ARGO not in sys.path:
    sys.path.insert(0, ARGO)

from slicing_env import EnvConfig  # noqa: E402
from scoring_credibility import oracle_reservation  # reuse the tested env-math  # noqa: E402

CFG = EnvConfig()


@dataclass
class ConstraintOutput:
    urllc_min_prb: int
    reliability_target: float = 0.0
    reason: str = ""
    citations: list = field(default_factory=list)
    raw: str = ""
    schema_ok: bool = True
    parse_fallback: bool = False


def clamp_snap(prb, cfg=CFG) -> int:
    prb = max(0, min(cfg.n_prb, int(round(prb))))
    return int(round(prb / cfg.prb_step) * cfg.prb_step)


# ---------------- JSON parse (reuse ARGO's robust extractor) ----------------
def _extract_json(text):
    try:
        from route_b_diag_task import _extract_json as argo_extract
        return argo_extract(text)
    except Exception:
        import re
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            try:
                return json.loads(m.group(0).replace(",}", "}").replace(",]", "]"))
            except Exception:
                return None


CONSTRAINT_KEYS = ("urllc_min_prb", "reliability_target", "reason", "citations")


def schema_compliance_constraint(obj) -> dict:
    if obj is None:
        return {"valid_json": False, "missing": list(CONSTRAINT_KEYS), "compliant": False}
    missing = [k for k in CONSTRAINT_KEYS if k not in obj]
    type_err = []
    if "urllc_min_prb" in obj:
        try:
            int(obj["urllc_min_prb"])
        except (ValueError, TypeError):
            type_err.append("urllc_min_prb")
    return {"valid_json": True, "missing": missing, "type_err": type_err,
            "compliant": not missing and not type_err}


# ---------------- prompt building ----------------
def build_state_summary(state) -> str:
    d, b, g = state["demand"], state["backlog"], float(state["channel"])
    cap = CFG.se["urllc"] * g
    return (
        "You are the safety layer of an O-RAN network-slicing scheduler.\n"
        f"Total PRBs = {CFG.n_prb}, allocated in steps of {CFG.prb_step} across three slices "
        "(eMBB, URLLC, mMTC).\n"
        f"Current offered load (Mbps): eMBB={d['embb']:.1f}, URLLC={d['urllc']:.1f}, mMTC={d['mmtc']:.1f}.\n"
        f"URLLC backlog (Mbps): {b['urllc']:.1f}.\n"
        f"Radio this slot: URLLC per-PRB capacity ~= {cap:.2f} Mbps/PRB (channel gain {g:.2f}).\n"
        "URLLC has a strict latency SLA: its offered load+backlog must be served THIS slot, or the SLA "
        "is violated. Channel can dip slot-to-slot, so a reliability margin above the nominal need is "
        "prudent. Reserving too many PRBs for URLLC starves eMBB/mMTC and wastes capacity.\n"
        "TASK: choose the minimum PRBs to RESERVE for URLLC (a floor the scheduler must honor)."
    )


def build_constraint_prompt(summary: str, evidence_block: str = "", valid_ids=None) -> str:
    ev = ""
    if evidence_block:
        ev = ("\n\nRetrieved O-RAN/3GPP SLA evidence (cite by id):\n" + evidence_block
              + (f"\nValid citation ids: {', '.join(valid_ids)}\n" if valid_ids else ""))
    return (
        summary + ev +
        "\n\nRespond with ONE JSON object and nothing else:\n"
        '{"urllc_min_prb": <int 0-100, multiple of 10>, '
        '"reliability_target": <float 0-1>, '
        '"reason": <short string>, '
        '"citations": [<ids you used; [] if none>]}'
    )


# ---------------- producers ----------------
class StaticProducer:
    name = "static"
    def produce(self, state) -> ConstraintOutput:
        return ConstraintOutput(urllc_min_prb=int(state["static_floor"]), reliability_target=0.0,
                                reason="fixed static floor", citations=[])


class OracleMarginProducer:
    name = "oracle_margin"
    def __init__(self, reliability=0.99):
        self.reliability = reliability
    def produce(self, state) -> ConstraintOutput:
        return ConstraintOutput(urllc_min_prb=oracle_reservation(state, reliability=self.reliability),
                                reliability_target=self.reliability,
                                reason="load-aware oracle (pessimistic channel)", citations=[])


class _LLMBase:
    def __init__(self, client, model_id, max_tokens=400):
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.calls = 0

    def _call(self, prompt) -> str:
        self.calls += 1
        resp = self.client.chat.completions.create(
            model=self.model_id, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return resp.choices[0].message.content or ""

    def _parse(self, text, state, valid_ids=None) -> ConstraintOutput:
        obj = _extract_json(text)
        sc = schema_compliance_constraint(obj)
        if not sc["compliant"]:
            return ConstraintOutput(urllc_min_prb=int(state["static_floor"]), reliability_target=0.0,
                                    reason="parse/schema fallback->static", citations=[], raw=text,
                                    schema_ok=False, parse_fallback=True)
        cites = obj.get("citations") or []
        if valid_ids is not None:
            cites = [c for c in cites]  # kept raw; validity scored separately in run_gate
        try:
            rel = float(obj.get("reliability_target", 0.0))
        except (ValueError, TypeError):
            rel = 0.0
        return ConstraintOutput(urllc_min_prb=clamp_snap(obj["urllc_min_prb"]),
                                reliability_target=rel, reason=str(obj.get("reason", ""))[:300],
                                citations=cites, raw=text, schema_ok=True)


class LLMNoRAGProducer(_LLMBase):
    name = "llm_no_rag"
    def produce(self, state) -> ConstraintOutput:
        prompt = build_constraint_prompt(build_state_summary(state))
        return self._parse(self._call(prompt), state)


class RAGLLMProducer(_LLMBase):
    name = "rag_llm"
    QUERY = ("URLLC ultra-reliable low-latency 5G/O-RAN network slice reliability target and latency "
             "SLA; how much radio resource margin to reserve for URLLC vs eMBB mMTC slices")

    def __init__(self, client, model_id, retriever, top_k=5, **kw):
        super().__init__(client, model_id, **kw)
        self.retriever = retriever
        self.top_k = top_k
        self._block = None        # constant query -> retrieve ONCE, reuse for all states
        self._valid_ids = None
        self.retrieval_calls = 0

    def _ensure_evidence(self):
        if self._block is not None:
            return
        hits = self.retriever.search(self.QUERY, top_k=self.top_k)
        self.retrieval_calls += 1
        lines, ids = [], []
        for chunk, score in hits:
            cid = str(getattr(chunk, "chunk_id", getattr(chunk, "source_doc", "src")))
            ids.append(cid)
            lines.append(f"[{cid}] {getattr(chunk, 'content', '')[:400].replace(chr(10), ' ')}")
        self._block = "\n".join(lines)
        self._valid_ids = ids

    def produce(self, state) -> ConstraintOutput:
        self._ensure_evidence()
        prompt = build_constraint_prompt(build_state_summary(state), self._block, self._valid_ids)
        out = self._parse(self._call(prompt), state, valid_ids=self._valid_ids)
        out._valid_ids = list(self._valid_ids)  # stash for citation-validity scoring in run_gate
        return out


# ---------------- offline self-test (no LLM/GPU): static + oracle + parser ----------------
if __name__ == "__main__":
    demo = {"demand": {"embb": 80.0, "urllc": 40.0, "mmtc": 12.0},
            "backlog": {"embb": 0.0, "urllc": 5.0, "mmtc": 0.0},
            "channel": 1.0, "static_floor": 50, "logged_action_idx": 0}
    print("state summary:\n", build_state_summary(demo))
    print("\nstatic ->", StaticProducer().produce(demo).urllc_min_prb)
    print("oracle ->", OracleMarginProducer().produce(demo).urllc_min_prb)
    good = '{"urllc_min_prb": 80, "reliability_target": 0.99999, "reason": "URLLC needs ~75 PRB + margin", "citations": ["3gpp_28914"]}'
    bad = "I think you should reserve about 80 PRBs for URLLC."
    base = _LLMBase(None, "x")
    print("parse good ->", base._parse(good, demo).__dict__)
    print("parse bad  ->", base._parse(bad, demo).__dict__)
    assert base._parse(good, demo).urllc_min_prb == 80
    assert base._parse(bad, demo).parse_fallback is True
    print("\nproducer parse self-test PASS")
