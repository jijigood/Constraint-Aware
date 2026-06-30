"""WS-A: real-LLM symbolic constraint compiler + evidence retrieval.

The LLM reads retrieved constraint evidence and the current state, then emits a
TYPED symbolic spec (never a PRB number). The decision rules that map a scenario
to a margin policy are deliberately NOT in the prompt — they live in the
retrieved evidence, so retrieval quality is load-bearing. The deterministic
verifier/solver then turn the spec into p_min.

Runs in `~/dify_vllm_uv310/bin/python` (openai + RealRetriever). Reuses the
Phase 2a client/call pattern (`01_code/rag/constraint_producers.py`) but with a
symbolic prompt instead of the `urllc_min_prb` one (the Phase 2a NO-GO).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from safe_oran.constraints.spec import (
    ALLOWED_MARGIN_POLICIES,
    DEFAULT_FORMULA_ID,
    DEFAULT_SERVICE_RULE,
)
from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.run_phase2c_mini_cer import FIELDS, MiniRetriever
from safe_oran.rag.cer_benchmark import _FIELD_QUERY_HINT, build_corpus_v2

GEN_DIR = PROJECT_ROOT / "04_results" / "phase2c_wsa" / "generations"
BGE_PATH = "/home/huangxiaolin/models/BGE-M3"

# Retrieval arms differ only in the evidence the LLM sees; the compiler is constant.
RETRIEVAL_ARMS = ("ordinary_rag_llm", "state_aware_rag_llm", "field_aware_cer_llm")


# ----------------------------- LLM client -----------------------------
def make_client():
    import openai

    ep = os.environ.get("TRACK_A_LLM_ENDPOINT", "http://127.0.0.1:8001/v1")
    key = os.environ.get("TRACK_A_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("TRACK_A_LLM_MODEL_ID", "qwen")
    return openai.OpenAI(base_url=ep, api_key=key), model, ep


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    for candidate in (m.group(0), m.group(0).replace(",}", "}").replace(",]", "]")):
        try:
            return json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
    return None


# ----------------------------- prompt -----------------------------
SYMBOLIC_SYSTEM = (
    "You are the constraint-compilation layer of an O-RAN network-slicing safety shield. "
    "You DO NOT choose PRB numbers. You translate SLA/policy evidence and the current state "
    "into a TYPED symbolic constraint spec. A downstream deterministic solver converts your "
    "spec into the numeric URLLC PRB reservation, so you must NEVER output a PRB count."
)

_FIELD_DOC = (
    "Fields (output exactly these keys):\n"
    f'- "formula_id": always "{DEFAULT_FORMULA_ID}".\n'
    '- "reliability_target": float, one of 0.99, 0.999, 0.9999. Higher only if the evidence says so.\n'
    f'- "channel_margin_policy": one of {sorted(ALLOWED_MARGIN_POLICIES)}. This sets which channel '
    "assumption the solver uses. Choose it from the retrieved evidence and the state — the evidence "
    "states which policy applies to which radio/SLA situation. Do not guess a rule that is not in the evidence.\n"
    f'- "service_rule": always "{DEFAULT_SERVICE_RULE}".\n'
    '- "citations": list of evidence ids you actually used (use the ids in [brackets]).'
)


def build_symbolic_prompt(intent: str, state_summary: str, evidence_block: str, valid_ids: list[str]) -> str:
    ev = ""
    if evidence_block:
        ev = (
            "\n\nRetrieved constraint evidence (cite by id):\n"
            + evidence_block
            + (f"\nValid citation ids: {', '.join(valid_ids)}\n" if valid_ids else "")
        )
    return (
        f"Intent: {intent}\n"
        f"Current state: {state_summary}"
        + ev
        + "\n\n"
        + _FIELD_DOC
        + "\n\nRespond with ONE JSON object and nothing else, e.g.:\n"
        '{"formula_id": "...", "reliability_target": 0.99, "channel_margin_policy": "...", '
        '"service_rule": "...", "citations": ["..."]}'
    )


@dataclass
class CompileResult:
    spec: dict[str, Any]
    raw: str
    parse_fallback: bool = False
    retrieved_ids: list[str] = field(default_factory=list)


def _fallback_spec(retrieved_ids: list[str]) -> dict[str, Any]:
    # Schema-incomplete generation -> safe conservative default (still symbolic, never a number).
    return {
        "formula_id": DEFAULT_FORMULA_ID,
        "reliability_target": 0.99,
        "channel_margin_policy": "pessimistic_quantile",
        "service_rule": DEFAULT_SERVICE_RULE,
        "priority_rank": 1,
        "citations": [],
        "retrieved_ids": retrieved_ids,
    }


def parse_spec(raw: str, retrieved_ids: list[str]) -> CompileResult:
    obj = _extract_json(raw)
    needed = ("channel_margin_policy", "reliability_target")
    if obj is None or any(k not in obj for k in needed):
        return CompileResult(_fallback_spec(retrieved_ids), raw, parse_fallback=True, retrieved_ids=retrieved_ids)
    try:
        rel = float(obj.get("reliability_target", 0.99))
    except (TypeError, ValueError):
        rel = 0.99
    spec = {
        "formula_id": str(obj.get("formula_id", DEFAULT_FORMULA_ID)),
        "reliability_target": rel,
        "channel_margin_policy": str(obj.get("channel_margin_policy", "pessimistic_quantile")),
        "service_rule": str(obj.get("service_rule", DEFAULT_SERVICE_RULE)),
        "priority_rank": 1,
        "citations": [str(c) for c in (obj.get("citations") or [])],
        "retrieved_ids": retrieved_ids,
    }
    # Preserve a direct-numeric leak so the verifier can reject it (C3 check).
    if "urllc_min_prb" in obj:
        spec["urllc_min_prb"] = obj["urllc_min_prb"]
    return CompileResult(spec, raw, parse_fallback=False, retrieved_ids=retrieved_ids)


def call_llm(client, model: str, prompt: str, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYMBOLIC_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


# ----------------------------- evidence retrieval -----------------------------
def _block_and_ids(hits) -> tuple[str, list[str]]:
    lines, ids = [], []
    for chunk, _score in hits:
        cid = str(getattr(chunk, "chunk_id", getattr(chunk, "doc_id", "src")))
        ids.append(cid)
        text = getattr(chunk, "content", getattr(chunk, "text", ""))
        lines.append(f"[{cid}] {text[:400].replace(chr(10), ' ')}")
    return "\n".join(lines), ids


def _dedup(pairs):
    seen, out = set(), []
    for chunk, score in pairs:
        cid = str(getattr(chunk, "chunk_id", getattr(chunk, "doc_id", "src")))
        if cid not in seen:
            seen.add(cid)
            out.append((chunk, score))
    return out


def retrieve_evidence(arm: str, sample, retriever, *, top_k: int = 5) -> tuple[str, list[str], list[str]]:
    """Return (evidence_block, valid_ids, retrieved_ids) for the given arm.

    Works for both MiniRetriever (TF-IDF, supports field= boost) and RealRetriever
    (BGE-M3, query text only). Field arm = union of per-field queries.
    """
    is_tfidf = isinstance(retriever, MiniRetriever)
    if arm == "ordinary_rag_llm":
        hits = retriever.search(sample.intent, top_k=top_k)
    elif arm == "state_aware_rag_llm":
        hits = retriever.search(f"{sample.intent} {sample.state_summary}", top_k=top_k)
    elif arm == "field_aware_cer_llm":
        merged = []
        for fname in FIELDS:
            q = f"{_FIELD_QUERY_HINT[fname]} {sample.intent} {sample.state_summary}"
            if is_tfidf:
                merged += retriever.search(q, top_k=3, field=fname)
            else:
                merged += retriever.search(q, top_k=3)
        hits = _dedup(sorted(merged, key=lambda x: -x[1]))[: top_k + 1]
    else:
        raise KeyError(arm)
    block, ids = _block_and_ids(hits)
    return block, ids, ids


# ----------------------------- BGE retriever over the v2 corpus -----------------------------
@dataclass
class _Chunk:
    chunk_id: str
    content: str
    source_doc: str = "cer_v2"


def build_bge_retriever(device: str = "cpu"):
    import sys

    argo = "/home/huangxiaolin/ARGO2-main/ARGO"
    if argo not in sys.path:
        sys.path.insert(0, argo)
    from track_a_real_backend import RealRetriever  # noqa: E402

    chunks = [_Chunk(d.doc_id, d.text + " " + " ".join(d.tags)) for d in build_corpus_v2()]
    cache = PROJECT_ROOT / "04_results" / "phase2c_wsa" / "bge_cache"
    return RealRetriever(
        chunks, embedding_path=BGE_PATH, device=device, use_reranker=False, cache_dir=str(cache)
    )


# ----------------------------- generation cache -----------------------------
class GenerationCache:
    """One JSONL per (model, retriever, arm), keyed by sample_id, for cheap reruns."""

    def __init__(self, model: str, retriever_tag: str, arm: str):
        GEN_DIR.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_")
        self.path = GEN_DIR / f"{safe_model}__{retriever_tag}__{arm}.jsonl"
        self._cache: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["sample_id"]] = rec
        self.hits = 0
        self.misses = 0

    def get(self, sample_id: str) -> dict | None:
        rec = self._cache.get(sample_id)
        if rec is not None:
            self.hits += 1
        return rec

    def put(self, rec: dict) -> None:
        self.misses += 1
        self._cache[rec["sample_id"]] = rec
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
