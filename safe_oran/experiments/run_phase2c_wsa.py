"""WS-A eval: does field-aware CER's advantage survive real-LLM compilation?

For one (model, retriever): retrieve evidence per arm, compile a symbolic spec
with a real LLM, verify + solve, and score the same metric set as Phase 2c-v2
(plus citation_validity). The verifier arm reuses the field arm's generations.

Run in `~/dify_vllm_uv310/bin/python` after starting vLLM + sourcing track_a_env.sh.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from safe_oran.constraints import ConstraintSpec, DeterministicSolver, Verifier
from safe_oran.envs.legacy import EnvConfig, PROJECT_ROOT
from safe_oran.experiments.run_phase2c_mini_cer import MiniRetriever
from safe_oran.experiments.run_phase2c_v2 import CATEGORIES, _aggregate
from safe_oran.rag.cer_benchmark import CATEGORY_COUNTS, build_corpus_v2, build_samples_v2
from safe_oran.rag import cer_llm

OUT_DIR = PROJECT_ROOT / "04_results" / "phase2c_wsa"
# arm -> (generation arm it reuses, requires_citations/fail-closed)
ARMS = {
    "ordinary_rag_llm": ("ordinary_rag_llm", False),
    "state_aware_rag_llm": ("state_aware_rag_llm", False),
    "field_aware_cer_llm": ("field_aware_cer_llm", False),
    "field_aware_cer_llm_verifier": ("field_aware_cer_llm", True),
}
GEN_ARMS = ("ordinary_rag_llm", "state_aware_rag_llm", "field_aware_cer_llm")


def _generate(gen_arm, samples, retriever, client, model, tag, retr_tag) -> dict[str, dict]:
    """Produce (or load cached) LLM spec + retrieved_ids per sample for one gen arm."""
    cache = cer_llm.GenerationCache(tag, retr_tag, gen_arm)
    out: dict[str, dict] = {}
    for s in samples:
        rec = cache.get(s.sample_id)
        if rec is None:
            block, valid_ids, retrieved_ids = cer_llm.retrieve_evidence(gen_arm, s, retriever)
            prompt = cer_llm.build_symbolic_prompt(s.intent, s.state_summary, block, valid_ids)
            raw = cer_llm.call_llm(client, model, prompt)
            res = cer_llm.parse_spec(raw, retrieved_ids)
            rec = {
                "sample_id": s.sample_id,
                "retrieved_ids": retrieved_ids,
                "raw": raw,
                "spec": res.spec,
                "parse_fallback": res.parse_fallback,
            }
            cache.put(rec)
        out[s.sample_id] = rec
    return out, {"hits": cache.hits, "misses": cache.misses}


def _row(arm, sample, rec, requires_cit, verifier, solver) -> dict[str, Any]:
    spec = rec["spec"]
    retrieved_ids = rec["retrieved_ids"]
    result = verifier.verify(spec, retrieved_ids, sample.state, z_mode="cer")
    used_fallback = bool(rec.get("parse_fallback"))
    if result.passed:
        executable = result.spec
    elif requires_cit:
        executable = verifier.fail_closed_spec(float(spec.get("reliability_target", 0.99)))
        used_fallback = True
    else:
        executable = None
    if executable is None:
        p_min = delta = under = over = None
    else:
        p_min = solver.solve(executable, sample.state).p_min
        delta = float(p_min - sample.gold_p_min)
        under, over = max(0.0, -delta), max(0.0, delta)

    cites = [str(c) for c in (spec.get("citations") or [])]
    ret_set = set(retrieved_ids)
    cit_validity = (sum(c in ret_set for c in cites) / len(cites)) if cites else 0.0
    gold = sample.gold_spec
    gold_ids, top5 = set(sample.gold_evidence_ids), set(retrieved_ids[:5])
    return {
        "sample_id": sample.sample_id, "category": sample.category, "arm": arm,
        "expected_effect": gold.get("expected_effect", "none"),
        "evidence_recall_at5": len(gold_ids & top5) / max(len(gold_ids), 1),
        "formula_accuracy": int(spec.get("formula_id") == gold["formula_id"]),
        "reliability_target_accuracy": int(float(spec.get("reliability_target", -1)) == float(gold["reliability_target"])),
        "channel_margin_policy_accuracy": int(spec.get("channel_margin_policy") == gold["channel_margin_policy"]),
        "service_rule_accuracy": int(spec.get("service_rule") == gold["service_rule"]),
        "spec_validity": int(result.passed), "verifier_reason": result.reason,
        "citation_validity": cit_validity, "fallback": int(used_fallback),
        "gold_p_min": sample.gold_p_min, "p_min": p_min, "delta_p_min": delta,
        "under_reservation_prb": under, "over_reservation_prb": over,
        "pred_spec": spec,
    }


def _agg_with_citation(rows, arm) -> dict[str, Any]:
    m = _aggregate(rows, arm)
    m["citation_validity"] = float(np.mean([r["citation_validity"] for r in rows]))
    return m


def run(model: str, tag: str, retr_tag: str, smoke: bool, client) -> dict[str, Any]:
    counts = {k: (1 if smoke else v) for k, v in CATEGORY_COUNTS.items()}
    samples = build_samples_v2(counts)
    if retr_tag == "tfidf":
        retriever = MiniRetriever(build_corpus_v2())
    elif retr_tag == "bge":
        retriever = cer_llm.build_bge_retriever()
    else:
        raise KeyError(retr_tag)

    gens, cache_stats = {}, {}
    for gen_arm in GEN_ARMS:
        gens[gen_arm], cache_stats[gen_arm] = _generate(gen_arm, samples, retriever, client, model, tag, retr_tag)

    solver = DeterministicSolver(EnvConfig())
    arms: dict[str, Any] = {}
    per_sample: list[dict] = []
    for arm, (gen_arm, requires_cit) in ARMS.items():
        verifier = Verifier(require_citations=requires_cit)
        rows = [_row(arm, s, gens[gen_arm][s.sample_id], requires_cit, verifier, solver) for s in samples]
        per_sample.extend(rows)
        m = _agg_with_citation(rows, arm)
        m["per_category"] = {c: _agg_with_citation([r for r in rows if r["category"] == c], arm)
                             for c in CATEGORIES if any(r["category"] == c for r in rows)}
        arms[arm] = m

    n_gen = sum(v["hits"] + v["misses"] for v in cache_stats.values())
    summary = {
        "schema_version": "safe_oran_phase2c_wsa",
        "kind": "phase2c_wsa_real_llm_compile",
        "model": tag, "served_model": model, "retriever": retr_tag, "smoke": bool(smoke),
        "paper_usable": (not smoke) and (sum(v["misses"] for v in cache_stats.values()) >= 0),
        "evidence": {
            "endpoint": os.environ.get("TRACK_A_LLM_ENDPOINT", ""),
            "llm_model_id": model, "retriever": retr_tag,
            "n_samples": len(samples), "n_generations": n_gen,
            "cache_stats": cache_stats,
        },
        "n_samples": len(samples), "categories": list(CATEGORIES), "arms": arms,
    }
    return {"summary": summary, "analysis": {"samples": per_sample}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TRACK_A_LLM_MODEL_ID", "qwen"),
                    help="served model name sent to the API (usually 'qwen')")
    ap.add_argument("--tag", default=None, help="size label for output/cache filenames, e.g. Qwen3-14B")
    ap.add_argument("--retriever", choices=("tfidf", "bge"), default="tfidf")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    client, model_id, ep = cer_llm.make_client()
    model = args.model or model_id
    tag = args.tag or model
    result = run(model, tag, args.retriever, args.smoke, client)
    out_dir = OUT_DIR / "smoke" if args.smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    file_tag = f"{tag.replace('/', '_')}__{args.retriever}"
    (out_dir / f"summary__{file_tag}.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True))
    (out_dir / f"analysis__{file_tag}.json").write_text(json.dumps(result["analysis"], indent=2, sort_keys=True))
    a = result["summary"]["arms"]
    brief = {arm: {"margin_acc": round(a[arm]["channel_margin_policy_accuracy"], 3),
                   "under_rsv": round(a[arm]["under_reservation_rate"] or 0.0, 3),
                   "cite_valid": round(a[arm]["citation_validity"], 3),
                   "validity": round(a[arm]["spec_validity"], 3)} for arm in ARMS}
    print(json.dumps({"model": tag, "retriever": args.retriever,
                      "n_gen": result["summary"]["evidence"]["n_generations"], "arms": brief}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
