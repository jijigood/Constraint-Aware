"""Build S6 event-level z-caches for M6 RAG ablation replay.

This script reads cached WS-A real-LLM generations. It never calls an LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from safe_oran.constraints import DeterministicSolver, Verifier, fallback_spec, oracle_spec
from safe_oran.constraints.z_source import ZCache
from safe_oran.envs.factory import make_legacy_env
from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.run_phase2c_mini_cer import FIELDS
from safe_oran.rag import cer_llm
from safe_oran.rag.cer_benchmark import _FIELD_QUERY_HINT, build_samples_v2

OUT_DIR = PROJECT_ROOT / "04_results" / "phase3_m6_ablation"
SCENARIO = "S6_moderate_decay"
MODEL_DEFAULT = "Qwen3-4B"
RETRIEVER_DEFAULT = "bge"
EVENTS = (
    {"t": 0, "sla": 0.99, "sample_id": "normal_00", "trigger_type": "initial_normal"},
    {"t": 250, "sla": 0.999, "sample_id": "upgrade_00", "trigger_type": "sla_upgrade"},
)
ABLATION_ARMS = ("oracle_z", "no_retrieval_z", "ordinary_rag_z", "state_aware_rag_z", "field_CER_z")
SEEDS = (42, 43, 44)
GEN_ARM_BY_ABLATION = {
    "ordinary_rag_z": "ordinary_rag_llm",
    "state_aware_rag_z": "state_aware_rag_llm",
    "field_CER_z": "field_aware_cer_llm",
}


def _generation_path(model_tag: str, retriever: str, gen_arm: str) -> Path:
    safe_model = model_tag.replace("/", "_")
    return PROJECT_ROOT / "04_results" / "phase2c_wsa" / "generations" / f"{safe_model}__{retriever}__{gen_arm}.jsonl"


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing cached WS-A generations: {path}")
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[str(rec["sample_id"])] = rec
    return out


def _state_at(t_event: int) -> dict[str, Any]:
    env = make_legacy_env(SCENARIO, seed=100_000)
    env.reset(seed=100_000)
    for _ in range(int(t_event)):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            raise RuntimeError(f"{SCENARIO} ended before t={t_event}")
    inner = env.inner if hasattr(env, "inner") else env
    return {
        "t": int(inner.t),
        "regime": getattr(inner, "regime", SCENARIO),
        "demand": {s: float(inner._pending[s]) for s in ("embb", "urllc", "mmtc")},
        "backlog": {s: float(inner.backlog[s]) for s in ("embb", "urllc", "mmtc")},
        "channel": float(getattr(inner, "_last_channel", inner.cfg.channel_mean)),
    }


def _sanitize_spec(raw_spec: dict[str, Any]) -> dict[str, Any]:
    if "urllc_min_prb" in raw_spec:
        raise ValueError("direct numeric PRB leak in cached symbolic spec")
    return {
        "formula_id": str(raw_spec.get("formula_id", "load_backlog_over_spectral_efficiency")),
        "reliability_target": float(raw_spec.get("reliability_target", 0.99)),
        "channel_margin_policy": str(raw_spec.get("channel_margin_policy", "nominal")),
        "service_rule": str(raw_spec.get("service_rule", "serve_offered_plus_backlog")),
        "priority_rank": int(raw_spec.get("priority_rank", 1)),
        "citations": [str(x) for x in raw_spec.get("citations", [])],
        "retrieved_ids": [str(x) for x in raw_spec.get("retrieved_ids", [])],
    }


def _shared_query(arm: str, sample) -> str:
    if arm == "ordinary_rag_z":
        return sample.intent
    if arm == "state_aware_rag_z":
        return f"{sample.intent} {sample.state_summary}"
    return ""


def _field_query(field_name: str, sample) -> str:
    return f"{_FIELD_QUERY_HINT[field_name]} {sample.intent} {sample.state_summary}"


def _spec_for_arm(
    arm: str,
    event: dict[str, Any],
    sample,
    generations: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[str], bool, str]:
    if arm == "oracle_z":
        z = oracle_spec(float(event["sla"])).to_dict()
        z["channel_margin_policy"] = sample.gold_spec["channel_margin_policy"]
        return z, [], False, "oracle"
    if arm == "no_retrieval_z":
        z = fallback_spec(float(event["sla"])).to_dict()
        z["citations"] = []
        z["retrieved_ids"] = []
        return z, [], True, "no_retrieval_default"
    gen_arm = GEN_ARM_BY_ABLATION[arm]
    rec = generations[gen_arm][sample.sample_id]
    spec = _sanitize_spec(rec["spec"])
    return spec, [str(x) for x in rec.get("retrieved_ids", spec.get("retrieved_ids", []))], False, gen_arm


def build(model_tag: str, retriever: str, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = {s.sample_id: s for s in build_samples_v2()}
    generations = {
        gen_arm: _load_jsonl(_generation_path(model_tag, retriever, gen_arm))
        for gen_arm in GEN_ARM_BY_ABLATION.values()
    }
    solver = DeterministicSolver()
    verifier = Verifier(require_citations=True)
    oracle_verifier = Verifier(require_citations=False)
    oracle_cache = ZCache()
    trace_rows: list[dict[str, Any]] = []
    caches: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}

    for arm in ABLATION_ARMS:
        entries: dict[str, dict[str, Any]] = {}
        event_rows = []
        for event in EVENTS:
            sample = samples[event["sample_id"]]
            spec, retrieved_ids, force_fallback, source_arm = _spec_for_arm(arm, event, sample, generations)
            key = ZCache.make_key(SCENARIO, float(event["sla"]), int(event["t"]))
            if arm == "oracle_z":
                result = oracle_verifier.verify(spec, retrieved_ids, sample.state, z_mode="oracle")
            else:
                result = verifier.verify(spec, retrieved_ids, sample.state, z_mode="cer")
            fallback_used = bool(force_fallback or not result.passed)
            executable = verifier.fail_closed_spec(float(event["sla"])) if fallback_used else result.spec
            if executable is None:
                executable = verifier.fail_closed_spec(float(event["sla"]))
                fallback_used = True
            entries[key] = executable.to_dict()

            state = _state_at(int(event["t"]))
            oracle_z = oracle_cache.get(SCENARIO, int(event["t"]), float(event["sla"]))
            p_oracle = solver.solve(oracle_z, state).p_min
            p_pred = solver.solve(entries[key], state).p_min
            delta = int(p_pred) - int(p_oracle)
            event_rows.append({
                "t_event": int(event["t"]),
                "sample_id": sample.sample_id,
                "trigger_type": event["trigger_type"],
                "verifier_passed": bool(result.passed),
                "fallback_used": fallback_used,
                "p_min_pred": int(p_pred),
                "p_min_oracle": int(p_oracle),
                "delta_p_min": int(delta),
                "under_reservation": max(0, -int(delta)),
                "over_reservation": max(0, int(delta)),
            })

            for seed in seeds:
                for field_name in FIELDS:
                    if arm == "field_CER_z":
                        query = _field_query(field_name, sample)
                    else:
                        query = _shared_query(arm, sample)
                    pred_value = entries[key].get(field_name)
                    gold_value = sample.gold_spec.get(field_name)
                    trace_rows.append({
                        "scenario": SCENARIO,
                        "seed": int(seed),
                        "t_event": int(event["t"]),
                        "trigger_type": event["trigger_type"],
                        "retrieval_arm": arm,
                        "field_name": field_name,
                        "query": query,
                        "topk_ids": " ".join(retrieved_ids),
                        "citations": " ".join(spec.get("citations", [])),
                        "gold_value": gold_value,
                        "predicted_value": pred_value,
                        "field_correct": str(pred_value == gold_value).lower(),
                        "p_min_pred": int(p_pred),
                        "p_min_oracle": int(p_oracle),
                        "delta_p_min": int(delta),
                        "under_reservation": max(0, -int(delta)),
                        "over_reservation": max(0, int(delta)),
                        "verifier_passed": str(result.passed).lower(),
                        "fallback_used": str(fallback_used).lower(),
                        "source_generation_arm": source_arm,
                    })

        parity = sum(int(r["delta_p_min"] == 0) for r in event_rows) / max(len(event_rows), 1)
        unsafe = sum(int(r["delta_p_min"] < 0) for r in event_rows) / max(len(event_rows), 1)
        fallback_rate = sum(int(r["fallback_used"]) for r in event_rows) / max(len(event_rows), 1)
        caches[arm] = entries
        summaries[arm] = {
            "events": event_rows,
            "p_min_parity_rate": float(parity),
            "unsafe_under_reservation_rate": float(unsafe),
            "fallback_rate": float(fallback_rate),
        }
        (OUT_DIR / f"z_cache__{arm}.json").write_text(json.dumps(entries, indent=2, sort_keys=True))

    trace_path = OUT_DIR / "m6_event_trace.csv"
    with trace_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(trace_rows)

    summary = {
        "schema_version": "safe_oran_m6_rag_ablation_caches",
        "kind": "m6_rag_ablation_event_caches",
        "scenario": SCENARIO,
        "model_tag": model_tag,
        "retriever": retriever,
        "arms": summaries,
        "cache_paths": {arm: str(OUT_DIR / f"z_cache__{arm}.json") for arm in ABLATION_ARMS},
        "event_trace": str(trace_path),
        "seeds": list(seeds),
    }
    (OUT_DIR / "cache_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default=MODEL_DEFAULT)
    ap.add_argument("--retriever", default=RETRIEVER_DEFAULT, choices=("bge", "tfidf"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()
    summary = build(args.model_tag, args.retriever, tuple(int(s) for s in args.seeds))
    print(json.dumps({
        "cache_paths": summary["cache_paths"],
        "event_trace": summary["event_trace"],
        "arms": summary["arms"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
