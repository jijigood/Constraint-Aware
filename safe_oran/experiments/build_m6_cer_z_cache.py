"""Build the event-level CER-z cache used by the M6 closed-loop controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safe_oran.constraints import DeterministicSolver, Verifier
from safe_oran.constraints.z_source import ZCache
from safe_oran.envs.factory import make_legacy_env
from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.rag.cer_benchmark import build_samples_v2

OUT_DIR = PROJECT_ROOT / "04_results" / "phase3_m6"
SCENARIO = "S6_moderate_decay"
EVENTS = (
    {"t": 0, "sla": 0.99, "sample_id": "normal_00"},
    {"t": 250, "sla": 0.999, "sample_id": "upgrade_00"},
)


def _generation_path(model_tag: str, retriever: str) -> Path:
    safe_model = model_tag.replace("/", "_")
    return PROJECT_ROOT / "04_results" / "phase2c_wsa" / "generations" / (
        f"{safe_model}__{retriever}__field_aware_cer_llm.jsonl"
    )


def _load_generation_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing WS-A generation cache: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records[str(rec["sample_id"])] = rec
    return records


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


def _sanitized_spec(rec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(rec["spec"])
    if "urllc_min_prb" in spec:
        raise ValueError(f"direct numeric PRB leak in {rec['sample_id']}")
    spec["retrieved_ids"] = [str(x) for x in rec.get("retrieved_ids", spec.get("retrieved_ids", []))]
    spec["citations"] = [str(x) for x in spec.get("citations", [])]
    return {
        "formula_id": spec["formula_id"],
        "reliability_target": float(spec["reliability_target"]),
        "channel_margin_policy": spec["channel_margin_policy"],
        "service_rule": spec["service_rule"],
        "priority_rank": int(spec.get("priority_rank", 1)),
        "citations": spec["citations"],
        "retrieved_ids": spec["retrieved_ids"],
    }


def build_cache(model_tag: str, retriever: str) -> dict[str, Any]:
    records = _load_generation_cache(_generation_path(model_tag, retriever))
    sample_states = {s.sample_id: s.state for s in build_samples_v2()}
    verifier = Verifier(require_citations=True)
    solver = DeterministicSolver()
    oracle_cache = ZCache()
    entries: dict[str, dict[str, Any]] = {}
    event_rows = []

    for event in EVENTS:
        sample_id = event["sample_id"]
        if sample_id not in records:
            raise KeyError(f"{sample_id} not found in WS-A generation cache")
        spec = _sanitized_spec(records[sample_id])
        result = verifier.verify(spec, spec["retrieved_ids"], sample_states[sample_id], z_mode="cer")
        if not result.passed:
            raise RuntimeError(f"{sample_id} failed M6 verifier: {result.reason}")

        key = ZCache.make_key(SCENARIO, float(event["sla"]), int(event["t"]))
        entries[key] = result.spec.to_dict() if result.spec is not None else spec

        state = _state_at(int(event["t"]))
        oracle_z = oracle_cache.get(SCENARIO, int(event["t"]), float(event["sla"]))
        oracle_p = solver.solve(oracle_z, state).p_min
        cer_p = solver.solve(entries[key], state).p_min
        delta = int(cer_p) - int(oracle_p)
        event_rows.append({
            "event_t": int(event["t"]),
            "sla": float(event["sla"]),
            "sample_id": sample_id,
            "cache_key": key,
            "channel_margin_policy": entries[key]["channel_margin_policy"],
            "reliability_target": entries[key]["reliability_target"],
            "citations": entries[key]["citations"],
            "retrieved_ids": entries[key]["retrieved_ids"],
            "verifier_passed": True,
            "oracle_p_min": int(oracle_p),
            "cer_p_min": int(cer_p),
            "delta_p_min_vs_oracle": int(delta),
            "unsafe_under_reservation": bool(delta < 0),
        })

    parity = sum(int(r["delta_p_min_vs_oracle"] == 0) for r in event_rows) / len(event_rows)
    unsafe = sum(int(r["unsafe_under_reservation"]) for r in event_rows) / len(event_rows)
    summary = {
        "schema_version": "safe_oran_phase3_m6_z_cache",
        "kind": "m6_event_level_cer_z_cache",
        "scenario": SCENARIO,
        "model_tag": model_tag,
        "retriever": retriever,
        "source_generation_cache": str(_generation_path(model_tag, retriever)),
        "events": event_rows,
        "gate": {
            "n_events": len(event_rows),
            "verifier_pass_rate": 1.0,
            "p_min_parity_rate": float(parity),
            "unsafe_under_reservation_rate": float(unsafe),
            "passed": bool(parity >= 1.0 and unsafe <= 0.0),
        },
    }
    return {"entries": entries, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="Qwen3-4B")
    ap.add_argument("--retriever", default="bge", choices=("bge", "tfidf"))
    args = ap.parse_args()

    result = build_cache(args.model_tag, args.retriever)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"cer_z_cache__{args.model_tag.replace('/', '_')}__{args.retriever}"
    cache_path = OUT_DIR / f"{stem}.json"
    summary_path = OUT_DIR / f"{stem}_summary.json"
    cache_path.write_text(json.dumps(result["entries"], indent=2, sort_keys=True))
    summary_path.write_text(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps({
        "cache": str(cache_path),
        "summary": str(summary_path),
        "gate": result["summary"]["gate"],
        "events": result["summary"]["events"],
    }, indent=2, sort_keys=True))
    return 0 if result["summary"]["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
