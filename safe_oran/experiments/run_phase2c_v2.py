"""Phase 2c-v2 retrieval + compile evaluation (G1/G2/G3) on the harder CER bench.

Deterministic, no LLM. Reuses the proven verifier/solver and the v1 metric set;
swaps in the v2 corpus/samples and the true per-field CER router
(``produce_spec_v2``). Adds a per-category breakdown and an under-reservation
(unsafe-spec) signal, which is the synthetic half of the dual-signal G4
(the real-env efficiency half lives in ``run_phase2c_downstream``).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.constraints import ConstraintSpec, DeterministicSolver, Verifier
from safe_oran.envs.legacy import EnvConfig, PROJECT_ROOT
from safe_oran.experiments.run_phase2c_mini_cer import ARMS, FIELDS, MiniRetriever
from safe_oran.rag.cer_benchmark import (
    CATEGORY_COUNTS,
    build_corpus_v2,
    build_samples_v2,
    produce_spec_v2,
)

OUT_DIR = PROJECT_ROOT / "04_results" / "phase2c_v2"
FIG_DIR = PROJECT_ROOT / "05_figures" / "phase2c_v2"
CATEGORIES = tuple(CATEGORY_COUNTS.keys())
ARM_LABELS = {
    "no_retrieval": "No retrieval",
    "ordinary_rag_intent_only": "Ordinary RAG",
    "state_aware_rag": "State-aware",
    "field_aware_cer": "Field-aware CER",
    "cer_verifier_solver": "CER+verifier",
}

FIELD_ACC_KEYS = (
    "formula_accuracy",
    "reliability_target_accuracy",
    "channel_margin_policy_accuracy",
    "service_rule_accuracy",
)
MEAN_KEYS = (
    "evidence_recall_at5",
    *FIELD_ACC_KEYS,
    "spec_validity",
    "fallback",
)


def _row(arm: str, sample, spec, hits, result, executable, used_fallback, solver) -> dict[str, Any]:
    retrieved_ids = [doc.doc_id for doc, _ in hits]
    if executable is None:
        p_min = delta = under = over = None
    else:
        p_min = solver.solve(executable, sample.state).p_min
        delta = float(p_min - sample.gold_p_min)
        under = max(0.0, -delta)
        over = max(0.0, delta)
    gold_ids = set(sample.gold_evidence_ids)
    ret_ids = set(retrieved_ids[:5])
    return {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "expected_effect": sample.gold_spec.get("expected_effect", "none"),
        "arm": arm,
        "retrieved_ids": retrieved_ids,
        "gold_evidence_ids": list(sample.gold_evidence_ids),
        "evidence_recall_at5": len(gold_ids & ret_ids) / max(len(gold_ids), 1),
        "formula_accuracy": int(spec.get("formula_id") == sample.gold_spec["formula_id"]),
        "reliability_target_accuracy": int(
            float(spec.get("reliability_target", -1)) == float(sample.gold_spec["reliability_target"])
        ),
        "channel_margin_policy_accuracy": int(
            spec.get("channel_margin_policy") == sample.gold_spec["channel_margin_policy"]
        ),
        "service_rule_accuracy": int(spec.get("service_rule") == sample.gold_spec["service_rule"]),
        "spec_validity": int(result.passed),
        "verifier_reason": result.reason,
        "fallback": int(used_fallback),
        "gold_p_min": sample.gold_p_min,
        "p_min": p_min,
        "delta_p_min": delta,
        "under_reservation_prb": under,
        "over_reservation_prb": over,
        "pred_spec": spec,
        "gold_spec": sample.gold_spec,
    }


def _aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {"arm": arm, "n": len(rows)}
    for key in MEAN_KEYS:
        metrics[key] = float(np.mean([r[key] for r in rows]))
    metrics["field_accuracy_mean"] = float(np.mean([metrics[k] for k in FIELD_ACC_KEYS]))
    deltas = np.asarray([r["delta_p_min"] for r in rows if r["delta_p_min"] is not None], dtype=float)
    metrics["mean_abs_delta_p_min"] = float(np.mean(np.abs(deltas))) if deltas.size else None
    metrics["mean_delta_p_min"] = float(np.mean(deltas)) if deltas.size else None
    metrics["under_reservation_rate"] = float(np.mean(deltas < 0)) if deltas.size else None
    metrics["over_reservation_rate"] = float(np.mean(deltas > 0)) if deltas.size else None
    metrics["mean_under_reservation_prb"] = float(np.mean([r["under_reservation_prb"] or 0.0 for r in rows]))
    metrics["mean_over_reservation_prb"] = float(np.mean([r["over_reservation_prb"] or 0.0 for r in rows]))
    return metrics


def evaluate_arm_v2(arm: str, samples, retriever) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    solver = DeterministicSolver(EnvConfig())
    verifier = Verifier(require_citations=(arm == "cer_verifier_solver"))
    rows = []
    for sample in samples:
        spec, hits = produce_spec_v2(arm, sample, retriever)
        retrieved_ids = [doc.doc_id for doc, _ in hits]
        result = verifier.verify(spec, retrieved_ids, sample.state, z_mode="cer")
        used_fallback = False
        executable = result.spec
        if arm == "cer_verifier_solver" and not result.passed:
            used_fallback = True
            executable = verifier.fail_closed_spec(sample.gold_spec["reliability_target"])
        elif not result.passed:
            executable = None
        rows.append(_row(arm, sample, spec, hits, result, executable, used_fallback, solver))

    metrics = _aggregate(rows, arm)
    per_cat: dict[str, Any] = {}
    for cat in CATEGORIES:
        crows = [r for r in rows if r["category"] == cat]
        if crows:
            per_cat[cat] = _aggregate(crows, arm)
    metrics["per_category"] = per_cat
    return metrics, rows


def gate(arms: dict[str, Any]) -> dict[str, Any]:
    ordn = arms["ordinary_rag_intent_only"]
    state = arms["state_aware_rag"]
    field = arms["field_aware_cer"]
    cer = arms["cer_verifier_solver"]
    checks = {
        "field_recall_beats_ordinary": field["evidence_recall_at5"] > ordn["evidence_recall_at5"],
        "field_accuracy_monotone": ordn["field_accuracy_mean"] <= state["field_accuracy_mean"] <= field["field_accuracy_mean"],
        "field_reduces_unsafe_vs_ordinary": field["under_reservation_rate"] < ordn["under_reservation_rate"],
        "field_resolves_conflict_vs_state": (
            field["per_category"]["conflict"]["under_reservation_rate"]
            < state["per_category"]["conflict"]["under_reservation_rate"]
        ),
        "cer_verifier_valid": cer["spec_validity"] >= 0.99,
        "cer_verifier_safe": cer["under_reservation_rate"] <= field["under_reservation_rate"],
    }
    return {"checks": checks, "PASS": all(checks.values())}


def run(smoke: bool = False) -> dict[str, Any]:
    docs = build_corpus_v2()
    counts = {k: (1 if smoke else v) for k, v in CATEGORY_COUNTS.items()}
    samples = build_samples_v2(counts)
    retriever = MiniRetriever(docs)
    arms: dict[str, Any] = {}
    per_sample: list[dict[str, Any]] = []
    for arm in ARMS:
        metrics, rows = evaluate_arm_v2(arm, samples, retriever)
        arms[arm] = metrics
        per_sample.extend(rows)
    summary = {
        "schema_version": "safe_oran_phase2c_v2",
        "kind": "phase2c_v2_cer_field_eval",
        "claim_scope": (
            "deterministic harder field-level CER benchmark with generic intents "
            "(scenario only in state summary); no real LLM generation"
        ),
        "smoke": bool(smoke),
        "paper_usable": not bool(smoke),
        "n_samples": len(samples),
        "categories": list(CATEGORIES),
        "category_counts": counts,
        "arms": arms,
    }
    summary["gate"] = gate(arms)
    return {"summary": summary, "analysis": {"samples": per_sample, "docs": [d.__dict__ for d in docs]}}


def write_table(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm", "n", "evidence_recall_at5", "field_accuracy_mean",
        "channel_margin_policy_accuracy", "reliability_target_accuracy",
        "spec_validity", "fallback", "mean_abs_delta_p_min",
        "under_reservation_rate", "mean_under_reservation_prb",
        "over_reservation_rate", "mean_over_reservation_prb",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm in ARMS:
            writer.writerow({k: summary["arms"][arm].get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    result = run(smoke=args.smoke)
    out_dir = OUT_DIR / "smoke" if args.smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True))
    (out_dir / "analysis.json").write_text(json.dumps(result["analysis"], indent=2, sort_keys=True))
    write_table(result["summary"], out_dir / "cer_v2_table.csv")
    print(json.dumps({
        "summary": str(out_dir / "summary.json"),
        "n_samples": result["summary"]["n_samples"],
        "gate": result["summary"]["gate"],
    }, indent=2, sort_keys=True))
    if not result["summary"]["gate"]["PASS"]:
        raise SystemExit("Phase2c-v2 retrieval/compile gate FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
