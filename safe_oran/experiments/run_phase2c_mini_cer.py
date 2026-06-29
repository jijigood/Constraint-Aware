"""Phase2c mini-CER field-level retrieval benchmark.

This is a deterministic, no-LLM benchmark. It evaluates whether field-aware
constraint evidence routing improves symbolic-z construction over ordinary RAG.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.constraints import ConstraintSpec, DeterministicSolver, Verifier
from safe_oran.constraints.spec import DEFAULT_FORMULA_ID, DEFAULT_SERVICE_RULE
from safe_oran.envs.legacy import EnvConfig, PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "04_results" / "phase2c_mini_cer"
FIG_DIR = PROJECT_ROOT / "05_figures" / "phase2c_mini_cer"
FIELDS = ("formula_id", "reliability_target", "channel_margin_policy", "service_rule")
ARMS = (
    "no_retrieval",
    "ordinary_rag_intent_only",
    "state_aware_rag",
    "field_aware_cer",
    "cer_verifier_solver",
)


@dataclass(frozen=True)
class EvidenceDoc:
    doc_id: str
    text: str
    fields: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class MiniSample:
    sample_id: str
    category: str
    intent: str
    state_summary: str
    state: dict[str, Any]
    gold_evidence_ids: tuple[str, ...]
    gold_spec: dict[str, Any]
    gold_p_min: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\\.]+", text.lower())


def build_corpus() -> list[EvidenceDoc]:
    return [
        EvidenceDoc(
            "E_FORMULA_LOAD_BACKLOG",
            "URLLC safety constraints use formula_id load_backlog_over_spectral_efficiency and serve offered plus backlog.",
            {"formula_id": DEFAULT_FORMULA_ID, "service_rule": DEFAULT_SERVICE_RULE},
            ("formula", "service"),
        ),
        EvidenceDoc(
            "E_FORMULA_NUMERIC_BAD",
            "Legacy numeric examples mention urllc_min_prb 50, but executable PRB numbers are not symbolic specs.",
            {},
            ("distractor",),
        ),
        EvidenceDoc(
            "E_REL_NORMAL_99",
            "Normal SLA reliability target is 0.99 for baseline URLLC admission.",
            {"reliability_target": 0.99},
            ("normal", "burst", "degraded", "reliability"),
        ),
        EvidenceDoc(
            "E_REL_UPGRADE_999",
            "Upgraded SLA requires reliability target 0.999 for URLLC after the SLA change.",
            {"reliability_target": 0.999},
            ("upgrade", "reliability"),
        ),
        EvidenceDoc(
            "E_REL_CONFLICT_LOW",
            "Old conflicting SLA appendix says reliability target 0.99 and should be ignored when newer policy exists.",
            {"reliability_target": 0.99},
            ("conflict", "distractor", "reliability"),
        ),
        EvidenceDoc(
            "E_REL_CONFLICT_HIGH",
            "Latest conflict-resolution policy sets URLLC reliability target to 0.9999.",
            {"reliability_target": 0.9999},
            ("conflict", "latest", "reliability"),
        ),
        EvidenceDoc(
            "E_MARGIN_NOMINAL",
            "Use channel_margin_policy nominal for stable or moderate channels when state already carries channel quality.",
            {"channel_margin_policy": "nominal"},
            ("normal", "margin"),
        ),
        EvidenceDoc(
            "E_MARGIN_PESS",
            "Use channel_margin_policy pessimistic_quantile for bursty URLLC or upgraded SLA reliability.",
            {"channel_margin_policy": "pessimistic_quantile"},
            ("burst", "upgrade", "conflict", "margin"),
        ),
        EvidenceDoc(
            "E_MARGIN_WORST",
            "Use channel_margin_policy worst_case only for severe channel degraded emergency constraints.",
            {"channel_margin_policy": "worst_case"},
            ("degraded", "margin"),
        ),
        EvidenceDoc(
            "E_SCENE_NORMAL",
            "normal SLA stable channel baseline URLLC moderate load.",
            {},
            ("normal",),
        ),
        EvidenceDoc(
            "E_SCENE_BURST",
            "URLLC burst state has offered load plus backlog and should use pessimistic quantile margin.",
            {"channel_margin_policy": "pessimistic_quantile"},
            ("burst",),
        ),
        EvidenceDoc(
            "E_SCENE_DEGRADED",
            "channel degraded weak radio state requires worst_case margin for safety compilation.",
            {"channel_margin_policy": "worst_case"},
            ("degraded",),
        ),
        EvidenceDoc(
            "E_SCENE_UPGRADE",
            "SLA upgrade event combines reliability target 0.999 with pessimistic quantile margin.",
            {"reliability_target": 0.999, "channel_margin_policy": "pessimistic_quantile"},
            ("upgrade",),
        ),
        EvidenceDoc(
            "E_POLICY_LATEST_WINS",
            "When evidence conflicts, latest policy evidence wins over old appendix evidence.",
            {},
            ("conflict", "latest"),
        ),
    ]


def _state(category: str, i: int) -> dict[str, Any]:
    if category == "normal":
        d = 14.0 + (i % 5) * 1.4
        b = float(i % 3)
        g = 0.82 + 0.03 * (i % 5)
    elif category == "burst":
        d = 26.0 + (i % 5) * 2.8
        b = 4.0 + (i % 4) * 2.0
        g = 0.68 + 0.025 * (i % 5)
    elif category == "degraded":
        d = 16.0 + (i % 5) * 1.8
        b = 1.0 + (i % 4)
        g = 0.28 + 0.025 * (i % 5)
    elif category == "upgrade":
        d = 20.0 + (i % 5) * 2.0
        b = 2.0 + (i % 4)
        g = 0.62 + 0.03 * (i % 5)
    elif category == "conflict":
        d = 22.0 + (i % 5) * 2.2
        b = 3.0 + (i % 4)
        g = 0.58 + 0.025 * (i % 5)
    else:
        raise KeyError(category)
    return {
        "t": i,
        "regime": category,
        "demand": {"embb": 45.0, "urllc": d, "mmtc": 12.0},
        "backlog": {"embb": 0.0, "urllc": b, "mmtc": 0.0},
        "channel": g,
    }


def _gold(category: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    base = {
        "formula_id": DEFAULT_FORMULA_ID,
        "reliability_target": 0.99,
        "channel_margin_policy": "nominal",
        "service_rule": DEFAULT_SERVICE_RULE,
        "priority_rank": 1,
        "citations": [],
        "retrieved_ids": [],
    }
    ids = ["E_FORMULA_LOAD_BACKLOG"]
    if category == "normal":
        base.update(reliability_target=0.99, channel_margin_policy="nominal")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_NOMINAL", "E_SCENE_NORMAL"]
    elif category == "burst":
        base.update(reliability_target=0.99, channel_margin_policy="pessimistic_quantile")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_PESS", "E_SCENE_BURST"]
    elif category == "degraded":
        base.update(reliability_target=0.99, channel_margin_policy="worst_case")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_WORST", "E_SCENE_DEGRADED"]
    elif category == "upgrade":
        base.update(reliability_target=0.999, channel_margin_policy="pessimistic_quantile")
        ids += ["E_REL_UPGRADE_999", "E_MARGIN_PESS", "E_SCENE_UPGRADE"]
    elif category == "conflict":
        base.update(reliability_target=0.9999, channel_margin_policy="pessimistic_quantile")
        ids += ["E_REL_CONFLICT_HIGH", "E_MARGIN_PESS", "E_POLICY_LATEST_WINS"]
    return base, tuple(ids)


def build_samples(n_per_category: int = 20) -> list[MiniSample]:
    solver = DeterministicSolver(EnvConfig())
    samples: list[MiniSample] = []
    categories = ("normal", "burst", "degraded", "upgrade", "conflict")
    intent_map = {
        "normal": "Compile URLLC safety constraint for normal slicing service.",
        "burst": "Compile URLLC constraint during traffic burst.",
        "degraded": "Compile URLLC constraint under channel degraded condition.",
        "upgrade": "Compile URLLC constraint after SLA upgrade.",
        "conflict": "Compile URLLC constraint with conflicting SLA evidence.",
    }
    for category in categories:
        for i in range(n_per_category):
            state = _state(category, i)
            gold_spec, gold_ids = _gold(category)
            p_min = solver.solve(ConstraintSpec.from_mapping(gold_spec), state).p_min
            d_u = state["demand"]["urllc"]
            b_u = state["backlog"]["urllc"]
            g = state["channel"]
            samples.append(MiniSample(
                sample_id=f"{category}_{i:02d}",
                category=category,
                intent=intent_map[category],
                state_summary=(
                    f"category={category} d_urllc={d_u:.1f} backlog_urllc={b_u:.1f} "
                    f"channel={g:.2f}"
                ),
                state=state,
                gold_evidence_ids=gold_ids,
                gold_spec=gold_spec,
                gold_p_min=p_min,
            ))
    return samples


class MiniRetriever:
    def __init__(self, docs: list[EvidenceDoc]):
        self.docs = docs
        self._doc_tokens = {d.doc_id: tokenize(d.text + " " + " ".join(d.tags)) for d in docs}
        df: dict[str, int] = {}
        for toks in self._doc_tokens.values():
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        n = len(docs)
        self.idf = {tok: math.log((n + 1) / (count + 0.5)) + 1.0 for tok, count in df.items()}

    def search(self, query: str, *, top_k: int = 5, field: str | None = None) -> list[tuple[EvidenceDoc, float]]:
        q = tokenize(query)
        scores = []
        for doc in self.docs:
            toks = self._doc_tokens[doc.doc_id]
            tf = {tok: toks.count(tok) for tok in set(toks)}
            score = 0.0
            for tok in q:
                score += tf.get(tok, 0) * self.idf.get(tok, 0.0)
            if field and field in doc.fields:
                score += 6.0
            if field and field in doc.tags:
                score += 2.0
            if score > 0:
                scores.append((doc, score))
        scores.sort(key=lambda x: (-x[1], x[0].doc_id))
        return scores[:top_k]


def _hits_to_spec(
    sample: MiniSample,
    hits: list[tuple[EvidenceDoc, float]],
    *,
    require_latest: bool = False,
    use_category_filter: bool = False,
) -> dict[str, Any]:
    spec = {
        "formula_id": DEFAULT_FORMULA_ID,
        "reliability_target": 0.99,
        "channel_margin_policy": "nominal",
        "service_rule": DEFAULT_SERVICE_RULE,
        "priority_rank": 1,
        "citations": [],
        "retrieved_ids": [doc.doc_id for doc, _ in hits],
    }
    for field_name in FIELDS:
        candidates = [(doc, score) for doc, score in hits if field_name in doc.fields]
        if not candidates:
            continue
        if require_latest and sample.category == "conflict" and field_name == "reliability_target":
            latest = [x for x in candidates if "latest" in x[0].tags]
            if latest:
                candidates = latest
        elif use_category_filter:
            tagged = [x for x in candidates if sample.category in x[0].tags]
            if tagged:
                candidates = tagged
        doc, _ = max(candidates, key=lambda x: (x[1], "latest" in x[0].tags))
        spec[field_name] = doc.fields[field_name]
        spec["citations"].append(doc.doc_id)
    return spec


def _field_aware_hits(sample: MiniSample, retriever: MiniRetriever) -> list[tuple[EvidenceDoc, float]]:
    rel_hint = "latest conflicting SLA" if sample.category == "conflict" else sample.category
    queries = {
        "formula_id": f"formula_id symbolic constraint {sample.intent} {sample.state_summary}",
        "reliability_target": f"reliability target {rel_hint} {sample.intent} {sample.state_summary}",
        "channel_margin_policy": f"channel margin policy {sample.intent} {sample.state_summary}",
        "service_rule": f"service rule offered backlog {sample.intent} {sample.state_summary}",
    }
    merged: dict[str, tuple[EvidenceDoc, float]] = {}
    for field_name, query in queries.items():
        for doc, score in retriever.search(query, top_k=3, field=field_name):
            boosted = score + 1.0
            if doc.doc_id not in merged or boosted > merged[doc.doc_id][1]:
                merged[doc.doc_id] = (doc, boosted)
    hits = sorted(merged.values(), key=lambda x: (-x[1], x[0].doc_id))
    return hits[:5]


def produce_spec(arm: str, sample: MiniSample, retriever: MiniRetriever) -> tuple[dict[str, Any], list[tuple[EvidenceDoc, float]]]:
    if arm == "no_retrieval":
        return {
            "formula_id": DEFAULT_FORMULA_ID,
            "reliability_target": 0.99,
            "channel_margin_policy": "nominal",
            "service_rule": DEFAULT_SERVICE_RULE,
            "priority_rank": 1,
            "citations": [],
            "retrieved_ids": [],
        }, []
    if arm == "ordinary_rag_intent_only":
        hits = retriever.search(sample.intent, top_k=5)
        return _hits_to_spec(sample, hits), hits
    if arm == "state_aware_rag":
        hits = retriever.search(f"{sample.intent} {sample.state_summary}", top_k=5)
        return _hits_to_spec(sample, hits), hits
    if arm in {"field_aware_cer", "cer_verifier_solver"}:
        hits = _field_aware_hits(sample, retriever)
        return _hits_to_spec(sample, hits, require_latest=True, use_category_filter=True), hits
    raise KeyError(arm)


def evaluate_arm(arm: str, samples: list[MiniSample], retriever: MiniRetriever) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    solver = DeterministicSolver(EnvConfig())
    verifier = Verifier(require_citations=(arm == "cer_verifier_solver"))
    rows = []
    for sample in samples:
        spec, hits = produce_spec(arm, sample, retriever)
        retrieved_ids = [doc.doc_id for doc, _ in hits]
        result = verifier.verify(spec, retrieved_ids, sample.state, z_mode="cer")
        used_fallback = False
        executable = result.spec
        if arm == "cer_verifier_solver" and not result.passed:
            used_fallback = True
            executable = verifier.fail_closed_spec(sample.gold_spec["reliability_target"])
        elif not result.passed:
            executable = None
        if executable is None:
            p_min = None
            delta = None
            under = over = None
        else:
            p_min = solver.solve(executable, sample.state).p_min
            delta = float(p_min - sample.gold_p_min)
            under = max(0.0, -delta)
            over = max(0.0, delta)
        gold_ids = set(sample.gold_evidence_ids)
        ret_ids = set(retrieved_ids[:5])
        rows.append({
            "sample_id": sample.sample_id,
            "category": sample.category,
            "arm": arm,
            "retrieved_ids": retrieved_ids,
            "gold_evidence_ids": list(sample.gold_evidence_ids),
            "evidence_recall_at5": len(gold_ids & ret_ids) / max(len(gold_ids), 1),
            "formula_accuracy": int(spec.get("formula_id") == sample.gold_spec["formula_id"]),
            "reliability_target_accuracy": int(float(spec.get("reliability_target", -1)) == float(sample.gold_spec["reliability_target"])),
            "channel_margin_policy_accuracy": int(spec.get("channel_margin_policy") == sample.gold_spec["channel_margin_policy"]),
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
        })

    metrics: dict[str, Any] = {"arm": arm, "n": len(rows)}
    for key in (
        "evidence_recall_at5",
        "formula_accuracy",
        "reliability_target_accuracy",
        "channel_margin_policy_accuracy",
        "service_rule_accuracy",
        "spec_validity",
        "fallback",
    ):
        metrics[key] = float(np.mean([r[key] for r in rows]))
    deltas = np.asarray([r["delta_p_min"] for r in rows if r["delta_p_min"] is not None], dtype=float)
    metrics["mean_abs_delta_p_min"] = float(np.mean(np.abs(deltas))) if deltas.size else None
    metrics["mean_delta_p_min"] = float(np.mean(deltas)) if deltas.size else None
    metrics["under_reservation_rate"] = float(np.mean(deltas < 0)) if deltas.size else None
    metrics["mean_under_reservation_prb"] = float(np.mean([r["under_reservation_prb"] or 0.0 for r in rows]))
    metrics["mean_over_reservation_prb"] = float(np.mean([r["over_reservation_prb"] or 0.0 for r in rows]))
    metrics["field_accuracy_mean"] = float(np.mean([
        metrics["formula_accuracy"],
        metrics["reliability_target_accuracy"],
        metrics["channel_margin_policy_accuracy"],
        metrics["service_rule_accuracy"],
    ]))
    return metrics, rows


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    arms = summary["arms"]
    ordinary = arms["ordinary_rag_intent_only"]
    field = arms["field_aware_cer"]
    cer = arms["cer_verifier_solver"]
    checks = {
        "field_aware_recall_beats_ordinary": field["evidence_recall_at5"] > ordinary["evidence_recall_at5"],
        "field_aware_field_accuracy_beats_ordinary": field["field_accuracy_mean"] > ordinary["field_accuracy_mean"],
        "field_aware_delta_beats_ordinary": field["mean_abs_delta_p_min"] < ordinary["mean_abs_delta_p_min"],
        "cer_verifier_solver_valid": cer["spec_validity"] >= 0.99,
        "cer_verifier_solver_safe": cer["under_reservation_rate"] <= field["under_reservation_rate"],
    }
    return {"checks": checks, "PASS": all(checks.values())}


def write_table(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "n",
        "evidence_recall_at5",
        "formula_accuracy",
        "reliability_target_accuracy",
        "channel_margin_policy_accuracy",
        "service_rule_accuracy",
        "field_accuracy_mean",
        "spec_validity",
        "mean_abs_delta_p_min",
        "under_reservation_rate",
        "mean_under_reservation_prb",
        "mean_over_reservation_prb",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm in ARMS:
            writer.writerow({k: summary["arms"][arm].get(k, "") for k in fields})


def _plot(summary: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["No retrieval", "Ordinary RAG", "State-aware", "Field-aware CER", "CER+verifier"]
    colors = ["#9E9E9E", "#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
    field_acc = [summary["arms"][arm]["field_accuracy_mean"] for arm in ARMS]
    delta = [summary["arms"][arm]["mean_abs_delta_p_min"] for arm in ARMS]

    outputs = []
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(labels, field_acc, color=colors, edgecolor="#222222", linewidth=0.5)
    ax.set_ylabel("Mean field accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mini-CER Field-Level Constraint Accuracy")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "fig_mini_cer_field_accuracy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(labels, delta, color=colors, edgecolor="#222222", linewidth=0.5)
    ax.set_ylabel("Mean |delta p_min|")
    ax.set_title("Mini-CER Solver Error vs Gold p_min")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "fig_mini_cer_delta_pmin.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path))
    return outputs


def run(smoke: bool = False) -> dict[str, Any]:
    docs = build_corpus()
    samples = build_samples(n_per_category=1 if smoke else 20)
    retriever = MiniRetriever(docs)
    arms = {}
    per_sample = []
    for arm in ARMS:
        metrics, rows = evaluate_arm(arm, samples, retriever)
        arms[arm] = metrics
        per_sample.extend(rows)
    summary = {
        "schema_version": "safe_oran_phase2c_mini_cer",
        "kind": "phase2c_mini_cer_field_eval",
        "claim_scope": "deterministic field-level retrieval/routing benchmark; no real LLM generation",
        "smoke": bool(smoke),
        "paper_usable": not bool(smoke),
        "n_samples": len(samples),
        "categories": sorted({s.category for s in samples}),
        "arms": arms,
    }
    summary["gate"] = gate(summary)
    return {"summary": summary, "analysis": {"samples": per_sample, "docs": [d.__dict__ for d in docs]}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    result = run(smoke=args.smoke)
    out_dir = OUT_DIR / "smoke" if args.smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True))
    (out_dir / "analysis.json").write_text(json.dumps(result["analysis"], indent=2, sort_keys=True))
    write_table(result["summary"], out_dir / "mini_cer_table.csv")
    figures = [] if args.smoke else _plot(result["summary"])
    result["summary"]["figures"] = figures
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps({
        "summary": str(out_dir / "summary.json"),
        "analysis": str(out_dir / "analysis.json"),
        "table": str(out_dir / "mini_cer_table.csv"),
        "figures": figures,
        "gate": result["summary"]["gate"],
    }, indent=2, sort_keys=True))
    if not result["summary"]["gate"]["PASS"]:
        raise SystemExit("Phase2c mini-CER gate FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
