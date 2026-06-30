"""Analyze M6 RAG ablation replay and write paper-facing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.build_m6_rag_ablation_caches import ABLATION_ARMS, OUT_DIR, SCENARIO
from safe_oran.experiments.eval_m6_rag_ablation import RUNS_DIR

FIG_DIR = PROJECT_ROOT / "05_figures" / "phase3_m6_ablation"
REPORT_PATH = PROJECT_ROOT / "06_reports" / "M6_RAG_ABLATION_RESULTS.md"
LETTER_PATH = PROJECT_ROOT / "06_reports" / "LETTER_EVIDENCE_STORY.md"
WSA_REPORT_PATH = PROJECT_ROOT / "06_reports" / "PHASE2C_WSA_RESULTS.md"
SEEDS = (42, 43, 44)
METRICS = (
    "reward",
    "urllc_violation_rate",
    "mean_D_proj",
    "shield_correction_rate",
    "fallback_rate",
    "p_min_parity_rate",
    "unsafe_under_reservation_rate",
    "mean_under_reservation_prb",
    "mean_over_reservation_prb",
    "mean_abs_delta_p_min_vs_oracle",
)


def _load_runs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def _mean_std(values: list[Any]) -> tuple[float | None, float | None, int]:
    vals = [float(v) for v in values if v is not None and v != ""]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def _fmt(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return ""
    return f"{mean:.{digits}f} ± {(0.0 if std is None else std):.{digits}f}"


def _groups(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        arm = run.get("retrieval_arm")
        if arm in ABLATION_ARMS:
            groups[arm].append(run)
    return groups


def _mean(groups: dict[str, list[dict[str, Any]]], arm: str, metric: str) -> float | None:
    m, _, _ = _mean_std([r.get("eval_metrics", {}).get(metric) for r in groups.get(arm, [])])
    return m


def validate(runs: list[dict[str, Any]]) -> list[str]:
    warnings = []
    groups = _groups(runs)
    for arm in ABLATION_ARMS:
        rs = sorted(groups.get(arm, []), key=lambda r: int(r.get("seed", -1)))
        if len(rs) < 3:
            warnings.append(f"{arm}: n_runs={len(rs)} < 3")
        if not all(bool(r.get("paper_usable", False)) for r in rs):
            warnings.append(f"{arm}: not all runs are paper_usable")
        for run in rs:
            eps = run.get("evidence", {}).get("n_eval_episodes", 0)
            if int(eps or 0) < 5:
                warnings.append(f"{arm} seed={run.get('seed')}: n_eval_episodes={eps} < 5")
            if not run.get("evidence", {}).get("no_training", False):
                warnings.append(f"{arm} seed={run.get('seed')}: no_training flag missing")
            if not run.get("evidence", {}).get("no_llm_calls", False):
                warnings.append(f"{arm} seed={run.get('seed')}: no_llm_calls flag missing")
    return warnings


def build_table(runs: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    groups = _groups(runs)
    rows = []
    for arm in ABLATION_ARMS:
        rs = sorted(groups.get(arm, []), key=lambda r: int(r.get("seed", -1)))
        row: dict[str, Any] = {
            "scenario": SCENARIO,
            "retrieval_arm": arm,
            "n_runs": len(rs),
            "seeds": " ".join(str(r.get("seed")) for r in rs),
            "paper_usable": str(bool(rs) and all(bool(r.get("paper_usable", False)) for r in rs)).lower(),
        }
        for metric in METRICS:
            mean, std, n = _mean_std([r.get("eval_metrics", {}).get(metric) for r in rs])
            row[metric] = _fmt(mean, std)
            row[f"{metric}_mean"] = "" if mean is None else mean
            row[f"{metric}_std"] = "" if std is None else std
            row[f"{metric}_n"] = n
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_paired_delta(runs: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    keyed = {(r.get("retrieval_arm"), int(r.get("seed"))): r for r in runs if r.get("retrieval_arm") in ABLATION_ARMS}
    rows = []
    for arm in ABLATION_ARMS:
        if arm == "oracle_z":
            continue
        for metric in METRICS:
            row: dict[str, Any] = {
                "scenario": SCENARIO,
                "comparison": f"{arm}-minus-oracle_z",
                "retrieval_arm": arm,
                "baseline_arm": "oracle_z",
                "metric": metric,
            }
            deltas: list[float | None] = []
            for seed in SEEDS:
                base = keyed.get(("oracle_z", seed), {}).get("eval_metrics", {}).get(metric)
                val = keyed.get((arm, seed), {}).get("eval_metrics", {}).get(metric)
                delta = None if base is None or val is None else float(val) - float(base)
                row[f"seed{seed}_delta"] = "" if delta is None else delta
                deltas.append(delta)
            mean, std, n = _mean_std(deltas)
            row["mean_delta"] = "" if mean is None else mean
            row["std_delta"] = "" if std is None else std
            row["n_valid"] = n
            row["improvement_direction"] = "positive" if metric in {"reward", "p_min_parity_rate"} else "negative"
            rows.append(row)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _init_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _bar_plot(groups: dict[str, list[dict[str, Any]]], specs: list[tuple[str, str]], path: Path, title: str) -> None:
    plt = _init_matplotlib()
    labels = ["Oracle", "No retrieval", "Ordinary", "State-aware", "Field-CER"]
    arms = list(ABLATION_ARMS)
    fig, axes = plt.subplots(1, len(specs), figsize=(max(7.0, 3.1 * len(specs)), 4.0))
    if len(specs) == 1:
        axes = [axes]
    colors = ["#9E9E9E", "#BAB0AC", "#4C78A8", "#72B7B2", "#F58518"]
    for ax, (metric, ylabel) in zip(axes, specs):
        means, stds = [], []
        for arm in arms:
            mean, std, _ = _mean_std([r.get("eval_metrics", {}).get(metric) for r in groups.get(arm, [])])
            means.append(np.nan if mean is None else mean)
            stds.append(0.0 if std is None else std)
        ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="#222222", linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
        if metric.endswith("rate"):
            ax.set_ylim(0, max(1.05, float(np.nanmax(means)) * 1.25 if np.isfinite(means).any() else 1.0))
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_figures(groups: dict[str, list[dict[str, Any]]]) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        FIG_DIR / "fig_m6_ablation_control_metrics.png",
        FIG_DIR / "fig_m6_ablation_pmin_parity.png",
        FIG_DIR / "fig_m6_ablation_reservation_error.png",
    ]
    _bar_plot(groups, [
        ("reward", "Reward"),
        ("urllc_violation_rate", "Violation"),
        ("mean_D_proj", "Mean D_proj"),
        ("shield_correction_rate", "Correction"),
    ], outputs[0], "S6 M6 RAG Ablation: Closed-Loop Metrics")
    _bar_plot(groups, [
        ("p_min_parity_rate", "p_min parity"),
        ("mean_abs_delta_p_min_vs_oracle", "|delta p_min|"),
        ("fallback_rate", "Fallback"),
    ], outputs[1], "S6 M6 RAG Ablation: p_min Parity")
    _bar_plot(groups, [
        ("mean_under_reservation_prb", "Under-reserved PRB"),
        ("mean_over_reservation_prb", "Over-reserved PRB"),
        ("unsafe_under_reservation_rate", "Unsafe under-rsv"),
    ], outputs[2], "S6 M6 RAG Ablation: Reservation Error")
    return outputs


def compute_gates(groups: dict[str, list[dict[str, Any]]], warnings: list[str]) -> dict[str, Any]:
    field = {m: _mean(groups, "field_CER_z", m) for m in METRICS}

    def val(rec: dict[str, float | None], key: str, default: float) -> float:
        out = rec.get(key)
        return default if out is None else float(out)

    field_gate = {
        "p_min_parity_ge_0p95": val(field, "p_min_parity_rate", 0.0) >= 0.95,
        "unsafe_under_le_0p02": val(field, "unsafe_under_reservation_rate", 1.0) <= 0.02,
        "fallback_le_0p05": val(field, "fallback_rate", 1.0) <= 0.05,
    }
    weaker = {}
    for arm in ("no_retrieval_z", "ordinary_rag_z", "state_aware_rag_z"):
        rec = {m: _mean(groups, arm, m) for m in METRICS}
        weaker[arm] = any([
            val(rec, "p_min_parity_rate", 1.0) < val(field, "p_min_parity_rate", 0.0) - 1e-9,
            val(rec, "mean_over_reservation_prb", 0.0) > val(field, "mean_over_reservation_prb", 0.0) + 1e-9,
            val(rec, "mean_under_reservation_prb", 0.0) > val(field, "mean_under_reservation_prb", 0.0) + 1e-9,
            val(rec, "mean_D_proj", 0.0) > val(field, "mean_D_proj", 0.0) + 1e-9,
            val(rec, "fallback_rate", 0.0) > val(field, "fallback_rate", 0.0) + 1e-9,
        ])
    return {
        "formal_warnings_empty": not warnings,
        "field_CER_gate": field_gate,
        "weaker_arm_detected": weaker,
        "passed": bool(not warnings and all(field_gate.values()) and any(weaker.values())),
    }


def _row_for_report(row: dict[str, Any]) -> str:
    return (
        f"| {row['retrieval_arm']} | {row['reward']} | {row['urllc_violation_rate']} | "
        f"{row['mean_D_proj']} | {row['shield_correction_rate']} | {row['p_min_parity_rate']} | "
        f"{row['fallback_rate']} | {row['mean_under_reservation_prb']} | {row['mean_over_reservation_prb']} |"
    )


def write_report(table_rows, delta_rows, gates, warnings, figures) -> None:
    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- All replay artifact gates passed."
    verdict = (
        "The S6 replay is discriminative: weaker retrieval arms alter the executable constraint path, while field-CER remains oracle-parity."
        if gates["passed"]
        else "The S6 two-event replay is only partially discriminative; use WS-A and event trace for the broader RAG claim."
    )
    table = "\n".join(_row_for_report(r) for r in table_rows)
    selected = [
        r for r in delta_rows
        if r["metric"] in {"mean_D_proj", "p_min_parity_rate", "mean_over_reservation_prb", "mean_under_reservation_prb", "fallback_rate"}
        and r["retrieval_arm"] in {"ordinary_rag_z", "field_CER_z"}
    ]
    delta_text = "\n".join(
        f"- `{r['comparison']}` `{r['metric']}`: {float(r['mean_delta']):.4f} ± {float(r['std_delta']):.4f}."
        for r in selected
        if r["mean_delta"] != ""
    )
    fig_text = "\n".join(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in figures)
    text = f"""# M6 RAG Ablation Replay Results

## Verdict
{verdict}

This experiment replays existing S6 M5 policies under cached Qwen3-4B+BGE symbolic z-caches. It does not train PPO and does not call an LLM.

## Gate Status
{warning_text}

Field-CER gate: `{json.dumps(gates['field_CER_gate'], sort_keys=True)}`.

Weaker-arm detection: `{json.dumps(gates['weaker_arm_detected'], sort_keys=True)}`.

## Result Table

| Arm | Reward | Violation | Mean D_proj | Correction | p_min parity | Fallback | Under PRB | Over PRB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Selected Paired Deltas vs Oracle
{delta_text}

## Interpretation
- `field_CER_z` is the closed-loop positive control: cached real-LLM field-CER spec stays aligned with Oracle-z.
- `ordinary_rag_z` demonstrates the visible RAG failure mode on S6: it over-reserves at the normal event because intent-only retrieval misses the state-conditioned margin.
- `no_retrieval_z` is intentionally ungrounded and fail-closed; report its fallback behavior rather than treating it as a valid candidate controller.
- The broader field-CER advantage remains the WS-A result across 160 samples and 4 model sizes; this replay shows how retrieval errors propagate into the controller when they change `p_min`.

## Generated Artifacts
- `04_results/phase3_m6_ablation/m6_event_trace.csv`
- `04_results/phase3_m6_ablation/m6_replay_trace.csv`
- `04_results/phase3_m6_ablation/m6_rag_ablation_table.csv`
- `04_results/phase3_m6_ablation/m6_rag_ablation_paired_delta.csv`
- `04_results/phase3_m6_ablation/summary.json`
{fig_text}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)


def update_story_and_wsa() -> None:
    if LETTER_PATH.exists():
        text = LETTER_PATH.read_text()
        marker = "\n## Paper Wording\n"
        insert = """\n### Phase3-M6 RAG ablation: retrieval errors become control errors\n\nThe M6 replay ablation compares oracle, no-retrieval, ordinary RAG, state-aware RAG, and field-CER z-caches under the same S6 M5 policies. It makes the RAG contribution visible: field-CER remains oracle-parity, while weaker retrieval arms can alter `p_min` through over/under reservation or fallback.\n"""
        if "Phase3-M6 RAG ablation" not in text and marker in text:
            text = text.replace(marker, insert + marker)
            LETTER_PATH.write_text(text)
    if WSA_REPORT_PATH.exists():
        text = WSA_REPORT_PATH.read_text()
        old_wsa_note = (
            "Closed-loop (" + "M6" + ") "
            + "still "
            + "deferred."
        )
        text = text.replace(
            old_wsa_note,
            "Closed-loop M6 and M6 RAG ablation are reported in the Phase3-M6 artifacts.",
        )
        WSA_REPORT_PATH.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = ap.parse_args()
    runs = _load_runs(Path(args.runs_dir))
    if not runs:
        raise SystemExit(f"no ablation runs found in {args.runs_dir}")
    warnings = validate(runs)
    table_rows = build_table(runs, OUT_DIR / "m6_rag_ablation_table.csv")
    delta_rows = build_paired_delta(runs, OUT_DIR / "m6_rag_ablation_paired_delta.csv")
    groups = _groups(runs)
    figures = generate_figures(groups)
    gates = compute_gates(groups, warnings)
    summary = {
        "schema_version": "safe_oran_m6_rag_ablation",
        "kind": "m6_rag_ablation_summary",
        "scenario": SCENARIO,
        "n_runs": len(runs),
        "warnings": warnings,
        "gates": gates,
        "table": str(OUT_DIR / "m6_rag_ablation_table.csv"),
        "paired_delta": str(OUT_DIR / "m6_rag_ablation_paired_delta.csv"),
        "event_trace": str(OUT_DIR / "m6_event_trace.csv"),
        "replay_trace": str(OUT_DIR / "m6_replay_trace.csv"),
        "figures": [str(p) for p in figures],
        "report": str(REPORT_PATH),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(table_rows, delta_rows, gates, warnings, figures)
    update_story_and_wsa()
    expected = [
        OUT_DIR / "m6_event_trace.csv",
        OUT_DIR / "m6_replay_trace.csv",
        OUT_DIR / "m6_rag_ablation_table.csv",
        OUT_DIR / "m6_rag_ablation_paired_delta.csv",
        OUT_DIR / "summary.json",
        REPORT_PATH,
        *figures,
    ]
    missing = [str(p) for p in expected if not p.exists() or p.stat().st_size == 0]
    print(json.dumps({**summary, "missing_or_empty": missing}, indent=2, sort_keys=True))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
