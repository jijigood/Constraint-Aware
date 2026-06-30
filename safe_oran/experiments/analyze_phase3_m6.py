"""Analyze the S6 M5-vs-M6 closed-loop CER-z comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.eval_phase3_m6 import OUT_DIR, RUNS_OUT_DIR, SCENARIO

FIG_DIR = PROJECT_ROOT / "05_figures" / "phase3_m6"
REPORT_PATH = PROJECT_ROOT / "06_reports" / "M6_CLOSED_LOOP_RESULTS.md"
METHODS = ("M5_constraint_aware", "M6_field_CER_z")
SEEDS = (42, 43, 44)
METRICS = (
    "reward",
    "urllc_violation_rate",
    "mean_D_proj",
    "shield_correction_rate",
    "adaptation_delay",
    "fallback_rate",
    "unsafe_under_reservation_rate",
    "p_min_parity_rate",
    "mean_abs_delta_p_min_vs_oracle",
    "mean_under_reservation_prb",
    "mean_over_reservation_prb",
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
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run.get("scenario") == SCENARIO and run.get("method") in METHODS:
            out[run["method"]].append(run)
    return out


def _metric_mean(groups: dict[str, list[dict[str, Any]]], method: str, metric: str) -> float | None:
    mean, _, _ = _mean_std([r.get("eval_metrics", {}).get(metric) for r in groups.get(method, [])])
    return mean


def validate(runs: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    groups = _groups(runs)
    for method in METHODS:
        rs = sorted(groups.get(method, []), key=lambda r: int(r.get("seed", -1)))
        if len(rs) < 3:
            warnings.append(f"{method}: n_runs={len(rs)} < 3")
        if any(bool(r.get("quick", False)) for r in rs):
            warnings.append(f"{method}: quick run detected")
        if not all(bool(r.get("paper_usable", False)) for r in rs):
            warnings.append(f"{method}: not all runs are paper_usable")
        for run in rs:
            ev = run.get("evidence", {})
            req = ev.get("source_total_timesteps_requested")
            consumed = ev.get("source_total_timesteps_consumed")
            eps = ev.get("n_eval_episodes", 0)
            if req is None or consumed is None:
                warnings.append(f"{method} seed={run.get('seed')}: missing source timesteps")
            elif float(consumed) < 0.99 * float(req):
                warnings.append(f"{method} seed={run.get('seed')}: consumed {consumed} < 0.99 * requested {req}")
            if int(eps or 0) < 5:
                warnings.append(f"{method} seed={run.get('seed')}: n_eval_episodes={eps} < 5")
    return warnings


def build_table(runs: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    rows = []
    groups = _groups(runs)
    for method in METHODS:
        rs = sorted(groups.get(method, []), key=lambda r: int(r.get("seed", -1)))
        row: dict[str, Any] = {
            "scenario": SCENARIO,
            "method": method,
            "n_runs": len(rs),
            "seeds": " ".join(str(r.get("seed")) for r in rs),
            "paper_usable": str(bool(rs) and all(bool(r.get("paper_usable", False)) for r in rs)).lower(),
            "quick_any": str(any(bool(r.get("quick", False)) for r in rs)).lower(),
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
    keyed = {(r["method"], int(r["seed"])): r for r in runs if r.get("method") in METHODS}
    rows = []
    for metric in METRICS:
        row: dict[str, Any] = {"scenario": SCENARIO, "metric": metric}
        deltas: list[float | None] = []
        for seed in SEEDS:
            m5 = keyed.get(("M5_constraint_aware", seed), {}).get("eval_metrics", {}).get(metric)
            m6 = keyed.get(("M6_field_CER_z", seed), {}).get("eval_metrics", {}).get(metric)
            delta = None if m5 is None or m6 is None else float(m6) - float(m5)
            row[f"seed{seed}_delta"] = "" if delta is None else delta
            deltas.append(delta)
        mean, std, n = _mean_std(deltas)
        row["mean_delta"] = "" if mean is None else mean
        row["std_delta"] = "" if std is None else std
        row["n_valid"] = n
        row["improvement_direction"] = "positive" if metric in {"reward", "p_min_parity_rate"} else "negative"
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
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


def plot_core_bars(groups: dict[str, list[dict[str, Any]]], path: Path) -> None:
    plt = _init_matplotlib()
    metrics = [
        ("reward", "Reward"),
        ("urllc_violation_rate", "Violation"),
        ("mean_D_proj", "Mean D_proj"),
        ("shield_correction_rate", "Correction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    colors = ["#F58518", "#54A24B"]
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        means, stds = [], []
        for method in METHODS:
            mean, std, _ = _mean_std([r.get("eval_metrics", {}).get(metric) for r in groups.get(method, [])])
            means.append(np.nan if mean is None else mean)
            stds.append(0.0 if std is None else std)
        ax.bar(["M5 Oracle-z", "M6 CER-z"], means, yerr=stds, capsize=4, color=colors, edgecolor="#222222", linewidth=0.6)
        ymax = np.nanmax(np.asarray(means, dtype=float)) if any(np.isfinite(means)) else 0.0
        if metric == "p_min_parity_rate":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, max(0.05, ymax * 1.25))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("S6 Closed-Loop: Oracle-z vs Field-CER-z")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_safety_bars(groups: dict[str, list[dict[str, Any]]], path: Path) -> None:
    plt = _init_matplotlib()
    metrics = [
        ("fallback_rate", "Fallback"),
        ("unsafe_under_reservation_rate", "Unsafe under-rsv"),
        ("p_min_parity_rate", "p_min parity"),
        ("mean_abs_delta_p_min_vs_oracle", "|delta p_min|"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    colors = ["#F58518", "#54A24B"]
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        means, stds = [], []
        for method in METHODS:
            mean, std, _ = _mean_std([r.get("eval_metrics", {}).get(metric) for r in groups.get(method, [])])
            means.append(np.nan if mean is None else mean)
            stds.append(0.0 if std is None else std)
        ax.bar(["M5 Oracle-z", "M6 CER-z"], means, yerr=stds, capsize=4, color=colors, edgecolor="#222222", linewidth=0.6)
        ymax = np.nanmax(np.asarray(means, dtype=float)) if any(np.isfinite(means)) else 0.0
        if metric == "p_min_parity_rate":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, max(0.05, ymax * 1.25))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("M6 CER-z Safety/Parity Diagnostics")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compute_gates(groups: dict[str, list[dict[str, Any]]], warnings: list[str]) -> dict[str, Any]:
    m5 = {m: _metric_mean(groups, "M5_constraint_aware", m) for m in METRICS}
    m6 = {m: _metric_mean(groups, "M6_field_CER_z", m) for m in METRICS}
    def val(rec: dict[str, float | None], key: str, default: float) -> float:
        out = rec.get(key)
        return default if out is None else float(out)

    hard = {
        "fallback_rate_le_0p05": val(m6, "fallback_rate", 1.0) <= 0.05,
        "unsafe_under_le_0p02": val(m6, "unsafe_under_reservation_rate", 1.0) <= 0.02,
        "p_min_parity_ge_0p95": val(m6, "p_min_parity_rate", 0.0) >= 0.95,
    }
    limited = {
        "reward_within_0p05": val(m6, "reward", -1e9) >= val(m5, "reward", 0.0) - 0.05,
        "violation_within_0p03": val(m6, "urllc_violation_rate", 1e9) <= val(m5, "urllc_violation_rate", 0.0) + 0.03,
        "mean_D_proj_within_7p5": val(m6, "mean_D_proj", 1e9) <= val(m5, "mean_D_proj", 0.0) + 7.5,
        "correction_within_0p10": val(m6, "shield_correction_rate", 1e9) <= val(m5, "shield_correction_rate", 0.0) + 0.10,
    }
    return {
        "hard_safety": hard,
        "limited_degradation": limited,
        "all_formal_gates": not warnings,
        "passed": bool(not warnings and all(hard.values()) and all(limited.values())),
        "m5_means": m5,
        "m6_means": m6,
    }


def write_report(
    table_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    warnings: list[str],
    gates: dict[str, Any],
    figures: list[Path],
) -> None:
    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- All formal-run gates passed."
    conclusion = (
        "CER-z can replace oracle-z in the S6 closed-loop controller with limited degradation."
        if gates["passed"]
        else "M6 is a boundary result: CER-z is evaluated in the closed loop, but at least one formal or degradation gate did not pass."
    )
    rows_text = "\n".join(
        f"| {r['method']} | {r['reward']} | {r['urllc_violation_rate']} | {r['mean_D_proj']} | "
        f"{r['shield_correction_rate']} | {r['fallback_rate']} | {r['unsafe_under_reservation_rate']} | "
        f"{r['p_min_parity_rate']} |"
        for r in table_rows
    )
    delta_text = "\n".join(
        f"- `{r['metric']}`: {float(r['mean_delta']):.4f} ± {float(r['std_delta']):.4f} (M6 - M5, n={r['n_valid']})."
        for r in delta_rows
        if r["metric"] in {"reward", "urllc_violation_rate", "mean_D_proj", "shield_correction_rate", "p_min_parity_rate"}
        and r["mean_delta"] != ""
    )
    fig_text = "\n".join(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in figures)
    text = f"""# M6 Closed-Loop CER-z Results

## Conclusion
{conclusion}

## Gate Status
{warning_text}

Hard safety gates: `{json.dumps(gates['hard_safety'], sort_keys=True)}`.

Limited-degradation gates: `{json.dumps(gates['limited_degradation'], sort_keys=True)}`.

## Result Table

| Method | Reward | Violation | Mean D_proj | Shield correction | Fallback | Unsafe under-rsv | p_min parity |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows_text}

## Paired Seed Delta
Positive delta improves reward/parity; negative delta improves violation, projection, and correction.

{delta_text}

## Interpretation
- M6 uses cached real-LLM `field-aware CER -> z_k` from WS-A; no LLM is called during DRL training or evaluation.
- The comparison is intentionally narrow: `S6_moderate_decay`, same PPO setup, same `p_min/Pmax` state augmentation, Oracle-z replaced by CER-z.
- Report M6 as the final closed-loop bridge after Phase2c: it tests whether the compiled CER-z remains usable once inserted into the controller.

## Generated Artifacts
- `04_results/phase3_m6/m6_table.csv`
- `04_results/phase3_m6/m6_paired_delta.csv`
- `04_results/phase3_m6/summary.json`
{fig_text}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_OUT_DIR))
    args = ap.parse_args()

    runs = _load_runs(Path(args.runs_dir))
    if not runs:
        raise SystemExit(f"no M6 eval runs found in {args.runs_dir}")
    warnings = validate(runs)
    table_rows = build_table(runs, OUT_DIR / "m6_table.csv")
    delta_rows = build_paired_delta(runs, OUT_DIR / "m6_paired_delta.csv")
    groups = _groups(runs)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = [FIG_DIR / "fig_m6_closed_loop_bars.png", FIG_DIR / "fig_m6_safety_parity.png"]
    plot_core_bars(groups, figures[0])
    plot_safety_bars(groups, figures[1])
    gates = compute_gates(groups, warnings)
    summary = {
        "schema_version": "safe_oran_phase3_m6",
        "kind": "phase3_m6_summary",
        "scenario": SCENARIO,
        "n_runs": len(runs),
        "warnings": warnings,
        "gates": gates,
        "table": str(OUT_DIR / "m6_table.csv"),
        "paired_delta": str(OUT_DIR / "m6_paired_delta.csv"),
        "figures": [str(p) for p in figures],
        "report": str(REPORT_PATH),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(table_rows, delta_rows, warnings, gates, figures)
    missing = [str(p) for p in [OUT_DIR / "m6_table.csv", OUT_DIR / "m6_paired_delta.csv", OUT_DIR / "summary.json", REPORT_PATH, *figures] if not p.exists() or p.stat().st_size == 0]
    print(json.dumps({**summary, "missing_or_empty": missing}, indent=2, sort_keys=True))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
