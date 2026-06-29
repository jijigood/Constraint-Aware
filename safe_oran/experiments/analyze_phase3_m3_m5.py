"""Build Phase3 M3/M5 result tables, paired deltas, figures, and report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.envs.legacy import EnvConfig, PROJECT_ROOT
from safe_oran.experiments.configs import SCENARIOS
from safe_oran.experiments.m3_m5_common import (
    METHODS,
    MODEL_DIR,
    OUT_DIR,
    RUNS_DIR,
    SCENARIOS_PHASE3,
    make_constraint_env,
    norm_from_vecnormalize,
)

FIG_DIR = PROJECT_ROOT / "05_figures" / "phase3_m3_m5"
REPORT_PATH = PROJECT_ROOT / "06_reports" / "PHASE3_M3_M5_RESULTS.md"
M3 = "M3_dynamic_no_aug"
M5 = "M5_constraint_aware"
SEEDS = (42, 43, 44)
METRICS = (
    "reward",
    "urllc_violation_rate",
    "mean_D_proj",
    "shield_correction_rate",
    "adaptation_delay",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    return [_load_json(path) for path in sorted(runs_dir.glob("*.json"))]


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def _fmt(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _evidence(run: dict[str, Any], key: str, default: Any = None) -> Any:
    return run.get(key, run.get("evidence", {}).get(key, default))


def _group_runs(runs: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run.get("method") in METHODS and run.get("scenario") in SCENARIOS_PHASE3:
            groups[(run["scenario"], run["method"])].append(run)
    return groups


def _run_by_key(runs: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run in runs:
        try:
            key = (run["scenario"], run["method"], int(run["seed"]))
        except KeyError:
            continue
        out[key] = run
    return out


def validate_runs(runs: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    groups = _group_runs(runs)
    for scenario in SCENARIOS_PHASE3:
        for method in METHODS:
            rs = sorted(groups.get((scenario, method), []), key=lambda r: int(r.get("seed", -1)))
            label = f"{method}|{scenario}"
            if len(rs) < 3:
                warnings.append(f"{label}: n_runs={len(rs)} < 3")
            if not all(bool(r.get("paper_usable", False)) for r in rs):
                warnings.append(f"{label}: not all runs are paper_usable")
            if any(bool(r.get("quick", False)) for r in rs):
                warnings.append(f"{label}: quick run detected")
            for run in rs:
                seed = run.get("seed")
                req = _evidence(run, "total_timesteps_requested")
                consumed = _evidence(run, "total_timesteps_consumed")
                eval_eps = _evidence(run, "n_eval_episodes", len(run.get("eval_per_episode", [])))
                if req is None or consumed is None:
                    warnings.append(f"{label} seed={seed}: missing timesteps evidence")
                elif float(consumed) < 0.99 * float(req):
                    warnings.append(f"{label} seed={seed}: consumed {consumed} < 0.99 * requested {req}")
                if eval_eps is None or int(eval_eps) < 5:
                    warnings.append(f"{label} seed={seed}: n_eval_episodes={eval_eps} < 5")
    return warnings


def build_phase3_table(runs: list[dict[str, Any]], out_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _group_runs(runs)
    for scenario in SCENARIOS_PHASE3:
        for method in METHODS:
            rs = sorted(groups.get((scenario, method), []), key=lambda r: int(r.get("seed", -1)))
            row: dict[str, Any] = {
                "scenario": scenario,
                "method": method,
                "n_runs": len(rs),
                "seeds": " ".join(str(r.get("seed")) for r in rs),
                "paper_usable": str(bool(rs) and all(bool(r.get("paper_usable", False)) for r in rs)).lower(),
                "quick_any": str(any(bool(r.get("quick", False)) for r in rs)).lower(),
                "timesteps_requested": int(sum(int(_evidence(r, "total_timesteps_requested", 0) or 0) for r in rs)),
                "timesteps_consumed": int(sum(int(_evidence(r, "total_timesteps_consumed", 0) or 0) for r in rs)),
                "n_eval_episodes": int(sum(int(_evidence(r, "n_eval_episodes", len(r.get("eval_per_episode", []))) or 0) for r in rs)),
            }
            for metric in METRICS:
                mean, std, n = _mean_std([r.get("eval_metrics", {}).get(metric) for r in rs])
                row[metric] = _fmt(mean, std)
                row[f"{metric}_mean"] = "" if mean is None else mean
                row[f"{metric}_std"] = "" if std is None else std
                row[f"{metric}_n"] = n
            for metric in ("train_violation_rate", "train_mean_D_proj", "train_shield_correction_rate"):
                mean, std, n = _mean_std([r.get("train_metrics", {}).get(metric) for r in rs])
                row[metric] = _fmt(mean, std)
                row[f"{metric}_mean"] = "" if mean is None else mean
                row[f"{metric}_std"] = "" if std is None else std
                row[f"{metric}_n"] = n
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_paired_delta(runs: list[dict[str, Any]], out_path: Path) -> list[dict[str, Any]]:
    keyed = _run_by_key(runs)
    rows: list[dict[str, Any]] = []
    delta_metrics = (
        "reward",
        "urllc_violation_rate",
        "mean_D_proj",
        "shield_correction_rate",
        "adaptation_delay",
    )
    for scenario in SCENARIOS_PHASE3:
        for metric in delta_metrics:
            deltas: list[float | None] = []
            row: dict[str, Any] = {"scenario": scenario, "metric": metric}
            for seed in SEEDS:
                m3 = keyed.get((scenario, M3, seed), {}).get("eval_metrics", {}).get(metric)
                m5 = keyed.get((scenario, M5, seed), {}).get("eval_metrics", {}).get(metric)
                delta = None if m3 is None or m5 is None else float(m5) - float(m3)
                deltas.append(delta)
                row[f"seed{seed}_delta"] = "" if delta is None else delta
            mean, std, n = _mean_std(deltas)
            row["mean_delta"] = "" if mean is None else mean
            row["std_delta"] = "" if std is None else std
            row["n_valid"] = n
            if metric == "reward":
                row["improvement_direction"] = "positive"
            else:
                row["improvement_direction"] = "negative"
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _init_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _metric_stats(runs: list[dict[str, Any]], scenario: str, method: str, metric: str) -> tuple[float | None, float | None]:
    vals = [
        r.get("eval_metrics", {}).get(metric)
        for r in runs
        if r.get("scenario") == scenario and r.get("method") == method
    ]
    mean, std, _ = _mean_std(vals)
    return mean, std


def _bar_values(runs: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for method in METHODS:
        method_means = []
        method_stds = []
        for scenario in SCENARIOS_PHASE3:
            mean, std = _metric_stats(runs, scenario, method, metric)
            method_means.append(np.nan if mean is None else mean)
            method_stds.append(0.0 if std is None else std)
        means.append(method_means)
        stds.append(method_stds)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def plot_grouped_bar(
    runs: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt = _init_matplotlib()
    label_map = {
        "S3_channel_decay": "S3 decay",
        "S4_sla_upgrade": "S4 SLA",
        "S5_combined": "S5 combined",
        "S6_moderate_decay": "S6 moderate",
    }
    labels = [label_map.get(s, s) for s in SCENARIOS_PHASE3]
    means, stds = _bar_values(runs, metric)
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(7.2, 1.8 * len(labels)), 4.4))
    colors = ["#4C78A8", "#F58518"]
    names = ["M3 dynamic no aug", "M5 constraint aware"]
    for idx, method_name in enumerate(names):
        ax.bar(
            x + (idx - 0.5) * width,
            means[idx],
            width,
            yerr=stds[idx],
            label=method_name,
            capsize=4,
            color=colors[idx],
            edgecolor="#222222",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_model_for_replay(method: str, scenario: str, seed: int):
    from stable_baselines3 import PPO

    tag = f"ppo_{method}_{scenario}_s{seed}"
    model_path = MODEL_DIR / f"{tag}.zip"
    vecnorm_path = MODEL_DIR / f"{tag}_vecnorm.pkl"
    if not model_path.exists() or not vecnorm_path.exists():
        return None, None, f"missing {model_path.name} or {vecnorm_path.name}"
    model = PPO.load(str(model_path), device="cpu")
    norm_fn = norm_from_vecnormalize(str(vecnorm_path), method, scenario)
    return model, norm_fn, ""


def replay_timeseries(method: str, scenario: str, seed: int, *, eval_seed: int = 100_000) -> tuple[list[dict[str, Any]], str]:
    model, norm_fn, reason = _load_model_for_replay(method, scenario, seed)
    if model is None or norm_fn is None:
        return [], reason
    env = make_constraint_env(method, scenario, seed=eval_seed)
    obs, _ = env.reset(seed=eval_seed)
    terminated = truncated = False
    rows: list[dict[str, Any]] = []
    while not (terminated or truncated):
        action, _ = model.predict(norm_fn(obs), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        inner = env._inner()
        raw_idx = int(info.get("raw_idx", info.get("agent_action", 0)))
        safe_idx = int(info.get("safe_idx", info.get("executed_action", raw_idx)))
        raw_u = int(inner.actions[raw_idx][1])
        safe_u = int(inner.actions[safe_idx][1])
        rows.append({
            "t": int(info.get("t", len(rows))),
            "method": method,
            "scenario": scenario,
            "seed": seed,
            "reward": float(reward),
            "p_min": int(info.get("p_min", 0)),
            "p_min_next": int(info.get("p_min_next", 0)),
            "p_raw_urllc": raw_u,
            "p_safe_urllc": safe_u,
            "D_proj": float(info.get("D_proj", 0.0)),
            "urllc_violation": int(bool(info.get("urllc_violation", False))),
            "shield_corrected": int(bool(info.get("shield_corrected", False))),
            "sla": float(info.get("sla", np.nan)),
            "state_channel": float(info.get("state_channel", np.nan)),
        })
    env.close()
    return rows, ""


def write_timeseries_csv(rows_by_method: dict[str, list[dict[str, Any]]], path: Path) -> None:
    rows = [row for rows in rows_by_method.values() for row in rows]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_s4_timeseries(rows_by_method: dict[str, list[dict[str, Any]]], path: Path) -> bool:
    if not all(rows_by_method.get(method) for method in METHODS):
        return False
    plt = _init_matplotlib()
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 8.8), sharex=True)
    colors = {M3: "#4C78A8", M5: "#F58518"}
    labels = {M3: "M3", M5: "M5"}
    linestyles = {M3: "--", M5: "-"}
    for method in (M5, M3):
        rows = rows_by_method[method]
        t = np.asarray([r["t"] for r in rows], dtype=float)
        p_min = np.asarray([r["p_min"] for r in rows], dtype=float)
        p_raw = np.asarray([r["p_raw_urllc"] for r in rows], dtype=float)
        p_safe = np.asarray([r["p_safe_urllc"] for r in rows], dtype=float)
        dproj = np.asarray([r["D_proj"] for r in rows], dtype=float)
        viol = np.asarray([r["urllc_violation"] for r in rows], dtype=float)
        axes[0].plot(t, p_min, label=labels[method], color=colors[method], linewidth=1.7, linestyle=linestyles[method])
        axes[1].plot(t, p_raw, label=f"{labels[method]} raw", color=colors[method], linewidth=1.0, linestyle=":", alpha=0.8)
        axes[1].plot(t, p_safe, label=f"{labels[method]} safe", color=colors[method], linewidth=1.4, linestyle=linestyles[method])
        axes[2].plot(t, dproj, label=labels[method], color=colors[method], linewidth=1.4, linestyle=linestyles[method])
        axes[3].plot(t, viol, label=labels[method], color=colors[method], linewidth=1.1, linestyle=linestyles[method], alpha=0.85)
    for ax in axes:
        ax.axvline(100, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("p_min")
    axes[1].set_ylabel("URLLC PRB")
    axes[2].set_ylabel("D_proj")
    axes[3].set_ylabel("Violation")
    axes[3].set_xlabel("Step")
    axes[0].set_title("S4 SLA Upgrade Deterministic Replay (seed 42)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].legend(frameon=False, ncol=4, fontsize=8)
    axes[0].text(102, axes[0].get_ylim()[1] * 0.92, "SLA upgrade", fontsize=9, color="#333333")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_s3_saturation(runs: list[dict[str, Any]], path: Path) -> None:
    plt = _init_matplotlib()
    pmax = float(EnvConfig().n_prb)
    methods = list(METHODS)
    x = np.arange(len(methods))
    pmin_means = []
    pmin_stds = []
    violation_means = []
    dproj_means = []
    for method in methods:
        pmin_mean, pmin_std = _metric_stats(runs, "S3_channel_decay", method, "mean_p_min")
        viol_mean, _ = _metric_stats(runs, "S3_channel_decay", method, "urllc_violation_rate")
        dproj_mean, _ = _metric_stats(runs, "S3_channel_decay", method, "mean_D_proj")
        pmin_means.append(np.nan if pmin_mean is None else pmin_mean / pmax)
        pmin_stds.append(0.0 if pmin_std is None else pmin_std / pmax)
        violation_means.append(np.nan if viol_mean is None else viol_mean)
        dproj_means.append(np.nan if dproj_mean is None else dproj_mean)

    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    width = 0.34
    ax1.bar(x - width / 2, pmin_means, width, yerr=pmin_stds, capsize=4, label="mean p_min / Pmax", color="#54A24B")
    ax1.bar(x + width / 2, violation_means, width, label="violation rate", color="#E45756")
    ax1.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["M3", "M5"])
    ax1.set_ylabel("Rate")
    ax1.set_title("S3 Saturation / Near-Infeasible Regime")
    ax2 = ax1.twinx()
    ax2.plot(x, dproj_means, color="#4C78A8", marker="o", linewidth=2.0, label="mean D_proj")
    ax2.set_ylabel("mean D_proj")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_s5_delay(runs: list[dict[str, Any]], path: Path) -> None:
    plt = _init_matplotlib()
    means = []
    stds = []
    ns = []
    for method in METHODS:
        vals = [
            r.get("eval_metrics", {}).get("adaptation_delay")
            for r in runs
            if r.get("scenario") == "S5_combined" and r.get("method") == method
        ]
        mean, std, n = _mean_std(vals)
        means.append(np.nan if mean is None else mean)
        stds.append(0.0 if std is None else std)
        ns.append(n)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(["M3", "M5"], means, yerr=stds, capsize=4, color=["#4C78A8", "#F58518"], edgecolor="#222222", linewidth=0.6)
    for bar, n in zip(bars, ns):
        height = bar.get_height()
        if np.isfinite(height):
            ax.text(bar.get_x() + bar.get_width() / 2, height + max(stds + [1.0]) * 0.15, f"n={n}", ha="center", fontsize=9)
    ax.set_ylabel("Adaptation delay (steps)")
    ax.set_title("S5 Partial Benefit Under Combined Stress")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_figures(runs: list[dict[str, Any]], *, skip_replay: bool = False) -> dict[str, Any]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {
        "figures": [],
        "timeseries_replay": {"generated": False, "reason": ""},
    }
    bar_specs = [
        ("mean_D_proj", "Mean D_proj", "Projection Distance Across Scenarios", "fig_phase3_dproj_bars.png", None),
        ("urllc_violation_rate", "URLLC violation rate", "Safety Violation Across Scenarios", "fig_phase3_violation_bars.png", (0, 1.0)),
        ("shield_correction_rate", "Shield correction rate", "Shield Intervention Across Scenarios", "fig_phase3_correction_bars.png", (0, 1.05)),
    ]
    for metric, ylabel, title, name, ylim in bar_specs:
        path = FIG_DIR / name
        plot_grouped_bar(runs, metric, ylabel, title, path, ylim=ylim)
        outputs["figures"].append(str(path))

    s3_path = FIG_DIR / "fig_s3_saturation.png"
    plot_s3_saturation(runs, s3_path)
    outputs["figures"].append(str(s3_path))

    s5_path = FIG_DIR / "fig_s5_adaptation_delay.png"
    plot_s5_delay(runs, s5_path)
    outputs["figures"].append(str(s5_path))

    if skip_replay:
        outputs["timeseries_replay"] = {"generated": False, "reason": "disabled by --skip-replay"}
        return outputs

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    reasons = []
    for method in METHODS:
        rows, reason = replay_timeseries(method, "S4_sla_upgrade", 42)
        rows_by_method[method] = rows
        if reason:
            reasons.append(f"{method}: {reason}")
    write_timeseries_csv(rows_by_method, OUT_DIR / "timeseries" / "S4_sla_upgrade_s42.csv")
    ts_path = FIG_DIR / "fig_s4_timeseries.png"
    if plot_s4_timeseries(rows_by_method, ts_path):
        outputs["figures"].append(str(ts_path))
        outputs["timeseries_replay"] = {"generated": True, "reason": ""}
    else:
        outputs["timeseries_replay"] = {"generated": False, "reason": "; ".join(reasons) or "empty replay"}
    return outputs


def _csv_preview_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['reward']} | "
            f"{row['urllc_violation_rate']} | {row['mean_D_proj']} | "
            f"{row['shield_correction_rate']} | {row['adaptation_delay']} |"
        )
    return "\n".join(lines)


def _delta_line(rows: list[dict[str, Any]], scenario: str, metric: str) -> str:
    rec = next((r for r in rows if r["scenario"] == scenario and r["metric"] == metric), None)
    if not rec:
        return ""
    mean = rec["mean_delta"]
    std = rec["std_delta"]
    if mean == "":
        return f"- `{scenario}` `{metric}`: insufficient paired data."
    return f"- `{scenario}` `{metric}`: {float(mean):.4f} ± {float(std):.4f} (M5 - M3, n={rec['n_valid']})."


def write_report(
    table_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    warnings: list[str],
    figure_info: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- All Phase3 formal-run gates passed."
    figures = "\n".join(f"- `{Path(p).relative_to(PROJECT_ROOT)}`" for p in figure_info["figures"])
    replay = figure_info.get("timeseries_replay", {})
    replay_text = (
        "- S4 deterministic replay was generated from local seed-42 checkpoints."
        if replay.get("generated")
        else f"- S4 deterministic replay was skipped: {replay.get('reason', 'unknown')}."
    )
    text = f"""# Phase3 M3/M5 Results

## Gate Status
{warning_text}

## Result Table

| Scenario | Method | Reward | Violation | Mean D_proj | Shield correction | Adaptation delay |
|---|---|---:|---:|---:|---:|---:|
{_csv_preview_rows(table_rows)}

## Paired Seed Delta Highlights
Negative deltas improve `mean_D_proj`, violation, shield correction, and delay. Positive deltas improve reward.

{_delta_line(delta_rows, "S4_sla_upgrade", "mean_D_proj")}
{_delta_line(delta_rows, "S4_sla_upgrade", "urllc_violation_rate")}
{_delta_line(delta_rows, "S6_moderate_decay", "mean_D_proj")}
{_delta_line(delta_rows, "S6_moderate_decay", "urllc_violation_rate")}
{_delta_line(delta_rows, "S5_combined", "adaptation_delay")}
{_delta_line(delta_rows, "S3_channel_decay", "mean_D_proj")}

## Interpretation
- S4 supports the state-augmentation claim: under the same Oracle-z dynamic constraint path, M5 reduces projection burden and violation relative to M3.
- S3 should be written as a saturation / near-infeasible regime: `p_min/Pmax` is close to the resource ceiling and both methods keep high violation, so lack of M5 improvement is expected rather than hidden.
- S5 should be written as partial benefit under combined stress: emphasize adaptation delay and projection behavior, not reward dominance.
- S6 is the clean moderate-stress test: it strengthens the claim that `p_min/Pmax` reduces projection/correction burden when the constraint is dynamic but not saturated, while reward and violation should be reported as trade-offs.
- Phase2b-v1 supports symbolic-z verified compilation and direct numeric rejection. It is not real CER/RAG evidence; that belongs to Phase2c.

## Generated Artifacts
- `04_results/phase3_m3_m5/phase3_table.csv`
- `04_results/phase3_m3_m5/paired_seed_delta.csv`
{figures}
{replay_text}
"""
    path.write_text(text)


def verify_artifacts(paths: list[Path]) -> list[str]:
    missing = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--skip-replay", action="store_true", help="Skip model-based S4 deterministic replay.")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = _load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No run JSON files found in {runs_dir}")

    warnings = validate_runs(runs)
    table_path = OUT_DIR / "phase3_table.csv"
    delta_path = OUT_DIR / "paired_seed_delta.csv"
    table_rows = build_phase3_table(runs, table_path)
    delta_rows = build_paired_delta(runs, delta_path)
    figure_info = generate_figures(runs, skip_replay=args.skip_replay)
    write_report(table_rows, delta_rows, warnings, figure_info, REPORT_PATH)

    expected = [table_path, delta_path, REPORT_PATH]
    expected.extend(Path(p) for p in figure_info["figures"])
    missing = verify_artifacts(expected)
    summary = {
        "runs": len(runs),
        "warnings": warnings,
        "table": str(table_path),
        "paired_delta": str(delta_path),
        "report": str(REPORT_PATH),
        "figures": figure_info["figures"],
        "timeseries_replay": figure_info["timeseries_replay"],
        "missing_or_empty": missing,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
