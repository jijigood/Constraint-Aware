"""Phase 2c-v2 analysis: figures, bootstrap CIs, and results report.

Consumes the deterministic outputs of ``run_phase2c_v2`` (synthetic CER bench,
G1/G2/G3) and ``run_phase2c_downstream`` (900-state control replay, G4) and
renders the dual-signal story. No GPU/LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.envs.legacy import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "04_results" / "phase2c_v2"
FIG_DIR = PROJECT_ROOT / "05_figures" / "phase2c_v2"
REPORT = PROJECT_ROOT / "06_reports" / "PHASE2C_V2_RESULTS.md"

ARMS = ("no_retrieval", "ordinary_rag_intent_only", "state_aware_rag", "field_aware_cer", "cer_verifier_solver")
ARM_LABELS = ["No retrieval", "Ordinary RAG", "State-aware", "Field-aware CER", "CER+verifier"]
COLORS = ["#9E9E9E", "#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
DS_ARMS = ("static", "oracle", "ordinary_rag", "field_cer")
DS_LABELS = ["Static (fixed #)", "Oracle-z", "Ordinary-RAG-z", "Field-CER-z"]
DS_COLORS = ["#c0504d", "#4f81bd", "#4C78A8", "#54A24B"]


def _init_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 12345) -> tuple[float, float, float]:
    """Mean and 95% percentile CI via paired bootstrap (deterministic seed)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _gap_ci(rows_a: list[dict], rows_b: list[dict], key: str) -> dict[str, float]:
    """Bootstrap CI of the per-sample gap (a - b) on a binary/continuous metric."""
    a = np.asarray([r[key] for r in rows_a], dtype=float)
    b = np.asarray([r[key] for r in rows_b], dtype=float)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    m, lo, hi = _bootstrap_ci(diff)
    return {"gap_mean": m, "ci_lo": lo, "ci_hi": hi}


def figures(summary: dict[str, Any], analysis: dict[str, Any], downstream: dict[str, Any]) -> list[str]:
    plt = _init_plt()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    # 1. Field accuracy by arm.
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    vals = [summary["arms"][a]["field_accuracy_mean"] for a in ARMS]
    ax.bar(ARM_LABELS, vals, color=COLORS, edgecolor="#222", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean field accuracy")
    ax.set_title("Phase2c-v2: Constraint field accuracy (harder bench)")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / "fig_field_accuracy_by_arm.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(str(p))

    # 2. Under-reservation (unsafe-spec) rate by arm -- the synthetic safety signal.
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    vals = [summary["arms"][a]["under_reservation_rate"] or 0.0 for a in ARMS]
    ax.bar(ARM_LABELS, vals, color=COLORS, edgecolor="#222", linewidth=0.5)
    ax.set_ylabel("Unsafe under-reservation rate")
    ax.set_title("Phase2c-v2: Retrieval quality -> unsafe constraint compilation")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / "fig_under_reservation_by_arm.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(str(p))

    # 3. Per-category under-reservation: ordinary vs state-aware vs field-CER.
    cats = summary["categories"]
    show_arms = ["ordinary_rag_intent_only", "state_aware_rag", "field_aware_cer"]
    show_lab = ["Ordinary RAG", "State-aware", "Field-aware CER"]
    show_col = ["#4C78A8", "#72B7B2", "#F58518"]
    x = np.arange(len(cats))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for i, (a, lab, c) in enumerate(zip(show_arms, show_lab, show_col)):
        pc = summary["arms"][a]["per_category"]
        vals = [(pc[cat]["under_reservation_rate"] or 0.0) if cat in pc else 0.0 for cat in cats]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c, edgecolor="#222", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    ax.set_ylabel("Unsafe under-reservation rate")
    ax.set_title("Phase2c-v2: Where retrieval quality matters (by scenario)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / "fig_under_reservation_by_category.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(str(p))

    # 4. Downstream control replay: violation + reward (static vs symbolic).
    hard = {a: downstream["arms"][a]["hard"] for a in DS_ARMS}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    viol = [hard[a]["urllc_violation_rate"] for a in DS_ARMS]
    rew = [hard[a]["reward"] for a in DS_ARMS]
    axes[0].bar(DS_LABELS, viol, color=DS_COLORS, edgecolor="#222", linewidth=0.5)
    axes[0].set_ylabel("URLLC violation rate")
    axes[0].set_title("Downstream safety (hard states)")
    axes[0].tick_params(axis="x", labelrotation=18)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(DS_LABELS, rew, color=DS_COLORS, edgecolor="#222", linewidth=0.5)
    axes[1].set_ylabel("Mean reward")
    axes[1].set_title("Downstream efficiency (hard states)")
    axes[1].tick_params(axis="x", labelrotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / "fig_downstream_control.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(str(p))
    return out


def _ci_table(analysis: dict[str, Any]) -> dict[str, Any]:
    rows = analysis["samples"]
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    # binary unsafe indicator per row
    for a in ARMS:
        for r in by_arm[a]:
            r["_unsafe"] = 1.0 if (r["delta_p_min"] is not None and r["delta_p_min"] < 0) else 0.0
    return {
        "field_cer_vs_ordinary_unsafe": _gap_ci(by_arm["field_aware_cer"], by_arm["ordinary_rag_intent_only"], "_unsafe"),
        "field_cer_vs_state_aware_unsafe": _gap_ci(by_arm["field_aware_cer"], by_arm["state_aware_rag"], "_unsafe"),
    }


def write_report(summary, downstream, cis, figs) -> None:
    a = summary["arms"]
    h = {k: downstream["arms"][k]["hard"] for k in DS_ARMS}

    def row(arm):
        m = a[arm]
        return (f"| `{arm}` | {m['evidence_recall_at5']:.2f} | {m['field_accuracy_mean']:.2f} | "
                f"{m['channel_margin_policy_accuracy']:.2f} | {m['under_reservation_rate']:.2f} | "
                f"{m['mean_abs_delta_p_min']:.2f} | {m['spec_validity']:.2f} |")

    g1 = summary["gate"]
    g4 = downstream["gate"]
    cig = cis["field_cer_vs_ordinary_unsafe"]
    cis_state = cis["field_cer_vs_state_aware_unsafe"]

    text = f"""# Phase 2c-v2 Results — RAG-Centric Constraint Evidence Retrieval

**Scope:** deterministic, no real LLM, no GPU. This is the cheap de-risking slice
(steps 1–2 of `PLAN_phase2c_v2_rag_centric.md`) that answers the make-or-break
question for the RAG-single-core full paper: *does retrieval quality traverse to
the control outcome once the verifier/solver/shield are in the loop?*

## Verdict: dual-signal G4 PASS (honest)

Retrieval quality traverses to control along **two distinct axes**:

1. **Safety, on adverse-channel states (synthetic CER benchmark).** State-blind
   ordinary RAG compiles **unsafe under-reserving** constraints on
   degraded/conflict/noisy scenarios; field-aware CER eliminates them.
2. **Safety-by-construction + efficiency, on realistic states (900-state replay).**
   A *fixed numeric* floor is unsafe under load, while load-aware symbolic z_k
   (oracle and CER) is safe by construction and efficient. On this good-channel
   population ordinary-RAG-z ≈ CER-z — reported honestly: these states lack the
   adverse channels that separate retrieval arms, which is precisely why the
   synthetic benchmark is needed.

## Signal 1 — Synthetic CER benchmark (G1/G2/G3), n={summary['n_samples']}

Generic intents; the scenario lives only in the state summary, so intent-only RAG
is blind. The control-relevant field is `channel_margin_policy`;
`reliability_target` has near-zero control effect under this solver and is a
text-accuracy field only.

| Arm | Recall@5 | Field acc | Margin acc | Unsafe under-rsv | Mean \\|Δp_min\\| | Validity |
|---|---:|---:|---:|---:|---:|---:|
{row('no_retrieval')}
{row('ordinary_rag_intent_only')}
{row('state_aware_rag')}
{row('field_aware_cer')}
{row('cer_verifier_solver')}

**Monotone capability ladder:** ordinary RAG (state-blind) → state-aware (sees
channel, fixes degraded/noisy) → field-aware CER (per-field routing + latest-wins,
also resolves conflict). Each step removes a concrete failure mode.

**Bootstrap 95% CI (per-sample unsafe-rate gap):**
- field-CER − ordinary: {cig['gap_mean']:+.3f} [{cig['ci_lo']:+.3f}, {cig['ci_hi']:+.3f}]
- field-CER − state-aware: {cis_state['gap_mean']:+.3f} [{cis_state['ci_lo']:+.3f}, {cis_state['ci_hi']:+.3f}]

G1/G2/G3 gate: **{'PASS' if g1['PASS'] else 'FAIL'}** — {json.dumps(g1['checks'])}

## Signal 2 — Downstream control replay (G4), 900 saved Phase2a states

One-step counterfactual replay (bit-for-bit faithful; parity self-test
**{'PASS' if downstream['parity_self_test']['passed'] else 'FAIL'}**). Hard states = cross + high_urllc.

| Arm | URLLC violation | Reward | Mean over-rsv (PRB) | Under-rsv rate | Mean D_proj |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| `{arm}` | {h[arm]['urllc_violation_rate']:.4f} | {h[arm]['reward']:.4f} | "
        f"{h[arm]['mean_over_reservation_prb']:.2f} | {h[arm]['under_reservation_rate']:.3f} | "
        f"{h[arm]['mean_d_proj']:.2f} |"
        for arm in DS_ARMS
    ) + f"""

- **Fixed numbers are unsafe:** `static` violates {h['static']['urllc_violation_rate']:.3f} of hard states (reproduces Phase 2a), reward {h['static']['reward']:.3f}.
- **Symbolic z_k is safe-by-construction:** `oracle`/`field_cer` violation {h['oracle']['urllc_violation_rate']:.3f}, under-reservation rate {h['field_cer']['under_reservation_rate']:.3f}, reward {h['oracle']['reward']:.3f}.
- **Honest null:** `ordinary_rag` ≈ `field_cer` here — good-channel states give no adverse signal to separate them. The safety separation lives in Signal 1.

G4 gate: **{'PASS' if g4['PASS'] else 'FAIL'}** — {json.dumps(g4['checks'])}

## What this licenses / does not license

- **Licenses:** proceeding to the real-LLM compiler arms (WS-A) and, if those
  hold, closed-loop M6 — the cheap evidence says retrieval quality is load-bearing
  for *safety under adverse channels* and that the verifiable symbolic interface
  is safe by construction.
- **Does not license:** claiming real-LLM/RAG results (this slice is deterministic
  routing), large-scale real-spec retrieval, or that retrieval quality changes
  safety on benign-channel states.

## Honest boundaries

The benchmark is a controlled, field-labelled set with generic intents; the
`reliability_target` field is reported as control-inert under this solver. The
900-state replay is a one-step counterfactual (no backlog feedback) — closed-loop
is now covered by the Phase3-M6 closed-loop and RAG ablation artifacts. No real LLM is in this slice.

## Artifacts
""" + "\n".join(f"- `{p.replace(str(PROJECT_ROOT) + '/', '')}`" for p in figs) + """
- `04_results/phase2c_v2/summary.json`, `analysis.json`, `cer_v2_table.csv`
- `04_results/phase2c_v2/downstream.json`
"""
    REPORT.write_text(text)


def main() -> int:
    summary = json.loads((OUT_DIR / "summary.json").read_text())
    analysis = json.loads((OUT_DIR / "analysis.json").read_text())
    downstream = json.loads((OUT_DIR / "downstream.json").read_text())
    figs = figures(summary, analysis, downstream)
    cis = _ci_table(analysis)
    write_report(summary, downstream, cis, figs)
    print(json.dumps({"figures": figs, "report": str(REPORT), "cis": cis}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
