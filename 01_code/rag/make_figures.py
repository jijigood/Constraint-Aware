"""
Render Phase 2a figures from summary.json (run in ~/safe_drl_oran/.venv, which has matplotlib).
Reads aggregates only — figures and JSON can never disagree.

Run:  ~/safe_drl_oran/.venv/bin/python 01_code/rag/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE2A = os.path.join(PROJ, "04_results", "phase2a")
FIG_DIR = os.path.join(PROJ, "05_figures", "phase2a")
SETS = ["cross", "high_urllc", "high_embb"]
EVAL_REGIME = {"cross": "high_urllc", "high_urllc": "high_urllc", "high_embb": "high_embb"}
ARMS = ["static", "oracle_margin", "llm_no_rag", "rag_llm"]
COLORS = {"static": "#c0504d", "oracle_margin": "#4f81bd", "llm_no_rag": "#9bbb59", "rag_llm": "#8064a2"}


def _err(rec, key):
    m = rec[key]["mean"]
    return max(0, m - rec[key]["lo"]), max(0, rec[key]["hi"] - m)


def fig_arm_safety(summary, phase0):
    fig, axes = plt.subplots(1, len(SETS), figsize=(5.3 * len(SETS), 4.3), squeeze=False)
    for j, sname in enumerate(SETS):
        ax = axes[0][j]
        arms = summary["sets"][sname]["arms"]
        labels = [a for a in ARMS if a in arms]
        vals = [arms[a]["urllc_violation_rate"]["mean"] for a in labels]
        lo = [_err(arms[a], "urllc_violation_rate")[0] for a in labels]
        hi = [_err(arms[a], "urllc_violation_rate")[1] for a in labels]
        ax.bar(range(len(labels)), vals, yerr=[lo, hi], capsize=3, color=[COLORS[a] for a in labels])
        oracle_ref = phase0["regimes"].get(EVAL_REGIME[sname], {}).get("oracle_margin", {}).get("urllc_violation_rate")
        if oracle_ref is not None:
            ax.axhline(oracle_ref, ls=":", c="green", lw=1, label=f"Phase0 oracle ({oracle_ref:.2f})")
            ax.legend(fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([a.replace("_", "\n") for a in labels], fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_ylabel("URLLC violation rate")
        ax.set_title(f"{sname}")
    fig.suptitle("Phase 2a — URLLC violation by arm (one-step counterfactual, 3 held-out sets)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_arm_safety.png"), dpi=130); plt.close(fig)


def fig_reward_safety(summary, phase0):
    fig, axes = plt.subplots(1, len(SETS), figsize=(5.3 * len(SETS), 4.3), squeeze=False)
    marks = {"static": "s", "oracle_margin": "*", "llm_no_rag": "P", "rag_llm": "^"}
    for j, sname in enumerate(SETS):
        ax = axes[0][j]
        arms = summary["sets"][sname]["arms"]
        p0 = phase0["regimes"].get(EVAL_REGIME[sname], {})
        st = p0.get("static", {})
        if st:
            xs = [st[k]["reward"] for k in st]; ys = [st[k]["urllc_violation_rate"] for k in st]
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], "-o", c="gray", ms=3, alpha=0.5,
                    label="Phase0 static frontier")
        for a in ARMS:
            if a in arms:
                ax.scatter(arms[a]["reward"]["mean"], arms[a]["urllc_violation_rate"]["mean"],
                           marker=marks[a], s=120, c=COLORS[a], edgecolor="k", lw=0.5, label=a, zorder=5)
        ax.set_xlabel("reward (higher better)"); ax.set_ylabel("URLLC violation (lower better)")
        ax.set_title(f"{sname}"); ax.legend(fontsize=6)
    fig.suptitle("Phase 2a — reward vs URLLC violation (arms vs Phase-0 static frontier)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_reward_safety.png"), dpi=130); plt.close(fig)


def fig_rag_vs_norag(summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(SETS)); w = 0.38
    for k, arm in enumerate(["llm_no_rag", "rag_llm"]):
        v = [summary["sets"][s]["arms"][arm]["urllc_violation_rate"]["mean"] for s in SETS]
        ax1.bar(x + (k - 0.5) * w, v, w, label=arm, color=COLORS[arm])
    ax1.set_xticks(x); ax1.set_xticklabels(SETS); ax1.set_ylabel("URLLC violation rate")
    ax1.set_title("RAG vs no-RAG: safety"); ax1.legend(fontsize=8)
    for k, arm in enumerate(["llm_no_rag", "rag_llm"]):
        c = [summary["sets"][s]["arms"][arm].get("citation_validity", 0) for s in SETS]
        ax2.bar(x + (k - 0.5) * w, c, w, label=arm, color=COLORS[arm])
    ax2.set_xticks(x); ax2.set_xticklabels(SETS); ax2.set_ylabel("citation validity"); ax2.set_ylim(0, 1.05)
    ax2.set_title("RAG vs no-RAG: citation grounding"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_rag_vs_norag.png"), dpi=130); plt.close(fig)


def fig_reservation(summary, phase0):
    """Mean URLLC reservation per arm vs oracle, per set (did the LLM learn the load-aware level)."""
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = np.arange(len(SETS)); w = 0.2
    for k, arm in enumerate(ARMS):
        vals = [summary["sets"][s]["arms"][arm]["mean_prb_urllc"] for s in SETS]
        ax.bar(x + (k - 1.5) * w, vals, w, label=arm, color=COLORS[arm])
    ax.set_xticks(x); ax.set_xticklabels(SETS); ax.set_ylabel("mean URLLC PRB reserved")
    ax.set_title("Phase 2a — mean URLLC reservation by arm (oracle = load-aware target)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_reservation.png"), dpi=130); plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summary = json.load(open(os.path.join(PHASE2A, "summary.json")))
    phase0 = json.load(open(os.path.join(PROJ, "04_results", "phase0_headroom.json")))
    fig_arm_safety(summary, phase0)
    fig_reward_safety(summary, phase0)
    fig_rag_vs_norag(summary)
    fig_reservation(summary, phase0)
    print(f"figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
