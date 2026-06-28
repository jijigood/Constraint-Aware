"""
Aggregate the Phase 2a-v2 model-size sweep: read summary_{tag}.json + analysis_{tag}.json per model,
build the constraint-quality-vs-size table + figure (run in ~/safe_drl_oran/.venv for matplotlib).

Run:  ~/safe_drl_oran/.venv/bin/python 01_code/rag/sweep_analyze.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE2A = os.path.join(PROJ, "04_results", "phase2a")
FIG_DIR = os.path.join(PROJ, "05_figures", "phase2a")
ORDER = [("1p7b", 1.7), ("4b", 4.0), ("14b", 14.0), ("32b", 32.0)]


def load(tag):
    sp = os.path.join(PHASE2A, f"summary_{tag}.json")
    ap = os.path.join(PHASE2A, f"analysis_{tag}.json")
    if not (os.path.exists(sp) and os.path.exists(ap)):
        return None
    return json.load(open(sp)), json.load(open(ap))


def main():
    rows, present = [], []
    static_ref = oracle_ref = None
    for tag, size in ORDER:
        got = load(tag)
        if not got:
            print(f"  (missing {tag}; skipping)")
            continue
        s, a = got
        cross = s["sets"]["cross"]["arms"]
        static_ref = cross["static"]["urllc_violation_rate"]["mean"]
        oracle_ref = cross["oracle_margin"]["urllc_violation_rate"]["mean"]
        g = a["gate"]
        row = {
            "tag": tag, "size_b": size, "paper_usable": s.get("paper_usable"),
            "cross_viol_norag": cross["llm_no_rag"]["urllc_violation_rate"]["mean"],
            "cross_viol_rag": cross["rag_llm"]["urllc_violation_rate"]["mean"],
            "cross_reward_rag": cross["rag_llm"]["reward"]["mean"],
            "cross_resv_rag": cross["rag_llm"]["mean_prb_urllc"],
            "schema_rag": cross["rag_llm"].get("schema_validity"),
            "cite_rag": cross["rag_llm"].get("citation_validity"),
            "PASS": g["PASS"], "rag_adds_value": g.get("rag_adds_value"),
            "verdict": g["verdict"][:60],
        }
        rows.append(row); present.append((tag, size))

    out = {"schema_version": "safe_drl_v1", "kind": "phase2a_size_sweep",
           "cross_static_ref": static_ref, "cross_oracle_ref": oracle_ref, "models": rows}
    with open(os.path.join(PHASE2A, "sweep_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # table
    print(f"\n{'='*92}\nMODEL-SIZE SWEEP — cross set (static_ref={static_ref}, oracle_ref={oracle_ref})\n{'='*92}")
    print(f"{'model':<7}{'size_b':>7}{'viol_norag':>12}{'viol_rag':>10}{'reward_rag':>12}"
          f"{'resv_rag':>10}{'schema':>8}{'cite':>7}{'PASS':>7}")
    for r in rows:
        print(f"{r['tag']:<7}{r['size_b']:>7.1f}{r['cross_viol_norag']:>12.3f}{r['cross_viol_rag']:>10.3f}"
              f"{r['cross_reward_rag']:>12.3f}{r['cross_resv_rag']:>10.0f}{(r['schema_rag'] or 0):>8.2f}"
              f"{(r['cite_rag'] or 0):>7.2f}{str(r['PASS']):>7}")

    # figure: cross URLLC violation vs model size
    if rows:
        sizes = [r["size_b"] for r in rows]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(sizes, [r["cross_viol_rag"] for r in rows], "-o", c="#8064a2", label="rag_llm")
        ax.plot(sizes, [r["cross_viol_norag"] for r in rows], "-s", c="#9bbb59", label="llm_no_rag")
        if static_ref is not None:
            ax.axhline(static_ref, ls="--", c="#c0504d", label=f"static floor ({static_ref:.2f})")
        if oracle_ref is not None:
            ax.axhline(oracle_ref, ls=":", c="#4f81bd", label=f"oracle ({oracle_ref:.2f})")
        ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{s:g}B" for s in sizes])
        ax.set_xlabel("producer model size (params, log)")
        ax.set_ylabel("cross-set URLLC violation (lower better)")
        ax.set_ylim(0, max(static_ref or 0.7, 0.7) * 1.05)
        ax.set_title("Phase 2a-v2 — constraint quality vs model size (decisive cross set)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_size_sweep.png"), dpi=130); plt.close(fig)
        print(f"\nfigure -> {os.path.join(FIG_DIR, 'fig_size_sweep.png')}")
    print(f"sweep_summary -> {os.path.join(PHASE2A, 'sweep_summary.json')}")


if __name__ == "__main__":
    main()
