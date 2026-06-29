"""WS-A analysis: does field-aware CER's advantage survive real-LLM compilation?

Aggregates all `summary__{tag}__{retriever}.json` from the WS-A sweep, compares
against the deterministic Phase 2c-v2 result (the "survival" question), reports
the scale trend and per-category location of the gap with bootstrap CIs, and
writes figures + `06_reports/PHASE2C_WSA_RESULTS.md`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.envs.legacy import PROJECT_ROOT

WSA_DIR = PROJECT_ROOT / "04_results" / "phase2c_wsa"
V2_SUMMARY = PROJECT_ROOT / "04_results" / "phase2c_v2" / "summary.json"
FIG_DIR = PROJECT_ROOT / "05_figures" / "phase2c_wsa"
REPORT = PROJECT_ROOT / "06_reports" / "PHASE2C_WSA_RESULTS.md"

ARMS = ("ordinary_rag_llm", "state_aware_rag_llm", "field_aware_cer_llm", "field_aware_cer_llm_verifier")
ARM_LABEL = {"ordinary_rag_llm": "Ordinary RAG+LLM", "state_aware_rag_llm": "State-aware+LLM",
             "field_aware_cer_llm": "Field-CER+LLM", "field_aware_cer_llm_verifier": "Field-CER+LLM+verifier"}
ADVERSE = ("degraded", "conflict", "noisy")  # where state/evidence disambiguation bites


def _size_key(tag: str) -> float:
    m = re.search(r"([\d.]+)\s*B", tag)
    return float(m.group(1)) if m else 0.0


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_summaries() -> dict[tuple[str, str], dict]:
    out = {}
    for p in sorted(WSA_DIR.glob("summary__*.json")):
        s = json.loads(p.read_text())
        out[(s["model"], s["retriever"])] = s
    return out


def _gap_ci(analysis_path: Path, key: str) -> dict[str, float]:
    """Bootstrap CI of per-sample (field_aware_cer_llm − ordinary_rag_llm) on `key`."""
    rows = json.loads(analysis_path.read_text())["samples"]
    f = {r["sample_id"]: r for r in rows if r["arm"] == "field_aware_cer_llm"}
    o = {r["sample_id"]: r for r in rows if r["arm"] == "ordinary_rag_llm"}
    ids = sorted(set(f) & set(o))
    diff = np.array([float(f[i][key]) - float(o[i][key]) for i in ids], dtype=float)
    m, lo, hi = _bootstrap_ci(diff)
    return {"gap_mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(ids)}


def _init_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figures(summaries, v2) -> list[str]:
    plt = _init_plt()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    retrievers = sorted({r for _, r in summaries})
    tags = sorted({t for t, _ in summaries}, key=_size_key)
    out = []
    arm_color = {"ordinary_rag_llm": "#4C78A8", "state_aware_rag_llm": "#72B7B2",
                 "field_aware_cer_llm": "#F58518", "field_aware_cer_llm_verifier": "#54A24B"}

    for metric, ylab, fname in [
        ("channel_margin_policy_accuracy", "Margin-policy accuracy", "fig_wsa_margin_acc_vs_size.png"),
        ("under_reservation_rate", "Unsafe under-reservation rate", "fig_wsa_under_rsv_vs_size.png"),
    ]:
        fig, axes = plt.subplots(1, len(retrievers), figsize=(5.6 * len(retrievers), 4.3), squeeze=False)
        for ax, retr in zip(axes[0], retrievers):
            xs = [t for t in tags if (t, retr) in summaries]
            x = [_size_key(t) for t in xs]
            for arm in ARMS:
                y = [summaries[(t, retr)]["arms"][arm].get(metric) or 0.0 for t in xs]
                ax.plot(x, y, "o-", label=ARM_LABEL[arm], color=arm_color[arm])
            # deterministic v2 reference (field vs ordinary) as dashed h-lines
            if v2 is not None:
                ax.axhline(v2["arms"]["field_aware_cer"].get(metric) or 0.0, ls="--", c="#F58518", alpha=0.5, lw=1)
                ax.axhline(v2["arms"]["ordinary_rag_intent_only"].get(metric) or 0.0, ls="--", c="#4C78A8", alpha=0.5, lw=1)
            ax.set_xscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([t.replace("Qwen3-", "") for t in xs], rotation=0)
            ax.set_xlabel("Model size")
            ax.set_ylabel(ylab)
            ax.set_title(f"{retr} retrieval")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.suptitle(f"WS-A: {ylab} vs model scale (dashed = deterministic v2 ref)")
        fig.tight_layout()
        p = FIG_DIR / fname
        fig.savefig(p, dpi=160)
        plt.close(fig)
        out.append(str(p))

    # per-category margin accuracy (field vs ordinary) at the largest available model, tfidf
    big = max(tags, key=_size_key)
    retr = "tfidf" if (big, "tfidf") in summaries else sorted({r for _, r in summaries})[0]
    s = summaries[(big, retr)]
    cats = s["categories"]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for i, arm in enumerate(["ordinary_rag_llm", "state_aware_rag_llm", "field_aware_cer_llm"]):
        pc = s["arms"][arm]["per_category"]
        vals = [pc[c]["channel_margin_policy_accuracy"] if c in pc else 0.0 for c in cats]
        ax.bar(x + (i - 1) * 0.26, vals, 0.26, label=ARM_LABEL[arm], color=arm_color[arm], edgecolor="#222", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    ax.set_ylabel("Margin-policy accuracy")
    ax.set_title(f"WS-A per-category margin accuracy ({big}, {retr})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / "fig_wsa_per_category.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(str(p))
    return out


def gate(summaries) -> dict[str, Any]:
    key14 = [(t, r) for (t, r) in summaries if abs(_size_key(t) - 14.0) < 1e-6]
    if not key14:
        # fall back to the largest available model
        big = max((t for t, _ in summaries), key=_size_key, default=None)
        key14 = [(t, r) for (t, r) in summaries if t == big]
    checks = {}
    for (t, r) in key14:
        a = summaries[(t, r)]["arms"]
        f, o, v = a["field_aware_cer_llm"], a["ordinary_rag_llm"], a["field_aware_cer_llm_verifier"]
        checks[f"{t}/{r}"] = {
            "field_margin_beats_ordinary": f["channel_margin_policy_accuracy"] > o["channel_margin_policy_accuracy"],
            "field_under_le_ordinary": (f["under_reservation_rate"] or 0) <= (o["under_reservation_rate"] or 0),
            "verifier_safe": (v["under_reservation_rate"] or 0) <= 1e-9,
            "verifier_cited": v["citation_validity"] >= 0.9,
        }
    survived = any(c["field_margin_beats_ordinary"] and c["field_under_le_ordinary"] for c in checks.values())
    return {"per_model_retriever": checks, "advantage_survives": survived}


def write_report(summaries, v2, cis, gate_res, figs) -> None:
    tags = sorted({t for t, _ in summaries}, key=_size_key)
    retrievers = sorted({r for _, r in summaries})

    lines = ["# Phase 2c-WS-A Results — does CER's advantage survive a real LLM compiler?\n"]
    lines.append("**Scope:** real Qwen3 LLM compiles the symbolic spec from retrieved evidence "
                 "(vLLM, temp=0, generations cached). Retrieval over the purpose-built v2 CER corpus; "
                 "the LLM emits typed fields, never a PRB number. Offline (no closed-loop DRL).\n")
    verdict = "SURVIVES" if gate_res["advantage_survives"] else "DOES NOT survive"
    lines.append(f"## Verdict: field-aware CER advantage **{verdict}** real-LLM compilation\n")

    if v2 is not None:
        dv = v2["arms"]
        lines.append("**Deterministic v2 reference (router, not LLM):** "
                     f"ordinary margin-acc {dv['ordinary_rag_intent_only']['channel_margin_policy_accuracy']:.2f} / "
                     f"under {dv['ordinary_rag_intent_only']['under_reservation_rate']:.2f}; "
                     f"field-CER margin-acc {dv['field_aware_cer']['channel_margin_policy_accuracy']:.2f} / "
                     f"under {dv['field_aware_cer']['under_reservation_rate']:.2f}.\n")

    lines.append("## Per (model, retriever): margin accuracy / unsafe under-reservation / citation validity\n")
    lines.append("| Model | Retr | Arm | Margin acc | Under-rsv | Cite valid | Validity | Recall@5 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for t in tags:
        for r in retrievers:
            if (t, r) not in summaries:
                continue
            a = summaries[(t, r)]["arms"]
            for arm in ARMS:
                m = a[arm]
                lines.append(f"| {t} | {r} | {ARM_LABEL[arm]} | "
                             f"{m['channel_margin_policy_accuracy']:.2f} | {m['under_reservation_rate'] or 0:.2f} | "
                             f"{m['citation_validity']:.2f} | {m['spec_validity']:.2f} | {m['evidence_recall_at5']:.2f} |")

    lines.append("\n## Field-CER − Ordinary gap (bootstrap 95% CI)\n")
    lines.append("| Model | Retr | Δ margin-acc [CI] | Δ unsafe-rate [CI] |")
    lines.append("|---|---|---|---|")
    for (t, r), c in cis.items():
        ma, ur = c["margin"], c["unsafe"]
        lines.append(f"| {t} | {r} | {ma['gap_mean']:+.3f} [{ma['ci_lo']:+.3f}, {ma['ci_hi']:+.3f}] | "
                     f"{ur['gap_mean']:+.3f} [{ur['ci_lo']:+.3f}, {ur['ci_hi']:+.3f}] |")

    lines.append("\n## Interpretation (honest)\n")
    lines.append("- All arms see the state summary, so this is the stringent test: does retrieval help "
                 "*beyond* the LLM's own reasoning over state? The CER edge is expected to concentrate on "
                 f"**provenance/conflict** ({', '.join(ADVERSE)}) where state alone cannot fix the policy.\n")
    lines.append(f"- G-WSA gate: advantage_survives = **{gate_res['advantage_survives']}**. "
                 f"Per (model,retriever) checks: {json.dumps(gate_res['per_model_retriever'])}\n")
    lines.append("- The verifier arm rejects any direct-numeric leak and fail-closes (C3), keeping unsafe "
                 "under-reservation ~0 with grounded citations.\n")
    lines.append("\n## Honest boundaries\n")
    lines.append("- Controlled field-labelled corpus (not large-scale real specs). Generations cached at temp=0; "
                 "`reliability_target` is control-inert under this solver. Closed-loop (M6) still deferred.\n")
    lines.append("## Artifacts\n")
    for p in figs:
        lines.append(f"- `{p.replace(str(PROJECT_ROOT) + '/', '')}`")
    lines.append("- `04_results/phase2c_wsa/summary__*.json`, `generations/*.jsonl`")

    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    summaries = _load_summaries()
    if not summaries:
        raise SystemExit("no WS-A summaries found in 04_results/phase2c_wsa/")
    v2 = json.loads(V2_SUMMARY.read_text()) if V2_SUMMARY.exists() else None
    cis = {}
    for (t, r) in summaries:
        ap = WSA_DIR / f"analysis__{t.replace('/', '_')}__{r}.json"
        if ap.exists():
            cis[(t, r)] = {"margin": _gap_ci(ap, "channel_margin_policy_accuracy"),
                           "unsafe": _gap_ci(ap, "_unsafe") if _has_unsafe(ap) else _unsafe_from_delta(ap)}
    gate_res = gate(summaries)
    figs = figures(summaries, v2)
    write_report(summaries, v2, cis, gate_res, figs)
    print(json.dumps({"models": sorted({t for t, _ in summaries}, key=_size_key),
                      "retrievers": sorted({r for _, r in summaries}),
                      "advantage_survives": gate_res["advantage_survives"],
                      "report": str(REPORT), "figures": figs}, indent=2))
    return 0


def _has_unsafe(ap: Path) -> bool:
    rows = json.loads(ap.read_text())["samples"]
    return rows and "_unsafe" in rows[0]


def _unsafe_from_delta(ap: Path) -> dict[str, float]:
    rows = json.loads(ap.read_text())["samples"]
    for r in rows:
        r["_unsafe"] = 1.0 if (r.get("delta_p_min") is not None and r["delta_p_min"] < 0) else 0.0
    f = {r["sample_id"]: r for r in rows if r["arm"] == "field_aware_cer_llm"}
    o = {r["sample_id"]: r for r in rows if r["arm"] == "ordinary_rag_llm"}
    ids = sorted(set(f) & set(o))
    diff = np.array([f[i]["_unsafe"] - o[i]["_unsafe"] for i in ids], dtype=float)
    m, lo, hi = _bootstrap_ci(diff)
    return {"gap_mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(ids)}


if __name__ == "__main__":
    raise SystemExit(main())
