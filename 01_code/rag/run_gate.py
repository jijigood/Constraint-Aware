"""
Phase 2a gate (runs in ~/dify_vllm_uv310 — has openai + RealRetriever). For each set x arm, produce a
per-state URLLC reservation, score it with the one-step counterfactual, aggregate, compute the gate, and
write summary.json + analysis.json (+ figures attempted). LLM calls are cached by discretized state.

Usage:
  ~/dify_vllm_uv310/bin/python 01_code/rag/run_gate.py --check-retriever   # just validate cache hit
  ~/dify_vllm_uv310/bin/python 01_code/rag/run_gate.py --smoke 5           # 5 states/set, real LLM
  ~/dify_vllm_uv310/bin/python 01_code/rag/run_gate.py                     # full gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAG_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(RAG_DIR), "env"))
ARGO = "/home/huangxiaolin/ARGO2-main/ARGO"
sys.path.insert(0, ARGO)

import counterfactual as CF  # noqa: E402
import constraint_producers as P  # noqa: E402
from scoring_credibility import oracle_reservation  # noqa: E402

PHASE2A = os.path.join(os.path.dirname(os.path.dirname(RAG_DIR)), "04_results", "phase2a")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(RAG_DIR)), "05_figures", "phase2a")
CACHE = os.path.join(PHASE2A, "llm_cache.json")
SETS_ORDER = ["cross", "high_urllc", "high_embb"]   # decisive set first
ARMS = ["static", "oracle_margin", "llm_no_rag", "rag_llm"]
SCHEMA_VERSION = "safe_drl_v1"


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    if len(v) <= 1:
        return float(v.mean()) if len(v) else 0.0, float(v.mean()) if len(v) else 0.0, float(v.mean()) if len(v) else 0.0
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(v, size=len(v), replace=True).mean()) for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_client():
    import openai
    ep = os.environ.get("TRACK_A_LLM_ENDPOINT", "http://127.0.0.1:8001/v1")
    key = os.environ.get("TRACK_A_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("TRACK_A_LLM_MODEL_ID", "qwen")
    return openai.OpenAI(base_url=ep, api_key=key), model, ep


def build_retriever():
    from track_a_experiment import load_chunks
    from track_a_real_backend import RealRetriever
    docs = os.path.join(ARGO, "ORAN_Docs")
    res = load_chunks(docs, chunk_words=120, chunk_overlap=20)
    chunks = res[0] if isinstance(res, tuple) else res
    r = RealRetriever(chunks, embedding_path="/home/huangxiaolin/models/BGE-M3",
                      reranker_path="/home/huangxiaolin/models/bge-reranker-v2-m3",
                      device="cpu", max_seq_len=512,
                      cache_dir=os.path.join(ARGO, ".track_a_emb_cache"))
    ev = r.evidence()
    print(f"  retriever: index_vectors={ev.get('index_vectors')} dim={ev.get('embedding_dim')} "
          f"cache_hit={ev.get('embedding_cache_hit')}")
    assert ev.get("embedding_cache_hit"), "embedding cache MISS — refusing to re-embed 710k chunks"
    assert int(ev.get("index_vectors", 0)) > 700_000, "unexpected index size"
    return r, ev


def run_arm(producer, states, gen_counter):
    """Return (reservations[list], meta[list of dict]). LLM arms run 8-wide (vLLM handles concurrency)."""
    is_llm = producer.name in ("llm_no_rag", "rag_llm")
    if is_llm:
        if hasattr(producer, "_ensure_evidence"):
            producer._ensure_evidence()      # retrieve ONCE before the parallel calls
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:
            outs = list(ex.map(producer.produce, states))
        gen_counter[0] += len(states)
    else:
        outs = [producer.produce(s) for s in states]
    reservations = [int(o.urllc_min_prb) for o in outs]
    meta = [{"prb": int(o.urllc_min_prb), "schema_ok": bool(o.schema_ok),
             "parse_fallback": bool(o.parse_fallback), "citations": list(o.citations or []),
             "valid_ids": list(getattr(o, "_valid_ids", []))} for o in outs]
    return reservations, meta


def llm_quality(meta, states):
    schema = 1.0 - float(np.mean([m["parse_fallback"] for m in meta]))
    def cite_v(m):
        c = m["citations"]
        if not c:
            return 0.0
        vid = set(m["valid_ids"])
        return sum(1 for x in c if x in vid) / len(c) if vid else 0.0
    citation = float(np.mean([cite_v(m) for m in meta]))
    def cons_v(m, s):
        o = oracle_reservation(s)
        return 1.0 if (m["schema_ok"] and 0 <= m["prb"] <= 100 and not (o > 0 and m["prb"] == 0)) else 0.0
    constraint = float(np.mean([cons_v(m, s) for m, s in zip(meta, states)]))
    return {"schema_validity": schema, "citation_validity": citation, "constraint_validity": constraint,
            "mean_reservation": float(np.mean([m["prb"] for m in meta]))}


def evaluate_set(set_name, states, producers, gen_counter):
    arm_res, arm_meta, arm_score = {}, {}, {}
    for arm, prod in producers.items():
        res, meta = run_arm(prod, states, gen_counter)
        arm_res[arm] = res; arm_meta[arm] = meta
        arm_score[arm] = CF.score_reservations(states, res)["per_state"]
    out = {"n": len(states), "arms": {}}
    for arm in producers:
        per = arm_score[arm]
        viol = [p["urllc_violation"] for p in per]
        rew = [p["reward"] for p in per]
        vm, vlo, vhi = bootstrap_ci(viol)
        rm, rlo, rhi = bootstrap_ci(rew)
        rec = {"urllc_violation_rate": {"mean": vm, "lo": vlo, "hi": vhi},
               "reward": {"mean": rm, "lo": rlo, "hi": rhi},
               "mean_prb_urllc": float(np.mean([p["prb_urllc"] for p in per]))}
        if arm in ("llm_no_rag", "rag_llm"):
            rec.update(llm_quality(arm_meta[arm], states))
        out["arms"][arm] = rec
    # paired diffs (same state order) for the gate
    vstatic = np.array([p["urllc_violation"] for p in arm_score["static"]])
    vrag = np.array([p["urllc_violation"] for p in arm_score["rag_llm"]])
    vnorag = np.array([p["urllc_violation"] for p in arm_score["llm_no_rag"]])
    dm, dlo, dhi = bootstrap_ci(vstatic - vrag)
    nm, nlo, nhi = bootstrap_ci(vnorag - vrag)
    out["paired"] = {"static_minus_rag_viol": {"mean": dm, "lo": dlo, "hi": dhi},
                     "norag_minus_rag_viol": {"mean": nm, "lo": nlo, "hi": nhi}}
    return out


def compute_gate(results):
    c = results.get("cross", {}).get("arms", {})
    paired = results.get("cross", {}).get("paired", {})
    checks, notes = {}, []
    if not c:
        return {"PASS": False, "verdict": "no cross set", "checks": {}, "notes": ["cross set missing"]}
    Vrag = c["rag_llm"]["urllc_violation_rate"]["mean"]
    Vora = c["oracle_margin"]["urllc_violation_rate"]["mean"]
    Vsta = c["static"]["urllc_violation_rate"]["mean"]
    Rrag = c["rag_llm"]["reward"]["mean"]
    Rora = c["oracle_margin"]["reward"]["mean"]
    # 1 RAG significantly reduces violations vs static (paired CI > 0 AND material gap)
    checks["rag_reduces_viol_vs_static"] = bool(paired["static_minus_rag_viol"]["lo"] > 0 and (Vsta - Vrag) >= 0.20)
    # 2 reward loss vs oracle small
    checks["reward_loss_vs_oracle_small"] = bool((Rora - Rrag) <= 0.10)
    # 3 approaches oracle safety
    checks["approaches_oracle_safety"] = bool(Vrag <= Vora + 0.05)
    pass_safety = checks["rag_reduces_viol_vs_static"] and checks["approaches_oracle_safety"]
    PASS = bool(pass_safety and checks["reward_loss_vs_oracle_small"])
    rag_adds_value = bool(paired["norag_minus_rag_viol"]["lo"] > 0)
    checks["rag_adds_value_over_norag"] = rag_adds_value
    # separately-reported quality (not blockers)
    quality = {arm: {k: c[arm][k] for k in ("schema_validity", "citation_validity", "constraint_validity",
                                            "mean_reservation") if k in c[arm]}
               for arm in ("llm_no_rag", "rag_llm")}
    # verdict branch
    if PASS and rag_adds_value:
        verdict = "GO closed-loop with RAG-LLM: it reduces violations vs static, approaches oracle safety at small reward cost, and RAG beats no-RAG."
    elif PASS and not rag_adds_value:
        verdict = ("GO closed-loop with LLM (no-RAG suffices): the LLM matches the oracle constraint, but "
                   "RAG does not change the decision vs parametric knowledge — RAG's value here is "
                   "provenance/citations, not the reservation (honest downgrade).")
        notes.append("RAG≈no-RAG: report retrieval as grounding/citations, not as a safety lever.")
    elif not pass_safety:
        verdict = "NO-GO: the LLM cannot match the oracle's load-aware safety offline. Honest negative; do not go closed-loop. Revisit prompt/state-summary or fall back to the oracle shield."
    else:
        verdict = "CAUTION/NO-GO: safety only via over-reserving (reward loss vs oracle too large). Revisit margin/prompt before closed-loop."
    return {"PASS": PASS, "pass_safety": pass_safety, "rag_adds_value": rag_adds_value,
            "checks": checks, "quality": quality, "verdict": verdict, "notes": notes,
            "cross_summary": {"V_static": Vsta, "V_norag": c["llm_no_rag"]["urllc_violation_rate"]["mean"],
                              "V_rag": Vrag, "V_oracle": Vora, "R_rag": Rrag, "R_oracle": Rora}}


def make_figures(results, phase0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (matplotlib unavailable in this venv: {e}; render figures separately from JSON)")
        return False
    os.makedirs(FIG_DIR, exist_ok=True)
    sets = [s for s in SETS_ORDER if s in results]
    # safety bars
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.2), squeeze=False)
    for j, sname in enumerate(sets):
        ax = axes[0][j]; arms = results[sname]["arms"]
        labels = [a for a in ARMS if a in arms]
        vals = [arms[a]["urllc_violation_rate"]["mean"] for a in labels]
        errs = [[arms[a]["urllc_violation_rate"]["mean"] - arms[a]["urllc_violation_rate"]["lo"] for a in labels],
                [arms[a]["urllc_violation_rate"]["hi"] - arms[a]["urllc_violation_rate"]["mean"] for a in labels]]
        colors = {"static": "#c0504d", "oracle_margin": "#4f81bd", "llm_no_rag": "#9bbb59", "rag_llm": "#8064a2"}
        ax.bar(range(len(labels)), vals, yerr=errs, capsize=3, color=[colors[a] for a in labels])
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(f"URLLC violation — {sname}"); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_arm_safety.png"), dpi=130); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-retriever", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.check_retriever:
        build_retriever(); print("retriever OK"); return

    states_by_set = {}
    for sname in SETS_ORDER:
        p = os.path.join(PHASE2A, f"states_{sname}.json")
        with open(p) as f:
            d = json.load(f)
        st = d["states"][:args.smoke] if args.smoke else d["states"]
        states_by_set[sname] = st

    client, model_id, endpoint = build_client()
    retriever, rev = build_retriever()
    producers = {"static": P.StaticProducer(), "oracle_margin": P.OracleMarginProducer(0.99),
                 "llm_no_rag": P.LLMNoRAGProducer(client, model_id),
                 "rag_llm": P.RAGLLMProducer(client, model_id, retriever, top_k=5)}

    gen_counter = [0]
    results = {}
    for sname in SETS_ORDER:
        print(f"[set {sname}] n={len(states_by_set[sname])} ...", flush=True)
        results[sname] = evaluate_set(sname, states_by_set[sname], producers, gen_counter)
        a = results[sname]["arms"]
        for arm in ARMS:
            extra = ""
            if arm in ("llm_no_rag", "rag_llm"):
                extra = f" schema={a[arm].get('schema_validity',0):.2f} cite={a[arm].get('citation_validity',0):.2f} resv={a[arm].get('mean_reservation',0):.0f}"
            print(f"   {arm:<14} viol={a[arm]['urllc_violation_rate']['mean']:.3f} "
                  f"reward={a[arm]['reward']['mean']:.3f} prb={a[arm]['mean_prb_urllc']:.0f}{extra}", flush=True)

    phase0 = json.load(open(os.path.join(os.path.dirname(os.path.dirname(RAG_DIR)),
                                          "04_results", "phase0_headroom.json")))
    gate = compute_gate(results)

    evidence = {"generation_calls": gen_counter[0], "index_vectors": int(rev.get("index_vectors", 0)),
                "embedding_dim": int(rev.get("embedding_dim", 0)),
                "embedding_model_id": rev.get("embedding_model_id", ""),
                "llm_model_id": model_id, "endpoint": endpoint,
                "n_states": sum(len(v) for v in states_by_set.values()), "smoke": bool(args.smoke),
                "tag": args.tag}
    paper_usable = (not args.smoke and evidence["index_vectors"] > 700_000 and evidence["embedding_dim"] > 0
                    and evidence["generation_calls"] > 0 and evidence["n_states"] >= 3 * 200
                    and bool(model_id) and bool(endpoint) and bool(evidence["embedding_model_id"]))

    summary = {"schema_version": SCHEMA_VERSION, "kind": "phase2a_gate_summary",
               "paper_usable": bool(paper_usable), "evidence": evidence, "sets": results}
    with open(os.path.join(PHASE2A, f"summary{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    analysis = {"schema_version": SCHEMA_VERSION, "kind": "phase2a_analysis", "tag": args.tag,
                "paper_usable": bool(paper_usable), "gate": gate}
    with open(os.path.join(PHASE2A, f"analysis{suffix}.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    make_figures(results, phase0)

    print("\n=== GATE ===")
    for k, v in gate["checks"].items():
        print(f"  [{'x' if v else ' '}] {k}")
    cs = gate["cross_summary"]
    print(f"\n  cross: V_static={cs['V_static']:.3f} V_norag={cs['V_norag']:.3f} V_rag={cs['V_rag']:.3f} "
          f"V_oracle={cs['V_oracle']:.3f} | R_rag={cs['R_rag']:.3f} R_oracle={cs['R_oracle']:.3f}")
    print(f"  PASS={gate['PASS']}  rag_adds_value={gate['rag_adds_value']}  paper_usable={paper_usable}")
    print(f"  verdict: {gate['verdict']}")
    for n in gate["notes"]:
        print(f"  note: {n}")


if __name__ == "__main__":
    main()
