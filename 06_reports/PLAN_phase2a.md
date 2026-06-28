# Phase 2a — Offline Constraint-Credibility Gate — RESULT REPORT

**Date:** 2026-06-23 · **Status:** DONE · **Gate: NO-GO (honest negative, pre-registered)** · `paper_usable=True`
**Artifacts:** `04_results/phase2a/{states_*.json, scoring_credibility.json, summary.json, analysis.json}` · `05_figures/phase2a/*.png`
**Stack:** Qwen3-14B (vLLM, served `qwen`) + BGE-M3/bge-reranker-v2-m3 (reused 2,070-doc O-RAN/3GPP KB, 710,045-chunk cache hit). Evidence: **1,800 real generations**, 900 held-out states, 3 sets × 4 arms.

## What Phase 2a tested (strictly offline — no LLM in any loop)
Can an LLM/RAG **produce** an executable, load-aware URLLC constraint (`urllc_min_prb`) that approximates the Phase-0 oracle, *before* committing to closed-loop? The LLM is called per held-out state; effects are measured by a **one-step counterfactual** (apply each arm's reservation via the existing `_project_to_min_urllc` shield to the logged action, step once). Arms: **static** (Phase-0 floor) · **oracle_margin** (load/SLA-aware UB) · **llm_no_rag** · **rag_llm**.

**Pre-gate scoring-credibility (no LLM): PASS** — the counterfactual scorer orders template reservations monotonically, penalizes over-reservation, reproduces static's cross-regime failure, and a broken (reservation-ignoring) scorer correctly fails the ordering (the control has teeth). Reconstruction self-test reproduced logged outcomes **bit-for-bit (0.00 over 900 states)**.

## Results (one-step counterfactual, 300 held-out states/set)

| set | arm | URLLC viol | reward | mean PRB | schema | citation |
|---|---|---:|---:|---:|---:|---:|
| **cross** (decisive) | static (floor 50) | 0.660 | −0.259 | 59 | — | — |
| | **oracle_margin** | **0.033** | 0.417 | 100 | — | — |
| | llm_no_rag | 0.370 | 0.152 | 73 | 1.00 | — |
| | **rag_llm** | **0.377** | 0.090 | 75 | 1.00 | 1.00 |
| high_urllc | static (60) | 0.083 | 0.798 | 78 | | |
| | oracle_margin | 0.033 | 0.633 | 91 | | |
| | rag_llm | 0.080 | 0.785 | 78 | 1.00 | 1.00 |
| high_embb | static (50) | 0.007 | 1.581 | 50 | | |
| | oracle_margin | 0.000 | 1.672 | 47 | | |
| | rag_llm | 0.010 | 1.775 | 41 | 1.00 | 1.00 |

## Gate verdict — **NO-GO** (`analysis.json`)
- `rag_reduces_viol_vs_static` ✓ (cross 0.660 → 0.377, gap 0.28, paired CI>0)
- `reward_loss_vs_oracle_small` ✗ (cross R_oracle 0.417 − R_rag 0.090 = 0.33 ≫ 0.10)
- `approaches_oracle_safety` ✗ (cross V_rag 0.377 ≫ V_oracle+0.05 = 0.083)
- `rag_adds_value_over_norag` ✗ (0.377 vs 0.370)

**Honest finding:** an off-the-shelf Qwen3-14B (with or without RAG) **reduces** URLLC violations vs a static floor but **cannot match the oracle's load-aware safety offline**, so **closed-loop RAG-shielding is not yet warranted**. JSON discipline is perfect (schema 1.00, citations valid) — the failure is the *decision*, not the format.

### Failure mode (the useful diagnostic)
- **Under-reservation under backlog.** On the cross set the oracle reserves ~100 PRB (URLLC demand + accumulated backlog at a pessimistic channel); the LLM reserves only ~60–75 → ~0.37 residual violations vs oracle 0.033.
- **Qualitatively right, quantitatively miscalibrated.** In the *easy* high_embb regime the LLM correctly reserves *less* than static (41 vs 50) → higher reward (1.775) at equal safety — so it does scale reservation with load. It just doesn't reserve aggressively enough on the hard, high-backlog, channel-margin states.
- **RAG ≈ no-RAG.** Retrieval surfaces the URLLC 99.999%/1 ms SLA (cited, citation_validity 1.00) but does **not change the reservation** vs parametric knowledge — confirming pre-registered negative (a): for this constraint, retrieval is provenance, not a decision lever.

### Honest caveats
- **One-step proxy** (no backlog feedback) bounds *instantaneous* safety — it is *generous* to the LLM (closed-loop, where under-reservation compounds backlog, would be worse), so the NO-GO is conservative-correct.
- Retrieval used a **constant domain query** (URLLC SLA facts are state-independent), fetched once — efficiency, not a method change.

## Decision / what this means
- **Do not proceed to closed-loop RAG-shielding as-is.** The program's solid, paper-ready contribution remains **Phase 0 + Phase 1**: the headroom oracle (static is fundamentally insufficient under traffic shift) + DRL baselines (safe-exploration and distribution-shift robustness; converged safety is learner-dependent). Phase 2a adds an honest boundary: *a 14B can't yet generate oracle-matching load-aware constraints offline, and RAG doesn't help.*
- **Pre-registered options (NOT pursued by p-hacking the prompt):** (a) a calibration-aware / few-shot prompt that forces "reserve ⌈(demand+backlog)/(se·g_pess)⌉" reasoning; (b) a stronger producer (Qwen3-32B, local); (c) keep the oracle/static shield and drop the LLM from the safety path. Any of these is a fresh, gated Phase-2a-v2 — to be chosen explicitly, not tuned-to-pass.

**Fallback already banked:** Phase 0+1 stand on their own; this negative sharpens the story (LLM-as-constraint-generator is the bottleneck, not the safe-DRL framing).

---

## Phase 2a-v2 — Model-size sweep (added 2026-06-24)
Question raised: "is the 14B too big — would a smaller model help?" Answer (empirical): **no** — the failure is *under-reservation* (numeric reasoning), so smaller is worse. Same offline gate, same held-out states/scorer; only the producer model changed (served sequentially via vLLM; 32B on GPU1). Artifacts: `04_results/phase2a/{summary,analysis}_{1p7b,4b,14b,32b}.json`, `sweep_summary.json`, `05_figures/phase2a/fig_size_sweep.png`.

| model | params | cross viol (rag_llm) | cross viol (no_rag) | reward (rag) | mean PRB | schema | cite | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | 1.7B | **0.660** | 0.660 | −0.259 | 59 | 1.00 | 1.00 | NO-GO |
| Qwen3-4B | 4B | 0.463 | 0.423 | 0.001 | 70 | 1.00 | 1.00 | NO-GO |
| Qwen3-14B | 14B | 0.367 | 0.360 | 0.107 | 76 | 1.00 | 1.00 | NO-GO |
| Qwen3-32B | 32B | **0.307** | 0.340 | 0.350 | 72 | 1.00 | 1.00 | NO-GO |
| *static ref* | — | 0.660 | | −0.259 | ~50 | | | |
| *oracle ref* | — | **0.033** | | 0.417 | ~100 | | | |

**Findings (honest):**
1. **Constraint quality improves monotonically with model size** (cross viol 0.660→0.463→0.367→0.307) — **smaller is strictly worse**, refuting the "try a 7B/3B" hypothesis. The 1.7B model is *no better than the static floor* (it effectively can't produce a useful reservation; reserves ≈floor).
2. **But it plateaus far above the oracle** — even **32B = 0.307 ≈ 9× the oracle's 0.033** and still fails the gate. The mean reservation tops out ~72–76 PRB while the oracle needs ~100 under backlog: **persistent under-reservation at every scale.**
3. **RAG ≈ no-RAG at all sizes** (32B shows the first faint RAG benefit, 0.340→0.307, still far short). JSON/citation discipline is perfect at every size (schema=cite=1.00) — the failure is the *decision*, never the format.
4. **Scale is not the lever.** Diminishing returns (Δ 14B→32B is only −0.06 for >2× params) imply no locally-available model crosses the gate by size alone.

**Updated decision:** the offline NO-GO stands across 1.7B→32B → closed-loop RAG-shielding with an off-the-shelf LLM is not warranted. The remaining levers are **calibration, not capacity**: a few-shot/explicit-arithmetic prompt (force `⌈(demand+backlog)/(se·g_pess)⌉`), or simply keep the oracle/static shield and drop the LLM from the safety path. Either is a fresh, explicitly-chosen step — not pursued by tuning-to-pass.
