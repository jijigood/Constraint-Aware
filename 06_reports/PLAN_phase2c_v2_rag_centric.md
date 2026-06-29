# PLAN — Phase 2c-v2: RAG-Centric Constraint Evidence Retrieval (full-paper upgrade)

**Date:** 2026-06-29
**Branch base:** `phase2b-v1-m3m5` (HEAD `ef74cb8`)
**Decision (user, 2026-06-29):** RAG/CER is the **single core contribution**; upgrade from letter to **full paper**.
**Status:** planning only — not implemented, not run.

---

## 0. Why this plan exists (the honest problem it must solve)

The earlier program ended at a clean letter-grade story: `direct-numeric LLM = NO-GO` → `symbolic-z verify/solve = PASS` → `p_min/Pmax state augmentation = control benefit`. Promoting RAG to the **single** headline forces us to confront two facts that the current code makes unavoidable:

1. **Phase 2c is not RAG yet.** `safe_oran/experiments/run_phase2c_mini_cer.py` is fully deterministic: a TF-IDF `MiniRetriever` plus hand-written field routing (`_field_aware_hits`, `_hits_to_spec`). There is **no LLM** anywhere in it. A full paper whose core claim is "RAG" cannot rest on TF-IDF + hand-coded routing — reviewers will reject the premise.

2. **The verifier/solver already washes out RAG-quality differences on the control metric.** In `04_results/phase2c_mini_cer/summary.json`, `state_aware_rag` already reaches `mean_abs_delta_p_min = 0.0` — identical to `field_aware_cer` and `cer_verifier_solver`. The recall/field-accuracy gap (0.80→0.90, 0.95→1.00) does **not** currently traverse to any control consequence. As-is, the data argues that retrieval quality beyond "state-aware" is irrelevant downstream — the opposite of "RAG is the core."

**This plan's job is therefore not to add experiments around a finished result. It is to (a) put a real LLM in the compile path, and (b) build a benchmark hard enough that retrieval quality demonstrably traverses to the control safety boundary — or honestly report that it does not.** The keystone is the downstream-control gate (G4). Everything else is in service of it.

We keep the program's honest-gate discipline: every gate has an explicit PASS / NO-GO and a pre-written fallback claim, so a negative result is still publishable.

---

## 1. Reframed contribution (CER as core)

**Task definition.** Constraint Evidence Retrieval (CER) for safety-critical slicing control:

```
(I_k intent, x_t state, D corpus)  -->  E_k field-level evidence  -->  z_k symbolic spec
```

The LLM/RAG output is **never** a PRB number and **never** free text answered to a user. It is the set of typed fields of `ConstraintSpec` (`safe_oran/constraints/spec.py`): `formula_id`, `reliability_target`, `channel_margin_policy`, `service_rule`, `priority_rank`, `citations`. A deterministic `Verifier` + `DeterministicSolver` turn `z_k` into `p_min`; the shield enforces it; `p_min/Pmax` is exposed to DRL.

**Four contribution points (paper):**

- **C1 — CER task.** RAG for *constraint compilation*, not QA, not direct control. Differentiator slogan: *"RAG for constraint compilation, not RAG for QA."*
- **C2 — Field-aware / state-aware CER.** Per-field routed retrieval (`q_reliability`, `q_margin`, `q_service`, `q_formula`) conditioned on `intent + state summary` beats intent-only RAG, *especially under conflict / missing / noisy evidence*.
- **C3 — Verifiable symbolic interface.** `E_k → z_k → Verifier` makes RAG output *checkable, not trusted*: whitelist, range, citation-grounding, fail-closed; direct-numeric PRB rejected (`direct_numeric_prb_is_not_a_symbolic_spec`).
- **C4 — Downstream control impact.** Retrieval quality → field accuracy → `Δp_min` (split into under/over) → `urllc_violation`, `fallback_rate`, `unsafe_pass`, `D_proj`. **This is the claim that makes RAG load-bearing and is the riskiest to establish.**

---

## 2. Five-link evidence chain → phases

| Link | Claim | Source | Status |
|---|---|---|---|
| L1 | Direct-numeric LLM constraint = NO-GO | Phase 2a / 2a-v2 (Qwen3 1.7B→32B) | DONE |
| L2 | Field-level CER beats ordinary RAG at building `z_k` | Phase 2c → **2c-v2 (real LLM, harder bench)** | TO BUILD |
| L3 | `z_k` verifiable → deterministic `p_min`; direct-numeric fail-closed | Phase 2b-v1 | DONE |
| L4 | **Retrieval quality traverses to control safety boundary** | **Phase 2c-v2 downstream replay (NEW, keystone)** | TO BUILD |
| L5 | `p_min/Pmax` state augmentation gives control benefit | Phase 3 M3/M5 | DONE |

The full paper stands on L1+L3+L5 (already solid) plus **L2+L4, which this plan must establish or honestly bound.**

---

## 3. Workstreams

### WS-A — Real LLM compiler (removes the "it's just TF-IDF" objection)

Add a real-LLM compile path alongside the deterministic baseline. Reuse the prior infra: BGE-M3 retriever + Qwen3 vLLM served via `serve_track_a_llm.sh`, OpenAI-compatible client (the `~/dify_vllm_uv310` env used in Phase 2a).

New module `safe_oran/rag/cer_llm.py`:
- `retrieve(intent, state_summary, corpus, mode)` — modes: `intent_only`, `state_aware`, `field_routed` (BGE-M3 dense retrieval; field_routed issues 4 per-field queries and merges, mirroring `_field_aware_hits` but over real embeddings).
- `compile_spec(intent, state_summary, evidence)` — LLM emits **only** the typed-field JSON (schema-constrained / function-call). System prompt forbids `urllc_min_prb`; if the model emits a number it must be rejected by the existing `Verifier`, which is a positive result for C3.
- Deterministic fallback to the current `MiniRetriever` path when vLLM is down, recorded as `compiler=deterministic` in outputs so it is never silently conflated with LLM runs.

**Arms (extends current `ARMS`):**
`no_retrieval` · `ordinary_rag_intent_only` · `state_aware_rag` · `field_aware_cer` · `cer_verifier_solver` · **`llm_field_cer`** (real LLM compile, field-routed) · **`llm_cer_verifier`** (real LLM + verifier + fail-closed). Keep the deterministic arms as ablation/oracle-routing references.

> Honesty rule: any arm using the LLM must be tagged `paper_usable` only with a live vLLM server + recorded generations (same discipline as Phase 2a). Deterministic-router arms are reported as *routing oracles / ablations*, not as "RAG".

### WS-B — Benchmark v2 (make differences traverse to control)

The current bench saturates because (i) only 5 clean categories and (ii) errors mostly produce *over*-reservation (safe-but-wasteful), so violations never move. Fix both.

New module `safe_oran/rag/cer_benchmark.py` (supersedes the inline corpus/sample builders, keeps the same `EvidenceDoc`/`MiniSample` dataclasses and `gold_p_min` via `DeterministicSolver`):

- **Size ~160**, category mix weighted so that **conflict + missing + noisy ≥ 40%** of samples:

  | Category | n | Tests | Intended control consequence |
  |---|--:|---|---|
  | normal | 24 | baseline | none (sanity) |
  | burst | 24 | margin routing | over-reservation if wrong |
  | degraded | 24 | worst_case margin | **under-reservation → unsafe** if missed |
  | upgrade | 24 | reliability 0.999 | under-reservation if missed |
  | conflict | 24 | latest-wins resolution | wrong target → under/over |
  | missing | 20 | required field absent | must fail-closed / fallback |
  | noisy | 20 | distractor-heavy corpus | recall degradation |

- **Corpus enlarged with distractors and near-duplicates** so retrieval is non-trivial (ordinary RAG should *fail* on conflict/noisy, not score 0.85).
- **Critical design constraint:** seed enough samples where a retrieval miss yields **under-reservation** (degraded/upgrade with tight headroom), so that field-accuracy differences show up as `urllc_violation` / `unsafe_pass`, not only as harmless over-reservation. Verify during smoke that ordinary RAG produces non-zero `under_reservation_rate` on these categories — otherwise the bench still can't separate arms and must be hardened further.
- **Per-sample control-consequence label** (`expected_effect ∈ {none, over, under, fallback}`) so the analysis can show *which* failures matter.
- **Light real-style evidence:** mix in a small set of public O-RAN / 3GPP-style technical snippets (paraphrased, clearly tagged `source=public_style`). **Do not** claim large-scale real-spec validation.

### WS-C — Downstream control replay (KEYSTONE, gate G4)

New experiment `safe_oran/experiments/run_phase2c_downstream.py`. For each arm, take its produced `z_k` per sample, run it through `Verifier` → `DeterministicSolver` → `p_min`, then **replay one control step against the realized environment** to convert spec error into control outcomes.

Two replay surfaces (do the cheap one first):
1. **Static-state replay (cheap, primary):** reuse the 900 saved states in `04_results/phase2a/states_{cross,high_embb,high_urllc}.json` and the Phase 2a one-step counterfactual (`01_code/rag/counterfactual.py`, `_project_to_min_urllc`). Per arm report: `urllc_violation`, `over_reservation_prb`, `under_reservation_prb`, `fallback_rate`, `unsafe_pass_rate` (spec passed verifier yet `p_min < oracle`), `mean_D_proj`. **No DRL retrain.**
2. **Closed-loop M6 (optional, only if G4 passes on (1)):** swap Oracle-z for CER-z in the M5 constraint-aware loop (`safe_oran/experiments/train_m3_m5.py`), one scenario (S6_moderate_decay — the clean non-saturated regime), 3 seeds. Report whether CER-z preserves the M5 benefit vs Oracle-z and vs ordinary-RAG-z.

z_k sources compared (same solver/shield): `oracle_z` · `ordinary_rag_z` · `state_aware_rag_z` · `field_cer_z` · `llm_cer_verifier_z`.

### WS-D — Metrics & analysis

Extend `evaluate_arm` outputs (already has `under/over_reservation_prb`, `delta_p_min`) and add at the analysis layer:
- `under_reservation_rate` and `mean_under_reservation_prb` **per category** (not just global mean — the global mean is what hid the effect before).
- `unsafe_pass_rate` = fraction where `spec_validity==1` but `p_min < gold_p_min` (verifier passed something unsafe).
- Δ⁻/Δ⁺ already computed as `under_reservation_prb` / `over_reservation_prb`; surface them as the headline safety-vs-waste decomposition.
- Bootstrap CIs over samples for the headline gaps (ordinary vs field-CER) on conflict/missing/noisy.

### WS-E — Paper integration

Phase 2c-v2 becomes a **main section (not appendix)**: CER task + field-aware retrieval + verifiable interface + downstream control impact. L1/L3/L5 become the framing and the closed-loop payoff.

---

## 4. Gates

| Gate | Check | PASS condition | Fallback claim if NO-GO |
|---|---|---|---|
| **G1** retrieval | recall@5, MRR, citation precision | field-routed > state-aware > intent-only, on **conflict/noisy** subsets (not just global) | "field routing helps only on clean SLAs" — narrow C2 |
| **G2** compilation | per-field accuracy, spec exact-match | field-CER field-acc > ordinary, gap concentrated on conflict/upgrade/degraded | report which fields LLM gets wrong; C2 partial |
| **G3** verify | spec validity, direct-numeric rejection, zero unsafe specs pass | direct-numeric 100% rejected; `unsafe_pass_rate(cer_verifier) ≈ 0` | C3 already solid from 2b-v1; keep |
| **G4** ★ downstream | does retrieval quality move control outcomes? | ordinary-RAG-z shows **non-zero** `urllc_violation`/`unsafe_pass` that field-CER-z **reduces**, with CI separation | **If washed out:** pivot headline to *"a verifier makes even mediocre RAG safe"* — C3-centric, RAG demoted back. Decide explicitly, no p-hacking. |
| **G5** closed-loop (opt) | CER-z preserves M5 benefit | CER-z violation/D_proj ≤ ordinary-RAG-z, ≈ Oracle-z on S6 | report CER-z gap to Oracle-z honestly as future work |

**G4 is the make-or-break gate for the "RAG single-core" decision.** Run it first and cheapest.

---

## 5. Run order & cost

1. **WS-B benchmark v2 + WS-D metrics** (deterministic, no GPU) — build hard bench; smoke-verify ordinary RAG produces non-zero under-reservation on degraded/upgrade. *Cheap.*
2. **WS-C static replay with deterministic arms** (no LLM, no GPU) — **first read on G4.** If even the oracle-routing field-CER cannot separate from state-aware on control outcomes, the bench is still too easy → return to step 1 before spending any GPU/LLM budget.
3. **WS-A real LLM arms** (vLLM on free GPU; GPU1 1×A100 per memory) — only after step 2 shows the bench can separate arms. Re-run G1/G2/G4 with `llm_field_cer`, `llm_cer_verifier`.
4. **WS-C closed-loop M6 / G5** (PPO on GPU, S6 ×3 seeds) — only if G4 passes.
5. **WS-E** paper integration.

Steps 1–2 are nearly free and de-risk the entire full-paper bet before any LLM/GPU spend. **Do not skip the step-2 checkpoint.**

---

## 6. Honest boundaries (write these into the paper)

- mini-CER v2 is a **controlled, field-labeled benchmark** mixing SLA templates, operator policy rules, and public-style O-RAN/3GPP snippets. **Not** large-scale real-spec retrieval; that is Phase 2c-v3 / future work.
- Deterministic-router arms are **routing oracles / ablations**, explicitly distinguished from real-LLM arms.
- Report reward/safety as **trade-offs** (carried from Phase 1/3 discipline), never reward alone.
- If G4 washes out, the paper's honest center is C3 (verifiable interface), and we say so — the verifier turning unreliable RAG into safe control is itself a defensible full-paper claim.

---

## 7. New code & artifacts

New code (branch `phase2c-v2-rag`):
- `safe_oran/rag/__init__.py`
- `safe_oran/rag/cer_benchmark.py` — corpus v2, samples v2, control-consequence labels
- `safe_oran/rag/cer_llm.py` — real-LLM retrieve + compile, deterministic fallback, arm tagging
- `safe_oran/experiments/run_phase2c_v2.py` — retrieval+compile eval (G1/G2/G3), extends current ARMS
- `safe_oran/experiments/run_phase2c_downstream.py` — control replay (G4), reuses `01_code/rag/counterfactual.py`
- (opt) extend `safe_oran/experiments/train_m3_m5.py` with `--z-source {oracle,cer}` for M6 (G5)
- `safe_oran/experiments/analyze_phase2c_v2.py` — per-category Δ⁻/Δ⁺, unsafe_pass, bootstrap CIs, figures

Artifacts:
- `04_results/phase2c_v2/{summary,analysis,downstream}.json`, `cer_v2_table.csv`, per-category CSVs
- `05_figures/phase2c_v2/{fig_retrieval_quality,fig_field_accuracy_by_category,fig_delta_pmin_under_over,fig_downstream_violation,fig_unsafe_pass}.png`
- `06_reports/PHASE2C_V2_RESULTS.md`

---

## 8. One-paragraph summary

Phase 2c-v2 turns the deterministic field-routing toy into a real RAG-centric study: a real-LLM constraint-evidence compiler, a harder field-labeled benchmark (≥40% conflict/missing/noisy, seeded so misses cause *unsafe* under-reservation, not just waste), and — the keystone — a downstream control-replay gate (G4) that tests whether retrieval quality actually moves the URLLC safety boundary once the verifier/solver are in the loop. Cheap deterministic steps 1–2 read G4 before any GPU/LLM spend; if differences wash out, the paper honestly recenters on the verifiable interface (C3) rather than overclaiming RAG.
