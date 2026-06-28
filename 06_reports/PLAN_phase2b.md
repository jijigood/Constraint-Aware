# Phase 2b — RAG-Compiled Verifiable Constraints with Deterministic Shield

**Date:** 2026-06-25 · **Status:** PLANNED · **Gate: pending (offline replay first; pre-registered)**
**Project:** `/home/huangxiaolin/safe_drl_oran/`
**Formalism:** see `06_reports/SYSTEM_COMPOSITION_AND_PROBLEM_MODELING_CN.md` (constrained MDP, URLLC constraint, compile→verify→solve→shield, arms, gates G1–G4, metrics). This document is the **executable phase plan** — it does not restate the formalism.

---

## 0. Why this phase, in one line

Phase 2a/2a-v2 proved an off-the-shelf LLM (1.7B→32B) **cannot emit the safety-critical number** (`urllc_min_prb`) — persistent under-reservation, monotone-but-plateaued ~9× above oracle, while **schema/citation = 1.00**. Phase 2b tests the resolution: **the LLM emits only a typed, verifiable constraint *spec*; a deterministic solver computes the reservation; a shield projects the action.** The LLM is removed from the numeric safety path.

**Core hypothesis (Q4 in the modeling doc):** `RAG-LLM symbolic compiler + verifier + deterministic solver + shield` approaches oracle safety, where direct LLM numeric generation failed.

---

## 1. Scope discipline (pre-registered — do NOT drift)

- **Offline replay BEFORE closed-loop.** The decisive gate (B3) is a one-step counterfactual on **saved Phase-2a states** — no DRL retraining, no environment rollout with the LLM in the loop. Closed-loop (B4) is attempted **only if B3 passes**.
- **Reuse, don't rebuild.** Replay the saved held-out states `04_results/phase2a/states_{cross,high_embb,high_urllc}.json` (900 states) via the existing one-step counterfactual `01_code/rag/counterfactual.py` + the `_project_to_min_urllc` shield in `01_code/env/slicing_env.py`. Reuse `oracle_margin` and `static` producers from `01_code/rag/constraint_producers.py` as the upper bound / weak baseline. Reuse `run_gate.py` orchestration and `scoring_credibility.py` as the pre-gate scorer check.
- **Template SLAs first, full-text standards later.** Start with 6–12 controlled SLA/intent templates (§4); only after the template path passes do we feed retrieved full standards text.
- **No RCA, no DPO, no DRL retrain, no data augmentation.** The RCA line is frozen (see master plan §E.2). Phase 2b adds only the symbolic compiler + verifier + solver and the offline replay around the *existing* environment/shield/states.
- **Honest gate.** `paper_usable=True` only with real served-LLM generations + runtime evidence. Pre-register the gate; a NO-GO is write-ready (Phase 0+1+2a already stand alone).

---

## 2. What is NEW vs reused

| Component | Status | Location |
|---|---|---|
| Typed constraint **spec schema** (the JSON contract) | NEW | `01_code/rag/constraint_spec.py` |
| **Verifier** (schema/type/unit/citation/range/formula-whitelist/feasibility/monotonicity + fail-closed) | NEW | `01_code/rag/verifier.py` |
| **Deterministic solver** `F(z, s) → urllc_min_prb` (formula-id → ⌈(d+b)/(se·g_pess)⌉_Δp) | NEW (extract/centralize) | `01_code/shield/solver.py` (populate the empty `01_code/shield/`) |
| **Symbolic-spec LLM producer** (emits spec, NOT a number) + optional RAG | NEW | extend `01_code/rag/constraint_producers.py` |
| Phase-2b gate driver + analysis | NEW | `01_code/rag/run_gate_phase2b.py`, `analyze_phase2b.py` |
| SLA/intent **templates** (6–12) | NEW | `01_code/rag/sla_templates.py` (or `02_data/`) |
| One-step counterfactual scorer | REUSE | `01_code/rag/counterfactual.py` |
| `_project_to_min_urllc` shield | REUSE | `01_code/env/slicing_env.py` |
| Saved held-out states | REUSE | `04_results/phase2a/states_*.json` |
| `static` / `oracle_margin` producers | REUSE | `01_code/rag/constraint_producers.py` |
| KB + BGE-M3 + vLLM serving | REUSE | 2,070-doc KB, cache, `serve_qwen.sh` |

The solver is the **same arithmetic the oracle already uses** — Phase 2b does not invent a new safety computation; it routes the oracle's computation through an LLM-chosen *spec* instead of an LLM-chosen *number*.

---

## 3. Arms (offline replay)

| Arm | What it isolates | Source |
|---|---|---|
| `static` | weak baseline (fixed floor) | reuse |
| `direct_llm_numeric` | the Phase-2a **failure baseline** (LLM emits `urllc_min_prb`) | reuse 2a producer |
| `llm_symbolic + solver` | does the **compiler** work without retrieval? (parametric spec → solver) | NEW |
| `rag_llm_symbolic + solver` | **primary method** (retrieved SLA → spec → solver) | NEW |
| `oracle_margin` | safety **upper bound** | reuse |

Decisive comparisons: `(rag_llm_symbolic) vs (direct_llm_numeric)` — does routing through a spec+solver beat direct numeric? And `(rag_llm_symbolic) vs (oracle_margin)` — does it approach the upper bound?

---

## 4. SLA / intent templates (6–12; design before any LLM call)

Purpose: test whether the LLM **compiles semantics**, not whether it parrots text. Vary the dimensions that should change the spec (and hence the reservation), plus adversarial cases that must trigger fail-closed:

1. URLLC reliability **0.95** (relaxed)
2. URLLC reliability **0.99**
3. URLLC reliability **0.999** (strict) → spec must drive a higher pessimistic-quantile margin
4. Latency **strict** vs (5) latency **relaxed**
6. **eMBB-priority** intent (URLLC floor only) vs (7) **URLLC-priority** intent
8. **Ambiguous** SLA (qualitative "high reliability", no number) → spec must pick a defensible default or fail-closed, never fabricate
9. **Conflicting evidence** (two docs disagree on the target) → verifier/fallback behavior
10. **Missing reliability target** → fail-closed to conservative shield
11. Mixed/bursty regime applicability
12. (optional) unit-mismatch / out-of-range adversarial (must be rejected by verifier)

**RAG-sensitivity check (G3):** templates 1–3 (and 4 vs 5, 6 vs 7) must produce *correctly different* specs/reservations — not identical specs with only different citations (the Phase-2a RAG≈no-RAG trap).

---

## 5. Implementation sequence (B0 → B4)

### B0 — Deterministic path + equivalence oracle (no LLM)
Hand-write canonical constraint specs → `verifier` → `solver`. **Establish the equivalence oracle:** for a spec encoding the oracle's reliability target, `solver(F(z,s))` must reproduce `oracle_margin`'s reservation **bit-for-bit** on the saved states. This de-risks the entire phase: it proves the only open question is whether the LLM emits a *correct spec*, not whether the deterministic path is right.
- **Exit:** verifier unit tests pass (valid specs accepted, each invalid class rejected → fail-closed); solver == oracle reservation on all 900 states for the matching spec; scorer-credibility (`scoring_credibility.py`) still PASS.

### B1 — LLM compiles clean SLA templates → spec (no RAG)
Symbolic-spec producer on the 6–12 templates (parametric knowledge only). Measure typed-spec validity, verifier rejection rate, fallback rate, and solver output vs oracle on replay.
- **Exit:** high spec validity; **zero unsafe specs pass the verifier**; invalid specs fail closed.

### B2 — RAG-LLM compiles retrieved SLA → spec
Add retrieval over the SLA templates / KB. Confirm RAG changes the spec **correctly** across reliability/latency/priority variation (G3), not just citations.
- **Exit:** G3 sensitivity holds; citation validity high; spec changes track the SLA changes.

### B3 — Counterfactual replay on saved states (THE GATE)
Run all 5 arms through `counterfactual.py` + `_project_to_min_urllc` on `states_{cross,high_embb,high_urllc}.json`. This is the pre-registered decision point.
- **Gates (from modeling doc §5):**
  - **G1 Safety+Reward:** `URLLC_viol(rag_symbolic) ≤ oracle + ε` AND `reward ≥ oracle − δ` AND safer than static under cross-regime shift (compare at matched safety, not unmatched).
  - **G2 Verifiable soundness:** typed-spec validity high; **unsafe-spec pass rate = 0**; invalid → fail-closed.
  - **G3 RAG sensitivity:** SLA/reliability/intent edits change the spec correctly.
  - **G4 Shift robustness:** holds across high_embb / high_urllc / bursty / reliability-variation / ambiguous-SLA / missing-evidence.
- **If B3 NO-GO:** write the honest negative (the spec+solver path still under-/over-reserves, or the LLM can't pick the right spec) — this remains a publishable boundary on top of Phase 0+1+2a. Do **not** p-hack the prompt to pass; a calibration-aware redesign is a fresh, separately-gated step.

### B4 — Closed-loop DRL integration (ONLY if B3 passes)
Insert the verified compiler→solver→shield around the **existing** PPO/DQN policies (reuse Phase-1 checkpoints; no retrain unless a gap demands it). Evaluate closed-loop URLLC violation, reward at matched safety, training-time vs eval-time safety, cross-regime robustness.

---

## 6. Metrics (report components, never scalar reward alone)

URLLC violation rate · mean reward · **reward at matched safety** · eMBB/mMTC SLA rate · Jain fairness · mean/p95 reservation · typed-spec validity · citation validity · verifier rejection rate · **unsafe-spec pass rate (must be 0)** · fallback (fail-closed) rate · RAG-sensitivity score. (See modeling doc §6.)

---

## 7. Environment / run notes (carry-over gotchas)

- **vLLM serving:** GPU1 is the free A100 (GPU0 held by another user). 32B needs a near-free GPU (util check fails if another proc holds >~7 GB; 32B compile/capture >600 s — poll generously). Use `01_code/rag/serve_qwen.sh` (or the Route-B `serve_route_b_lora.sh` pattern). Run serving in background; **stop by explicit PID** (`pkill -f`/`pgrep -f` self-match).
- **Proxy:** `source track_a_env.sh` (NO_PROXY=127.0.0.1 + key) before any local-endpoint call, or httpx/openai 502s.
- **Python:** RAG/LLM-in-loop pieces run under `~/dify_vllm_uv310` (openai + RealRetriever); pure-numpy replay/figures under the project `.venv`; figures rendered in `.venv` (anaconda has a numpy/scipy conflict).
- **Default producer model:** Qwen3-14B (with a 32B confirmation pass available, per the 2a-v2 precedent).

---

## 8. Deliverables / artifacts

- Code: `01_code/rag/{constraint_spec.py, verifier.py, sla_templates.py, run_gate_phase2b.py, analyze_phase2b.py}`, `01_code/shield/solver.py`, extended `constraint_producers.py`.
- Results: `04_results/phase2b/{b0_equivalence.json, b1_templates.json, b2_rag.json, replay_{cross,high_embb,high_urllc}.json, summary.json, analysis.json}`; figures `05_figures/phase2b/*.png`.
- Report: append the B3 verdict (GO/NO-GO with the four sub-gates) to this file, mirroring the `PLAN_phase2a.md` result-report style.

---

## 9. Relationship to the paper

The paper structure is already drafted in the modeling doc §10 (Intro → Empirical motivation [Phase 0/1/2a] → Method [compile→verify→shield] → Problem formulation → Experiments [Phase 2b offline replay + optional closed-loop] → Results → Discussion → Conclusion). Drafting can proceed **in parallel** with B0–B3 — the empirical-motivation and problem-formulation sections do not depend on the Phase 2b outcome; only the Experiments/Results sections do. The RCA/RAFT supporting study (master plan Appendix A + §B.3 per-class hardening) is the motivation/boundary, not a contribution.

## 10. One-sentence thesis

> RAG-LLM compiles SLA + intent into typed, verifiable constraint specs; the safety-critical reservation is computed by a deterministic solver and enforced by a shield — the LLM interprets and compiles rules, it does not compute safety-critical numbers.
