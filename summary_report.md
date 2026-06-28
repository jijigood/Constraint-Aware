# RAG-Conditioned Safe DRL for Intent-Aware 6G O-RAN Slicing — Summary Report

**Project:** `~/safe_drl_oran/` · **Owner:** Owner A · **Engine:** Claude Code (meta-supervisor)
**Span:** 2026-06-23 → 2026-06-24 · **Lineage:** second paper after the *budgeted / distractor-robust evidence* program (`~/telecom_rag_research`); realizes **Theme A** of `~/LLM_6G_RAG_DRL_research_directions.md`.
**Status:** Phase 0 ✅ gate PASS · Phase 1 ✅ gate PASS (honest) · Phase 2a ✅ **NO-GO** (honest negative) · Phase 2a-v2 (size sweep) ✅ confirms NO-GO.

---

## 1. Executive summary

The thesis: **DRL is the controller; the LLM/RAG layer's job is a safety shield** that turns retrieved SLA/standards text into dynamic, load-aware constraints. We de-risked it the way the prior program did — oracle-first, honest gates, component-level reporting, pre-registered negatives — and reached a clean, defensible result set:

1. **Phase 0 (PASS):** In a lightweight O-RAN slicing simulator, a *fixed* (static) URLLC reservation **cannot be safe-and-rewarding across traffic regimes** — in the URLLC-heavy regime *no* static reservation reaches safety — while a **load/SLA-aware (oracle) constraint** sits beyond the static Pareto frontier. → real headroom for an adaptive shield.
2. **Phase 1 (PASS, honest):** Real PPO/DQN learners confirm the shield's value on three axes — **safe exploration** (DRL-only commits 35k–118k URLLC violations *during training*; shielded ≈0), **distribution-shift robustness** (a policy/static-shield tuned on calm traffic fails catastrophically when traffic surges; the load-aware shield holds), and **converged safety for the weaker learner**. Honestly: converged single-regime safety is *learner-dependent* (PPO largely self-learns it), so we do **not** rest the contribution on it.
3. **Phase 2a (NO-GO, honest negative):** Can an off-the-shelf LLM *generate* the oracle's constraint from SLA text + state, offline? **No.** Qwen3-14B (±RAG) reduces violations vs static but lands ~11× above the oracle — it **under-reserves under backlog** — and **RAG adds nothing** over parametric knowledge.
4. **Phase 2a-v2 (size sweep):** Across **Qwen3-1.7B → 4B → 14B → 32B**, constraint quality improves **monotonically** (refuting "try a smaller model") but **plateaus ~9× above the oracle with diminishing returns**. **Scale is not the lever.**

**Bottom line:** the safe-DRL framing is sound and the *problem* is real and well-characterized; the bottleneck is the **LLM-as-constraint-generator**, not the architecture. The paper-ready contribution is **Phase 0 + Phase 1**, with Phase 2a/2a-v2 as an honest boundary result.

---

## 2. System & method

**Environment (`01_code/env/slicing_env.py`, pure-numpy, gym-compatible).** Three slices share 100 PRBs (allocated in steps of 10 → 66 discrete actions): **eMBB** (throughput SLA), **URLLC** (latency SLA — demand+backlog must be served *this slot* or it is a violation), **mMTC** (access SLA). Time-varying per-slice demand with regimes {balanced, high_embb, high_urllc, bursty}, diurnal cycle, bursts, and channel variation. The headline safety concern: starving URLLC during eMBB bursts.

**Shields.** `none` · `static(min_prb)` · `oracle_margin(reliability)` = reserve URLLC demand+backlog at a pessimistic channel quantile derived from the SLA reliability target (the load/SLA-aware upper bound the LLM is asked to approximate). The executable knob is `urllc_min_prb`, projected by `_project_to_min_urllc`.

**Stack.** DRL: Stable-Baselines3 2.6.0 + Gymnasium 1.1.1 + torch 2.5.1, **CPU** (small MLP; both A100s shared). LLM: Qwen3 family via vLLM (OpenAI-compatible). RAG: reused 2,070-doc O-RAN/3GPP KB + BGE-M3 + bge-reranker-v2-m3 (710,045-chunk cache) from the prior program.

**Discipline (carried over).** Honest `paper_usable` gate computed from *runtime evidence* (real timesteps/generations, on-disk checkpoints, index size) — never a CLI flag; component-level metrics (never a lone scalar); explicit per-phase go/no-go; pre-registered honest negatives; fresh isolated venvs.

---

## 3. Phase 0 — Headroom oracle (offline, no GPU, no LLM)

**Make-or-break claim tested:** does a safety shield even have headroom — i.e., can no *fixed* constraint be both safe and rewarding across regimes, while a load-aware one can?

**Result** (`04_results/phase0_headroom.json`; reward-seeking policy, seeds 42/43/44):

| regime | shield | reward | URLLC viol |
|---|---:|---:|---:|
| high_embb | none | 0.110 | **1.000** |
| | best static (50) | 1.634 | 0.002 |
| | **oracle_margin** | **1.718** | **0.000** |
| high_urllc | none | −0.416 | **1.000** |
| | best static (60) | −0.178 | **0.637** |
| | **oracle_margin** | **0.691** | **0.018** |

**Gate PASS.** Unshielded reward-seeking = 100% URLLC violations; the best static reservation **differs by regime** and in `high_urllc` **no static reservation reaches safety** (best = 63.7%); the load-aware oracle lies on/beyond the static Pareto frontier in both regimes. → headroom for an adaptive shield is real and large.

*(Honest note: the first gate draft failed on a comparison-logic bug and a missing reliability margin in the oracle — both fixed transparently, not tuned-to-pass.)*

---

## 4. Phase 1 — DRL baselines (CPU; 36 runs = 2 algos × 2 regimes × 3 shields × 3 seeds × 300k steps)

Wrapper verified **bit-for-bit faithful** to the env (smoke-test parity 0.00 over 144 cells). Rule-based proportional-fair (PF) baseline; all 36 runs `paper_usable=true`.

**In-regime (seed mean):**

| regime | arm | reward | URLLC viol | train-time viol |
|---|---|---:|---:|---:|
| high_embb | rule PF | −0.098 | 0.943 | — |
| | ppo/none | **1.808** | 0.010 | 0.122 |
| | ppo/oracle_margin | 1.683 | **0.000** | **0.000** |
| | dqn/none | 1.554 | 0.065 | 0.142 |
| high_urllc | rule PF | −0.536 | 0.888 | — |
| | ppo/none | **0.905** | 0.085 | 0.316 |
| | ppo/oracle_margin | 0.661 | **0.030** | **0.030** |
| | dqn/none | 0.412 | 0.333 | 0.383 |
| | dqn/oracle_margin | 0.647 | **0.030** | **0.018** |

**Cross-regime distribution shift (train high_embb → eval high_urllc, URLLC viol):** none **0.76–0.82** · static (traveling floor) **0.57–0.84** · **oracle_margin 0.030**.

**Gate PASS — the shield's value, stated honestly:**
1. **Safe exploration (universal, robust):** DRL-only commits **35k–118k cumulative URLLC violations during training**; shielded arms ≈0–3%. Holds even where the converged policy is safe.
2. **Distribution-shift robustness (the headline motivation):** a policy + static shield tuned on calm traffic fail (0.57–0.84) when traffic surges; only the **load-aware** shield holds (0.030).
3. **Converged safety helps the weaker learner** (DQN high_urllc 0.333→0.030).

**Honest caveat (pre-registered):** converged single-regime safety is **learner-dependent** — PPO largely *internalizes* it (high_embb 0.010, high_urllc 0.085) — so the contribution does **not** rest on it. The shield also has an honest reward cost (over-reserving) except where unsafety was very costly. Figures: `05_figures/phase1/` (safety bars, reward-safety Pareto, safe-exploration cumulative, cross-regime, convergence). Report: `06_reports/PLAN_phase1.md`.

---

## 5. Phase 2a — Offline constraint-credibility gate (Qwen3-14B; 900 states; 1,800 generations)

**Question:** can an LLM *produce* the executable constraint `{urllc_min_prb, reliability_target, reason, citations}` approximating the oracle, **offline**, before any closed loop? Arms: static / oracle_margin / LLM-no-RAG / RAG-LLM. Scored by a **one-step counterfactual** (reconstruct env, force realized channel, project reservation, step once) — reconstruction self-test reproduced logged outcomes **bit-for-bit (0.00 / 900)**; a no-LLM **scoring-credibility pre-gate PASSED** (monotone, over-reservation penalized, discriminating control with teeth). `paper_usable=true`.

| set | static | oracle | LLM-no-RAG | RAG-LLM |
|---|---:|---:|---:|---:|
| **cross** (decisive) | 0.660 | **0.033** | 0.370 | 0.377 |
| high_urllc | 0.083 | 0.033 | 0.083 | 0.080 |
| high_embb | 0.007 | 0.000 | 0.010 | 0.010 |

**Verdict: NO-GO** (honest negative). The LLM **reduces** violations vs static (cross 0.66→0.38) but **cannot match the oracle** (0.033) — it **under-reserves under backlog** (~60–75 PRB where the oracle needs ~100) — and **RAG ≈ no-RAG** (citations valid, schema 1.00; the *decision* is unchanged, not the format). The one-step proxy is *generous* to the LLM (no backlog feedback), so the NO-GO is conservative-correct. Report: `06_reports/PLAN_phase2a.md`.

---

## 6. Phase 2a-v2 — Model-size sweep (Qwen3-1.7B / 4B / 14B / 32B; same gate)

Prompted by "is the model too big — try smaller?". Same gate, same states/scorer; only the producer model changed (32B served on a free GPU1).

| model | params | cross viol (RAG) | reward (RAG) | mean PRB | gate |
|---|---:|---:|---:|---:|---|
| Qwen3-1.7B | 1.7B | **0.660** (= static; useless) | −0.259 | 59 | NO-GO |
| Qwen3-4B | 4B | 0.463 | 0.001 | 70 | NO-GO |
| Qwen3-14B | 14B | 0.367 | 0.107 | 76 | NO-GO |
| Qwen3-32B | 32B | **0.307** | 0.350 | 72 | NO-GO |
| *oracle* | — | **0.033** | 0.417 | ~100 | — |

**Findings:** quality improves **monotonically** with size (0.660→0.307) — **smaller is strictly worse** (the 1.7B can't beat the static floor), refuting the "smaller model" hypothesis — but **plateaus ~9× above the oracle with diminishing returns** (32B gains only 0.06 over 14B for >2× params), with **persistent under-reservation at every scale**. JSON/citation discipline perfect throughout (schema=cite=1.00). **Scale is not the lever.** Figure: `05_figures/phase2a/fig_size_sweep.png`.

*(The sweep's 14B value here, 0.367, vs Phase 2a §5's 0.377 differ by ~0.01 — two separate runs of the same model; vLLM run-to-run non-determinism under 8-wide concurrency. Both NO-GO; conclusions unchanged.)*

---

## 7. Honest limitations

- **Simulator, not a real testbed.** A lightweight slicing gym; ColO-RAN / ns-O-RAN validation is deferred (planned realism backstop).
- **Converged single-regime safety is learner-dependent** (Phase 1) — claimed only as safe-exploration + distribution-shift, not universally.
- **One-step counterfactual** (Phase 2a) bounds *instantaneous* safety, not closed-loop equilibrium (it omits backlog feedback) — which is exactly why closed-loop was gated behind it; the gate's NO-GO makes closed-loop moot for now.
- **N=3 seeds** (Phase 1) — bootstrap CIs reported; not large-N.
- **RAG used a constant domain query** (URLLC SLA facts are state-independent) — efficiency, not a method change.

---

## 8. Overall verdict & what is paper-ready

- **Paper-ready now: Phase 0 + Phase 1.** The headroom oracle (static is *fundamentally insufficient* under traffic shift) + DRL baselines (safe-exploration + distribution-shift robustness; honest learner-dependent converged-safety) form a coherent, reproducible, honestly-reported contribution.
- **Phase 2a + 2a-v2 are an honest boundary result:** an off-the-shelf LLM (1.7B–32B, ±RAG) cannot generate oracle-matching load-aware constraints offline; the bottleneck is constraint-generation calibration, **not** model scale and **not** the safe-DRL framing. This *sharpens* the story rather than weakening it.
- **Closed-loop RAG-shielding is NOT warranted** on current evidence.

## 9. Next options (choose explicitly; no tuning-to-pass)
1. **Calibration, not capacity:** a few-shot / explicit-arithmetic prompt forcing `urllc_min_prb ≥ ⌈(demand+backlog)/(se·g_pess)⌉`; re-run the *same* gate. The most direct test of whether the failure is elicitation vs capability.
2. **Keep the oracle/static shield, drop the LLM from the safety path** — and write up Phase 0+1 (+2a as the boundary).
3. **Realism validation** (ColO-RAN offline traces) of the Phase 0/1 result, independent of the LLM question.

---

## Appendix — artifacts
- **Code:** `01_code/env/{slicing_env.py, slicing_gym_env.py}`, `01_code/smoke_test.py`, `01_code/drl/{drl_common,train_baselines,eval_baselines}.py` + `run_phase1.sh`, `01_code/rag/{state_replay,counterfactual,scoring_credibility,constraint_producers,run_gate,make_figures,sweep_analyze}.py` + `run_size_sweep.sh`, `serve_qwen.sh`.
- **Results:** `04_results/phase0_headroom.json`; `04_results/phase1/{runs/, summary.json, analysis.json}`; `04_results/phase2a/{states_*, scoring_credibility.json, summary[_tag].json, analysis[_tag].json, sweep_summary.json}`.
- **Figures:** `05_figures/phase1/*.png`, `05_figures/phase2a/*.png`.
- **Reports:** `06_reports/{PLAN_phase0.md, PLAN_phase1.md, PLAN_phase2a.md}`, this file.
- **Models:** `02_models/*.zip(+_vecnorm.pkl)` (12 PPO/DQN × shields × seeds). vLLM servers stopped (GPUs freed).
