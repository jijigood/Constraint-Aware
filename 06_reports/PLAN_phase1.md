# Phase 1 — DRL Baseline Validation — RESULT REPORT

**Date:** 2026-06-23 · **Status:** DONE, **gate PASS** (honestly, with nuance) · **Compute:** CPU, 36 runs × 300k steps × 3 seeds
**Artifacts:** `04_results/phase1/{runs/*.json, summary.json, analysis.json}` · `05_figures/phase1/*.png` · models `02_models/*.zip(+_vecnorm.pkl)` · `all_paper_usable=true`
**Stack:** fresh `uv` venv — torch 2.5.1, gymnasium 1.1.1, **SB3 2.6.0** (pinned; SB3 2.6 requires gymnasium <1.2 — caught at install).

## What Phase 1 tested
Confirm — **with a learning agent**, not Phase-0 scripted policies — that the slicing safety shield has real value, and report it honestly even if DRL learns to be safe on its own. Strictly DRL baselines; no LLM/RAG. Wrapper verified **bit-for-bit faithful** to the Phase-0 env (smoke-test parity = 0.00 over 144 cells).

**Arms:** {PPO, DQN} × {high_embb, high_urllc} × shields {none, static(Phase-0 floor 50/60), oracle_margin(rel 0.99)} × seeds {42,43,44}. Baseline = rule-based proportional-fair (PF). Honest `paper_usable` from runtime evidence (consumed timesteps from `model.num_timesteps`, on-disk checkpoint, 20-ep eval from the reloaded model) — never a flag.

## In-regime results (seed mean; full table in `summary.json`)

| regime | arm | reward | URLLC viol | train-time viol | eMBB SLA | mMTC SLA |
|---|---|---:|---:|---:|---:|---:|
| **high_embb** | rule PF | −0.098 | 0.943 | — | — | — |
| | ppo/none | **1.808** | 0.010 | 0.122 | 1.000 | 0.076 |
| | ppo/oracle_margin | 1.683 | **0.000** | **0.000** | 0.854 | 0.170 |
| | dqn/none | 1.554 | 0.065 | 0.142 | 0.745 | 0.182 |
| | dqn/oracle_margin | 1.295 | **0.000** | **0.000** | 0.390 | 0.306 |
| **high_urllc** | rule PF | −0.536 | 0.888 | — | — | — |
| | ppo/none | **0.905** | 0.085 | 0.316 | 0.299 | 0.054 |
| | ppo/oracle_margin | 0.661 | **0.030** | **0.030** | 0.011 | 0.126 |
| | dqn/none | 0.412 | 0.333 | 0.383 | 0.263 | 0.292 |
| | dqn/oracle_margin | 0.647 | **0.030** | **0.018** | 0.004 | 0.216 |

(static arms in `summary.json`: high_embb static≈0.001–0.005 viol; high_urllc static 0.040 (ppo) / 0.269 (dqn).)

## Cross-regime distribution shift (train→eval URLLC viol; `summary.json:cross_regime`)

| train→eval | none | static (traveling floor) | oracle_margin |
|---|---:|---:|---:|
| ppo high_embb→high_urllc | **0.824** | **0.844** | **0.030** |
| dqn high_embb→high_urllc | 0.758 | 0.574 | 0.030 |
| * high_urllc→high_embb | ~0.00–0.05 | ~0.00 | 0.000 |

## Gate verdict — **PASS**, with honest framing
`drl_beats_rule_based_reward_all_regimes` ✓ (DRL 0.41–1.81 ≫ PF −0.10/−0.54, non-overlapping CIs) · `shield_value_in_key_regime` (high_urllc) ✓ · `oracle_is_safety_upper_bound` ✓ (~0.03 floor, ≈ Phase-0 oracle). `PASS=True`.

**The shield's value is real but NOT primarily converged single-regime safety — stated honestly (per the pre-registered branch):**
1. **Safe exploration (robust, universal):** DRL-only commits **12–38%** URLLC violations *during training* in every cell; shielded arms **0–3%**. This holds even where the converged policy is safe — the strongest, most general benefit.
2. **Distribution-shift robustness (the headline motivation for Phase 2):** a DRL policy *and* a static shield tuned on calm `high_embb` traffic fail catastrophically (0.57–0.84 viol) when traffic surges to `high_urllc`; only the **load-aware** shield holds (0.030). Static safety does not transfer; adaptive (load/SLA-aware) safety does.
3. **Converged safety helps the weaker learner:** DQN-only stays unsafe in `high_urllc` (0.333 → 0.030 with shield); PPO-only is already fairly safe there (0.085) and essentially safe in `high_embb` (0.010).

**Honest negative / caveat (do NOT overclaim):** converged single-regime safety is **regime- and algorithm-dependent**. PPO largely *internalizes* safety on its own (high_embb 0.010; high_urllc 0.085), so the shield's converged in-regime benefit for PPO is small. We therefore do **not** rest the contribution on converged single-regime safety. (The binary `drl_only_internalizes_safety` gate flag is False only because DQN/high_embb sits at 0.065 > 0.05; the underlying pattern is mixed and is reported as such.) The shield also has an honest **reward cost** in-regime (oracle_margin reduces reward in 3/4 in-regime cells by over-reserving URLLC — e.g. ppo high_urllc 0.905→0.661) *except* where unsafety was so costly that safety pays for itself (dqn high_urllc 0.412→0.647). All components reported separately, never a lone scalar.

**On increasing difficulty:** because safe-exploration + distribution-shift already establish the shield's value honestly, we do **not** need to inflate a converged gap. If a stronger converged single-regime gap is later wanted, the right knob is **non-stationary/mixed regimes within an episode** (so a single learned reservation can't internalize safety), not raising `beta_violation` (which makes safety easier to learn). Logged as an option, not forced.

## Figures (`05_figures/phase1/`)
`fig_safety_bars` (per-arm viol vs PF + Phase-0 oracle line) · `fig_reward_safety_pareto` (DRL arms vs Phase-0 static frontier+oracle — DRL+oracle reaches the safe corner static can't, esp. high_urllc) · `fig_safe_exploration` (cumulative training violations: none ≫ shield) · `fig_cross_regime` (static breaks under shift, oracle holds) · `fig_convergence`.

## Phase 2 go/no-go → **GO**
The load-aware oracle is the safety upper bound and static fails under traffic shift, so the well-posed Phase-2 problem is exact: **a RAG-grounded LLM shield must approximate the oracle's per-step URLLC reservation from retrieved SLA/standards text + observed load, recovering the cross-regime robustness without the oracle's hand-coded `demand/se/g` knowledge.** Arms: static · LLM-shield-no-RAG · RAG-LLM-shield · oracle (UB); gated by a **constraint-credibility** check (analog of `route_b_reward_credibility.py`) before any constraint enters the closed loop. Reuse the 2,070-doc O-RAN/3GPP KB + BGE-M3 + Qwen3-14B vLLM. Fallback negative is write-ready: "DRL needs safe exploration + load-aware (not static) shielding; converged single-regime safety is learner-dependent."
