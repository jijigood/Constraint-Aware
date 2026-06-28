# Theme A — RAG-Conditioned Safe DRL for Intent-Aware 6G O-RAN Slicing

**Started:** 2026-06-23 · **Owner:** Owner A · **Engine:** Claude Code (meta-supervisor)
**Lineage:** second paper after the *budgeted/distractor-robust evidence* program (`~/telecom_rag_research`).
Carries over that program's discipline: **de-risk the make-or-break claim offline first**, gate every
later phase on it, reuse existing machinery, never report a number that isn't in a result JSON, respect
the GPU/network walls.

## Scoping (user-locked 2026-06-23)
- **Control substrate:** lightweight reproducible Python slicing gym **now**; validate on real
  **ColO-RAN** offline traces **later** (phased — sim de-risks, ColO-RAN is the realism backstop).
- **Headline novelty:** a **RAG-grounded LLM safety shield + dynamic constraints**. The LLM converts
  retrieved SLA/standards text into *executable, load-aware* constraints; a shield projects unsafe DRL
  actions into the feasible set. DRL stays the controller. (Theme C of the directions doc — chosen as the
  least-overlapping-with-ORAN-GUIDE, most-defensible angle; reuses the reward-credibility / honest-gate
  discipline from the prior program.)

## Environment facts (probed 2026-06-23)
- GPU0 saturated (100%, ~62 GB) by another user; **GPU1 free** (1× A100 80 GB available).
- **No** O-RAN control platform installed (ns-O-RAN / ORANSlice / ColO-RAN all absent).
- **No** DRL stack (`gymnasium` / `stable_baselines3` absent in every venv; anaconda has `ray`+`torch`).
- Network reachable (GitHub + HF via proxy). Staged locally: COOPER, TMF921 intent-to-config,
  telecom-intent-config-sft-10k, telecom-knowledge-graph-rel19, ElectricalElectronicsIR.

---

## The make-or-break claim (test BEFORE installing DRL or touching the LLM)
> In a realistic slicing env, (1) a reward-seeking policy genuinely produces non-trivial **URLLC SLA
> violations** (unsafe actions occur and cost reward); (2) a **fixed static** safety constraint cannot
> simultaneously minimize violations *and* preserve reward across traffic regimes; and (3) a
> **load/SLA-aware (oracle) dynamic** constraint reaches ~0 violations at strictly higher reward than the
> best static constraint — i.e. **there is headroom an adaptive RAG-grounded shield can fill.**

If (3) is false (a single static reservation already dominates), the LLM shield is pointless → report the
"static-shield-suffices" negative and stop. Everything below is gated on this. This is the direct analog
of the prior program's "a ~110-token sufficient subset exists but no deployable signal finds it" oracle.

## Phase 0 — offline headroom oracle (NO GPU, NO LLM, pure numpy) — **DONE, GATE PASS (2026-06-23)**

**Result (`04_results/phase0_headroom.json`, policy=throughput_greedy, seeds 42/43/44):**
- Unshielded reward-seeking = **100% URLLC violations** in both regimes (unsafe actions are real & costly).
- **Best static reservation differs by regime** (high_embb→50 PRB, high_urllc→60) and deploying one in
  the other loses safety/reward → no universal fixed constraint.
- **high_urllc: NO static reservation reaches safety** (best = static(60) @ 63.7% violations) — static
  shielding is *fundamentally insufficient*, not just suboptimal, when URLLC demand surges.
- **Load/SLA-aware oracle lies on/beyond the static Pareto frontier in both regimes:** high_embb
  oracle_margin 1.718 reward @ **0.000** viol (beats static(50) 1.634 @ 0.002); high_urllc oracle_margin
  0.691 reward @ **0.018** viol (vs best static −0.178 @ 0.637). → **headroom for an adaptive RAG shield
  is real and large.**

**Honest notes (so the gate stays trustworthy):** (a) the first gate draft FAILED because it compared the
oracle to an arbitrary "safe" static point instead of the Pareto frontier — fixed to a proper
non-domination test. (b) The first oracle reserved the bare minimum (no reliability margin) and leaked ~8%
violations from in-slot channel noise; added `oracle_margin` that reserves at a pessimistic channel
quantile derived from a **reliability target** (the same SLA quantity the LLM shield will read) — this is
the principled upper bound, not tuning-to-pass. (c) In high_urllc, oracle_margin drives eMBB SLA to ~0
(URLLC eats nearly all PRBs): expected — when URLLC demand is genuinely huge, safety *requires* sacrificing
eMBB; the point is the dynamic constraint finds that frontier and static cannot.

### Phase 0 — original spec (executed as below)
- Build a lightweight slicing gym (eMBB throughput-SLA / URLLC latency-SLA / mMTC access-SLA; PRB
  allocation; time-varying demand with regimes + diurnal + bursts; channel variation). gym-style
  `reset/step`, numpy-only so it runs on CPU now and wraps cleanly for SB3 later.
- Shields: `none`, `static(min_prb_urllc)` swept over a grid, `dynamic_oracle` (load-aware min PRB =
  ceil(demand/se/channel) — the perfect-knowledge upper bound the LLM shield will approximate from SLA
  text + observed load).
- Policies (no DRL needed): `random`, `throughput_greedy` (ignores URLLC — mimics unsafe exploration),
  `reward_greedy` (myopic argmax of immediate reward).
- **Report (component-level, never just scalar reward — prior-program rule):** URLLC violation rate,
  SLA-satisfaction per slice, mean reward, Jain fairness, unsafe-action rate, PRB utilization.
- **GATE (go/no-go):** across ≥2 regimes (e.g. high-eMBB vs high-URLLC), (a) `none` shield violation
  rate ≥ ~15%; (b) the static-shield reward-vs-violation Pareto frontier has a knee (reward drops as the
  reservation grows); (c) the best static `min_prb_urllc` *differs by regime*; (d) `dynamic_oracle`
  Pareto-dominates the best static (≥0 violations at ≥ best-static reward). → proceed; else stop + report.
- Artifact: `04_results/phase0_headroom.json` (`paper_usable=false`, offline oracle), printed table + an
  explicit go/no-go line.

## Phase 1 — DRL baselines on the sim (GPU1; gated on Phase 0) — PLANNED
- Install a clean DRL venv (gymnasium + stable_baselines3 + torch, CUDA on GPU1; mirror the prior
  program's "fresh uv venv, don't pollute existing envs" rule). Wrap the env as a Gymnasium env.
- Baselines: rule-based (proportional-fair) · DRL-only (PPO + DQN) · DRL + static shield · DRL + oracle
  dynamic shield (upper bound). Report the same components + reward convergence + sample efficiency.
- **GATE:** DRL-only beats rule-based on reward but has materially higher URLLC violations than
  DRL+shield; DRL+oracle-shield is the safety upper bound. Confirms the shield's value with a real learner.

## Phase 2 — RAG-grounded LLM safety shield (GPU; gated on Phase 1) — PLANNED
- KB: reuse the existing 2,070-doc O-RAN/3GPP KB + BGE-M3 cache from the prior program; add SLA/intent
  templates (TMF921 intent-to-config + telecom-knowledge-graph-rel19 are staged locally).
- LLM (served via vLLM, reuse Qwen3-14B stack): given retrieved SLA/standards + current network summary
  → emit a **structured, executable constraint** (e.g. per-slice min-PRB / max-latency-prob / priority),
  validated by a schema + a **constraint-credibility gate** (the analog of `route_b_reward_credibility.py`:
  the LLM-derived constraint must, on held-out steps, reduce violations without needless reward loss vs
  static, BEFORE it's allowed into the closed loop). No constraint enters the shield un-gated.
- Arms: static shield · LLM-shield-no-RAG · RAG-LLM-shield · oracle (upper bound). Robustness: noisy
  retrieval, wrong/ambiguous SLA text (reuse the distractor-robustness ethos).
- **GATE:** RAG-LLM-shield approaches the oracle Pareto frontier and beats static + LLM-no-RAG at equal
  reward; constraint-credibility passes before any headline claim.

## Phase 3 — ColO-RAN realism validation + ablations — PLANNED
- Port the shield/eval to ColO-RAN offline traces (download when Phase 2 passes); ablations (no-RAG,
  noisy retrieval, wrong SLA, unseen traffic), bootstrap CIs, error analysis. Optional ns-O-RAN PoC.

## Guardrails (carried from the prior program)
- Component-level reporting always; scalar reward only as a summary.
- Honest gate: a run is `paper_usable` only with the real learner/LLM + runtime evidence, never CLI flags.
- Fresh venvs; pin `CUDA_VISIBLE_DEVICES=1` (GPU0 is another user's); greedy/seeded for reproducibility.
- Every phase has an explicit go/no-go; the fallback negative is always write-ready.
- Don't overbuild: no unified multi-env framework until a second env (ColO-RAN) forces it.
