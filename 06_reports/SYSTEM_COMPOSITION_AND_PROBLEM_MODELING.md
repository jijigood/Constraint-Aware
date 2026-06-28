# RAG-Compiled Verifiable Constraints with Deterministic Shields for Safe DRL O-RAN Slicing

**Document type:** paper-oriented system composition and problem modeling note  
**Date:** 2026-06-25  
**Project root:** `/home/huangxiaolin/safe_drl_oran/`  
**Positioning:** primary research direction after Phase 0/1/2a; RCA/post-training line is now a supporting empirical study.

---

## Abstract-Style Summary

This project studies safe deep reinforcement learning for O-RAN network slicing under dynamic traffic, SLA constraints, and distribution shift. Prior experiments show that a load/SLA-aware dynamic safety shield can reach URLLC safety where static reservations fail, while DRL improves reward and learning efficiency. However, direct off-the-shelf LLM generation of safety-critical numeric constraints, such as `urllc_min_prb`, fails across model scales from Qwen3-1.7B to Qwen3-32B: schema and citation quality are perfect, but the generated reservations systematically under-protect URLLC.

The resulting system design is therefore not an end-to-end LLM controller. Instead, the LLM/RAG module acts as a **symbolic constraint compiler**: it reads SLA, intent, and standards evidence and outputs a typed, verifiable constraint specification. A deterministic verifier and shield then compute and enforce the safety-critical numeric reservation around a DRL slicing controller. The core research claim is that safe O-RAN control should split semantic interpretation from numerical enforcement: RAG-LLMs handle policy semantics and provenance, while deterministic shields handle safety-critical arithmetic and action projection.

---

## 1. Research Positioning

### 1.1 Motivation

O-RAN slicing control must allocate radio resources among heterogeneous slices such as eMBB, URLLC, and mMTC. The slices have different objectives:

- eMBB prefers high throughput.
- URLLC requires low-latency, high-reliability service.
- mMTC requires broad access/service coverage.

A DRL controller can optimize long-run reward under time-varying demand, but unsafe exploration or distribution shift can cause URLLC latency violations. A safety shield can project unsafe actions into a feasible set, but a fixed static reservation is not enough when traffic regimes change. The central question is therefore:

> Can RAG and LLMs help produce adaptive safety constraints for DRL-based O-RAN slicing without putting the LLM on the safety-critical numeric path?

### 1.2 Empirical Premises Already Established

The current design is grounded in five empirical facts:

1. **Dynamic shield headroom exists.** In high-URLLC traffic, no static reservation reaches safety; the load/SLA-aware oracle shield reaches near-safe behavior.
2. **DRL benefits from shielding.** Shielded training reduces unsafe exploration and improves cross-regime robustness.
3. **Direct LLM numeric reservation fails.** Qwen3-1.7B/4B/14B/32B all under-reserve relative to the oracle; scale improves but does not close the safety gap.
4. **LLM format/citation behavior is reliable.** Phase 2a-v2 shows schema and citation validity can reach 1.00, even when numeric decisions fail.
5. **The prior RCA line gives the same lesson.** LLMs can learn structured formats and robustness, but scalar reward and source-aware ordering are not reliable enough to be trusted as the main safety mechanism.

These facts motivate the compile -> verify -> shield decomposition.

---

## 2. System Composition

The proposed system has seven layers.

```text
SLA / intent / standards evidence
        |
        v
RAG retrieval
        |
        v
LLM symbolic constraint compiler
        |
        v
typed constraint specification
        |
        v
verifier and fail-closed policy
        |
        v
deterministic safety shield
        |
        v
DRL slicing controller -> safe PRB allocation
        |
        v
environment + evaluator + honest gate
```

### 2.1 O-RAN Slicing Environment

The control substrate is a lightweight O-RAN slicing environment implemented in:

- `01_code/env/slicing_env.py`
- `01_code/env/slicing_gym_env.py`

It models three slices sharing a fixed pool of physical resource blocks:

| Slice | Role | Main SLA pressure |
|---|---|---|
| eMBB | throughput-oriented broadband | minimum throughput / high reward for served traffic |
| URLLC | low-latency reliable service | offered load + backlog must be served in the slot |
| mMTC | massive access | served fraction / access coverage |

The environment includes:

- PRB allocation with discrete action granularity;
- high-eMBB and high-URLLC traffic regimes;
- diurnal traffic variation;
- bursts;
- channel variation;
- backlog;
- per-slice SLA indicators;
- reward and component metrics.

### 2.2 DRL Controller

The DRL controller proposes a PRB allocation action. Phase 1 evaluates:

- PPO;
- DQN;
- rule-based proportional-fair baseline;
- no-shield, static-shield, and oracle-shield variants.

Relevant code:

- `01_code/drl/train_baselines.py`
- `01_code/drl/eval_baselines.py`
- `01_code/drl/drl_common.py`

The DRL controller remains the optimization engine. It is not replaced by an LLM.

### 2.3 Deterministic Safety Shield

The shield receives a proposed action and a constraint such as a URLLC minimum PRB floor. It projects the action to the nearest feasible allocation that satisfies the constraint.

Conceptually:

\[
a_t^{safe} = \Pi_{\mathcal{C}_t}(a_t)
\]

where:

- \(a_t\) is the DRL-proposed allocation;
- \(\mathcal{C}_t\) is the feasible action set induced by the compiled constraint;
- \(\Pi_{\mathcal{C}_t}\) is a deterministic projection operator.

The existing implementation is currently embedded in the environment/RAG components rather than a standalone `shield/` directory:

- `_project_to_min_urllc` and oracle shield logic in `01_code/env/slicing_env.py`;
- Phase 2a producer/scoring logic in `01_code/rag/constraint_producers.py`;
- counterfactual one-step safety scoring in `01_code/rag/counterfactual.py`;
- oracle reservation logic reused through `01_code/rag/scoring_credibility.py`.

For the next engineering phase, this shield should be factored into a dedicated module, but the paper model can already describe it as a deterministic projection layer.

### 2.4 RAG Evidence Layer

The RAG layer retrieves SLA, standards, intent, and policy evidence. It reuses the existing telecom/O-RAN knowledge assets:

- 2,070-document O-RAN/3GPP KB;
- BGE-M3 embedding cache;
- existing retrieval/serving stack from the earlier telecom RAG program.

Its role is not to produce the reservation number. Its role is to provide textual grounding for:

- SLA type;
- reliability target;
- latency semantics;
- slice priority;
- applicable standards/policies;
- citation provenance.

### 2.5 LLM Symbolic Constraint Compiler

The LLM reads:

- retrieved evidence;
- operator intent;
- SLA template;
- current network summary if needed.

It outputs a typed constraint specification, not a safety-critical reservation number.

Example output:

```json
{
  "constraint_type": "urllc_latency_reliability",
  "slice": "urllc",
  "metric": "slot_level_latency_service",
  "service_rule": "serve_offered_load_plus_backlog_in_current_slot",
  "reliability_target": 0.99,
  "channel_margin_policy": "pessimistic_quantile",
  "formula_id": "load_backlog_over_spectral_efficiency",
  "units": {
    "load": "Mbps",
    "capacity": "Mbps_per_PRB",
    "reservation": "PRB"
  },
  "applicability": {
    "traffic_regime": ["high_urllc", "bursty", "mixed"],
    "slice_priority": "URLLC before eMBB throughput"
  },
  "citations": ["sla_doc_3", "oran_policy_7"]
}
```

The critical design rule is:

> The LLM may select the constraint type and semantic parameters, but it may not directly set `urllc_min_prb`.

### 2.6 Verifier

The verifier checks that the compiled specification is usable before it reaches the shield.

Checks include:

| Check | Purpose |
|---|---|
| schema validity | output is parseable and complete |
| type validity | fields have valid types and units |
| citation validity | cited IDs exist in retrieved evidence |
| range validity | thresholds/reliability values are in accepted ranges |
| formula validity | `formula_id` maps to an approved deterministic formula |
| feasibility | resulting constraint can be enforced under the PRB budget |
| monotonic safety sanity | stronger reliability target should not reduce reservation |

If verification fails, the system should fail closed:

```text
invalid spec -> conservative default shield or oracle-style fallback
```

This turns LLM errors into detectable specification failures rather than unsafe control actions.

### 2.7 Evaluation and Honest Gate

All result artifacts should preserve the existing discipline:

- `paper_usable=true` only with runtime evidence;
- component-level metrics before scalar summaries;
- JSON result files as the source of all reported numbers;
- explicit go/no-go gates;
- write-ready negative if a phase fails.

Relevant result directories:

- `04_results/phase0_headroom.json`
- `04_results/phase1/summary.json`
- `04_results/phase1/analysis.json`
- `04_results/phase2a/summary*.json`
- `04_results/phase2a/analysis*.json`
- `04_results/phase2a/sweep_summary.json`

---

## 3. Problem Modeling

### 3.1 Slicing as a Constrained Markov Decision Process

We model O-RAN slicing as a constrained MDP:

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \mathcal{C}, \gamma)
\]

where:

- \(\mathcal{S}\): network states;
- \(\mathcal{A}\): PRB allocation actions;
- \(P\): traffic/channel/backlog transition dynamics;
- \(r\): reward function;
- \(\mathcal{C}\): SLA-induced safety constraints;
- \(\gamma\): discount factor.

At each time step \(t\), the controller observes state \(s_t\), proposes an action \(a_t\), the shield projects it to \(a_t^{safe}\), and the environment returns reward and component metrics.

### 3.2 State Space

The environment state contains:

\[
s_t =
\{
d_t^{embb}, d_t^{urllc}, d_t^{mmtc},
b_t^{embb}, b_t^{urllc}, b_t^{mmtc},
g_t,
\tau_t
\}
\]

where:

- \(d_t^i\): offered load for slice \(i\);
- \(b_t^i\): backlog for slice \(i\);
- \(g_t\): channel gain;
- \(\tau_t\): normalized time within the episode.

In the implementation this appears as an 8-dimensional observation vector.

### 3.3 Action Space

An action is a PRB allocation vector:

\[
a_t = (p_t^{embb}, p_t^{urllc}, p_t^{mmtc})
\]

subject to:

\[
p_t^{embb} + p_t^{urllc} + p_t^{mmtc} = P_{\max}
\]

and quantization:

\[
p_t^i \in \{0, \Delta p, 2\Delta p, \dots, P_{\max}\}
\]

In the current environment:

- \(P_{\max}=100\);
- \(\Delta p=10\).

### 3.4 Reward Function

The scalar reward summarizes served traffic and SLA satisfaction:

\[
r_t =
w_e \cdot \text{served}^{embb}_t
 w_m \cdot \text{served}^{mmtc}_t
 w_u \cdot \mathbb{1}[\text{URLLC safe}]
- \beta \cdot \mathbb{1}[\text{URLLC violation}]
\]

The paper should not rely on the scalar reward alone. It must report:

- URLLC violation rate;
- eMBB SLA satisfaction;
- mMTC SLA satisfaction;
- mean reward;
- Jain fairness;
- PRB utilization/allocation;
- training-time unsafe exploration.

### 3.5 URLLC Safety Constraint

The core safety constraint is:

\[
\text{capacity}^{urllc}_t \ge d_t^{urllc} + b_t^{urllc}
\]

where:

\[
\text{capacity}^{urllc}_t = p_t^{urllc} \cdot se^{urllc} \cdot g_t
\]

For a reliability-aware shield, use a pessimistic channel estimate:

\[
g_t^{pess} = f(g_t, \rho)
\]

where \(\rho\) is the reliability target. The deterministic reservation is:

\[
p_{min,t}^{urllc}
= \left\lceil
\frac{d_t^{urllc} + b_t^{urllc}}
{se^{urllc} \cdot g_t^{pess}}
\right\rceil_{\Delta p}
\]

where \(\lceil \cdot \rceil_{\Delta p}\) means snap upward to the PRB action granularity.

This is the number the LLM should not directly generate. The solver computes it from verified symbolic constraints and runtime state.

### 3.6 Shield Projection

Given the DRL action \(a_t\), the shield enforces:

\[
p_t^{urllc} \ge p_{min,t}^{urllc}
\]

The projected action is:

\[
a_t^{safe}
= \arg\min_{a \in \mathcal{A}}
\|a - a_t\|_1
\quad
\text{s.t.}
\quad
p^{urllc} \ge p_{min,t}^{urllc}
\]

The shield is deterministic, auditable, and independent of LLM numeric calibration.

### 3.7 RAG-Compiled Constraint Specification

Let:

- \(D\): corpus of SLA/standard/policy documents;
- \(q\): retrieval query derived from intent and SLA;
- \(E_q = \text{Retrieve}(q, D)\): retrieved evidence;
- \(x_t\): current network summary or control context;
- \(I\): operator intent.

The LLM compiler produces:

\[
z_t = \text{Compile}_{LLM}(I, E_q, x_t)
\]

where \(z_t\) is a symbolic constraint specification, not an action and not a PRB reservation.

The verifier maps:

\[
V(z_t) \rightarrow \{\text{valid}, \text{invalid}\}
\]

If valid, the deterministic solver computes:

\[
p_{min,t}^{urllc} = F(z_t, s_t)
\]

If invalid:

\[
p_{min,t}^{urllc} = F_{fallback}(s_t)
\]

This makes the safety path:

```text
LLM semantic compile -> verifier -> deterministic formula -> shield projection
```

not:

```text
LLM directly emits safety-critical PRB value
```

### 3.8 Optimization Objective

The constrained objective is:

\[
\max_{\pi}
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T}
\gamma^t r(s_t, a_t^{safe})
\right]
\]

subject to:

\[
\Pr[\text{URLLC violation}] \le \epsilon
\]

and:

\[
a_t^{safe} = \Pi_{\mathcal{C}(z_t, s_t)}(\pi(s_t))
\]

The compiler is successful if it yields constraints that let the deterministic shield approach oracle safety while preserving reward.

---

## 4. Experimental Questions

### Q1. Is a dynamic shield necessary?

Already supported by Phase 0:

- static reservation cannot cover both high-eMBB and high-URLLC regimes;
- dynamic oracle reaches near-safe behavior under high-URLLC.

### Q2. Does shielding matter for DRL?

Already supported by Phase 1:

- safe exploration improves strongly;
- cross-regime robustness improves;
- DQN high-URLLC converged safety improves with oracle shield;
- PPO may internalize safety in single-regime training, so the strongest claim should be distribution-shift and training-time safety.

### Q3. Can an off-the-shelf LLM directly emit the numeric safety reservation?

Answered negatively by Phase 2a/2a-v2:

- larger models improve monotonically but remain far from oracle;
- under-reservation persists;
- RAG helps citations more than decisions;
- schema/citation success does not imply control correctness.

### Q4. Can an LLM compile verifiable symbolic constraints that a deterministic shield can enforce?

This is the next primary research question.

Required arms:

| Arm | Description |
|---|---|
| static shield | fixed URLLC reservation |
| direct LLM numeric | Phase 2a-style `urllc_min_prb` generation |
| LLM symbolic compiler + solver | no RAG |
| RAG-LLM symbolic compiler + solver | proposed method |
| oracle margin | upper-bound safety reference |

---

## 5. Go / No-Go Gates

### G1 — Safety and Reward

The compiled-constraint shield should approach the oracle safety frontier:

```text
URLLC violation <= oracle_margin + epsilon
AND reward >= oracle_margin - delta
AND safer than static under cross-regime shift
```

Avoid comparing reward against an unsafe static baseline without matching safety.

### G2 — Verifiable Soundness

The system should show:

```text
typed_spec_validity high
zero unsafe specs pass verifier
invalid specs fail closed
```

A failed verifier is acceptable if the fallback is safe and reported honestly.

### G3 — RAG Sensitivity

RAG must affect the compiled specification in the right way:

```text
changing SLA/reliability/intent text changes the symbolic spec correctly
```

The system should not repeat the Phase 2a failure where RAG improved citations but not decisions.

### G4 — Shift Robustness

The compiled shield should remain safe across:

- high-eMBB regime;
- high-URLLC regime;
- bursty regime if added;
- altered reliability targets;
- noisy or ambiguous SLA text.

---

## 6. Evaluation Metrics

Report metrics by component:

| Metric | Meaning |
|---|---|
| URLLC violation rate | primary safety metric |
| reward | scalar control utility |
| eMBB SLA rate | throughput service preservation |
| mMTC SLA rate | access service preservation |
| Jain fairness | allocation balance |
| reservation mean/p95 | conservatism of shield |
| reward at matched safety | fair comparison against static |
| typed spec validity | compiler format/syntax quality |
| citation validity | RAG grounding discipline |
| verifier rejection rate | spec failure detection |
| unsafe-spec pass rate | should be zero |
| fallback rate | how often fail-closed path is used |
| RAG sensitivity score | whether retrieved evidence changes the correct fields |

---

## 7. Expected Paper Claims

### Claims the current evidence already supports

1. Static shielding is insufficient under traffic-regime shift.
2. DRL benefits from shielding in safe exploration and cross-regime robustness.
3. Direct LLM numeric safety generation fails across local model scales.
4. Schema/citation correctness is not enough for safety-critical control.

### Claims Phase 2b should test

1. RAG-LLM symbolic compilation plus deterministic shielding can recover oracle-like safety.
2. Verification prevents LLM semantic errors from entering the safety-critical numeric path.
3. RAG matters when it changes typed constraints correctly under SLA/intent variation.

### Claims to avoid

- LLM directly controls the network.
- LLM directly computes reliable safety reservations.
- RAG automatically improves control decisions.
- This is ready for real O-RAN deployment.
- The RCA/source-aware line is the main contribution.

---

## 8. Relationship to the Supporting RCA Study

The Track A -> Route B -> Route B′-Train line is not discarded. It supports the main thesis by showing:

- inference-time sufficiency signals can fail even when oracle headroom exists;
- LLM confidence and scalar reward are unreliable as control signals;
- training can improve robustness, but source-aware priors and scalar DPO are not automatically reliable.

The RCA line should be cited as a supporting empirical study, not as the primary method contribution. The primary system is the Safe DRL constraint-compiler architecture.

---

## 9. Reproducibility Pointers

Current relevant assets:

| Asset | Path |
|---|---|
| Phase 0 plan | `/home/huangxiaolin/safe_drl_oran/06_reports/PLAN_phase0.md` |
| Phase 1 plan | `/home/huangxiaolin/safe_drl_oran/06_reports/PLAN_phase1.md` |
| Phase 2a plan | `/home/huangxiaolin/safe_drl_oran/06_reports/PLAN_phase2a.md` |
| slicing env | `/home/huangxiaolin/safe_drl_oran/01_code/env/slicing_env.py` |
| gym wrapper | `/home/huangxiaolin/safe_drl_oran/01_code/env/slicing_gym_env.py` |
| DRL training/eval | `/home/huangxiaolin/safe_drl_oran/01_code/drl/` |
| Phase 2a producers | `/home/huangxiaolin/safe_drl_oran/01_code/rag/constraint_producers.py` |
| counterfactual scorer | `/home/huangxiaolin/safe_drl_oran/01_code/rag/counterfactual.py` |
| scoring credibility | `/home/huangxiaolin/safe_drl_oran/01_code/rag/scoring_credibility.py` |
| Phase 0 result | `/home/huangxiaolin/safe_drl_oran/04_results/phase0_headroom.json` |
| Phase 1 results | `/home/huangxiaolin/safe_drl_oran/04_results/phase1/` |
| Phase 2a results | `/home/huangxiaolin/safe_drl_oran/04_results/phase2a/` |

---

## 10. Suggested Paper Outline

```text
1. Introduction
   - safe DRL for O-RAN slicing
   - why LLMs should not directly control numeric safety constraints

2. Motivation and Empirical Boundary
   - static shield failure
   - DRL shield value
   - direct LLM numeric NO-GO

3. System Architecture
   - RAG evidence layer
   - LLM symbolic constraint compiler
   - verifier
   - deterministic shield
   - DRL controller

4. Problem Formulation
   - constrained MDP
   - URLLC latency/reliability constraint
   - shield projection
   - compile -> verify -> solve formalization

5. Experiments
   - Phase 0/1 recap
   - direct LLM numeric baseline
   - symbolic compiler + solver
   - RAG/no-RAG and SLA-shift ablations

6. Results
   - safety/reward Pareto
   - verifier validity
   - RAG sensitivity
   - shift robustness

7. Discussion
   - LLM reliability boundary
   - why deterministic enforcement is necessary
   - limitations and real O-RAN path

8. Conclusion
```

---

## 11. One-Sentence Thesis

> RAG-LLMs are useful for compiling textual SLA and intent into typed, verifiable constraint specifications, but safe O-RAN DRL requires deterministic numerical enforcement; the LLM should explain and compile the rule, not compute the safety-critical reservation.

