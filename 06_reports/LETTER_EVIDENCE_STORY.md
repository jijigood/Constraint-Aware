# Letter Evidence Story

## Core Claim
Language models can help interpret SLA and policy evidence, but they should not directly output safety-critical PRB numbers. The safer design is:

`SLA / intent / evidence -> symbolic z_k -> verifier -> solver -> p_min -> shield -> DRL`

The DRL policy receives only the verified scalar constraint strength `p_min/Pmax`, not natural language, JSON fields, or citations.

## Evidence Chain

### Phase2a: direct numeric LLM control is unsafe / unreliable

The direct numeric path `LLM/RAG -> urllc_min_prb` was a pre-registered NO-GO:

- Cross-regime violation: static `0.6600`, no-RAG `0.3700`, RAG `0.3767`, oracle `0.0333`.
- RAG did not add value over no-RAG, and the reward/safety gap to oracle remained large.

### Phase2b-v1: symbolic-z verified compilation works

The symbolic path gate is `PASS: symbolic-z verifier/solver path is deterministic and direct numeric outputs are fail-closed.`.

- Direct numeric outputs are rejected before entering the safety path.
- Oracle/template symbolic specs reproduce the deterministic solver reservation.
- This phase proves symbolic compilation mechanics, not real CER/RAG retrieval.

### Phase2c: mini-CER shows field-aware retrieval is useful

Gate: `True`.

| Arm | Recall@5 | Field accuracy | Mean abs delta p_min | Spec validity |
|---|---:|---:|---:|---:|
| `no_retrieval` | 0.0000 | 0.7000 | 4.1000 | 1.0000 |
| `ordinary_rag_intent_only` | 0.6500 | 0.8500 | 12.2000 | 1.0000 |
| `state_aware_rag` | 0.8000 | 0.9500 | 0.0000 | 1.0000 |
| `field_aware_cer` | 0.9000 | 1.0000 | 0.0000 | 1.0000 |
| `cer_verifier_solver` | 0.9000 | 1.0000 | 0.0000 | 1.0000 |

This supports the narrower retrieval claim: CER is useful when it is evaluated as field-level evidence routing for symbolic constraint construction, rather than as ordinary QA-style RAG.

### Phase3: p_min/Pmax reduces projection burden when the regime is not saturated

| Scenario | Method | Reward | Violation | Mean D_proj | Shield correction | Adaptation delay |
|---|---|---:|---:|---:|---:|---:|
| S3_channel_decay | M3_dynamic_no_aug | -1.5732 ± 0.0007 | 0.8340 ± 0.0000 | 105.3013 ± 30.1234 | 0.9964 ± 0.0040 |  |
| S3_channel_decay | M5_constraint_aware | -1.5713 ± 0.0007 | 0.8340 ± 0.0000 | 125.8907 ± 19.7089 | 0.9968 ± 0.0045 |  |
| S4_sla_upgrade | M3_dynamic_no_aug | 0.6755 ± 0.0237 | 0.0820 ± 0.0165 | 84.0800 ± 24.6743 | 0.7407 ± 0.1080 |  |
| S4_sla_upgrade | M5_constraint_aware | 0.6876 ± 0.0320 | 0.0747 ± 0.0186 | 56.6333 ± 8.8533 | 0.6813 ± 0.1019 |  |
| S5_combined | M3_dynamic_no_aug | -0.4178 ± 0.0475 | 0.4552 ± 0.0156 | 83.2213 ± 15.5029 | 0.8904 ± 0.0898 | 86.0000 ± 4.0000 |
| S5_combined | M5_constraint_aware | -0.4295 ± 0.0132 | 0.4569 ± 0.0093 | 75.2587 ± 11.3539 | 0.9020 ± 0.0368 | 47.0000 ± 35.0000 |
| S6_moderate_decay | M3_dynamic_no_aug | 0.7732 ± 0.0496 | 0.0519 ± 0.0254 | 30.3680 ± 3.0745 | 0.4288 ± 0.0205 | 0.2667 ± 0.3771 |
| S6_moderate_decay | M5_constraint_aware | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 1.4000 ± 1.9799 |

Paired deltas are computed as `M5 - M3`:

- S4 `Delta D_proj`: -27.4467 ± 18.6449; `Delta violation`: -0.0073 ± 0.0190.
- S6 `Delta D_proj`: -7.0640 ± 9.9571; `Delta violation`: 0.0157 ± 0.0225.
- S5 `Delta adaptation delay`: -70.0000 ± 0.0000.
- S3 is the boundary case: the constraint saturates near the resource ceiling, so M5 has little room to improve and should be reported as a near-infeasible regime.


### Phase3-M6: closed-loop CER-z completed

Gate: `passed`.

| Method | Reward | Violation | Mean D_proj | Shield correction | Fallback | Unsafe under-rsv | p_min parity |
|---|---:|---:|---:|---:|---:|---:|---:|
| M5_constraint_aware | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| M6_field_CER_z | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |

This closes the final system loop: the S6 controller replaces Oracle-z with cached real-LLM field-CER-z while keeping the same verified solver/shield/DRL path.


### Phase3-M6 RAG ablation: retrieval errors become control errors

The M6 replay ablation compares oracle, no-retrieval, ordinary RAG, state-aware RAG, and field-CER z-caches under the same S6 M5 policies. It makes the RAG contribution visible: field-CER remains oracle-parity, while weaker retrieval arms can alter `p_min` through over/under reservation or fallback.

## Paper Wording

Use this claim:

> We propose a verifiable symbolic constraint compilation framework for safe O-RAN slicing control. Direct numeric LLM constraint generation is unreliable; symbolic-z verified compilation safely produces `p_min`; and exposing `p_min/Pmax` to the DRL policy reduces shield projection under dynamic, non-saturated constraints, with reward/safety trade-offs reported explicitly.

Avoid this claim:

> The system is a real O-RAN deployment or a fully validated real-world CER/RAG controller.

Phase2c is a controlled mini benchmark for field-level retrieval. Real retrieval over external standards and field attribution remains Phase2c-v2 / future work.
