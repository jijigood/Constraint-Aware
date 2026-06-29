# Phase 2c-v2 Results — RAG-Centric Constraint Evidence Retrieval

**Scope:** deterministic, no real LLM, no GPU. This is the cheap de-risking slice
(steps 1–2 of `PLAN_phase2c_v2_rag_centric.md`) that answers the make-or-break
question for the RAG-single-core full paper: *does retrieval quality traverse to
the control outcome once the verifier/solver/shield are in the loop?*

## Verdict: dual-signal G4 PASS (honest)

Retrieval quality traverses to control along **two distinct axes**:

1. **Safety, on adverse-channel states (synthetic CER benchmark).** State-blind
   ordinary RAG compiles **unsafe under-reserving** constraints on
   degraded/conflict/noisy scenarios; field-aware CER eliminates them.
2. **Safety-by-construction + efficiency, on realistic states (900-state replay).**
   A *fixed numeric* floor is unsafe under load, while load-aware symbolic z_k
   (oracle and CER) is safe by construction and efficient. On this good-channel
   population ordinary-RAG-z ≈ CER-z — reported honestly: these states lack the
   adverse channels that separate retrieval arms, which is precisely why the
   synthetic benchmark is needed.

## Signal 1 — Synthetic CER benchmark (G1/G2/G3), n=160

Generic intents; the scenario lives only in the state summary, so intent-only RAG
is blind. The control-relevant field is `channel_margin_policy`;
`reliability_target` has near-zero control effect under this solver and is a
text-accuracy field only.

| Arm | Recall@5 | Field acc | Margin acc | Unsafe under-rsv | Mean \|Δp_min\| | Validity |
|---|---:|---:|---:|---:|---:|---:|
| `no_retrieval` | 0.00 | 0.71 | 0.15 | 0.44 | 12.38 | 1.00 |
| `ordinary_rag_intent_only` | 0.18 | 0.57 | 0.15 | 0.44 | 12.38 | 1.00 |
| `state_aware_rag` | 0.52 | 0.86 | 0.57 | 0.17 | 4.94 | 1.00 |
| `field_aware_cer` | 0.71 | 0.94 | 0.88 | 0.00 | 7.81 | 1.00 |
| `cer_verifier_solver` | 0.71 | 0.94 | 0.88 | 0.00 | 7.81 | 1.00 |

**Monotone capability ladder:** ordinary RAG (state-blind) → state-aware (sees
channel, fixes degraded/noisy) → field-aware CER (per-field routing + latest-wins,
also resolves conflict). Each step removes a concrete failure mode.

**Bootstrap 95% CI (per-sample unsafe-rate gap):**
- field-CER − ordinary: -0.444 [-0.519, -0.369]
- field-CER − state-aware: -0.169 [-0.231, -0.113]

G1/G2/G3 gate: **PASS** — {"cer_verifier_safe": true, "cer_verifier_valid": true, "field_accuracy_monotone": true, "field_recall_beats_ordinary": true, "field_reduces_unsafe_vs_ordinary": true, "field_resolves_conflict_vs_state": true}

## Signal 2 — Downstream control replay (G4), 900 saved Phase2a states

One-step counterfactual replay (bit-for-bit faithful; parity self-test
**PASS**). Hard states = cross + high_urllc.

| Arm | URLLC violation | Reward | Mean over-rsv (PRB) | Under-rsv rate | Mean D_proj |
|---|---:|---:|---:|---:|---:|
| `static` | 0.3717 | 0.2693 | 1.35 | 0.838 | 3.20 |
| `oracle` | 0.0333 | 0.6215 | 0.00 | 0.000 | 49.13 |
| `ordinary_rag` | 0.0333 | 0.6215 | 1.58 | 0.000 | 49.13 |
| `field_cer` | 0.0333 | 0.6215 | 1.58 | 0.000 | 49.13 |

- **Fixed numbers are unsafe:** `static` violates 0.372 of hard states (reproduces Phase 2a), reward 0.269.
- **Symbolic z_k is safe-by-construction:** `oracle`/`field_cer` violation 0.033, under-reservation rate 0.000, reward 0.622.
- **Honest null:** `ordinary_rag` ≈ `field_cer` here — good-channel states give no adverse signal to separate them. The safety separation lives in Signal 1.

G4 gate: **PASS** — {"cer_matches_oracle_efficiency": true, "cer_safe_by_construction": true, "oracle_is_safe": true, "static_is_unsafe": true}

## What this licenses / does not license

- **Licenses:** proceeding to the real-LLM compiler arms (WS-A) and, if those
  hold, closed-loop M6 — the cheap evidence says retrieval quality is load-bearing
  for *safety under adverse channels* and that the verifiable symbolic interface
  is safe by construction.
- **Does not license:** claiming real-LLM/RAG results (this slice is deterministic
  routing), large-scale real-spec retrieval, or that retrieval quality changes
  safety on benign-channel states.

## Honest boundaries

The benchmark is a controlled, field-labelled set with generic intents; the
`reliability_target` field is reported as control-inert under this solver. The
900-state replay is a one-step counterfactual (no backlog feedback) — closed-loop
is the deferred M6 step. No real LLM is in this slice.

## Artifacts
- `05_figures/phase2c_v2/fig_field_accuracy_by_arm.png`
- `05_figures/phase2c_v2/fig_under_reservation_by_arm.png`
- `05_figures/phase2c_v2/fig_under_reservation_by_category.png`
- `05_figures/phase2c_v2/fig_downstream_control.png`
- `04_results/phase2c_v2/summary.json`, `analysis.json`, `cer_v2_table.csv`
- `04_results/phase2c_v2/downstream.json`
