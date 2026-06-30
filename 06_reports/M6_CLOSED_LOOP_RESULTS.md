# M6 Closed-Loop CER-z Results

## Conclusion
CER-z can replace oracle-z in the S6 closed-loop controller with limited degradation.

## Gate Status
- All formal-run gates passed.

Hard safety gates: `{"fallback_rate_le_0p05": true, "p_min_parity_ge_0p95": true, "unsafe_under_le_0p02": true}`.

Limited-degradation gates: `{"correction_within_0p10": true, "mean_D_proj_within_7p5": true, "reward_within_0p05": true, "violation_within_0p03": true}`.

## Result Table

| Method | Reward | Violation | Mean D_proj | Shield correction | Fallback | Unsafe under-rsv | p_min parity |
|---|---:|---:|---:|---:|---:|---:|---:|
| M5_constraint_aware | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| M6_field_CER_z | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## Paired Seed Delta
Positive delta improves reward/parity; negative delta improves violation, projection, and correction.

- `reward`: 0.0000 ± 0.0000 (M6 - M5, n=3).
- `urllc_violation_rate`: 0.0000 ± 0.0000 (M6 - M5, n=3).
- `mean_D_proj`: 0.0000 ± 0.0000 (M6 - M5, n=3).
- `shield_correction_rate`: 0.0000 ± 0.0000 (M6 - M5, n=3).
- `p_min_parity_rate`: 0.0000 ± 0.0000 (M6 - M5, n=3).

## Interpretation
- M6 uses cached real-LLM `field-aware CER -> z_k` from WS-A; no LLM is called during DRL training or evaluation.
- The comparison is intentionally narrow: `S6_moderate_decay`, same PPO setup, same `p_min/Pmax` state augmentation, Oracle-z replaced by CER-z.
- Report M6 as the final closed-loop bridge after Phase2c: it tests whether the compiled CER-z remains usable once inserted into the controller.

## Generated Artifacts
- `04_results/phase3_m6/m6_table.csv`
- `04_results/phase3_m6/m6_paired_delta.csv`
- `04_results/phase3_m6/summary.json`
- `05_figures/phase3_m6/fig_m6_closed_loop_bars.png`
- `05_figures/phase3_m6/fig_m6_safety_parity.png`
