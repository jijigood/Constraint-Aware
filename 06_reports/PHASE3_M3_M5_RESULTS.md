# Phase3 M3/M5 Results

## Gate Status
- All Phase3 formal-run gates passed.

## Result Table

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

## Paired Seed Delta Highlights
Negative deltas improve `mean_D_proj`, violation, shield correction, and delay. Positive deltas improve reward.

- `S4_sla_upgrade` `mean_D_proj`: -27.4467 ± 18.6449 (M5 - M3, n=3).
- `S4_sla_upgrade` `urllc_violation_rate`: -0.0073 ± 0.0190 (M5 - M3, n=3).
- `S6_moderate_decay` `mean_D_proj`: -7.0640 ± 9.9571 (M5 - M3, n=3).
- `S6_moderate_decay` `urllc_violation_rate`: 0.0157 ± 0.0225 (M5 - M3, n=3).
- `S5_combined` `adaptation_delay`: -70.0000 ± 0.0000 (M5 - M3, n=1).
- `S3_channel_decay` `mean_D_proj`: 20.5893 ± 15.8118 (M5 - M3, n=3).

## Interpretation
- S4 supports the state-augmentation claim: under the same Oracle-z dynamic constraint path, M5 reduces projection burden and violation relative to M3.
- S3 should be written as a saturation / near-infeasible regime: `p_min/Pmax` is close to the resource ceiling and both methods keep high violation, so lack of M5 improvement is expected rather than hidden.
- S5 should be written as partial benefit under combined stress: emphasize adaptation delay and projection behavior, not reward dominance.
- S6 is the clean moderate-stress test: it strengthens the claim that `p_min/Pmax` reduces projection/correction burden when the constraint is dynamic but not saturated, while reward and violation should be reported as trade-offs.
- Phase2b-v1 supports symbolic-z verified compilation and direct numeric rejection. It is not real CER/RAG evidence; that belongs to Phase2c.

## Generated Artifacts
- `04_results/phase3_m3_m5/phase3_table.csv`
- `04_results/phase3_m3_m5/paired_seed_delta.csv`
- `05_figures/phase3_m3_m5/fig_phase3_dproj_bars.png`
- `05_figures/phase3_m3_m5/fig_phase3_violation_bars.png`
- `05_figures/phase3_m3_m5/fig_phase3_correction_bars.png`
- `05_figures/phase3_m3_m5/fig_s3_saturation.png`
- `05_figures/phase3_m3_m5/fig_s5_adaptation_delay.png`
- `05_figures/phase3_m3_m5/fig_s4_timeseries.png`
- S4 deterministic replay was generated from local seed-42 checkpoints.
