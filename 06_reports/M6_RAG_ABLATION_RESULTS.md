# M6 RAG Ablation Replay Results

## Verdict
The S6 replay is discriminative: weaker retrieval arms alter the executable constraint path, while field-CER remains oracle-parity.

This experiment replays existing S6 M5 policies under cached Qwen3-4B+BGE symbolic z-caches. It does not train PPO and does not call an LLM.

## Gate Status
- All replay artifact gates passed.

Field-CER gate: `{"fallback_le_0p05": true, "p_min_parity_ge_0p95": true, "unsafe_under_le_0p02": true}`.

Weaker-arm detection: `{"no_retrieval_z": true, "ordinary_rag_z": true, "state_aware_rag_z": false}`.

## Result Table

| Arm | Reward | Violation | Mean D_proj | Correction | p_min parity | Fallback | Under PRB | Over PRB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_z | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| no_retrieval_z | 0.6131 ± 0.0430 | 0.1116 ± 0.0217 | 23.4267 ± 7.6946 | 0.3553 ± 0.0674 | 0.7556 ± 0.0007 | 1.0000 ± 0.0000 | 1.4880 ± 0.0259 | 1.1160 ± 0.0000 |
| ordinary_rag_z | 0.6145 ± 0.0425 | 0.1109 ± 0.0216 | 23.4613 ± 7.6916 | 0.3557 ± 0.0670 | 0.7557 ± 0.0009 | 0.0000 ± 0.0000 | 1.4720 ± 0.0247 | 1.1280 ± 0.0000 |
| state_aware_rag_z | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| field_CER_z | 0.7339 ± 0.0107 | 0.0676 ± 0.0028 | 23.3040 ± 6.9433 | 0.3461 ± 0.0847 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## Selected Paired Deltas vs Oracle
- `ordinary_rag_z-minus-oracle_z` `mean_D_proj`: 0.1573 ± 1.2608.
- `ordinary_rag_z-minus-oracle_z` `fallback_rate`: 0.0000 ± 0.0000.
- `ordinary_rag_z-minus-oracle_z` `p_min_parity_rate`: -0.2443 ± 0.0009.
- `ordinary_rag_z-minus-oracle_z` `mean_under_reservation_prb`: 1.4720 ± 0.0247.
- `ordinary_rag_z-minus-oracle_z` `mean_over_reservation_prb`: 1.1280 ± 0.0000.
- `field_CER_z-minus-oracle_z` `mean_D_proj`: 0.0000 ± 0.0000.
- `field_CER_z-minus-oracle_z` `fallback_rate`: 0.0000 ± 0.0000.
- `field_CER_z-minus-oracle_z` `p_min_parity_rate`: 0.0000 ± 0.0000.
- `field_CER_z-minus-oracle_z` `mean_under_reservation_prb`: 0.0000 ± 0.0000.
- `field_CER_z-minus-oracle_z` `mean_over_reservation_prb`: 0.0000 ± 0.0000.

## Interpretation
- `field_CER_z` is the closed-loop positive control: cached real-LLM field-CER spec stays aligned with Oracle-z.
- `ordinary_rag_z` demonstrates the visible RAG failure mode on S6: it over-reserves at the normal event because intent-only retrieval misses the state-conditioned margin.
- `no_retrieval_z` is intentionally ungrounded and fail-closed; report its fallback behavior rather than treating it as a valid candidate controller.
- The broader field-CER advantage remains the WS-A result across 160 samples and 4 model sizes; this replay shows how retrieval errors propagate into the controller when they change `p_min`.

## Generated Artifacts
- `04_results/phase3_m6_ablation/m6_event_trace.csv`
- `04_results/phase3_m6_ablation/m6_replay_trace.csv`
- `04_results/phase3_m6_ablation/m6_rag_ablation_table.csv`
- `04_results/phase3_m6_ablation/m6_rag_ablation_paired_delta.csv`
- `04_results/phase3_m6_ablation/summary.json`
- `05_figures/phase3_m6_ablation/fig_m6_ablation_control_metrics.png`
- `05_figures/phase3_m6_ablation/fig_m6_ablation_pmin_parity.png`
- `05_figures/phase3_m6_ablation/fig_m6_ablation_reservation_error.png`
