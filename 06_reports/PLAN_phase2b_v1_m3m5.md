# Phase2b-v1 + Phase3 M3/M5 Implementation Note

**Status:** implemented scaffold + offline gate + quick training smoke.  
**Branch:** `phase2b-v1-m3m5`.  
**Scope:** no real LLM/RAG calls; no full 300k-step DRL grid yet.

## What This Adds

This phase turns the Phase2a direct numeric NO-GO into a safer symbolic path:

```text
symbolic z_k -> verifier -> deterministic solver -> p_min -> shield
```

It also adds the first closed-loop entrypoints for the M3/M5 comparison:

```text
M3_dynamic_no_aug:      obs = x_t
M5_constraint_aware:   obs = [x_t, p_min/Pmax]
```

The M3/M5 path uses Oracle-z only in this phase, so the effect of state
augmentation is not mixed with CER/RAG errors.

## New Experiment Entrypoints

```bash
# Phase2b-v1 symbolic-z offline gate
.venv/bin/python -m safe_oran.experiments.run_phase2b_offline --smoke 5
.venv/bin/python -m safe_oran.experiments.run_phase2b_offline

# Phase3 M3/M5 quick smoke
.venv/bin/python -m safe_oran.experiments.train_m3_m5 \
  --method M3_dynamic_no_aug --scenario S4_sla_upgrade --seed 42 --quick
.venv/bin/python -m safe_oran.experiments.train_m3_m5 \
  --method M5_constraint_aware --scenario S4_sla_upgrade --seed 42 --quick
.venv/bin/python -m safe_oran.experiments.eval_m3_m5
```

## Implemented Checks

- S3/S5 now support explicit linear channel decay in `safe_oran` scenario configs.
- S3 reaches the configured channel endpoint within one episode.
- S5 combines bursty traffic, SLA upgrade, and channel decay.
- Direct numeric legacy outputs are rejected by the verifier.
- Oracle-z and template symbolic-z reproduce oracle `p_min` exactly on the saved Phase2a states.
- M3/M5 quick training writes checkpoints and JSON metrics.

## Current Artifacts

- `04_results/phase2b_v1/summary.json`
- `04_results/phase2b_v1/analysis.json`
- `04_results/phase3_m3_m5/runs/*.json`
- `04_results/phase3_m3_m5/summary.json`

## Next Full Runs

Run the full M3/M5 grid after the quick smoke:

```bash
for SCENARIO in S3_channel_decay S4_sla_upgrade S5_combined; do
  for METHOD in M3_dynamic_no_aug M5_constraint_aware; do
    for SEED in 42 43 44; do
      .venv/bin/python -m safe_oran.experiments.train_m3_m5 \
        --method "$METHOD" --scenario "$SCENARIO" --seed "$SEED" --timesteps 300000
    done
  done
done
.venv/bin/python -m safe_oran.experiments.eval_m3_m5
```

Only after the Oracle-z M3/M5 result is stable should Phase2c add real CER/RAG
retrieval and M6 closed-loop experiments.

