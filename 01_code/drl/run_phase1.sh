#!/usr/bin/env bash
# Phase 1 orchestrator: smoke-gate -> train grid (bounded concurrency) -> eval/gate/figures.
# Usage: bash run_phase1.sh [--quick]
set -uo pipefail

PROJ=/home/huangxiaolin/safe_drl_oran
PY="$PROJ/.venv/bin/python"
TRAIN="$PROJ/01_code/drl/train_baselines.py"
EVAL="$PROJ/01_code/drl/eval_baselines.py"
SMOKE="$PROJ/01_code/smoke_test.py"
cd "$PROJ" || exit 1
mkdir -p logs

QUICK=""; TS=300000; SEEDS=(42 43 44)
if [[ "${1:-}" == "--quick" ]]; then QUICK="--quick"; TS=30000; SEEDS=(42); fi
ALGOS=(ppo dqn); REGIMES=(high_embb high_urllc); SHIELDS=(none static oracle_margin)
MAXJOBS=${MAXJOBS:-5}
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""

echo "=== [$(date -u +%H:%M:%S)] Phase 1 (quick='${QUICK}' ts=$TS seeds=${SEEDS[*]} maxjobs=$MAXJOBS) ==="

echo "=== smoke test (gate) ==="
"$PY" "$SMOKE" > logs/smoke.log 2>&1
if ! grep -q "SMOKE TEST PASSED" logs/smoke.log; then
  echo "SMOKE TEST FAILED -- aborting (see logs/smoke.log)"; tail -20 logs/smoke.log; exit 1
fi
echo "smoke OK"

echo "=== training grid ==="
for algo in "${ALGOS[@]}"; do for regime in "${REGIMES[@]}"; do
  for shield in "${SHIELDS[@]}"; do for seed in "${SEEDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
    tag="${algo}_${regime}_${shield}_s${seed}"
    ( "$PY" "$TRAIN" --algo "$algo" --regime "$regime" --shield "$shield" --seed "$seed" \
        --timesteps "$TS" $QUICK > "logs/train_${tag}.log" 2>&1 \
      && echo "  done $tag" || echo "  FAIL $tag (see logs/train_${tag}.log)" ) &
  done; done
done; done
wait
echo "=== training grid complete [$(date -u +%H:%M:%S)] ==="

echo "=== eval + gate + figures ==="
"$PY" "$EVAL" 2>&1 | tee logs/eval.log
echo "=== Phase 1 complete [$(date -u +%H:%M:%S)] ==="
