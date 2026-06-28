#!/usr/bin/env bash
# Phase 2a-v2 model-size sweep: for each model, (re)serve on GPU0 and run the SAME tagged gate.
# Reuses the dumped states + scorer; only the producer model changes. Server left stopped at the end.
set -uo pipefail

PROJ=/home/huangxiaolin/safe_drl_oran
PY=~/dify_vllm_uv310/bin/python
SERVE=/home/huangxiaolin/track_a_deep_researcher/serve_track_a_llm.sh
cd "$PROJ" || exit 1
mkdir -p logs
source /home/huangxiaolin/track_a_deep_researcher/track_a_env.sh
KEY="$(cat /home/huangxiaolin/track_a_deep_researcher/llm_endpoint.key 2>/dev/null || echo EMPTY)"
export TRACK_A_LLM_MAXLEN=4096   # prompts ~2.5k tok + 400 out; smaller KV, fits 32B comfortably

# "path tag util"
MODELS=(
  "/home/huangxiaolin/models/Qwen3-1.7B 1p7b 0.40"
  "/home/huangxiaolin/models/Qwen3-4B   4b   0.40"
  "/home/huangxiaolin/models/Qwen3-14B  14b  0.60"
  "/home/huangxiaolin/models/Qwen3-32B  32b  0.92"
)

stop_8001() {
  local pid
  pid=$(ss -ltnp 2>/dev/null | grep ':8001 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  if [ -n "${pid:-}" ]; then
    echo "  stopping vLLM pid=$pid on :8001"; kill "$pid" 2>/dev/null
    for _ in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':8001 ' || break; sleep 1; done
  fi
}

wait_ready() {
  for _ in $(seq 1 90); do
    curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q '"id":"qwen"' && return 0
    sleep 5
  done
  return 1
}

for entry in "${MODELS[@]}"; do
  set -- $entry; MPATH="$1"; TAG="$2"; UTIL="$3"
  echo "=== [$(date -u +%H:%M:%S)] model=$TAG path=$MPATH util=$UTIL ==="
  stop_8001
  TRACK_A_LLM_MODEL="$MPATH" TRACK_A_LLM_GPU=0 TRACK_A_LLM_GPU_UTIL="$UTIL" bash "$SERVE" \
    > "logs/serve_${TAG}.log" 2>&1
  if ! wait_ready; then echo "  SERVE TIMEOUT for $TAG"; tail -25 "logs/serve_${TAG}.log"; continue; fi
  echo "  served $TAG; running gate ..."
  "$PY" "$PROJ/01_code/rag/run_gate.py" --tag "$TAG" > "logs/gate_${TAG}.log" 2>&1
  rc=$?
  echo "  gate rc=$rc $(date -u +%H:%M:%S)"
  grep -A3 "=== GATE ===" "logs/gate_${TAG}.log" 2>/dev/null | tail -8
done

stop_8001
echo "=== sweep complete [$(date -u +%H:%M:%S)]; server stopped (GPU0 freed) ==="
