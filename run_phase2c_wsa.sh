#!/usr/bin/env bash
# WS-A orchestration: for each Qwen3 size, serve vLLM, run the real-LLM CER
# compiler eval for both retrievers (tfidf, bge), then stop the server.
# Generations are cached, so re-runs are free. Run from the repo root.
set -uo pipefail

REPO=/home/huangxiaolin/safe_drl_oran
SERVE=/home/huangxiaolin/track_a_deep_researcher/serve_track_a_llm.sh
ENVSH=/home/huangxiaolin/track_a_deep_researcher/track_a_env.sh
PY=/home/huangxiaolin/dify_vllm_uv310/bin/python
PORT="${TRACK_A_LLM_PORT:-8001}"
KEY_FILE=/home/huangxiaolin/track_a_deep_researcher/llm_endpoint.key

# GPU precheck: pick the freest GPU unless TRACK_A_LLM_GPU is set.
if [[ -z "${TRACK_A_LLM_GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  export TRACK_A_LLM_GPU="${GPU:-1}"
fi
echo "Using GPU ${TRACK_A_LLM_GPU}"

source "$ENVSH"
cd "$REPO"
export PYTHONPATH="$REPO"

# size label : model path : gpu util
SIZES=(
  "Qwen3-1.7B:/home/huangxiaolin/models/Qwen3-1.7B:0.40"
  "Qwen3-4B:/home/huangxiaolin/models/Qwen3-4B:0.45"
  "Qwen3-14B:/home/huangxiaolin/models/Qwen3-14B:0.60"
  "Qwen3-32B:/home/huangxiaolin/models/Qwen3-32B:0.92"
)
# Override with: SIZES_FILTER="Qwen3-14B" ./run_phase2c_wsa.sh
FILTER="${SIZES_FILTER:-}"

wait_ready() {
  local key; key="$(cat "$KEY_FILE")"
  for _ in $(seq 1 180); do
    curl -s -H "Authorization: Bearer $key" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
      | grep -q '"id":"qwen"' && return 0
    sleep 5
  done
  return 1
}

stop_vllm() {
  # Avoid `pkill -f ...` here: when this script is launched through `bash -lc`,
  # the pattern can match the current shell command and kill itself. Stop by port
  # instead, which is the actual resource the next vLLM server needs.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":${PORT} " || return 0
    sleep 2
  done
}

for entry in "${SIZES[@]}"; do
  IFS=':' read -r LABEL MODEL UTIL <<< "$entry"
  [[ -n "$FILTER" && "$FILTER" != "$LABEL" ]] && continue
  echo "==================== $LABEL ===================="
  stop_vllm
  TRACK_A_LLM_MODEL="$MODEL" TRACK_A_LLM_GPU_UTIL="$UTIL" bash "$SERVE" || { echo "serve failed"; continue; }
  if ! wait_ready; then echo "[$LABEL] vLLM not ready; skipping"; stop_vllm; continue; fi
  for RETR in tfidf bge; do
    echo "---- $LABEL / $RETR ----"
    "$PY" -m safe_oran.experiments.run_phase2c_wsa --model qwen --tag "$LABEL" --retriever "$RETR" \
      || echo "[$LABEL/$RETR] eval failed"
  done
  stop_vllm
done
echo "WS-A sweep done."
