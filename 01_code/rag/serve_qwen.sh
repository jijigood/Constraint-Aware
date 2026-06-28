#!/usr/bin/env bash
# Launch Qwen3-14B (vLLM, OpenAI-compatible) on GPU0 for the Phase-2a constraint gate.
# Reuses ~/track_a_deep_researcher/serve_track_a_llm.sh (which backgrounds vLLM + prints the endpoint).
# Poll readiness afterwards with: curl -s -H "Authorization: Bearer $KEY" .../v1/models | grep qwen
set -uo pipefail
LOGD=/home/huangxiaolin/safe_drl_oran/logs; mkdir -p "$LOGD"
source /home/huangxiaolin/track_a_deep_researcher/track_a_env.sh
PORT="${TRACK_A_LLM_PORT:-8001}"
KEY="$(cat /home/huangxiaolin/track_a_deep_researcher/llm_endpoint.key 2>/dev/null || echo EMPTY)"
if curl -s -H "Authorization: Bearer $KEY" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q qwen; then
  echo "qwen already serving on :${PORT}"; exit 0
fi
TRACK_A_LLM_GPU=0 TRACK_A_LLM_GPU_UTIL=0.6 bash /home/huangxiaolin/track_a_deep_researcher/serve_track_a_llm.sh
echo "launch invoked on GPU0; model load ~1-2 min; poll /v1/models for 'qwen'."
