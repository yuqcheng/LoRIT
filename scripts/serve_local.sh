#!/bin/bash
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR to the local model directory}"
cd "$(dirname "$0")"
exec vllm serve "$MODEL_DIR" \
    --served-model-name "${SERVED_NAME:-local-model}" \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --api-key EMPTY \
    ${NOTHINK:+--chat-template "$PWD/nothink.jinja"} \
    --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
    --max-model-len "${MAX_LEN:-32768}" \
    --seed "${SEED:-2025}"
