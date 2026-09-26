#!/bin/bash
# Serve one local model with vLLM for the runs in this repository.
#
#   MODEL_DIR=/path/to/Qwen3-8B PORT=8000 ./scripts/serve_local.sh
#
# The chat template forces the hybrid-reasoning models used here into their non-thinking mode, so every arm is
# measured on the same generation behaviour. GPU_UTIL is a fraction of the WHOLE card and must fit the memory
# free at launch; lower it when the card is shared.
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
