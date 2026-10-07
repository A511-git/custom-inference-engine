#!/usr/bin/env bash
# ==============================================================================
# Launch NInfer Serve for 2x Tesla T4 (SM75 TP2)
# Qwen3.6-27B groupwise-int + MTP3 Speculative Decoding + INT8 KV Cache
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ENGINE_DIR}/build"
SERVER_BIN="${BUILD_DIR}/apps/ninfer-serve"
MODEL_FILE="/tmp/models/qwen3_6_27b.ninfer"

PORT="${PORT:-8080}"
MAX_CONTEXT="${MAX_CONTEXT:-32768}"

if [ ! -f "${SERVER_BIN}" ]; then
    echo "Error: Server binary not found at ${SERVER_BIN}."
    echo "Please run ${SCRIPT_DIR}/kaggle_setup.sh first."
    exit 1
fi

if [ ! -f "${MODEL_FILE}" ]; then
    echo "Error: Model artifact not found at ${MODEL_FILE}."
    echo "Please download the artifact first via ${SCRIPT_DIR}/kaggle_setup.sh."
    exit 1
fi

echo "===================================================================="
echo " Starting NInfer Server on port ${PORT}..."
echo " Config: 2x Tesla T4, TP2, INT8 KV, MTP3 speculative decoding"
echo " Context Ceiling: ${MAX_CONTEXT} tokens"
echo "===================================================================="

exec "${SERVER_BIN}" "${MODEL_FILE}" \
    --tp 2 --devices 0,1 \
    --port "${PORT}" \
    --kv-dtype int8 \
    --kv-capacity auto \
    --max-context "${MAX_CONTEXT}" \
    --max-concurrency 1 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --reasoning-effort medium
