#!/usr/bin/env bash
# ==============================================================================
# Kaggle 2x Tesla T4 One-Click Setup & Build Script
# Tailored for NInfer SM75 TP2 (Qwen3.6-27B groupwise-int)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ENGINE_DIR}/build"
MODELS_DIR="/tmp/models"
MODEL_FILE="${MODELS_DIR}/qwen3_6_27b.ninfer"
MODEL_URL="https://huggingface.co/mr-september/Qwen3.6-27B-NInfer/resolve/main/qwen3_6_27b.ninfer"

echo "===================================================================="
echo " [Step 1/5] Verifying System Environment & GPUs"
echo "===================================================================="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "CUDA Compiler: $(nvcc --version | grep release)"
echo "Available space in /tmp: $(df -h /tmp | awk 'NR==2 {print $4}')"

echo "===================================================================="
echo " [Step 2/5] Installing Build Tooling (Ninja, CMake, Aria2)"
echo "===================================================================="
if ! command -v ninja &>/dev/null || ! command -v aria2c &>/dev/null || ! pkg-config --exists libavformat 2>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq cmake ninja-build aria2 pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
fi

echo "===================================================================="
echo " [Step 3/5] Compiling and Executing Transport Probes"
echo "===================================================================="
mkdir -p /tmp/probes
echo "Compiling P2P Probe..."
nvcc -arch=sm_75 -O2 "${ENGINE_DIR}/tools/tp2/p2p_probe.cu" -o /tmp/probes/p2p_probe || true
if [ -f /tmp/probes/p2p_probe ]; then /tmp/probes/p2p_probe 0 1 || true; fi

echo "Compiling Collective Transport Probe..."
nvcc -arch=sm_75 -O2 "${ENGINE_DIR}/tools/tp2/transport_probe.cu" -o /tmp/probes/transport_probe || true
if [ -f /tmp/probes/transport_probe ]; then /tmp/probes/transport_probe 0 1 || true; fi

echo "Compiling Mailbox Fallback Probe..."
nvcc -arch=sm_75 -O2 "${ENGINE_DIR}/tools/tp2/mailbox_probe.cu" -o /tmp/probes/mailbox_probe || true
if [ -f /tmp/probes/mailbox_probe ]; then /tmp/probes/mailbox_probe 0 1 || true; fi

echo "===================================================================="
echo " [Step 4/5] Building NInfer Engine (Target: sm_75, INT8 KV Only)"
echo "===================================================================="
cmake -B "${BUILD_DIR}" -S "${ENGINE_DIR}" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DNINFER_BUILD_APPS=ON \
    -DNINFER_BUILD_TESTS=OFF \
    -DNINFER_SM75_INT8_KV_ONLY=ON

ninja -C "${BUILD_DIR}" -j"$(nproc)"

echo "Engine built successfully: ${BUILD_DIR}/apps/ninfer-serve"

echo "===================================================================="
echo " [Step 5/5] Downloading Model Artifact to /tmp"
echo "===================================================================="
mkdir -p "${MODELS_DIR}"
if [ ! -f "${MODEL_FILE}" ]; then
    echo "Downloading Qwen3.6-27B artifact (~16.29 GiB) via aria2c..."
    aria2c -x 16 -s 16 -k 1M -c "${MODEL_URL}" -d "${MODELS_DIR}" -o "qwen3_6_27b.ninfer"
else
    echo "Model artifact already present at ${MODEL_FILE} ($(du -h "${MODEL_FILE}" | cut -f1))"
fi

echo "===================================================================="
echo " SETUP COMPLETE! To start serving, run:"
echo "   ${SCRIPT_DIR}/start_server.sh"
echo "===================================================================="
