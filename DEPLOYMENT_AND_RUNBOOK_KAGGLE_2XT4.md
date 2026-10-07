# Deployment & Runbook: Dual Tesla T4 (SM75) TP2 Inference on Kaggle

This guide provides end-to-end, copy-paste executable commands for building and running **Qwen3.6-27B (or Qwen3.8-27B) `groupwise-int` on 2× NVIDIA Tesla T4 (16GB) GPUs** in a Kaggle Linux environment.

---

## 1. Environment & Architecture Summary

- **Target Hardware**: 2× NVIDIA Tesla T4 (TU104 / `sm_75`, 16 GB VRAM each).
- **PCIe Topology**: PCIe 3.0 x16 via Host Bridge (PHB). Peer-to-peer (P2P) access is **supported and enabled** (`cudaDeviceCanAccessPeer(0,1) == 1`).
- **P2P Bandwidth**: Measured at **~9.18 GB/s** bidirectional.
- **Model**: Qwen3.6-27B / Qwen3.8-27B `groupwise-int` (W8A16, Q4/Q5 projections, Q6 vocab).
  - Single-GPU size: 16.29 GiB (does not fit on one 16GB T4).
  - **TP2 size per GPU**: ~8.15 GiB weights + ~3-4 GiB INT8 KV pool = **~11.5 - 12.5 GiB** resident per card (comfortably fits in 16 GB!).
- **KV Cache**: INT8 group-64 paged KV cache (`--kv-dtype int8`).
- **Speculative Decoding**: Greedy MTP0 baseline (`--spec mtp --draft-tokens 0` or omit `--spec`), followed by MTP3 (`--spec mtp --draft-tokens 3 --lm-head-draft`).

---

## 2. Kaggle Notebook One-Click Setup Script

Run this block in a Kaggle Linux notebook cell with the `2x T4` accelerator selected:

```bash
%%bash
set -e

echo "=== Step 1: Install Build Dependencies ==="
apt-get update -qq && apt-get install -y -qq cmake ninja-build aria2 git

echo "=== Step 2: Verify GPU and CUDA ==="
nvidia-smi
nvcc --version

echo "=== Step 3: Check /tmp Storage Space ==="
df -h /tmp
```

---

## 3. Clone Repository & Apply Patches

In Kaggle, clone the primary base tree (`ninfer-2080ti-22g-tp2`) or push your local directory:

```bash
%%bash
set -e
cd /tmp
if [ ! -d "ninfer" ]; then
  git clone https://github.com/zsq13767593046-bit/ninfer-2080ti-22g-tp2.git ninfer
fi
cd ninfer

# Ensure SM75 poison NaN fix is present in src/ops/linear_pair/w8/w8_pair_plan.cpp
# (Already applied in our workspace base)
```

---

## 4. Run Hardware & Collective Transport Probes

Before running any models, compile and run the standalone transport probes to verify PCIe P2P latency and throughput between GPU 0 and GPU 1:

```bash
%%bash
cd /tmp/ninfer

echo "=== Compiling P2P Probe ==="
nvcc -arch=sm_75 -O2 tools/tp2/p2p_probe.cu -o /tmp/p2p_probe
/tmp/p2p_probe 0 1

echo "=== Compiling Transport Probe ==="
nvcc -arch=sm_75 -O2 tools/tp2/transport_probe.cu -o /tmp/transport_probe
/tmp/transport_probe 0 1

echo "=== Compiling Mailbox Probe (Fallback Check) ==="
nvcc -arch=sm_75 -O2 tools/tp2/mailbox_probe.cu -o /tmp/mailbox_probe
/tmp/mailbox_probe 0 1
```

*Expected output*:
- `p2p_probe`: Reports `canAccessPeer(0, 1) = 1` and `canAccessPeer(1, 0) = 1`. Bandwidth ~9.18 GB/s.
- `transport_probe`: 10 KiB reduction latency ~2–7 µs over direct UVA P2P.
- `mailbox_probe`: ~40–50 µs over pinned-host memory (validating fallback).

---

## 5. Build NInfer Engine for SM75

Compile with Ninja using `CMAKE_CUDA_ARCHITECTURES=75`. You can optionally pass `-DNINFER_SM75_INT8_KV_ONLY=ON` to eliminate the multi-hour BF16 decode ptxas compilation step:

```bash
%%bash
cd /tmp/ninfer

mkdir -p build
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_TESTS=OFF \
  -DNINFER_SM75_INT8_KV_ONLY=ON

ninja -C build -j$(nproc)
```

---

## 6. Download Model Artifact to Fast Local Storage (`/tmp`)

Download the quantized model artifact (`qwen3_6_27b.ninfer` or `qwen3_8_27b.ninfer`) directly to `/tmp` (RAM-backed/NVMe, ~100GB available on Kaggle):

```bash
%%bash
mkdir -p /tmp/models
cd /tmp/models

# Using aria2c with 16 connections for maximum download speed
aria2c -x 16 -s 16 -k 1M \
  "https://huggingface.co/mr-september/Qwen3.6-27B-NInfer/resolve/main/qwen3_6_27b.ninfer" \
  -o qwen3_6_27b.ninfer
```

---

## 7. Execution Runbook

### A. Smoke Test via CLI (Single Prompt, Greedy MTP0)

```bash
%%bash
/tmp/ninfer/build/apps/ninfer /tmp/models/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --kv-dtype int8 --kv-capacity auto \
  --max-context 8192 \
  --prompt "Explain the difference between process and thread in three sentences." \
  --max-new 128 \
  --sampling greedy
```

### B. High-Performance OpenAI/Anthropic Server (MTP3 Speculative Decoding)

Launch the HTTP server listening on port 8080:

```bash
%%bash
/tmp/ninfer/build/apps/ninfer-serve /tmp/models/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --port 8080 \
  --kv-dtype int8 --kv-capacity auto \
  --max-context 32768 \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --reasoning-effort medium
```

### C. Client Curl Request

In a separate terminal or Python cell:

```python
import requests
import json

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "qwen3_6_27b",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.0,
        "max_tokens": 64
    }
)
print(response.json())
```

---

## 8. Troubleshooting & Common Pitfalls

| Symptom | Root Cause | Solution |
|---|---|---|
| `unaligned or oversized direct read` | Direct I/O staging slot was not 4096-byte aligned. | Verified fixed in `materializer.cpp` with `kPayloadAlignment`. |
| Output contains poison NaNs after token 160 | Schedule `DualSplitKMediumC192` ran truncated `<160>` kernel on SM75. | Verified fixed in `w8_pair_plan.cpp`: routes to `ConcatMmaR32C64`. |
| `ptxas` hangs for hours during build | 96 emulated BF16 decode templates overloading ptxas on Turing. | Build with `-DNINFER_SM75_INT8_KV_ONLY=ON` or compile `gqa_attention_decode_tp2.cu` separately. |
| Memory allocation failure at startup | Context was sized too large for 16GB cards. | Use `--kv-dtype int8 --kv-capacity auto --max-context 32768` (or 65536). |
| Multi-turn chat re-prefills entire history | MTP prefix reuse reset at TP2. | Apply Dmitriy Ivanov's checkpoint mirroring patch from `fix/tp2-mtp-prefix-reuse`. |
