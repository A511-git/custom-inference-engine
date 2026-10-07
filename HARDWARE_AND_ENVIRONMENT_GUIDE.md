# Hardware and Deployment Guide: Kaggle 2× Tesla T4

**Target System:** Kaggle Notebook Container  
**GPUs:** 2× NVIDIA Tesla T4 (Turing TU104, Compute Capability 7.5, 15,360 MiB VRAM each)  
**Host:** 2 physical cores / 4 logical threads (Intel Xeon @ 2.0 GHz), 31.35 GiB System RAM, 0 swap  
**CUDA Toolkit:** CUDA 12.8 / 12.9 / 13.x  
**Storage:** `/tmp` (Overlayfs, practical observed capacity ~100 GB usable)  

---

## 1. Machine Identification & Context

- **Local Machine (Windows Workstation):**
  - Used for code development, repository maintenance, diff analysis, and staging patches.
  - Does not have the physical Tesla T4 cards.
- **Remote Execution Machine (Kaggle Linux Environment):**
  - Contains the 2× Tesla T4 hardware.
  - Recorded in [`kaggle_2xT4_complete_hardware_report.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/kaggle_2xT4_complete_hardware_report.md) and [`Machine.MD`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/Machine.MD).

---

## 2. Hardware Topology & Transfer Baselines

| Property | Measured Value | Notes |
| :--- | :--- | :--- |
| **P2P Accessibility** | `cudaDeviceCanAccessPeer(0,1) == 1`<br>`cudaDeviceCanAccessPeer(1,0) == 1` | Peer access is physically granted |
| **Topology** | **PHB** (PCIe Host Bridge) | No NVLink bridge |
| **GPU $\leftrightarrow$ GPU P2P Bandwidth** | **~9.18 – 9.91 GB/s** | PCIe Gen 3 ×16 |
| **Host-to-Device (H2D) Bandwidth** | **~11.43 GiB/s** | Direct pinned transfers |
| **Device-to-Host (D2H) Bandwidth** | **~12.18 GiB/s** | Direct pinned transfers |

### Transport Strategy:
Because peer access is granted over the host bridge, **stream-capturable UVA device-to-device `cudaMemcpyAsync`** is operational. However, to guard against driver synchronization hiccups or when running without P2P, the **Pinned-Host PeerMailbox** provides a GPU-side polling fallback.

---

## 3. Storage Hierarchy on Kaggle

| Path | Filesystem | Capacity | Usage Rule |
| :--- | :--- | :--- | :--- |
| **`/tmp`** | Overlayfs | **~100 GB practical** | **Primary model storage location.** Place `.ninfer` artifacts here. |
| **`/dev/shm`** | tmpfs (RAM) | 14 GB | Used for shared memory / IPC. Do NOT store model files here. |
| **`/kaggle/working`** | ext4 | ~20 GB | For build outputs, logs, binaries. Too small for large weights. |
| **`/kaggle/input`** | ext4 / NFS | ~20 GB | Read-only input mount. |

---

## 4. Build Instructions for Kaggle (Linux)

### Step 1: Install Build Dependencies
```bash
apt-get update && apt-get install -y cmake ninja-build libcurl4-openssl-dev pkg-config
```

### Step 2: Configure & Compile for SM75
To avoid the multi-hour ptxas compile hang from software-emulated BF16 decode attention, compile with `NINFER_SM75_INT8_KV_ONLY`:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DNINFER_SM75_INT8_KV_ONLY=ON

cmake --build build --parallel $(nproc)
```

Target binaries generated:
- `build/apps/ninfer`: CLI engine runner
- `build/apps/ninfer-serve`: OpenAI & Anthropic compatible HTTP server
- `build/tools/tp2/transport_probe`: Transport & P2P verification probe
- `build/tools/tp2/mailbox_probe`: Pinned-host mailbox probe

---

## 5. Artifact Setup & Execution Commands

### Step 1: Download Model Artifact to `/tmp`
```bash
pip install huggingface-hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='neroued/Qwen3.6-27B-NInfer', filename='qwen3_6_27b.ninfer', local_dir='/tmp')
"
```

### Step 2: Run Transport Probe
Verify P2P and mailbox communication latency:
```bash
./build/tools/tp2/transport_probe
./build/tools/tp2/mailbox_probe
```

### Step 3: Run TP2 Inference (Greedy MTP0 Baseline)
```bash
./build/apps/ninfer /tmp/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 8192 \
  --max-new 256 \
  --kv-dtype int8 \
  --prompt "Explain quantum decoherence in three clear sentences."
```

### Step 4: Run TP2 Inference with Speculative MTP3
```bash
./build/apps/ninfer /tmp/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 32768 \
  --max-new 256 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --prompt "Write a Python script implementing a thread-safe LRU cache."
```

### Step 5: Start HTTP Server for 2× T4
```bash
./build/apps/ninfer-serve /tmp/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --host 0.0.0.0 --port 8080 \
  --max-context 65536 \
  --kv-dtype int8 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```
