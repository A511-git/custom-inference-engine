# Custom Inference Engine: Dual Tesla T4 (TP2) Inference System

High-performance, memory-optimized tensor-parallel (TP=2) inference engine purpose-built to run **Qwen3-27B** models on **2× NVIDIA Tesla T4 GPUs (32 GB total VRAM, Turing SM75)**.

---

## 📁 Repository & Workspace Layout

```
Custom infrence Engine/
├── engine_knowledge_base/       # Authoritative engineering documentation & audit matrices
│   ├── 01_provenance_and_lineage/           # Fork lineage & Change Impact Matrix (CHG-01..23)
│   ├── 02_hardware_and_physics/             # Dual T4 PCIe bandwidth & NUMA characteristics
│   ├── 03_tensor_parallelism_sharding/      # TP2 sharding mathematics & weight layout
│   ├── 04_cuda_kernels_and_sm75_execution/  # SM75 MMA emulation & W8 GEMM Split-K
│   ├── 05_attention_and_fmha_sm75/          # FA75 Head-256 FlashAttention decode kernel
│   ├── 06_memory_and_kv_cache/              # 16GB VRAM budgeting & INT8 paged KV cache
│   ├── 07_communication_and_collectives/    # Pipelined all-reduce & 8-byte argmax collective
│   └── 08_speculation_and_runtime/          # Phase 0 & Phase 1 optimization roadmaps
│
├── ninfer-t4-tp2/               # Production C++/CUDA engine codebase
│   ├── include/ninfer/          # Public API headers (ops, types, engine)
│   ├── src/                     # Core runtime, CUDA kernels, memory materializer, serve
│   ├── apps/cli/                # Interactive terminal generator (ninfer-cli)
│   ├── tests/                   # Unit test suites & qualification harnesses
│   └── deploy/                  # Automated Kaggle 2× T4 bootstrap and benchmark scripts
│
├── cloned_repos/                # Upstream & fork references (local reference only, gitignored)
├── ARCHITECTURAL_AUDIT_SM75_TP2.md          # Comprehensive architectural audit
├── PERFORMANCE_AND_ROOFLINE_MODEL_2XT4.md   # Roofline analysis and memory bandwidth limits
├── DEPLOYMENT_AND_RUNBOOK_KAGGLE_2XT4.md    # Production deployment runbook
└── ChatLookUP.MD                            # Master engineering specifications
```

---

## ⚡ Core Engine Features

1. **Dual T4 Tensor Parallelism (TP=2)**:
   - Symmetrically splits 27B model layers across 2 GPUs (halved weight matrices, column/row parallel projections).
   - Vocab-split LM-head ($124,160$ logits per rank).

2. **Pinned Host Token Embedding (`--embedding-host`)**:
   - Offloads the $248,320 \times 5,120$ token embedding table to pinned host CPU memory (`cudaHostAlloc`).
   - Slashes VRAM consumption by ~1.2 to 2.4 GiB per GPU, unlocking **+70,000 extra KV cache context tokens**.

3. **Local LM-Head Argmax Shortcut**:
   - In greedy decode, replaces 496 KiB `allgather_rows` collective transfers with an **8-byte scalar exchange** (`LocalArgmaxScalar { float, int32 }`).
   - 99.998% bandwidth reduction per token, saving 35–50 µs latency per step.

4. **FA75 Head-256 FlashAttention Decode Kernel**:
   - Tailored for Turing SM75 with $D=256$ head dimension.
   - Strictly keeps shared memory $<48\text{ KiB}$ ($38,440\text{ B}$ static tile).
   - Collaborative 128-bit vectorized loads (`int4` pack) without Ampere `cp.async`.
   - SM75 `mma_s8` Tensor Cores with online FP32 softmax.

5. **2-Stage Pipelined All-Reduce**:
   - Divides row-parallel collective data into two 16-byte aligned chunks.
   - Overlaps Chunk 1 PCIe transfer with Chunk 0 local addition kernel.
   - 100% CUDA Graph capture safe with zero host synchronization.

6. **Hardware Topology & Small-$M$ Autotuning**:
   - Detects bidirectional UVA, asymmetric P2P, or non-P2P mailbox fallbacks at startup.
   - Dynamically calibrates SIMT vs MMA crossover thresholds based on SM count (T4 vs 2080 Ti).

---

## 🚀 Quickstart

### Build the Engine (Fast SM75 Preset)
```bash
cd ninfer-t4-tp2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNINFER_SM75_INT8_KV_ONLY=ON
cmake --build . -j$(nproc)
```

### Run Interactive CLI
```bash
./apps/cli/ninfer-cli \
  --model /path/to/qwen3_6_27b_w8 \
  --tp 2 \
  --embedding-host \
  --prompt "Explain the concept of quantum superposition."
```

### Launch OpenAI-Compatible HTTP Server
```bash
./src/serve/ninfer-server \
  --model /path/to/qwen3_6_27b_w8 \
  --tp 2 \
  --port 8080 \
  --embedding-host
```

### Automated Kaggle 2× T4 Deployment
```bash
cd ninfer-t4-tp2
bash deploy/kaggle_setup.sh   # Full build in <4 minutes
bash deploy/start_server.sh   # Boots server on dual T4
bash deploy/benchmark_tp2.sh  # Evaluates tok/s decode throughput
```
