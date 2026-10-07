# Master Knowledge Base: Custom SM75 TP2 Inference Engine

> **ENGINE ARCHITECTURE & DECISION RECORD FOR AGENTS & ENGINEERS**  
> This directory constitutes the authoritative technical documentation for the custom inference engine running **Qwen3.6-27B** and **Qwen3.8-27B** (`groupwise-int`) across **2× NVIDIA Tesla T4 GPUs (Turing SM75)**.  
> Any future agent session or engineer can obtain exhaustive answers on every architectural choice, kernel selection, memory formula, collective protocol, and bug fix from this knowledge base **without reading raw C++/CUDA code**.

---

## 🗺️ Master Navigation & Topic Index

```
engine_knowledge_base/
├── 01_provenance_and_lineage/         # Source code origins, commit SHAs & fork delta matrix
│   ├── lineage_and_commits.md         # Evolutionary tree from upstream to our tailored engine
│   ├── fork_comparisons.md            # Side-by-side comparison of all 7 repos (taken vs rejected)
│   └── change_impact_and_verification_matrix.md # Full 14-point change impact, risk & verification registry
│
├── 02_hardware_and_physics/           # Physical constraints, measured topology & roofline math
│   ├── turing_sm75_architecture.md    # TU104 die, 40 SMs, Tensor Cores, smem limits, 70W TDP
│   ├── dual_t4_pcie_topology.md       # PHB Host Bridge, 9.91 GB/s P2P, UVA direct transfers
│   └── roofline_and_performance_math.md # Mathematical FLOP & bandwidth roofline derivations
│
├── 03_tensor_parallelism_sharding/    # Sharding theory, layer mappings & tensor geometry
│   ├── tp2_architecture_overview.md   # Intra-layer TP2 vs Pipeline Parallelism theory
│   ├── layer_sharding_matrix.md       # Full layer-by-layer tensor split table (Attention/GDN/MLP)
│   └── q4_q5_shard_geometry.md        # Row-slicing vs K-slicing in group-64 quantized weights
│
├── 04_cuda_kernels_and_sm75_execution/ # Kernel implementations, bug fixes & compile opts
│   ├── w8_gemm_and_splitk.md          # W8 Split-K GEMM, Issue #3 NaN poison bug deep dive
│   ├── mma_fragment_emulation.md      # PTX m16n8k32.s8 / m16n8k16.f16 fragment bug & fix
│   ├── small_t_exact_kernels.md       # Small-T exact shard kernels (T <= 16, +61% decode speed)
│   └── ptxas_compilation_and_templates.md # BF16 template explosion, TU split & INT8-only build flag
│
├── 05_collectives_and_transport/      # Inter-GPU communication subsystems
│   ├── uva_direct_p2p_transport.md    # pull_peer via cudaMemcpyAsync(DeviceToDevice), graph capture
│   └── pinned_host_peer_mailbox.md    # Spinlock fallback transport, slab layout, epoch protocol
│
├── 06_memory_and_kv_cache/            # Memory layout, paged KV pool & budgeting
│   ├── int8_group64_kv_cache.md       # Paged KV cache, 16.9 KiB/tok/GPU, prefill indexing fix
│   └── 16gb_vram_budgeting.md         # Exact byte budgeting for 16GB cards (weights + KV + OS)
│
├── 07_artifact_and_materializer/      # Model weight container & streaming loader
│   ├── ninfer_artifact_container.md   # .ninfer format, direct I/O 4KB alignment fix
│   └── batched_2d_materialization.md  # cudaMemcpy2DAsync strided upload (9.5s -> 3.0s load)
│
└── 08_speculation_and_runtime/        # Multi-Token Prediction & HTTP server
    ├── mtp3_speculative_decoding.md   # MTP verification loop, proposal fix, draft tokens
    ├── prefix_reuse_and_serving.md    # Multi-turn prefix reuse (4.7x TTFT boost), HTTP server
    └── phase1_optimizations_and_roadmap.md # Embedding-host, FA75, small-M GEMV, local argmax
```

---

## ⚡ Quick Reference: Critical Findings & Decisions

| Decision / Issue | Root Cause | Solution / Chosen Architecture | Source Document |
|---|---|---|---|
| **Why TP2 instead of Pipeline Parallelism?** | Pipeline bubbles waste 50% of GPU compute time; VRAM footprint remains uneven. | Intra-layer Megatron TP2: both GPUs compute every layer simultaneously, halving weight & KV cache memory per GPU. | [TP2 Architecture Overview](03_tensor_parallelism_sharding/tp2_architecture_overview.md) |
| **Poison NaN Bug (Issue #3)** | $T \in [161, 192]$ launched truncated `<160>` kernel, leaving 32 columns unwritten. | Routed $T \in [161, 192]$ to `ConcatMmaR32C64` under `#if defined(NINFER_SM75)`. | [W8 GEMM & Split-K](04_cuda_kernels_and_sm75_execution/w8_gemm_and_splitk.md) |
| **Corrupted INT8 MMA Decomposition** | Turing emulated `m16n8k32.s8` paired registers as $(M_0, K_0)/(M_0, K_1)$, mixing row halves. | Corrected fragment decomposition in `mma.cuh` to $(a_0, a_2)$ and $(a_1, a_3)$. | [MMA Fragment Emulation](04_cuda_kernels_and_sm75_execution/mma_fragment_emulation.md) |
| **Decode Speed Jump (10.4 $\to$ 27.5 tok/s)** | Column-sharded Q4/Q5 projections fell back to prefill-sized grouped-MMA kernels (2.1 ms). | Implemented dedicated small-T exact kernels ($T \le 16$) for 3584/2048 shard extents. | [Small-T Exact Kernels](04_cuda_kernels_and_sm75_execution/small_t_exact_kernels.md) |
| **PTXAS 6.5-Hour Compilation Hang** | 96 emulated BF16 decode templates created an 8.16M line PTX file. | Split TUs (`gqa_attention_decode_tp2.cu`) + added `-DNINFER_SM75_INT8_KV_ONLY=ON`. | [PTXAS Compilation & Templates](04_cuda_kernels_and_sm75_execution/ptxas_compilation_and_templates.md) |
| **Primary Collective Transport** | Direct PCIe 3.0 Host Bridge P2P measured at 9.91 GB/s and 2.5–7 µs latency. | UVA direct `pull_peer` via destination-stream `cudaMemcpyAsync(DeviceToDevice)`. | [UVA Direct P2P Transport](05_collectives_and_transport/uva_direct_p2p_transport.md) |
| **Fallback Collective Transport** | Systems without P2P suffer 277 µs driver-event staging latency. | Valerio's Pinned-Host `PeerMailbox` spinlock transport (41 µs latency). | [Pinned-Host PeerMailbox](05_collectives_and_transport/pinned_host_peer_mailbox.md) |
| **Model Weight Load (9.5s $\to$ 3.0s)** | 4,000,000 tiny scalar ~4 KiB row copies overloaded the driver. | Batched run detection uploading strided column shards via `cudaMemcpy2DAsync`. | [Batched 2D Materialization](07_artifact_and_materializer/batched_2d_materialization.md) |
| **Direct I/O 4KB Read Abort** | Small pinned allocations packed by driver lacked 4096-byte alignment. | Padded `Slot` buffer with `kPayloadAlignment` (4096 B) and aligned pointer. | [NInfer Artifact Container](07_artifact_and_materializer/ninfer_artifact_container.md) |
| **Multi-Turn Chat Speedup (4.7x TTFT)** | TP2 previously reset prefix cache on every continuation turn. | Mirrored `rewrite_checkpoint_hidden` across ranks to restore from retained hidden state. | [Prefix Reuse & Serving](08_speculation_and_runtime/prefix_reuse_and_serving.md) |
| **Comprehensive Change & Risk Audit** | Tracking base $\to$ additions $\to$ cascading breaking risks & verification proofs. | Formal 14-point impact matrix with 6-step RFC checklist and living registry. | [Change Impact & Verification Matrix](01_provenance_and_lineage/change_impact_and_verification_matrix.md) |
| **Phase 1 Performance & Memory Roadmap** | Freeing 1.2+ GiB VRAM per card & slashing decode PCIe overhead by 99.99%. | `--embedding-host`, Local LM-Head Argmax, Small-M GEMV, and FA75 attention. | [Phase 1 Optimizations & Roadmap](08_speculation_and_runtime/phase1_optimizations_and_roadmap.md) |
