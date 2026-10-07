# Lineage & Commit Provenance

> **Location**: `engine_knowledge_base/01_provenance_and_lineage/lineage_and_commits.md`  
> **Related Documents**: [Fork Comparisons](fork_comparisons.md) | [Change Impact & Verification Matrix](change_impact_and_verification_matrix.md) | [TP2 Architecture Overview](../03_tensor_parallelism_sharding/tp2_architecture_overview.md) | [W8 GEMM & Split-K](../04_cuda_kernels_and_sm75_execution/w8_gemm_and_splitk.md)

This document details the exact evolutionary lineage of our custom engine, documenting the origin repositories, commit hashes, authors, and functional contributions of every constituent technology.

---

## 1. The Evolutionary Tree

```
Neroued/ninfer (Upstream Base, Blackwell sm_120a focus)
  │
  ├──> mr-september/ninfer-2080ti-22g (Single-GPU SM75 Turing Port)
  │      └── Custom W8 GEMM, Split-K, GDN MmaUnsplit (T >= 9)
  │
  ├──> wamansou/ninfer-tp2-1m / giocom (Original TP2 Design & YaRN 1M)
  │      └── 2-GPU Column/Row-parallel sharding, UVA pull_peer allreduce
  │
  ├──> ValerioDolci/ninfer-tp2 (Mature Production TP2 Runtime)
  │      └── Pinned-Host PeerMailbox, CUDA Graph decode, concurrent serving
  │
  ├──> ivanov84/ninfer-windows-tp2
  │      └── MTP prefix reuse fix (eab0a929), ScopedDevice fix (2cf050bf)
  │
  ├──> zsq13767593046-bit/ninfer-2080ti-22g-tp2 (The SM75 + TP2 Convergence)
  │      └── Combined SM75 Turing kernels with Valerio TP2 framework
  │      └── fe4590c0: Small-T exact kernels for Q4/Q5 shards (+61% decode speed)
  │      └── d3b079cf: Batched 2D materializer uploads (9.5s -> 3.0s load)
  │      └── 10af75b4: Corrected SM75 mma_s8/mma_f16 fragment decomposition
  │      └── 3c9c7caa: INT8 decode dynamic smem tail & prefill page offset fix
  │
  └──> ninfer-t4-tp2 (OUR CUSTOM ENGINE - Hardened for 2x Tesla T4)
         └── Fixed Issue #3 poison NaN bug (w8_pair_plan.cpp)
         └── Fixed direct-I/O 4KB alignment in materializer.cpp
         └── Added NINFER_SM75_INT8_KV_ONLY fast-build CMake preset (<4 min)
         └── Integrated PeerMailbox fallback transport
         └── Automated Kaggle deployment kit (deploy/*.sh)
```

---

## 2. Key Commits & Historical Provenance

### A. The Turing SM75 Kernel Port (`mr-september`)
- **Repository**: `cloned_repos/ninfer-2080ti-22g`
- **Key Commit `4d67c7d7`**: `feat(arch): port NInfer to NVIDIA Turing sm_75`
  - Replaced Blackwell-specific TMA (Tensor Memory Accelerator) and native FP8/NVFP4 instructions with Turing-compatible W8 Split-K GEMM.
  - Introduced Turing register and shared-memory bounds.
- **Key Commit `b53e21c2`**: `fix(sm75): route all T >= 9 GDN gating to MmaUnsplit on Turing`
  - Turing has cooperative-launch and shared-memory limits that cause CTA occupancy failures during wide token counts. Slicing gating projections to `MmaUnsplit` resolved cooperative launch limits.

### B. The Tensor-Parallel Execution Framework (`ValerioDolci`)
- **Repository**: `cloned_repos/ninfer-tp2`
- **Key Architecture**:
  - Implemented one-process, two-device execution with rank-local state (`ExecutionContext`, `DeviceSelection`).
  - Implemented stream-capturable UVA allreduce and allgather in `src/ops/common/allreduce.cu`.
  - Added Pinned-Host `PeerMailbox` in `src/ops/common/peer_mailbox.cu` for zero-event cross-device reductions on consumer PCIe buses.
- **Key Commit `795f590b`**: `artifact: align the direct-I/O staging slots to the payload alignment`
  - Aligned direct-read buffers to 4096 bytes (`kPayloadAlignment`) to prevent OS direct-read aborts.

### C. The SM75 + TP2 Convergence (`zsq13767593046-bit`)
- **Repository**: `cloned_repos/ninfer-2080ti-22g-tp2`
- **Key Commit `10af75b4`**: `fix(ops): correct SM75 mma_s8 and mma_f16 Turing fragment decomposition`
  - Corrected Turing emulated MMA register pairings. Upstream `mr-september` accumulated $(M_0, K_0)$ and $(M_0, K_1)$ into accumulator row 0, corrupting all INT8 KV attention. Fixed via exact integer oracle test (`test_mma_sm75.cu`).
- **Key Commit `3c9c7caa`**: `fix(ops): fit SM75 INT8 decode shared memory and prefill page offsets`
  - Staged page IDs at dynamic shared memory tail, preventing 328-byte static smem overflow past 48 KiB ceiling.
  - Fixed prefill KV indexing bug where tile 2 read tile 1's bytes (`page_offset + key_l`).
- **Key Commit `fe4590c0`**: `perf(ops): route the TP2 Q4/Q5 shards through small-T exact kernels`
  - Column shards previously dispatched to generic grouped-MMA kernels (2.1 ms). Implemented exact small-T kernels for $T \le 16$, lifting decode speed from 10.4 to 27.5 tok/s on Turing hardware.
- **Key Commit `d3b079cf`**: `perf(artifact): batch strided shard uploads into 2D copies`
  - Grouped ~4,000,000 scalar row uploads into batched `cudaMemcpy2DAsync` calls, reducing two-device load time from 9.5s to 3.0s.

### D. Multi-Turn Prefix Reuse Fix (`ivanov84`)
- **Repository**: `cloned_repos/ninfer-windows-tp2`
- **Branch**: `fix/tp2-mtp-prefix-reuse`
- **Key Commit `eab0a929`**: `fix(tp2): enable MTP prefix reuse via per-rank retained hidden state`
  - Mirrored `rewrite_checkpoint_hidden` across ranks. On conversation turns, the engine restores from resident KV cache rather than re-prefilling history, cutting TTFT by 4.7x (from 905 ms to 193 ms).

### E. Our Hardened Tailored Engine (`ninfer-t4-tp2`)
- **Workspace Location**: `ninfer-t4-tp2/`
- **Commit `230d28a4`**: `feat: tailor NInfer SM75 TP2 for 2x Tesla T4 deployment on Kaggle`
  - Fixed Issue #3 poison NaN bug in `w8_pair_plan.cpp` by routing $T \in [161, 192]$ on SM75 to `ConcatMmaR32C64`.
  - Fixed direct-I/O 4KB alignment in `materializer.cpp`.
  - Added `-DNINFER_SM75_INT8_KV_ONLY=ON` default build preset in `CMakeLists.txt` and `gqa_attention_decode_launch.cuh` (<4 min compile time).
  - Ported Pinned-Host `PeerMailbox` fallback transport.
  - Added one-click Kaggle deployment kit in `deploy/`.
