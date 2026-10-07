# Fork Comparisons & Component Selection Matrix

> **Location**: `engine_knowledge_base/01_provenance_and_lineage/fork_comparisons.md`  
> **Related Documents**: [Lineage & Commits](lineage_and_commits.md) | [Change Impact & Verification Matrix](change_impact_and_verification_matrix.md) | [TP2 Architecture Overview](../03_tensor_parallelism_sharding/tp2_architecture_overview.md) | [UVA Direct P2P Transport](../05_collectives_and_transport/uva_direct_p2p_transport.md)

This document provides a side-by-side technical matrix comparing all seven upstream and fork repositories analyzed in this project. It explains exactly what code and features were incorporated, what was rejected, and the technical reasons behind each decision.

---

## 1. Comprehensive Repository Comparison Matrix

| Repository | Focus Architecture | Quantization Support | Multi-GPU Strategy | Status for Dual T4 SM75 |
|---|:---:|:---:|:---:|---|
| **`Neroued/ninfer`** | Blackwell (`sm_120a`) | NVFP4 native, FP8, W8A16 | Single-GPU only | Architectural reference. Single-GPU contracts and container format specs adopted; Blackwell-only kernels rejected. |
| **`mr-september/ninfer-2080ti-22g`** | Turing (`sm_75`) | `groupwise-int` (W8A16, Q4/Q5, Q6) | Single-GPU only | SM75 W8 GEMM and GDN `MmaUnsplit` adopted; lacked TP2 multi-GPU support and contained bugs (MMA decomposition, Issue #3 NaNs). |
| **`wamansou/ninfer-tp2-1m`** | Blackwell (`sm_120a`) | NVFP4 native | TP2 (UVA P2P) + YaRN 1M | Geometry reference. Shard planning and YaRN math adopted; NVFP4-only kernels incompatible with Turing. |
| **`ValerioDolci/ninfer-tp2`** | Ampere / Blackwell | NVFP4 native | Mature TP2 + CUDA Graph + Mailbox | Runtime reference. Rank-local state, `allreduce.cu`, and `PeerMailbox` adopted; lacked SM75 Q4/Q5 shard routes. |
| **`parallelno/ninfer-dflash2-tp2`** | Ada / Blackwell | NVFP4 native | TP2 + DFlash2 + Prefill Boost | Quality reference. Mirrored KV sequence row fix (PR #2) verified; DFlash2 deferred to post-MTP3 milestone. |
| **`ivanov84/ninfer-windows-tp2`** | Ada / Hopper | NVFP4 native | TP2 (Mailbox on WDDM) | Feature source. `ScopedDevice` leak fix (`2cf050bf`) and MTP prefix reuse (`eab0a929`) adopted. |
| **`zsq13767593046-bit/ninfer-2080ti-22g-tp2`** | Turing (`sm_75`) | `groupwise-int` (W8A16, Q4/Q5, Q6) | TP2 (SM75) + YaRN 1M | **PRIMARY BASE CODEBASE**. Solved SM75 Q4/Q5 small-T exact shard dispatch, batched 2D upload, corrected MMA decomposition. |
| **`ninfer-t4-tp2` (OUR REPO)** | **Tesla T4 (`sm_75`)** | **`groupwise-int` (W8A16, INT8 KV)** | **TP2 (Hardware P2P + Mailbox fallback)** | **FINAL INTEGRATED ENGINE**. Applied Issue #3 fix, 4KB direct I/O alignment, fast compile preset, and Kaggle deployment kit. |

---

## 2. Granular Extraction vs Rejection Breakdown

### A. What Was Extracted from `mr-september/ninfer-2080ti-22g`
- **Adopted**:
  - Turing W8 GEMM and Split-K schedules in `src/ops/linear/w8/` and `src/ops/linear_pair/w8/`.
  - Turing shared-memory bounds ($48\text{ KiB}$ static ceiling).
  - GDN gating projection routing to `MmaUnsplit` for token counts $T \ge 9$ in `src/ops/gdn_gating_proj/`.
- **Rejected**:
  - Unpatched MMA emulation in `src/ops/common/mma.cuh` (contained register transposition that corrupted INT8 KV).
  - Single-GPU model execution path (lacked TP2 collectives).

### B. What Was Extracted from `ValerioDolci/ninfer-tp2`
- **Adopted**:
  - Multi-device execution abstractions: `ExecutionContext`, `DeviceSelection`, and `CurrentDeviceGuard`.
  - Stream-capturable collective implementations: `allreduce_sum()` and `allgather_rows()` in `src/ops/common/allreduce.cu`.
  - Pinned-Host `PeerMailbox` subsystem: `include/ninfer/ops/peer_mailbox.h`, `src/ops/common/peer_mailbox.cu`, `src/ops/kernel/peer_exchange.cuh`, `tools/tp2/mailbox_probe.cu`.
  - Direct I/O 4096-byte alignment (`kPayloadAlignment`) in `src/artifact/materializer.cpp`.
  - Device synchronization guard on request startup failure (`caee6083`).
  - Speculative verification lookup miss rejection (`350136a8`).
- **Rejected**:
  - Pure NVFP4 assumption in model load plans (incompatible with Turing hardware).

### C. What Was Extracted from `zsq13767593046-bit/ninfer-2080ti-22g-tp2`
- **Adopted**:
  - **The Missing Link**: Q4/Q5 attention and GDN input projection TP2 sharding dispatch in `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp` and `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp`.
  - Small-T exact shard kernels (`q4_q5_attn_input_small_t_shard_launch` and `q4_q5_gdn_input_independent_shard_launch`) for $T \le 16$, cutting decode latency by 57%.
  - Batched 2D `cudaMemcpy2DAsync` uploads in `src/artifact/materializer.cpp` (3.0s load time).
  - Corrected Turing `mma_s8` and `mma_f16` decompositions in `src/ops/common/mma.cuh`.
  - Paged INT8 KV cache dynamic shared-memory tail staging and Bc=32 prefill page offset index fix in `src/ops/kernel/gqa_attention_decode_i8.cuh` and `gqa_attention_prefill_i8.cuh`.
  - Compilation TU splitting (`gqa_attention_decode_tp2.cu`).
- **Required Fixes Applied by Us**:
  - Issue #3 Poison NaN bug fix in `src/ops/linear_pair/w8/w8_pair_plan.cpp`.
  - Direct I/O `kPayloadAlignment` buffer padding in `src/artifact/materializer.cpp`.
  - `NINFER_SM75_INT8_KV_ONLY=ON` fast build option in `CMakeLists.txt`.

### D. What Was Extracted from `ivanov84/ninfer-windows-tp2`
- **Adopted**:
  - `ScopedDevice` leak fix around arena creation (`2cf050bf`).
  - MTP multi-turn prefix reuse via per-rank retained hidden states and mirrored `rewrite_checkpoint_hidden` (`eab0a929` on branch `fix/tp2-mtp-prefix-reuse`).

### E. What Was Extracted from `parallelno/ninfer-dflash2-tp2`
- **Adopted**:
  - Mirrored KV table row fix: `set_peer_i32(peer->io.text_kv_table_row, sequence.kv->text_peer->bound_row())` in `bind_sequence_kv` (PR #2).
  - Materializer destination bounds checks.
- **Deferred to Later Milestones**:
  - DFlash2 drafter integration (deferred until baseline TP2 MTP0/MTP3 is established on Kaggle).
  - Prefill pipeline boost (`prefill_boost2` branch).
