# Exhaustive Audit Report: Issues, Pull Requests, Bug Fixes & Breakthroughs Across All Repositories

This report synthesizes the deep code audit of all seven repositories in `cloned_repos/`. It details every known issue, pull request, bug fix, race condition, memory leak, and performance breakthrough to ensure that no future agent session or engineer needs to re-investigate or duplicate these findings.

---

## 1. Repository Inventory & Branch Map

| Repository Path | Origin / Fork Author | Crucial Role & Key Branches |
|---|---|---|
| `cloned_repos/ninfer-2080ti-22g-tp2` | `zsq13767593046-bit` | **Primary Working Base**. SM75 Turing kernels + Valerio TP2 execution framework + Q4/Q5 TP2 shard routes + YaRN 1M context. Branches: `master`, `origin/tp2-sm75`. |
| `cloned_repos/ninfer-2080ti-22g` | `mr-september` | Base SM75 Turing port. W8 GEMM, Split-K, GDN `MmaUnsplit` ($T \ge 9$). Branches: `master`. |
| `cloned_repos/ninfer-tp2` | `ValerioDolci` | Mature TP2 runtime, CUDA Graph decode, concurrent serving, Pinned-Host PeerMailbox. Branches: `main`. |
| `cloned_repos/ninfer-windows-tp2` | `ivanov84` | Windows TP2 runtime, device leak fix, **MTP prefix-reuse at TP2**. Branches: `windows-tp2`, `fix/tp2-mtp-prefix-reuse`. |
| `cloned_repos/ninfer-dflash2-tp2` | `parallelno` | DFlash2 TP2 handling, mirrored KV tables, **prefill pipeline boost**. Branches: `main`, `prefill_boost2`. |
| `cloned_repos/ninfer-tp2-1m` | `wamansou` / `giocom` | Original TP2 pull collective design & YaRN 1M context. Sharding geometry documentation. |
| `cloned_repos/ninfer-upstream` | `Neroued` | Current upstream reference. Single-GPU contracts, prompt cache boundaries, metric endpoints. |

---

## 2. Exhaustive Bug Catalog & Critical Corrections

### Bug 1: Silent NaN / Output Poisoning Bug (Issue #3 in `mr-september/ninfer-2080ti-22g`)
- **Location**: `src/ops/linear_pair/w8/w8_pair_plan.cpp` (line 42) & `src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu` (lines 205-228).
- **Reported By**: Davis-Liang.
- **Root Cause**:
  In `w8_pair_gemm_splitk.cu`, the schedule `DualSplitKMediumC192` ($T \in [161, 192]$) on Turing SM75 (`#if defined(NINFER_SM75)`) was assigned:
  ```cpp
  case W8PairScheduleId::DualSplitKMediumC192:
      if (x.ne[1] <= 192) {
          launch_medium<160, 2, 2, 2>(x, first_weight, second_weight, first_out, second_out, stream);
          return;
      }
  ```
  Because the kernel was compiled with tile width `<160>`, it only computed output columns $0 \dots 159$. Columns $160 \dots 191$ were never written by the GPU kernel, leaving uninitialized memory and poison NaNs in the output tensor!
  The tile was capped at 160 because Turing's static shared memory limit is 48 KiB (a 192-column tile would require 50 KiB and fail).
- **Fix Applied**:
  In `src/ops/linear_pair/w8/w8_pair_plan.cpp`:
  ```cpp
  #if defined(NINFER_SM75)
      {161, 192, W8PairScheduleId::ConcatMmaR32C64},
  #else
      {161, 192, W8PairScheduleId::DualSplitKMediumC192},
  #endif
  ```
  On SM75, token counts in $[161, 192]$ route directly to `ConcatMmaR32C64`, which computes all columns without truncation and fits inside Turing shared memory.

---

### Bug 2: Corrupted INT8 KV QK Products & Prefill PV Path (Commit `10af75b4` in `ninfer-2080ti-22g-tp2`)
- **Location**: `src/ops/common/mma.cuh` (lines 148-185).
- **Root Cause**:
  In Turing SM75 emulation of PTX `m16n8k32.s8`, PTX registers interleave row halves with K halves:
  - $a_0 = (M_0, K_0)$, $a_1 = (M_1, K_0)$, $a_2 = (M_0, K_1)$, $a_3 = (M_1, K_1)$.
  In `mr-september/ninfer-2080ti-22g`, the decomposition was:
  ```cpp
  mma_s8_m8n8k16(c0, c1, a0, b0);
  mma_s8_m8n8k16(c0, c1, a1, b1); // WRONG: a1 is M1, but c0/c1 is M0!
  mma_s8_m8n8k16(c2, c3, a2, b0); // WRONG: a2 is M0, but c2/c3 is M1!
  mma_s8_m8n8k16(c2, c3, a3, b1);
  ```
  Similarly, `mma_f16` paired $a_0/a_2$ and $a_1/a_3$ incorrectly. Every INT8 KV attention computation and FP16 prefill PV computation was numerically corrupted.
- **Fix in `ninfer-2080ti-22g-tp2`**:
  Decomposition correctly rewritten to:
  ```cpp
  mma_s8_m8n8k16(c0, c1, a0, b0);
  mma_s8_m8n8k16(c0, c1, a2, b1);
  mma_s8_m8n8k16(c2, c3, a1, b0);
  mma_s8_m8n8k16(c2, c3, a3, b1);
  ```
  Validated via exact integer oracle test (`tests/ops/test_mma_sm75.cu`).

---

### Bug 3: INT8 Decode Shared Memory Overflow & Prefill Page Offset Bug (Commit `3c9c7caa` in `ninfer-2080ti-22g-tp2`)
- **Location**: `src/ops/kernel/gqa_attention_decode_i8.cuh` & `src/ops/kernel/gqa_attention_prefill_i8.cuh`.
- **Root Cause**:
  1. *Decode*: Widening the visible-key domain to 1,048,576 tokens grew the page-id array, pushing static shared memory 328 bytes past the 48 KiB ceiling (96 nvlink link errors). Moved the page-id array to the tail of dynamic shared memory.
  2. *Prefill*: Halving prefill tile size ($B_c = 64 \to 32$) on SM75 caused KV cache indexing to use `key_l` instead of `page_offset + key_l`. The second 32-token tile in every 64-token page read the first tile's data, corrupting any prompt prefill longer than 32 tokens!
- **Fix in `ninfer-2080ti-22g-tp2`**:
  Calculated `const int page_offset = tile_k0 & kPagedKVPageMask;` and indexed with `page_offset + key_l`.

---

### Bug 4: Multi-Hour PTXAS Compilation Hang (Issue #4 & PR #5 in `mr-september/ninfer-2080ti-22g`)
- **Location**: `src/ops/launcher/gqa_attention_decode.cu`.
- **Root Cause**:
  SM75 lacks native BF16 tensor cores. Emulated `mma_bf16` instantiated 96 template variations in one translation unit, producing an 8.16 million line (361 MB) PTX file that caused `ptxas` to run for over 6.5 hours and consume >12 GB RAM.
- **Fix**:
  1. Header separation: `gqa_attention_decode_launch.cuh` allows each geometry to instantiate in its own TU (`gqa_attention_decode_tp2.cu`), dropping compile time to minutes.
  2. Build flag `-DNINFER_SM75_INT8_KV_ONLY=ON`: Compiles out emulated BF16 decode-attention templates completely for lean INT8-KV builds on Kaggle.

---

### Bug 5: Multi-Lane Output Corruption via Unmirrored KV Table Row (PR #2 in `parallelno/ninfer-dflash2-tp2`)
- **Location**: `src/targets/qwen3_6/impl/runtime/program_impl.h` (`bind_sequence_kv`).
- **Author**: Valerio Dolci.
- **Root Cause**:
  `peer->io.text_kv_table_row` was set to 0 once at startup and was never updated when binding active sequences. In concurrent execution, rank 1 decoded from whatever old KV page was at slot 0, causing garbled responses.
- **Fix in `ninfer-2080ti-22g-tp2`**:
  Line 1215: `set_peer_i32(peer->io.text_kv_table_row, sequence.kv->text_peer->bound_row());` inside `ScopedDevice` scope.

---

### Bug 6: Direct I/O Staging Unaligned Memory Failure (Commit `795f590b` in `ValerioDolci/ninfer-tp2`)
- **Location**: `src/artifact/materializer.cpp` (`Slot` class).
- **Root Cause**:
  `reader.read_direct` requires 4096-byte alignment (`kPayloadAlignment`). Small page-locked allocations (`cudaMallocHost`) can be packed into shared pages by the CUDA driver, resulting in an unaligned starting address and triggering OS errors ("unaligned or oversized direct read").
- **Fix**:
  Allocate `bytes + kPayloadAlignment` in `Slot` and align `data` pointer to `kPayloadAlignment`.

---

### Bug 7: Asynchronous Device Work Leak on Startup Failure (Commit `caee6083` in `ValerioDolci/ninfer-tp2`)
- **Location**: Request commit / startup failure handling.
- **Root Cause**:
  When request startup failed (e.g. publication check failure), buffers and pages were returned to the pool immediately. If kernels were already queued asynchronously on GPU 0 and GPU 1, they continued writing into memory that had been handed to other requests.
- **Fix**:
  Execute `synchronize_devices()` before recycling buffers and sequence rows on failure.

---

### Bug 8: CUDA Device Context Leak in Arena Allocation (Commit `2cf050bf` in `ivanov84/ninfer-windows-tp2`)
- **Location**: `RequestMemory` / arena construction.
- **Root Cause**:
  `cudaSetDevice(1)` was invoked during rank 1 setup without saving and restoring the calling thread's active device. Subsequent rank 0 operations executed on device 1, leading to `cudaErrorInvalidValue`.
- **Fix**:
  Wrap all rank switches in `ScopedDevice` / `CurrentDeviceGuard`.

---

### Bug 9: Accidental Acceptance on Proposal Probability Miss (Commit `350136a8` in `ValerioDolci/ninfer-tp2`)
- **Location**: `src/ops/kernel/speculative_round.cuh`.
- **Root Cause**:
  In speculative verification, if the drafted token was missing from the candidate support, the lookup returned $q_d = 0$. The accept condition was `!(pd >= qd || u * qd < pd)`. With $q_d = 0$, $u \cdot q_d < p_d \implies 0 < p_d$ evaluated to true, accidentally accepting an invalid draft unconditionally!
- **Fix**:
  `reject = qd <= 0.0f || !(pd >= qd || u * qd < pd);`.

---

## 3. High-Value Technical Breakthroughs & Implementations

### Breakthrough 1: TP2 Q4/Q5 Small-T Exact Shard Routing (Commit `fe4590c0` in `ninfer-2080ti-22g-tp2`)
- **Mechanism**:
  Attention input projections and GDN input projections have column-sharded weights:
  - Attention input parent: $7168 \times 5120 \to 3584 \times 5120$ per rank.
  - GDN input parent: $4096 \times 5120 \to 2048 \times 5120$ per rank.
  Previously, shards dispatched to generic grouped-MMA kernels (designed for prefill), taking 2.1 ms per GDN projection on Turing.
  `zsq` added exact-kernel instantiations for column widths $T \le 16$:
  - `q4_q5_attn_input_small_t_shard_launch`
  - `q4_q5_gdn_input_independent_shard_launch`
- **Impact**:
  Cut decode kernel time by 57%; decode speed increased from 10.4 to **27.5 tok/s** on dual Turing GPUs (+61% over TP1).

---

### Breakthrough 2: Batched 2D `cudaMemcpy2DAsync` Shard Uploads (Commit `d3b079cf` in `ninfer-2080ti-22g-tp2`)
- **Mechanism**:
  Naively uploading column shards row-by-row generated ~4,000,000 tiny ~4 KiB copies, taking 9.5 seconds.
  Batched detection groups identical stride runs across ranks into `cudaMemcpy2DAsync` calls.
- **Impact**:
  Model load time dropped from 9.5s down to **3.0s** on dual-GPU systems.

---

### Breakthrough 3: Pinned-Host PeerMailbox Transport (`ValerioDolci/ninfer-tp2`)
- **Mechanism**:
  On systems lacking PCIe P2P, driver-staged `cudaMemcpyAsync(DeviceToDevice)` incurs 277 µs latency per 10 KiB reduction due to driver event chaining.
  `PeerMailbox` uses mapped pinned host memory with GPU-side polling loops (`peer_exchange.cuh`), achieving **41 µs** reduction latency.
- **Application to Kaggle 2×T4**:
  On Kaggle, hardware P2P is fully enabled (~9.18 GB/s, 2-7 µs latency via direct UVA `pull_peer`). `PeerMailbox` serves as a verified zero-event fallback for environments where P2P is blocked.

---

### Breakthrough 4: TP2 + MTP Multi-Turn Prefix Reuse (Commit `eab0a929` in `ivanov84/ninfer-windows-tp2`)
- **Mechanism**:
  In multi-turn chat conversations, previously TP2 reset prefix caching to full re-prefill.
  Dmitriy Ivanov's fix mirrors `rewrite_checkpoint_hidden` across ranks and restores from per-rank retained hidden states at the `TurnClosure` frontier.
- **Impact**:
  - TTFT on conversation turn 2 dropped from 905 ms to 193 ms (**4.7x speedup**).
  - TTFT on conversation turn 3 dropped from 957 ms to 210 ms (**4.5x speedup**).

---

### Breakthrough 5: Prefill Pipeline with Pinned Relay (Commit `87f7c4d6` in `parallelno/ninfer-dflash2-tp2`)
- **Mechanism**:
  Pipelining chunked prefill transfers across PCIe concurrently with computation using pinned host relay buffers.
- **Impact**:
  +22% prefill throughput with only 5k context memory trade-off.

---

## 4. Synthesis & Recommendations for Deployment

1. **Base Tree**: Use `cloned_repos/ninfer-2080ti-22g-tp2` as the authoritative source. It already contains the Turing SM75 kernels, the corrected MMA decompositions, and the exact Q4/Q5 TP2 small-T shard dispatch.
2. **Immediate Patches to Maintain**:
   - `w8_pair_plan.cpp`: Route $T \in [161, 192]$ to `ConcatMmaR32C64` on SM75 (prevents NaN poison bug).
   - `materializer.cpp`: Align `Slot` staging buffer to `kPayloadAlignment`.
3. **Execution on Kaggle 2×T4**:
   - UVA direct P2P is active and measured at 9.18 GB/s. `allreduce.cu` with `pull_peer` executes native hardware P2P transfers.
   - Run with `--tp 2 --devices 0,1 --kv-dtype int8 --kv-capacity auto`.
