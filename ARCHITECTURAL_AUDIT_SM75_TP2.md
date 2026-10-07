# Architectural Audit: SM75 TP2 Qwen3.6-27B Groupwise-INT

**Target Architecture:** 2× NVIDIA Tesla T4 (Turing `sm_75`, Compute Capability 7.5, 15.0 GiB VRAM each)  
**Target Model:** Qwen3.6-27B (or Qwen3.8-27B) Dense `groupwise-int` (W8A16)  
**Execution Mode:** Tensor Parallelism = 2 (`--tp 2 --devices 0,1`), MTP0 baseline, MTP3 draft-3 target, INT8 group-64 KV cache  
**Transport:** Dual-mode: Capturable UVA direct peer copies when P2P is granted, and Pinned-Host PeerMailbox fallback  

---

## 1. Cloned Repositories Overview (`cloned_repos/`)

All key repositories are stored in the local [`cloned_repos/`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos) directory:

1. **[`cloned_repos/ninfer-2080ti-22g-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2)**:
   - **Origin:** Fork by `zsq13767593046-bit` (submitted as PR #6 to `mr-september/ninfer-2080ti-22g`).
   - **Significance:** Already solved the Q4/Q5 groupwise-int sharding routes for Turing SM75! It merged `wamansou/ninfer-tp2-1m`'s TP2 architecture with `mr-september/ninfer-2080ti-22g`'s SM75 kernel backend.
   - **Key Commits:**
     - `fe4590c0`: *"perf(ops): route the TP2 Q4/Q5 shards through small-T exact kernels"*
     - `d3b079cf`: *"perf(artifact): batch strided shard uploads into 2D copies"*
     - `10af75b4`: *"fix(ops): correct SM75 mma_s8 and mma_f16 Turing fragment decomposition"*
     - `3c9c7caa`: *"fix(ops): fit SM75 INT8 decode shared memory and prefill page offsets"*
2. **[`cloned_repos/ninfer-2080ti-22g`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g)**:
   - **Origin:** Port by `mr-september` for modded 22GB RTX 2080 Ti.
   - **Significance:** Base ISA and kernel foundation for Turing SM75 (custom W8 GEMM, Split-K, GDN MmaUnsplit, INT8 KV decode).
3. **[`cloned_repos/ninfer-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2)**:
   - **Origin:** Fork by `ValerioDolci` (v0.4.4+).
   - **Significance:** Primary blueprint for mature TP2 runtime: contains Pinned-Host PeerMailbox (`peer_mailbox.cu`), pipelined exchange kernels, CUDA Graph all-reduce capture, startup mailbox probe, and concurrent TP2 request scheduling.
4. **[`cloned_repos/ninfer-dflash2-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-dflash2-tp2)**:
   - **Origin:** Fork by `parallelno`.
   - **Significance:** Solved per-device CUDA function attributes memoization ([`kernel_attr_once.h`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-dflash2-tp2/src/ops/launcher/kernel_attr_once.h)), multi-device materializer disjointness checks ([`src/artifact/materializer.cpp`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-dflash2-tp2/src/artifact/materializer.cpp)), and mirrored rank-1 KV page tables.
5. **[`cloned_repos/ninfer-windows-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-windows-tp2)**:
   - **Origin:** Fork by `ivanov84`.
   - **Significance:** Historical implementation of GPU0 $\leftrightarrow$ Pinned-Host Mailbox $\leftrightarrow$ GPU1 transport without NVLink.
6. **[`cloned_repos/ninfer-tp2-1m`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2-1m)**:
   - **Origin:** Fork by `wamansou`.
   - **Significance:** Original foundational two-rank execution model, 4-event pull collective, and YaRN 1M context.
7. **[`cloned_repos/ninfer-upstream`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-upstream)**:
   - **Origin:** Upstream `Neroued/ninfer`.
   - **Significance:** Upstream contracts and artifact definitions.

---

## 2. Source-Level Audit: Groupwise-INT Linear Sharding

In upstream NInfer and ValerioDolci TP2, groupwise-int artifacts were rejected at startup because the attention and GDN input projections are paired Q4/Q5 parents that lacked split routes.

In [`cloned_repos/ninfer-2080ti-22g-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2), this problem was solved as follows:

### A. Attention Input Projection (`attn_input_proj`)
- **Parent Tensor:** Packed Q \| K \| Gate \| V (`[14336, 5120]`, input rows 5120, query rows 6144, KV rows 1024).
- **TP2 Shard Geometry:**
  - Rank 0: Query rows $[0, 3072)$, KV rows $[0, 512)$ (12 Q heads, 2 KV heads)
  - Rank 1: Query rows $[3072, 6144)$, KV rows $[512, 1024)$ (12 Q heads, 2 KV heads)
  - Shard shape: `[7168, 5120]`
- **Implementation in Code:**
  - [`src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp#L114-L143):
    ```cpp
    bool q4_q5_attn_input_admits_shard(const Q4Q5AttnInputProblem& problem) noexcept {
        return problem.input_rows == 5120 && problem.query_rows == 3072 && problem.kv_rows == 512 &&
               problem.padded_k == 5120 && problem.cols >= 1;
    }
    void q4_q5_attn_input_dispatch_shard(const Tensor& x, const Weight& query_key_weight,
                                         const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, cudaStream_t stream) {
        if (problem.cols <= 16) {
            q4_q5_attn_input_small_t_shard_launch(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
            return;
        }
        if (problem.cols <= 20) {
            q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
            return;
        }
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
    }
    ```
  - [`src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu):
    Provides exact-kernel instantiation `q4_q5_attn_input_small_t_shard_launch` dedicated to 3072/512 rows for $T \le 16$ (decode widths).

### B. GDN Input Projection (`gdn_input_proj`)
- **Parent Tensor:** Packed Q \| K \| V \| Z (`[16384, 5120]`, input rows 5120, QK rows 4096, VZ rows 12288).
- **TP2 Shard Geometry:**
  - QK rows: 2048 per rank (8 key heads of 256)
  - VZ rows: 6144 per rank (24 value heads of 256)
  - Shard shape: `[8192, 5120]`
- **Implementation in Code:**
  - [`src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp#L81-L140):
    ```cpp
    if (supported_shard_shape(problem)) {
        if (problem.cols <= 16) { return {Q4Q5GdnInputScheduleId::IndependentDirectFixed}; }
        return {Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128};
    }
    ```
  - [`src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu):
    Implements `q4_q5_gdn_input_independent_shard_launch` templated for the local 2048/3072/3072 extents.

### C. Linear Projections Matrix

| Operation | Weight Format | Parent Shape | Shard Shape | Parallel Strategy | SM75 Kernel Route |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attn Out Proj** | W8 A16 | `[5120, 6144]` | `[5120, 3072]` | Col split + All-Reduce | W8 A16 LinearPair / Split-K (`w8_gemm_splitk.cu`) |
| **GDN Gating Proj** | BF16 | `[48, 5120]` | `[24, 5120]` | Row split (no reduce) | BF16 Linear / MmaUnsplit (`bf16_gdn_gating_proj_kernels.cu`) |
| **GDN Out Proj** | W8 A16 | `[5120, 6144]` | `[5120, 3072]` | Col split + All-Reduce | W8 A16 LinearPair / Split-K (`w8_gemm_splitk.cu`) |
| **SwiGLU Gate/Up** | W8 A16 | `[34816, 5120]` | `[17408, 5120]` | Row split (no reduce) | W8 Linear SwiGLU Small-T (`w8_linear_swiglu_small_t.cu`) |
| **SwiGLU Down** | W8 A16 | `[5120, 17408]` | `[5120, 8704]` | Col split + All-Reduce | W8 A16 `linear_add` (`w8_linear_add.cu`) |
| **LM Output Head** | W8 A16 | `[248320, 5120]` | `[124160, 5120]` | Vocab Row split + All-Gather | W8 Linear Head + Logit All-Gather |

---

## 3. Discovered Bugs & Mandatory Patches

### Bug 1: W8 LinearPair Silent Poison Columns on SM75 (Issue #3)
- **File:** [`src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu#L205-L212)
- **Problem:** For schedule `DualSplitKMediumC192` ($T \in [161, 192]$), the code calls `launch_medium<160, 2, 2, 2>`. It only computes the first 160 columns, leaving columns $161 \dots T$ unwritten with poison/NaN data.
- **Turing Constraint:** Turing static shared memory is capped at 48 KiB. A 192-column tile needs $2 \times 192 \times 128 + 2048 = 51,200\text{ bytes} = 50\text{ KiB}$, which cannot fit static smem.
- **Fix:** In [`src/ops/linear_pair/w8/w8_pair_plan.cpp`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/linear_pair/w8/w8_pair_plan.cpp#L42):
  ```cpp
  constexpr W8PairScheduleId kK2048Route161To192 =
  #if defined(NINFER_SM75)
      W8PairScheduleId::ConcatMmaR32C64;
  #else
      W8PairScheduleId::DualSplitKMediumC192;
  #endif
  ```

### Bug 2: 361 MB PTX Module / 6-Hour `ptxas` Build Hang (Issue #4 / PR #5)
- **File:** [`src/ops/launcher/gqa_attention_decode.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/launcher/gqa_attention_decode.cu)
- **Problem:** SM75 lacks native BF16 MMA instructions. The software emulation of `mma_bf16` expands into thousands of warp-shuffles and scalar FMAs. Instantiating all 96 BF16 partial decode kernels causes `ptxas` to compile an 8.16 million-line PTX file, running for >6.5 hours without producing a cubin.
- **Fix:** Adopt PR #5 (`NINFER_SM75_INT8_KV_ONLY`):
  In `CMakeLists.txt`:
  ```cmake
  option(NINFER_SM75_INT8_KV_ONLY "Omit software-emulated BF16 decode attention on SM75" ON)
  ```
  In `src/ops/launcher/gqa_attention_decode.cu`:
  ```cpp
  #if defined(NINFER_SM75_INT8_KV_ONLY)
  constexpr bool kSm75Int8KvOnly = true;
  #else
  constexpr bool kSm75Int8KvOnly = false;
  #endif
  ```
  Throw when BF16 KV decode is requested on an INT8-only build, eliminating the 96 template instantiations.

### Bug 3: Per-Device `cudaFuncSetAttribute` Cache Hazard
- **File:** [`src/ops/launcher/kernel_attr_once.h`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/launcher/kernel_attr_once.h)
- **Problem:** `cudaFuncSetAttribute` was previously cached in a process-static variable on device 0, failing rank 1 launches when dynamic shared memory was needed on longer prompts.
- **Status:** [`ninfer-2080ti-22g-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2) already contains `kernel_attr_once.h` with device-indexed caching.

---

## 4. Collective Transport: UVA Direct Copy vs Pinned-Host PeerMailbox

On Kaggle 2× Tesla T4:
- Hardware Topology: `PHB` (PCIe Host Bridge, PCIe Gen3 x16, 0 swap).
- `cudaDeviceCanAccessPeer(0,1) == 1`, `cudaDeviceCanAccessPeer(1,0) == 1`.
- Measured direct P2P Bandwidth: **~9.18 – 9.91 GB/s**.

### Transport Paths in NInfer TP2:
1. **Direct UVA Peer Copies (`allreduce.cu`):**
   - Stream-capturable `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)` over unified virtual addresses.
   - Used when peer access is enabled.
2. **Pinned-Host PeerMailbox (`peer_mailbox.cu`):**
   - In [`cloned_repos/ninfer-tp2/src/ops/common/peer_mailbox.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2/src/ops/common/peer_mailbox.cu) and [`include/ninfer/ops/peer_mailbox.h`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2/include/ninfer/ops/peer_mailbox.h).
   - GPU-side poll/consume through mapped pinned host memory.
   - Bypasses CUDA driver event overhead during captured graph execution.
   - On non-NVLink PCIe, drops 10 KiB reduction latency from ~277 µs down to **~41 µs**!

**Action:** Port `peer_mailbox.h`, `peer_mailbox.cu`, `peer_exchange.cuh`, and `mailbox_probe.cu` into the unified engine tree.

---

## 5. VRAM Budget for 2× Tesla T4 (15.0 GiB each)

| Allocation Item | Per-Card Sizing (GiB) | Aggregate (2× T4) | Notes |
| :--- | :--- | :--- | :--- |
| **Weights (Qwen3.6-27B groupwise-int)** | **8.97 GiB** | 16.29 GiB | Halved via TP2 row/col sharding |
| **CUDA Context & Driver overhead** | ~0.48 GiB | ~0.96 GiB | Fixed driver baseline |
| **Runtime Workspace & Graph nodes** | ~1.50 GiB | ~3.00 GiB | Allocations for 1888-node TP2 decode graph |
| **Free Headroom for Paged KV Cache** | **~4.05 GiB** | **~8.10 GiB** | Available for KV pool |
| **KV Footprint (INT8 group-64)** | **16.9 KiB / token** | 33.8 KiB / token | 16 layers × 2 KV heads/card × 256 dim × 1B |
| **Max Context at `--tp 2`** | **~245,000 tokens** | — | Comfortably supports up to **131,072 (128K)** or **262,144 (262K)** context |
