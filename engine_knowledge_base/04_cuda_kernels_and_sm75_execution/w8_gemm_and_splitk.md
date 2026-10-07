# W8 GEMM, Split-K & The Issue #3 Poison NaN Bug

> **Location**: `engine_knowledge_base/04_cuda_kernels_and_sm75_execution/w8_gemm_and_splitk.md`  
> **Related Documents**: [Turing SM75 Architecture](../02_hardware_and_physics/turing_sm75_architecture.md) | [MMA Fragment Emulation](mma_fragment_emulation.md) | [Small-T Exact Kernels](small_t_exact_kernels.md)

This document details the Turing W8 Split-K GEMM kernel implementation, the scheduling logic, and an exhaustive root-cause analysis of **Issue #3 (The Silent Poison NaN Bug)** along with the applied fix.

---

## 1. Turing W8 Split-K Kernel Architecture

On Turing SM75, dense matrix multiplications during small-to-medium token counts ($T \le 192$) do not possess sufficient arithmetic work to saturate 40 SMs with standard 2D grid tiling.
- **Split-K Strategy**: Slices the inner contraction dimension $K$ across multiple Thread Blocks (CTAs).
- Each CTA computes a partial accumulation along its slice of $K$.
- An atomic reduction or secondary reduction kernel combines partial sums into the final output buffer.

### Source Files in Engine
- `src/ops/linear/w8/w8_rowsplit_gemm_splitk.cu`
- `src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu`
- `src/ops/linear_pair/w8/w8_pair_plan.cpp`

---

## 2. Issue #3 Deep Dive: The Silent Poison NaN Bug

### A. The Symptom
Reported by Davis-Liang on the base Turing fork (`mr-september/ninfer-2080ti-22g`):
Whenever the engine processed prompts with token counts in the range $T \in [161, 192]$, generated text became garbled, corrupted, or contained poison NaNs and infinite loops. At other token counts (e.g. $T = 128$ or $T = 256$), output was completely normal.

### B. Root Cause Analysis
In `src/ops/linear_pair/w8/w8_pair_plan.cpp`, line 42 defined the routing schedule table for contraction size $K = 2048$:
```cpp
constexpr std::array<W8PairRouteSpec, 37> kK2048Routes{{
    ...
    {129, 160, W8PairScheduleId::DualSplitKMediumC160},
    {161, 192, W8PairScheduleId::DualSplitKMediumC192},  // <-- Selected for T in [161, 192]
    {193, 384, W8PairScheduleId::ConcatMmaR32C64},
    ...
}};
```

Now inspect the launch dispatch in `src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu` (lines 205–212) under `#if defined(NINFER_SM75)`:
```cpp
#if defined(NINFER_SM75)
    case W8PairScheduleId::DualSplitKMediumC192:
        if (x.ne[1] <= 192) {
            launch_medium<160, 2, 2, 2>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
#endif
```

**Notice the catastrophic bug**:
- The schedule is for column width up to **192** (`DualSplitKMediumC192`).
- But on SM75, it called `launch_medium<160, 2, 2, 2>`!
- **Why did upstream do that?** Because a 192-column tile requires $\approx 50\text{ KiB}$ of static shared memory, which exceeds Turing SM75's strict 48 KiB hardware limit!
- **The Disaster**: Because the template was compiled for tile width 160, the GPU thread block loop only evaluated columns $0 \dots 159$. Columns $160 \dots 191$ **were never written by the GPU**!
- The memory buffer for columns $160 \dots 191$ retained whatever garbage or uninitialized NaNs were left in VRAM. This poisoned the downstream layer activations, corrupting all tokens in that range.

---

## 3. The Resolution

The solution is to **never route $T \in [161, 192]$ to `DualSplitKMediumC192` on Turing SM75**.  
Instead, route it to `ConcatMmaR32C64`, which uses smaller, flexible tiles that fit comfortably inside 48 KiB shared memory and compute every single output column without truncation.

### Applied Patch in `src/ops/linear_pair/w8/w8_pair_plan.cpp`
```diff
--- a/src/ops/linear_pair/w8/w8_pair_plan.cpp
+++ b/src/ops/linear_pair/w8/w8_pair_plan.cpp
@@ -41,7 +41,11 @@ constexpr std::array<W8PairRouteSpec, 37> kK2048Routes{{
     {129, 160, W8PairScheduleId::DualSplitKMediumC160},
+#if defined(NINFER_SM75)
+    {161, 192, W8PairScheduleId::ConcatMmaR32C64},
+#else
     {161, 192, W8PairScheduleId::DualSplitKMediumC192},
+#endif
     {193, 384, W8PairScheduleId::ConcatMmaR32C64},
```

### Verification & Validation
1. The route table array size remains exactly 37, satisfying the static assertion `routes_are_closed(kK2048Routes)`.
2. Token counts $T \in [161, 192]$ execute `w8_pair_concat_mma_launch` with tile `<32, 64, 32, 16, 3>`.
3. All 192 columns are computed with zero uninitialized memory and zero NaNs.
