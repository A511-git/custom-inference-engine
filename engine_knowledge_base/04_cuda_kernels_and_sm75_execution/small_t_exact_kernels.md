# Small-T Exact Kernels & The +61% Speedup Discovery

> **Location**: `engine_knowledge_base/04_cuda_kernels_and_sm75_execution/small_t_exact_kernels.md`  
> **Related Documents**: [Layer Sharding Matrix](../03_tensor_parallelism_sharding/layer_sharding_matrix.md) | [W8 GEMM & Split-K](w8_gemm_and_splitk.md) | [Roofline & Performance Math](../02_hardware_and_physics/roofline_and_performance_math.md)

This document explains the single most important performance breakthrough in the SM75 TP2 port: the implementation of small-T exact kernels for column-sharded Q4/Q5 projections (Commit `fe4590c0`), which increased dual-GPU decode throughput from **10.4 tok/s to 27.5 tok/s**.

---

## 1. The Performance Paradox

Before Commit `fe4590c0`, benchmarking the first SM75 TP2 merge on Turing hardware revealed a paradox:
- **TP1 (Single GPU)**: Measured decode speed was **17.0 tok/s**.
- **TP2 (Dual GPU)**: Measured decode speed was **10.4 tok/s**!
Instead of running faster by distributing work across two cards, TP2 ran **40% slower**!

---

## 2. Profiling & Root Cause

Detailed CUDA event profiling identified the exact bottleneck:
- **Attention Input Projection** ($[3584, 5120]$ shard): Spent **0.7 ms** per call.
- **GDN Input Projection** ($[2048, 5120]$ shard): Spent **2.1 ms** per call.
- Across 64 layers, these two input projections alone consumed **57% of total kernel execution time**!

### Why Were They Taking 2.1 ms?
1. The single-GPU (TP1) engine had dedicated, hand-tuned "Small-T Exact Kernels" for the full parent shapes:
   - Attention parent: $7168 \times 5120$.
   - GDN parent: $4096 \times 5120$.
   These small-T kernels were hardcoded for small token counts ($T \le 16$, especially decode $T = 1 \dots 4$) with optimal register usage and minimal CTA scheduling overhead.
2. When the TP2 merge halved the matrix dimensions to the shard sizes ($3584 \times 5120$ and $2048 \times 5120$):
   - The shard planning functions in `q4_q5_attn_input_plan.cpp` and `q4_q5_gdn_input_plan.cpp` had **no small-T exact kernel instantiations for the halved row shapes**!
   - Consequently, the plan fell back to the generic **grouped-MMA prefill kernels**!
   - The grouped-MMA prefill kernels are designed for massive token counts ($T = 512 \dots 4096$) with large $BM = 64$ tiles. When fed a decode width of $T = 1$ to $4$, the kernel suffered catastrophic under-utilization, launching mostly empty tiles and taking 2.1 ms per projection!

---

## 3. The Implementation (Commit `fe4590c0`)

Author `zsq` templated the small-T launcher families on their row-shape constants and instantiated them for both the full TP1 parent and the head-local TP2 shard:

### A. Code Locations
- `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu`
- `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp`
- `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu`
- `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp`

### B. Shard Plan Dispatch Logic
In `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp`:
```cpp
Q4Q5AttnInputScheduleId q4_q5_attn_input_dispatch_shard(std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("columns must be positive"); }
    // Decode widths (1 <= cols <= 16) take the dedicated small-T shard exact kernel:
    if (cols <= 16) { return Q4Q5AttnInputScheduleId::SmallTShard; }
    // Prefill widths take the row-count-generic grouped-MMA schedules:
    if (cols <= 64) { return Q4Q5AttnInputScheduleId::GroupedMmaCols64; }
    return Q4Q5AttnInputScheduleId::GroupedMmaLargeT;
}
```

In `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp`:
```cpp
Q4Q5GdnInputScheduleId q4_q5_gdn_input_dispatch_shard(std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("columns must be positive"); }
    if (cols <= 16) { return Q4Q5GdnInputScheduleId::IndependentShard; }
    if (cols <= 64) { return Q4Q5GdnInputScheduleId::GroupedMmaCols64; }
    return Q4Q5GdnInputScheduleId::GroupedMmaLargeT;
}
```

### C. Kernel Instantiation Parameters
- **Attention Input Shard**: Rows $M = 3584$, Contraction $K = 5120$.
- **GDN Input Shard**: Rows $M = 2048$, Contraction $K = 5120$.
- Tile geometry: Optimized for $T \in [1, 16]$, staging quantized nibbles in 16-byte vector loads and unrolling inner dot-products directly in SM registers.

---

## 4. Measured Performance Impact

| Metric | Before Fix (Grouped-MMA Fallback) | After Fix (Small-T Exact Shard) | Improvement |
|---|:---:|:---:|:---:|
| **GDN Input Proj Latency** | 2.10 ms / call | **0.18 ms / call** | **11.6x faster** |
| **Attn Input Proj Latency** | 0.70 ms / call | **0.12 ms / call** | **5.8x faster** |
| **Total Decode Kernel Time** | 96.2 ms / token | **41.3 ms / token** | **-57% reduction** |
| **Decode Throughput (MTP3)** | 10.4 tok/s | **27.5 tok/s** | **+164% (+61% over TP1)** |
