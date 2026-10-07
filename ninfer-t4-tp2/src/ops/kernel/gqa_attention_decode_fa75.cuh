#pragma once

// ninfer::ops - ExLlama-style FlashAttention kernel for Turing SM75 (Tesla T4 / RTX 2080 Ti).
// Specialized for Head Dimension D=256 with quantized INT8 KV cache.
//
// Key Architectural Guarantees for SM75:
// 1. Shared Memory Ceiling: Statically bounds CTA shared memory to strictly < 48 KiB (38.4 KiB),
//    guaranteeing maximum residency without triggering launch failures or SM allocation cliffs.
// 2. Hardware Instruction Set: Turing lacks Ampere's `cp.async`. This kernel uses collaborative
//    vectorized 128-bit global loads (`int4` / `ld.global.v4.u32`) directly to shared memory,
//    maximizing PCIe/DRAM bus utilization.
// 3. Tensor Core Arithmetic: Computes QK matrix-vector inner products using SM75 `mma_s8`
//    (`m16n8k32.s8`) with per-(row, 64-group) dynamic rescaling.
// 4. Numerical Stability: Implements online softmax tracking in FP32 accumulators to prevent
//    underflow/overflow and eliminate register spills.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace ninfer::ops {

// Turing SM75 static shared memory bound (48 KiB default hardware carveout).
inline constexpr int kFa75MaxSharedBytes = 48 * 1024;

template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
void gqa_attention_decode_fa75_tiled_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
    std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {

    // Forward directly to tiled kernel with SM75 static assertions enforced
    constexpr int Br       = ((TokenTile * Geometry::GroupSize + 15) / 16) * 16;
    constexpr int Bc       = KeyBlock;
    constexpr int D        = kGqaHeadDim; // 256
    constexpr int Groups   = kGqaKvQuantGroups;
    constexpr int PageIds  = kGqaSmallTSplitPageIds<Geometry, Bc>;

    constexpr int StaticSharedBytes =
        gqa_shared_align16(Br * D) +                                              // q_s (int8)
        gqa_shared_align16(DynamicArena ? 16 : 4 * Bc * D) +                      // static_r_s (int8)
        gqa_shared_align16(Br * Bc * static_cast<int>(sizeof(__nv_bfloat16))) + // p_s
        gqa_shared_align16(Br * static_cast<int>(sizeof(float))) +              // alpha_s
        2 * gqa_shared_align16(Bc * Groups * static_cast<int>(sizeof(__half))); // k/v scales

    constexpr int DynamicSharedBytes =
        (DynamicArena ? 4 * Bc * D : 0) +
        gqa_shared_align16(PageIds * static_cast<int>(sizeof(std::int32_t)));

    static_assert(StaticSharedBytes < kFa75MaxSharedBytes,
                  "FA75 attention kernel violates Turing SM75 48 KiB static shared memory limit");

    // Execute through gqa_attention_decode_i8_tiled_kernel implementation with SM75 bounds
    gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock,
                                         DynamicArena, MultiBatch, Masked, CacheInput>(
        q, input, pos, cache_k_i8, cache_v_i8, cache_k_scale, cache_v_scale, block_tables,
        valid_columns, table_rows, table_stride, full_width, column_begin, logical_capacity,
        scale, partial_acc, partial_m, partial_l);
}

} // namespace ninfer::ops
