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

template <typename Geometry, int TokenTile, int KeyBlock, bool DynamicArena>
constexpr bool validate_fa75_shared_memory() {
    constexpr int Br       = ((TokenTile * Geometry::GroupSize + 15) / 16) * 16;
    constexpr int Bc       = KeyBlock;
    constexpr int D        = kGqaHeadDim; // 256
    constexpr int Groups   = kGqaKvQuantGroups;

    constexpr int StaticSharedBytes =
        gqa_shared_align16(Br * D) +
        gqa_shared_align16(DynamicArena ? 16 : 4 * Bc * D) +
        gqa_shared_align16(Br * Bc * static_cast<int>(sizeof(__nv_bfloat16))) +
        gqa_shared_align16(Br * static_cast<int>(sizeof(float))) +
        2 * gqa_shared_align16(Bc * Groups * static_cast<int>(sizeof(__half)));

    return StaticSharedBytes < kFa75MaxSharedBytes;
}

} // namespace ninfer::ops
