# MMA Fragment Emulation & Register Decomposition Fix

> **Location**: `engine_knowledge_base/04_cuda_kernels_and_sm75_execution/mma_fragment_emulation.md`  
> **Related Documents**: [Turing SM75 Architecture](../02_hardware_and_physics/turing_sm75_architecture.md) | [W8 GEMM & Split-K](w8_gemm_and_splitk.md) | [INT8 Group-64 KV Cache](../06_memory_and_kv_cache/int8_group64_kv_cache.md)

This document explains the mathematical decomposition of modern PTX MMA operations into Turing-native `m8n8k16` sub-operations, detailing the register transposition bug in upstream `mr-september` (Commit `10af75b4`) and its fix.

---

## 1. Hardware Background: Turing MMA Limits

Modern GPU architectures (Ampere `sm_80`, Ada `sm_89`, Blackwell `sm_120`) provide native hardware PTX instructions:
- `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (INT8 inputs, INT32 accumulation, 32 K-elements per step).
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (FP16 inputs, FP32 accumulation, 16 K-elements per step).

Turing SM75 hardware **does not possess `m16n8k32.s8` or `m16n8k16.f16`**.  
Turing hardware only implements:
- `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` (Turing native INT8).
- `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` (Turing native FP16).

Therefore, any portable CUDA kernel targeting modern MMA shapes must emulate them on SM75 by composing multiple Turing-native instructions inside `src/ops/common/mma.cuh`.

---

## 2. PTX Register Layout vs Turing Sub-Operations

### A. PTX `m16n8k32.s8` Register Contract
The PTX specification defines the four 32-bit registers of operand fragment $A$ ($a_0, a_1, a_2, a_3$, each holding four 8-bit integers) as follows:
- $a_0$: Rows $0 \dots 7$, Columns $0 \dots 15 \implies (M_0, K_0)$
- $a_1$: Rows $8 \dots 15$, Columns $0 \dots 15 \implies (M_1, K_0)$
- $a_2$: Rows $0 \dots 7$, Columns $16 \dots 31 \implies (M_0, K_1)$
- $a_3$: Rows $8 \dots 15$, Columns $16 \dots 31 \implies (M_1, K_1)$

Operand $B$ holds two 32-bit registers ($b_0, b_1$):
- $b_0$: Columns $0 \dots 15 \implies K_0$
- $b_1$: Columns $16 \dots 31 \implies K_1$

Accumulator $C$ holds four 32-bit integers:
- $(c_0, c_1)$: Output rows $0 \dots 7 \implies M_0$
- $(c_2, c_3)$: Output rows $8 \dots 15 \implies M_1$

### B. The Bug in Upstream `mr-september`
In `mr-september/ninfer-2080ti-22g/src/ops/common/mma.cuh`, the decomposition was written as:
```cpp
// INCORRECT UPSTREAM IMPLEMENTATION:
mma_s8_m8n8k16(c0, c1, a0, b0); // (M0, K0) * K0 -> accumulates into M0 (c0, c1)
mma_s8_m8n8k16(c0, c1, a1, b1); // WRONG! a1 is (M1, K0), accumulating into M0 (c0, c1)!
mma_s8_m8n8k16(c2, c3, a2, b0); // WRONG! a2 is (M0, K1), accumulating into M1 (c2, c3)!
mma_s8_m8n8k16(c2, c3, a3, b1); // (M1, K1) * K1 -> accumulates into M1 (c2, c3)
```

**The Consequence**:
The second sub-product added Row Half 1 into Row Half 0!  
The third sub-product added Row Half 0 into Row Half 1!  
Every INT8-KV query-key ($QK^T$) product and every INT8 prefill projection produced completely invalid arithmetic results.

---

## 3. The Corrected Decomposition (Commit `10af75b4`)

In `ninfer-2080ti-22g-tp2`, author `zsq` corrected the mapping in `src/ops/common/mma.cuh`:

```cpp
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3,
                                       unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // PTX m16n8k32 fragments interleave row halves with K halves:
    // a0=(M0,K0), a1=(M1,K0), a2=(M0,K1), a3=(M1,K1).
    // Compose four Turing m8n8k16 operations in that order:
    mma_s8_m8n8k16(c0, c1, a0, b0); // M0: a0 * b0
    mma_s8_m8n8k16(c0, c1, a2, b1); // M0: a2 * b1  (CORRECT: stays in M0)
    mma_s8_m8n8k16(c2, c3, a1, b0); // M1: a1 * b0  (CORRECT: stays in M1)
    mma_s8_m8n8k16(c2, c3, a3, b1); // M1: a3 * b1
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
```

Similarly, `mma_f16` was corrected:
- Decomposing `m16n8k16` into two `m16n8k8` operations paired as $(a_0, a_1) \times b_0$ and $(a_2, a_3) \times b_1$.

### Validation
Verified by an exact integer-oracle microtest (`tests/ops/test_mma_sm75.cu`).  
Before the fix, the oracle failed 128/128 elements; after the fix, it achieved **100% bit-exact parity**.
