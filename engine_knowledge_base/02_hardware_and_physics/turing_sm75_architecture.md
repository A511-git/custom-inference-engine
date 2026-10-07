# Turing SM75 Architecture & Constraints

> **Location**: `engine_knowledge_base/02_hardware_and_physics/turing_sm75_architecture.md`  
> **Related Documents**: [Dual T4 PCIe Topology](dual_t4_pcie_topology.md) | [Roofline & Performance Math](roofline_and_performance_math.md) | [MMA Fragment Emulation](../04_cuda_kernels_and_sm75_execution/mma_fragment_emulation.md)

This document details the microarchitectural characteristics, compute limits, memory hierarchy, and register/shared-memory constraints of the **NVIDIA Tesla T4 (Turing TU104 / SM75)**.

---

## 1. Physical Specifications (Tesla T4)

| Microarchitectural Parameter | Specification | Functional Implication |
|---|:---:|---|
| **Die & Process** | TU104 (12nm FFN, TSMC) | First-generation RT & Tensor Core architecture |
| **Compute Capability** | `sm_75` | Lacks Ampere async copies (`cp.async`) and Hopper TMA |
| **Streaming Multiprocessors (SMs)** | **40 SMs** | 2,560 CUDA cores (64 FP32 cores / SM) |
| **Turing Tensor Cores** | **320 Tensor Cores** (8 per SM) | Support INT8, INT4, FP16 MMA. **No native BF16 MMA**. |
| **FP16 Tensor Core Peak** | **65.1 TFLOPS** | Compute ceiling for FP16 GEMM |
| **INT8 Tensor Core Peak** | **130.3 TOPS** | Compute ceiling for INT8 W8 MMA |
| **INT4 Tensor Core Peak** | **260.6 TOPS** | Hardware INT4 matrix multiply capability |
| **Base / Boost Clocks** | 585 MHz / 1,590 MHz | Typical sustained boost: ~1,000–1,400 MHz under heavy load |
| **Thermal Design Power (TDP)** | **70 W** | Passive cooling; aggressive thermal throttling if uncooled |
| **VRAM Capacity** | **16,384 MiB** (15,360 MiB usable) | 16 GB GDDR6 |
| **Memory Bus Width** | **256-bit** | 8 memory controllers (32-bit each) |
| **Memory Clock / Data Rate** | 5,000 MHz / 10 Gbps | **320 GB/s peak** (~240–260 GB/s sustained) |
| **L2 Cache Capacity** | **4,096 KiB (4 MiB)** | Shared crossbar cache |

---

## 2. On-Chip Memory Ceilings & Occupancy Constraints

The Turing SM memory partition imposes strict limits that dictate kernel launch configurations:

### A. Shared Memory Hierarchy
- **Physical Shared Memory + L1 Cache per SM**: 96 KiB total configurable SRAM per SM.
- **Turing Default Partition**: 64 KiB shared memory / 32 KiB L1 cache (or 32 KiB smem / 64 KiB L1).
- **Static Shared Memory Ceiling**: **48 KiB (49,152 bytes)** per Thread Block (CTA).
  - *Hard constraint*: Any static `__shared__` allocation exceeding 48 KiB triggers an immediate compilation / link error (`nvlink: shared memory allocation exceeded`).
  - *Opt-in Dynamic Shared Memory*: Up to **64 KiB (65,536 bytes)** per CTA can be allocated dynamically, provided `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` is explicitly called before launch.
- **Architectural Impact on NInfer**:
  - In `src/ops/kernel/gqa_attention_decode_i8.cuh`, widening the context window to 1M keys grew the page-id array past the 48 KiB ceiling. The array had to be relocated to the tail of **dynamic shared memory** (Commit `3c9c7caa`).
  - In `src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu`, tile width 192 requires ~50 KiB static smem, which Turing rejects. This caused the truncated `<160>` kernel launch and the poison NaN bug (Issue #3).

### B. Register File Bounds
- **Register File per SM**: 64K 32-bit registers (256 KiB) per SM.
- **Maximum Registers per Thread**: 255 registers.
- **Warp Schedulers per SM**: 4 warp schedulers per SM (each dispatching up to 1 instruction per clock).
- **Maximum Active Warps per SM**: 32 warps (1,024 threads per SM).
- **Maximum Active CTAs per SM**: 16 blocks per SM.
- **Occupancy Reality on Turing**:
  - Heavy quantized GEMM kernels consume 128 to 192 registers per thread, capping occupancy at 1 to 2 active warps per scheduler (33% to 50% theoretical occupancy).
  - In memory-bound decode, latency hiding is governed by memory subsystem queue depth rather than arithmetic occupancy.

---

## 3. Tensor Core Instruction Set & MMA Emulation

### A. Supported Matrix Multiply and Accumulate (MMA) Instructions
Turing Tensor Cores natively support:
1. `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` (FP16 inputs, FP32 accumulator).
2. `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` (INT8 inputs, INT32 accumulator).

### B. What Turing Lacks (The Missing Hardware Features)
1. **No Native `m16n8k32.s8`**:
   - Modern engines (Ampere, Hopper, Blackwell) execute 32 K-elements per step.
   - On Turing SM75, PTX `m16n8k32.s8` must be **decomposed into four `m8n8k16` sub-operations** in software (`mma.cuh`).
2. **No Native BF16 Tensor Cores**:
   - Turing has no native bfloat16 math. Any BF16 operation requires software conversions (`__bfloat162float`) or fragment emulation.
   - This absence is why compiling 96 BF16 decode attention templates overloaded `ptxas` for >6.5 hours, making `-DNINFER_SM75_INT8_KV_ONLY=ON` mandatory for lean builds.
3. **No Asynchronous Copy Engine (`cp.async`)**:
   - SM75 requires manual register staging (`L2 -> Registers -> Shared Memory`) via standard vector loads (`ld.global.v4`).
4. **No Tensor Memory Accelerator (TMA)**:
   - All tensor address indexing and strided pitch calculations must be evaluated by SM integer ALUs.
