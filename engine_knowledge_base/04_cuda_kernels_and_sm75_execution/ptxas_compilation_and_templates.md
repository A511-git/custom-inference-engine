# PTXAS Compilation Hang & Template Optimization

> **Location**: `engine_knowledge_base/04_cuda_kernels_and_sm75_execution/ptxas_compilation_and_templates.md`  
> **Related Documents**: [Turing SM75 Architecture](../02_hardware_and_physics/turing_sm75_architecture.md) | [Small-T Exact Kernels](small_t_exact_kernels.md) | [INT8 Group-64 KV Cache](../06_memory_and_kv_cache/int8_group64_kv_cache.md)

This document analyzes the root causes of the multi-hour `ptxas` compiler hang on Turing SM75, detailing the two solutions implemented in our engine: Translation Unit (TU) decoupling and the `-DNINFER_SM75_INT8_KV_ONLY=ON` compilation flag.

---

## 1. The Compilation Bottleneck

When compiling NInfer on Turing architecture (`CMAKE_CUDA_ARCHITECTURES=75`), the compilation of decode attention kernels in `src/ops/launcher/gqa_attention_decode.cu` exhibited severe resource exhaustion:
- **Build Time**: `ptxas` ran continuously for **>6.5 hours** without terminating.
- **Memory Consumption**: Exceeded **12 GiB of host RAM** during register allocation.
- **Intermediate Assembly**: Generated an **8.16 million line (361 MB) PTX module** from a single translation unit!

On remote cloud environments (like Kaggle Linux containers with strict 9-hour total execution timeouts and limited CPU RAM), a 6.5-hour compilation job inevitably crashes or exhausts notebook timeouts before inference can ever begin.

---

## 2. Root Cause Analysis

### A. Template Explosion Across Head Geometries
GQA decode attention kernels are templated on head geometries, KV data types, batch multi-threading, token counts, and split-KV intervals:
1. `Gqa27Geometry` (40 Q / 8 KV heads - TP1)
2. `Gqa27Tp2Geometry` (20 Q / 4 KV heads - TP2 text)
3. `Gqa35Tp2Geometry` (35B variant)

Compiling all three geometries in a single translation unit instantiated over **96 full kernel variations**.

### B. Emulated BF16 MMA Register Pressure on Turing
Because Turing SM75 **has no native BF16 tensor cores**, every single BF16 MMA operation is synthesized using emulated warp-shuffle primitives and 32-bit register packing.
- When `ptxas` attempts global register allocation and instruction scheduling across 96 massive emulated template functions in a single compilation unit, the register interference graph undergoes quadratic blowup ($O(N^2)$ to $O(N^3)$ complexity).
- The compiler enters an exponential backtracking loop attempting to satisfy the 255-register hardware limit per thread.

---

## 3. Dual-Layer Solution in Our Engine

### Layer 1: Translation Unit (TU) Decoupling (Commit `3c9c7caa`)
Rather than instantiating all geometries inside `gqa_attention_decode.cu`:
1. The launcher template definitions were extracted into a shared header:  
   `src/ops/launcher/gqa_attention_decode_launch.cuh`.
2. Dedicated translation units were created for each geometry:
   - `src/ops/launcher/gqa_attention_decode.cu`: Compiles TP1 geometries.
   - `src/ops/launcher/gqa_attention_decode_tp2.cu`: Compiles `Gqa27Tp2Geometry` in complete isolation:
     ```cpp
     // src/ops/launcher/gqa_attention_decode_tp2.cu
     #include "ops/launcher/gqa_attention_decode_launch.cuh"
     namespace ninfer::ops::detail {
     NINFER_GQA_DECODE_INSTANTIATE(Gqa27Tp2Geometry)
     }
     ```
3. CMake compiles both TUs concurrently across parallel CPU cores, preventing any single TU from accumulating millions of PTX lines.

### Layer 2: Fast-Build Flag (`-DNINFER_SM75_INT8_KV_ONLY=ON`)
Because 16GB Tesla T4 GPUs **strictly require INT8 KV cache** to fit Qwen3.6-27B into memory, compiling emulated BF16 decode attention kernels is completely redundant for our deployment.

We introduced the `NINFER_SM75_INT8_KV_ONLY` build flag:

#### In `CMakeLists.txt`:
```cmake
option(NINFER_SM75_INT8_KV_ONLY "Compile only INT8 KV attention decode kernels on SM75 (fast compile)" ON)
if(NINFER_SM75_INT8_KV_ONLY)
  add_compile_definitions(NINFER_SM75_INT8_KV_ONLY=1)
endif()
```

#### In `src/ops/launcher/gqa_attention_decode_launch.cuh`:
```cpp
    if (cache.dtype == DType::I8) {
        launch_for_dtype.template operator()<true>();
    } else {
#if defined(NINFER_SM75_INT8_KV_ONLY)
        throw std::invalid_argument("BF16 KV attention omitted in NINFER_SM75_INT8_KV_ONLY build");
#else
        launch_for_dtype.template operator()<false>();
#endif
    }
```

---

## 4. Compilation Results & Comparison

| Build Configuration | Translation Units | Emulated Templates | Total ptxas Time | Peak RAM |
|---|:---:|:---:|:---:|:---:|
| **Upstream `mr-september` Base** | 1 monolithic TU | 96 templates (BF16 + INT8) | **>6.5 hours (Hangs)** | >12 GiB |
| **TU Decoupled (`gqa_attention_decode_tp2.cu`)** | 2 parallel TUs | 96 templates | **~1.5 hours** | ~4 GiB / TU |
| **Hardened (`NINFER_SM75_INT8_KV_ONLY=ON`)** | 2 parallel TUs | 24 templates (INT8 only) | **< 4 minutes** | **< 1.2 GiB** |

Enabling `-DNINFER_SM75_INT8_KV_ONLY=ON` by default reduces full engine build times on Kaggle from hours down to **under 4 minutes**, with 100% full functionality for INT8 KV inference.
