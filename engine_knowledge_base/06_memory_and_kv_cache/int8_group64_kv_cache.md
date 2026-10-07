# INT8 Group-64 KV Cache & Prefill Indexing Fixes

> **Location**: `engine_knowledge_base/06_memory_and_kv_cache/int8_group64_kv_cache.md`  
> **Related Documents**: [16GB VRAM Budgeting](16gb_vram_budgeting.md) | [Turing SM75 Architecture](../02_hardware_and_physics/turing_sm75_architecture.md) | [Roofline & Performance Math](../02_hardware_and_physics/roofline_and_performance_math.md)

This document details the paged INT8 group-64 KV cache layout, the mathematical memory savings, and two critical bug fixes implemented in `gqa_attention_decode_i8.cuh` and `gqa_attention_prefill_i8.cuh` (Commit `3c9c7caa`).

---

## 1. INT8 Group-64 KV Cache Memory Model

### A. Memory Savings vs Standard BF16 KV Cache
In standard autoregressive Transformers, the KV cache stores $K$ and $V$ activations for all past tokens across all layers.
- For Qwen3.6-27B (64 layers, head dimension $D = 128$, 8 KV heads globally $\to$ **4 KV heads per GPU in TP2**):
  - **BF16 Footprint per Token (per GPU)**:
    $$\text{Bytes}_{\text{BF16}} = 2\text{ (K + V)} \times 64\text{ layers} \times 4\text{ heads} \times 128\text{ dim} \times 2\text{ bytes} = \mathbf{131,072\text{ bytes (128.0 KiB/token)}}$$
  - **INT8 Group-64 Footprint per Token (per GPU)**:
    Each 128-element head vector is quantized into two 64-element groups. Each group requires 64 INT8 values + one 16-bit FP16 scaling factor ($64 + 2 = 66\text{ bytes}$ per head):
    $$\text{Bytes}_{\text{INT8}} = 2 \times 64\text{ layers} \times 4\text{ heads} \times (128 + 4)\text{ bytes} = \mathbf{67,584\text{ bytes (66.0 KiB/token)}} \approx \mathbf{16.9\text{ KiB/head-layer/GPU}}$$
  - **Result**: INT8 group-64 quantization **halves the KV cache memory consumption**, allowing a 16GB Tesla T4 to support context lengths over **65,000 to 130,000 tokens**!

---

## 2. Bug Fix 1: Prefill Page Offset Indexing Bug (Commit `3c9c7caa`)

### A. The Defect in Upstream Prefill
In `src/ops/kernel/gqa_attention_prefill_i8.cuh`, Turing SM75 requires halving the prefill key tile dimension from $B_c = 64$ down to $B_c = 32$ to fit inside Turing's register and shared memory budgets.
- Paged KV pages are fixed at **64 tokens per physical page** (`kPagedKVPageTokens = 64`).
- When $B_c = 32$, two consecutive prefill key tiles share the **same 64-token physical page**.
- **The Upstream Flaw**:
  Upstream indexed the KV cache buffer strictly by tile-local key index `key_l`:
  ```cpp
  // INCORRECT UPSTREAM INDEXING:
  const std::int64_t off = gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d, key_l);
  ```
- **The Consequence**: For the second 32-token tile of each 64-token page, `key_l` ranged from $0 \dots 31$. It read from the **beginning** of the page instead of starting at byte offset 32!
- The second 32-token tile read the first tile's data, causing silent corruption for any prompt longer than 32 tokens!

### B. The Applied Fix
In `src/ops/kernel/gqa_attention_prefill_i8.cuh`:
```cpp
// CORRECTED IMPLEMENTATION:
const int page_offset = tile_k0 & kPagedKVPageMask; // 0 for tile 1, 32 for tile 2
...
const std::int64_t off = gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d,
                                                           page_offset + key_l);
```
Both key and scale indices now offset by `page_offset + key_l`, completely resolving prefill data corruption.

---

## 3. Bug Fix 2: INT8 Decode Shared Memory Overflow (Commit `3c9c7caa`)

### A. The Defect in Decode Kernels
In `src/ops/kernel/gqa_attention_decode_i8.cuh`, widening the addressable context window to 1,048,576 keys scaled up the page-id array `physical_pages_s[PageIds]`.
- On SM75, the static shared memory allocation reached **49,480 bytes**—exactly **328 bytes over the 48 KiB static ceiling** (49,152 bytes)!
- The CUDA linker aborted with **96 `nvlink` errors** (`static shared memory limit exceeded`).

### B. The Applied Fix
The page-id array was relocated from a static shared array to the **tail of the dynamic shared memory arena**:
```cpp
constexpr int DynamicSharedBytes =
    (DynamicArena ? 4 * Bc * D : 0) + gqa_shared_align16(PageIds * static_cast<int>(sizeof(std::int32_t)));

// Bind pointer to dynamic tail:
std::int32_t* physical_pages_s = reinterpret_cast<std::int32_t*>(
    dynamic_r_s + (DynamicArena ? 4 * Bc * D : 0));
```
Dynamic shared memory is charged against the 64 KiB opt-in budget rather than the 48 KiB static ceiling, enabling clean compilation and execution.
