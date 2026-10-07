# Q4/Q5 Shard Geometry & Memory Layout

> **Location**: `engine_knowledge_base/03_tensor_parallelism_sharding/q4_q5_shard_geometry.md`  
> **Related Documents**: [TP2 Architecture Overview](tp2_architecture_overview.md) | [Layer Sharding Matrix](layer_sharding_matrix.md) | [Batched 2D Materialization](../07_artifact_and_materializer/batched_2d_materialization.md)

This document explains the physical bit-level and byte-level memory layouts of groupwise quantized tensors (`q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`) and why row-slicing ($N/2$) preserves quantization integrity without modifying the storage format.

---

## 1. Groupwise Quantization Memory Layout

In NInfer's `groupwise-int` weight format, matrices are stored in $[N, K]$ row-major format, where:
- $N$ is the number of output channels / rows.
- $K$ is the contraction dimension / columns.
- Quantization groups live along the **$K$ dimension** with a fixed block size:
  - `q4_g64_fp16`: Group size $G = 64$.
  - `q5_g64_fp16`: Group size $G = 64$.
  - `q6_g64_fp16`: Group size $G = 64$.
  - `q8_g32_fp16`: Group size $G = 32$.

```
Row 0: [ Group 0 (64 values + FP16 scale) ][ Group 1 (64 values + FP16 scale) ] ... [ Group K/64 ]
Row 1: [ Group 0 (64 values + FP16 scale) ][ Group 1 (64 values + FP16 scale) ] ... [ Group K/64 ]
...
Row N-1: [ Group 0 (64 values + FP16 scale) ][ Group 1 (64 values + FP16 scale) ] ... [ Group K/64 ]
```

### Critical Layout Invariant
**Quantization groups NEVER cross row boundaries**.  
Each row is an independently self-contained byte sequence consisting of:
1. Packed quantized integer nibbles/bytes.
2. Group scaling factors (FP16 / half-precision).
3. Group minimums/zeros (if asymmetric).

---

## 2. Row Slicing ($N$) vs Contraction Slicing ($K$)

### A. Why Row Slicing ($N \to N/2$) Is Naturally Trivial
When sharding along the output row dimension $N$ (Column-Parallel):
$$[N, K] \implies \text{Rank 0: } [N/2, K], \quad \text{Rank 1: } [N/2, K]$$
- **Integrity**: Each rank receives complete, intact rows.
- **Quantization Compatibility**: Every quantization group along $K$ remains 100% complete.
- **Bit-Stream Compatibility**: No repacking, no bit-shifting, and no scale recomputations are required!
- Both rank shards are **valid standalone `q4_g64_fp16` or `q5_g64_fp16` tensors** that can be directly passed into existing single-GPU SM75 linear kernels.

### B. Why Contraction Slicing ($K \to K/2$) Requires Alignment Care
When sharding along the contraction dimension $K$ (Row-Parallel):
$$[N, K] \implies \text{Rank 0: } [N, K/2], \quad \text{Rank 1: } [N, K/2]$$
- **Alignment Requirement**: The contraction dimension $K/2$ must be a strict multiple of the quantization group size $G$ (64) and the MMA memory tile stride (128).
- **Verification on Qwen3.6-27B**:
  - `down_proj`: $K = 17408 \implies K/2 = 8704$.
    $$8704 / 64 = 136\text{ groups (Exact integer)}$$
    $$8704 / 128 = 68\text{ tiles (Exact integer)}$$
  - `o_proj`: $K = 5120 \implies K/2 = 2560$.
    $$2560 / 64 = 40\text{ groups (Exact integer)}$$
    $$2560 / 128 = 20\text{ tiles (Exact integer)}$$
  - `gdn.out_proj`: $K = 6144 \implies K/2 = 3072$.
    $$3072 / 64 = 48\text{ groups (Exact integer)}$$
    $$3072 / 128 = 24\text{ tiles (Exact integer)}$$
- Every contraction split lands on an exact group-64 and tile-128 boundary. Neither rank ever cuts through the middle of a quantization block.

---

## 3. Storage Formula per Quantization Format

| Quantization Codec | Stored Bits / Weight | Scale Format | Bytes per Group of 64 | Effective Bits / Weight |
|---|:---:|:---:|:---:|:---:|
| `q4_g64_fp16` | 4-bit (packed nibbles) | 16-bit FP16 | $32\text{ B (nibbles)} + 2\text{ B (scale)} = 34\text{ bytes}$ | **4.25 bits/weight** |
| `q5_g64_fp16` | 5-bit (4-bit + 1-bit high) | 16-bit FP16 | $32\text{ B} + 8\text{ B} + 2\text{ B} = 42\text{ bytes}$ | **5.25 bits/weight** |
| `q6_g64_fp16` | 6-bit (4-bit + 2-bit high) | 16-bit FP16 | $32\text{ B} + 16\text{ B} + 2\text{ B} = 50\text{ bytes}$ | **6.25 bits/weight** |
| `q8_g32_fp16` | 8-bit (int8 bytes) | 16-bit FP16 | $32\text{ B} + 2\text{ B} = 34\text{ bytes}$ | **8.50 bits/weight** |

### Total Weight Volume
- Qwen3.6-27B packed storage: **16.29 GiB** ($17.49 \times 10^9$ bytes).
- Sharded storage per GPU in TP2: **~8.15 GiB** ($8.75 \times 10^9$ bytes) + replicated embeddings/norms (~0.8 GiB) = **~8.95 GiB total resident weight footprint**.
