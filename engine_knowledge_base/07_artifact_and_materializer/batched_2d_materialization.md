# Batched 2D Materialization & The 3.0-Second Model Load

> **Location**: `engine_knowledge_base/07_artifact_and_materializer/batched_2d_materialization.md`  
> **Related Documents**: [NInfer Artifact Container](ninfer_artifact_container.md) | [Q4/Q5 Shard Geometry](../03_tensor_parallelism_sharding/q4_q5_shard_geometry.md) | [Layer Sharding Matrix](../03_tensor_parallelism_sharding/layer_sharding_matrix.md)

This document details the batched 2D strided memory upload optimization in `src/artifact/materializer.cpp` (Commit `d3b079cf`), which reduced dual-GPU weight loading times from **9.5 seconds down to 3.0 seconds**.

---

## 1. The Multi-GPU Shard Loading Problem

When loading a single-GPU (TP1) model, weights are contiguous byte spans:
- Uploading an 8 GiB layer takes a single, uninterrupted `cudaMemcpyAsync` call.
- The GPU copy engine reaches near-peak PCIe bandwidth (~12 GB/s) instantly.

### The TP2 Sharding Complication
In Tensor Parallelism, column-sharded matrices (like attention $Q/K$ and MLP `gate_up`) arrive interleaved in source order:
- Row 0 belongs to GPU 0.
- Row 1 belongs to GPU 1.
- Row 2 belongs to GPU 0, etc.

In naive materializers, each row range is submitted as an individual 1D copy (`cudaMemcpyAsync`).
- For a 16.29 GiB model, this issued over **4,000,000 separate ~4 KiB memory copy calls**!
- Each call requires an OS driver transition, ring-buffer command packet creation, and PCIe doorbell write.
- **The Result**: The CPU driver queue drowned in overhead, stretching two-device model load time to **9.5 seconds** despite fast NVMe drives.

---

## 2. The Solution: Strided Run Detection & `cudaMemcpy2DAsync`

Author `zsq` introduced a strided run detector in `src/artifact/materializer.cpp` (lines 350–415):

```cpp
// Lookahead scan across upcoming copy ranges:
if (copy_begin == range.source_begin && copy_end == range.source_end) {
    const std::uint64_t row_bytes = range.source_end - range.source_begin;
    for (std::size_t scan = range_index + 1;
         scan < ranges.size() && ranges[scan].source_begin < chunk_end;
         ++scan) {
        ...
        // Check for constant source stride and constant destination stride:
        const std::uint64_t source_stride = candidate.source_begin - previous.source_begin;
        const std::uint64_t dest_stride   = candidate.destination - previous.destination;
        ...
        ++run;
    }
}
```

### The 2D Copy Dispatch
Whenever a sequence of rows shares identical byte lengths, constant source strides (stepping over the peer's bytes in the host staging buffer), and constant destination strides:
```cpp
if (run >= 2) {
    const std::size_t row_bytes = static_cast<std::size_t>(range.source_end - range.source_begin);
    CUDA_CHECK(cudaMemcpy2DAsync(
        range.destination, static_cast<std::size_t>(first_dest_stride),
        slot.data + static_cast<std::size_t>(range.source_begin - source),
        static_cast<std::size_t>(first_source_stride),
        row_bytes, run,
        cudaMemcpyHostToDevice, devices[device_slot]->load_stream));
    ...
    continue;
}
```

Instead of 4,000,000 individual copy submissions, the driver submits **a few hundred batched 2D matrix copies**.

---

## 3. Impact & Measurements

| Metric | Naive Scalar 1D Copies | Batched 2D Materializer (`cudaMemcpy2DAsync`) | Improvement |
|---|:---:|:---:|:---:|
| **Driver API Calls** | ~4,000,000 calls | **~450 calls** | **8,800x fewer calls** |
| **Copy Engine Submissions** | Continuous queue stalls | Streamlined bursts | Minimal CPU overhead |
| **Two-Device Load Time** | 9.5 seconds | **3.0 seconds** | **3.2x faster load** |
| **Numerical Equivalence** | Baseline | Bit-identical | **100% exact match** |
