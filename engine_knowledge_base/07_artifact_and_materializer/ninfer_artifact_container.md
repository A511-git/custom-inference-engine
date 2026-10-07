# NInfer Artifact Container & Direct-I/O 4KB Alignment Fix

> **Location**: `engine_knowledge_base/07_artifact_and_materializer/ninfer_artifact_container.md`  
> **Related Documents**: [Batched 2D Materialization](batched_2d_materialization.md) | [Q4/Q5 Shard Geometry](../03_tensor_parallelism_sharding/q4_q5_shard_geometry.md)

This document details the `.ninfer` container format and the direct I/O memory alignment fix (`kPayloadAlignment`) in `src/artifact/materializer.cpp` (Commit `795f590b`).

---

## 1. The `.ninfer` Artifact Container Format

A `.ninfer` file is a standalone container format designed for zero-copy memory mapping and direct OS disk streaming:

```
+-------------------------------------------------------------------------------+
| Header Magic | JSON Metadata (Shapes/Offsets) | 4096-Byte Padding | Payload   |
|   (8 bytes)  |      (Dynamic size)            |    (Alignment)    |  Weights  |
+-------------------------------------------------------------------------------+
```

1. **Self-Describing JSON Metadata**: Defines tensor shapes, quantization formats, scale offsets, and byte spans.
2. **4096-Byte Payload Alignment**: The payload begins at an offset strictly aligned to `kPayloadAlignment = 4096` bytes.
3. **Direct OS Read (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`)**:
   - `reader.read_direct` bypasses the OS page cache entirely, streaming multi-gigabyte weight chunks directly from NVMe/RAM into page-locked GPU staging buffers.
   - Requirement: **Both the file offset and the user-space destination buffer pointer must be strict multiples of 4096 bytes**.

---

## 2. The Direct I/O Unaligned Read Failure (Commit `795f590b`)

### A. The Defect
In `src/artifact/materializer.cpp`, the staging `Slot` class allocated memory using `cudaMallocHost`:
```cpp
// DEFECTIVE IMPLEMENTATION:
class Slot {
public:
    Slot(std::size_t bytes, std::span<DeviceContext* const> devices) : buffer(bytes) {
        ...
    }
    PinnedHostBuffer buffer;
};
```
- In CUDA, `cudaMallocHost` pools small allocations into shared 64 KiB memory pages.
- If a small allocation occurs before a slot, the slot's starting pointer `buffer.data()` may land at offset 1,536 or 2,048 inside the physical page.
- When `reader.read_direct` was called with this unaligned buffer pointer, the Linux kernel rejected the read with `EINVAL` (`unaligned or oversized direct read`).

### B. The Applied Fix
In `src/artifact/materializer.cpp`:
```cpp
constexpr std::size_t kPayloadAlignment = 4096;

class Slot {
public:
    Slot(std::size_t bytes, std::span<DeviceContext* const> devices)
        : buffer(bytes + kPayloadAlignment),
          data(static_cast<std::byte*>(buffer.data()) +
               (kPayloadAlignment - reinterpret_cast<std::uintptr_t>(buffer.data()) %
                                        kPayloadAlignment) %
                   kPayloadAlignment) {
        for (std::size_t i = 0; i < devices.size(); ++i) {
            CUDA_CHECK(cudaSetDevice(devices[i]->device));
            CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
        }
    }
    ...
    PinnedHostBuffer buffer;
    std::byte* data = nullptr; // GUARANTEED 4096-BYTE ALIGNED START
};
```
- The buffer allocates `bytes + kPayloadAlignment` and aligns `data` internally.
- All `read_direct` calls and downstream host-to-device memory copies use `slot.data`.
- This completely eliminated direct-I/O alignment failures across all operating systems.
