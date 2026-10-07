# UVA Direct P2P Transport & CUDA Graph Capturability

> **Location**: `engine_knowledge_base/05_collectives_and_transport/uva_direct_p2p_transport.md`  
> **Related Documents**: [Dual T4 PCIe Topology](../02_hardware_and_physics/dual_t4_pcie_topology.md) | [TP2 Architecture Overview](../03_tensor_parallelism_sharding/tp2_architecture_overview.md) | [Pinned-Host PeerMailbox](pinned_host_peer_mailbox.md)

This document details the primary inter-GPU communication mechanism implemented in `src/ops/common/allreduce.cu`: Unified Virtual Addressing (UVA) direct peer-to-peer copies, stream synchronization choreography, and why it is fully capturable inside CUDA Graphs.

---

## 1. Why `cudaMemcpyAsync(DeviceToDevice)` Over UVA?

### A. The Rejection of `cudaMemcpyPeerAsync`
Developers familiar with older CUDA multi-GPU programming often attempt to use `cudaMemcpyPeerAsync`.  
In modern CUDA (CUDA 12.x / 13.x), **`cudaMemcpyPeerAsync` is strictly rejected inside a CUDA stream capture region** with error code `cudaErrorStreamCaptureUnsupported`.
- If an engine uses `cudaMemcpyPeerAsync`, the entire tensor-parallel forward pass **cannot be captured into a CUDA Graph**.
- Without CUDA Graphs, every single token step incurs ~40 host API launches and CPU-GPU synchronization points per layer $\times$ 64 layers $\approx 2,560$ kernel launches per token, drowning performance in CPU driver overhead.

### B. The UVA Solution (`pull_peer`)
Under 64-bit Linux with Unified Virtual Addressing (UVA), every device pointer uniquely identifies its backing device in a single virtual address space.
In `src/ops/common/allreduce.cu`:
```cpp
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}
```
- **Execution**: The copy is issued on the **destination device's stream**.
- **Hardware Routing**: Because `cudaDeviceCanAccessPeer(0, 1) == 1` on our Kaggle PHB topology, the CUDA driver executes this call as a **direct hardware DMA transfer over PCIe at 9.91 GB/s**.
- **Capturability**: `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)` is **100% capturable inside CUDA stream capture**, allowing the entire cross-device forward pass to be recorded into a single re-executable CUDA Graph!

---

## 2. Three-Phase Collective Choreography

A cross-device reduction requires strict ordering between two independent GPU streams. If issued naively, a stream wait could snapshot an event before the peer stream had even recorded it.

In `src/ops/common/allreduce.cu`, both `allreduce_sum()` and `allgather_rows()` execute a verified three-phase choreography:

```
[GPU 0 Stream]                                           [GPU 1 Stream]
      |                                                        |
======+================== PHASE A: ISSUE RECORD ===============+======
cudaEventRecord(inputs_ready[0])                         cudaEventRecord(inputs_ready[1])
      |                                                        |
======+================== PHASE B: PULL PEER ==================+======
cudaStreamWaitEvent(inputs_ready[1])                     cudaStreamWaitEvent(inputs_ready[0])
cudaMemcpyAsync(peer -> local)                           cudaMemcpyAsync(peer -> local)
cudaEventRecord(pull_done[0])                            cudaEventRecord(pull_done[1])
      |                                                        |
======+================== PHASE C: LOCAL COMBINE ==============+======
cudaStreamWaitEvent(pull_done[1])                        cudaStreamWaitEvent(pull_done[0])
residual_add_launch(local += pulled)                     residual_add_launch(local += pulled)
      |                                                        |
[Both ranks hold final sum Z]                            [Both ranks hold final sum Z]
```

### Detailed Phase Logic:
1. **Phase A (Both Ranks)**:
   - Rank 0 records `inputs_ready[0]` on device 0's stream.
   - Rank 1 records `inputs_ready[1]` on device 1's stream.
2. **Phase B (Both Ranks)**:
   - Rank 0 stream waits on `inputs_ready[1]`, pulls rank 1's partial vector into its private staging buffer, then records `pull_done[0]`.
   - Rank 1 stream waits on `inputs_ready[0]`, pulls rank 0's partial vector into its private staging buffer, then records `pull_done[1]`.
3. **Phase C (Local Combine)**:
   - Rank 0 stream waits on `pull_done[1]`, then calls `residual_add_launch` in-place ($Z = Z_0 + Z_1$).
   - Rank 1 stream waits on `pull_done[0]`, then calls `residual_add_launch` in-place ($Z = Z_0 + Z_1$).

---

## 3. Local Combine Arithmetic Reuse

Rather than implementing a standalone reduction kernel, `allreduce_sum` reuses `detail::residual_add_launch` from `src/ops/launcher/residual_add.h`:
- Accumulation: High-precision FP32 intermediate accumulation.
- Rounding: Single round-to-nearest-even on the final BF16 store.
- **Benefit**: Guarantees identical numerical rounding behavior between normal residual connections and inter-GPU allreduce operations.
