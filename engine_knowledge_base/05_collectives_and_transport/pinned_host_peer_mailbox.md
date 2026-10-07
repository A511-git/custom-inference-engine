# Pinned-Host PeerMailbox Fallback Transport

> **Location**: `engine_knowledge_base/05_collectives_and_transport/pinned_host_peer_mailbox.md`  
> **Related Documents**: [UVA Direct P2P Transport](uva_direct_p2p_transport.md) | [Dual T4 PCIe Topology](../02_hardware_and_physics/dual_t4_pcie_topology.md) | [Roofline & Performance Math](../02_hardware_and_physics/roofline_and_performance_math.md)

This document specifies the fallback transport implementation: the Pinned-Host `PeerMailbox` subsystem developed by Valerio Dolci and Dmitriy Ivanov. It explains how kernel-level spinlocks in mapped host memory eliminate driver event staging on systems where direct P2P is restricted.

---

## 1. Why PeerMailbox Exists

On many multi-GPU systems—such as GeForce consumer hardware (RTX 3090, 4090, 5090) or virtualized Windows/WSL2/WDDM environments—the driver disables direct hardware P2P between PCIe slots.
- Under such conditions, `cudaMemcpyAsync(DeviceToDevice)` is staged through system RAM by driver worker threads.
- The CUDA event chain that synchronizes this staged copy costs **~277 µs per 10 KiB reduction**.
- Across 128 allreduces per token:
  $$128 \times 277\text{ µs} \approx \mathbf{35.5\text{ ms of pure driver overhead!}}$$
  This cuts decode throughput in half!

### The PeerMailbox Alternative
Instead of relying on the CUDA driver's event-staged copy engine:
- Both GPUs allocate a shared, page-locked, system-mapped host slab (`cudaHostAllocMapped`).
- Both GPUs run a **concurrent GPU kernel** (`peer_exchange_pipelined_kernel`).
- The GPUs publish their vectors directly to system RAM, flip a hardware flag, spin on the peer's flag, and combine the vectors directly in GPU registers.
- **Latency**: Drops from **~277 µs down to ~41 µs** (a **6.7x latency reduction**)!

---

## 2. Memory Slab Layout in System RAM

In `src/ops/common/peer_mailbox.cu`, the slab is laid out as a single contiguous allocation per slot set:

```
+---------------------------------------------------------------------------------------------+
|                                  PINNED HOST MEMORY SLAB                                    |
+------------------------------+------------------------------+---------------+---------------+
| Payload Rank 0 (Slots 0..N-1)| Payload Rank 1 (Slots 0..N-1)| Flags Rank 0  | Flags Rank 1  |
|      (256-byte aligned)      |      (256-byte aligned)      | (64-byte line)| (64-byte line)|
+------------------------------+------------------------------+---------------+---------------+
```

1. **Payload Alignment**: Each payload slot is aligned to **256 bytes**, ensuring that 16-byte vectorized stores (`float4` / `int4`) never cross cache line boundaries.
2. **Flag Striding**: Each release flag sits on its own **64-byte CPU cache line** (`kPeerFlagStride = 64`) to eliminate false sharing between CPU and GPU memory controllers.
3. **Sticky Hang Word**: Sits at the end of the slab. If either GPU experiences a kernel timeout or unrecoverable lock, it asserts the hang word and bails out cleanly rather than hanging the machine.

---

## 3. Epoch Synchronization Protocol

Because both GPUs execute identical token schedules, each slot's epoch counter advances in lockstep without requiring host CPU resets between graph replays:

```
[GPU 0 Kernel Execution]                                      [GPU 1 Kernel Execution]
         |                                                                 |
target = epoch + 1                                                target = epoch + 1
Store 16B payload to Rank 0 slot                                  Store 16B payload to Rank 1 slot
__threadfence_system()                                            __threadfence_system()
atomicAdd(&arrival_counter, 1)                                    atomicAdd(&arrival_counter, 1)
Last CTA block sets mine_flag = target                            Last CTA block sets mine_flag = target
         |                                                                 |
Spin on peer_flag >= target (relaxed load)                        Spin on peer_flag >= target (relaxed load)
Acquire fence: ld.global.acquire                                  Acquire fence: ld.global.acquire
Read peer payload (ld.global.cv)                                  Read peer payload (ld.global.cv)
Combine in registers (FP32 sum)                                   Combine in registers (FP32 sum)
Store final sum to local VRAM                                     Store final sum to local VRAM
```

### Safety Features:
- **Wrap-Safe Comparisons**: Epoch counters use unsigned integers with modulo comparisons, surviving millions of tokens without overflow bugs.
- **Spin Limit (`kPeerSpinLimit`)**: If the peer flag does not arrive after $500,000$ iterations, the thread block sets the host hang word and returns an error rather than wedging the GPU kernel driver.

---

## 4. Role in Our Dual Tesla T4 Engine

In our Kaggle 2× Tesla T4 system:
- **Primary Transport**: Hardware UVA direct P2P is fully functional (**9.91 GB/s**, **2.5–5.0 µs** latency) and is active by default.
- **Fail-Safe Fallback**: `PeerMailbox` files (`peer_mailbox.h`, `peer_mailbox.cu`, `peer_exchange.cuh`, `mailbox_probe.cu`) are compiled into the engine. If the engine is ever deployed to a machine without P2P access (e.g. cloud instances with strict IOMMU isolation or Windows WDDM), it automatically shifts to `PeerMailbox` without crashing.
