# Dual Tesla T4 PCIe Topology & Interconnect Physics

> **Location**: `engine_knowledge_base/02_hardware_and_physics/dual_t4_pcie_topology.md`  
> **Related Documents**: [Turing SM75 Architecture](turing_sm75_architecture.md) | [Roofline & Performance Math](roofline_and_performance_math.md) | [UVA Direct P2P Transport](../05_collectives_and_transport/uva_direct_p2p_transport.md) | [Pinned-Host PeerMailbox](../05_collectives_and_transport/pinned_host_peer_mailbox.md)

This document analyzes the physical PCIe interconnect, measured Peer-to-Peer (P2P) performance, and collective reduction physics for the **2× NVIDIA Tesla T4** deployment on Kaggle Linux.

---

## 1. Measured Interconnect Diagnostics

The following empirical data is taken directly from the hardware diagnostic probe executed on the target Kaggle environment ([`Machine.MD`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/Machine.MD)):

```
========================================
CUDA device count: 2
GPU0: Tesla T4 | CC 7.5
GPU1: Tesla T4 | CC 7.5

P2P 0 -> 1: 1 (SUPPORTED)
P2P 1 -> 0: 1 (SUPPORTED)

=== GPU0 -> GPU1 ===
Transferred: 5.37 GB
Time:        542.248 ms
Bandwidth:   9.90 GB/s

=== GPU1 -> GPU0 ===
Transferred: 5.37 GB
Time:        541.813 ms
Bandwidth:   9.91 GB/s

========================================
FINAL P2P RESULT: SUPPORTED (9.91 GB/s)
========================================
```

---

## 2. PCIe Physical Topology & Routing

```
                    +------------------------------------+
                    |        Host CPU / Memory           |
                    +-----------------+------------------+
                                      |
                     +----------------+----------------+
                     | PCIe Root Complex / Host Bridge |
                     |             (PHB)               |
                     +--------+---------------+--------+
                              |               |
               PCIe 3.0 x16   |               |   PCIe 3.0 x16
               (15.75 GB/s)   |               |   (15.75 GB/s)
                              v               v
                     +--------+-------+ +-----+--------+
                     |  GPU 0 (Tesla) | | GPU 1 (Tesla)|
                     |  TU104 (sm_75) | | TU104 (sm_75)|
                     +----------------+ +--------------+
                             <----------------->
                        Peer-to-Peer (P2P) via PHB
                           Measured: 9.91 GB/s
```

### A. Bus Topology Classification
- **Connection Type**: PCIe 3.0 x16 per device.
- **Topology Path**: Host Bridge / PCI Root Port (`PHB`).
- **Absence of NVLink**: Tesla T4 cards do not possess physical NVLink fingers. All cross-device traffic travels across the server motherboard PCIe bus.
- **P2P Traversal**: Because both T4 cards sit behind the same Root Complex, the PCIe transaction layer permits Direct Memory Access (DMA) reads/writes across the Root Complex without round-tripping through system RAM.
- **Measured Bandwidth Efficiency**:
  - Theoretical single-direction PCIe 3.0 x16 bandwidth: $16 \times 985\text{ MB/s} = 15.75\text{ GB/s}$.
  - Real-world payload ceiling after 128b/130b encoding, packet headers, and flow control: $\approx 12.0\text{ GB/s}$.
  - Measured sustained P2P throughput: **9.90 – 9.91 GB/s** (83% of usable PCIe 3.0 payload ceiling).

---

## 3. Communication Latency Physics

In Tensor Parallelism (TP2), inter-GPU communication consists of **AllReduce Sums** ($10\text{ KiB}$ vectors) and **AllGather Logits** ($124\text{ KiB}$ vectors).

### A. Small-Payload Reduction Latency (10 KiB)
- In Qwen3.6-27B, the hidden dimension is $H = 5120$.
- Each intermediate reduction vector is $5120 \times 2\text{ bytes (BF16)} = 10,240\text{ bytes (10 KiB)}$.
- **Transfer Time over 9.91 GB/s PCIe**:
  $$t_{\text{transfer}} = \frac{10,240\text{ bytes}}{9.91 \times 10^9\text{ bytes/s}} \approx 1.03\text{ µs}$$
- **PCIe Packet Overhead & Driver Dispatch**: $\approx 1.5\text{ – }2.5\text{ µs}$.
- **Local Residual Combine Kernel**: $\approx 1.0\text{ µs}$.
- **Total Reduction Latency**: **~2.5 – 5.0 µs** per allreduce.

### B. Aggregate Communication Overhead per Token Step
- Each Transformer layer performs 2 allreduces (Attention output projection + MLP down projection).
- Across 64 layers:
  $$\text{Total Allreduces per Token} = 64 \times 2 = 128\text{ operations}$$
- Total communication wall-clock time:
  $$t_{\text{comm}} = 128 \times 3.5\text{ µs} \approx \mathbf{0.45\text{ ms (450 microseconds)}}$$
- **Conclusion**: On a 39.8 ms decode token step, communication accounts for **only 1.1% of execution time**. The system is virtually 99% compute and memory streaming bound.

---

## 4. Hardware P2P vs Fallback Transport Comparison

| Attribute | UVA Direct P2P (`pull_peer`) | Pinned-Host `PeerMailbox` | Driver Host-Staged Fallback |
|---|:---:|:---:|:---:|
| **Target Topology** | Direct P2P via PHB / NVLink | Restricted PCIe / WDDM | No P2P, standard CUDA |
| **API Call** | `cudaMemcpyAsync(DeviceToDevice)` | Kernel-side spinlock in mapped RAM | Driver-managed host staging |
| **10 KiB Latency** | **2.5 – 5.0 µs** | **~41 µs** | **~277 µs** |
| **Event Overhead** | Zero (dest-stream copy) | Zero (GPU-side polling) | High (event chain per copy) |
| **CUDA Graph Safe** | **Yes** (stream capturable) | **Yes** (kernel-based) | No (graph capture rejects) |
| **Role in Engine** | **Primary Active Transport** | **Embedded Fallback Transport** | Rejected |
