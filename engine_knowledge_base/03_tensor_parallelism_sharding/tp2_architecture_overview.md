# Tensor Parallelism (TP2) Architecture Overview

> **Location**: `engine_knowledge_base/03_tensor_parallelism_sharding/tp2_architecture_overview.md`  
> **Related Documents**: [Layer Sharding Matrix](layer_sharding_matrix.md) | [Q4/Q5 Shard Geometry](q4_q5_shard_geometry.md) | [UVA Direct P2P Transport](../05_collectives_and_transport/uva_direct_p2p_transport.md)

This document explains the mathematical theory, architectural benefits, and operational mechanics of 2-way Tensor Parallelism (TP2) implemented in our custom inference engine, contrasting it with Pipeline Parallelism.

---

## 1. Why Tensor Parallelism (TP2) Instead of Pipeline Parallelism (PP2)?

| Parallelism Strategy | Execution Pattern | Memory Distribution | Bubble Overhead | Suitability for Dual T4 |
|---|---|---|:---:|:---:|
| **Pipeline Parallelism (PP2)** | GPU 0 runs Layers 1–32; passes activation to GPU 1 for Layers 33–64. | Weights halved per GPU, but KV cache remains full on active card. | **50% idle bubbles** during single-request decode. | **Poor**: 1 GPU sits idle while the other computes, destroying latency. |
| **Tensor Parallelism (TP2)** | **Both GPUs execute all 64 layers concurrently**, splitting matrix widths and attention heads. | **Both weights and KV cache are halved** on each GPU. | **0% bubbles**: 100% active utilization on both GPUs. | **Optimal**: Maximizes aggregate memory bandwidth (500 GB/s) and cuts latency. |

---

## 2. Megatron-Style Intra-Layer Sharding Theory

Tensor Parallelism shards individual linear operations $Y = X W^T$ across devices. In Transformer blocks, linear layers occur in pairs (Projection $\to$ Contraction), which allows interleaving **Column-Parallel** and **Row-Parallel** operations to minimize communication.

```
                            [Input Activation X: 5120]
                                      |
                     +----------------+----------------+
                     | (Replicated Broadcast)           |
                     v                                 v
           [GPU 0: First Half Rows]          [GPU 1: Second Half Rows]
             W0: [17408 x 5120]                W1: [17408 x 5120]
                     |                                 |
                     v                                 v
            Y0 = X * W0^T                     Y1 = X * W1^T
             [17408 features]                  [17408 features]
                     |                                 |
                     +----------------+----------------+
                                      | (No communication needed!)
                                      | (SwiGLU activation applied locally)
                                      v
                     +----------------+----------------+
                     |                                 |
           [GPU 0: Down Projection]          [GPU 1: Down Projection]
             W_down0: [5120 x 8704]            W_down1: [5120 x 8704]
                     |                                 |
                     v                                 v
               Partial Sum Y0                    Partial Sum Y1
               [5120 features]                   [5120 features]
                     |                                 |
                     +----------------+----------------+
                                      |
                       <====== ALLREDUCE SUM ======> (Hardware PCIe P2P, 3.5 µs)
                                      |
                           [Final Output: 5120]
```

### A. Column-Parallel (Output-Row Split — Zero Communication)
In a column-parallel layer, the weight matrix rows $N$ are sliced in half across ranks:
$$W = \begin{bmatrix} W_0 \\ W_1 \end{bmatrix}, \quad W_0 \in \mathbb{R}^{\frac{N}{2} \times K}, \quad W_1 \in \mathbb{R}^{\frac{N}{2} \times K}$$
Both GPUs take the identical input vector $X \in \mathbb{R}^{1 \times K}$ and compute locally:
$$Y_0 = X W_0^T \in \mathbb{R}^{1 \times \frac{N}{2}}, \quad Y_1 = X W_1^T \in \mathbb{R}^{1 \times \frac{N}{2}}$$
**Result**:
$$Y = \begin{bmatrix} Y_0 & Y_1 \end{bmatrix} \in \mathbb{R}^{1 \times N}$$
- **Zero cross-device communication is required**.
- Applied to: Attention input projection (Q, K, V), MLP `gate_up` projection, GDN input projection.

### B. Row-Parallel (Input-Column / Contraction Split + AllReduce Sum)
In a row-parallel layer, the contraction dimension $K$ is sliced in half:
$$W = \begin{bmatrix} W_0 & W_1 \end{bmatrix}, \quad W_0 \in \mathbb{R}^{M \times \frac{K}{2}}, \quad W_1 \in \mathbb{R}^{M \times \frac{K}{2}}$$
Because the previous column-parallel layer produced $Y_0$ and $Y_1$, each GPU already holds its corresponding half-activation:
$$X = \begin{bmatrix} X_0 & X_1 \end{bmatrix}, \quad X_0 \in \mathbb{R}^{1 \times \frac{K}{2}}, \quad X_1 \in \mathbb{R}^{1 \times \frac{K}{2}}$$
Each GPU computes a partial contraction sum locally:
$$Z_0 = X_0 W_0^T \in \mathbb{R}^{1 \times M}, \quad Z_1 = X_1 W_1^T \in \mathbb{R}^{1 \times M}$$
The full output requires the mathematical sum:
$$Z = Z_0 + Z_1 = X_0 W_0^T + X_1 W_1^T$$
- **AllReduce Sum**: Both GPUs exchange partial vectors $Z_0$ and $Z_1$ over PCIe P2P in **~3.5 µs**, computing the sum in-place so both ranks exit with the identical output vector $Z$.
- Applied to: Attention output projection (`o_proj`), MLP `down_proj`, GDN output projection.

---

## 3. Communication Cost Breakdown per Transformer Block

In every Transformer layer of Qwen3.6-27B:
1. **Self-Attention Sub-Block**:
   - Q/K/V Projections: Column-Parallel $\implies$ **0 communication**.
   - Attention Core: Head-Local $\implies$ **0 communication**.
   - Output Projection: Row-Parallel $\implies$ **1 AllReduce Sum (10 KiB)**.
2. **MLP Sub-Block**:
   - `gate_up` Projection: Column-Parallel $\implies$ **0 communication**.
   - SwiGLU Activation: Local Element-wise $\implies$ **0 communication**.
   - `down_proj` Projection: Row-Parallel $\implies$ **1 AllReduce Sum (10 KiB)**.

**Total Collective Operations per Layer**: Exactly **2 AllReduce Sums**.  
Across 64 layers: $128 \text{ AllReduces} \times 3.5\text{ µs} \approx \mathbf{0.45\text{ ms}}$, leaving **98.9% of GPU time dedicated entirely to compute and memory streaming**.
