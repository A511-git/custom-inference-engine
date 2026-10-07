# Layer Sharding Matrix: Qwen3.6-27B on TP2

> **Location**: `engine_knowledge_base/03_tensor_parallelism_sharding/layer_sharding_matrix.md`  
> **Related Documents**: [TP2 Architecture Overview](tp2_architecture_overview.md) | [Q4/Q5 Shard Geometry](q4_q5_shard_geometry.md) | [Small-T Exact Kernels](../04_cuda_kernels_and_sm75_execution/small_t_exact_kernels.md)

This document contains the complete, parameter-by-parameter tensor sharding reference table for **Qwen3.6-27B `groupwise-int`**. Every layer, parent tensor, quantization format, split axis, and rank-local dimension is explicitly cataloged.

---

## 1. Master Tensor Sharding Reference Table

| Module & Operation | Logical Binding Name | Stored Quant Format | Global Dimensions $[N, K]$ | Split Axis | Rank 0 Dimensions | Rank 1 Dimensions | Communication Operation |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Token Embedding** | `model.embed_tokens` | BF16 / INT8 | $[248320, 5120]$ | None (Replicated) | $[248320, 5120]$ | $[248320, 5120]$ | None |
| **Input RMSNorm** | `model.layers[i].input_layernorm` | FP32 | $[5120]$ | None (Replicated) | $[5120]$ | $[5120]$ | None |
| **Attention Query/Key** | `model.layers[i].self_attn.query_key` | `q4_g64_fp16` | $[7168, 5120]$ | Output Rows ($N$) | $[3584, 5120]$ | $[3584, 5120]$ | None (Column-Parallel) |
| **Attention Value/Gate** | `model.layers[i].self_attn.value_gate` | `q5_g64_fp16` | $[7168, 5120]$ | Output Rows ($N$) | $[3584, 5120]$ | $[3584, 5120]$ | None (Column-Parallel) |
| **Attention Core & KV** | Paged KV Cache Pool | `int8` | 40 Q / 8 KV heads | Head-Local | 20 Q / 4 KV heads | 20 Q / 4 KV heads | None (50% VRAM saved) |
| **Attention Output** | `model.layers[i].self_attn.o_proj` | `q5_g64_fp16` | $[5120, 5120]$ | Input Cols ($K$) | $[5120, 2560]$ | $[5120, 2560]$ | **AllReduce Sum (10 KiB)** |
| **Post-Attn RMSNorm** | `model.layers[i].post_attention_layernorm`| FP32 | $[5120]$ | None (Replicated) | $[5120]$ | $[5120]$ | None |
| **MLP Gate / Up** | `model.layers[i].mlp.gate_up_proj` | `q4_g64_fp16` | $[34816, 5120]$ | Output Rows ($N$) | $[17408, 5120]$ | $[17408, 5120]$ | None (Column-Parallel) |
| **MLP SwiGLU** | Activation Element-wise | FP16 / BF16 | $[17408]$ | Local features | $[8704]$ features | $[8704]$ features | None |
| **MLP Down** | `model.layers[i].mlp.down_proj` | `q5_g64_fp16` | $[5120, 17408]$ | Input Cols ($K$) | $[5120, 8704]$ | $[5120, 8704]$ | **AllReduce Sum (10 KiB)** |
| **GDN Query / Key** | `model.layers[i].gdn.query_key` | `q4_g64_fp16` | $[4096, 5120]$ | Output Rows ($N$) | $[2048, 5120]$ | $[2048, 5120]$ | None (Column-Parallel) |
| **GDN Value / Z** | `model.layers[i].gdn.value_z` | `q5_g64_fp16` | $[12288, 5120]$ | Output Rows ($N$) | $[6144, 5120]$ | $[6144, 5120]$ | None (Column-Parallel) |
| **GDN Recurrent State** | `gated_delta_rule_recurrent` | BF16 state | $D = 128$ | Head-Local | 2048 recurrent | 2048 recurrent | None |
| **GDN Output** | `model.layers[i].gdn.out_proj` | `q5_g64_fp16` | $[5120, 6144]$ | Input Cols ($K$) | $[5120, 3072]$ | $[5120, 3072]$ | **AllReduce Sum (10 KiB)** |
| **Final RMSNorm** | `model.norm` | FP32 | $[5120]$ | None (Replicated) | $[5120]$ | $[5120]$ | None |
| **LM Output Head** | `model.output_head` | `q6_g64_fp16` | $[248320, 5120]$ | Vocab Rows ($N$) | $[124160, 5120]$ | $[124160, 5120]$ | **AllGather Rows / Top-K** |

---

## 2. Attention Sharding In-Depth

In Qwen3.6-27B, the attention projections use fused parent weights:
- **`query_key` Parent**: Physically stores $Q$ ($5120$ rows) and $K$ ($2048$ rows) interleaved:
  $$\text{Global Shape} = [7168, 5120]$$
  - At TP2, Rank 0 takes rows $0 \dots 2559$ ($Q_0$) and $5120 \dots 6143$ ($K_0$) $\implies 3584$ rows total.
  - Rank 1 takes rows $2560 \dots 5119$ ($Q_1$) and $6144 \dots 7167$ ($K_1$) $\implies 3584$ rows total.
- **`value_gate` Parent**: Physically stores $\text{Gate}$ ($5120$ rows) and $V$ ($2048$ rows):
  $$\text{Global Shape} = [7168, 5120] \implies \text{Rank 0: } 3584 \text{ rows}, \text{ Rank 1: } 3584 \text{ rows}$$
- **Head Decomposition**:
  - Global: 40 Query heads ($40 \times 128 = 5120$), 8 KV heads ($8 \times 128 = 1024$).
  - Per GPU: **20 Query heads**, **4 KV heads**.
  - Head ratio remains $20 / 4 = 5$ on each card.

---

## 3. Gated Delta Net (GDN) Sharding In-Depth

GDN is a linear recurrent attention mechanism. In Qwen3.6-27B:
- **Input Projections**:
  - `query_key`: $[4096, 5120] \implies$ sliced to $[2048, 5120]$ per GPU.
  - `value_z`: $[12288, 5120] \implies$ sliced to $[6144, 5120]$ per GPU.
- **Head-Local Recurrence**:
  - The chunked delta rule evaluates recurrent state transitions independently per head.
  - Slicing input heads by 50% guarantees that each GPU updates only its local recurrent state slices without any cross-device communication inside the sequence loop.
- **Output Contraction**:
  - `out_proj`: $[5120, 6144] \implies$ sliced along input contraction dimension to $[5120, 3072]$ per GPU.
  - Evaluated via local GEMM followed by **AllReduce Sum**.
