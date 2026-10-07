# 16GB VRAM Budgeting & Sizing on Tesla T4

> **Location**: `engine_knowledge_base/06_memory_and_kv_cache/16gb_vram_budgeting.md`  
> **Related Documents**: [INT8 Group-64 KV Cache](int8_group64_kv_cache.md) | [Roofline & Performance Math](../02_hardware_and_physics/roofline_and_performance_math.md) | [Turing SM75 Architecture](../02_hardware_and_physics/turing_sm75_architecture.md)

This document provides the byte-level memory budget for running Qwen3.6-27B across 16 GB Tesla T4 GPUs, explaining how available VRAM is divided between weights, KV cache, draft structures, and CUDA runtime reserves.

---

## 1. Physical VRAM Budget per Tesla T4 (15.0 GiB Usable)

On Linux under the NVIDIA proprietary driver, a 16,384 MiB physical card exposes approximately **15,360 MiB (15.00 GiB)** of addressable user memory after driver context initialization and display/system reservations.

```
Total Usable VRAM: 15,360 MiB (15.00 GiB)
+-------------------------------------------------------------------------------+
| Sharded Weights  | MTP Draft | CUDA Workspaces | Paged INT8 KV Pool | Headroom|
|    8,340 MiB     |  800 MiB  |     650 MiB     |     4,570 MiB      | 1000 MiB|
|    (8.15 GiB)    | (0.78 GiB)|    (0.63 GiB)   |     (4.46 GiB)     | (0.98 G)|
+-------------------------------------------------------------------------------+
 0%              54%         60%               64%                  93%      100%
```

---

## 2. Granular Memory Allocation Breakdown

| Component | Size (MiB) | Size (GiB) | Fraction of VRAM | Description |
|---|:---:|:---:|:---:|---|
| **Sharded Linear Weights** | 8,340 MiB | 8.15 GiB | 54.3% | Half of the 16.29 GiB Q4/Q5/Q6 weight artifact |
| **Replicated Layers (Embedding/Norms)** | 620 MiB | 0.61 GiB | 4.0% | Replicated token embedding rows & RMSNorm scales |
| **MTP Drafter Weights & State** | 800 MiB | 0.78 GiB | 5.2% | Multi-Token Prediction layer weights and draft heads |
| **CUDA Runtime & Scratch Arenas** | 650 MiB | 0.63 GiB | 4.2% | Split-K atomic buffers, GDN recurrent scratch, graphs |
| **Safety Headroom & OS Buffer** | 1,000 MiB | 0.98 GiB | 6.5% | Reserved headroom preventing out-of-memory aborts |
| **Paged INT8 KV Cache Pool** | **4,950 MiB** | **4.83 GiB** | **32.2%** | **Dedicated pool for active conversation context** |
| **TOTAL ALLOCATED** | **15,360 MiB** | **15.00 GiB** | **100.0%** | Full addressable VRAM envelope |

---

## 3. Context Capacity Sizing (`--kv-capacity auto`)

The paged KV cache pool uses **66.0 KiB per token per GPU** (in INT8 group-64 mode with 4 KV heads per GPU):

$$\text{KV Memory per 1,000 Tokens} = 1000 \times 66.0\text{ KiB} \approx 64.5\text{ MiB}$$

With **4,950 MiB** allocated to the paged KV pool:

$$\text{Maximum Concurrent Tokens} = \frac{4,950 \times 1,024\text{ KiB}}{66.0\text{ KiB/token}} \approx \mathbf{76,800\text{ tokens}}$$

### Context Ceiling Recommendations:
- **Single Request Serving (`--max-concurrency 1`)**:
  - `--max-context 32768`: Consumes **2,112 MiB** of KV memory (leaving ample ~2.8 GiB safety margin).
  - `--max-context 65536`: Consumes **4,224 MiB** of KV memory (supported, comfortable fit).
- **Multi-Tenant Serving (`--max-concurrency 4`)**:
  - Each request at 8,192 tokens: $4 \times 8,192 = 32,768 \text{ tokens total} \implies \mathbf{2,112\text{ MiB}}$ allocated.
- **Why BF16 KV Fails on 16GB Cards**:
  BF16 KV requires **128.0 KiB per token**. At 65,536 tokens, BF16 would require **8,192 MiB** of KV memory—which exceeds available headroom by >3 GiB, triggering instant `cudaErrorMemoryAllocation`!
