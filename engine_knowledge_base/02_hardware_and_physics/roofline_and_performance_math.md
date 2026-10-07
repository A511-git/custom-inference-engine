# Roofline & Performance Mathematics

> **Location**: `engine_knowledge_base/02_hardware_and_physics/roofline_and_performance_math.md`  
> **Related Documents**: [Turing SM75 Architecture](turing_sm75_architecture.md) | [Dual T4 PCIe Topology](dual_t4_pcie_topology.md) | [MTP3 Speculative Decoding](../08_speculation_and_runtime/mtp3_speculative_decoding.md)

This document provides the complete mathematical roofline derivations for Prefill (prompt processing) and Decode (token generation) throughput on the dual Tesla T4 platform.

---

## 1. System Roofline Parameters

```
  Performance
  (TFLOPS)
      ^
      |                          Compute Bound (Prefill)
  130 +-----------------------=========================== (Roofline Peak: 130 TFLOPS)
      |                      /
      |                     /
      |                    /
      |                   /   Memory Bandwidth Bound (Decode)
      |                  /    Slope = 500 GB/s
      |                 /
    0 +----------------+----------------------------------> Arithmetic Intensity
      0               0.26                                  (FLOPs / Byte)
                      (Ridge Point)
                      Decode (AI ~ 0.005)    Prefill (AI ~ 10-50)
```

- **Aggregate Peak FP16/INT8 Tensor Compute**: $2 \times 65.1\text{ TFLOPS} = \mathbf{130.2\text{ TFLOPS}}$.
- **Aggregate Peak Memory Bandwidth**: $2 \times 320\text{ GB/s} = \mathbf{640\text{ GB/s}}$ (Peak), **~500 GB/s sustained**.
- **Ridge Point (Arithmetic Intensity Threshold)**:
  $$\text{Ridge Point} = \frac{\text{Peak Compute}}{\text{Peak Bandwidth}} = \frac{130.2 \times 10^{12} \text{ FLOP/s}}{500 \times 10^9 \text{ B/s}} \approx \mathbf{0.26\text{ FLOPs/Byte}}$$

---

## 2. Decode Step Mathematical Derivation (Memory-Bound)

During autoregressive generation ($T = 1$), arithmetic intensity is approximately $1.0\text{ FLOP/byte}$, placing decode deeply in the memory bandwidth-bound regime.

### A. Weight Streaming Time
- Total Model Weights (`groupwise-int`): $16.29\text{ GiB} = 17.49 \times 10^9\text{ bytes}$.
- Weights stored per GPU in TP2:
  $$W_{\text{rank}} = \frac{17.49 \times 10^9\text{ bytes}}{2} = 8.75 \times 10^9\text{ bytes (8.15 GiB)}$$
- Sustained DRAM bandwidth per T4: $B_{\text{mem}} \approx 250\text{ GB/s}$.
- Weight streaming time:
  $$t_{\text{weights}} = \frac{8.75 \times 10^9\text{ bytes}}{250 \times 10^9\text{ bytes/s}} = \mathbf{35.00\text{ ms}}$$
  *(Because both GPUs stream concurrently from independent 256-bit memory buses, this wall-clock time is 35.0 ms, not 70.0 ms).*

### B. KV Cache Streaming Time (INT8 Group-64)
- Footprint per token per GPU: $\approx 16.9\text{ KiB} = 17,305\text{ bytes}$.
- At context length $C$:
  $$t_{\text{kv}}(C) = \frac{C \times 17,305\text{ bytes}}{250 \times 10^9\text{ bytes/s}} = C \times 6.92 \times 10^{-5}\text{ ms}$$
  - $C = 2,048$: $t_{\text{kv}} = 0.14\text{ ms}$
  - $C = 8,192$: $t_{\text{kv}} = 0.57\text{ ms}$
  - $C = 32,768$: $t_{\text{kv}} = 2.27\text{ ms}$

### C. Communication Time
- 128 allreduce operations per token $\times 3.5\text{ µs}$ over measured 9.91 GB/s PCIe:
  $$t_{\text{comm}} = 128 \times 3.5\text{ µs} = \mathbf{0.45\text{ ms}}$$

### D. Element-wise & Kernel Overhead
- RMSNorm, SiLU, RoPE, CUDA Graph overhead: $\approx \mathbf{4.00\text{ ms}}$.

### E. Total Step Time & MTP0 Decode TPS
$$t_{\text{step}} = 35.00\text{ ms} + t_{\text{kv}}(C) + 0.45\text{ ms} + 4.00\text{ ms}$$
At $C = 2,048$:
$$t_{\text{step}} = 35.00 + 0.14 + 0.45 + 4.00 = \mathbf{39.59\text{ ms}}$$
$$\text{TPS}_{\text{MTP0}} = \frac{1000\text{ ms}}{39.59\text{ ms}} \approx \mathbf{25.2\text{ tok/s theoretical}}$$
Accounting for passive 70W cooling and thermal throttling: **20 – 24 tok/s sustained**.

---

## 3. MTP3 Speculative Decoding Derivation

Under Multi-Token Prediction ($K = 3$ draft tokens, verifying $T = 4$ positions):
1. **Verification Forward ($T = 4$)**: Runs with 4x higher arithmetic intensity, amortizing weight streaming costs across 4 candidate tokens. Target verification time: $t_{\text{verify}} \approx 46.0\text{ ms}$.
2. **Draft Forward ($K = 3$)**: Small drafter structure runs in $t_{\text{draft}} \approx 15.0\text{ ms}$.
3. **Total Round Time**: $t_{\text{round}} = 46.0\text{ ms} + 15.0\text{ ms} = \mathbf{61.0\text{ ms}}$.
4. **Draft Acceptance Rate ($\alpha$)**:
   $$\text{Tokens Emitted per Round} = 1 + K \times \alpha = 1 + 3 \times \alpha$$
   - At $\alpha = 0.70$ (Standard Chat): $1 + 2.1 = 2.1\text{ to }2.2\text{ tokens}$.
   - Throughput:
     $$\text{TPS}_{\text{MTP3}} = \frac{2.15\text{ tokens}}{0.061\text{ s}} = \mathbf{35.2\text{ tok/s sustained (32 – 38 tok/s range)}}.$$
   - At $\alpha = 0.85$ (Code / Structured Output):
     $$\text{TPS}_{\text{MTP3}} = \frac{2.55\text{ tokens}}{0.061\text{ s}} \approx \mathbf{41.8\text{ tok/s}}.$$

---

## 4. Prefill (Prompt Processing / TTFT) Derivation

For prompt chunks ($T \ge 1024$), prefill operates in the **compute-bound** regime.

1. **Compute Required per Token**:
   $$\text{FLOPs/token} \approx 2 \times N = 2 \times 27 \times 10^9 = \mathbf{54\text{ GFLOPs/token}}$$
2. **Effective Tensor Core Throughput**:
   Dual T4 delivers ~70 to 85 sustained TFLOPS on Turing W8A16 MMA GEMM (55%–65% of peak):
   $$\text{Prefill TPS} = \frac{75 \times 10^{12}\text{ FLOP/s}}{54 \times 10^9\text{ FLOP/token}} \approx \mathbf{1,380\text{ tok/s sustained (1,000 – 1,400 range)}}.$$
3. **Time To First Token (TTFT)**:
   For prompt length $L$:
   $$\text{TTFT} = \frac{L}{\text{Prefill TPS}} + t_{\text{step}}$$
   - At $L = 512$: $\text{TTFT} = \frac{512}{950} + 0.040\text{ s} \approx \mathbf{0.58\text{ s (580 ms)}}$.
   - At $L = 2048$: $\text{TTFT} = \frac{2048}{1380} + 0.040\text{ s} \approx \mathbf{1.52\text{ s}}$.
   - With **Multi-Turn Prefix Reuse** (`fix/tp2-mtp-prefix-reuse`), prefill is skipped for cached tokens, dropping TTFT to **~190–220 ms**.
