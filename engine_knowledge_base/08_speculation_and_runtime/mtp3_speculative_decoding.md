# MTP3 Speculative Decoding & Verification Protocol

> **Location**: `engine_knowledge_base/08_speculation_and_runtime/mtp3_speculative_decoding.md`  
> **Related Documents**: [Roofline & Performance Math](../02_hardware_and_physics/roofline_and_performance_math.md) | [Prefix Reuse & Serving](prefix_reuse_and_serving.md) | [Small-T Exact Kernels](../04_cuda_kernels_and_sm75_execution/small_t_exact_kernels.md)

This document details Multi-Token Prediction (MTP) speculative decoding, the verification forward loop, the draft acceptance predicate, and the proposal probability lookup fix in `src/ops/kernel/speculative_round.cuh` (Commit `350136a8`).

---

## 1. What Is Multi-Token Prediction (MTP)?

In standard autoregressive generation, a model predicts exactly one token per forward pass ($T = 1$), running bound to memory bandwidth.
- **Qwen3.6-27B MTP Architecture**: Embeds a lightweight specialized drafter structure:
  - `mtp/input_projection`
  - `mtp/layer/attention/...`
  - `mtp/layer/mlp/...`
  - `mtp/final_norm`
- With draft window $K = 3$ (`--spec mtp --draft-tokens 3`), the drafter proposes **3 candidate future tokens** ($d_1, d_2, d_3$) at negligible compute cost.
- The main target model then evaluates all $K+1 = 4$ positions ($T = 4$) **simultaneously in one parallel forward pass**.

---

## 2. The Verification Forward Loop & Acceptance Predicate

```
[Drafter Forward] ───────────────> Proposes 3 tokens: d1, d2, d3
                                               │
                                               v
[Target Model Forward (T = 4)] ───> Computes target probabilities p(x | context)
                                               │
                                               v
[Speculative Verification] ───────> Rejection Sampling Predicate per position i:
                                      Accept di if: u < p(di) / q(di)
                                      where u ~ Uniform(0, 1)
```

1. If $p(d_i) \ge q(d_i)$, token $d_i$ is accepted unconditionally.
2. If $p(d_i) < q(d_i)$, token $d_i$ is accepted with probability $p(d_i) / q(d_i)$.
3. On the first rejected token at position $j$, the target distribution is adjusted to resample a replacement token, and speculation halts for that round.
4. If all $K$ tokens are accepted, the target model emits an additional **bonus token**, yielding $K+1 = 4$ tokens in a single round.

---

## 3. The Proposal Lookup Miss Bug & Fix (Commit `350136a8`)

### A. The Defect in Upstream
In `src/ops/kernel/speculative_round.cuh`, candidate probabilities are stored in a sparse top-16 candidate support structure:
- If a drafted token $d$ was outside the top-16 support, the probability lookup function returned $q_d = 0.0\text{f}$.
- **The Upstream Acceptance Check**:
  ```cpp
  // DEFECTIVE UPSTREAM CODE:
  const float qd = speculative_sparse_probability(candidate_ids + at, proposal_q + at, d);
  const float u  = sampling_uniform(...);
  reject = !(pd >= qd || u * qd < pd);
  ```
- **The Disaster**:
  When $q_d = 0.0\text{f}$, the expression `u * qd < pd` became `0.0f < pd`.
  As long as target probability $p_d > 0$, `u * qd < pd` evaluated to **TRUE**!
  The negation `!(true)` meant `reject = false`!
- **Accidental Acceptance**: An invalid draft whose proposal probability was completely missing from the candidate list was **accepted unconditionally**, corrupting the output stream!

### B. The Applied Fix
In `src/ops/kernel/speculative_round.cuh`:
```cpp
// CORRECTED IMPLEMENTATION:
reject = qd <= 0.0f || !(pd >= qd || u * qd < pd);
```
If $q_d \le 0.0\text{f}$, the draft is **immediately and correctly rejected**, and the residual distribution correctly resamples from the unmodified target distribution.

---

## 4. Operational Commands for MTP3

To run MTP3 speculative decoding on dual Tesla T4:

### CLI:
```bash
./build/apps/ninfer /tmp/models/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --kv-dtype int8 --kv-capacity auto \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --prompt "Write a quicksort implementation in Python."
```

### HTTP Server:
```bash
./build/apps/ninfer-serve /tmp/models/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --kv-dtype int8 --kv-capacity auto \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --port 8080
```
- **Throughput**: Achieves **32 – 38 tok/s** on standard conversational prompts and **up to 44 tok/s** on code/structured prompts.
