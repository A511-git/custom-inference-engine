# Multi-Turn Prefix Reuse & HTTP Serving

> **Location**: `engine_knowledge_base/08_speculation_and_runtime/prefix_reuse_and_serving.md`  
> **Related Documents**: [MTP3 Speculative Decoding](mtp3_speculative_decoding.md) | [Roofline & Performance Math](../02_hardware_and_physics/roofline_and_performance_math.md) | [16GB VRAM Budgeting](../06_memory_and_kv_cache/16gb_vram_budgeting.md)

This document details the multi-turn conversation prefix reuse optimization (Commit `eab0a929` on branch `fix/tp2-mtp-prefix-reuse`), which unlocks a 4.7x speedup in Time to First Token (TTFT), and documents the production HTTP server API.

---

## 1. The Multi-Turn Prefix Problem in TP2

In chat assistants and agentic coding loops, users converse across multiple back-and-forth turns.
- In Turn 1, the user sends a system prompt + user question (e.g. 500 tokens).
- In Turn 2, the client re-sends the history (system prompt + turn 1 answer) + turn 2 question (e.g. 950 tokens).
- **The Upstream TP2 Defect**:
  In early TP2 trees, MTP speculative state and GDN recurrent states reset on every new request. The engine performed a **Full Reset**, re-computing prefill over all 950 tokens from scratch every turn!
  - Turn 2 prefill took **~905 ms**.
  - Turn 3 prefill took **~957 ms**.

---

## 2. The Solution: Checkpoint Mirroring & Retained Hidden State (Commit `eab0a929`)

Author Dmitriy Ivanov resolved this by introducing per-rank retained state mirroring across devices:
1. **`TurnClosure` Rewrite Checkpoint**:
   - At the conclusion of each generation turn, the engine commits a checkpoint capturing the resident KV cache frontier, the MTP drafter state, and the GDN recurrent state.
2. **Mirroring `rewrite_checkpoint_hidden`**:
   - The hidden state vector at the checkpoint frontier is mirrored across devices:
     ```cpp
     // Rank 1 resumes from its own bit-identical, all-reduced copy:
     peer->io.rewrite_checkpoint_hidden = local_hidden;
     ```
3. **Planner Logic**:
   - When the next prompt arrives, the request planner computes the Longest Common Prefix (LCP) against the resident `TurnClosure` frontier.
   - If identical, the planner selects `RestoreTurnCheckpoint`.
   - The engine skips prefill for the cached 850+ tokens and computes **only the new appended tokens**!

### Measured Empirical Impact

| Conversation Turn | Cold Baseline (Full Reset) | With Prefix Reuse (This Fix) | TTFT Latency Reduction |
|---|:---:|:---:|:---:|
| **Round 1 (Cold Prompt)** | 891 tokens prefilled (~905 ms) | 891 tokens prefilled (~905 ms) | Baseline |
| **Round 2 (Follow-up Turn)** | 891 tokens prefilled (~905 ms) | **38 tokens computed (193 ms)** | **4.7x faster TTFT** |
| **Round 3 (Multi-turn Chat)** | 933 tokens prefilled (~957 ms) | **46 tokens computed (210 ms)** | **4.5x faster TTFT** |

---

## 3. OpenAI & Anthropic Compatible HTTP Serving

The engine provides an embedded HTTP server (`ninfer-serve`) listening on port 8080:

### A. Starting the Server
```bash
./build/apps/ninfer-serve /tmp/models/qwen3_6_27b.ninfer \
  --tp 2 --devices 0,1 \
  --port 8080 \
  --kv-dtype int8 --kv-capacity auto \
  --max-context 32768 \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --reasoning-effort medium
```

### B. Standard OpenAI Chat Completion Request
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_6_27b",
    "messages": [
      {"role": "system", "content": "You are a concise AI assistant."},
      {"role": "user", "content": "Explain virtual memory in two sentences."}
    ],
    "temperature": 0.0,
    "max_tokens": 128
  }'
```

### C. Server Response Format
Returns standard OpenAI JSON format containing:
- `choices[0].message.content`: Generated response text.
- `choices[0].finish_reason`: `stop` or `length`.
- `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`.
- Supports streaming SSE (`"stream": true`) with token-by-token delta events.
