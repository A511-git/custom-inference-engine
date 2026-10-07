# Agent Session Context & Master Guide: Custom Inference Engine (SM75 TP2)

> **FOR FUTURE AGENTS / SESSIONS:**  
> **READ THIS DOCUMENT FIRST BEFORE EXECUTING ANY WORK.**  
> Do **NOT** re-scan the web or re-clone repositories. All source repositories, architectural audits, hardware measurements, and bug analyses are organized right here.

---

## 1. Project Goal & Target Architecture

- **Primary Goal:** Run **Qwen3.6-27B** (or Qwen3.8-27B) `groupwise-int` (W8A16) across **2× NVIDIA Tesla T4 GPUs** using Tensor Parallelism (`--tp 2 --devices 0,1`).
- **Target Hardware:** 2× Tesla T4 (Turing `sm_75`, 15.0 GiB VRAM per GPU).
- **Environment Distinction:**
  - **Local Host (Current Windows Machine):** Development, code inspection, repository merging, patch creation. Does NOT have the physical T4 cards.
  - **Target Execution Host (Kaggle Linux):** 2× Tesla T4, CUDA 12.8, PHB P2P supported @ 9.18 GB/s, practical ~100 GB `/tmp` storage. Full hardware spec in [`kaggle_2xT4_complete_hardware_report.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/kaggle_2xT4_complete_hardware_report.md).
- **Execution Target:**
  - Weights: ~8.97 GiB / GPU
  - KV Cache: INT8 group-64 (~16.9 KiB / token / GPU)
  - VRAM Headroom: ~4.05 GiB / GPU for KV pool $\rightarrow$ supports up to 240,000 tokens context.

---

## 2. Directory Layout & Cloned Repositories

All reference repositories have been organized inside [`cloned_repos/`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos), while your dedicated tailored codebase lives in [`ninfer-t4-tp2/`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/ninfer-t4-tp2):

| Directory | Upstream / Fork | Status / Key Contribution |
| :--- | :--- | :--- |
| [`ninfer-t4-tp2/`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/ninfer-t4-tp2) | **Tailored Production Engine** | Dedicated codebase with all SM75 TP2 Turing kernels, NaN bug fix, direct-I/O 4KB alignment, PeerMailbox, and Kaggle deploy scripts. |
| [`cloned_repos/ninfer-2080ti-22g-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2) | `zsq13767593046-bit` | Upstream SM75 TP2 base. Contains Q4/Q5 groupwise-int shard routes, small-T exact kernels, and batched 2D materializer. |
| [`cloned_repos/ninfer-2080ti-22g`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g) | `mr-september` | Turing SM75 single-GPU baseline (W8 GEMM, Split-K, GDN MmaUnsplit). |
| [`cloned_repos/ninfer-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2) | `ValerioDolci` | Mature TP2 execution framework, CUDA Graph decode, concurrent serving, and Pinned-Host PeerMailbox. |
| [`cloned_repos/ninfer-dflash2-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-dflash2-tp2) | `parallelno` | Per-device `kernel_attr_once.h` memoization and materializer destination bounds checks. |
| [`cloned_repos/ninfer-windows-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-windows-tp2) | `ivanov84` | Pinned-host mailbox reference implementation. |
| [`cloned_repos/ninfer-tp2-1m`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2-1m) | `wamansou` | Original TP2 pull collective design & YaRN 1M context. |
| [`cloned_repos/ninfer-upstream`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-upstream) | `Neroued` | Upstream single-GPU Blackwell-oriented engine. |

---

## 3. Critical Discoveries from PR & Issue Trackers

1. **PR #6 in `mr-september/ninfer-2080ti-22g` (`zsq13767593046-bit`):**
   - Implemented the missing piece: Q4/Q5 attention and GDN input projection TP2 sharding routes.
   - Code location: [`src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp#L114-L143) and [`src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp#L81-L140).
   - Speedup: TP2 decode rose from 10.4 tok/s to 27.5 tok/s on Turing hardware.
2. **Issue #3 in `mr-september/ninfer-2080ti-22g` (Davis-Liang):**
   - **Bug:** In [`src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/linear_pair/w8/w8_pair_gemm_splitk.cu#L205-L212), tokens $T \in [161, 192]$ route to `DualSplitKMediumC192` which instantiates the 160-column tile, silently dropping 32 columns and writing NaNs!
   - **Fix:** In [`src/ops/linear_pair/w8/w8_pair_plan.cpp`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2/src/ops/linear_pair/w8/w8_pair_plan.cpp#L42), route window `161..192` on SM75 directly to `ConcatMmaR32C64`.
3. **PR #5 / Issue #4 in `mr-september/ninfer-2080ti-22g` (agorevski / xausky):**
   - **Bug:** Emulated BF16 decode-attention instantiates 96 templates with warp shuffles, generating a 361 MB PTX module that hangs `ptxas` for >6.5 hours.
   - **Fix:** Compile with `-DNINFER_SM75_INT8_KV_ONLY=ON` to eliminate emulated BF16 decode instantiations for lean INT8 builds.
4. **Pinned-Host Mailbox in `ValerioDolci/ninfer-tp2`:**
   - Provides GPU-side polling in pinned memory, dropping 10 KiB reduction latency from ~277 µs down to ~41 µs on non-NVLink PCIe topologies.
   - Files: [`include/ninfer/ops/peer_mailbox.h`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2/include/ninfer/ops/peer_mailbox.h), [`src/ops/common/peer_mailbox.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2/src/ops/common/peer_mailbox.cu), [`tools/tp2/mailbox_probe.cu`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-tp2/tools/tp2/mailbox_probe.cu).

---

## 4. Master Patch & Merge Plan

Our target engine builds on [`cloned_repos/ninfer-2080ti-22g-tp2`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/cloned_repos/ninfer-2080ti-22g-tp2) with four specific additions:

1. **Apply Issue #3 Fix:**
   Patch `src/ops/linear_pair/w8/w8_pair_plan.cpp` so $T \in [161, 192]$ uses `ConcatMmaR32C64` on SM75.
2. **Apply PR #5 Build Option:**
   Add `-DNINFER_SM75_INT8_KV_ONLY=ON` to `CMakeLists.txt` and `gqa_attention_decode.cu`.
3. **Integrate Pinned-Host PeerMailbox:**
   Copy `peer_mailbox.h`, `peer_mailbox.cu`, `peer_exchange.cuh`, and `mailbox_probe.cu` from `cloned_repos/ninfer-tp2` into the tree, enabling dual-path transport (UVA peer copies + Mailbox fallback).
4. **Configure for Kaggle 2× T4:**
   Follow [`HARDWARE_AND_ENVIRONMENT_GUIDE.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/HARDWARE_AND_ENVIRONMENT_GUIDE.md) to build and deploy.

---

## 5. Companion Documents

- Change impact, cascading risk, & verification matrix: [`engine_knowledge_base/01_provenance_and_lineage/change_impact_and_verification_matrix.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/engine_knowledge_base/01_provenance_and_lineage/change_impact_and_verification_matrix.md)
- Performance & Roofline throughput calculation: [`PERFORMANCE_AND_ROOFLINE_MODEL_2XT4.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/PERFORMANCE_AND_ROOFLINE_MODEL_2XT4.md)
- Exhaustive issue and PR audit report: [`ISSUES_AND_PR_AUDIT_REPORT.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/ISSUES_AND_PR_AUDIT_REPORT.md)
- Complete deployment & runbook for Kaggle: [`DEPLOYMENT_AND_RUNBOOK_KAGGLE_2XT4.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/DEPLOYMENT_AND_RUNBOOK_KAGGLE_2XT4.md)
- Detailed architectural audit & layer mapping: [`ARCHITECTURAL_AUDIT_SM75_TP2.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/ARCHITECTURAL_AUDIT_SM75_TP2.md)
- Deployment & hardware specifications: [`HARDWARE_AND_ENVIRONMENT_GUIDE.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/HARDWARE_AND_ENVIRONMENT_GUIDE.md)
- Raw Kaggle benchmark: [`kaggle_2xT4_complete_hardware_report.md`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/kaggle_2xT4_complete_hardware_report.md)
- Machine summary: [`Machine.MD`](file:///c:/Users/A-511/OneDrive/Desktop/projects/Custom%20infrence%20Engine/Machine.MD)
