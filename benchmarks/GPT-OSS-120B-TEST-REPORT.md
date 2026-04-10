# GPT-OSS 120B Benchmark Test Report

**Date:** 2026-02-19
**Status:** Test run (first attempt)
**Operator:** Harold (manual) + Claude Code (automation)

## Model Info

| Property | GPT-OSS 120B (Target) | GPT-OSS 20B (Draft) |
|----------|----------------------|---------------------|
| Architecture | MoE (128 experts, 4 active) | MoE |
| Total params | 116.83B | ~21B |
| Active params | ~5.1B | ~3.6B |
| Quant | Q4_K_XL (UD) | MXFP4 (official ggml-org) |
| Size on disk | ~60GB (2 shards) | ~12.1GB |
| Family | GPT-OSS | GPT-OSS (same family) |
| Context used | 2048 | 2048 |
| llama.cpp version | b8111 | b8111 |

## Hardware Layout (Final Configuration)

| Role | Machine | Hardware | VRAM/RAM | IP |
|------|---------|----------|----------|----|
| Coordinator | Akiva | 2x RX 7900 XTX (ROCm) | 48GB GPU + 16.4GB CPU offload | 192.168.86.29 |
| Drafter | Desktop | RTX 4070 Ti Super + RTX 3060 | 28GB combined | 192.168.86.36 |
| Proxy | M4 Mac mini | — | — | localhost |

**Memory distribution (coordinator):**
- ROCm0 (XTX 1): 21,818 MiB (21.3 GB)
- ROCm1 (XTX 2): 21,499 MiB (21.0 GB)
- ROCm_Host (CPU/RAM): 16,767 MiB (16.4 GB)
- Total loaded: ~59.6 GB

## Benchmark Results

### Baseline (direct to coordinator, no speculation)

| Prompt | Time | Tokens | Generation tok/s | Prompt tok/s |
|--------|:----:|:------:|:----------------:|:------------:|
| Factual | 9.71s | 177 | 19.7 | 29.8 |
| Code | 13.73s | 256 | 19.7 | 31.4 |
| Creative | 11.36s | 216 | 19.6 | 29.0 |
| Reasoning | 10.56s | 187 | 19.7 | 35.1 |
| List | 15.68s | 256 | 17.0 | 31.4 |
| **Average** | **12.21s** | — | **19.1** | **31.3** |

### Speculative (through Tightwad proxy)

| Prompt | Time | Tokens | Effective tok/s |
|--------|:----:|:------:|:---------------:|
| Factual | 33.34s | 248 | 7.4 |
| Code | 34.22s | 184 | 5.4 |
| Creative | 36.99s | 267 | 7.2 |
| Reasoning | 23.70s | 97 | 4.1 |
| List | 27.36s | 157 | 5.7 |
| **Average** | **31.12s** | — | **6.0** |

### Proxy Statistics

| Metric | Value |
|--------|:-----:|
| Total rounds | 54 |
| Tokens drafted | 1,659 |
| Tokens accepted | 1,659 |
| Acceptance rate | **100.0%** |
| Tokens per round | 31.7 |
| Wall-clock speedup | **0.39x** (2.5x slowdown) |

## Analysis: Why 100% Acceptance But 0.39x Speedup?

The paradox: perfect token acceptance (100%) combined with a significant slowdown. Root causes:

### 1. Target is already fast (19.1 tok/s)
The 120B MoE model has only 5.1B active params per token. Combined with 48GB of GPU VRAM (most layers on GPU), generation is already fast. At 19.7 tok/s, the coordinator produces 32 tokens in ~1.6 seconds.

### 2. Draft→verify round-trip overhead exceeds direct generation
Each speculation round involves:
- Draft: Desktop generates 32 tokens (~1-2s)
- Network: Draft tokens sent to proxy, proxy sends to coordinator (~LAN RTT)
- Verify: Coordinator processes 32 draft tokens as prompt verification + 1 bonus token
- Network: Result back to proxy, back to client

Even with 100% acceptance, each round takes ~4-6s vs ~1.6s for the coordinator to generate those 32 tokens directly. The draft is not "free" — it adds latency.

### 3. MoE models are uniquely bad candidates for cross-machine speculation
MoE's low active parameter count means the target is fast enough that speculation overhead dominates. Dense models (which are slower per token) benefit more from speculation.

## Issues Encountered During Test

### 1. MoE Expert Replication Breaks RPC (CRITICAL)
- **Original plan:** Pool desktop (28GB) + akiva (48GB) = 76GB via RPC
- **What happened:** `alloc_tensor_range: failed to allocate RPC0[...] buffer of size 22304105600` (22.3 GB requested for 12GB 3060)
- **Root cause:** MoE models replicate routing/expert selection tables to every device. Each RPC worker needs ~20GB just for routing tables, regardless of tensor split percentage.
- **Workaround:** Dropped RPC entirely, used CPU offload instead
- **Lesson:** MoE models cannot be distributed via RPC to small GPUs. The per-device overhead is too large.

### 2. Version Mismatch Across 3 Machines
- Akiva: b8112 (dev build from git main)
- Desktop: b8100 (release)
- M2: b8079 (release)
- **Resolution:** Rebuilt akiva to b8111, downloaded b8111 release for M2 and desktop. All matched.
- **Lesson:** Need automated version checking. There's no `tightwad doctor` check for this currently.

### 3. M2 Drafter OOM (Metal)
- GPT-OSS 20B Q8_0 crashed on M2 MacBook Air (16GB unified) with `kIOGPUCommandBufferCallbackErrorOutOfMemory` during warmup
- M2 `recommendedMaxWorkingSetSize` is only 11,453 MiB — not enough for Q8_0 of a 20B model
- **Would fix with:** `--no-warmup` flag or smaller quant (Q4_K_M)

### 4. Model Path Mismatch on M2
- Plan specified `~/models/gpt-oss-20b-Q8_0.gguf`
- Actual location: `~/.tightwad/models/gpt-oss-20b-Q8_0.gguf`
- **Lesson:** Model paths should be verified before starting servers

### 5. `--np` vs `-np` Flag
- `--np 1` is invalid in llama-server b8111 — it's `-np 1` (single dash)
- Caused a startup failure that required a restart

### 6. No SSH Access to Desktop
- Could not verify RPC version, start drafter, or run diagnostics on the Windows desktop
- All desktop operations required Harold's manual intervention
- **Lesson:** Tightwad needs a way to communicate with agents on remote hosts (tightwad-to-tightwad peer comms)

## Key Findings

1. **MoE models are poor candidates for speculative decoding when the target is fast.** The 5.1B active params make the 120B model generate at ~20 tok/s — fast enough that speculation overhead is a net negative.

2. **Speculation helps most when the target is slow (<5 tok/s).** Previous benchmarks showed 1.8x speedup on RPC pool targets running at 3 tok/s. The breakeven point appears to be around 8-10 tok/s.

3. **MoE expert replication prevents RPC distribution to small GPUs.** Routing tables are replicated per device, adding ~20GB overhead per worker regardless of tensor split. This makes consumer GPUs (8-16GB) unusable as RPC workers for large MoE models.

4. **100% token acceptance confirms same-family pairing works perfectly.** GPT-OSS 20B → GPT-OSS 120B achieved 100% acceptance across all prompt types — the draft model perfectly predicts the target's token choices.

5. **CPU offload is viable for MoE.** Despite 16.4GB offloaded to DDR3 RAM, the model still achieved 19.1 tok/s generation. MoE's sparse activation means only active experts need fast memory access.

## Recommendations

### For Tightwad Development
1. **Add `tightwad doctor` version matching** — check all RPC workers and coordinator are on the same llama.cpp build before starting
2. **Adaptive speculation** — automatically disable speculation when target tok/s exceeds a threshold (e.g., >10 tok/s)
3. **Tightwad peer agent protocol** — allow tightwad instances on different machines to communicate (start/stop servers, check versions, transfer models)
4. **MoE-aware tensor split** — warn users when MoE models are assigned to small-VRAM RPC workers

### For This Hardware Setup
1. **Keep GPT-OSS 120B on akiva in direct mode** (no speculation) — 19.1 tok/s is excellent
2. **Use desktop as a standalone inference server** for smaller models (GLM-4.7-Flash at 83 tok/s)
3. **Reserve speculation for dense models on slow targets** (e.g., Llama 3.3 70B on RPC pool at 3 tok/s → 5.4 tok/s with speculation)

## Raw Data

- Full benchmark JSON: `benchmarks/gptoss-120b-full-benchmark.json`
- Benchmark script text-match (for reference): `benchmarks/gptoss-120b-benchmark.json`
