# Tightwad — Agent Setup Guide

> If you're an AI agent helping someone set up Tightwad, this is your playbook.
> Skip the marketing. Here's what you need to know to get it running.
>
> **Docs:** [tightwad.dev](https://tightwad.dev) | [FAQ & Best Practices](https://youngharold.github.io/tightwad/) | [Wiki](https://github.com/youngharold/tightwad/wiki) | [PyPI](https://pypi.org/project/tightwad/)

---

## Prerequisites Check

Run these on the human's machine(s) to assess what they have:

```bash
# CUDA GPUs (NVIDIA)
nvidia-smi

# AMD GPUs (ROCm)
rocm-smi

# CPU info
lscpu                                    # Linux
sysctl -n machdep.cpu.brand_string       # macOS

# RAM
free -h                                  # Linux
vm_stat | head -5                        # macOS

# Network — check machines can reach each other
ping 192.168.1.x                         # replace with other machine's IP
```

**What you're looking for:**
- How many GPUs, how much VRAM each
- CPU core count and speed (relevant for CPU drafting)
- Total RAM (needed for CPU inference)
- Whether machines are on the same LAN (required for cross-machine setup)

---

## Decision Tree

Use this to pick the right Tightwad configuration:

| Situation | Recommendation |
|-----------|---------------|
| 1 machine, 1 GPU | Tightwad won't help much. Use Ollama directly. |
| 1 machine, 2+ GPUs | RPC cluster mode — pool the GPUs into one endpoint |
| 2+ machines, any GPUs | Speculative decoding proxy — **this is where Tightwad shines** |
| 2+ machines, some CPU-only | CPU drafting + GPU verification — still works, contributes |
| Any machine + cloud API key | GPU/CPU draft locally, verify via API — slash API costs |
| MoE target (Mixtral, GPT-OSS, Qwen3-MoE, MiniMax, DeepSeek) | **Defuse the GGUF first**, then enable `moe_placement: balanced` — see "MoE Models" section below |

**Rule of thumb:** If there's more than one machine with any compute at all, Tightwad can help.

---

## Quick Setup (5 steps)

### Step 1: Install on all machines

```bash
# On every machine that will participate:
pip install tightwad
```

Or install Ollama on each machine if using the Ollama backend (simpler for most users):

```bash
# Linux/macOS:
curl -fsSL https://ollama.com/install.sh | sh

# Make Ollama accessible over the network (required for multi-machine):
OLLAMA_HOST=0.0.0.0 ollama serve
```

### Step 2: Detect hardware

Run on each machine and note the output:

```bash
# NVIDIA GPU — note the GPU name, VRAM, and whether CUDA is available
nvidia-smi --query-gpu=name,memory.total --format=csv

# AMD GPU
rocm-smi --showmeminfo vram

# CPU-only — note core count and speed
lscpu | grep -E "Model name|CPU\(s\)|MHz"

# How much RAM (important for CPU inference)
free -h | grep Mem
```

**Classify each machine:**
- `≥16GB VRAM` → good target (big model)
- `8–15GB VRAM` → good draft (small model) or medium target
- `2–7GB VRAM` → draft only (1.7B–3B models)
- `<2GB VRAM or CPU only` → CPU draft with tiny models (0.5B–1.7B)

### Step 3: Generate cluster.yaml

The proxy config lives at `configs/cluster.yaml`. Fill in the template:

```yaml
proxy:
  host: 0.0.0.0
  port: 8088                          # The port your apps will point at
  max_draft_tokens: auto              # auto-tunes based on acceptance rate (or pin at 32)
  fallback_on_draft_failure: true     # Keep serving even if draft machine goes down

  draft:
    url: http://192.168.1.20:11434    # IP of the draft machine — replace with actual IP
    model_name: qwen3:8b              # Draft model — must be same family as target
    backend: ollama                   # "ollama" or "llamacpp"

  target:
    url: http://192.168.1.10:11434    # IP of the target machine — replace with actual IP
    model_name: qwen3:32b             # Target model — bigger = slower but smarter
    backend: ollama
```

**Field guide:**
- `url`: The machine running the model. Use `ip addr` (Linux) or `ipconfig` (Windows) to find IPs.
- `model_name`: Must be pulled on that machine already (`ollama pull qwen3:8b`).
- `backend`: Use `ollama` for quick setup. Use `llamacpp` for best performance + logprobs.
- `max_draft_tokens`: Use `auto` (recommended) or pin at `32` for manual control.

**For CPU-only draft machines**, use a tiny model:

```yaml
  draft:
    url: http://192.168.1.30:8081     # CPU machine running llama-server
    model_name: qwen3:1.7b
    backend: llamacpp
```

**For RPC cluster with ROCm multi-GPU**, add coordinator config:

```yaml
coordinator:
  host: 0.0.0.0
  port: 8080
  backend: hip                         # "hip" for AMD, "cuda" for NVIDIA
  gpus:
    - name: "7900 XTX #0"
      vram_gb: 24
    - name: "7900 XTX #1"
      vram_gb: 24
  # Optional: extra CLI args for llama-server
  extra_args: ["--no-mmap", "--no-warmup"]
  # Optional: env vars passed to llama-server (overrides auto-presets)
  env:
    HSA_ENABLE_SDMA: "1"              # override if your board handles SDMA fine
```

**Backend presets:** Tightwad auto-injects `HSA_ENABLE_SDMA=0` and `GPU_MAX_HW_QUEUES=1` for `hip` backends with 2+ GPUs. This prevents SDMA hangs on most AMD boards without any manual config. Explicit `env` values in YAML override presets.

**`flash_attn`** is a boolean (`true`/`false`). Emitted as `--flash-attn on` for llama.cpp b8112+:

```yaml
models:
  qwen3-32b:
    path: /models/Qwen3-32B-Q4_K_M.gguf
    flash_attn: true                   # emitted as --flash-attn on
    default: true
```

### Step 3.5: Run diagnostics

Before starting, verify everything is configured correctly:

```bash
tightwad doctor --fix
# Checks: config syntax, model files, binaries, RPC connectivity,
# llama.cpp version matching, model family compatibility, MoE VRAM,
# peer agent reachability
```

**Key v0.4+ features to know about:**
- **Family auto-detection** — Tightwad detects model architecture families at startup and warns on mismatch. No more silent 3% acceptance from cross-family pairs.
- **`max_draft_tokens: auto`** — cost-aware adaptive tuning. Adjusts based on acceptance rate AND draft-vs-verify timing. Recommend for all setups.
- **Chat template auto-detection** — `/v1/chat/completions` auto-detects Llama 3, Mistral, Gemma, Phi, DeepSeek, Command-R formats. Override with `proxy.chat_template: llama3` if needed.
- **Peer agent** — `tightwad peer start` on Windows/remote machines replaces SSH. Set `peer_port: 9191` on workers.
- **Quality gate** — `tightwad gate start` for CPU fleet + GPU verifier mode (datacenter cost reduction).
- **Consensus** — `proxy.consensus_mode: majority` with multiple drafters skips the target when drafters agree.
- **A/B benchmark** — `tightwad bench` compares proxy vs direct target speed.

### Step 4: Start the cluster

```bash
# On draft machine — start the draft model
OLLAMA_HOST=0.0.0.0 ollama run qwen3:8b

# On target machine — start the target model
OLLAMA_HOST=0.0.0.0 ollama run qwen3:32b

# On proxy machine — start Tightwad
tightwad proxy start

# Expected output:
# ✓ Draft model healthy  (qwen3:8b @ 192.168.1.20:11434)
# ✓ Target model healthy (qwen3:32b @ 192.168.1.10:11434)
# ✓ Proxy listening on http://localhost:8088
```

### Step 5: Verify it works

```bash
# Basic smoke test
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2 + 2?"}], "max_tokens": 20}'

# Should return a valid JSON response with the answer.

# Check acceptance rate (the key metric)
tightwad proxy status

# What success looks like:
# Acceptance rate: 50-89%   ← good (higher = more tokens from draft = faster)
# Acceptance rate: 3-10%    ← bad (probably cross-family models — see Model Selection)
# Acceptance rate: 0%       ← broken (check connectivity)

# Detailed stats
curl http://localhost:8088/v1/tightwad/status
```

---

## Model Selection Guide

**Same-family matching is critical.** Cross-family (e.g., Llama draft → Qwen target) drops acceptance to ~3%. Same family → high acceptance.

| Target Model Size | Recommended Draft Model | Expected Acceptance |
|---|---|---|
| 7B–14B | 0.5B–1.5B | 40–60% |
| 32B–72B | 3B–8B | 50–70% |
| 70B–405B | 8B–14B | 60–80% |

**Best combos (proven):**
- `qwen3:1.7b` → `qwen3:8b` — lightweight, fast
- `qwen3:8b` → `qwen3:32b` — the classic homelab setup (~64% acceptance)
- `qwen3:8b` → `qwen3.5:397b` (API) — 80% acceptance, massive API cost reduction
- `qwen3:1.7b` → `qwen3:32b` — CPU draft machine → GPU target

**Avoid:**
- Llama draft → Qwen target (~3% acceptance — different tokenizers)
- Mismatched quantization levels (Q2 draft → Q8 target can hurt acceptance)

---

## MoE Models (v0.5+)

Mixture-of-Experts models (Mixtral, GPT-OSS, Qwen3-MoE, MiniMax M2.5, DeepSeek-MoE) are first-class in Tightwad 0.5. If the user wants to run an MoE target across a pool, this is the path.

### Step 1: Defuse the GGUF (almost always required)

Most MoE GGUFs ship with **fused** expert tensors (`blk.L.ffn_*_exps.weight`) — one tensor per layer covering every expert. llama.cpp cannot per-expert-split a fused tensor, so placement will silently no-op. Tightwad ships a one-time rewriter:

```bash
tightwad moe defuse /path/to/mixtral-8x7b.Q4_K_M.gguf /path/to/mixtral-8x7b-indexed.Q4_K_M.gguf
```

- Output size equals input size. No quantization change. Weights are byte-identical when sliced back.
- Load the indexed file with `llama-server` exactly like the original.
- Skip this step only if you already have an indexed-form GGUF — `tightwad doctor` will warn you if placement is configured on a fused model.

### Step 2: Configure `moe_placement` on the model

```yaml
models:
  mixtral-8x7b:
    path: /models/mixtral-8x7b-indexed.Q4_K_M.gguf
    moe_placement: balanced          # off | balanced | profile-guided
    default: true
```

- **`balanced`** (recommended default) — bin-packs whole experts across GPUs proportional to VRAM. Emits llama.cpp `--override-tensor` flags automatically at `tightwad start`.
- **`profile-guided`** — pins frequently-hit experts to the fastest GPU using a captured profile. Experimental; requires the llama.cpp patch at `scripts/patches/llamacpp-moe-log.patch` + `env: { LLAMA_LOG_MOE: "1" }`. Recommend starting with `balanced` first.

### Step 3: Preview before launch (optional but valuable)

```bash
tightwad moe plan /models/mixtral-8x7b-indexed.Q4_K_M.gguf               # human-readable table
tightwad moe plan /models/mixtral-8x7b-indexed.Q4_K_M.gguf --emit-ot     # just the -ot flags
tightwad moe plan /models/mixtral-8x7b-indexed.Q4_K_M.gguf --json        # machine-readable
```

If the plan looks balanced (per-device bytes within ±15% of VRAM ratio), proceed.

### Step 4: Start the cluster and verify `-ot` emission

```bash
tightwad start
```

Check `~/.tightwad/logs/coordinator.log` — you should see `--override-tensor "^blk\.0\.ffn_(gate|up|down)\.(0|1|...)\.weight$=CUDA0"` lines in the llama-server invocation. If not, placement silently no-op'd (most common cause: still fused, re-run defuse).

### Step 5 (optional): Capture a profile and upgrade to profile-guided

```bash
# After the cluster has served representative traffic for 5+ minutes:
tightwad moe profile --follow-coord --duration 300 -o ~/.tightwad/moe-profile.json
tightwad moe summary ~/.tightwad/moe-profile.json

# Then update cluster.yaml:
#   moe_placement: profile-guided
#   moe_hot_profile: ~/.tightwad/moe-profile.json
# Restart with `tightwad stop && tightwad start`.
```

Hot-expert capture requires the instrumented llama.cpp build — see `scripts/patches/README.md`. Without it, profile-guided degrades to balanced (doctor warns).

### MoE model selection guide

| Target | Params | Recommended draft (same family!) | Notes |
|---|---|---|---|
| Mixtral 8x7B (~47B MoE, 13B active) | ~26 GB Q4 | Mistral 7B | Classic MoE entry point; fits a single 4090 |
| GPT-OSS 120B | ~60 GB Q4 | GPT-OSS 20B | Needs pooled VRAM or a very large single card |
| Qwen3-MoE 30B (MoE w/ 3B active) | ~18 GB Q4 | Qwen3-1.7B or 7B | Lean on activation sparsity |
| DeepSeek-MoE 16B (2B active) | ~10 GB Q4 | DeepSeek 1.3B | Fits a 12GB GPU comfortably |
| MiniMax M2.5 (229B) | ~130 GB Q4 | Qwen3-1.7B (via LM Studio) | Treat as API target; use speculation proxy, not RPC placement |

### MoE benchmark

Use the MoE-specific bench for acceptance/TTFT/speedup numbers against an OpenAI-compatible target (LM Studio, vLLM, llama-server):

```bash
tightwad moe bench \
  --target-url http://<host>:1234 \
  --target-model <model-id> \
  --max-tokens 256 \
  --json benchmarks/<model>-benchmark.json
```

Live-streams per-prompt TTFT, rolling acceptance, speedup.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Connection refused` | Ollama/llama-server not running on remote machine | SSH in, check `ollama ps` or `ps aux | grep llama`. Check firewall: `sudo ufw allow 11434` |
| `Connection refused` | Ollama bound to localhost only | Set `OLLAMA_HOST=0.0.0.0` and restart Ollama |
| Low acceptance rate (<10%) | Cross-family models | Switch draft to same model family as target |
| Low acceptance rate (10–40%) | Quantization mismatch or different fine-tune | Try same quant level; prefer base models over instruct for drafting |
| Slow despite high acceptance | Network bottleneck | Check `ping` between machines. Cross-machine latency >10ms hurts. Try `max_draft_tokens: 64` to reduce round trips |
| Proxy starts but no speedup | `max_draft_tokens` too low | Set `max_draft_tokens: auto` or `32` in cluster.yaml |
| Chat works but 0% acceptance | Chat template mismatch | Set `proxy.chat_template: llama3` for Llama models (auto-detects in v0.4+) |
| `alloc_tensor_range` error | MoE model routing overhead | MoE models replicate ~20GB to every GPU. Use 24GB+ GPUs only. `tightwad doctor` warns about this |
| `moe_placement` set but no `--override-tensor` flags in coordinator.log | GGUF still uses fused expert tensors | Run `tightwad moe defuse <fused.gguf> <indexed.gguf>` and point the model path at the indexed file |
| Doctor warns "profile-guided without LLAMA_LOG_MOE" | llama.cpp not instrumented for expert logging | Apply `scripts/patches/llamacpp-moe-log.patch`, rebuild llama.cpp, add `env: { LLAMA_LOG_MOE: "1" }` to cluster.yaml. Or downgrade to `moe_placement: balanced` |
| `tightwad moe profile` returns empty hits but nonzero total_tokens | Same as above — unpatched llama.cpp | Patch required for per-expert capture. `total_tokens > 0, hits == []` is the "working but unpatched" state |
| MoE model loads but per-device VRAM lopsided | `per_device_bytes` skew check — placement plan is wrong | Run `tightwad moe plan <model> --json` and inspect. File an issue if the deviation from VRAM ratio is >15% |
| Version mismatch at startup | llama.cpp versions differ | Build same version on all machines, or `--skip-version-check` |
| Draft machine keeps failing | Hardware too slow | CPU draft needs ≥15 tok/s to help. Try a smaller model (0.5B) |
| "Model not found" | Model not pulled on remote | SSH to that machine, run `ollama pull <model>` |
| llama-server hangs on ROCm multi-GPU | SDMA engine hang | Tightwad auto-injects `HSA_ENABLE_SDMA=0` for `hip` + 2+ GPUs. If running manually: `HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 llama-server ...` |

---

## Hardware Classification Quick Reference

| Hardware | Role | Best Model Size |
|----------|------|----------------|
| RTX 4090 / 4080 (24GB) | Target | 32B–70B |
| RTX 4070 Ti Super (16GB) | Target | 32B |
| RTX 3090 / 4070 (12–16GB) | Target | 14B–32B |
| RTX 2070 / 3060 (8–12GB) | Draft | 7B–8B |
| GTX 1060 / 1070 (6–8GB) | Draft | 3B–7B |
| GTX 770 / old cards (2–4GB) | Draft | 1.7B–3B |
| CPU (any, 16+ GB RAM) | Draft | 0.5B–1.7B |
| M-series Mac (unified memory) | Draft or Target | Depends on RAM |
| RX 7900 XTX / AMD (24GB) | Target | 32B–70B |
| RX 6700 / AMD mid-range | Draft | 7B–8B |

---

## Tell Your Human

After setup is complete, explain it in plain terms:

> "I set up Tightwad. Your chat app now points at port 8088 instead of 11434.
> Same models, same output, just faster. The small model on [Machine B] drafts tokens,
> the big model on [Machine A] verifies them. You won't notice anything different except speed.
> The acceptance rate is X% — that means X% of tokens are coming from the fast machine,
> not the slow one."

If they ask what to do with their old hardware:
> "That [old GPU/CPU machine] is now contributing to inference. It's not just sitting there anymore."

---

## Expanding the Cluster

Adding a new machine is non-destructive. The cluster grows:

1. Install Ollama (or llama-server) on the new machine
2. Pull an appropriate draft model
3. Add it to `cluster.yaml` as a draft source or fallback
4. Restart `tightwad proxy`

No downtime on existing machines required. The new node starts contributing immediately.
