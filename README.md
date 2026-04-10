# Tightwad

[![PyPI](https://img.shields.io/pypi/v/tightwad)](https://pypi.org/project/tightwad/) [![CI](https://github.com/youngharold/tightwad/actions/workflows/ci.yml/badge.svg)](https://github.com/youngharold/tightwad/actions) [![License](https://img.shields.io/github/license/youngharold/tightwad)](LICENSE) [![Python](https://img.shields.io/pypi/pyversions/tightwad)](https://pypi.org/project/tightwad/)

Mixed-vendor GPU inference cluster manager with speculative decoding proxy. Pools CUDA and ROCm GPUs across machines using [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc), and accelerates inference via application-layer speculative decoding across network-separated servers.

> **AI Agents:** See **[AGENTS.md](AGENTS.md)** for a complete setup playbook — hardware assessment, decision tree, step-by-step config, model selection guide, and troubleshooting. Designed for Claude, ChatGPT, Copilot, [OpenClaw](https://github.com/openclaw/openclaw), and any AI assistant helping users deploy Tightwad.

## How It Works in 10 Seconds

```
YOUR HARDWARE (any mix works)                         TIGHTWAD
                                                         │
  RTX 4070 Ti Super (16GB, NVIDIA) ──────────┐           │
  RTX 3060 (12GB, NVIDIA) ───────────────────┤           │
  RTX 2070 (8GB, NVIDIA) ────────────────────┤  cluster  │   ┌──────────────-┐
  GTX 770 (2GB — yes, really) ───────────────┤──────────►│──►│ OpenAI API    │
  RX 7900 XTX (24GB, AMD!) ─────────────────-┤  yaml     │   │ localhost:8088│
  Old Xeon workstation (CPU only) ───────────┤           │   └──────────────-┘
  Your laptop (M2, CPU draft) ───────────────┘           │
                                                         │
  CUDA ✓   ROCm ✓   CPU ✓   Mixed ✓            One endpoint.
```

> It's not 2 matching GPUs. It's your entire junk drawer of compute unified into one API.
> That dusty 770 in your closet? Put it to work.

```
  Without Tightwad: big model generates every token, one at a time
  With Tightwad:    big model only works on the tokens it disagrees with
  Output quality:   IDENTICAL (greedy) / EQUIVALENT (sampling)
  Speed:            Up to 2-3x faster
```

> The small model is fast but sometimes wrong. The big model is slow but always right.
> Tightwad uses the small model to do most of the work, and the big model to catch mistakes.
> Because catching mistakes is cheap — it's one batch operation, not N serial ones.

## What Does This Look Like as a User?

**You change nothing about your workflow.** Tightwad is invisible.

| | Before | After |
|---|---|---|
| **Your chat app** | Open WebUI, ChatBot UI, etc. | Same app, no changes |
| **Points at** | `http://192.168.1.10:11434` (Ollama on one machine) | `http://192.168.1.10:8088` (Tightwad proxy) |
| **Model you talk to** | Qwen3-32B | Qwen3-32B (same model, same output) |
| **What you see** | Normal chat responses | Normal chat responses, just faster |
| **The small model** | Doesn't exist | Hidden — drafting on a different machine entirely |
| **Other machines** | Idle, wasted | RTX 2070, old Xeon, laptop — all contributing |

**That's it.** One URL change. Same UI, same model, same quality. Tightwad handles everything behind the scenes:

1. Your chat sends a message to Tightwad's port
2. Behind the curtain, a small model quickly predicts the next several tokens
3. The big model verifies them all in one shot (instead of generating one at a time)
4. Tightwad streams the verified tokens back to your chat
5. You see the response faster — and it's **identical** (greedy decoding) or **statistically equivalent** (sampling) to what the big model would have produced alone

The small model is like autocomplete on your phone — it suggests, the big model accepts or corrects. You only ever see the final, verified output.

## Four Modes

### 1. Speculative Decoding Proxy — Draft + Verify across machines

A fast small model (e.g., 1.7B on any CPU or cheap GPU) drafts candidate tokens, a large model (e.g., 32B-72B) verifies them in batch. Output quality is **equivalent to running the large model alone**, but up to 2x faster because batch verification is much cheaper than autoregressive generation. Network traffic: **bytes** (token IDs only).

### 2. RPC Cluster — Pool GPUs into one endpoint

Combine GPUs from different machines and vendors into a single OpenAI-compatible API. The coordinator distributes model layers across local and remote GPUs. Use this when a model doesn't fit on any single machine.

> **Note:** The coordinator machine needs enough **system RAM** for the full model file (not just its GPU share). llama.cpp mmaps the entire GGUF before distributing tensors to workers. A 70B Q4_K_M (~40GB) needs ~44GB RAM on the coordinator.

### 3. Combined Mode — Speculation Over a Pool (the killer feature)

**When a model doesn't fit on one machine, pool the GPUs AND speculate on top.** The RPC pool is slow autoregressive (3 tok/s over WiFi), but batch verification amortizes the RPC overhead — 32 tokens per round instead of 1 token per round-trip. Result: **1.8x speedup** over pool-only, making models that don't fit on one machine actually usable.

```
ANY junk hardware (P400 2GB, GTX 770, laptop CPU, Raspberry Pi)
    │ runs a small 1-2B draft model (~30 tok/s)
    │ sends token IDs (bytes, not megabytes)
    ▼
Tightwad Proxy (:8088)
    │ sends draft to pool for BATCH verification
    ▼
RPC GPU Pool (any mix: CUDA + ROCm + Metal, running 70B)
    │ verifies 32 tokens in ONE forward pass
    │ 1 RPC round-trip for 32 tokens instead of 32 round-trips
    ▼
5+ tok/s instead of 3 tok/s — and the 70B model fits nowhere else
```

### 4. Quality Gate — CPU Fleet Drafts, GPU Verifies

A fleet of cheap agents (CPUs, small GPUs) generate full responses using small models. A single powerful GPU reviews each output — approving, correcting, or rejecting it. 60-80% of responses pass unchanged, so the GPU only processes the hard 20-40%. `tightwad gate start` to run this mode.

> The draft model needs: (1) same model family as the target, (2) llamacpp backend (not Ollama) for prompt-append verification, (3) any hardware that can run a 1-2B model. That's it.

```
Client (OpenAI API)
        │
        ▼
┌──────────────────────────────┐
│   Tightwad Proxy (:8088)      │  Python async server
│   Speculation Loop:          │
│   1. Draft N tokens (auto)    │──► Draft: Qwen3-8B (fast, local)
│   2. Verify batch            │──► Target: Qwen3-32B (accurate, local or API)
│   3. Accept/reject           │
│   4. Stream to client        │
└──────────────────────────────┘
```

**Why not just use RPC?** RPC ships 100-300 MB of tensor data per step over the network. The speculative proxy ships token IDs (bytes). For models that fit on a single machine's VRAM, speculation is dramatically faster.

## Docker Quick Start

The fastest way to get a speculative decoding proxy running. No config files needed — just set your draft and target server URLs:

```bash
# One-liner with Docker
docker run --rm --network host \
  -e TIGHTWAD_DRAFT_URL=http://192.168.1.10:11434 \
  -e TIGHTWAD_DRAFT_MODEL=qwen3:8b \
  -e TIGHTWAD_TARGET_URL=http://192.168.1.20:11434 \
  -e TIGHTWAD_TARGET_MODEL=qwen3:32b \
  ghcr.io/youngharold/tightwad

# Or with Docker Compose (edit docker-compose.yml with your IPs first)
docker compose up
# Logs persist in ./logs/ across restarts
```

> **Mac/Docker Desktop:** Replace `--network host` with `-p 8088:8088` and use `host.docker.internal` instead of LAN IPs.

Docker Compose includes a healthcheck (`/v1/models` every 10s) and mounts `./logs/` for persistent proxy logs.

All `TIGHTWAD_*` env vars:

| Env Var | Default | Description |
|---------|---------|-------------|
| `TIGHTWAD_DRAFT_URL` | *required* | Draft server URL |
| `TIGHTWAD_DRAFT_MODEL` | `draft` | Draft model name |
| `TIGHTWAD_DRAFT_BACKEND` | `ollama` | `ollama` or `llamacpp` |
| `TIGHTWAD_TARGET_URL` | *required* | Target server URL |
| `TIGHTWAD_TARGET_MODEL` | `target` | Target model name |
| `TIGHTWAD_TARGET_BACKEND` | `ollama` | `ollama` or `llamacpp` |
| `TIGHTWAD_PORT` | `8088` | Proxy listen port |
| `TIGHTWAD_HOST` | `0.0.0.0` | Proxy bind host |
| `TIGHTWAD_MAX_DRAFT_TOKENS` | `8` | Tokens per draft round (or `auto` for adaptive tuning) |
| `TIGHTWAD_PROXY_TOKEN` | *(unset)* | Bearer token for proxy API auth (recommended) |
| `TIGHTWAD_ALLOW_PRIVATE_UPSTREAM` | `true` | SSRF: `false` = block private/LAN upstream IPs |
| `TIGHTWAD_MAX_TOKENS_LIMIT` | `16384` | Hard cap on `max_tokens` in requests — rejects higher values with 400 (DoS mitigation) |
| `TIGHTWAD_MAX_BODY_SIZE` | `10485760` | Max request body size in bytes (10 MB) — rejects oversized payloads with 413 before buffering |

### Proxy Authentication

The proxy API binds to `0.0.0.0:8088` by default, making it reachable by any
device on the LAN (or the internet if ports are forwarded). **Set a token to
prevent unauthorized use of your GPU compute.**

**Via environment variable (Docker / Docker Compose):**

```bash
docker run --rm --network host \
  -e TIGHTWAD_DRAFT_URL=http://192.168.1.10:11434 \
  -e TIGHTWAD_TARGET_URL=http://192.168.1.20:11434 \
  -e TIGHTWAD_PROXY_TOKEN=my-secret-token \
  ghcr.io/youngharold/tightwad
```

**Via cluster.yaml:**

```yaml
proxy:
  host: 0.0.0.0
  port: 8088
  auth_token: "${TIGHTWAD_PROXY_TOKEN}"   # or paste the token directly
  draft:
    url: http://127.0.0.1:8081
    model_name: qwen3-1.7b
  target:
    url: http://192.168.1.100:8090
    model_name: qwen3-32b
```

**Making authenticated requests:**

```bash
curl http://localhost:8088/v1/chat/completions \
  -H "Authorization: Bearer my-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

If no token is configured the proxy operates in open (unauthenticated) mode
for backward compatibility, but logs a **security warning** on startup.
`TIGHTWAD_TOKEN` (the swarm seeder token) is also accepted as a fallback alias.

### SSRF Protection (upstream URL validation)

Tightwad validates all upstream URLs before opening connections (audit ref: SEC-5).

**What is always enforced:**
- **Scheme allowlist** — only `http://` and `https://` are accepted. `file://`, `gopher://`, `ftp://`, and every other scheme are rejected with a clear error.

**What is enforced when `allow_private_upstream: false`:**
- **Private/internal IP blocking** — requests to RFC-1918 ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), link-local / IMDS (`169.254.0.0/16`), and IPv6 equivalents (`::1`, `fc00::/7`, `fe80::/10`) are blocked.
- **DNS-rebinding protection** — hostnames are resolved via DNS and the resolved IPs are also checked, so a domain that resolves to an internal address is caught even if the URL looks public.

**Homelab default (`allow_private_upstream: true`):**

Because Tightwad's most common use case targets LAN servers, the private-IP check **defaults to allowed**. The scheme check is still always enforced.

```yaml
proxy:
  # Default: LAN/loopback targets are fine (common homelab setup)
  allow_private_upstream: true   # omit or set true for home/LAN use

  # Strict mode: useful in cloud or multi-tenant environments
  # allow_private_upstream: false

  draft:
    url: http://192.168.1.101:11434   # OK in default mode
    model_name: qwen3-1.7b
  target:
    url: http://192.168.1.100:8080
    model_name: qwen3-32b
```

**Via environment variable:**

```bash
# Strict mode (block private/internal targets)
export TIGHTWAD_ALLOW_PRIVATE_UPSTREAM=false

# Homelab mode (default)
export TIGHTWAD_ALLOW_PRIVATE_UPSTREAM=true   # or omit entirely
```

If a URL fails validation the proxy refuses to start and prints a clear error explaining why and how to fix it.

## Quick Start

```bash
# Install
pip install tightwad

# Auto-discover LAN servers and generate config
tightwad init

# Or edit topology manually
vim configs/cluster.yaml

# Verify your setup
tightwad doctor        # check config, binaries, network, versions
tightwad doctor --fix  # show fix suggestions for any issues
```

### Speculative Decoding Proxy

```bash
# Start the proxy (draft + target servers must be running)
tightwad proxy start

# Check health and acceptance rate stats
tightwad proxy status

# Test it
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# Detailed stats
curl http://localhost:8088/v1/tightwad/status

# Stop
tightwad proxy stop
```

### RPC Cluster

```bash
# Check cluster status
tightwad status

# Start (after rpc-server instances are running on workers)
tightwad start

# View coordinator logs
tightwad logs              # last 50 lines
tightwad logs -f           # live tail

# Hot-swap to a different model (RPC workers persist)
tightwad swap deepseek-r1-70b

# Benchmark
tightwad benchmark

# Stop
tightwad stop
```

## Homelab Recipe

A realistic three-machine setup you can reproduce in ~30 minutes. Start with two machines, add more anytime. The cluster grows.

**Hardware:**
- **Machine A (target):** Desktop with RTX 4070 Ti Super + RTX 3060 (28GB VRAM combined) — the workhorse
- **Machine B (draft):** Old gaming PC with RTX 2070 (8GB VRAM) — that box you almost sold
- **Machine C (CPU draft):** Server or workstation with no GPU — CPU-only, still contributes

**Expected results:** 58–64% average token acceptance, up to 88% on reasoning tasks. Machine C adds throughput even without a GPU.

> **Replace all `192.168.1.x` addresses below with your actual machine IPs.** Find them with `ip addr` (Linux), `ipconfig` (Windows), or `ipconfig getifaddr en0` (macOS).

---

### Step 1 — On Machine B: Start the draft model

```bash
# Machine B — RTX 2070 (8GB VRAM)
ollama run qwen3:8b
# Confirm it works:
ollama ps
# Should show: qwen3:8b running
```

Ollama listens on `0.0.0.0:11434` by default. If not, set `OLLAMA_HOST=0.0.0.0` before starting.

### Step 2 — On Machine C: Start a CPU draft model

```bash
# Machine C — CPU only, no GPU needed
# llama-server with a tiny model is ideal for CPU drafting
llama-server -m qwen3-1.7b-q4_k_m.gguf --port 8081 --host 0.0.0.0
# Or with Ollama:
OLLAMA_HOST=0.0.0.0 ollama run qwen3:1.7b
```

Even at 15–30 tok/s on CPU, Machine C reduces load on Machine B and adds redundancy.

### Step 3 — On Machine A: Start the target model

```bash
# Machine A — RTX 4070 Ti Super + RTX 3060 (28GB combined via llama.cpp RPC)
ollama run qwen3:32b
# Confirm:
ollama ps
# Should show: qwen3:32b running
```

Same note: ensure Ollama is accessible on the network (`OLLAMA_HOST=0.0.0.0`).

### Step 4 — On whichever machine runs the proxy: Install Tightwad

```bash
pip install tightwad
```

### Step 5 — Generate config

**Option A: Auto-discover with `tightwad init` (recommended)**

```bash
tightwad init
# Scans your LAN for Ollama and llama-server instances
# Shows a table of discovered servers
# You pick target (big model) and draft (small model) by number
# Writes configs/cluster.yaml automatically
```

Example output:
```
Scanning LAN for inference servers...

          Discovered Servers (192.168.1.0/24)
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
┃ # ┃ Host          ┃ Port  ┃ Backend ┃ Models     ┃ Status  ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
│ 1 │ 192.168.1.10  │ 11434 │ ollama  │ qwen3:32b  │ healthy │
│ 2 │ 192.168.1.20  │ 11434 │ ollama  │ qwen3:8b   │ healthy │
└───┴───────────────┴───────┴─────────┴────────────┴─────────┘

Select TARGET server (big model): 1
Select DRAFT server (small fast model): 2
Write to configs/cluster.yaml? [Y/n] y
```

If your subnet isn't auto-detected correctly, specify it manually:
```bash
tightwad init --subnet 192.168.1.0/24
```

**Option B: Manual config**

Edit `configs/cluster.yaml` directly:

```yaml
proxy:
  host: 0.0.0.0
  port: 8088
  max_draft_tokens: auto            # auto-tunes based on acceptance rate
  fallback_on_draft_failure: true
  draft:
    url: http://192.168.1.20:11434    # Machine B (RTX 2070) — replace with your IP
    model_name: qwen3:8b
    backend: ollama
  target:
    url: http://192.168.1.10:11434    # Machine A (4070 Ti + 3060) — replace with your IP
    model_name: qwen3:32b
    backend: ollama
```

Replace all IPs with your actual machine IPs (`ip addr` on Linux, `ipconfig` on Windows).

**Option C: Docker (no config file at all)**

Skip config entirely and use environment variables:

```bash
docker run --rm --network host \
  -e TIGHTWAD_DRAFT_URL=http://192.168.1.20:11434 \
  -e TIGHTWAD_DRAFT_MODEL=qwen3:8b \
  -e TIGHTWAD_TARGET_URL=http://192.168.1.10:11434 \
  -e TIGHTWAD_TARGET_MODEL=qwen3:32b \
  ghcr.io/youngharold/tightwad
```

**Adding Machine C later** (CPU draft as fallback or parallel drafter):

```yaml
  # Add to configs/cluster.yaml:
  draft_fallback:
    url: http://192.168.1.30:8081     # Machine C (CPU only) — replace with your IP
    model_name: qwen3:1.7b
    backend: llamacpp
```

### Step 6 — Start the proxy

```bash
tightwad proxy start
# Expected output:
# ✓ Draft model healthy  (qwen3:8b @ 192.168.1.20:11434)   — Machine B
# ✓ Target model healthy (qwen3:32b @ 192.168.1.10:11434)  — Machine A
# ✓ Proxy listening on http://localhost:8088
```

### Step 7 — Test it

```bash
# Basic test
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 17 * 24?"}],
    "max_tokens": 100
  }'

# Check acceptance rate stats
tightwad proxy status
# Expected: Acceptance rate: ~58% | Rounds: N | Tokens saved: N

# Detailed stats
curl http://localhost:8088/v1/tightwad/status
```

### Step 8 — Point your app at it

Any OpenAI-compatible client works. Just change the base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8088/v1",
    api_key="not-needed"  # Tightwad doesn't require an API key
)

response = client.chat.completions.create(
    model="tightwad",
    messages=[{"role": "user", "content": "Explain recursion"}]
)
```

**Acceptance rates you can expect with this setup:**

| Task | Acceptance Rate |
|------|:--------------:|
| Reasoning / math | ~88% |
| Code generation | ~73% |
| Factual Q&A | ~52% |
| Creative writing | ~34% |
| **Average** | **~58%** |

> **The cluster grows.** Start with Machines A + B. Add Machine C when you're ready. Add a fourth machine (that GTX 770 you haven't thrown out yet) whenever. Each new node contributes without disrupting the existing setup — just add it to `cluster.yaml` or re-run `tightwad init`. Tightwad doesn't care what generation or vendor the hardware is from. CUDA, ROCm, Metal, CPU-only — it all pools together. The only thing that matters is that your draft and target models share the same family.

> **Note on the bigger picture:** With Qwen3-8B drafting for Qwen3.5-397B (via API), we've seen 80% acceptance after whitespace normalization — meaning 4 in 5 tokens come from your local GPU, not the cloud. Reasoning tasks hit 88%. The bigger the gap between draft and target quality, the more you save.

## Configuration

Edit `configs/cluster.yaml` (or generate one with `tightwad init`):

```yaml
# ⚠️  Replace all IPs and model paths below with your own values.
# Find your IPs: ip addr (Linux), ipconfig (Windows), ipconfig getifaddr en0 (macOS)

# Speculative decoding proxy
proxy:
  host: 0.0.0.0
  port: 8088
  max_draft_tokens: auto            # auto-tunes based on acceptance rate (or pin at 32)
  fallback_on_draft_failure: true
  max_tokens_limit: 16384           # Hard cap on max_tokens per request (DoS mitigation, CQ-1)
  max_body_size: 10485760           # Max request body bytes — 10 MB (memory-exhaustion mitigation, CQ-5)
  draft:
    url: http://192.168.1.50:8081    # ← your draft machine's IP + port
    model_name: qwen3-8b
    backend: llamacpp                  # or "ollama"
  target:
    url: http://192.168.1.100:8080   # ← your target machine's IP + port
    model_name: qwen3-32b
    backend: llamacpp

# RPC cluster (optional, for tensor-parallel across machines)
# Pool GPUs from multiple machines into a single model
coordinator:
  host: 0.0.0.0
  port: 8090
  backend: cuda          # "cuda" (NVIDIA) or "hip" (AMD/ROCm)
  gpus:                  # Local GPUs on the coordinator machine
    - name: "RTX 4070 Ti Super"
      vram_gb: 16
    - name: "RTX 3060"
      vram_gb: 12

workers:                 # Remote machines running rpc-server
  - host: 192.168.1.20   # ← your worker's IP
    gpus:
      - name: "RTX 2070"
        vram_gb: 8
        rpc_port: 50052
  - host: 192.168.1.30   # ← your worker's IP
    gpus:
      - name: "Apple M2 Metal"
        vram_gb: 11       # Use recommendedMaxWorkingSetSize, not total unified memory
        rpc_port: 50052

models:
  qwen3-32b:
    path: /models/Qwen3-32B-Q4_K_M.gguf  # ← absolute path on coordinator machine
    ctx_size: 8192
    flash_attn: true              # boolean (legacy string values "on"/"off" also accepted)
    default: true
```

### Backend Presets & Passthrough

Tightwad auto-injects known-good settings per backend. For example, ROCm multi-GPU setups automatically get `HSA_ENABLE_SDMA=0` and `GPU_MAX_HW_QUEUES=1` to prevent SDMA hangs — no manual configuration needed.

You can override presets or add your own with `extra_args` and `env` in the coordinator section:

```yaml
coordinator:
  backend: hip
  gpus:
    - name: "7900 XTX #0"
      vram_gb: 24
    - name: "7900 XTX #1"
      vram_gb: 24
  # Additional CLI args passed to llama-server
  extra_args: ["--no-mmap", "--no-warmup"]
  # Environment variables (override auto-injected presets)
  env:
    HSA_ENABLE_SDMA: "1"  # override the default preset
```

### Server Backends

The proxy supports two backend types for draft and target servers:

| Backend | Endpoint | Best for |
|---------|----------|----------|
| `ollama` | `/api/generate` (raw mode) | Quick setup, any Ollama instance |
| `llamacpp` | `/v1/completions` (with logprobs) | Best performance, full logprobs support |

## How Speculative Decoding Works

1. **Draft:** The small model generates N candidate tokens (fast, ~100+ tok/s)
2. **Verify:** The large model evaluates all N tokens in a single forward pass
3. **Accept/reject:** Keep tokens where both models agree, take the large model's token at the first disagreement
4. **Repeat** until done

The output is **equivalent** to running the large model alone — the small model just proposes shortcuts.

### Benchmark Results

#### Wall-Clock Speedup (Qwen3-8B → Qwen3-32B, cross-machine llama-server)

Draft on RTX 2070 (8GB), target on RTX 4070 Ti Super + RTX 3060 (28GB). Both via llama-server with prompt-append verification.

| Prompt | Baseline | Speculative | Speedup |
|--------|:--------:|:-----------:|:-------:|
| Capital of France | 1.17s | 0.90s | **1.30x** |
| Thermodynamics | 12.73s | 9.09s | **1.40x** |
| Prime checker | 12.76s | 10.15s | **1.28x** |
| Average speed | 13.24s | 10.95s | **1.21x** |
| TCP vs UDP | 5.58s | 4.88s | **1.14x** |
| **Total** | **45.43s** | **35.96s** | **1.27x** |

**1.27x overall speedup** with `max_draft_tokens: 32` (50 rounds, 31.7 tokens/round, 100% acceptance).

##### Tuning `max_draft_tokens`

| Setting | Rounds | Tok/Round | Overall Speedup |
|:-------:|:------:|:---------:|:---------------:|
| 8 | 96 | 8.8 | 0.63x (slower) |
| **32** | **50** | **31.7** | **1.27x** |
| 64 | 16 | 56.5 | 1.21x |

The sweet spot is **32 draft tokens** — fewer rounds reduce HTTP overhead, but going too high (64) adds draft latency that outweighs the savings.

#### Acceptance Rate Details (logprobs verification)

| Metric | Value |
|--------|:-----:|
| **Acceptance Rate** | **73.5%** |
| **Effective tokens/round** | **6.6** (at max_draft_tokens=8) |
| Total rounds | 87 |
| Drafted tokens | 671 |
| Accepted tokens | 493 |

#### Text-Match Benchmarks (Ollama, for acceptance rate comparison)

Same-family (Qwen3-8B → Qwen3-32B, local Ollama):

| Prompt Type | Acceptance Rate | Rounds | Notes |
|-------------|:--------------:|:------:|-------|
| Reasoning   | **89%**        | 32     | Highest — deterministic math answers |
| Code        | **76%**        | 34     | High — structured syntax overlap |
| Factual     | 73%            | 16     | Strong agreement on facts |
| List        | 42%            | 40     | Varied phrasing causes divergence |
| Creative    | 39%            | 6      | Lowest — many valid outputs |
| **Average** | **63.8%**      | 25.6   | |

#### Cloud API Benchmarks (OpenRouter)

| Draft | Target | Size Gap | Acceptance |
|-------|--------|:--------:|:----------:|
| Llama 3.1 8B | Llama 3.1 405B | 50x | **18.9%** |
| Qwen3 1.7B | Qwen3.5 397B | 233x | **10.8%** |
| Llama 3.1 8B | Llama 3.1 70B | 9x | **9.9%** |
| Qwen3 1.7B | Qwen3 235B | 138x | **6.6%** |
| Qwen3 8B | Llama 3.3 70B | cross-family | **~3%** |

> **Important:** Over cloud APIs, the per-round network latency (~3-8s per API call) makes speculative decoding *slower* than baseline despite positive acceptance rates. Spec decoding shines when both models are local or very low-latency.

**Key findings:**
- Same-family drafting is critical — cross-family drops to ~3% regardless of model size
- Draft model size matters — the 1.7B is too small to predict 200B+ target phrasing
- Larger targets don't always mean lower acceptance (405B beat 70B with the same 8B draft)
- Cloud API latency negates wall-clock speedup even with decent acceptance rates

#### CPU Draft Results (Qwen3-1.7B CPU → Qwen3-32B GPU)

| Draft Host | Draft Speed | Acceptance | Wall-Clock Speedup |
|------------|:-----------:|:----------:|:------------------:|
| M4 Mac CPU (llama-server) | 32.8 tok/s | 68% | 0.80x |
| Unraid CPU (Ollama, text-match) | 14.9 tok/s | 68% | 0.14x |

CPU drafting with a 1.7B model works but doesn't achieve speedup at `max_draft_tokens=8` due to HTTP round-trip overhead. At `max_draft_tokens=32`, CPU drafting achieves significant speedup (see Combined Mode below).

#### Combined Mode: Speculation Over RPC Pool

**The killer feature.** When a model is too large for any single machine, pool GPUs via RPC and use speculative decoding to overcome RPC's per-token latency. The draft model runs on any junk hardware (CPU, 2GB GPU) and the pooled target verifies 32 tokens per batch instead of generating one at a time.

**Qwen3-32B (4-GPU pool, Qwen3-1.7B draft on M4 CPU):**

| Mode | Speed | Notes |
|------|:-----:|-------|
| RPC pool direct (autoregressive) | 3.0 tok/s | Each token = full RPC round-trip to all workers |
| **RPC pool + speculation** | **5.4 tok/s** | 32 tokens verified per batch, 100% acceptance |
| **Speedup** | **1.8x** | |

**Llama 3.3 70B (4-GPU pool, Llama 3.1 8B draft on M4 Metal):**

| Mode | Tokens | Time | Speed |
|------|:------:|:----:|:-----:|
| RPC pool direct (autoregressive) | 512 | 231s | 2.2 tok/s |
| **RPC pool + speculation** | **519** | **127s** | **4.1 tok/s** |
| **Speedup** | | | **1.86x** |

100% acceptance rate, 33 tokens/round. The 70B model doesn't fit on any single machine — it's distributed across 4 GPUs (4070 Ti Super + 3060 + 2070 + M2 Metal = 52GB VRAM) over WiFi. Without speculation: painfully slow. With speculation: usable.

> **Critical lesson: draft and target MUST be the same model family.** Llama 3.2 3B → Llama 3.3 70B got 1.6% acceptance (10x slower than no speculation) despite sharing a tokenizer. Llama 3.1 8B → Llama 3.3 70B gets 100% acceptance because they share the same architecture. Tightwad automatically detects model families at proxy startup and in `tightwad doctor`, warning loudly on mismatch.

```
Why this works:

  Pool autoregressive: 1 token → full RPC round-trip → 1 token → full RPC round-trip → ...
                       2-3 tok/s (network latency per token)

  Pool + speculation:  Draft 32 tokens (local GPU, fast, no network)
                       → Verify 32 tokens in ONE batch (one RPC round-trip for 32 tokens)
                       → 4-5 tok/s (network latency amortized over 32 tokens)
```

**This means any model that fits across your pooled GPUs is usable — even over WiFi.** The draft model just needs to be from the same model family as the target and small enough to run on your local hardware.

#### RPC Pool Without Speculation (for comparison)

Don't do this over WiFi. RPC tensor-parallelism ships 100-300 MB per inference step.

| Setup | Speed |
|-------|:-----:|
| Desktop local only (4070+3060, 32B) | 17.0 tok/s |
| 4-GPU RPC pool (4070+3060+2070+M2, 32B) | 3.0 tok/s |
| Same pool + speculation (Qwen3-1.7B draft) | 5.4 tok/s |
| 4-GPU RPC pool (4070+3060+2070+M2, **70B**) | 2.2 tok/s |
| Same pool + speculation (Llama 3.1 8B draft) | **4.1 tok/s** |

RPC pooling is only useful when the model doesn't fit on one machine. When it does fit locally, don't pool — just use speculation with a remote drafter.

### Use Cases

- **Models too big for one machine:** Pool GPUs via RPC, then speculate on top — the draft model turns 3 tok/s into 5+ tok/s. A 70B model across 4 consumer GPUs becomes usable
- **Local multi-GPU:** Draft on a consumer GPU ($200), verify on a larger GPU/rig
- **Cloud cost reduction:** Draft locally, verify via cloud API — fewer API calls for the same output quality
- **CPU draft, GPU verify:** Run a tiny model (0.6B-2B) on CPU/RAM, verify on GPU. Turns every idle CPU into usable inference compute
- **Multi-drafter parallelism:** Multiple CPUs each run a draft model in parallel, the GPU target picks the best candidate
- **Legacy GPU revival:** A 12-year-old GPU with 2GB VRAM can run a 1-2B draft model for a 70B+ target — turning e-waste into productive infrastructure
- **Junk drawer inference:** Pool ALL your hardware — CUDA, ROCm, Metal, CPU — into one endpoint. The speculative proxy handles the coordination. No GPU left behind

## Swarm Transfer — P2P Model Distribution

When you need to get a 40 GB model onto 5 worker machines, rsync from one source = 200 GB of outbound transfer. Swarm transfer splits the model into 64 MB pieces with SHA256 hashes and lets workers pull from **any peer** that has pieces — including each other.

```
rsync (single-source):                swarm (P2P):

  Source ──► Worker 1 (40 GB)           Source ──► Worker 1 ──► Worker 3
  Source ──► Worker 2 (40 GB)           Source ──► Worker 2 ──► Worker 4
  Source ──► Worker 3 (40 GB)           Worker 1 ──► Worker 5
  Source ──► Worker 4 (40 GB)           Worker 2 ──► Worker 5
  Source ──► Worker 5 (40 GB)
  Total: 200 GB from source            Total: ~80 GB from source (peers share the rest)
```

| | rsync (`tightwad distribute`) | swarm (`tightwad swarm`) |
|---|---|---|
| **Transfer pattern** | Single source → each worker | Any peer → any peer |
| **Source bandwidth** | O(N × model_size) | O(model_size) |
| **Resume on interrupt** | Restart from beginning | Continue from last piece |
| **Integrity** | Trust the network | SHA256 per piece |
| **Best for** | 1-2 workers, small models | 3+ workers, large models |

> **Note:** Swarm transfer is intended for distributing models across your own cluster. Ensure you have the right to distribute any model files you share. Some models (e.g., Llama 3) require license acceptance on HuggingFace.

```bash
# On the source machine (--token requires auth, --allowed-ips restricts by subnet)
tightwad manifest create ~/models/Qwen3-32B-Q4_K_M.gguf
tightwad swarm seed ~/models/Qwen3-32B-Q4_K_M.gguf \
  --token mysecret \
  --allowed-ips 192.168.1.0/24

# On each worker (can pull from source + other workers)
tightwad swarm pull ~/models/Qwen3-32B-Q4_K_M.gguf \
  --manifest http://192.168.1.10:9080/manifest \
  --peer http://192.168.1.10:9080 \
  --peer http://192.168.1.20:9080 \
  --token mysecret

# Check progress
tightwad swarm status ~/models/Qwen3-32B-Q4_K_M.gguf
```

See the [Swarm Transfer](#swarm-transfer) section above for architecture details on rarest-first piece selection and bitfield tracking.

## Why Tightwad?

You've probably heard of the other tools. Here's how Tightwad fits in.

### vs vLLM

vLLM is excellent production inference software. It's also primarily CUDA-focused, ROCm support is experimental. If you have an AMD GPU, getting it working takes extra effort. Tightwad pools CUDA and ROCm GPUs on the same model, same endpoint.

vLLM does support speculative decoding, but only within a single machine. Tightwad's proxy does it across your network — your draft model can be on a completely different box than your target.

**Critically: vLLM cannot pool heterogeneous hardware.** You can't mix a GTX 770 with a 4070 Ti in vLLM. You can't combine a CUDA machine with an AMD machine. You can't add a CPU-only node to the cluster. vLLM assumes uniform, high-end CUDA hardware throughout. Tightwad assumes you have a junk drawer.

vLLM is built for ML teams running production workloads at scale. Tightwad is built for anyone with two machines and a network cable.

| | vLLM | Tightwad |
|--|------|----------|
| AMD / ROCm support | Experimental | ✓ |
| Cross-machine speculative decoding | ✗ | ✓ |
| Mix old + new GPU generations | ✗ | ✓ |
| CPU nodes in the cluster | ✗ | ✓ |
| Works with Ollama | ✗ | ✓ |
| Target audience | Production ML teams | Homelab / anyone |

### vs Ollama

Ollama is great. It's the reason most people have local models running at all. But Ollama runs one model on one machine. When you outgrow one GPU, Ollama can't help you — it has no concept of pooling or cross-machine inference.

**Ollama cannot combine machines at all.** Your RTX 2070 on the old gaming PC and your RTX 4070 on the main rig are completely isolated from each other in Ollama's world. They'll never cooperate on a single request.

Tightwad is the next step after Ollama. Keep using Ollama as the backend on each machine — Tightwad just coordinates between them.

### vs llama.cpp RPC

Tightwad is built *on top of* llama.cpp RPC. We didn't replace it — we added the orchestration layer, YAML configuration, CLI, and speculative decoding proxy that you'd otherwise have to script yourself.

The key difference for speculative decoding: llama.cpp RPC ships 100–300 MB of tensor data over the network per step. Tightwad's proxy ships token IDs — a few bytes. For models that fit on individual machines, the proxy approach is dramatically faster over a standard home network.

### vs TGI (HuggingFace Text Generation Inference)

TGI is optimized for the HuggingFace ecosystem and integrates well with their services. It's an excellent tool if you're already in that ecosystem.

Tightwad is MIT licensed, has no vendor affiliation, and works with your existing Ollama or llama.cpp setup without any additional accounts or services. It's backend-agnostic by design.

### The honest summary

If you have a single powerful CUDA machine and need production-grade throughput: use vLLM.

If you have one machine and just want to run models: use Ollama.

If you have two or more machines — mixed vendors, mixed GPU generations, mixed budgets, some with no GPU at all — and want them to work together intelligently: that's what Tightwad is for.

That GTX 770 from 2013? Put it to work drafting tokens. The old Xeon server with no GPU? CPU drafting. Your gaming PC, your workstation, your NAS, your laptop — Tightwad doesn't judge what you have. It just pools it.

## RAM Management

llama-server mmaps the entire GGUF file into RAM before copying tensors to VRAM. On Windows, pages stay resident forever. On Linux, pages linger in the page cache. This means a 16 GB RAM machine can't load an 18 GB model even if the GPU has 24 GB VRAM.

Tightwad solves this with three tools:

### `tightwad load` — Pre-warm + Load + Reclaim

For standalone GGUF loading with memory-aware pre-warming:

```bash
tightwad load /path/to/model.gguf                    # auto pre-warm if needed
tightwad load /path/to/model.gguf --no-prewarm        # skip pre-warming
```

When the model exceeds 80% of available RAM, tightwad reads the file sequentially (with `posix_fadvise(SEQUENTIAL)` on Linux) to warm the page cache before llama-server mmaps it. After `/health` confirms the model is in VRAM, RAM is reclaimed.

This also happens automatically via `tightwad start` when `ram_reclaim` is `auto` or `on`.

### `tightwad reclaim` — Free RAM After Loading

After the model is fully loaded to VRAM (`/health` returns 200), tell the OS to release the file's page cache:

```bash
# Auto-reclaim after starting coordinator
tightwad start -m qwen3-32b --ram-reclaim on

# Or reclaim manually for any running llama-server
tightwad reclaim                        # auto-detects coordinator PID
tightwad reclaim --pid 12345            # any llama-server process
tightwad reclaim --pid 12345 --model-path /models/model.gguf
```

**How it works per platform:**

| Platform | Method | Effect |
|----------|--------|--------|
| Linux | `posix_fadvise(DONTNEED)` | Drops the GGUF file's page cache. Targeted — only affects that file. |
| Windows | `SetProcessWorkingSetSize(-1, -1)` | Trims working set, moves mmap'd pages to standby list. |
| macOS | No-op | Unified memory — GPU and CPU share physical RAM. Reclaim is unnecessary. |

**`ram_reclaim` modes** (in `cluster.yaml` or `--ram-reclaim` flag):

| Mode | Behavior |
|------|----------|
| `off` | Default llama-server behavior. No reclaim. |
| `on` | Always reclaim after model loads to VRAM. |
| `auto` | Reclaim if model > 50% of available RAM. Skip if plenty of headroom. (Default) |

### `tightwad tune` — System Readiness Check

For machines where the model is bigger than RAM, NVMe swap must be configured:

```bash
tightwad tune                              # general system check
tightwad tune --model /models/qwen3-32b.gguf  # check against specific model
```

Example output on a 16 GB machine with no swap:

```
System Resources:
  RAM:        16.0 GB (12.3 GB available)
  Swap:       0.0 GB (0.0 GB used)
  Swappiness: 60

Model: qwen3-32b-Q4_K_M.gguf (18.1 GB)

  [!] CRITICAL: No swap configured. This model (18.1 GB) exceeds available
      RAM (12.3 GB). Loading will fail. Configure NVMe swap:

      sudo fallocate -l 32G /swapfile
      sudo chmod 600 /swapfile
      sudo mkswap /swapfile
      sudo swapon /swapfile
      echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

  [i] Tip: After loading, run 'tightwad reclaim' to free RAM.
```

## CLI Reference

**Setup & Config**

| Command | Description |
|---------|-------------|
| `tightwad init` | Auto-discover LAN servers and generate cluster.yaml |
| `tightwad init --local` | Auto-detect local GPUs and generate coordinator-only config |
| `tightwad doctor [--fix] [--json]` | Diagnose config, binaries, network, versions, families, MoE |

**Speculative Decoding Proxy**

| Command | Description |
|---------|-------------|
| `tightwad proxy start` | Start proxy (dashboard at `/dashboard`) |
| `tightwad proxy stop / status` | Stop proxy or show health + acceptance rate |
| `tightwad chat [--direct]` | Interactive chat with inline stats (or bypass proxy) |
| `tightwad bench [--json]` | A/B benchmark: proxy vs direct target comparison |

**Quality Gate (CPU Fleet + GPU Verifier)**

| Command | Description |
|---------|-------------|
| `tightwad gate start` | Start quality gate proxy |
| `tightwad gate status` | Show approve/correct/reject rates |

**RPC Cluster**

| Command | Description |
|---------|-------------|
| `tightwad start [-m MODEL]` | Start coordinator (enforces version matching) |
| `tightwad stop` | Stop the coordinator |
| `tightwad status` | Show RPC cluster status |
| `tightwad swap MODEL` | Hot-swap model (workers persist) |
| `tightwad benchmark` | Benchmark the running coordinator |

**Peer Agent (Cross-Platform Management)**

| Command | Description |
|---------|-------------|
| `tightwad peer start [--port N]` | Start peer agent daemon (default port 9191) |
| `tightwad peer stop / status` | Stop or check peer agent |

**Models & Distribution**

| Command | Description |
|---------|-------------|
| `tightwad pull <model> [--list]` | Download GGUF from HuggingFace (ensure you've accepted the model's license) |
| `tightwad inspect <model.gguf> [--plan]` | Show model info, MoE details, distribution plan |
| `tightwad distribute MODEL [--method rsync\|swarm] [--dry-run]` | Copy model to workers |
| `tightwad swarm seed / pull / status` | P2P model distribution |
| `tightwad manifest create <model.gguf>` | Generate swarm manifest |

**System**

| Command | Description |
|---------|-------------|
| `tightwad logs [coordinator\|proxy] [-f] [--clear]` | View or tail logs |
| `tightwad tune [--model <gguf>]` | Check RAM/swap readiness |
| `tightwad load <model.gguf> [--no-prewarm]` | Memory-aware model loading |
| `tightwad reclaim [--pid PID]` | Free RAM after model loads to VRAM |
| `tightwad start --ram-reclaim auto` | Start with automatic RAM reclaim |
| `tightwad service install / uninstall / status` | Manage systemd/launchd service |
| `tightwad deploy <host> [--ssh-user USER]` | Deploy to remote host via SSH |

Global option: `-c /path/to/cluster.yaml` or `TIGHTWAD_CONFIG` env var. Config is auto-discovered from `./tightwad.yaml`, `./configs/cluster.yaml`, `~/.tightwad/config.yaml`, or the package default if `-c` is not provided.

`tightwad inspect` requires the `gguf` package: `pip install tightwad[inspect]`


## API Endpoints (Proxy)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Browser-based chat UI |
| `/dashboard` | GET | Live monitoring dashboard (health, charts, request log) |
| `/v1/completions` | POST | Text completion (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/v1/tightwad/status` | GET | Proxy stats: acceptance rate, rounds, throughput |
| `/v1/tightwad/events` | GET | SSE stream of live stats, health, and request events |
| `/v1/tightwad/ws` | WebSocket | Bidirectional dashboard updates (stats, health, commands) |
| `/v1/tightwad/history` | GET | JSON array of recent request records (max 50) |
| `/metrics` | GET | Prometheus text exposition format (requests, tokens, acceptance rate, uptime) |

All endpoints support `stream: true` for SSE streaming. The WebSocket endpoint at `/v1/tightwad/ws` enables bidirectional communication — live stats push and inbound commands (e.g., adjusting `max_draft_tokens` in real time). The chat UI at `/` provides an instant browser-based interface with streaming — no additional software required.

> **Chat Templates:** The `/v1/chat/completions` endpoint auto-detects the correct chat template from the target model family (Llama 3, Mistral, Gemma, Phi, DeepSeek, Command-R, ChatML/Qwen). Override with `proxy.chat_template` in cluster.yaml or `TIGHTWAD_CHAT_TEMPLATE` env var. Use `/v1/completions` with a pre-formatted prompt if you need full control over the prompt format.

> **Auto-Tune:** Set `max_draft_tokens: auto` in your config (or `TIGHTWAD_MAX_DRAFT_TOKENS=auto`) to let the proxy optimize draft count at runtime. It adjusts based on rolling acceptance rates and draft-vs-verify timing — increasing when drafts are cheap and acceptance is high, decreasing when they're not. Most users should use `auto`.

> **Draft-Verify Pipelining:** The proxy can overlap drafting round N+1 with verifying round N. When acceptance is high (73-100% with same-family models), the optimistic draft is ready immediately when verification completes, nearly doubling throughput.

## Hardware Setup

### RPC Workers

Each machine that contributes a GPU runs `rpc-server`. Download pre-built binaries from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) or build from source:

**Windows (CUDA):**
```bash
# Pre-built: download llama-b8079-bin-win-cuda-cu12.2.0-x64.zip
# rpc-server.exe is included alongside llama-server.exe
rpc-server.exe --host 0.0.0.0 --port 50052
```

**macOS (Metal):**
```bash
# Pre-built: download llama-b8079-bin-macos-arm64.tar.gz
# Restrict to Metal GPU only (avoids exposing CPU as second device):
./rpc-server --host 0.0.0.0 --port 50052 --device MTL0
```

**Linux (ROCm/CUDA):**
```bash
# Build from source:
cmake -B build -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)
./build/bin/rpc-server --host 0.0.0.0 --port 50052
```

**Important:** The coordinator's `llama-server` and all `rpc-server` instances **must be the same build version**. Version mismatches cause silent failures — no error, tensors just don't distribute. `tightwad start` enforces this automatically via SSH version checks and refuses to launch on mismatch. Use `--skip-version-check` to bypass if needed. `tightwad doctor` also reports version mismatches as errors.

### Coordinator

The coordinator runs `llama-server` with `--rpc` pointing to all workers:

```bash
llama-server -m model.gguf --host 0.0.0.0 --port 8090 -ngl 999 \
  --rpc 192.168.1.20:50052,192.168.1.30:50052 \
  --tensor-split 0.34,0.26,0.17,0.23 \
  --flash-attn --jinja
```

### Firewall Notes

`rpc-server` listens on port 50052 by default. You'll need to open this port:

- **Windows:** `New-NetFirewallRule -DisplayName 'llama-rpc' -Direction Inbound -Program 'C:\llama\rpc-server.exe' -Action Allow`
- **macOS:** Add `rpc-server` to System Settings → Network → Firewall → Options, or: `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/rpc-server && sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /path/to/rpc-server`
- **Linux:** `sudo ufw allow 50052/tcp`

## Benchmarking

Benchmark scripts measure acceptance rates and wall-clock speedup across model pairs. See the scripts in the `benchmarks/` directory and `scripts/benchmark_proxy.py` for full methodology and reproduction instructions.

```bash
# Multi-config benchmark (local Ollama + cloud APIs)
OPENROUTER_API_KEY="sk-or-..." .venv/bin/python scripts/benchmark_proxy.py

# Llama 3.1 8B (local) → Llama 3.1 405B (OpenRouter)
# Requires local llama-server running on port 8081
llama-server -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8081 -ngl 999
OPENROUTER_API_KEY="sk-or-..." .venv/bin/python scripts/benchmark_openrouter.py
```

Results are saved as JSON in `benchmarks/`.

## Development

```bash
git clone https://github.com/youngharold/tightwad.git
cd tightwad
pip install -e ".[dev]"
pytest tests/ --ignore=tests/e2e_loader_test.py -v
```

## Project Structure

```
tightwad/
├── cli/                # CLI package (split into subcommand modules)
│   ├── __init__.py     # Main Click group, shared helpers
│   ├── proxy.py        # proxy start/stop/status
│   ├── cluster.py      # start, stop, status, swap, benchmark, bench
│   ├── gate.py         # gate start/status (quality gate)
│   ├── peer.py         # peer start/stop/status
│   ├── tools.py        # doctor, init, inspect, distribute, deploy, etc.
│   └── service.py      # service install/uninstall/status
├── proxy.py            # Speculative decoding proxy (+ auto-tune, pipelining, consensus)
├── speculation.py      # Verification algorithms (greedy, stochastic, consensus)
├── config.py           # YAML/env config loader with SSRF validation
├── family.py           # Model family detection (Ollama, llama-server, GGUF)
├── chat_templates.py   # Per-family chat templates (Llama 3, Mistral, Gemma, Phi, etc.)
├── quality_gate.py     # CPU fleet drafts, GPU verifies (approve/correct/reject)
├── peer.py             # Peer agent HTTP daemon (replaces SSH)
├── tree_speculation.py # Tree-based speculation data structures
├── bench.py            # A/B benchmark: proxy vs direct target
├── coordinator.py      # llama-server lifecycle + version enforcement
├── worker.py           # RPC health checks + version detection (SSH + peer)
├── doctor.py           # Diagnostic checks (config, network, versions, families, MoE)
├── dashboard.py        # Live web dashboard (SSE + WebSocket)
├── gguf_reader.py      # Pure-Python GGUF v2/v3 parser (zero deps)
├── gguf_inspect.py     # GGUF analysis + MoE detection + distribution planning
├── ssrf.py             # SSRF protection (scheme, private-IP, DNS-rebinding)
├── validation.py       # Request body validation + DoS limits
├── init_wizard.py      # LAN auto-discovery + config generator
├── reclaim.py          # Cross-platform RAM reclaim after model loading
├── loader.py           # Memory-aware model pre-warming
├── distribute.py       # Model distribution (rsync/scp + swarm P2P)
├── manifest.py         # Swarm manifest + piece bitfield tracking
├── swarm_transfer.py   # P2P seeder + puller
├── gpu_detect.py       # GPU auto-detection (NVIDIA, AMD, Apple)
├── service.py          # systemd/launchd service management
├── model_hub.py        # HuggingFace GGUF downloader
├── deploy.py           # Remote deployment via SSH
└── tune.py             # System RAM/swap diagnostics
tests/                  # 765+ tests (unit + integration)
configs/                # Example cluster.yaml configs
```

## License

Tightwad is provided as-is under the [MIT License](LICENSE). Performance claims are based on our testing and may vary with your hardware, model pairing, network, and configuration. All trademarks are the property of their respective owners.
