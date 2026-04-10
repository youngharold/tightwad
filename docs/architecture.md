# Architecture

How Tightwad's components fit together.

## Components

```
┌─────────────────────────────────────────────────────────┐
│                     Your Chat UI                         │
│              (Open WebUI, ChatBot UI, etc.)              │
└──────────────────────┬──────────────────────────────────┘
                       │ OpenAI API
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Tightwad Proxy (:8088)                  │
│            Speculative Decoding Coordinator              │
│                                                         │
│  ┌─────────────┐              ┌──────────────────────┐  │
│  │ Draft Client │              │   Target Client      │  │
│  └──────┬──────┘              └──────────┬───────────┘  │
└─────────┼───────────────────────────────┼───────────────┘
          │                               │
          ▼                               ▼
┌──────────────────┐          ┌──────────────────────────┐
│   Draft Server   │          │     Coordinator (:8080)  │
│  (Ollama/llama)  │          │      llama-server        │
│                  │          │                          │
│  Small model     │          │  Distributes layers to   │
│  e.g. 1.7B      │          │  RPC workers via tensor   │
│  Fast, cheap     │          │  parallelism             │
└──────────────────┘          └─────────┬────────────────┘
                                        │ RPC
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │  RPC Worker  │  │  RPC Worker  │  │  RPC Worker  │
            │  GPU 0       │  │  GPU 1       │  │  GPU 2       │
            │  (NVIDIA)    │  │  (AMD)       │  │  (NVIDIA)    │
            └──────────────┘  └──────────────┘  └──────────────┘
```

## Three Modes

### Mode 1: RPC Pool Only

Pool GPUs into one endpoint. No speculation.

```
Chat UI  →  Coordinator (:8080)  →  RPC Workers (GPU 0, GPU 1, ...)
```

The coordinator runs `llama-server` with `--rpc` flags pointing at each worker. Model layers are distributed across GPUs based on `tensor_split`. Generation is autoregressive — one token at a time, each requiring a round-trip to all workers.

**When to use:** Model doesn't fit on one GPU, and you don't have a spare machine for drafting.

### Mode 2: Speculative Decoding Only

Two servers, no pooling. Draft + verify.

```
Chat UI  →  Proxy (:8088)  →  Draft Server (small model)
                            →  Target Server (big model)
```

The proxy coordinates the speculation loop. No RPC workers involved — both servers run independently.

**When to use:** Big model fits on one GPU, you just want it faster.

### Mode 3: Combined (the killer feature)

Speculation on top of a GPU pool.

```
Chat UI  →  Proxy (:8088)  →  Draft Server (small model)
                            →  Coordinator (:8080)  →  RPC Workers
```

The pool is slow autoregressive (3 tok/s over WiFi), but batch verification amortizes the overhead — 32 tokens verified per round instead of 1 token per round-trip.

**When to use:** Model doesn't fit on one GPU AND you want it faster.

## Speculative Decoding Flow

```
1. Chat UI sends prompt to Proxy

2. Proxy forwards prompt to Draft Server
   Draft Server generates K candidate tokens (fast, ~30 tok/s)
   Returns: [t1, t2, t3, ..., tK]

3. Proxy sends prompt + candidates to Target Server
   Target Server verifies ALL K tokens in ONE batch forward pass
   Returns: [accept, accept, accept, reject, ...]

4. Proxy accepts verified tokens, discards from first rejection
   Accepted: [t1, t2, t3]  (3 tokens from 1 round-trip to target)

5. Proxy streams accepted tokens back to Chat UI

6. Repeat from step 2 with updated context
```

**Why this is fast:** Without speculation, the target generates 1 token per forward pass. With speculation, it verifies K tokens in 1 forward pass (batch verification is almost as cheap as single-token generation). If the draft model is good, most tokens get accepted.

**Why output is identical:** The target model has final say on every token. Rejected tokens are replaced with the target's choice. With greedy decoding (temperature=0), output is mathematically equivalent to running the target alone.

## Network Traffic

- **RPC (pool):** Tensor data between coordinator and workers. Can be significant over WiFi.
- **Speculation:** Token IDs only (bytes, not megabytes). Draft → Proxy → Target. Negligible bandwidth.

This is why combined mode works over WiFi: the expensive RPC round-trips happen less often (once per batch of 32 tokens instead of once per token).

## File Layout

```
tightwad/
├── cli.py              # Click CLI entry point
├── config.py           # YAML config loading and validation
├── coordinator.py      # llama-server process management
├── worker.py           # RPC worker health checks
├── proxy.py            # Speculative decoding proxy (Starlette app)
├── speculation.py      # Core speculation loop (draft → verify → accept)
├── dashboard.py        # Live web dashboard for proxy stats
├── doctor.py           # Diagnostic checks
├── init_wizard.py      # LAN scanner and config generator
├── distribute.py       # Model file distribution to workers
├── manifest.py         # Swarm manifest (chunked file hashing)
├── swarm_transfer.py   # P2P model transfer (seeder/puller)
└── gguf_inspect.py     # GGUF metadata and tensor inspection
```

## Dependencies

- **llama.cpp** — The inference engine. Tightwad manages it, doesn't replace it.
- **Ollama** — Optional. Can serve as draft or target backend.
- **Python 3.10+** — Tightwad itself is pure Python.
- **httpx** — HTTP client for health checks and API calls.
- **Starlette/Uvicorn** — Proxy server.
- **Rich** — Terminal output formatting.
- **Click** — CLI framework.
- **PyYAML** — Config parsing.
