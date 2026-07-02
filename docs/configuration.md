# Configuration Reference

Tightwad uses a YAML config file, typically at `configs/cluster.yaml`. Override the path with `-c` or the `TIGHTWAD_CONFIG` environment variable.

## Full Example

```yaml
coordinator:
  host: 0.0.0.0
  port: 8080
  backend: cuda          # "cuda" (NVIDIA) or "hip" (AMD/ROCm)
  gpus:                  # Local GPUs on the coordinator machine
    - name: "RTX 4070 Ti Super"
      vram_gb: 16
    - name: "RTX 3060"
      vram_gb: 12
  extra_args: ["--flash-attn", "on"]

workers:                 # Remote machines running rpc-server
  - host: 192.168.1.20
    gpus:
      - name: "RTX 2070"
        vram_gb: 8
        rpc_port: 50052

models:
  llama-70b:
    name: Llama 3.3 70B
    path: /models/llama-3.3-70b-instruct-Q4_K_M.gguf
    ctx_size: 8192
    flash_attn: true
    default: true

binaries:
  coordinator: /usr/local/bin/llama-server
  rpc_server: /usr/local/bin/rpc-server

proxy:
  host: 0.0.0.0
  port: 8088
  draft:
    url: http://192.168.1.30:11434
    model_name: qwen3:1.7b
    backend: ollama
  target:
    url: http://127.0.0.1:8080
    model_name: llama-70b
    backend: llamacpp
  max_draft_tokens: 8
```

## Sections

### `coordinator`

The machine that runs `llama-server` and distributes layers to RPC workers.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `0.0.0.0` | Bind address for the coordinator |
| `port` | int | `8080` | API port (OpenAI-compatible at `/v1`) |
| `backend` | string | `hip` | `cuda` (NVIDIA) or `hip` (AMD/ROCm); selects auto-injected presets |
| `gpus` | list | `[]` | Local GPUs on the coordinator machine (see below) |
| `extra_args` | list | backend preset | Additional args passed to llama-server (e.g. `["--flash-attn", "on"]`). Overrides the backend preset when set. |
| `env` | map | backend preset | Environment variables for llama-server; merged over (and overriding) the backend preset env |

Each entry under `coordinator.gpus` takes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | GPU name (for display) |
| `vram_gb` | int | yes | VRAM in GB (used for tensor-split calculation) |

> **Note:** the llama-server / rpc-server binary paths are configured under the top-level `binaries:` section (below), **not** under `coordinator`.

> **RAM requirement:** The coordinator needs enough system RAM for the full GGUF file. llama.cpp mmaps the entire file before distributing tensors. A 70B Q4_K_M (~40GB) needs ~44GB RAM on the coordinator.

### `workers`

Remote machines running `rpc-server` that contribute GPU (or CPU) compute to the pool.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `host` | string | yes | IP or hostname |
| `ssh_user` | string | no | SSH user (enables remote version checks / `tightwad distribute`) |
| `model_dir` | string | no | Path to models on this worker (for `tightwad distribute`) |
| `peer_port` | int | no | Peer-agent port on this worker (default 9191) |
| `gpus` | list | yes | GPUs this worker contributes (see below) |

Each entry under a worker's `gpus` list takes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | GPU name (for display) |
| `vram_gb` | int | yes | VRAM in GB (used for tensor-split calculation) |
| `rpc_port` | int | yes | Port the worker's `rpc-server` listens on (e.g. 50052) |

**Tensor split:** Tightwad computes the split automatically, proportional to each GPU's `vram_gb` (coordinator GPUs first, then worker GPUs). There is no manual `tensor_split` field.

### `models`

Model definitions. Each key is the model's config name; mark the one to load on `tightwad start` with `default: true`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `<name>.name` | string | — | Display name |
| `<name>.path` | string | — | Path to GGUF file (absolute, on coordinator) |
| `<name>.ctx_size` | int | `8192` | Context window size |
| `<name>.predict` | int | `4096` | Max tokens to predict (`-n`) |
| `<name>.flash_attn` | bool | `true` | Enable flash attention (legacy string `"on"`/`"off"` also accepted) |
| `<name>.default` | bool | `false` | Load this model on `tightwad start` |
| `<name>.moe_placement` | string | — | Expert-placement strategy for MoE models (e.g. `balanced`, `profile-guided`) |
| `<name>.moe_hot_profile` | string | — | Path to a hot-expert profile (for `moe_placement: profile-guided`) |

### `binaries`

Paths to the executables Tightwad launches. Optional — both resolve from `PATH` by default.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `coordinator` | string | `llama-server` | Path to the llama-server binary |
| `rpc_server` | string | `rpc-server` | Path to the rpc-server binary |

### `proxy`

Speculative decoding proxy configuration. Optional — only needed if you want speculation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `0.0.0.0` | Bind address |
| `port` | int | `8088` | Proxy listen port |
| `max_draft_tokens` | int \| `"auto"` | `8` | Tokens per draft round; `auto` enables cost-aware adaptive tuning |
| `fallback_on_draft_failure` | bool | `true` | Fall back to target-only generation if the draft fails |
| `auth_token` | string | — | Bearer token protecting `/v1/` (also from `TIGHTWAD_PROXY_TOKEN`). Required to bind a non-loopback host. |
| `consensus_mode` | string | `off` | `off`, `strict`, `majority`, or `any` (multi-drafter consensus) |
| `chat_template` | string | `auto` | Chat template (`auto` detects Llama/Mistral/Gemma/Phi/DeepSeek/Command-R) |
| `max_tokens_limit` | int | `16384` | Reject requests asking for more than this many tokens |

#### `proxy.draft` / `proxy.target`

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Server URL (e.g. `http://192.168.1.10:11434`) |
| `model_name` | string | Model name as the server knows it |
| `backend` | string | `ollama` or `llamacpp` (default `llamacpp`) |

#### `proxy.drafters` (multi-drafter mode)

Instead of a single `draft`, you can specify multiple drafters. Tightwad picks the fastest one per round (or votes across them in `consensus_mode`).

```yaml
proxy:
  drafters:
    - url: http://192.168.1.10:11434
      model_name: qwen3:1.7b
      backend: ollama
    - url: http://192.168.1.20:11434
      model_name: llama3.2:1b
      backend: ollama
  target:
    url: http://192.168.1.30:8080
    model_name: qwen3-32b
    backend: llamacpp
```

### `peer`

Optional local peer-agent daemon (`tightwad peer start`) used for cross-machine coordination.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `0.0.0.0` | Bind address |
| `port` | int | `9191` | Peer-agent port |
| `auth_token` | string | — | Bearer token (also from `TIGHTWAD_PEER_TOKEN`). Required to bind a non-loopback host — the agent can spawn/kill processes. |
| `model_dirs` | list | `[]` | Directories the agent scans for GGUF files |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TIGHTWAD_CONFIG` | Path to cluster.yaml (overrides default) |
| `TIGHTWAD_PROXY_TOKEN` | Proxy bearer token (overrides `proxy.auth_token`) |
| `TIGHTWAD_PEER_TOKEN` | Peer-agent bearer token (overrides `peer.auth_token`) |
| `TIGHTWAD_MAX_DRAFT_TOKENS` | Default `max_draft_tokens` (default `8`) |
| `TIGHTWAD_ALLOW_UNAUTHENTICATED` | Set `true` to allow a non-loopback bind without a token |

## Validation

Run `tightwad doctor` to check your config for common issues:

```bash
tightwad doctor        # Check everything
tightwad doctor --fix  # Show suggested fix commands
tightwad doctor --json # Machine-readable output
```
