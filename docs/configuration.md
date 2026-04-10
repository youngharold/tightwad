# Configuration Reference

Tightwad uses a YAML config file, typically at `configs/cluster.yaml`. Override the path with `-c` or the `TIGHTWAD_CONFIG` environment variable.

## Full Example

```yaml
coordinator:
  host: 127.0.0.1
  port: 8080
  binary: /usr/local/bin/llama-server
  extra_args: ["--flash-attn"]

workers:
  - name: gpu0
    host: 192.168.1.10
    port: 50052
    gpu:
      vendor: nvidia
      model: RTX 4070 Ti Super
      vram_gb: 16
    model_dir: /models

  - name: gpu1
    host: 192.168.1.20
    port: 50052
    gpu:
      vendor: amd
      model: RX 7900 XTX
      vram_gb: 24
    model_dir: /models

models:
  default: llama-70b
  llama-70b:
    name: Llama 3.3 70B
    path: /models/llama-3.3-70b-instruct-Q4_K_M.gguf
    tensor_split: "16,24"

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
  max_draft_tokens: 32
```

## Sections

### `coordinator`

The machine that runs `llama-server` and distributes layers to RPC workers.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `127.0.0.1` | Bind address for the coordinator |
| `port` | int | `8080` | API port (OpenAI-compatible at `/v1`) |
| `binary` | string | `llama-server` | Path to llama-server binary |
| `extra_args` | list | `[]` | Additional args passed to llama-server (e.g. `--flash-attn`) |

> **RAM requirement:** The coordinator needs enough system RAM for the full GGUF file. llama.cpp mmaps the entire file before distributing tensors. A 70B Q4_K_M (~40GB) needs ~44GB RAM on the coordinator.

### `workers`

RPC workers that contribute GPU (or CPU) compute to the pool.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Human-readable name |
| `host` | string | yes | IP or hostname |
| `port` | int | yes | RPC port (default: 50052) |
| `gpu.vendor` | string | no | `nvidia`, `amd`, or `cpu` |
| `gpu.model` | string | no | GPU model name (for display) |
| `gpu.vram_gb` | int | no | VRAM in GB (used for tensor split calculation) |
| `model_dir` | string | no | Path to models on this worker (for `tightwad distribute`) |

### `models`

Model definitions. The `default` key specifies which model loads on `tightwad start`.

| Field | Type | Description |
|-------|------|-------------|
| `default` | string | Key of the default model |
| `<name>.name` | string | Display name |
| `<name>.path` | string | Path to GGUF file (on coordinator) |
| `<name>.tensor_split` | string | Comma-separated VRAM allocation per GPU |

**Tensor split:** Proportional to each GPU's VRAM. For a 16GB + 24GB setup, use `"16,24"`. Order matches the `workers` list.

### `proxy`

Speculative decoding proxy configuration. Optional — only needed if you want speculation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `0.0.0.0` | Bind address |
| `port` | int | `8088` | Proxy listen port |
| `max_draft_tokens` | int | `32` | Tokens per draft round |

#### `proxy.draft` / `proxy.target`

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Server URL (e.g. `http://192.168.1.10:11434`) |
| `model_name` | string | Model name as the server knows it |
| `backend` | string | `ollama` or `llamacpp` |

#### `proxy.drafters` (multi-drafter mode)

Instead of a single `draft`, you can specify multiple drafters. Tightwad picks the fastest one per round.

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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TIGHTWAD_CONFIG` | Path to cluster.yaml (overrides default) |

## Validation

Run `tightwad doctor` to check your config for common issues:

```bash
tightwad doctor        # Check everything
tightwad doctor --fix  # Show suggested fix commands
tightwad doctor --json # Machine-readable output
```
