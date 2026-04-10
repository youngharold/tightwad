# Quickstart

From zero to running inference in 5 minutes.

## Install

```bash
pip install tightwad
```

Or from source:
```bash
git clone https://github.com/youngharold/tightwad.git
cd tightwad
pip install -e .
```

## Prerequisites

You need [llama.cpp](https://github.com/ggml-org/llama.cpp) built with RPC support on each machine:

```bash
cmake -B build -DGGML_RPC=ON -DGGML_CUDA=ON  # or -DGGML_HIP=ON for AMD
cmake --build build --config Release -j
```

And a GGUF model file. Download one from [HuggingFace](https://huggingface.co/models?search=gguf).

## Option A: Speculative Decoding (fastest to set up)

If you have two machines — one with a big model, one with a small model — you can get speculative decoding in 2 minutes.

### 1. Start both servers

**Machine A** (draft — any CPU or cheap GPU):
```bash
# Using Ollama (easiest)
ollama serve  # default port 11434
ollama pull qwen3:1.7b
```

**Machine B** (target — your GPU):
```bash
# Using Ollama
ollama serve
ollama pull qwen3:32b
```

### 2. Auto-generate config

```bash
tightwad init \
  --draft-url http://192.168.1.10:11434 \
  --draft-model qwen3:1.7b \
  --target-url http://192.168.1.20:11434 \
  --target-model qwen3:32b
```

### 3. Start the proxy

```bash
tightwad proxy start
```

### 4. Use it

Point your chat UI (Open WebUI, etc.) at `http://localhost:8088/v1` and chat normally. Same model, same output, faster.

Check stats:
```bash
tightwad proxy status
```

## Option B: GPU Pool (for models that don't fit on one card)

### 1. Start RPC workers on each machine

```bash
# On each worker machine:
llama-rpc-server --host 0.0.0.0 --port 50052
```

### 2. Auto-discover and generate config

```bash
tightwad init
```

This scans your LAN for inference servers and walks you through setup.

### 3. Validate

```bash
tightwad doctor
```

### 4. Start the coordinator

```bash
tightwad start
```

### 5. Verify

```bash
tightwad status
```

Your pooled endpoint is at `http://localhost:8080/v1`.

## Option C: Combined Mode (pool + speculation)

Set up the GPU pool (Option B), then add a `proxy` section to your `cluster.yaml` pointing at a draft server. See [examples/combined-mode.yaml](../examples/combined-mode.yaml).

```bash
tightwad start        # Start the pool
tightwad proxy start  # Start speculation on top
```

Your endpoint is `http://localhost:8088/v1` (the proxy).

## Next Steps

- [Configuration Reference](configuration.md) — every cluster.yaml field explained
- [Architecture](architecture.md) — how the pieces fit together
- [Examples](../examples/) — sample configs for common setups
- Run `tightwad benchmark` to measure your setup
