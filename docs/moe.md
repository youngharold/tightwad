# MoE Support in Tightwad

Tightwad treats Mixture-of-Experts models as first-class citizens in v0.5. This
doc covers what Tightwad does for MoE, how the placement algorithm works, and
where the tooling stops short.

## What you get

| Feature | What it does | Command |
|---|---|---|
| GGUF defusion | Rewrites fused expert tensors to indexed form so per-expert placement is possible | `tightwad moe defuse` |
| Expert-aware placement | Keeps whole experts on one card; `--override-tensor` flags emitted to llama-server | `moe_placement: balanced` |
| Profile-guided placement | Pins frequently-hit experts to the fastest device | `moe_placement: profile-guided` |
| Hot-expert profiling | Captures per-expert hit counts from a running cluster | `tightwad moe profile` |
| MoE benchmark | Speculation-over-MoE-pool A/B with TTFT, acceptance, expert skew | `tightwad moe bench` |

## Defusion first

Most modern MoE GGUFs ship **fused** expert weights (`blk.L.ffn_*_exps.weight`)
— one tensor per layer covering every expert. llama.cpp can't per-expert-split
a fused tensor, so placement degrades to whole-layer splits. Tightwad's
defusion pass rewrites fused tensors into indexed form
(`blk.L.ffn_*.E.weight`) by slicing along the expert dimension:

```
tightwad moe defuse gpt-oss-120b.gguf gpt-oss-120b-indexed.gguf
```

Output size equals input size. No quantization change. Load the indexed file
with llama-server exactly like the fused original.

## Config

```yaml
models:
  gpt-oss-120b:
    path: /models/gpt-oss-120b-indexed.gguf
    moe_placement: balanced          # off | balanced | profile-guided
    moe_hot_profile: ~/.tightwad/moe-profile.json   # required for profile-guided
```

Default behavior is unchanged if `moe_placement` is omitted.

## How placement works

1. Every `(layer, expert)` unit is sized from the GGUF tensor inventory.
2. Each GPU becomes a `DeviceSlot` with capacity = `vram_gb * 0.85 - routing_overhead_gb`.
3. Units are bin-packed proportionally to VRAM.
4. Profile-guided placement weights hot experts by `1 + 3.0 * frequency` and
   pins the top-K to the highest-scoring device.
5. One `--override-tensor` flag is emitted per `(layer, device)` pair.

Device scoring uses real TCP-RTT measurements from `tightwad moe device-bench`
(cached 24h at `~/.tightwad/device-scores.json`). Coordinator-local GPUs get a
baseline score; RPC workers scale inversely with measured latency.

## Profile-guided in practice

```
# 1. Start the coordinator normally (balanced placement for the initial run)
tightwad start

# 2. Capture a profile while the cluster handles representative traffic
tightwad moe profile --follow-coord --duration 300 -o ~/.tightwad/profile.json

# 3. Summarize to see which experts dominate
tightwad moe summary ~/.tightwad/profile.json

# 4. Flip placement to profile-guided and restart
#    (update moe_placement + moe_hot_profile in cluster.yaml)
tightwad stop && tightwad start
```

### Hot-expert capture requires instrumented llama.cpp

llama.cpp's public C API does not expose expert-routing decisions. Tightwad
ships a 15-line patch at `scripts/patches/llamacpp-moe-log.patch` that adds an
`LLAMA_LOG_MOE=1`-gated log line:

```
moe: layer=12 chosen=[47,88,12,3]
```

Apply the patch, rebuild llama.cpp, and set `LLAMA_LOG_MOE=1` in `env:` in
`cluster.yaml`. `tightwad doctor` warns when profile-guided placement is
configured without the env var.

Without the patch, profiles still populate `total_tokens` from
`n_expert_used` counts but `hits` stays empty, and profile-guided degrades to
balanced.

## MoE-specific benchmarking

```
tightwad moe bench \
  --target-url http://192.168.86.29:1234 \
  --target-model minimax/minimax-m2.5 \
  --max-tokens 256 \
  --json benchmarks/minimax-m2.5-benchmark.json
```

The bench streams a per-prompt table (TTFT, direct tok/s, proxy tok/s, rolling
acceptance, speedup). Output JSON mirrors the standard proxy-bench schema plus
a `moe` section with acceptance rate and total rounds.

LM Studio's OpenAI-compatible endpoint on `:1234` works as-is; no proxy
changes required.

## When NOT to use placement

- **Model fits on one GPU.** Single-GPU inference is always faster than any
  split. Leave `moe_placement` unset.
- **Model is dense.** Doctor will warn if you enable it anyway.
- **You're running over 2.4 GHz WiFi with a 229B target.** Consider the
  Combined Mode (speculation over RPC pool) benchmarks in `docs/` first —
  without speculation the pool is slower than idle.

## Limitations

- Quantized GGUF defusion relies on contiguous per-expert byte slicing. Works
  for every known MoE layout today but may break on future formats. Tests
  validate round-trip equivalence on F32 fixtures.
- `-ot` flag count is ~O(layers). A 128-expert × 32-layer model emits ~32
  flags (one per layer with a multi-expert alternation). `ps` output gets
  chatty.
- Profile-guided is experimental pending upstream llama.cpp hook support.

## Related files

- `tightwad/moe_placement.py` — bin-pack + regex emission.
- `tightwad/moe_defuse.py` — GGUF rewriter.
- `tightwad/moe_profile.py` — stderr parser + profile persistence.
- `tightwad/moe_device_bench.py` — device scoring.
- `tightwad/cli/moe.py` — `tightwad moe …` commands.
