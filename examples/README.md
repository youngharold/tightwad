# Example Configurations

Copy-paste-edit these for your setup. Replace IPs and model paths with your own.

| File | Setup | GPUs |
|------|-------|------|
| [minimal-spec-decode.yaml](minimal-spec-decode.yaml) | Simplest speculative decoding (2 Ollama servers) | Any 2 machines |
| [cpu-draft-gpu-target.yaml](cpu-draft-gpu-target.yaml) | CPU drafts, GPU verifies | 1 GPU + 1 spare CPU |
| [two-gpu-single-machine.yaml](two-gpu-single-machine.yaml) | RPC pool on one box | 2 GPUs, same machine |
| [mixed-vendor-pool.yaml](mixed-vendor-pool.yaml) | NVIDIA + AMD pool | Mixed vendors |
| [combined-mode.yaml](combined-mode.yaml) | Speculation over GPU pool (killer feature) | 4+ GPUs across machines |

## Quick Start

1. Pick the example closest to your setup
2. Copy it to `configs/cluster.yaml`
3. Replace IPs, ports, and model paths
4. Run `tightwad doctor` to validate
5. Start with `tightwad start` (pool) or `tightwad proxy start` (speculation)
