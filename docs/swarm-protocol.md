# Tightwad Swarm Protocol

BitTorrent-inspired peer-to-peer model distribution and cooperative inference for distributed consumer GPUs.

## Overview

The swarm protocol extends Tightwad from a manually-configured homelab tool into a self-organizing network of consumer GPUs that cooperate on inference. Three layers:

1. **Model Distribution** — get the model to the machines that need it (torrent-style)
2. **Peer Discovery** — find available GPUs without manual config (DHT-style)
3. **Cooperative Inference** — share GPU cycles fairly across the network (tit-for-tat)

Each layer is independently useful and can be adopted incrementally.

---

## Layer 1: Swarm Model Distribution

### Problem

`tightwad distribute` currently uses single-source rsync — one machine uploads to all workers sequentially. A 70B Q4_K_M GGUF is ~40GB. Uploading to 5 workers = 200GB of bandwidth from one machine. Over the internet, this takes hours.

### Solution: Torrent-Style Chunked Transfer

Split the GGUF into fixed-size pieces (default 64MB), hash each piece, and let machines pull pieces from any peer that has them. As each machine finishes a piece, it immediately becomes a seeder for that piece.

#### Manifest File (`.tightwad.manifest`)

Generated once per model, stored alongside the GGUF or distributed independently:

```json
{
  "model": "Llama-3.3-70B-Instruct-Q4_K_M",
  "filename": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
  "total_size": 42949672960,
  "piece_size": 67108864,
  "pieces": [
    {"index": 0, "offset": 0, "size": 67108864, "sha256": "a1b2c3..."},
    {"index": 1, "offset": 67108864, "size": 67108864, "sha256": "d4e5f6..."},
    ...
  ],
  "metadata": {
    "architecture": "llama",
    "parameters": "70B",
    "quantization": "Q4_K_M",
    "context_length": 131072,
    "min_ram_gb": 44,
    "compatible_drafters": ["Llama-3.1-8B-Instruct"]
  }
}
```

The `compatible_drafters` field solves the same-family problem upfront — peers know which draft models work before downloading anything.

#### Transfer Protocol

```
Peer A (has full model)          Peer B (new, wants model)         Peer C (new, wants model)
        │                                │                                │
        ├─── piece 0 ──────────────────►│                                │
        ├─── piece 1 ──────────────────►│                                │
        │                                ├─── piece 0 ──────────────────►│  (B seeds to C)
        ├─── piece 2 ──────────────────►│                                │
        │                                ├─── piece 1 ──────────────────►│  (B seeds to C)
        ├─── piece 3 ─────────────────────────────────────────────────►│  (A sends to C directly)
        │                                ├─── piece 2 ──────────────────►│  (B seeds to C)
```

With N peers, effective bandwidth scales roughly as O(N) — each completed piece creates a new seeder. A 40GB model distributes to 10 machines in roughly the time it takes to transfer once, not 10x.

#### Piece Verification

Each piece is verified against the SHA256 hash in the manifest before being committed to disk. Corrupted pieces are re-downloaded from a different peer. This is critical for internet transfers where bit rot and interrupted connections are common.

#### Resume Support

Each peer tracks which pieces it has in a local bitfield file (`.tightwad.pieces`). Interrupted transfers resume from the last verified piece — no re-downloading completed work.

#### CLI

```bash
# Generate manifest for a model
tightwad manifest create ~/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf

# Seed a model (announce to swarm that you have it)
tightwad swarm seed llama-3.3-70b

# Download a model from the swarm
tightwad swarm pull llama-3.3-70b

# Check swarm status for a model
tightwad swarm status llama-3.3-70b
```

---

## Layer 2: Peer Discovery (DHT)

### Problem

Current setup requires manually editing `cluster.yaml` with IP addresses, ports, GPU specs, and model paths. This works for a homelab but breaks for internet pooling where peers are dynamic.

### Solution: Lightweight DHT

A simplified Kademlia-style DHT where each peer announces:

```json
{
  "peer_id": "a1b2c3d4...",
  "endpoint": "100.64.0.5:50052",
  "capabilities": {
    "gpus": [
      {"name": "RTX 4070 Ti Super", "vram_gb": 16, "backend": "cuda"}
    ],
    "ram_gb": 64,
    "models_available": ["llama-3.3-70b", "llama-3.1-8b"],
    "roles": ["rpc_worker", "drafter"],
    "bandwidth_mbps": 500
  },
  "reputation": {
    "uptime_hours": 240,
    "tokens_served": 1500000,
    "tokens_consumed": 800000,
    "ratio": 1.875
  }
}
```

#### Bootstrap

New peers bootstrap by connecting to one known peer (or a well-known bootstrap node). From there, the DHT self-organizes. For Tailscale networks, peers can be discovered via Tailscale's DNS (all machines in the tailnet are reachable by hostname).

```bash
# Join a swarm via bootstrap peer
tightwad swarm join --bootstrap 100.64.0.1:9090

# Join via Tailscale (auto-discovers peers on the tailnet)
tightwad swarm join --tailscale

# List discovered peers
tightwad swarm peers
```

#### Automatic Topology Formation

Given the set of available peers and their capabilities, the swarm automatically determines:

1. **Coordinator** — peer with the most RAM (for mmap)
2. **RPC workers** — peers with GPUs, ordered by VRAM
3. **Drafters** — peers with compatible small models loaded
4. **Tensor split** — calculated from actual VRAM across all active peers

This replaces manual `cluster.yaml` configuration entirely. The swarm is the config.

#### Heartbeat and Churn

Peers send heartbeats every 30 seconds. If a peer misses 3 heartbeats (90s), it's considered dead and its layers are redistributed. The coordinator triggers a model reload with an updated tensor-split excluding the dead peer.

For graceful departure, peers announce they're leaving so redistribution starts immediately.

---

## Layer 3: Cooperative Inference (Tit-for-Tat)

### Problem

Why would strangers share GPU cycles? Without incentives, an internet GPU pool would be dominated by freeloaders who consume inference but never contribute.

### Solution: BitTorrent-Style Tit-for-Tat

Every peer maintains a local ledger of tokens served vs. tokens consumed. The ratio determines priority:

| Ratio | Status | Effect |
|-------|--------|--------|
| > 1.5 | Super seeder | Priority queue, lowest latency slot assignment |
| 1.0 - 1.5 | Fair | Normal queue |
| 0.5 - 1.0 | Leech-ish | Deprioritized, may wait for idle capacity |
| < 0.5 | Freeloader | Rate-limited, last priority |
| 0.0 | New peer | Grace period (first 10,000 tokens free) |

#### How It Works

```
Alice (ratio 1.8, super seeder)    Bob (ratio 0.3, freeloader)
        │                                │
        │── inference request ──►        │── inference request ──►
        │                                │
        │◄── immediate serve ───         │◄── queued, wait... ───
        │    (priority slot)              │    (deprioritized)
```

Alice contributes more than she consumes, so she gets priority. Bob consumes without contributing, so he waits. This naturally incentivizes contribution without any payment system.

#### Accounting

Ledgers are maintained locally by each peer and cross-signed by the serving peer. No blockchain, no central authority. If peers disagree on accounting, they simply stop peering with each other (like BitTorrent's choking mechanism).

```json
{
  "entry": {
    "from": "alice_peer_id",
    "to": "bob_peer_id",
    "tokens_served": 5000,
    "timestamp": "2026-02-17T22:30:00Z",
    "model": "llama-3.3-70b"
  },
  "signature": "bob_signs_receipt..."
}
```

#### Optimistic Unchoking

Like BitTorrent, periodically give a random deprioritized peer a chance to prove itself. This prevents permanent lockout and lets new peers build reputation.

### Rarest-Layer Replication

The swarm tracks which layers are held by which peers. Layers with fewer replicas are higher priority for replication:

```
Layer 0-15:  held by 4 peers (healthy)
Layer 16-31: held by 4 peers (healthy)
Layer 32-47: held by 2 peers (⚠️ replicate!)
Layer 48-63: held by 3 peers (okay)
```

When a new peer joins with spare VRAM, it preferentially loads the rarest layers. This ensures the model stays available even as peers churn.

### Speculative Branching Swarm

Extend multi-drafter from LAN to internet scale:

```
Peer A (Llama 3.1 8B, latency 15ms)  ──► draft 32 tokens ──┐
Peer B (Llama 3.1 8B, latency 45ms)  ──► draft 32 tokens ──┤
Peer C (Llama 3.1 8B, latency 80ms)  ──► draft 32 tokens ──┼──► pick best ──► target verifies
Peer D (Llama 3.1 3B, latency 20ms)  ──► draft 32 tokens ──┤
                                                              │
                                              (first response + grace period,
                                               prefer highest acceptance history)
```

Over the internet, drafter latency varies. The proxy races multiple remote drafters (existing multi-drafter architecture scales to this) and picks the winner. The grace period adapts to observed latency — high-latency peers get more time before being cancelled.

---

## Security and Privacy

### Transport Encryption

All swarm communication runs over WireGuard/Tailscale. Tensor data, token IDs, and peer messages are encrypted in transit. No plaintext GPU data crosses the public internet.

### Inference Privacy

**Important limitation:** The coordinator and RPC workers see intermediate tensor activations. A malicious peer acting as RPC worker could potentially reconstruct input/output tokens from tensor data.

Mitigations:
- **Trusted swarms only** — peers explicitly accept each other (Tailscale ACLs, invite-only)
- **Drafter-only mode** — contribute as a drafter (sees only the draft model's output, not the target's) for reduced trust requirements
- **Future work:** Encrypted tensor computation (homomorphic or secure enclaves), though this is a research problem not a near-term feature

### Model Licensing

The swarm distributes model weights. Peers are responsible for complying with model licenses (Llama's community license, etc.). The manifest includes a `license` field for visibility:

```json
"metadata": {
  "license": "llama3.3",
  "license_url": "https://llama.meta.com/llama3_3/license/"
}
```

---

## Implementation Phases

### Phase 1: Manifest + Swarm Distribution
- `tightwad manifest create` — generate manifest with piece hashes
- `tightwad swarm seed` / `tightwad swarm pull` — chunked P2P transfer
- Piece verification, resume support, bitfield tracking
- **Dependencies:** asyncio, existing config infrastructure
- **New files:** `tightwad/manifest.py`, `tightwad/swarm_transfer.py`

### Phase 2: Peer Discovery
- DHT-based peer announcement and discovery
- Tailscale integration for zero-config LAN/VPN discovery
- Automatic topology formation (coordinator, workers, drafters)
- `tightwad swarm join` / `tightwad swarm peers`
- **Dependencies:** Phase 1 (need to know what models peers have)
- **New files:** `tightwad/dht.py`, `tightwad/topology.py`

### Phase 3: Cooperative Inference
- Tit-for-tat accounting with signed ledger entries
- Priority queue based on contribution ratio
- Optimistic unchoking for new peers
- Rarest-layer replication
- **Dependencies:** Phase 2 (need peer discovery to know who to account with)
- **New files:** `tightwad/accounting.py`, `tightwad/scheduler.py`

### Phase 4: Internet-Scale Speculation
- Speculative branching across internet-distributed drafters
- Adaptive grace periods based on observed peer latency
- Draft routing to lowest-latency compatible peer
- **Dependencies:** Phase 2 + 3 (need discovery + accounting)
- **Modifies:** `tightwad/proxy.py` (extend multi-drafter)

---

## Comparison to Existing Systems

| System | Model Distribution | Peer Discovery | Cooperative Inference | Speculation |
|--------|:-:|:-:|:-:|:-:|
| BitTorrent | Chunked P2P | DHT | Tit-for-tat (files) | N/A |
| Petals | Manual | Centralized | Basic | No |
| Exo | Manual | mDNS (LAN only) | No | Pipeline parallel |
| BOINC | Centralized | Centralized | Credit system | No |
| Golem | Centralized | Marketplace | Payment (crypto) | No |
| **Tightwad Swarm** | **Chunked P2P** | **DHT + Tailscale** | **Tit-for-tat (compute)** | **Yes (core feature)** |

The key differentiator: speculation is the core protocol primitive, not an afterthought. Every design decision optimizes for the draft→verify loop because that's what makes consumer GPU pooling viable over high-latency networks.
