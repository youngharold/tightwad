# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-04-19

### Added
- **Expert-aware MoE placement** — new `moe_placement: balanced | profile-guided` per-model config. Emits llama.cpp `--override-tensor` flags that pin whole experts to specific GPUs instead of relying on layer-level splits. Works with indexed-form MoE GGUFs; `tightwad moe defuse` handles the fused-tensor majority.
- **`tightwad moe defuse`** — rewrites fused expert tensors (`blk.L.ffn_*_exps.weight`) to indexed form (`blk.L.ffn_*.E.weight`) so per-expert placement is actually achievable. Streaming I/O, single pass, identical weights.
- **`tightwad moe plan`** — generates and previews the expert placement map for the current cluster. `--emit-ot` prints shell-paste-ready flags; `--json` emits the full plan as JSON.
- **`tightwad moe profile` / `moe summary`** — captures per-expert routing frequencies from a running cluster (llama.cpp stderr, local logs, or peer agent). Requires the instrumented llama.cpp build from `scripts/patches/llamacpp-moe-log.patch` for full fidelity; unpatched builds capture aggregate routing-event counts.
- **`tightwad moe bench`** — MoE-specific A/B benchmark with live-streaming per-prompt table (TTFT, rolling acceptance, speedup). Targets any OpenAI-compatible endpoint, including LM Studio on `:1234`.
- **Auto-measured device scores** (`tightwad/moe_device_bench.py`) — profile-guided placement uses real TCP-RTT measurements per device with a 24-hour cache at `~/.tightwad/device-scores.json`.
- **Peer endpoint `GET /v1/peer/moe/profile`** — aggregated hot-expert counts from `rpc-server` stderr (now captured to `~/.tightwad/logs/rpc-{port}.log` with 10 MB rotation).
- **Doctor checks** — warns on MoE+dense mismatch, fused-without-defuse, profile-guided without a hot profile or without `LLAMA_LOG_MOE=1`.
- Reference config `configs/cluster-moe-youngharold.yaml` and `docs/moe.md` guide.

## [0.4.3] - 2026-04-16

### Changed
- **README modes section** expanded from four to six to match tightwad.dev. Adds first-class entries for **Multi-Drafter Consensus** and **Swarm Transfer**, reorders with **Combined Mode** first to match the killer-feature framing on the website. Wiki `Home.md` updated in lockstep.

## [0.4.2] - 2026-03-28

### Fixed
- **JSON error handling** (#54) — proxy HTTP paths now catch invalid JSON responses gracefully
- **YAML config validation** (#55) — missing 'coordinator' key gives clear error instead of KeyError
- **Default mismatch** (#56) — env var default for max_draft_tokens aligned with YAML default (8)
- **Env var type safety** (#59) — non-numeric env vars fall back to defaults instead of crashing
- **extra_args merging** (#60) — explicit empty `[]` in config no longer ignored
- **Constant-time auth** (#61) — bearer token comparison uses `hmac.compare_digest()`
- **Unused imports** — cleaned across coordinator, family, quality_gate, dashboard

### Added
- **Integration tests** (#51, #52, #53) — speculation_round, quality gate pipeline, manifest.py
- **FAQ & best practices site** — https://youngharold.github.io/tightwad/

## [0.4.1] - 2026-03-28

### Added
- **Quality gate mode** (#34) — new proxy mode where a fleet of cheap CPU/GPU agents generate responses and a powerful GPU reviews them. Three verdicts: approve (pass through, ~60-80%), correct (fix inline), reject (regenerate on GPU). LRU response cache, round-robin routing, configurable verification prompts. CLI: `tightwad gate start/status`
- **A/B benchmark mode** (#42) — `tightwad bench` compares proxy vs direct target speed with per-prompt breakdown, JSON output, custom prompt files
- **WebSocket dashboard** (#39) — bidirectional `/v1/tightwad/ws` endpoint alongside existing SSE. Supports inbound commands (e.g. live draft token adjustment)
- **Draft-verify pipelining** (#47) — overlap next draft with current verification for up to 2x additional throughput. Optimistic drafting reused when acceptance is high
- **Cost-aware adaptive tuning** (#48) — draft/verify timing ratio scales aggression factor. Cheap drafts → increase aggressively; expensive drafts → conservative changes
- **Tree-based speculation** (#49) — tree data structures and path selection for branching multi-drafter outputs. Ready for integration when llama.cpp adds tree attention

### Changed
- **HTTP/2 enabled** (#45) — proxy client connections now use HTTP/2 for multiplexed requests
- **httpx[http2] promoted to runtime dependency** from dev-only
- **CLI split into subcommand modules** (#36) — `cli.py` (1500+ lines) split into `cli/` package with 7 modules (cluster, proxy, gate, peer, tools, service)

### Fixed
- **Ephemeral httpx clients** (#43) — Ollama code paths now reuse persistent connections (5-15ms/round saved)
- **Redundant /tokenize calls** (#44) — skipped for same-family models (2-6ms/round saved)
- **KV cache slot pinning** (#46) — `id_slot: 0` ensures cache hits across speculation rounds
- **Tensor split GPU ordering** — `plan_distribution` now matches workers-first convention
- **CI: exclude e2e_loader_test.py** — standalone script was crashing pytest collection

### Evaluated
- **Built-in GGUF parser** (#38) — kept for zero-dep installs (family detection, loader). Official `gguf` package used for advanced inspection via `[inspect]` extra
- **Pipeline parallelism** (#50) — closed as future RFC. Requires llama-server hidden-state passthrough not yet available

## [0.4.0] - 2026-03-27

### Added
- GitHub Actions CI workflow with Python 3.10–3.13 test matrix (#18)
- CHANGELOG.md, CONTRIBUTING.md, and CODE_OF_CONDUCT.md (#19)
- GitHub issue templates (bug report, feature request) and PR template (#20)
- Example configs: minimal spec decode, CPU draft, two-GPU, mixed-vendor, combined mode (#21)
- Documentation: quickstart guide, configuration reference, architecture overview (#22)
- README badges: PyPI version, CI status, license, Python versions (#23)
- **Backend presets** — auto-inject known-good environment variables per backend (e.g. `HSA_ENABLE_SDMA=0` and `GPU_MAX_HW_QUEUES=1` for ROCm multi-GPU), preventing SDMA hangs without manual configuration
- **`extra_args` and `env` coordinator config** — passthrough fields in `cluster.yaml` for backend-specific CLI args and environment variables; user values override presets
- **Model family compatibility validation** (#37) — auto-detect draft/target architecture families via Ollama `/api/show`, llama-server `/props`, or GGUF metadata. Warns at proxy startup and in `tightwad doctor` when families mismatch (the #1 footgun in speculative decoding setup — causes <5% acceptance with no obvious error)
- **Version enforcement between coordinator and RPC workers** (#30) — `tightwad start` checks llama.cpp versions via SSH and refuses to launch on mismatch. `tightwad doctor` escalates version mismatches from WARN to FAIL. Bypass with `--skip-version-check`
- **MoE model VRAM warnings** (#32) — detect Mixture-of-Experts models via GGUF metadata (`expert_count`) or tensor name patterns. `tightwad inspect` shows expert count, active experts, and per-device shared overhead. `tightwad doctor` and `tightwad start` warn when MoE shared overhead exceeds a GPU's VRAM
- **Peer agent for cross-platform cluster management** (#31) — lightweight HTTP daemon (port 9191) replacing SSH for remote worker management. 7 REST endpoints: version, health, GPU, models, RPC start/stop, logs. Bearer token auth, process manager, LAN-friendly. CLI: `tightwad peer start/stop/status`. Workers with `peer_port` configured use HTTP instead of SSH for version checks
- **Consensus verification: multi-drafter token voting** (#33) — when multiple drafters are configured, compare their outputs before contacting the target. Three modes: `strict` (unanimous), `majority` (>50%), `any_disagree`. Unanimous positions skip the target call entirely. Configurable via `proxy.consensus_mode` in cluster.yaml or `TIGHTWAD_CONSENSUS_MODE` env var. Prometheus metrics for consensus accept/fallback rates
- **Configurable chat templates** (#40) — `/v1/chat/completions` now auto-detects the correct chat template from the target model family instead of hardcoding Qwen3/ChatML format. Built-in templates for Llama 3, Mistral, Gemma, Phi, DeepSeek, Command-R. Configurable via `proxy.chat_template` or `TIGHTWAD_CHAT_TEMPLATE` env var
- **Auto-tune max_draft_tokens** (#41) — adaptive mode adjusts draft tokens at runtime based on rolling acceptance rate. Increases when acceptance >80%, decreases when <40%, clamped to 4-64. Enable via `proxy.max_draft_tokens: auto` or `TIGHTWAD_MAX_DRAFT_TOKENS=auto`
- **A/B benchmark mode** (#42) — `tightwad bench` runs the same prompts through proxy and target directly, reporting speedup, avg/median tok/s, p95 latency, per-prompt breakdown. Supports custom prompt files (`--prompts`), JSON output (`--json`), and warmup runs

### Fixed
- **`--flash-attn on` value format** — llama.cpp b8112+ changed `--flash-attn` from a bare flag to requiring a value (`on|off|auto`). Reverts the v0.1.5 bare-flag change. Without this fix, `--flash-attn` consumes the next argument (e.g. `--rpc`) as its value, breaking RPC pool startup
- **Tensor split GPU ordering** — `plan_distribution` now uses workers-first ordering consistent with `tensor_split()` and llama.cpp RPC convention. Zero-VRAM GPUs no longer receive remainder layers

## [0.3.0] - 2026-02-19

### Added
- **JSON pidfile metadata** — coordinator PID file now stores JSON with pid, port, config path, model name, and start timestamp. Backward compatible with legacy plain-int format. Enables `tightwad status` without `-c` flag
- **`tightwad deploy <host>`** — deploy tightwad to a remote host via SSH: checks connectivity, verifies python3, installs via pip, copies config, starts tightwad, and verifies health endpoint
- **Prometheus `/metrics` endpoint** — text/plain Prometheus exposition format with metrics: `tightwad_requests_total`, `tightwad_tokens_generated_total`, `tightwad_tokens_drafted_total`, `tightwad_tokens_accepted_total`, `tightwad_speculation_acceptance_rate`, `tightwad_speculation_rounds_total`, `tightwad_uptime_seconds`, `tightwad_bonus_tokens_total`, `tightwad_resampled_total`

## [0.2.1] - 2026-02-19

### Added
- **`tightwad pull`** — download GGUF models from HuggingFace with resume support, progress bar, and GGUF validation. Includes curated model registry with short specs (e.g. `llama3.3:70b-q4_k_m`, `qwen3:32b-q4_k_m`) and supports direct URLs and HF repo paths
- `tightwad pull --list` — show all available models in the curated registry

## [0.2.0] - 2026-02-19

### Added
- **GPU auto-detection** (`tightwad/gpu_detect.py`) — auto-detect NVIDIA (via nvidia-smi), AMD (via rocm-smi), and Apple Silicon (via system_profiler) GPUs with VRAM, backend, and device index
- **`tightwad init --local`** — one-command setup: detect local GPUs, find llama-server binary, generate a coordinator-only config YAML. Supports `--model-path` and `--port` options
- **`tightwad service install/uninstall/status`** — install tightwad as a persistent system service (systemd on Linux, launchd on macOS) with auto-restart on failure
- `detect_binary()` — searches PATH and common build locations for llama-server

## [0.1.5] - 2026-02-19

### Fixed
- **`--flash-attn` bare flag** — llama-server expects `--flash-attn` as a bare flag, not `--flash-attn on`. The spurious `"on"` argument was silently ignored, meaning flash attention was never actually enabled. This caused a ~10% generation speed regression on ROCm setups
- **posix_fadvise error reporting** — errno codes (EINVAL, EBADF, ESPIPE) are now mapped to human-readable messages. Failed reclaim is reported as "failed" (yellow) distinct from "skipped" (dimmed), logged at debug instead of warning level
- **`flash_attn` type coercion** — `ModelConfig.flash_attn` is now strictly `bool`. Legacy YAML string values (`"on"`, `"off"`, `"auto"`) are coerced to boolean during config loading for backward compatibility

### Added
- **Config auto-discovery** — tightwad now searches `./tightwad.yaml`, `./configs/cluster.yaml`, `~/.tightwad/config.yaml`, and the package default before requiring an explicit `-c` flag. `FileNotFoundError` now lists all searched paths and suggests `tightwad init`
- Friendly quick-start guide printed on config-not-found errors instead of a raw traceback

## [0.1.4] - 2026-02-18

### Added
- **`tightwad load` command** — standalone GGUF loading with pre-warming, memory-aware startup, and post-load RAM reclaim
- **Pure-Python GGUF parser** (`gguf_reader.py`) — parses GGUF v2/v3 headers, KV metadata, and tensor info without the `gguf` package dependency
- **Pre-warm lifecycle** (`loader.py`) — sequential page cache warming with `posix_fadvise(SEQUENTIAL)` on Linux before llama-server mmaps the file
- **Auto pre-warm in `tightwad start`** — when `ram_reclaim` is `auto` or `on` and model > 80% of available RAM, pre-warming is applied automatically
- `needs_streaming_load()` detection — returns True when model exceeds 80% of (available RAM + swap)
- `TIGHTWAD_DISABLE_PREWARM=1` environment variable kill-switch
- `LoadResult` dataclass with timing, throughput, and peak RSS metrics
- `model_summary()` for extracting arch, layers, quant, context length from GGUF headers

## [0.1.3] - 2026-02-18

### Added
- **`tightwad reclaim` command** — free RAM after model loads to VRAM. Cross-platform: `posix_fadvise(DONTNEED)` on Linux, `SetProcessWorkingSetSize` on Windows, no-op on macOS (unified memory)
- **`tightwad tune` command** — diagnose system RAM/swap readiness for large models with platform-specific fix commands
- **`--ram-reclaim` flag on `tightwad start`** — modes: `off`, `on`, `auto` (default). Auto reclaims if model > 50% of available RAM
- `ram_reclaim` config option in cluster.yaml (top-level, defaults to `auto`)
- `start_and_reclaim()` coordinator function — waits for `/health` 200 before reclaiming (prevents evicting pages the server still needs)
- Auto-detection of model path from `/proc/{pid}/maps` on Linux
- Auto-detection of coordinator PID from pidfile for `tightwad reclaim`
- Cross-platform RSS and available RAM reading without psutil dependency

## [0.1.2] - 2026-02-18

### Added
- `--version` flag to CLI
- Non-interactive `tightwad init` with `--draft-url` and `--target-url` flags
- SECURITY.md with vulnerability reporting guidelines
- Docker healthcheck and persistent log volume

### Security
- **Bearer-token authentication for proxy API** — optional `auth_token` config or `TIGHTWAD_PROXY_TOKEN` env var; logs warning if proxy starts without auth (#6)
- **SSRF protection on upstream URLs** — scheme allowlist (http/https only), optional private-IP blocking via `allow_private_upstream` config, DNS-rebinding protection (#7)
- **Input validation and request size limits** — `max_tokens_limit` (default 16384) rejects excessive generation requests; `max_body_size` (default 10 MB) rejects oversized payloads before buffering (#8)
- Fixed XSS vulnerability in live dashboard (SEC-2, #9)
- Code hardening: replaced assert guards with proper exceptions, fixed log file descriptor leak (#10)

### Fixed
- Broken wiki links in documentation
- Replaced nonexistent Qwen3-72B references with Qwen3-32B
- Sanitized example configs and scripts (removed hardcoded IPs)

## [0.1.1] - 2026-02-17

### Added
- `tightwad doctor` diagnostic command for cluster validation
- `tightwad logs` command with `--follow` and `--clear` options
- `tightwad chat` interactive mode with inline speculation stats
- Live web dashboard for proxy monitoring (`/dashboard`)
- Docker Compose support with environment variable configuration
- `tightwad init` wizard for auto-discovering LAN inference servers
- Swarm security: `--token` and `--allowed-ips` flags for seeder

### Fixed
- Mocked `draft_client.post` in verify_with_logprobs tests

## [0.1.0] - 2026-02-16

### Added
- Initial release
- Mixed-vendor GPU pooling via llama.cpp RPC
- Speculative decoding proxy (draft + verify across machines)
- Combined mode: speculation over RPC pool
- GGUF model inspection (`tightwad inspect`)
- Model distribution to workers (`tightwad distribute`)
- Swarm P2P transfer protocol with manifest and chunked downloads
- Benchmark scripts for local and OpenRouter models
- Support for CUDA, ROCm, and CPU backends
- OpenAI-compatible API endpoint

[Unreleased]: https://github.com/youngharold/tightwad/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/youngharold/tightwad/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/youngharold/tightwad/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/youngharold/tightwad/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/youngharold/tightwad/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/youngharold/tightwad/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/youngharold/tightwad/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/youngharold/tightwad/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/youngharold/tightwad/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/youngharold/tightwad/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/youngharold/tightwad/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/youngharold/tightwad/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/youngharold/tightwad/releases/tag/v0.1.0
