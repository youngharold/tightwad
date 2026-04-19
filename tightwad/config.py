"""Cluster configuration loader."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger("tightwad.config")


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cluster.yaml"


@dataclass
class GPU:
    name: str
    vram_gb: int
    rpc_port: int | None = None  # None for local coordinator GPUs


@dataclass
class Worker:
    host: str
    gpus: list[GPU] = field(default_factory=list)
    ssh_user: str | None = None
    model_dir: str | None = None
    peer_port: int | None = None

    @property
    def rpc_addresses(self) -> list[str]:
        return [f"{self.host}:{gpu.rpc_port}" for gpu in self.gpus]


@dataclass
class ModelConfig:
    name: str
    path: str
    ctx_size: int = 8192
    predict: int = 4096
    flash_attn: bool = True
    default: bool = False
    #: Expert-placement strategy for MoE models.
    #:
    #: ``None`` / ``"off"`` (default) — no per-expert placement; the existing
    #: layer-only ``--tensor-split`` behavior applies.
    #:
    #: ``"balanced"`` — bin-pack expert tensors across GPUs proportional to
    #: VRAM. Requires an indexed-form GGUF (see ``tightwad moe defuse``).
    #:
    #: ``"profile-guided"`` — same as balanced, but consumes a captured
    #: hot-expert profile at ``moe_hot_profile`` to pin frequently-hit experts
    #: onto the highest-scoring device.
    moe_placement: str | None = None
    moe_hot_profile: str | None = None


@dataclass
class ServerEndpoint:
    url: str  # e.g. "http://192.168.1.100:8081"
    model_name: str  # display only
    backend: str = "llamacpp"  # "llamacpp" or "ollama"


@dataclass
class PeerConfig:
    host: str = "0.0.0.0"
    port: int = 9191
    auth_token: str | None = None
    model_dirs: list[str] = field(default_factory=list)


@dataclass
class ProxyConfig:
    draft: ServerEndpoint
    target: ServerEndpoint
    host: str = "0.0.0.0"
    port: int = 8088
    max_draft_tokens: int = 8
    auto_draft_tokens: bool = False
    fallback_on_draft_failure: bool = True
    drafters: list[ServerEndpoint] = field(default_factory=list)
    #: Optional Bearer token that protects all /v1/ endpoints.
    #: When set, every request must include ``Authorization: Bearer <token>``.
    #: When unset the proxy operates in open (unauthenticated) mode and logs
    #: a startup warning — this preserves backward compatibility.
    auth_token: str | None = None
    #: Allow upstream URLs that resolve to private / internal IP ranges.
    #:
    #: Tightwad's most common deployment targets LAN servers (e.g.
    #: ``http://192.168.1.10:11434``) so the private-IP SSRF check defaults to
    #: ``True`` (opted in / allowed).  Set this to ``False`` in environments
    #: where the proxy should never reach internal infrastructure.
    #:
    #: The scheme check (http/https only) is **always** enforced regardless of
    #: this flag.
    #:
    #: Audit ref: SEC-5 / issue #7
    allow_private_upstream: bool = True
    #: Maximum value allowed for ``max_tokens`` in completion requests.
    #:
    #: Requests that exceed this limit are rejected with ``400 Bad Request``
    #: before the downstream server is contacted.  Very large ``max_tokens``
    #: values can force the downstream llama.cpp server into an extremely long
    #: generation — effectively a DoS against the backend.
    #:
    #: Configurable via ``TIGHTWAD_MAX_TOKENS_LIMIT`` env var or
    #: ``proxy.max_tokens_limit`` in cluster.yaml.
    #:
    #: Audit ref: CQ-1 / issue #8
    max_tokens_limit: int = 16384
    #: Maximum allowed request body size in bytes.
    #:
    #: Requests whose ``Content-Length`` header exceeds this value are rejected
    #: with ``413 Content Too Large`` *before* the body is buffered in memory,
    #: preventing memory-exhaustion DoS via multi-gigabyte payloads.
    #:
    #: Configurable via ``TIGHTWAD_MAX_BODY_SIZE`` env var (bytes) or
    #: ``proxy.max_body_size`` in cluster.yaml.
    #:
    #: Audit ref: CQ-5 / issue #8
    max_body_size: int = 10 * 1024 * 1024  # 10 MB default
    #: Consensus verification mode for multi-drafter setups.
    #:
    #: When multiple drafters are configured, consensus mode compares their
    #: outputs *before* contacting the target model.  Positions where drafters
    #: agree can be accepted directly, skipping the expensive target call.
    #:
    #: Supported values:
    #:   ``"off"``           — disabled (default), use normal speculation
    #:   ``"strict"``        — require unanimous agreement
    #:   ``"majority"``      — >50 % of drafters must agree
    #:   ``"any_disagree"``  — accept unanimous, verify on any disagreement
    #:
    #: Configurable via ``TIGHTWAD_CONSENSUS_MODE`` env var or
    #: ``proxy.consensus_mode`` in cluster.yaml.
    consensus_mode: str = "off"
    #: Chat template for ``/v1/chat/completions``.
    #:
    #: Controls how OpenAI-format chat messages are rendered into the raw
    #: prompt the model expects.  Mismatched templates silently destroy
    #: speculation acceptance rates.
    #:
    #: Supported values: ``"auto"`` (detect from target model family),
    #: ``"chatml"`` (Qwen/Yi), ``"llama3"``, ``"mistral"``, ``"gemma"``,
    #: ``"phi"``, ``"deepseek"``, ``"command-r"``.
    #:
    #: Configurable via ``TIGHTWAD_CHAT_TEMPLATE`` env var or
    #: ``proxy.chat_template`` in cluster.yaml.
    chat_template: str = "auto"


@dataclass
class ClusterConfig:
    coordinator_host: str
    coordinator_port: int
    coordinator_backend: str
    coordinator_gpus: list[GPU]
    workers: list[Worker]
    models: dict[str, ModelConfig]
    coordinator_binary: str
    rpc_server_binary: str
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    proxy: ProxyConfig | None = None
    peer: PeerConfig | None = None
    quality_gate: object | None = None  # QualityGateConfig (lazy import)
    ram_reclaim: str = "auto"  # "off", "on", "auto"

    @property
    def all_gpus(self) -> list[GPU]:
        gpus = list(self.coordinator_gpus)
        for w in self.workers:
            gpus.extend(w.gpus)
        return gpus

    @property
    def total_vram_gb(self) -> int:
        return sum(g.vram_gb for g in self.all_gpus)

    @property
    def rpc_addresses(self) -> list[str]:
        addrs: list[str] = []
        for w in self.workers:
            addrs.extend(w.rpc_addresses)
        return addrs

    def tensor_split(self) -> list[float]:
        """Calculate --tensor-split proportions from VRAM sizes.

        Order: RPC worker GPUs first, then coordinator local GPUs.
        This matches llama-server's mapping: --rpc workers get the first
        split slots, local GPUs get the remaining slots.
        """
        # Workers first, then coordinator locals
        gpus: list[GPU] = []
        for w in self.workers:
            gpus.extend(w.gpus)
        gpus.extend(self.coordinator_gpus)
        total = sum(g.vram_gb for g in gpus)
        return [round(g.vram_gb / total, 2) for g in gpus]

    def default_model(self) -> ModelConfig | None:
        for m in self.models.values():
            if m.default:
                return m
        # Return first model if none marked default
        return next(iter(self.models.values()), None)


def backend_presets(backend: str, gpu_count: int) -> dict:
    """Return known-good env vars and extra args for a backend.

    Users can override any of these via explicit ``env`` and ``extra_args``
    in their cluster.yaml coordinator section.
    """
    presets: dict = {"env": {}, "extra_args": []}

    if backend == "hip" and gpu_count > 1:
        # Multi-GPU ROCm: prevent SDMA hangs on most AMD boards
        presets["env"]["HSA_ENABLE_SDMA"] = "0"
        presets["env"]["GPU_MAX_HW_QUEUES"] = "1"

    return presets


def _parse_draft_tokens(value: int | str) -> int:
    """Parse max_draft_tokens value, returning default 16 for 'auto'."""
    if isinstance(value, str) and value.lower() == "auto":
        return 16  # sensible starting point for auto-tuning
    return int(value)


def load_proxy_from_env() -> ProxyConfig | None:
    """Build a ProxyConfig from TIGHTWAD_* environment variables.

    Returns None if required env vars (TIGHTWAD_DRAFT_URL, TIGHTWAD_TARGET_URL)
    are not set.

    Raises
    ------
    ValueError
        If a supplied URL fails SSRF validation (bad scheme, private IP when
        not allowed, etc.).
    """
    draft_url = os.environ.get("TIGHTWAD_DRAFT_URL")
    target_url = os.environ.get("TIGHTWAD_TARGET_URL")
    if not draft_url or not target_url:
        return None

    # Token may be supplied via TIGHTWAD_PROXY_TOKEN (primary) or the legacy
    # TIGHTWAD_TOKEN alias used by the swarm seeder.
    auth_token = (
        os.environ.get("TIGHTWAD_PROXY_TOKEN")
        or os.environ.get("TIGHTWAD_TOKEN")
        or None
    )

    # TIGHTWAD_ALLOW_PRIVATE_UPSTREAM: set to "false" / "0" / "no" to
    # enforce private-IP blocking even in env-var mode.  Defaults to True
    # so that homelab LAN URLs continue to work without extra configuration.
    _priv_raw = os.environ.get("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", "true").lower()
    allow_private = _priv_raw not in ("false", "0", "no")

    # Validate URLs before building the config (SSRF: SEC-5)
    _validate_proxy_urls(
        draft_url=draft_url,
        target_url=target_url,
        allow_private=allow_private,
        source="environment variable",
    )

    try:
        max_tokens_limit = int(os.environ.get("TIGHTWAD_MAX_TOKENS_LIMIT", "16384"))
    except ValueError:
        logger.warning("Invalid TIGHTWAD_MAX_TOKENS_LIMIT value, using default 16384")
        max_tokens_limit = 16384
    try:
        max_body_size = int(os.environ.get("TIGHTWAD_MAX_BODY_SIZE", str(10 * 1024 * 1024)))
    except ValueError:
        logger.warning("Invalid TIGHTWAD_MAX_BODY_SIZE value, using default 10 MB")
        max_body_size = 10 * 1024 * 1024

    return ProxyConfig(
        draft=ServerEndpoint(
            url=draft_url,
            model_name=os.environ.get("TIGHTWAD_DRAFT_MODEL", "draft"),
            backend=os.environ.get("TIGHTWAD_DRAFT_BACKEND", "ollama"),
        ),
        target=ServerEndpoint(
            url=target_url,
            model_name=os.environ.get("TIGHTWAD_TARGET_MODEL", "target"),
            backend=os.environ.get("TIGHTWAD_TARGET_BACKEND", "ollama"),
        ),
        host=os.environ.get("TIGHTWAD_HOST", "0.0.0.0"),
        port=int(os.environ.get("TIGHTWAD_PORT", "8088")),
        max_draft_tokens=_parse_draft_tokens(os.environ.get("TIGHTWAD_MAX_DRAFT_TOKENS", "8")),
        auto_draft_tokens=os.environ.get("TIGHTWAD_MAX_DRAFT_TOKENS", "").lower() == "auto",
        auth_token=auth_token,
        allow_private_upstream=allow_private,
        max_tokens_limit=max_tokens_limit,
        max_body_size=max_body_size,
        consensus_mode=os.environ.get("TIGHTWAD_CONSENSUS_MODE", "off"),
        chat_template=os.environ.get("TIGHTWAD_CHAT_TEMPLATE", "auto"),
    )


def _validate_proxy_urls(
    *,
    draft_url: str,
    target_url: str,
    drafters: list[str] | None = None,
    allow_private: bool,
    source: str = "config",
) -> None:
    """Run SSRF validation on all proxy upstream URLs.

    Parameters
    ----------
    draft_url:
        The draft model's upstream URL.
    target_url:
        The target model's upstream URL.
    drafters:
        Optional list of additional drafter URLs to validate.
    allow_private:
        Forwarded to :func:`~tightwad.ssrf.validate_upstream_url`.
    source:
        Human-readable description of where the URLs came from (used in
        log messages and error context).

    Raises
    ------
    ValueError
        If any URL fails SSRF validation.
    """
    # Lazy import to avoid circular dependencies at module load time.
    from .ssrf import validate_upstream_url

    endpoints = [("proxy.draft", draft_url), ("proxy.target", target_url)]
    for label, url in (drafters or []):
        endpoints.append((label, url))

    for label, url in endpoints:
        try:
            validate_upstream_url(url, allow_private=allow_private)
        except ValueError as exc:
            raise ValueError(
                f"SSRF validation failed for {label} URL from {source}: {exc}"
            ) from exc
        logger.debug("ssrf: %s URL %r validated OK (allow_private=%s)", label, url, allow_private)


def _resolve_config_path(explicit: str | Path | None = None) -> Path:
    """Search for config file in standard locations.

    Search order:
    1. Explicit path (``-c`` flag)
    2. ``TIGHTWAD_CONFIG`` environment variable
    3. ``./tightwad.yaml`` (current working directory)
    4. ``./configs/cluster.yaml`` (project convention)
    5. ``~/.tightwad/config.yaml`` (user config)
    6. Package-bundled ``DEFAULT_CONFIG``
    """
    if explicit is not None:
        return Path(explicit)

    env_path = os.environ.get("TIGHTWAD_CONFIG")
    if env_path:
        return Path(env_path)

    candidates = [
        Path.cwd() / "tightwad.yaml",
        Path.cwd() / "configs" / "cluster.yaml",
        Path.home() / ".tightwad" / "config.yaml",
        DEFAULT_CONFIG,
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.debug("Auto-discovered config: %s", candidate)
            return candidate

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No tightwad config found. Searched:\n  {searched}\n\n"
        "Quick start:\n"
        "  tightwad init --local          # auto-detect GPUs and generate config\n"
        "  tightwad init --draft-url ...   # non-interactive proxy config\n"
        "  tightwad -c /path/to/config.yaml start"
    )


def load_config(path: str | Path | None = None) -> ClusterConfig:
    """Load cluster config from YAML file, falling back to env vars for proxy-only mode."""
    try:
        config_path = _resolve_config_path(path)
    except FileNotFoundError:
        # Before raising, check env vars for proxy-only mode
        proxy = load_proxy_from_env()
        if proxy is not None:
            return ClusterConfig(
                coordinator_host="0.0.0.0",
                coordinator_port=8080,
                coordinator_backend="cuda",
                coordinator_gpus=[],
                workers=[],
                models={},
                coordinator_binary="llama-server",
                rpc_server_binary="rpc-server",
                proxy=proxy,
            )
        raise

    if not config_path.exists():
        proxy = load_proxy_from_env()
        if proxy is not None:
            return ClusterConfig(
                coordinator_host="0.0.0.0",
                coordinator_port=8080,
                coordinator_backend="cuda",
                coordinator_gpus=[],
                workers=[],
                models={},
                coordinator_binary="llama-server",
                rpc_server_binary="rpc-server",
                proxy=proxy,
            )
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if "coordinator" not in raw:
        raise ValueError(
            "Missing required 'coordinator' section in cluster.yaml"
        )

    coord = raw["coordinator"]
    coordinator_gpus = [
        GPU(name=g["name"], vram_gb=g["vram_gb"])
        for g in coord.get("gpus", [])
    ]

    workers = []
    for w in raw.get("workers", []):
        gpus = [
            GPU(name=g["name"], vram_gb=g["vram_gb"], rpc_port=g["rpc_port"])
            for g in w.get("gpus", [])
        ]
        workers.append(Worker(
            host=w["host"],
            gpus=gpus,
            ssh_user=w.get("ssh_user"),
            model_dir=w.get("model_dir"),
            peer_port=w.get("peer_port"),
        ))

    models = {}
    for name, m in raw.get("models", {}).items():
        models[name] = ModelConfig(
            name=name,
            path=m["path"],
            ctx_size=m.get("ctx_size", 8192),
            predict=m.get("predict", 4096),
            flash_attn=m.get("flash_attn", True) not in (False, "off", "false", "no", 0),
            default=m.get("default", False),
            moe_placement=m.get("moe_placement"),
            moe_hot_profile=m.get("moe_hot_profile"),
        )

    binaries = raw.get("binaries", {})

    proxy = None
    if "proxy" in raw:
        p = raw["proxy"]
        draft = p["draft"]
        target = p["target"]
        drafter_endpoints = []
        drafter_url_pairs: list[tuple[str, str]] = []
        for i, d in enumerate(p.get("drafters", [])):
            drafter_endpoints.append(ServerEndpoint(
                url=d["url"],
                model_name=d["model_name"],
                backend=d.get("backend", "llamacpp"),
            ))
            drafter_url_pairs.append((f"proxy.drafters[{i}]", d["url"]))

        # auth_token: read from YAML, then fall back to env vars so that
        # tokens can be injected at runtime without editing the config file.
        yaml_token = p.get("auth_token") or None
        env_token = (
            os.environ.get("TIGHTWAD_PROXY_TOKEN")
            or os.environ.get("TIGHTWAD_TOKEN")
            or None
        )
        # YAML value takes precedence; env var is the fallback.
        resolved_token = yaml_token or env_token

        # allow_private_upstream defaults to True so that common homelab
        # configs (targeting LAN addresses like 192.168.x.x) work without
        # any extra config.  Operators who want strict SSRF enforcement can
        # set allow_private_upstream: false in cluster.yaml.
        allow_private = p.get("allow_private_upstream", True)

        # Validate all upstream URLs before constructing clients (SSRF: SEC-5).
        _validate_proxy_urls(
            draft_url=draft["url"],
            target_url=target["url"],
            drafters=drafter_url_pairs,
            allow_private=allow_private,
            source=str(config_path),
        )

        proxy = ProxyConfig(
            draft=ServerEndpoint(url=draft["url"], model_name=draft["model_name"], backend=draft.get("backend", "llamacpp")),
            target=ServerEndpoint(url=target["url"], model_name=target["model_name"], backend=target.get("backend", "llamacpp")),
            host=p.get("host", "0.0.0.0"),
            port=p.get("port", 8088),
            max_draft_tokens=_parse_draft_tokens(p.get("max_draft_tokens", 8)),
            auto_draft_tokens=str(p.get("max_draft_tokens", 8)).lower() == "auto",
            fallback_on_draft_failure=p.get("fallback_on_draft_failure", True),
            drafters=drafter_endpoints,
            auth_token=resolved_token,
            allow_private_upstream=allow_private,
            max_tokens_limit=p.get("max_tokens_limit", 16384),
            max_body_size=p.get("max_body_size", 10 * 1024 * 1024),
            consensus_mode=p.get("consensus_mode", "off"),
            chat_template=p.get("chat_template", "auto"),
        )

    peer = None
    if "peer" in raw:
        pe = raw["peer"]
        peer = PeerConfig(
            host=pe.get("host", "0.0.0.0"),
            port=pe.get("port", 9191),
            auth_token=pe.get("auth_token") or os.environ.get("TIGHTWAD_PEER_TOKEN"),
            model_dirs=pe.get("model_dirs", []),
        )

    quality_gate = None
    if "quality_gate" in raw:
        from .quality_gate import QualityGateConfig, AgentEndpoint

        qg = raw["quality_gate"]
        verifier = qg.get("verifier", {})
        qg_agents = []
        for a in qg.get("agents", []):
            qg_agents.append(AgentEndpoint(
                url=a["url"],
                model_name=a["model_name"],
                backend=a.get("backend", "ollama"),
            ))
        quality_gate = QualityGateConfig(
            verifier_url=verifier.get("url", ""),
            verifier_model=verifier.get("model_name", ""),
            verifier_backend=verifier.get("backend", "llamacpp"),
            agents=qg_agents,
            routing=qg.get("routing", "round_robin"),
            verification_prompt=qg.get("verification_prompt", ""),
            max_retries=qg.get("max_retries", 1),
            cache_identical=qg.get("cache_identical", True),
            host=qg.get("host", "0.0.0.0"),
            port=qg.get("port", 8088),
            auth_token=qg.get("auth_token") or os.environ.get("TIGHTWAD_PROXY_TOKEN"),
        )

    ram_reclaim = raw.get("ram_reclaim", "auto")
    if ram_reclaim not in ("off", "on", "auto"):
        logger.warning(
            "Invalid ram_reclaim value %r, defaulting to 'auto'", ram_reclaim
        )
        ram_reclaim = "auto"

    # Backend presets (env vars, extra args) merged with explicit config
    coord_backend = coord.get("backend", "hip")
    presets = backend_presets(coord_backend, len(coordinator_gpus))
    merged_env = {**presets["env"], **coord.get("env", {})}
    merged_extra = coord.get("extra_args") if coord.get("extra_args") is not None else presets.get("extra_args", [])

    return ClusterConfig(
        coordinator_host=coord.get("host", "0.0.0.0"),
        coordinator_port=coord.get("port", 8080),
        coordinator_backend=coord_backend,
        coordinator_gpus=coordinator_gpus,
        workers=workers,
        models=models,
        coordinator_binary=binaries.get("coordinator", "llama-server"),
        rpc_server_binary=binaries.get("rpc_server", "rpc-server"),
        extra_args=merged_extra,
        env=merged_env,
        proxy=proxy,
        peer=peer,
        quality_gate=quality_gate,
        ram_reclaim=ram_reclaim,
    )
