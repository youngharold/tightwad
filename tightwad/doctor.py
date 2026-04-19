"""Diagnostic checks for Tightwad cluster health."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx
from rich.console import Console
from rich.rule import Rule

from . import worker
from .config import ClusterConfig, load_config


class Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


_ICONS = {
    Status.PASS: "[green]✓[/green]",
    Status.FAIL: "[red]✗[/red]",
    Status.WARN: "[yellow]![/yellow]",
    Status.SKIP: "[dim]○[/dim]",
}


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""
    fix: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Section:
    title: str
    results: list[CheckResult] = field(default_factory=list)


@dataclass
class DoctorReport:
    sections: list[Section] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(
            r.status != Status.FAIL
            for s in self.sections
            for r in s.results
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "sections": [
                {
                    "title": s.title,
                    "results": [
                        {
                            "name": r.name,
                            "status": r.status.value,
                            "detail": r.detail,
                            "fix": r.fix,
                            "data": r.data,
                        }
                        for r in s.results
                    ],
                }
                for s in self.sections
            ],
        }


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_config(config_path: str | Path | None) -> tuple[Section, ClusterConfig | None]:
    """Check config file: exists, parses, has models and workers."""
    section = Section(title="Configuration")

    # Resolve path the same way load_config does
    from .config import DEFAULT_CONFIG
    resolved = Path(config_path) if config_path else Path(
        os.environ.get("TIGHTWAD_CONFIG", DEFAULT_CONFIG)
    )

    if not resolved.exists():
        section.results.append(CheckResult(
            name="Config file",
            status=Status.FAIL,
            detail=f"Not found: {resolved}",
            fix=f"cp configs/cluster.yaml {resolved}  # then edit for your hardware",
        ))
        return section, None

    section.results.append(CheckResult(
        name="Config file",
        status=Status.PASS,
        detail=str(resolved),
    ))

    try:
        config = load_config(config_path)
    except Exception as e:
        section.results.append(CheckResult(
            name="YAML parse",
            status=Status.FAIL,
            detail=str(e),
            fix="Fix the YAML syntax in your config file",
        ))
        return section, None

    section.results.append(CheckResult(
        name="YAML parse",
        status=Status.PASS,
    ))

    # Models defined?
    if not config.models:
        section.results.append(CheckResult(
            name="Models defined",
            status=Status.FAIL,
            detail="No models section in config",
            fix="Add a 'models:' section to your cluster.yaml",
        ))
    else:
        section.results.append(CheckResult(
            name="Models defined",
            status=Status.PASS,
            detail=f"{len(config.models)} model(s): {', '.join(config.models)}",
        ))

    # Workers defined?
    if not config.workers:
        section.results.append(CheckResult(
            name="Workers defined",
            status=Status.WARN,
            detail="No workers — running coordinator-only (single machine)",
        ))
    else:
        gpu_count = sum(len(w.gpus) for w in config.workers)
        section.results.append(CheckResult(
            name="Workers defined",
            status=Status.PASS,
            detail=f"{len(config.workers)} worker(s), {gpu_count} remote GPU(s)",
        ))

    # Structural validation on the parsed config
    _validate_config(section, config)

    return section, config


def _validate_config(section: Section, config: ClusterConfig) -> None:
    """Add structural validation checks to the config section."""
    from urllib.parse import urlparse

    # Port range checks
    def _check_port(name: str, port: int) -> None:
        if not (1 <= port <= 65535):
            section.results.append(CheckResult(
                name=f"Port range: {name}",
                status=Status.WARN,
                detail=f"{port} is outside valid range 1-65535",
                fix=f"Set {name} to a port between 1 and 65535",
            ))

    _check_port("coordinator_port", config.coordinator_port)
    for w in config.workers:
        for gpu in w.gpus:
            if gpu.rpc_port is not None:
                _check_port(f"rpc_port ({w.host}/{gpu.name})", gpu.rpc_port)
    if config.proxy:
        _check_port("proxy.port", config.proxy.port)

    # VRAM positive
    for gpu in config.all_gpus:
        if gpu.vram_gb <= 0:
            section.results.append(CheckResult(
                name=f"VRAM positive: {gpu.name}",
                status=Status.WARN,
                detail=f"vram_gb={gpu.vram_gb} — must be > 0",
                fix=f"Set vram_gb to the actual VRAM size for {gpu.name}",
            ))

    # Coordinator binary not empty
    if not config.coordinator_binary.strip():
        section.results.append(CheckResult(
            name="Coordinator binary",
            status=Status.WARN,
            detail="Empty string",
            fix="Set binaries.coordinator to 'llama-server' or an absolute path",
        ))

    # Proxy URL validation
    if config.proxy:
        for label, endpoint in [("proxy.draft", config.proxy.draft), ("proxy.target", config.proxy.target)]:
            parsed = urlparse(endpoint.url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                section.results.append(CheckResult(
                    name=f"URL valid: {label}",
                    status=Status.WARN,
                    detail=f"'{endpoint.url}' is not a valid http(s) URL",
                    fix=f"Set {label}.url to a valid URL like http://host:port",
                ))

        # Backend enum
        valid_backends = {"llamacpp", "ollama"}
        for label, endpoint in [("proxy.draft", config.proxy.draft), ("proxy.target", config.proxy.target)]:
            if endpoint.backend not in valid_backends:
                section.results.append(CheckResult(
                    name=f"Backend valid: {label}",
                    status=Status.WARN,
                    detail=f"'{endpoint.backend}' — expected one of {valid_backends}",
                    fix=f"Set {label}.backend to 'llamacpp' or 'ollama'",
                ))

        # max_draft_tokens range
        mdt = config.proxy.max_draft_tokens
        if not (1 <= mdt <= 256):
            section.results.append(CheckResult(
                name="max_draft_tokens range",
                status=Status.WARN,
                detail=f"{mdt} is outside range 1-256",
                fix="Set proxy.max_draft_tokens between 1 and 256",
            ))

    # Duplicate RPC addresses
    seen: dict[str, str] = {}
    for w in config.workers:
        for gpu in w.gpus:
            if gpu.rpc_port is None:
                continue
            addr = f"{w.host}:{gpu.rpc_port}"
            if addr in seen:
                section.results.append(CheckResult(
                    name="Duplicate RPC address",
                    status=Status.WARN,
                    detail=f"{addr} used by both {seen[addr]} and {gpu.name}",
                    fix="Each RPC worker needs a unique host:port",
                ))
            else:
                seen[addr] = gpu.name


def check_binaries(config: ClusterConfig) -> Section:
    """Check that coordinator and rpc-server binaries are findable."""
    section = Section(title="Binaries")

    for label, binary in [
        ("Coordinator binary", config.coordinator_binary),
        ("RPC server binary", config.rpc_server_binary),
    ]:
        path = Path(binary)
        if path.is_absolute():
            found = path.exists()
        else:
            found = shutil.which(binary) is not None

        if found:
            resolved = shutil.which(binary) or str(path)
            section.results.append(CheckResult(
                name=label,
                status=Status.PASS,
                detail=f"{binary} → {resolved}",
            ))
        else:
            # On a different OS from the coordinator, this is expected
            if _is_cross_platform_path(binary):
                section.results.append(CheckResult(
                    name=label,
                    status=Status.SKIP,
                    detail=f"{binary} (cross-platform path, not checkable locally)",
                ))
            else:
                section.results.append(CheckResult(
                    name=label,
                    status=Status.FAIL,
                    detail=f"{binary} not found",
                    fix=f"Install llama.cpp and ensure '{binary}' is in PATH or use an absolute path in config",
                ))

    return section


def check_models(config: ClusterConfig) -> Section:
    """Check model file existence and tensor split sanity."""
    section = Section(title="Models")

    for name, model in config.models.items():
        model_path = Path(model.path)

        # Windows drive-letter paths on a non-Windows machine
        if _is_cross_platform_path(model.path):
            section.results.append(CheckResult(
                name=f"Model: {name}",
                status=Status.SKIP,
                detail=f"{model.path} (remote path, not checkable locally)",
            ))
        elif model_path.exists():
            size_gb = model_path.stat().st_size / (1024 ** 3)
            section.results.append(CheckResult(
                name=f"Model: {name}",
                status=Status.PASS,
                detail=f"{model.path} ({size_gb:.1f} GB)",
            ))
        else:
            section.results.append(CheckResult(
                name=f"Model: {name}",
                status=Status.FAIL,
                detail=f"File not found: {model.path}",
                fix=f"Download the model or update the path in config",
            ))

    # Tensor split sanity
    split = config.tensor_split()
    total = sum(split)
    if abs(total - 1.0) > 0.05:
        section.results.append(CheckResult(
            name="Tensor split",
            status=Status.WARN,
            detail=f"Proportions sum to {total:.2f} (expected ~1.0): {split}",
        ))
    else:
        section.results.append(CheckResult(
            name="Tensor split",
            status=Status.PASS,
            detail=f"{split} ({config.total_vram_gb} GB total across {len(config.all_gpus)} GPUs)",
        ))

    # MoE + small GPU checks
    _check_moe_vram(section, config)
    _check_moe_placement(section, config)

    return section


def _check_moe_vram(section: Section, config: ClusterConfig) -> None:
    """Check if MoE models will fit on the cluster's GPUs."""
    try:
        from .gguf_inspect import inspect_model, check_moe_vram
    except ImportError:
        return  # gguf package not installed

    for name, model in config.models.items():
        model_path = Path(model.path)
        if not model_path.exists() or _is_cross_platform_path(model.path):
            continue

        try:
            model_info = inspect_model(str(model_path))
        except Exception:
            continue

        if not model_info.is_moe:
            continue

        moe = model_info.moe
        section.results.append(CheckResult(
            name=f"MoE detected: {name}",
            status=Status.PASS,
            detail=f"{moe.n_expert} experts, ~{moe.routing_overhead_gb:.1f}GB shared overhead/GPU",
        ))

        # Check each GPU against MoE minimum
        gpu_vram = {gpu.name: gpu.vram_gb for gpu in config.all_gpus}
        warnings = check_moe_vram(model_info, gpu_vram=gpu_vram)
        for w in warnings:
            # Determine if it's a specific GPU warning or general info
            is_oom = "OOM" in w
            section.results.append(CheckResult(
                name=f"MoE VRAM: {name}",
                status=Status.WARN if is_oom else Status.PASS,
                detail=w,
                fix=(
                    "Use only GPUs with sufficient VRAM as RPC workers, "
                    "or offload to CPU on a single large-RAM machine."
                ) if is_oom else "",
            ))


def _check_moe_placement(section: Section, config: ClusterConfig) -> None:
    """Validate `moe_placement` config against the model file."""
    try:
        from .gguf_inspect import inspect_model
        from .moe_defuse import is_fused_expert
    except ImportError:
        return

    for name, model in config.models.items():
        placement = getattr(model, "moe_placement", None)
        if not placement or placement == "off":
            continue

        model_path = Path(model.path)
        if not model_path.exists() or _is_cross_platform_path(model.path):
            continue

        try:
            info = inspect_model(str(model_path))
        except Exception:
            continue

        if not info.is_moe:
            section.results.append(CheckResult(
                name=f"MoE placement: {name}",
                status=Status.WARN,
                detail=f"moe_placement: {placement} but model is dense",
                fix="Remove `moe_placement` from this model entry.",
            ))
            continue

        fused = any(is_fused_expert(t.name) for t in info.tensors)
        if fused:
            section.results.append(CheckResult(
                name=f"MoE placement: {name}",
                status=Status.WARN,
                detail="Fused expert tensors — `moe_placement` will no-op.",
                fix=f"Run `tightwad moe defuse {model.path} <out.gguf>` and point `moe_placement` at the indexed file.",
            ))
            continue

        if placement == "profile-guided":
            hot_profile = getattr(model, "moe_hot_profile", None)
            if not hot_profile:
                section.results.append(CheckResult(
                    name=f"MoE placement: {name}",
                    status=Status.WARN,
                    detail="profile-guided requires `moe_hot_profile`",
                    fix="Capture a profile via `tightwad moe profile` first, then set moe_hot_profile to the JSON path.",
                ))
                continue
            hot_path = Path(hot_profile).expanduser()
            if not hot_path.exists():
                section.results.append(CheckResult(
                    name=f"MoE placement: {name}",
                    status=Status.WARN,
                    detail=f"moe_hot_profile not found: {hot_profile}",
                    fix="Re-run `tightwad moe profile` or correct the path.",
                ))
                continue
            env_ok = config.env.get("LLAMA_LOG_MOE") == "1"
            if not env_ok:
                section.results.append(CheckResult(
                    name=f"MoE placement: {name}",
                    status=Status.WARN,
                    detail="profile-guided configured but LLAMA_LOG_MOE=1 not set — future profiles will be empty",
                    fix="Add `env: { LLAMA_LOG_MOE: \"1\" }` and rebuild llama.cpp with scripts/patches/llamacpp-moe-log.patch.",
                ))

        section.results.append(CheckResult(
            name=f"MoE placement: {name}",
            status=Status.PASS,
            detail=f"moe_placement: {placement}",
        ))


def check_network(config: ClusterConfig) -> Section:
    """Check RPC worker reachability and latency."""
    section = Section(title="Network")

    if not config.workers:
        section.results.append(CheckResult(
            name="RPC workers",
            status=Status.SKIP,
            detail="No workers configured",
        ))
        return section

    for w in config.workers:
        for gpu in w.gpus:
            if not gpu.rpc_port:
                continue
            status = worker.check_rpc_port(w.host, gpu.rpc_port)
            if status.alive:
                if status.latency_ms and status.latency_ms > 10:
                    section.results.append(CheckResult(
                        name=f"RPC {w.host}:{gpu.rpc_port}",
                        status=Status.WARN,
                        detail=f"{gpu.name} — {status.latency_ms}ms (>10ms, WiFi?)",
                        fix="Use wired Ethernet for best RPC performance",
                    ))
                else:
                    section.results.append(CheckResult(
                        name=f"RPC {w.host}:{gpu.rpc_port}",
                        status=Status.PASS,
                        detail=f"{gpu.name} — {status.latency_ms}ms",
                    ))
            else:
                fix = f"Start rpc-server on {w.host}:{gpu.rpc_port}"
                if w.ssh_user:
                    fix += f"\n  ssh {w.ssh_user}@{w.host} '{config.rpc_server_binary} -p {gpu.rpc_port}'"
                fix += _firewall_hint(w.host, gpu.rpc_port)
                section.results.append(CheckResult(
                    name=f"RPC {w.host}:{gpu.rpc_port}",
                    status=Status.FAIL,
                    detail=f"{gpu.name} — {status.error}",
                    fix=fix,
                ))

    return section


def check_services(config: ClusterConfig) -> Section:
    """Check coordinator, proxy, and swarm seeder PIDs and health endpoints."""
    section = Section(title="Services")

    # Coordinator PID
    coord_pid_path = Path.home() / ".tightwad" / "coordinator.pid"
    coord_running = False
    if coord_pid_path.exists():
        try:
            pid = int(coord_pid_path.read_text().strip())
            os.kill(pid, 0)
            coord_running = True
            section.results.append(CheckResult(
                name="Coordinator process",
                status=Status.PASS,
                detail=f"PID {pid}",
            ))
        except (ProcessLookupError, ValueError):
            section.results.append(CheckResult(
                name="Coordinator process",
                status=Status.WARN,
                detail="Stale PID file (process not running)",
                fix="tightwad start",
            ))
    else:
        section.results.append(CheckResult(
            name="Coordinator process",
            status=Status.SKIP,
            detail="Not started",
        ))

    # Coordinator /health
    if coord_running:
        health = worker.check_coordinator_health("127.0.0.1", config.coordinator_port)
        if health.get("alive"):
            section.results.append(CheckResult(
                name="Coordinator /health",
                status=Status.PASS,
                detail=f"http://127.0.0.1:{config.coordinator_port}/health",
            ))
        else:
            section.results.append(CheckResult(
                name="Coordinator /health",
                status=Status.FAIL,
                detail=health.get("error", "unhealthy"),
                fix=f"Check coordinator logs — process running but /health failing",
            ))

    # Proxy
    if config.proxy:
        proxy_pid_path = Path.home() / ".tightwad" / "proxy.pid"
        proxy_running = False
        if proxy_pid_path.exists():
            try:
                pid = int(proxy_pid_path.read_text().strip())
                os.kill(pid, 0)
                proxy_running = True
            except (ProcessLookupError, ValueError):
                pass

        if proxy_running:
            section.results.append(CheckResult(
                name="Proxy process",
                status=Status.PASS,
                detail=f"PID {pid}",
            ))
            # Check proxy status endpoint
            try:
                resp = httpx.get(
                    f"http://127.0.0.1:{config.proxy.port}/v1/tightwad/status",
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    section.results.append(CheckResult(
                        name="Proxy /status",
                        status=Status.PASS,
                        detail=f"http://127.0.0.1:{config.proxy.port}/v1/tightwad/status",
                    ))
                else:
                    section.results.append(CheckResult(
                        name="Proxy /status",
                        status=Status.WARN,
                        detail=f"HTTP {resp.status_code}",
                    ))
            except Exception as e:
                section.results.append(CheckResult(
                    name="Proxy /status",
                    status=Status.FAIL,
                    detail=str(e),
                ))
        else:
            section.results.append(CheckResult(
                name="Proxy process",
                status=Status.SKIP,
                detail="Not started (tightwad proxy start)",
            ))

        # Draft endpoint
        try:
            resp = httpx.get(f"{config.proxy.draft.url}/health", timeout=5.0)
            section.results.append(CheckResult(
                name=f"Draft endpoint ({config.proxy.draft.model_name})",
                status=Status.PASS if resp.status_code == 200 else Status.WARN,
                detail=config.proxy.draft.url,
            ))
        except Exception as e:
            section.results.append(CheckResult(
                name=f"Draft endpoint ({config.proxy.draft.model_name})",
                status=Status.WARN,
                detail=f"{config.proxy.draft.url} — {e}",
                fix=f"Start the draft model server at {config.proxy.draft.url}",
            ))

        # Target endpoint
        try:
            resp = httpx.get(f"{config.proxy.target.url}/health", timeout=5.0)
            section.results.append(CheckResult(
                name=f"Target endpoint ({config.proxy.target.model_name})",
                status=Status.PASS if resp.status_code == 200 else Status.WARN,
                detail=config.proxy.target.url,
            ))
        except Exception as e:
            section.results.append(CheckResult(
                name=f"Target endpoint ({config.proxy.target.model_name})",
                status=Status.WARN,
                detail=f"{config.proxy.target.url} — {e}",
                fix=f"Start the target model server at {config.proxy.target.url}",
            ))

    return section


def check_versions(config: ClusterConfig) -> Section:
    """Check llama-server version locally and on workers via SSH."""
    from .worker import check_version_match

    section = Section(title="Versions")

    ver = check_version_match(config)

    # Local version
    if ver.local.version:
        section.results.append(CheckResult(
            name="Local llama-server",
            status=Status.PASS,
            detail=ver.local.version,
        ))
    else:
        if _is_cross_platform_path(config.coordinator_binary):
            section.results.append(CheckResult(
                name="Local llama-server",
                status=Status.SKIP,
                detail="Cross-platform binary, not checkable locally",
            ))
        else:
            section.results.append(CheckResult(
                name="Local llama-server",
                status=Status.WARN,
                detail=f"Could not determine version ({ver.local.error or 'unknown'})",
                fix=f"{config.coordinator_binary} --version",
            ))

    # Worker versions
    for w_info in ver.workers:
        if w_info.version:
            section.results.append(CheckResult(
                name=f"Worker {w_info.host}",
                status=Status.PASS,
                detail=w_info.version,
                data={"version": w_info.version},
            ))
        else:
            section.results.append(CheckResult(
                name=f"Worker {w_info.host}",
                status=Status.SKIP,
                detail=f"SSH failed ({w_info.error or 'unknown'})",
                fix=f"ssh-copy-id user@{w_info.host}",
            ))

    # Unchecked workers (no ssh_user)
    for host in ver.unchecked:
        section.results.append(CheckResult(
            name=f"Worker {host}",
            status=Status.SKIP,
            detail="No ssh_user configured — cannot check remote version",
            fix=f"Add ssh_user to worker {host} in config for version checks",
        ))

    # Mismatches — FAIL, not WARN
    for w_info in ver.mismatched:
        section.results.append(CheckResult(
            name=f"Version match ({w_info.host})",
            status=Status.FAIL,
            detail=f"Local: {ver.local.version} ≠ Remote: {w_info.version}",
            fix=(
                "Mismatched llama.cpp versions cause silent failures during RPC inference.\n"
                "  Build the same commit on both machines.\n"
                "  Or use --skip-version-check with 'tightwad start' to bypass."
            ),
        ))

    # Summary pass if all matched
    if ver.matched and ver.workers:
        section.results.append(CheckResult(
            name="Version consistency",
            status=Status.PASS,
            detail=f"All nodes running {ver.local.version}",
        ))

    return section


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def check_peer_agents(config: ClusterConfig) -> Section:
    """Check reachability and version of peer agents on workers."""
    section = Section(title="Peer Agents")

    workers_with_peer = [w for w in config.workers if w.peer_port]
    if not workers_with_peer:
        section.results.append(CheckResult(
            name="Peer agents",
            status=Status.SKIP,
            detail="No workers have peer_port configured",
        ))
        return section

    # Check local peer agent
    from . import peer as peer_mod
    local_pid = peer_mod.read_pidfile()
    if local_pid is not None:
        try:
            os.kill(local_pid, 0)
            section.results.append(CheckResult(
                name="Local peer agent",
                status=Status.PASS,
                detail=f"PID {local_pid}",
            ))
        except ProcessLookupError:
            section.results.append(CheckResult(
                name="Local peer agent",
                status=Status.WARN,
                detail="Stale PID file (process not running)",
                fix="tightwad peer start",
            ))
    else:
        section.results.append(CheckResult(
            name="Local peer agent",
            status=Status.SKIP,
            detail="Not started (tightwad peer start)",
        ))

    # Resolve peer auth token
    peer_token = None
    if config.peer and config.peer.auth_token:
        peer_token = config.peer.auth_token

    # Check remote peer agents
    for w in workers_with_peer:
        url = f"http://{w.host}:{w.peer_port}/v1/peer/version"
        headers = {}
        if peer_token:
            headers["Authorization"] = f"Bearer {peer_token}"

        try:
            resp = httpx.get(url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                tw_ver = data.get("tightwad_version", "unknown")
                llama_ver = data.get("llama_server_version", "unknown")
                section.results.append(CheckResult(
                    name=f"Peer {w.host}:{w.peer_port}",
                    status=Status.PASS,
                    detail=f"tightwad={tw_ver}, llama-server={llama_ver}",
                    data=data,
                ))
            elif resp.status_code == 401:
                section.results.append(CheckResult(
                    name=f"Peer {w.host}:{w.peer_port}",
                    status=Status.FAIL,
                    detail="Authentication failed (401)",
                    fix="Check peer.auth_token in config matches the remote peer agent",
                ))
            else:
                section.results.append(CheckResult(
                    name=f"Peer {w.host}:{w.peer_port}",
                    status=Status.WARN,
                    detail=f"HTTP {resp.status_code}",
                ))
        except Exception as e:
            section.results.append(CheckResult(
                name=f"Peer {w.host}:{w.peer_port}",
                status=Status.FAIL,
                detail=str(e),
                fix=f"Start peer agent on {w.host}: tightwad peer start --port {w.peer_port}",
            ))

    return section


def check_model_families(config: ClusterConfig) -> Section:
    """Check draft/target model family compatibility for speculative decoding."""
    import asyncio
    from .family import check_proxy_families

    section = Section(title="Model Families")

    if not config.proxy:
        section.results.append(CheckResult(
            name="Family compatibility",
            status=Status.SKIP,
            detail="No proxy configured — family check not applicable",
        ))
        return section

    try:
        result = asyncio.run(check_proxy_families(
            draft_url=config.proxy.draft.url,
            draft_model=config.proxy.draft.model_name,
            draft_backend=config.proxy.draft.backend,
            target_url=config.proxy.target.url,
            target_model=config.proxy.target.model_name,
            target_backend=config.proxy.target.backend,
        ))
    except Exception as exc:
        section.results.append(CheckResult(
            name="Family detection",
            status=Status.WARN,
            detail=f"Could not detect model families: {exc}",
            fix="Ensure draft and target servers are running and reachable",
        ))
        return section

    # Report detected architectures
    if result.draft:
        section.results.append(CheckResult(
            name=f"Draft architecture ({result.draft.model_name})",
            status=Status.PASS,
            detail=f"{result.draft.arch} (family: {result.draft.family}, via {result.draft.source})",
        ))
    else:
        section.results.append(CheckResult(
            name=f"Draft architecture ({config.proxy.draft.model_name})",
            status=Status.WARN,
            detail="Could not detect architecture",
            fix=f"Ensure the draft server at {config.proxy.draft.url} is running",
        ))

    if result.target:
        section.results.append(CheckResult(
            name=f"Target architecture ({result.target.model_name})",
            status=Status.PASS,
            detail=f"{result.target.arch} (family: {result.target.family}, via {result.target.source})",
        ))
    else:
        section.results.append(CheckResult(
            name=f"Target architecture ({config.proxy.target.model_name})",
            status=Status.WARN,
            detail="Could not detect architecture",
            fix=f"Ensure the target server at {config.proxy.target.url} is running",
        ))

    # Compatibility verdict
    if result.draft and result.target:
        if result.compatible:
            section.results.append(CheckResult(
                name="Family compatibility",
                status=Status.PASS,
                detail=result.message,
            ))
        else:
            section.results.append(CheckResult(
                name="Family compatibility",
                status=Status.FAIL,
                detail=result.message,
                fix=(
                    f"Replace your draft model with a smaller model from the "
                    f"{result.target.family} family. Mismatched families cause "
                    f"<5% acceptance rates (10x slower than no speculation)."
                ),
            ))

    return section


def run_doctor(config_path: str | Path | None = None) -> DoctorReport:
    """Run all diagnostic checks and return a report."""
    report = DoctorReport()

    # 1. Config (short-circuits if config fails)
    config_section, config = check_config(config_path)
    report.sections.append(config_section)
    if config is None:
        return report

    # 2. Binaries
    report.sections.append(check_binaries(config))

    # 3. Models
    report.sections.append(check_models(config))

    # 4. Network
    report.sections.append(check_network(config))

    # 5. Services
    report.sections.append(check_services(config))

    # 6. Versions
    report.sections.append(check_versions(config))

    # 7. Peer agents
    report.sections.append(check_peer_agents(config))

    # 8. Model family compatibility (proxy only)
    report.sections.append(check_model_families(config))

    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_report(console: Console, report: DoctorReport, show_fix: bool = False) -> None:
    """Render the doctor report to the console with Rich formatting."""
    for section in report.sections:
        console.print()
        console.print(Rule(section.title, style="bold"))
        for r in section.results:
            icon = _ICONS[r.status]
            line = f"  {icon} {r.name}"
            if r.detail:
                line += f"  [dim]{r.detail}[/dim]"
            console.print(line)

            if show_fix and r.fix and r.status in (Status.FAIL, Status.WARN):
                for fix_line in r.fix.split("\n"):
                    console.print(f"      [cyan]→ {fix_line}[/cyan]")

    # Summary
    console.print()
    total = sum(len(s.results) for s in report.sections)
    passed = sum(
        1 for s in report.sections for r in s.results if r.status == Status.PASS
    )
    failed = sum(
        1 for s in report.sections for r in s.results if r.status == Status.FAIL
    )
    warned = sum(
        1 for s in report.sections for r in s.results if r.status == Status.WARN
    )
    skipped = sum(
        1 for s in report.sections for r in s.results if r.status == Status.SKIP
    )

    parts = [f"[green]{passed} passed[/green]"]
    if failed:
        parts.append(f"[red]{failed} failed[/red]")
    if warned:
        parts.append(f"[yellow]{warned} warnings[/yellow]")
    if skipped:
        parts.append(f"[dim]{skipped} skipped[/dim]")

    summary = ", ".join(parts)
    if report.passed:
        console.print(f"[bold green]All checks passed![/bold green]  ({summary})")
    else:
        console.print(f"[bold red]{failed} check(s) failed.[/bold red]  ({summary})")
        if not show_fix:
            console.print("[dim]Run with --fix to see suggested fixes.[/dim]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_cross_platform_path(path: str) -> bool:
    """Detect Windows-style paths when running on a non-Windows platform."""
    if platform.system() == "Windows":
        return False
    return bool(re.match(r"^[A-Za-z]:[/\\]", path))


def _firewall_hint(host: str, port: int) -> str:
    """Return platform-specific firewall hint."""
    system = platform.system()
    if system == "Darwin":
        return f"\n  Check macOS firewall: /usr/libexec/ApplicationFirewall/socketfilterfw --listapps"
    elif system == "Linux":
        return f"\n  Check firewall: sudo ufw status  # or: sudo iptables -L -n | grep {port}"
    return ""


def _get_llama_version(binary: str) -> str | None:
    """Get llama-server version from local binary."""
    resolved = shutil.which(binary) or binary
    try:
        result = subprocess.run(
            [resolved, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        output = (result.stdout + result.stderr).strip()
        if output:
            # Extract first meaningful line
            for line in output.splitlines():
                line = line.strip()
                if line:
                    return line
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _get_remote_llama_version(ssh_user: str, host: str) -> str | None:
    """Get llama-server version from a remote machine via SSH BatchMode."""
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                f"{ssh_user}@{host}",
                "llama-server --version 2>&1 || rpc-server --version 2>&1 || echo unknown",
            ],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout.strip()
        if output and output != "unknown":
            for line in output.splitlines():
                line = line.strip()
                if line:
                    return line
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None
