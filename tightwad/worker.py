"""RPC worker health checks and lifecycle management."""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field

import httpx

from .config import ClusterConfig, Worker

logger = logging.getLogger("tightwad.worker")


@dataclass
class WorkerStatus:
    host: str
    port: int
    gpu_name: str
    alive: bool
    latency_ms: float | None = None
    error: str | None = None


@dataclass
class VersionInfo:
    """Version information for a coordinator or worker node."""

    host: str
    version: str | None
    error: str | None = None


@dataclass
class VersionCheckResult:
    """Result of comparing coordinator and worker versions."""

    matched: bool
    local: VersionInfo
    workers: list[VersionInfo] = field(default_factory=list)
    mismatched: list[VersionInfo] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.matched:
            return f"All versions match: {self.local.version}"
        parts = [f"Coordinator ({self.local.host}): {self.local.version}"]
        for w in self.mismatched:
            parts.append(f"  Worker {w.host}: {w.version}")
        return (
            "Version mismatch — mismatched llama.cpp versions cause silent RPC "
            "failures.\n" + "\n".join(parts) + "\n"
            "Build the same llama.cpp commit on all machines, or use "
            "--skip-version-check to proceed anyway."
        )


def check_rpc_port(host: str, port: int, timeout: float = 2.0) -> WorkerStatus:
    """Check if an RPC worker is reachable via TCP connect."""
    gpu_name = f"{host}:{port}"
    start = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            latency = (time.monotonic() - start) * 1000
            return WorkerStatus(
                host=host, port=port, gpu_name=gpu_name,
                alive=True, latency_ms=round(latency, 1),
            )
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        return WorkerStatus(
            host=host, port=port, gpu_name=gpu_name,
            alive=False, error=str(e),
        )


def check_all_workers(config: ClusterConfig) -> list[WorkerStatus]:
    """Check all RPC workers defined in the cluster config."""
    statuses = []
    for worker in config.workers:
        for gpu in worker.gpus:
            if gpu.rpc_port:
                statuses.append(check_rpc_port(worker.host, gpu.rpc_port))
    return statuses


def check_coordinator_health(
    host: str = "127.0.0.1", port: int = 8080, timeout: float = 5.0
) -> dict:
    """Check if the coordinator llama-server is healthy via /health endpoint."""
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=timeout)
        return {"alive": resp.status_code == 200, "status": resp.json()}
    except Exception as e:
        return {"alive": False, "error": str(e)}


def wait_for_workers(
    config: ClusterConfig,
    timeout: float = 30.0,
    interval: float = 2.0,
) -> list[WorkerStatus]:
    """Wait for all RPC workers to become available."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = check_all_workers(config)
        if all(s.alive for s in statuses):
            return statuses
        time.sleep(interval)
    return check_all_workers(config)


# ---------------------------------------------------------------------------
# Version detection and matching
# ---------------------------------------------------------------------------


def get_local_version(binary: str = "llama-server") -> VersionInfo:
    """Get llama-server version from the local binary."""
    resolved = shutil.which(binary) or binary
    try:
        result = subprocess.run(
            [resolved, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        output = (result.stdout + result.stderr).strip()
        for line in output.splitlines():
            line = line.strip()
            if line:
                return VersionInfo(host="localhost", version=line)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return VersionInfo(host="localhost", version=None, error=str(e))
    return VersionInfo(host="localhost", version=None, error="no version output")


def get_remote_version(ssh_user: str, host: str) -> VersionInfo:
    """Get llama-server version from a remote worker via SSH."""
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
                    return VersionInfo(host=host, version=line)
    except (subprocess.TimeoutExpired, OSError) as e:
        return VersionInfo(host=host, version=None, error=str(e))
    return VersionInfo(host=host, version=None, error="could not determine version")


def get_remote_version_via_peer(
    host: str, port: int, token: str | None = None
) -> VersionInfo:
    """Get llama-server version from a remote worker via its peer agent."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(
            f"http://{host}:{port}/v1/peer/version",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            version = data.get("llama_server_version")
            return VersionInfo(host=host, version=version)
        return VersionInfo(
            host=host, version=None,
            error=f"peer agent returned HTTP {resp.status_code}",
        )
    except Exception as e:
        return VersionInfo(host=host, version=None, error=str(e))


def check_version_match(config: ClusterConfig) -> VersionCheckResult:
    """Compare coordinator and worker llama.cpp versions.

    Returns a VersionCheckResult indicating whether all versions match.
    Workers without ssh_user are listed in unchecked (can't verify).
    """
    local = get_local_version(config.coordinator_binary)

    workers: list[VersionInfo] = []
    mismatched: list[VersionInfo] = []
    unchecked: list[str] = []

    # Resolve peer auth token from config if available
    peer_token = None
    if config.peer and config.peer.auth_token:
        peer_token = config.peer.auth_token

    seen_hosts: set[str] = set()
    for w in config.workers:
        if w.host in seen_hosts:
            continue
        seen_hosts.add(w.host)

        # Try peer agent first when peer_port is configured
        info = None
        if w.peer_port:
            info = get_remote_version_via_peer(w.host, w.peer_port, peer_token)
            if info.version:
                workers.append(info)
                if local.version and info.version != local.version:
                    mismatched.append(info)
                continue
            # Peer agent failed — fall back to SSH if available
            logger.debug(
                "Peer agent on %s:%d failed (%s), trying SSH",
                w.host, w.peer_port, info.error,
            )

        if not w.ssh_user:
            unchecked.append(w.host)
            continue

        info = get_remote_version(w.ssh_user, w.host)
        workers.append(info)

        if info.version and local.version and info.version != local.version:
            mismatched.append(info)

    matched = len(mismatched) == 0
    return VersionCheckResult(
        matched=matched,
        local=local,
        workers=workers,
        mismatched=mismatched,
        unchecked=unchecked,
    )
