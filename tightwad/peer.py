"""Peer agent HTTP daemon for local network communication between tightwad instances."""

from __future__ import annotations

import hmac
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import PeerConfig

logger = logging.getLogger("tightwad.peer")

PIDFILE = Path.home() / ".tightwad" / "peer.pid"

# ---------------------------------------------------------------------------
# Startup timestamp (set when create_app is called)
# ---------------------------------------------------------------------------

_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Auth middleware (same pattern as proxy.py)
# ---------------------------------------------------------------------------


class TokenAuthMiddleware:
    """Starlette ASGI middleware that enforces Bearer-token authentication.

    When a token is configured every HTTP request must include::

        Authorization: Bearer <token>

    Requests missing or presenting an incorrect token receive a ``401
    Unauthorized`` response.  Non-HTTP scopes (WebSocket, lifespan) pass
    through unchanged.

    When no token is configured the middleware is a transparent no-op.
    """

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.token:
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {self.token}"

        if not hmac.compare_digest(auth_header, expected):
            response = Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Process manager for rpc-server child processes
# ---------------------------------------------------------------------------


@dataclass
class ManagedProcess:
    """A child process started by the peer agent."""

    pid: int
    port: int
    cmd: list[str]
    started_at: float


class ProcessManager:
    """Tracks child processes started via the peer agent (PID tracking, start/stop)."""

    def __init__(self) -> None:
        self._processes: dict[int, ManagedProcess] = {}

    def start(self, cmd: list[str], port: int, capture_stderr: bool = True) -> ManagedProcess:
        """Start a subprocess and track it.

        When ``capture_stderr`` is True (default), stderr is rotated into
        ``~/.tightwad/logs/rpc-{port}.log`` (10 MB cap) so
        ``tightwad moe profile`` can tail it for expert-routing events.
        """
        if capture_stderr:
            log_path = rpc_log_path(port)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(log_path)
            stderr_target = open(log_path, "ab")
        else:
            stderr_target = subprocess.DEVNULL

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_target,
        )
        managed = ManagedProcess(
            pid=proc.pid,
            port=port,
            cmd=cmd,
            started_at=time.time(),
        )
        self._processes[port] = managed
        logger.info("Started process PID %d on port %d: %s", proc.pid, port, cmd)
        return managed

    def stop(self, port: int) -> bool:
        """Stop a managed process by port. Returns True if stopped."""
        managed = self._processes.pop(port, None)
        if managed is None:
            return False
        try:
            os.kill(managed.pid, signal.SIGTERM)
            logger.info("Stopped process PID %d on port %d", managed.pid, port)
            return True
        except ProcessLookupError:
            logger.warning("Process PID %d already exited", managed.pid)
            return True

    def list_processes(self) -> list[dict]:
        """Return info about all managed processes."""
        result = []
        for port, mp in self._processes.items():
            alive = True
            try:
                os.kill(mp.pid, 0)
            except ProcessLookupError:
                alive = False
            result.append({
                "pid": mp.pid,
                "port": mp.port,
                "cmd": mp.cmd,
                "started_at": mp.started_at,
                "alive": alive,
            })
        return result


# Module-level process manager instance
_process_manager = ProcessManager()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_llama_server_version() -> str | None:
    """Get llama-server version string."""
    binary = shutil.which("llama-server")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        output = (result.stdout + result.stderr).strip()
        for line in output.splitlines():
            line = line.strip()
            if line:
                return line
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _get_memory_info() -> dict:
    """Get memory info without psutil.

    Uses /proc/meminfo on Linux, vm_stat on macOS.
    """
    system = platform.system()

    if system == "Linux":
        try:
            info: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        val_kb = int(parts[1])
                        info[key] = val_kb
            return {
                "total_mb": info.get("MemTotal", 0) // 1024,
                "available_mb": info.get("MemAvailable", 0) // 1024,
                "free_mb": info.get("MemFree", 0) // 1024,
            }
        except (OSError, ValueError):
            pass

    if system == "Darwin":
        try:
            # Total memory
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            total_bytes = int(result.stdout.strip())
            total_mb = total_bytes // (1024 * 1024)

            # vm_stat for free/inactive pages
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True, text=True, timeout=5,
            )
            pages: dict[str, int] = {}
            for line in result.stdout.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip().rstrip(".")
                    try:
                        pages[key.strip()] = int(val)
                    except ValueError:
                        pass
            page_size = 16384  # ARM64 macOS
            free_pages = pages.get("Pages free", 0)
            inactive_pages = pages.get("Pages inactive", 0)
            available_mb = (free_pages + inactive_pages) * page_size // (1024 * 1024)

            return {
                "total_mb": total_mb,
                "available_mb": available_mb,
                "free_mb": free_pages * page_size // (1024 * 1024),
            }
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass

    return {}


def _get_load_avg() -> list[float] | None:
    """Get system load averages (1, 5, 15 min)."""
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        # Windows doesn't have getloadavg
        return None


def _scan_gguf_files(model_dirs: list[str]) -> list[dict]:
    """Scan directories for GGUF files."""
    results = []
    for dir_path in model_dirs:
        p = Path(dir_path).expanduser()
        if not p.is_dir():
            continue
        for gguf in p.glob("*.gguf"):
            try:
                size = gguf.stat().st_size
                results.append({
                    "name": gguf.name,
                    "path": str(gguf),
                    "size_bytes": size,
                    "size_gb": round(size / (1024**3), 2),
                })
            except OSError:
                continue
    return results


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


async def version_endpoint(request: Request) -> JSONResponse:
    """GET /v1/peer/version -- returns tightwad version, llama-server version, platform."""
    from . import __version__

    return JSONResponse({
        "tightwad_version": __version__,
        "llama_server_version": _get_llama_server_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    })


async def health_endpoint(request: Request) -> JSONResponse:
    """GET /v1/peer/health -- returns uptime, CPU load, memory, disk."""
    uptime = time.time() - _start_time

    disk = {}
    try:
        usage = shutil.disk_usage("/")
        disk = {
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
        }
    except OSError:
        pass

    return JSONResponse({
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "load_avg": _get_load_avg(),
        "memory": _get_memory_info(),
        "disk": disk,
    })


async def gpu_endpoint(request: Request) -> JSONResponse:
    """GET /v1/peer/gpu -- returns GPU info via gpu_detect."""
    from .gpu_detect import detect_gpus

    gpus = detect_gpus()
    return JSONResponse({
        "gpus": [
            {
                "name": gpu.name,
                "vram_mb": gpu.vram_mb,
                "backend": gpu.backend,
                "index": gpu.index,
            }
            for gpu in gpus
        ],
    })


async def models_endpoint(request: Request) -> JSONResponse:
    """GET /v1/peer/models -- lists GGUF files in configured model directories."""
    model_dirs = request.app.state.model_dirs
    models = _scan_gguf_files(model_dirs)
    return JSONResponse({
        "models": models,
        "model_dirs": model_dirs,
    })


async def rpc_start_endpoint(request: Request) -> JSONResponse:
    """POST /v1/peer/rpc/start -- start an rpc-server process on given port."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    port = body.get("port")
    if not port or not isinstance(port, int):
        return JSONResponse({"error": "port (int) is required"}, status_code=400)

    binary = body.get("binary", "rpc-server")
    resolved = shutil.which(binary) or binary

    if not Path(resolved).exists() and not shutil.which(resolved):
        return JSONResponse(
            {"error": f"binary not found: {binary}"},
            status_code=400,
        )

    host = body.get("host", "0.0.0.0")
    cmd = [resolved, "-H", host, "-p", str(port)]

    try:
        managed = _process_manager.start(cmd, port)
        return JSONResponse({
            "status": "started",
            "pid": managed.pid,
            "port": managed.port,
            "cmd": managed.cmd,
        })
    except Exception as e:
        logger.error("Failed to start rpc-server: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


async def rpc_stop_endpoint(request: Request) -> JSONResponse:
    """POST /v1/peer/rpc/stop -- stop a managed rpc-server process."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    port = body.get("port")
    if not port or not isinstance(port, int):
        return JSONResponse({"error": "port (int) is required"}, status_code=400)

    stopped = _process_manager.stop(port)
    if stopped:
        return JSONResponse({"status": "stopped", "port": port})
    return JSONResponse(
        {"error": f"no managed process on port {port}"},
        status_code=404,
    )


RPC_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def rpc_log_path(port: int) -> Path:
    return Path.home() / ".tightwad" / "logs" / f"rpc-{port}.log"


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > RPC_LOG_MAX_BYTES:
            rotated = path.with_suffix(path.suffix + ".1")
            rotated.unlink(missing_ok=True)
            path.rename(rotated)
    except OSError as exc:
        logger.warning("Failed to rotate %s: %s", path, exc)


async def moe_profile_endpoint(request: Request) -> JSONResponse:
    """GET /v1/peer/moe/profile?port=N — aggregated hot-expert counts."""
    port_str = request.query_params.get("port")
    if not port_str or not port_str.isdigit():
        return JSONResponse({"error": "port (int) query param is required"},
                            status_code=400)
    port = int(port_str)
    log_path = rpc_log_path(port)
    if not log_path.exists():
        return JSONResponse({"error": f"no log for port {port}"},
                            status_code=404)
    try:
        from .moe_profile import parse_log_file
    except ImportError:
        return JSONResponse({"error": "moe_profile module unavailable"},
                            status_code=500)
    profile = parse_log_file(log_path)
    top = [{"layer": h.layer, "expert": h.expert, "count": h.count}
           for h in profile.top_n(64)]
    return JSONResponse({
        "port": port,
        "total_tokens": profile.total_tokens,
        "source": profile.source,
        "top_experts": top,
        "per_layer_skew": profile.per_layer_skew(),
    })


async def logs_endpoint(request: Request) -> JSONResponse:
    """GET /v1/peer/logs -- return recent log lines."""
    service = request.query_params.get("service", "peer")
    lines = int(request.query_params.get("lines", "50"))
    lines = min(lines, 1000)  # cap at 1000

    log_dir = Path.home() / ".tightwad" / "logs"
    log_file = log_dir / f"{service}.log"

    if not log_file.exists():
        return JSONResponse({
            "service": service,
            "lines": [],
            "error": f"log file not found: {log_file}",
        })

    try:
        text = log_file.read_text()
        all_lines = text.splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return JSONResponse({
            "service": service,
            "lines": tail,
            "total_lines": len(all_lines),
        })
    except OSError as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: PeerConfig) -> Starlette:
    """Create the Starlette ASGI application for the peer agent."""
    global _start_time
    _start_time = time.time()

    middleware = []
    if config.auth_token:
        from starlette.middleware import Middleware
        middleware.append(
            Middleware(TokenAuthMiddleware, token=config.auth_token)
        )
    else:
        logger.warning(
            "Peer agent starting WITHOUT authentication. "
            "Set peer.auth_token in config or TIGHTWAD_PEER_TOKEN env var."
        )

    app = Starlette(
        routes=[
            Route("/v1/peer/version", version_endpoint, methods=["GET"]),
            Route("/v1/peer/health", health_endpoint, methods=["GET"]),
            Route("/v1/peer/gpu", gpu_endpoint, methods=["GET"]),
            Route("/v1/peer/models", models_endpoint, methods=["GET"]),
            Route("/v1/peer/rpc/start", rpc_start_endpoint, methods=["POST"]),
            Route("/v1/peer/rpc/stop", rpc_stop_endpoint, methods=["POST"]),
            Route("/v1/peer/moe/profile", moe_profile_endpoint, methods=["GET"]),
            Route("/v1/peer/logs", logs_endpoint, methods=["GET"]),
        ],
        middleware=middleware,
    )

    # Store model_dirs on app state for the models endpoint
    app.state.model_dirs = config.model_dirs

    return app


# ---------------------------------------------------------------------------
# Pidfile management
# ---------------------------------------------------------------------------


def write_pidfile() -> None:
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))


def remove_pidfile() -> None:
    PIDFILE.unlink(missing_ok=True)


def read_pidfile() -> int | None:
    if PIDFILE.exists():
        try:
            return int(PIDFILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def stop_peer() -> bool:
    """Stop the peer agent daemon. Returns True if it was running."""
    pid = read_pidfile()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        remove_pidfile()
        return True
    except ProcessLookupError:
        remove_pidfile()
        return False
