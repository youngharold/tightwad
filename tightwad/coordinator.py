"""Coordinator: launches llama-server with RPC backend flags."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from .config import ClusterConfig, ModelConfig
from .worker import (
    check_all_workers,
    check_coordinator_health,
    check_version_match,
)

logger = logging.getLogger("tightwad.coordinator")

PIDFILE = Path.home() / ".tightwad" / "coordinator.pid"
LOGDIR = Path.home() / ".tightwad" / "logs"
COORDINATOR_LOG = LOGDIR / "coordinator.log"


def _pid_alive(pid: int) -> bool:
    """Portable process-liveness probe.

    ``os.kill(pid, 0)`` is not a liveness check on Windows: CPython routes
    signal 0 to GenerateConsoleCtrlEvent(CTRL_C_EVENT), which succeeds (or
    raises plain OSError, never ProcessLookupError) for dead PIDs — so
    stale-pidfile recovery would never trigger there.
    """
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user
        return True
    except OSError:
        return False


def _health_host(host: str | None) -> str:
    """Map a llama-server bind address to an address health checks can reach.

    Wildcard binds (0.0.0.0 / ::) accept loopback connections; a specific
    bind address does not, so health checks must target it directly.
    """
    if host in (None, "", "0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return host


def _boot_time() -> float | None:
    """Best-effort system boot time as an epoch timestamp, None if unknown."""
    if os.name == "nt":
        try:
            import ctypes

            return time.time() - ctypes.windll.kernel32.GetTickCount64() / 1000.0
        except Exception:
            return None
    # Linux: /proc/stat "btime <epoch>" line
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    # macOS/BSD: sysctl kern.boottime → "{ sec = 1719900000, usec = ... } ..."
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.boottime"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r"sec\s*=\s*(\d+)", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _write_pidfile(pid: int, port: int, config_path: str | None = None,
                   model_name: str | None = None,
                   host: str | None = None) -> None:
    """Write JSON metadata to the pidfile."""
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pid": pid,
        "port": port,
        "host": host,
        "config": config_path,
        "model": model_name,
        "started": time.time(),
    }
    PIDFILE.write_text(json.dumps(data))


def _read_pidfile() -> dict | None:
    """Read pidfile, handling both JSON (new) and plain-int (legacy) formats.

    Returns dict with at least {"pid": int} or None if no pidfile.
    """
    if not PIDFILE.exists():
        return None
    text = PIDFILE.read_text().strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "pid" in data:
            # A pidfile written before the current boot cannot refer to our
            # process — at best its PID has been recycled by an unrelated
            # process, which stop() must not signal and start() must not
            # treat as "already running". Only discard when boot time is
            # known and clearly after "started" (10s clock-skew tolerance).
            started = data.get("started")
            if isinstance(started, (int, float)):
                boot = _boot_time()
                if boot is not None and started < boot - 10:
                    logger.info(
                        "Discarding stale pidfile (PID %s predates current "
                        "boot)", data.get("pid"),
                    )
                    PIDFILE.unlink(missing_ok=True)
                    return None
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    # Legacy: plain integer PID
    try:
        pid = int(text)
        return {"pid": pid}
    except ValueError:
        return None


def build_server_args(config: ClusterConfig, model: ModelConfig) -> list[str]:
    """Build llama-server command-line arguments."""
    args = [
        config.coordinator_binary,
        "-m", model.path,
        "-ngl", "999",
        "--host", config.coordinator_host,
        "--port", str(config.coordinator_port),
        "--ctx-size", str(model.ctx_size),
        "-n", str(model.predict),
    ]

    if model.flash_attn:
        args.extend(["--flash-attn", "on"])

    # RPC workers
    rpc_addrs = config.rpc_addresses
    if rpc_addrs:
        args.extend(["--rpc", ",".join(rpc_addrs)])

    # Tensor split across all GPUs (coordinator locals first, then RPC workers)
    split = config.tensor_split()
    if len(split) > 1:
        args.extend(["--tensor-split", ",".join(str(s) for s in split)])

    # Expert-aware placement: emit one --override-tensor flag per (layer, device)
    # pair when the model opts in via moe_placement. Silently no-op for dense
    # models, opted-out models, fused-expert GGUFs, or missing gguf package.
    if getattr(model, "moe_placement", None) and model.moe_placement != "off":
        for flag in _moe_override_tensor_flags(config, model):
            args.extend(["--override-tensor", flag])

    # Backend-specific and user-supplied extra arguments
    args.extend(config.extra_args)

    return args


def _moe_override_tensor_flags(config: ClusterConfig, model: ModelConfig) -> list[str]:
    try:
        from . import gguf_inspect as _gguf_inspect
        from .moe_placement import build_slots, plan_expert_placement
    except ImportError:
        return []

    try:
        info = _gguf_inspect.inspect_model(model.path)
    except Exception as exc:
        logger.warning("moe_placement: inspect_model(%s) failed: %s", model.path, exc)
        return []
    if not info.is_moe:
        return []

    hot = None
    if model.moe_placement == "profile-guided" and model.moe_hot_profile:
        try:
            from .moe_profile import HotExpertProfile
            hot = HotExpertProfile.load(model.moe_hot_profile).frequency()
        except ImportError:
            logger.warning(
                "moe_placement: profile-guided requested but moe_profile "
                "module not available; falling back to balanced",
            )
        except Exception as exc:
            logger.warning(
                "moe_placement: failed to load hot profile %s: %s",
                model.moe_hot_profile, exc,
            )

    plan = plan_expert_placement(
        info, build_slots(config),
        hot_experts=hot, strategy=model.moe_placement,
    )
    if plan.fused_fallback:
        logger.warning(
            "moe_placement: %s has fused expert tensors — run `tightwad moe "
            "defuse` to enable per-expert placement. Falling back to "
            "layer-only split.", model.path,
        )
    return plan.override_tensor_args


def start(
    config: ClusterConfig,
    model_name: str | None = None,
    skip_version_check: bool = False,
) -> int:
    """Start the coordinator llama-server.

    Parameters
    ----------
    config:
        Cluster configuration.
    model_name:
        Model to load (default from config if None).
    skip_version_check:
        If False (default), refuse to start when coordinator and worker
        llama.cpp versions don't match.  Set True or pass
        ``--skip-version-check`` on the CLI to bypass.

    Returns the subprocess PID.
    """
    # Resolve model
    if model_name:
        model = config.models.get(model_name)
        if not model:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {', '.join(config.models)}"
            )
    else:
        model = config.default_model()
        if not model:
            raise ValueError("No models configured")

    # Check if already running
    pidfile_data = _read_pidfile()
    if pidfile_data is not None:
        pid = pidfile_data["pid"]
        if _pid_alive(pid):
            raise RuntimeError(
                f"Coordinator already running (PID {pid}). "
                "Use 'tightwad stop' first."
            )
        PIDFILE.unlink()

    # Health-check RPC workers
    worker_statuses = check_all_workers(config)
    dead = [s for s in worker_statuses if not s.alive]
    if dead:
        dead_str = ", ".join(f"{s.host}:{s.port}" for s in dead)
        raise RuntimeError(
            f"RPC workers not reachable: {dead_str}\n"
            "Start rpc-server on the worker machine first."
        )

    # Version matching: refuse to start if coordinator/worker versions differ
    if config.workers and not skip_version_check:
        ver = check_version_match(config)
        if not ver.matched:
            raise RuntimeError(ver.message)
        if ver.local.version:
            logger.info("Version check passed: %s", ver.local.version)
        if ver.unchecked:
            logger.warning(
                "Could not check versions for workers without ssh_user: %s",
                ", ".join(ver.unchecked),
            )

    # MoE VRAM warning (non-blocking)
    if config.workers:
        _warn_moe_vram(config, model)

    # Build and launch
    args = build_server_args(config, model)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    # Open the log file, pass it to Popen (child inherits the FD), then
    # immediately close the parent-side reference.  The child process keeps
    # the file open as long as it runs, which is intentional.  Closing the
    # parent's copy prevents FD accumulation when start() is called
    # repeatedly (e.g. during model swaps that call swap_model()).
    run_env = {**os.environ, **config.env} if config.env else None

    log_fh = open(COORDINATOR_LOG, "a")
    try:
        proc = subprocess.Popen(
            args,
            stdout=log_fh,
            stderr=log_fh,
            env=run_env,
        )
    finally:
        # Always release the parent-side reference, even if Popen fails.
        log_fh.close()

    _write_pidfile(
        pid=proc.pid,
        port=config.coordinator_port,
        model_name=model.name,
        host=config.coordinator_host,
    )

    return proc.pid


def stop(wait_timeout: float = 30.0) -> bool:
    """Stop the coordinator llama-server.

    Blocks until the process has actually exited (up to *wait_timeout*
    seconds, escalating to SIGKILL) so callers like ``swap_model()`` can
    immediately rebind the HTTP port and reuse the freed VRAM. Returning
    right after SIGTERM would race the replacement server against a
    multi-second llama-server teardown.
    """
    pidfile_data = _read_pidfile()
    if pidfile_data is None:
        return False

    pid = pidfile_data["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        # Already gone (POSIX ProcessLookupError) or an invalid handle on
        # Windows, where os.kill(SIGTERM) is TerminateProcess.
        pass
    else:
        deadline = time.monotonic() + wait_timeout
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _pid_alive(pid):
            sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            logger.warning(
                "Coordinator (PID %d) still running %.0fs after SIGTERM — "
                "sending SIGKILL", pid, wait_timeout,
            )
            try:
                os.kill(pid, sigkill)
            except OSError:
                pass
            kill_deadline = time.monotonic() + 5.0
            while _pid_alive(pid) and time.monotonic() < kill_deadline:
                time.sleep(0.1)
    PIDFILE.unlink(missing_ok=True)
    return True


def status(config: ClusterConfig | None = None) -> dict:
    """Get full cluster status.

    If *config* is None, attempts to read port from pidfile metadata
    for config-less status checks.
    """
    # Coordinator
    coord_running = False
    coord_pid = None
    pidfile_data = _read_pidfile()
    if pidfile_data is not None:
        coord_pid = pidfile_data["pid"]
        if _pid_alive(coord_pid):
            coord_running = True
        else:
            PIDFILE.unlink(missing_ok=True)
            coord_pid = None
            pidfile_data = None

    # Determine port/host: config takes precedence, then pidfile metadata
    port = config.coordinator_port if config else pidfile_data.get("port", 8080) if pidfile_data else 8080
    bind_host = (
        config.coordinator_host if config
        else pidfile_data.get("host") if pidfile_data
        else None
    )

    coord_health = None
    if coord_running:
        coord_health = check_coordinator_health(_health_host(bind_host), port)

    # Workers (requires config)
    worker_statuses = check_all_workers(config) if config else []

    result = {
        "coordinator": {
            "running": coord_running,
            "pid": coord_pid,
            "port": port,
            "health": coord_health,
        },
        "workers": [
            {
                "address": f"{s.host}:{s.port}",
                "alive": s.alive,
                "latency_ms": s.latency_ms,
                "error": s.error,
            }
            for s in worker_statuses
        ],
    }

    # Config summary (only when config is available)
    if config:
        result["config"] = {
            "total_vram_gb": config.total_vram_gb,
            "gpu_count": len(config.all_gpus),
            "models": list(config.models.keys()),
            "tensor_split": config.tensor_split(),
        }
    elif pidfile_data:
        result["config"] = {
            "model": pidfile_data.get("model"),
            "started": pidfile_data.get("started"),
        }

    return result


def start_and_reclaim(
    config: ClusterConfig,
    model_name: str | None = None,
    ram_reclaim: str | None = None,
    wait_timeout: float = 300.0,
    skip_version_check: bool = False,
) -> tuple[int, "ReclaimResult | None"]:
    """Start coordinator, wait for /health, then reclaim RAM based on mode.

    Parameters
    ----------
    config:
        Cluster configuration.
    model_name:
        Model to load (default from config if None).
    ram_reclaim:
        Override mode: "off", "on", "auto".  Falls back to ``config.ram_reclaim``.
    wait_timeout:
        Seconds to wait for /health to return 200.

    Returns
    -------
    (pid, ReclaimResult or None)
    """
    from .reclaim import reclaim_ram, should_reclaim, get_available_ram_bytes, get_swap_free_bytes

    mode = ram_reclaim or config.ram_reclaim

    if mode == "off":
        pid = start(config, model_name, skip_version_check=skip_version_check)
        return pid, None

    # Check if model needs streaming load (> 80% of available RAM)
    model = (
        config.models.get(model_name) if model_name else config.default_model()
    )
    model_path = model.path if model else None

    if mode in ("auto", "on") and model_path:
        try:
            model_file_size = Path(model_path).stat().st_size
        except OSError:
            model_file_size = 0

        if model_file_size > 0:
            from .loader import needs_streaming_load, load_model
            available = get_available_ram_bytes()
            swap_free = get_swap_free_bytes()
            if needs_streaming_load(model_file_size, available, swap_free_bytes=swap_free):
                # Model won't fit comfortably — pre-warm + start + reclaim
                result = load_model(
                    config, model_name, ram_reclaim=mode,
                    wait_timeout=wait_timeout,
                    skip_version_check=skip_version_check,
                )
                return result.pid, result.reclaim_result

    # Normal path: model fits in RAM, start then optionally reclaim
    pid = start(config, model_name, skip_version_check=skip_version_check)

    # Wait for /health to return 200 before reclaiming
    deadline = time.monotonic() + wait_timeout
    healthy = False
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            logger.error(
                "Coordinator (PID %d) exited during startup — see %s",
                pid, COORDINATOR_LOG,
            )
            return pid, None
        health = check_coordinator_health(
            _health_host(config.coordinator_host), config.coordinator_port
        )
        if health.get("alive"):
            healthy = True
            break
        time.sleep(2.0)

    if not healthy:
        logger.warning(
            "Coordinator did not become healthy within %.0fs — skipping reclaim",
            wait_timeout,
        )
        return pid, None

    if mode == "auto":
        model_file_size = 0
        if model_path:
            try:
                model_file_size = Path(model_path).stat().st_size
            except OSError:
                pass
        if model_file_size > 0 and not should_reclaim(model_file_size):
            logger.info("RAM reclaim: skipped (sufficient RAM for %.1f GB model)",
                        model_file_size / (1024**3))
            return pid, None

    result = reclaim_ram(pid, model_path)
    return pid, result


def _warn_moe_vram(config: ClusterConfig, model: ModelConfig) -> None:
    """Log warnings if an MoE model's shared overhead may exceed worker VRAM."""
    model_path = Path(model.path)
    if not model_path.exists():
        return

    try:
        from .gguf_inspect import inspect_model, check_moe_vram
    except ImportError:
        return

    try:
        model_info = inspect_model(str(model_path))
    except Exception:
        return

    if not model_info.is_moe:
        return

    gpu_vram = {gpu.name: gpu.vram_gb for gpu in config.all_gpus}
    warnings = check_moe_vram(model_info, gpu_vram=gpu_vram)
    for w in warnings:
        logger.warning("MoE VRAM: %s", w)


def swap_model(config: ClusterConfig, model_name: str) -> int:
    """Hot-swap the active model (stop coordinator, restart with new model).

    RPC workers persist — only the coordinator restarts.
    """
    stop()
    return start(config, model_name)
