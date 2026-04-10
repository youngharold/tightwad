"""GGUF pre-warming and smart loading lifecycle.

v0.1.4 focuses on advisory approach: sequential pre-warm with
``posix_fadvise(SEQUENTIAL)`` hints, then reclaim after model loads to VRAM.

Hard memory constraining (systemd-run, cgroups, Job Objects) deferred to v0.1.5
behind ``--force-constrain`` flag.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

from .reclaim import ReclaimResult, get_available_ram_bytes, reclaim_ram

logger = logging.getLogger("tightwad.loader")

_SYSTEM = platform.system().lower()

# Environment kill-switch
_PREWARM_DISABLED = os.environ.get("TIGHTWAD_DISABLE_PREWARM", "").strip() == "1"


@dataclass
class LoadResult:
    """Result of a full load lifecycle."""

    pid: int
    model_size_gb: float
    peak_rss_mb: float
    load_time_seconds: float
    prewarm_time_seconds: float
    prewarm_throughput_gbs: float   # GB/s during pre-warm
    healthy: bool
    reclaim_result: ReclaimResult | None


def needs_streaming_load(
    model_size_bytes: int,
    available_ram_bytes: int,
    swap_free_bytes: int = 0,
) -> bool:
    """Returns True if model > 80% of (available RAM + swap free).

    When True, pre-warming + reclaim are applied automatically.
    """
    total_available = available_ram_bytes + swap_free_bytes
    if total_available <= 0:
        return True
    return model_size_bytes > (total_available * 0.8)


def _get_swap_free_bytes() -> int:
    """Get free swap in bytes. Best-effort, returns 0 on failure."""
    if _SYSTEM == "linux":
        try:
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.splitlines():
                if line.startswith("SwapFree:"):
                    kb = int(line.split()[1])
                    return kb * 1024
        except (FileNotFoundError, ValueError):
            pass
        return 0
    if _SYSTEM == "darwin":
        # macOS unified memory — swap is managed transparently
        return 0
    if _SYSTEM == "windows":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(mem)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return mem.ullAvailPageFile
        except Exception:
            pass
        return 0
    return 0


def prewarm_sequential(
    model_path: str | Path,
    file_size: int,
    chunk_size: int = 4 * 1024 * 1024,
    progress_callback=None,
) -> float:
    """Read GGUF file sequentially in chunks to warm the page cache.

    On Linux, calls ``posix_fadvise(SEQUENTIAL)`` first so the kernel does
    aggressive readahead and (optionally) drops pages behind the cursor.

    Parameters
    ----------
    model_path:
        Path to the GGUF file.
    file_size:
        Total file size in bytes.
    chunk_size:
        Read chunk size (default 4 MB).
    progress_callback:
        Called as ``callback(bytes_read, file_size)`` after each chunk.

    Returns
    -------
    Elapsed time in seconds.
    """
    model_path = str(model_path)
    t0 = time.monotonic()

    flags = os.O_RDONLY
    if _SYSTEM == "windows":
        flags |= os.O_BINARY  # prevent text-mode truncation at 0x1A
    fd = os.open(model_path, flags)
    try:
        # Advisory hint on Linux: sequential access pattern
        if _SYSTEM == "linux":
            try:
                POSIX_FADV_SEQUENTIAL = 2
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL)
            except Exception:
                pass  # best-effort

        bytes_read = 0
        buf = bytearray(chunk_size)
        while bytes_read < file_size:
            n = os.readv(fd, [buf]) if hasattr(os, "readv") else len(os.read(fd, chunk_size))
            if n == 0:
                break
            bytes_read += n
            if progress_callback:
                progress_callback(bytes_read, file_size)
    finally:
        os.close(fd)

    elapsed = time.monotonic() - t0
    return elapsed


def load_model(
    config,
    model_name: str | None = None,
    prewarm: bool = True,
    ram_reclaim: str | None = None,
    wait_timeout: float = 300.0,
    progress_callback=None,
) -> LoadResult:
    """Full load lifecycle: parse GGUF, pre-warm, start coordinator, reclaim.

    Parameters
    ----------
    config:
        ClusterConfig instance.
    model_name:
        Model to load (default from config if None).
    prewarm:
        Whether to pre-warm the page cache before starting.
    ram_reclaim:
        Override mode: "off", "on", "auto".
    wait_timeout:
        Seconds to wait for /health to return 200.
    progress_callback:
        Called as ``callback(bytes_read, total_bytes)`` during pre-warm.

    Returns
    -------
    LoadResult with timing, RSS, and reclaim info.
    """
    from . import coordinator
    from .reclaim import get_process_rss_mb
    from .worker import check_coordinator_health

    # Resolve model
    model_cfg = (
        config.models.get(model_name) if model_name else config.default_model()
    )
    if not model_cfg:
        raise ValueError("No model specified and no default configured")

    model_path = Path(model_cfg.path)
    file_size = model_path.stat().st_size
    model_size_gb = file_size / (1024**3)

    # Try to parse GGUF header for metadata (best-effort)
    gguf_info = None
    try:
        from .gguf_reader import read_header, model_summary
        header = read_header(model_path)
        gguf_info = model_summary(header)
    except Exception as e:
        logger.debug("GGUF parse failed (using file size only): %s", e)

    # Pre-warm if needed and not disabled
    prewarm_time = 0.0
    prewarm_throughput = 0.0
    should_prewarm = prewarm and not _PREWARM_DISABLED

    if should_prewarm:
        available = get_available_ram_bytes()
        swap_free = _get_swap_free_bytes()
        if needs_streaming_load(file_size, available, swap_free):
            logger.info(
                "Pre-warming %s (%.1f GB, available RAM: %.1f GB)",
                model_path.name, model_size_gb, available / (1024**3),
            )
            prewarm_time = prewarm_sequential(
                model_path, file_size,
                progress_callback=progress_callback,
            )
            if prewarm_time > 0:
                prewarm_throughput = model_size_gb / prewarm_time
            logger.info(
                "Pre-warm complete: %.1fs (%.2f GB/s)",
                prewarm_time, prewarm_throughput,
            )
        else:
            logger.info(
                "Model fits comfortably in RAM (%.1f GB model, "
                "%.1f GB available) — skipping pre-warm",
                model_size_gb, available / (1024**3),
            )

    # Start coordinator
    t_start = time.monotonic()
    pid = coordinator.start(config, model_name)

    # Wait for health
    deadline = time.monotonic() + wait_timeout
    healthy = False
    while time.monotonic() < deadline:
        health = check_coordinator_health("127.0.0.1", config.coordinator_port)
        if health.get("alive"):
            healthy = True
            break
        time.sleep(2.0)

    load_time = time.monotonic() - t_start
    peak_rss = get_process_rss_mb(pid)

    # Reclaim RAM
    mode = ram_reclaim or config.ram_reclaim
    reclaim_result = None
    if healthy and mode != "off":
        reclaim_result = reclaim_ram(pid, str(model_path))

    if not healthy:
        logger.warning(
            "Coordinator did not become healthy within %.0fs", wait_timeout
        )

    return LoadResult(
        pid=pid,
        model_size_gb=round(model_size_gb, 2),
        peak_rss_mb=round(peak_rss, 1),
        load_time_seconds=round(load_time, 1),
        prewarm_time_seconds=round(prewarm_time, 1),
        prewarm_throughput_gbs=round(prewarm_throughput, 2),
        healthy=healthy,
        reclaim_result=reclaim_result,
    )
