"""RAM reclaim for llama-server processes after model loading.

llama-server mmaps the entire GGUF file into RAM before copying tensors to
VRAM.  On Windows ``unmap_fragment()`` is a no-op so pages stay resident
forever.  On Linux pages linger in the page cache.  This module tells the OS
to release those pages once the model is fully loaded to GPU.

Cross-platform: Linux (posix_fadvise), Windows (SetProcessWorkingSetSize),
macOS (no-op — unified memory).
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("tightwad.reclaim")

_SYSTEM = platform.system().lower()


@dataclass
class ReclaimResult:
    """Result of a RAM reclaim operation."""

    platform: str  # "linux", "windows", "darwin"
    pid: int
    rss_before_mb: float
    rss_after_mb: float
    reclaimed_mb: float
    method: str  # "posix_fadvise", "SetProcessWorkingSetSize", "skipped"
    error: str | None = None


def get_process_rss_mb(pid: int) -> float:
    """Get resident set size in MB without psutil dependency."""
    if _SYSTEM == "linux":
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    # VmRSS: 123456 kB
                    kb = int(line.split()[1])
                    return kb / 1024.0
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        return 0.0

    if _SYSTEM == "windows":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
            )
            if not handle:
                return 0.0
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
                if psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return counters.WorkingSetSize / (1024 * 1024)
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            pass
        return 0.0

    # macOS / other: use ps
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        kb = int(out.strip())
        return kb / 1024.0
    except Exception:
        return 0.0


def get_available_ram_bytes() -> int:
    """Get available (not total) RAM in bytes. Cross-platform."""
    if _SYSTEM == "linux":
        try:
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.splitlines():
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
        except (FileNotFoundError, ValueError):
            pass
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
                return mem.ullAvailPhys
        except Exception:
            pass
        return 0

    # macOS: use sysctl for total, vm_stat for free pages
    if _SYSTEM == "darwin":
        try:
            out = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            total = int(out.strip())
            # Use vm_stat to estimate available
            vm_out = subprocess.check_output(
                ["vm_stat"], text=True, stderr=subprocess.DEVNULL
            )
            page_size = 16384  # Apple Silicon default
            free_pages = 0
            for line in vm_out.splitlines():
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                elif "Pages free:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))
                elif "Pages speculative:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))
                elif "Pages purgeable:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))
            return free_pages * page_size
        except Exception:
            pass
        return 0

    return 0


def get_swap_free_bytes() -> int:
    """Get free swap space in bytes. Cross-platform."""
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
                # PageFile includes both RAM and swap; subtract physical to get swap-only
                return max(0, mem.ullAvailPageFile - mem.ullAvailPhys)
        except Exception:
            pass
        return 0

    if _SYSTEM == "darwin":
        try:
            import re
            out = subprocess.check_output(
                ["sysctl", "-n", "vm.swapusage"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            # Format: "total = 2048.00M  used = 512.00M  free = 1536.00M  ..."
            m = re.search(r"free\s*=\s*([\d.]+)M", out)
            if m:
                return int(float(m.group(1)) * 1024 * 1024)
        except Exception:
            pass
        return 0

    return 0


def should_reclaim(model_size_bytes: int) -> bool:
    """Auto-detect: returns True if model > 50% of available RAM."""
    available = get_available_ram_bytes()
    if available <= 0:
        # Can't determine available RAM — reclaim to be safe
        return True
    return model_size_bytes > (available * 0.5)


def detect_model_path_from_proc(pid: int) -> str | None:
    """Read /proc/{pid}/maps to find mmap'd .gguf files. Linux only."""
    if _SYSTEM != "linux":
        return None
    try:
        maps = Path(f"/proc/{pid}/maps").read_text()
        for line in maps.splitlines():
            parts = line.split()
            if len(parts) >= 6:
                path = parts[-1]
                if path.endswith(".gguf"):
                    return path
    except (FileNotFoundError, PermissionError):
        pass
    return None


_FADVISE_ERRORS: dict[int, str] = {
    9: "Bad file descriptor (EBADF)",
    22: "Invalid argument (EINVAL) — file may not support fadvise",
    29: "Illegal seek (ESPIPE) — not a regular file",
}


def _reclaim_linux(pid: int, model_path: str | None) -> tuple[str, str | None]:
    """Linux: use posix_fadvise(DONTNEED) on the GGUF file.

    Returns (method, error_message | None) where method is one of:
    - "posix_fadvise" — success
    - "failed" — fadvise returned a non-zero errno
    - "skipped" — could not attempt (no model path)
    """
    if model_path is None:
        model_path = detect_model_path_from_proc(pid)
    if model_path is None:
        return "skipped", "no model path detected"

    POSIX_FADV_DONTNEED = 4
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        fd = os.open(model_path, os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size
            ret = libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED)
            if ret != 0:
                human_msg = _FADVISE_ERRORS.get(ret, f"unknown error (errno {ret})")
                logger.debug("posix_fadvise returned %d: %s", ret, human_msg)
                return "failed", human_msg
        finally:
            os.close(fd)
        return "posix_fadvise", None
    except Exception as e:
        logger.debug("posix_fadvise exception: %s", e)
        return "failed", str(e)


def _reclaim_windows(pid: int) -> str:
    """Windows: trim working set via SetProcessWorkingSetSize(-1, -1)."""
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_QUERY_INFORMATION = 0x0400
        handle = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, pid
        )
        if not handle:
            return "skipped"
        try:
            # -1, -1 tells Windows to trim the working set
            result = kernel32.SetProcessWorkingSetSize(
                handle,
                ctypes.c_size_t(-1 & 0xFFFFFFFFFFFFFFFF),
                ctypes.c_size_t(-1 & 0xFFFFFFFFFFFFFFFF),
            )
            if not result:
                return "skipped"
        finally:
            kernel32.CloseHandle(handle)
        return "SetProcessWorkingSetSize"
    except Exception as e:
        logger.warning("SetProcessWorkingSetSize failed: %s", e)
        return "skipped"


def reclaim_ram(pid: int, model_path: str | None = None) -> ReclaimResult:
    """Reclaim RAM from a llama-server process after model loading.

    Parameters
    ----------
    pid:
        PID of the llama-server process.
    model_path:
        Path to the GGUF file.  On Linux, auto-detected from ``/proc/{pid}/maps``
        if not provided.

    Returns
    -------
    ReclaimResult with before/after RSS and method used.
    """
    rss_before = get_process_rss_mb(pid)

    if _SYSTEM == "darwin":
        logger.info(
            "macOS uses unified memory — GPU and CPU share physical RAM. "
            "Reclaim is unnecessary."
        )
        return ReclaimResult(
            platform=_SYSTEM,
            pid=pid,
            rss_before_mb=rss_before,
            rss_after_mb=rss_before,
            reclaimed_mb=0.0,
            method="skipped",
        )

    error = None
    if _SYSTEM == "linux":
        method, error = _reclaim_linux(pid, model_path)
    elif _SYSTEM == "windows":
        method = _reclaim_windows(pid)
    else:
        method = "skipped"
        error = f"Unsupported platform: {_SYSTEM}"

    rss_after = get_process_rss_mb(pid)
    reclaimed = max(0.0, rss_before - rss_after)

    return ReclaimResult(
        platform=_SYSTEM,
        pid=pid,
        rss_before_mb=round(rss_before, 1),
        rss_after_mb=round(rss_after, 1),
        reclaimed_mb=round(reclaimed, 1),
        method=method,
        error=error,
    )
