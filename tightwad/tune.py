"""System diagnostics and tuning recommendations for large model loading.

When the model is BIGGER than available RAM, NVMe swap must be configured so
the OS can page during mmap loading.  After loading + RAM reclaim, swap usage
drops too.  This module diagnoses system readiness and suggests fixes.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("tightwad.tune")

_SYSTEM = platform.system().lower()


@dataclass
class SystemInfo:
    """Snapshot of system RAM, swap, and storage configuration."""

    platform: str
    total_ram_gb: float
    available_ram_gb: float
    swap_total_gb: float
    swap_used_gb: float
    swap_on_nvme: bool | None = None  # None = unknown
    vm_swappiness: int | None = None  # Linux only


@dataclass
class Recommendation:
    """A single tuning recommendation."""

    severity: str  # "info", "warn", "critical"
    message: str
    commands: list[str] = field(default_factory=list)


def _get_total_ram_bytes() -> int:
    """Get total physical RAM in bytes."""
    if _SYSTEM == "linux":
        try:
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (FileNotFoundError, ValueError):
            pass
        return 0

    if _SYSTEM == "windows":
        try:
            import ctypes

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

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(mem)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return mem.ullTotalPhys
        except Exception:
            pass
        return 0

    if _SYSTEM == "darwin":
        try:
            out = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return int(out.strip())
        except Exception:
            pass
        return 0

    return 0


def _get_swap_info() -> tuple[float, float]:
    """Return (swap_total_gb, swap_used_gb)."""
    if _SYSTEM == "linux":
        try:
            meminfo = Path("/proc/meminfo").read_text()
            total = used = 0
            for line in meminfo.splitlines():
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) * 1024
                    used = total - free
            return total / (1024**3), used / (1024**3)
        except (FileNotFoundError, ValueError):
            pass
        return 0.0, 0.0

    if _SYSTEM == "windows":
        try:
            out = subprocess.check_output(
                ["wmic", "pagefile", "list", "/format:csv"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 4 and parts[-2].isdigit():
                    # AllocatedBaseSize is in MB
                    total_mb = int(parts[-2])
                    used_mb = int(parts[-1]) if parts[-1].isdigit() else 0
                    return total_mb / 1024, used_mb / 1024
        except Exception:
            pass
        return 0.0, 0.0

    if _SYSTEM == "darwin":
        try:
            out = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            # "total = 2048.00M  used = 1024.00M  free = 1024.00M"
            parts = out.strip().split()
            total = used = 0.0
            for i, p in enumerate(parts):
                if p == "total":
                    val = parts[i + 2].rstrip("M")
                    total = float(val) / 1024
                elif p == "used":
                    val = parts[i + 2].rstrip("M")
                    used = float(val) / 1024
            return total, used
        except Exception:
            pass
        return 0.0, 0.0

    return 0.0, 0.0


def _get_swappiness() -> int | None:
    """Get Linux vm.swappiness value."""
    if _SYSTEM != "linux":
        return None
    try:
        val = Path("/proc/sys/vm/swappiness").read_text().strip()
        return int(val)
    except (FileNotFoundError, ValueError):
        return None


def _detect_swap_on_nvme() -> bool | None:
    """Best-effort check if swap is on NVMe storage."""
    if _SYSTEM == "darwin":
        return True  # Apple Silicon always uses NVMe

    if _SYSTEM == "linux":
        try:
            swaps = Path("/proc/swaps").read_text()
            for line in swaps.splitlines()[1:]:  # skip header
                parts = line.split()
                if not parts:
                    continue
                swap_path = parts[0]
                # Check if device is NVMe
                try:
                    import os

                    real_path = os.path.realpath(swap_path)
                    # /dev/nvme0n1p2 or swapfile on nvme
                    if "nvme" in real_path:
                        return True
                    # Check mount point for swap files
                    out = subprocess.check_output(
                        ["df", swap_path],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    if "nvme" in out:
                        return True
                except Exception:
                    pass
            return None  # couldn't determine
        except (FileNotFoundError, PermissionError):
            pass
        return None

    return None


def diagnose() -> SystemInfo:
    """Gather system RAM, swap, and storage info."""
    from .reclaim import get_available_ram_bytes

    total_ram = _get_total_ram_bytes()
    available_ram = get_available_ram_bytes()
    swap_total, swap_used = _get_swap_info()
    swappiness = _get_swappiness()
    swap_nvme = _detect_swap_on_nvme()

    return SystemInfo(
        platform=_SYSTEM,
        total_ram_gb=round(total_ram / (1024**3), 1),
        available_ram_gb=round(available_ram / (1024**3), 1),
        swap_total_gb=round(swap_total, 1),
        swap_used_gb=round(swap_used, 1),
        swap_on_nvme=swap_nvme,
        vm_swappiness=swappiness,
    )


def recommend(
    info: SystemInfo, model_size_gb: float | None = None
) -> list[Recommendation]:
    """Generate tuning recommendations based on system state and model size."""
    recs: list[Recommendation] = []

    if model_size_gb is None:
        # No model specified — just report system state
        if info.swap_total_gb < 1.0 and info.platform == "linux":
            recs.append(
                Recommendation(
                    severity="info",
                    message="No swap configured. Large models may fail to load.",
                    commands=[
                        "sudo fallocate -l 32G /swapfile",
                        "sudo chmod 600 /swapfile",
                        "sudo mkswap /swapfile",
                        "sudo swapon /swapfile",
                        "echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab",
                    ],
                )
            )
        recs.append(
            Recommendation(
                severity="info",
                message="System is ready. Specify --model to check against a specific model.",
            )
        )
        return recs

    available = info.available_ram_gb
    swap_free = info.swap_total_gb - info.swap_used_gb

    # Critical: model > RAM and no swap
    if model_size_gb > available and info.swap_total_gb < 1.0:
        if info.platform == "linux":
            swap_size = max(32, int(model_size_gb * 1.5))
            recs.append(
                Recommendation(
                    severity="critical",
                    message=(
                        f"No swap configured. This model ({model_size_gb:.1f} GB) "
                        f"exceeds available RAM ({available:.1f} GB). "
                        f"Loading will fail. Configure NVMe swap:"
                    ),
                    commands=[
                        f"sudo fallocate -l {swap_size}G /swapfile",
                        "sudo chmod 600 /swapfile",
                        "sudo mkswap /swapfile",
                        "sudo swapon /swapfile",
                        "echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab",
                    ],
                )
            )
        elif info.platform == "windows":
            recs.append(
                Recommendation(
                    severity="critical",
                    message=(
                        f"Insufficient pagefile. This model ({model_size_gb:.1f} GB) "
                        f"exceeds available RAM ({available:.1f} GB). "
                        "Increase the Windows pagefile on your NVMe drive."
                    ),
                    commands=[
                        "System Properties > Advanced > Performance Settings > "
                        "Advanced > Virtual Memory > Change",
                        "Set pagefile on NVMe drive to System Managed or "
                        f"custom min {int(model_size_gb * 1.5 * 1024)} MB",
                    ],
                )
            )
        elif info.platform == "darwin":
            recs.append(
                Recommendation(
                    severity="info",
                    message="Swap is managed automatically on macOS. Your system is ready.",
                )
            )
        return recs

    # Warn: model > RAM but swap might not be enough
    if model_size_gb > available:
        needed_swap = model_size_gb - available + 2.0  # 2 GB headroom
        if swap_free < needed_swap:
            recs.append(
                Recommendation(
                    severity="warn",
                    message=(
                        f"Insufficient swap for model loading. Model ({model_size_gb:.1f} GB) "
                        f"exceeds RAM ({available:.1f} GB) by "
                        f"{model_size_gb - available:.1f} GB, but only "
                        f"{swap_free:.1f} GB swap free."
                    ),
                )
            )

    # Linux swappiness check
    if info.vm_swappiness is not None and info.vm_swappiness < 60:
        recs.append(
            Recommendation(
                severity="info",
                message=(
                    f"vm.swappiness is {info.vm_swappiness}. "
                    "Increase during loading for better page-out behavior."
                ),
                commands=["sudo sysctl vm.swappiness=100"],
            )
        )

    # Swap not on NVMe
    if info.swap_on_nvme is False:
        recs.append(
            Recommendation(
                severity="warn",
                message="Swap does not appear to be on NVMe. Loading will be very slow.",
            )
        )

    # Everything is sufficient
    if not recs:
        recs.append(
            Recommendation(
                severity="info",
                message=f"System is ready for this model ({model_size_gb:.1f} GB).",
            )
        )

    # Always suggest reclaim if model is large relative to RAM
    if model_size_gb > available * 0.5:
        recs.append(
            Recommendation(
                severity="info",
                message="Tip: After loading, run 'tightwad reclaim' to free RAM.",
            )
        )

    return recs
