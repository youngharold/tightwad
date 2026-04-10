"""GPU auto-detection for NVIDIA (CUDA), AMD (ROCm), and Apple (Metal)."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("tightwad.gpu_detect")

SUBPROCESS_TIMEOUT = 10  # seconds


@dataclass
class DetectedGPU:
    """A GPU discovered on the local machine."""
    name: str
    vram_mb: int
    backend: str  # "cuda", "hip", "metal"
    index: int = 0


def _run(cmd: list[str], timeout: int = SUBPROCESS_TIMEOUT) -> str | None:
    """Run a command, returning stdout or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Command %s failed: %s", cmd, e)
    return None


def _detect_nvidia() -> list[DetectedGPU]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                idx = int(parts[0])
                name = parts[1]
                vram_mb = int(float(parts[2]))
                gpus.append(DetectedGPU(
                    name=name, vram_mb=vram_mb, backend="cuda", index=idx,
                ))
            except (ValueError, IndexError):
                continue
    return gpus


def _detect_rocm() -> list[DetectedGPU]:
    """Detect AMD GPUs via rocm-smi."""
    # Get product names
    name_out = _run(["rocm-smi", "--showproductname"])
    if not name_out:
        return []

    # Parse GPU names — format varies, look for "GPU[N]" lines
    gpu_names: dict[int, str] = {}
    for line in name_out.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        # Try to find GPU index and card model
        parts = line.split()
        for i, part in enumerate(parts):
            if part.startswith("GPU[") and part.endswith("]"):
                try:
                    idx = int(part[4:-1])
                except ValueError:
                    continue
                # Rest of line after ":" is the name
                if ":" in line:
                    gpu_names[idx] = line.split(":", 1)[1].strip()
                break

    # Get VRAM info
    vram_out = _run(["rocm-smi", "--showmeminfo", "vram"])
    gpu_vram: dict[int, int] = {}
    if vram_out:
        for line in vram_out.splitlines():
            line = line.strip()
            if "Total" in line and "GPU[" in line:
                for part in line.split():
                    if part.startswith("GPU[") and part.endswith("]"):
                        try:
                            idx = int(part[4:-1])
                        except ValueError:
                            continue
                        # Find the number after "Total"
                        try:
                            total_idx = line.index("Total")
                            remainder = line[total_idx + 5:].strip().lstrip(":").strip()
                            val = int(remainder.split()[0])
                            # rocm-smi reports in bytes typically
                            gpu_vram[idx] = val // (1024 * 1024)
                        except (ValueError, IndexError):
                            pass

    gpus = []
    for idx in sorted(set(gpu_names) | set(gpu_vram)):
        gpus.append(DetectedGPU(
            name=gpu_names.get(idx, f"AMD GPU {idx}"),
            vram_mb=gpu_vram.get(idx, 0),
            backend="hip",
            index=idx,
        ))
    return gpus


def _detect_metal() -> list[DetectedGPU]:
    """Detect Apple Silicon GPU via system_profiler."""
    if platform.system() != "Darwin":
        return []

    out = _run(["system_profiler", "SPDisplaysDataType"])
    if not out:
        return []

    # Parse chipset name and check for Apple Silicon
    chipset = None
    for line in out.splitlines():
        stripped = line.strip()
        if "Chipset Model:" in stripped:
            chipset = stripped.split(":", 1)[1].strip()

    if not chipset:
        return []

    # Estimate unified memory — Metal shares system RAM
    mem_out = _run(["/usr/sbin/sysctl", "-n", "hw.memsize"])
    vram_mb = 0
    if mem_out:
        try:
            vram_mb = int(mem_out) // (1024 * 1024)
        except ValueError:
            pass

    return [DetectedGPU(
        name=chipset,
        vram_mb=vram_mb,
        backend="metal",
        index=0,
    )]


def detect_gpus() -> list[DetectedGPU]:
    """Auto-detect all GPUs on the system.

    Tries NVIDIA first, then ROCm, then Metal. Returns all found GPUs.
    """
    gpus: list[DetectedGPU] = []

    nvidia = _detect_nvidia()
    if nvidia:
        gpus.extend(nvidia)
        logger.info("Detected %d NVIDIA GPU(s)", len(nvidia))

    rocm = _detect_rocm()
    if rocm:
        gpus.extend(rocm)
        logger.info("Detected %d AMD GPU(s)", len(rocm))

    metal = _detect_metal()
    if metal:
        gpus.extend(metal)
        logger.info("Detected %d Metal GPU(s)", len(metal))

    return gpus


# Common locations to search for llama-server
_COMMON_BINARY_PATHS = [
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
    os.path.expanduser("~/llama.cpp/build/bin/llama-server.exe"),
    "/usr/local/bin/llama-server",
    "/usr/bin/llama-server",
]


def detect_binary() -> str | None:
    """Find llama-server binary on the system.

    Searches PATH first, then common build locations.
    """
    # Check PATH
    found = shutil.which("llama-server")
    if found:
        return found

    # Check common locations
    for path in _COMMON_BINARY_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None
