"""Tests for GPU auto-detection module."""

from unittest.mock import patch, MagicMock

import pytest

from tightwad.gpu_detect import (
    DetectedGPU,
    _detect_nvidia,
    _detect_metal,
    detect_gpus,
    detect_binary,
)


def test_detected_gpu_dataclass():
    gpu = DetectedGPU(name="RTX 4070", vram_mb=16384, backend="cuda", index=0)
    assert gpu.name == "RTX 4070"
    assert gpu.vram_mb == 16384
    assert gpu.backend == "cuda"


def test_detect_nvidia_parses_csv():
    csv_output = (
        "0, NVIDIA GeForce RTX 4070 Ti Super, 16384\n"
        "1, NVIDIA GeForce RTX 3060, 12288"
    )
    with patch("tightwad.gpu_detect._run", return_value=csv_output):
        gpus = _detect_nvidia()
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA GeForce RTX 4070 Ti Super"
    assert gpus[0].vram_mb == 16384
    assert gpus[0].backend == "cuda"
    assert gpus[0].index == 0
    assert gpus[1].name == "NVIDIA GeForce RTX 3060"
    assert gpus[1].vram_mb == 12288
    assert gpus[1].index == 1


def test_detect_nvidia_no_smi():
    with patch("tightwad.gpu_detect._run", return_value=None):
        assert _detect_nvidia() == []


def test_detect_metal_on_darwin():
    profiler_output = (
        "Graphics/Displays:\n"
        "    Apple M4:\n"
        "      Chipset Model: Apple M4\n"
        "      Type: GPU\n"
        "      Bus: Built-In\n"
    )
    with patch("tightwad.gpu_detect.platform.system", return_value="Darwin"), \
         patch("tightwad.gpu_detect._run") as mock_run:
        def side_effect(cmd, timeout=10):
            if "SPDisplaysDataType" in cmd:
                return profiler_output
            if "hw.memsize" in cmd:
                return "17179869184"  # 16 GB
            return None
        mock_run.side_effect = side_effect
        gpus = _detect_metal()

    assert len(gpus) == 1
    assert gpus[0].name == "Apple M4"
    assert gpus[0].vram_mb == 16384
    assert gpus[0].backend == "metal"


def test_detect_metal_not_darwin():
    with patch("tightwad.gpu_detect.platform.system", return_value="Linux"):
        assert _detect_metal() == []


def test_detect_gpus_combines_results():
    """detect_gpus returns GPUs from all backends."""
    nvidia_gpus = [DetectedGPU(name="RTX 4070", vram_mb=16384, backend="cuda")]
    with patch("tightwad.gpu_detect._detect_nvidia", return_value=nvidia_gpus), \
         patch("tightwad.gpu_detect._detect_rocm", return_value=[]), \
         patch("tightwad.gpu_detect._detect_metal", return_value=[]):
        gpus = detect_gpus()
    assert len(gpus) == 1
    assert gpus[0].backend == "cuda"


def test_detect_binary_in_path():
    with patch("tightwad.gpu_detect.shutil.which", return_value="/usr/local/bin/llama-server"):
        assert detect_binary() == "/usr/local/bin/llama-server"


def test_detect_binary_not_found():
    with patch("tightwad.gpu_detect.shutil.which", return_value=None), \
         patch("tightwad.gpu_detect.os.path.isfile", return_value=False):
        assert detect_binary() is None
