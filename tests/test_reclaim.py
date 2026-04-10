"""Tests for RAM reclaim module."""

import os
import platform
from unittest.mock import patch, MagicMock

import pytest

from tightwad.reclaim import (
    ReclaimResult,
    get_process_rss_mb,
    get_available_ram_bytes,
    should_reclaim,
    detect_model_path_from_proc,
    reclaim_ram,
)


def test_reclaimresult_dataclass():
    """ReclaimResult fields are populated correctly."""
    r = ReclaimResult(
        platform="linux",
        pid=1234,
        rss_before_mb=18000.0,
        rss_after_mb=500.0,
        reclaimed_mb=17500.0,
        method="posix_fadvise",
    )
    assert r.platform == "linux"
    assert r.pid == 1234
    assert r.reclaimed_mb == 17500.0
    assert r.method == "posix_fadvise"
    assert r.error is None


def test_reclaimresult_with_error():
    r = ReclaimResult(
        platform="freebsd",
        pid=999,
        rss_before_mb=0,
        rss_after_mb=0,
        reclaimed_mb=0,
        method="skipped",
        error="Unsupported platform: freebsd",
    )
    assert r.error == "Unsupported platform: freebsd"


def test_get_rss_mb_current_process():
    """RSS for the current process should be a positive float."""
    rss = get_process_rss_mb(os.getpid())
    assert isinstance(rss, float)
    assert rss > 0


def test_get_rss_mb_nonexistent_process():
    """RSS for a non-existent PID should return 0."""
    rss = get_process_rss_mb(999999999)
    assert rss == 0.0


def test_get_available_ram():
    """Available RAM should return a positive value on any platform."""
    available = get_available_ram_bytes()
    assert isinstance(available, int)
    assert available > 0


def test_should_reclaim_large_model():
    """A model larger than 50% of available RAM should trigger reclaim."""
    available = get_available_ram_bytes()
    # Model that is 2x available RAM
    assert should_reclaim(available * 2) is True


def test_should_reclaim_small_model():
    """A tiny model should not trigger reclaim."""
    # 1 MB model — always fits comfortably
    assert should_reclaim(1024 * 1024) is False


def test_should_reclaim_zero_available():
    """If available RAM can't be determined (0), reclaim to be safe."""
    with patch("tightwad.reclaim.get_available_ram_bytes", return_value=0):
        assert should_reclaim(1024) is True


def test_detect_model_path_linux_mock():
    """Mock /proc/{pid}/maps with a .gguf entry."""
    maps_content = (
        "7f0000000000-7f0100000000 r--s 00000000 08:01 12345  "
        "/models/qwen3-32b-Q4_K_M.gguf\n"
        "7f0200000000-7f0300000000 r-xp 00000000 08:01 99999  "
        "/usr/lib/libc.so.6\n"
    )
    with patch("tightwad.reclaim._SYSTEM", "linux"), \
         patch("tightwad.reclaim.Path") as MockPath:
        MockPath.return_value.read_text.return_value = maps_content
        result = detect_model_path_from_proc(1234)
        assert result == "/models/qwen3-32b-Q4_K_M.gguf"


def test_detect_model_path_not_linux():
    """Non-Linux platforms return None."""
    with patch("tightwad.reclaim._SYSTEM", "darwin"):
        assert detect_model_path_from_proc(1234) is None


def test_reclaim_skips_on_darwin():
    """On macOS, reclaim returns method='skipped' (unified memory)."""
    with patch("tightwad.reclaim._SYSTEM", "darwin"), \
         patch("tightwad.reclaim.get_process_rss_mb", return_value=500.0):
        result = reclaim_ram(os.getpid())
        assert result.method == "skipped"
        assert result.platform == "darwin"
        assert result.reclaimed_mb == 0.0


def test_reclaim_unsupported_platform():
    """Unsupported platforms skip with an error message."""
    with patch("tightwad.reclaim._SYSTEM", "freebsd"), \
         patch("tightwad.reclaim.get_process_rss_mb", return_value=100.0):
        result = reclaim_ram(os.getpid())
        assert result.method == "skipped"
        assert result.error is not None
        assert "freebsd" in result.error


def test_reclaim_linux_no_model_path():
    """Linux reclaim without model path tries /proc auto-detect."""
    with patch("tightwad.reclaim._SYSTEM", "linux"), \
         patch("tightwad.reclaim.get_process_rss_mb", return_value=100.0), \
         patch("tightwad.reclaim.detect_model_path_from_proc", return_value=None):
        result = reclaim_ram(os.getpid())
        assert result.method == "skipped"


def test_reclaim_linux_fadvise_einval():
    """posix_fadvise returning EINVAL (22) should report method='failed' with human message."""
    from tightwad.reclaim import _reclaim_linux
    with patch("tightwad.reclaim._SYSTEM", "linux"), \
         patch("tightwad.reclaim.detect_model_path_from_proc", return_value="/models/test.gguf"):
        # Mock ctypes and os to simulate fadvise returning 22
        mock_libc = MagicMock()
        mock_libc.posix_fadvise.return_value = 22
        with patch("tightwad.reclaim.ctypes.CDLL", return_value=mock_libc), \
             patch("tightwad.reclaim.os.open", return_value=3), \
             patch("tightwad.reclaim.os.fstat") as mock_fstat, \
             patch("tightwad.reclaim.os.close"):
            mock_fstat.return_value = MagicMock(st_size=1024)
            method, error = _reclaim_linux(1234, None)
            assert method == "failed"
            assert "EINVAL" in error


def test_reclaim_linux_fadvise_success():
    """posix_fadvise returning 0 should report method='posix_fadvise' with no error."""
    from tightwad.reclaim import _reclaim_linux
    mock_libc = MagicMock()
    mock_libc.posix_fadvise.return_value = 0
    with patch("tightwad.reclaim.ctypes.CDLL", return_value=mock_libc), \
         patch("tightwad.reclaim.os.open", return_value=3), \
         patch("tightwad.reclaim.os.fstat") as mock_fstat, \
         patch("tightwad.reclaim.os.close"):
        mock_fstat.return_value = MagicMock(st_size=1024)
        method, error = _reclaim_linux(1234, "/models/test.gguf")
        assert method == "posix_fadvise"
        assert error is None
