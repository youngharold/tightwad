"""Tests for the GGUF loader module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tightwad.loader import (
    LoadResult,
    needs_streaming_load,
    prewarm_sequential,
    _get_swap_free_bytes,
)
from tightwad.reclaim import ReclaimResult


class TestNeedsStreamingLoad:
    def test_model_fits_comfortably(self):
        # 1 GB model, 16 GB available — 6.25%, well under 80%
        assert needs_streaming_load(1 * 1024**3, 16 * 1024**3) is False

    def test_model_too_large(self):
        # 14 GB model, 16 GB available — 87.5%, over 80%
        assert needs_streaming_load(14 * 1024**3, 16 * 1024**3) is True

    def test_model_at_boundary(self):
        # Exactly 80% — not strictly greater
        avail = 10 * 1024**3
        model = int(avail * 0.8)
        assert needs_streaming_load(model, avail) is False
        # One byte over
        assert needs_streaming_load(model + 1, avail) is True

    def test_zero_available_ram(self):
        # Unknown available RAM — should return True to be safe
        assert needs_streaming_load(1 * 1024**3, 0) is True

    def test_with_swap(self):
        # 14 GB model, 8 GB RAM, 8 GB swap = 16 GB total — 87.5%, over 80%
        assert needs_streaming_load(
            14 * 1024**3, 8 * 1024**3, swap_free_bytes=8 * 1024**3
        ) is True
        # 10 GB model, 8 GB RAM, 8 GB swap = 16 GB total — 62.5%, under 80%
        assert needs_streaming_load(
            10 * 1024**3, 8 * 1024**3, swap_free_bytes=8 * 1024**3
        ) is False


class TestPrewarmSequential:
    def test_reads_file(self):
        """Pre-warm reads a temporary file and returns elapsed > 0."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            data = b"\x00" * (1024 * 1024)  # 1 MB
            f.write(data)
            f.flush()
            path = Path(f.name)

        try:
            elapsed = prewarm_sequential(path, len(data), chunk_size=64 * 1024)
            assert elapsed >= 0
        finally:
            path.unlink()

    def test_progress_callback(self):
        """Progress callback is called during pre-warm."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            data = b"\x00" * (256 * 1024)  # 256 KB
            f.write(data)
            f.flush()
            path = Path(f.name)

        calls = []
        try:
            prewarm_sequential(
                path, len(data), chunk_size=64 * 1024,
                progress_callback=lambda done, total: calls.append((done, total)),
            )
            assert len(calls) > 0
            # Last call should have bytes_read close to file size
            assert calls[-1][0] >= len(data) - 64 * 1024
        finally:
            path.unlink()

    def test_empty_file(self):
        """Pre-warming an empty file returns quickly."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            path = Path(f.name)

        try:
            elapsed = prewarm_sequential(path, 0)
            assert elapsed >= 0
        finally:
            path.unlink()


class TestLoadResultDataclass:
    def test_fields(self):
        result = LoadResult(
            pid=12345,
            model_size_gb=18.1,
            peak_rss_mb=2100.0,
            load_time_seconds=43.2,
            prewarm_time_seconds=12.1,
            prewarm_throughput_gbs=1.5,
            healthy=True,
            reclaim_result=None,
        )
        assert result.pid == 12345
        assert result.model_size_gb == 18.1
        assert result.healthy is True
        assert result.reclaim_result is None

    def test_with_reclaim(self):
        reclaim = ReclaimResult(
            platform="linux",
            pid=12345,
            rss_before_mb=18000.0,
            rss_after_mb=108.0,
            reclaimed_mb=17892.0,
            method="posix_fadvise",
        )
        result = LoadResult(
            pid=12345,
            model_size_gb=18.1,
            peak_rss_mb=2100.0,
            load_time_seconds=43.2,
            prewarm_time_seconds=12.1,
            prewarm_throughput_gbs=1.5,
            healthy=True,
            reclaim_result=reclaim,
        )
        assert result.reclaim_result.reclaimed_mb == 17892.0
        assert result.reclaim_result.method == "posix_fadvise"


class TestSelectStrategy:
    @patch("tightwad.loader.get_available_ram_bytes", return_value=32 * 1024**3)
    def test_noop_when_fits(self, mock_avail):
        """When model fits, needs_streaming_load returns False."""
        assert needs_streaming_load(1 * 1024**3, 32 * 1024**3) is False

    @patch("tightwad.loader._SYSTEM", "darwin")
    def test_darwin_prewarm_still_works(self):
        """macOS still benefits from pre-warm (unified memory cache warming)."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"\x00" * 1024)
            f.flush()
            path = Path(f.name)
        try:
            elapsed = prewarm_sequential(path, 1024)
            assert elapsed >= 0
        finally:
            path.unlink()


class TestGetSwapFreeBytes:
    @patch("tightwad.loader._SYSTEM", "darwin")
    def test_darwin_returns_zero(self):
        assert _get_swap_free_bytes() == 0

    @patch("tightwad.loader._SYSTEM", "freebsd")
    def test_unsupported_returns_zero(self):
        assert _get_swap_free_bytes() == 0
