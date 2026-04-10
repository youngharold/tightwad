"""Tests for version detection and enforcement between coordinator and workers."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from tightwad.worker import (
    VersionCheckResult,
    VersionInfo,
    check_version_match,
    get_local_version,
    get_remote_version,
)


# ---------------------------------------------------------------------------
# get_local_version
# ---------------------------------------------------------------------------


class TestGetLocalVersion:
    def test_parses_version_output(self):
        fake = MagicMock()
        fake.stdout = "llama-server version b8112\n"
        fake.stderr = ""
        with patch("tightwad.worker.subprocess.run", return_value=fake):
            result = get_local_version("llama-server")
        assert result.version == "llama-server version b8112"
        assert result.host == "localhost"

    def test_returns_none_on_not_found(self):
        with patch("tightwad.worker.subprocess.run", side_effect=FileNotFoundError):
            result = get_local_version("nonexistent-binary")
        assert result.version is None
        assert result.error is not None

    def test_returns_none_on_timeout(self):
        with patch("tightwad.worker.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("cmd", 10)):
            result = get_local_version("llama-server")
        assert result.version is None

    def test_returns_none_on_empty_output(self):
        fake = MagicMock()
        fake.stdout = ""
        fake.stderr = ""
        with patch("tightwad.worker.subprocess.run", return_value=fake):
            result = get_local_version("llama-server")
        assert result.version is None

    def test_multiline_takes_first(self):
        fake = MagicMock()
        fake.stdout = "  \nllama-server b8112\nextra stuff\n"
        fake.stderr = ""
        with patch("tightwad.worker.subprocess.run", return_value=fake):
            result = get_local_version("llama-server")
        assert result.version == "llama-server b8112"


# ---------------------------------------------------------------------------
# get_remote_version
# ---------------------------------------------------------------------------


class TestGetRemoteVersion:
    def test_parses_ssh_output(self):
        fake = MagicMock()
        fake.stdout = "llama-server version b8112\n"
        with patch("tightwad.worker.subprocess.run", return_value=fake):
            result = get_remote_version("user", "192.168.1.100")
        assert result.version == "llama-server version b8112"
        assert result.host == "192.168.1.100"

    def test_returns_none_on_ssh_failure(self):
        with patch("tightwad.worker.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("ssh", 15)):
            result = get_remote_version("user", "unreachable")
        assert result.version is None
        assert result.host == "unreachable"

    def test_returns_none_on_unknown(self):
        fake = MagicMock()
        fake.stdout = "unknown"
        with patch("tightwad.worker.subprocess.run", return_value=fake):
            result = get_remote_version("user", "host")
        assert result.version is None


# ---------------------------------------------------------------------------
# check_version_match
# ---------------------------------------------------------------------------


def _make_config(workers=None):
    """Build a minimal ClusterConfig for version check tests."""
    import yaml
    from tightwad.config import load_config

    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "cuda",
            "gpus": [{"name": "GPU0", "vram_gb": 16}],
        },
        "workers": workers or [],
        "models": {
            "test": {
                "path": "/models/test.gguf",
                "ctx_size": 4096,
                "default": True,
            }
        },
    }
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        return load_config(f.name)


class TestCheckVersionMatch:
    def test_no_workers_matches(self):
        config = _make_config(workers=[])
        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")):
            result = check_version_match(config)
        assert result.matched is True
        assert result.local.version == "b8112"
        assert result.workers == []

    def test_matching_versions(self):
        config = _make_config(workers=[{
            "host": "192.168.1.100",
            "ssh_user": "user",
            "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
        }])
        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")), \
             patch("tightwad.worker.get_remote_version",
                   return_value=VersionInfo("192.168.1.100", "b8112")):
            result = check_version_match(config)
        assert result.matched is True
        assert len(result.mismatched) == 0

    def test_mismatched_versions(self):
        config = _make_config(workers=[{
            "host": "192.168.1.100",
            "ssh_user": "user",
            "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
        }])
        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")), \
             patch("tightwad.worker.get_remote_version",
                   return_value=VersionInfo("192.168.1.100", "b8100")):
            result = check_version_match(config)
        assert result.matched is False
        assert len(result.mismatched) == 1
        assert result.mismatched[0].version == "b8100"
        assert "mismatch" in result.message.lower()

    def test_worker_without_ssh_user_unchecked(self):
        config = _make_config(workers=[{
            "host": "192.168.1.100",
            "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
        }])
        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")):
            result = check_version_match(config)
        assert result.matched is True
        assert "192.168.1.100" in result.unchecked

    def test_deduplicates_hosts(self):
        """Multiple GPUs on same host should only check version once."""
        config = _make_config(workers=[{
            "host": "192.168.1.100",
            "ssh_user": "user",
            "gpus": [
                {"name": "GPU0", "vram_gb": 16, "rpc_port": 50052},
                {"name": "GPU1", "vram_gb": 12, "rpc_port": 50053},
            ],
        }])
        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")), \
             patch("tightwad.worker.get_remote_version",
                   return_value=VersionInfo("192.168.1.100", "b8112")) as mock_remote:
            result = check_version_match(config)
        assert mock_remote.call_count == 1
        assert result.matched is True

    def test_multiple_workers_one_mismatched(self):
        config = _make_config(workers=[
            {
                "host": "192.168.1.100",
                "ssh_user": "user",
                "gpus": [{"name": "GPU0", "vram_gb": 16, "rpc_port": 50052}],
            },
            {
                "host": "192.168.1.101",
                "ssh_user": "user",
                "gpus": [{"name": "GPU1", "vram_gb": 8, "rpc_port": 50052}],
            },
        ])

        def fake_remote(user, host):
            if host == "192.168.1.100":
                return VersionInfo(host, "b8112")
            return VersionInfo(host, "b8100")

        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")), \
             patch("tightwad.worker.get_remote_version", side_effect=fake_remote):
            result = check_version_match(config)
        assert result.matched is False
        assert len(result.mismatched) == 1
        assert result.mismatched[0].host == "192.168.1.101"

    def test_worker_ssh_fails_not_mismatched(self):
        """If SSH fails, worker version is None — should not count as mismatch."""
        config = _make_config(workers=[{
            "host": "192.168.1.100",
            "ssh_user": "user",
            "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
        }])
        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")), \
             patch("tightwad.worker.get_remote_version",
                   return_value=VersionInfo("192.168.1.100", None, error="SSH failed")):
            result = check_version_match(config)
        # Can't confirm mismatch if we can't get the version
        assert result.matched is True
        assert len(result.mismatched) == 0


# ---------------------------------------------------------------------------
# VersionCheckResult.message
# ---------------------------------------------------------------------------


class TestVersionCheckResultMessage:
    def test_matched_message(self):
        result = VersionCheckResult(
            matched=True,
            local=VersionInfo("localhost", "b8112"),
        )
        assert "b8112" in result.message

    def test_mismatched_message_includes_hosts(self):
        result = VersionCheckResult(
            matched=False,
            local=VersionInfo("localhost", "b8112"),
            mismatched=[VersionInfo("192.168.1.100", "b8100")],
        )
        assert "192.168.1.100" in result.message
        assert "b8100" in result.message
        assert "b8112" in result.message
        assert "--skip-version-check" in result.message
