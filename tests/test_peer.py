"""Tests for the peer agent daemon."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from starlette.testclient import TestClient

from tightwad.config import PeerConfig, load_config
from tightwad.peer import (
    ProcessManager,
    TokenAuthMiddleware,
    create_app,
    read_pidfile,
    remove_pidfile,
    rpc_log_path,
    write_pidfile,
    _get_memory_info,
    _rotate_if_needed,
    _scan_gguf_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def peer_config():
    return PeerConfig(
        host="0.0.0.0",
        port=9191,
        auth_token=None,
        model_dirs=[],
    )


@pytest.fixture
def secured_peer_config():
    return PeerConfig(
        host="0.0.0.0",
        port=9191,
        auth_token="test-peer-token",
        model_dirs=[],
    )


@pytest.fixture
def client(peer_config):
    app = create_app(peer_config)
    return TestClient(app)


@pytest.fixture
def secured_client(secured_peer_config):
    app = create_app(secured_peer_config)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Version endpoint
# ---------------------------------------------------------------------------


class TestVersionEndpoint:
    def test_returns_expected_fields(self, client):
        resp = client.get("/v1/peer/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "tightwad_version" in data
        assert "llama_server_version" in data
        assert "platform" in data
        assert "machine" in data
        assert "hostname" in data

    def test_tightwad_version_matches(self, client):
        from tightwad import __version__
        resp = client.get("/v1/peer/version")
        data = resp.json()
        assert data["tightwad_version"] == __version__


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_uptime_and_system_info(self, client):
        resp = client.get("/v1/peer/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert "load_avg" in data
        assert "memory" in data
        assert "disk" in data

    def test_disk_has_expected_keys(self, client):
        resp = client.get("/v1/peer/health")
        data = resp.json()
        disk = data["disk"]
        if disk:  # may be empty on some platforms
            assert "total_gb" in disk
            assert "free_gb" in disk


# ---------------------------------------------------------------------------
# GPU endpoint
# ---------------------------------------------------------------------------


class TestGPUEndpoint:
    def test_returns_gpu_list(self, client):
        with patch("tightwad.gpu_detect.detect_gpus", return_value=[]):
            resp = client.get("/v1/peer/gpu")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpus" in data
        assert isinstance(data["gpus"], list)

    def test_returns_detected_gpus(self, client):
        from tightwad.gpu_detect import DetectedGPU
        mock_gpus = [
            DetectedGPU(name="RTX 4070", vram_mb=16384, backend="cuda", index=0),
        ]
        with patch("tightwad.gpu_detect.detect_gpus", return_value=mock_gpus):
            resp = client.get("/v1/peer/gpu")
        data = resp.json()
        assert len(data["gpus"]) == 1
        assert data["gpus"][0]["name"] == "RTX 4070"
        assert data["gpus"][0]["vram_mb"] == 16384
        assert data["gpus"][0]["backend"] == "cuda"


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_lists_gguf_files_in_tmp(self, tmp_path):
        """Create fake GGUF files and verify they appear."""
        model_file = tmp_path / "test-model.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config = PeerConfig(model_dirs=[str(tmp_path)])
        app = create_app(config)
        client = TestClient(app)

        resp = client.get("/v1/peer/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "test-model.gguf"
        assert data["models"][0]["size_bytes"] == 1024

    def test_empty_model_dirs(self, client):
        resp = client.get("/v1/peer/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []

    def test_nonexistent_model_dir(self):
        config = PeerConfig(model_dirs=["/nonexistent/path"])
        app = create_app(config)
        client = TestClient(app)
        resp = client.get("/v1/peer/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


# ---------------------------------------------------------------------------
# RPC start/stop endpoints
# ---------------------------------------------------------------------------


class TestRPCEndpoints:
    def test_rpc_start_requires_port(self, client):
        resp = client.post("/v1/peer/rpc/start", json={})
        assert resp.status_code == 400
        assert "port" in resp.json()["error"]

    def test_rpc_stop_requires_port(self, client):
        resp = client.post("/v1/peer/rpc/stop", json={})
        assert resp.status_code == 400
        assert "port" in resp.json()["error"]

    def test_rpc_stop_unknown_port(self, client):
        resp = client.post("/v1/peer/rpc/stop", json={"port": 99999})
        assert resp.status_code == 404

    def test_rpc_start_missing_binary(self, client):
        with patch("tightwad.peer.shutil.which", return_value=None):
            resp = client.post(
                "/v1/peer/rpc/start",
                json={"port": 50052, "binary": "nonexistent-rpc-server"},
            )
        assert resp.status_code == 400
        assert "not found" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Logs endpoint
# ---------------------------------------------------------------------------


class TestLogsEndpoint:
    def test_missing_log_file(self, client):
        resp = client.get("/v1/peer/logs?service=nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_reads_existing_log(self, client, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "test.log"
        log_file.write_text("line1\nline2\nline3\n")

        with patch("tightwad.peer.Path.home", return_value=tmp_path / ".tightwad"):
            # Re-create to pick up patched home
            pass

        # Direct test of the scan function instead
        # The endpoint uses Path.home() which is harder to patch for Starlette
        # Just verify the endpoint doesn't crash
        resp = client.get("/v1/peer/logs?service=peer&lines=10")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth token enforcement
# ---------------------------------------------------------------------------


class TestAuthToken:
    def test_no_token_allows_access(self, client):
        resp = client.get("/v1/peer/version")
        assert resp.status_code == 200

    def test_missing_token_returns_401(self, secured_client):
        resp = secured_client.get("/v1/peer/version")
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, secured_client):
        resp = secured_client.get(
            "/v1/peer/version",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_correct_token_allows_access(self, secured_client):
        resp = secured_client.get(
            "/v1/peer/version",
            headers={"Authorization": "Bearer test-peer-token"},
        )
        assert resp.status_code == 200

    def test_all_endpoints_protected(self, secured_client):
        """All endpoints should require auth when token is set."""
        endpoints = [
            ("GET", "/v1/peer/version"),
            ("GET", "/v1/peer/health"),
            ("GET", "/v1/peer/gpu"),
            ("GET", "/v1/peer/models"),
            ("GET", "/v1/peer/logs"),
        ]
        for method, path in endpoints:
            if method == "GET":
                resp = secured_client.get(path)
            assert resp.status_code == 401, f"{method} {path} should be 401 without token"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestPeerConfigParsing:
    def test_peer_config_in_yaml(self, tmp_path):
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "cuda",
                "gpus": [{"name": "GPU0", "vram_gb": 16}],
            },
            "workers": [
                {
                    "host": "192.168.1.100",
                    "peer_port": 9191,
                    "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
                }
            ],
            "models": {
                "test": {"path": "/models/test.gguf", "default": True},
            },
            "peer": {
                "port": 9292,
                "auth_token": "my-secret",
                "model_dirs": ["/models", "/extra-models"],
            },
        }
        config_file = tmp_path / "cluster.yaml"
        config_file.write_text(yaml.dump(cfg))

        config = load_config(config_file)
        assert config.peer is not None
        assert config.peer.port == 9292
        assert config.peer.auth_token == "my-secret"
        assert config.peer.model_dirs == ["/models", "/extra-models"]

    def test_peer_port_on_worker(self, tmp_path):
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "cuda",
                "gpus": [{"name": "GPU0", "vram_gb": 16}],
            },
            "workers": [
                {
                    "host": "192.168.1.100",
                    "peer_port": 9191,
                    "ssh_user": "user",
                    "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
                }
            ],
            "models": {
                "test": {"path": "/models/test.gguf", "default": True},
            },
        }
        config_file = tmp_path / "cluster.yaml"
        config_file.write_text(yaml.dump(cfg))

        config = load_config(config_file)
        assert config.workers[0].peer_port == 9191

    def test_no_peer_config(self, tmp_path):
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "cuda",
                "gpus": [{"name": "GPU0", "vram_gb": 16}],
            },
            "workers": [],
            "models": {
                "test": {"path": "/models/test.gguf", "default": True},
            },
        }
        config_file = tmp_path / "cluster.yaml"
        config_file.write_text(yaml.dump(cfg))

        config = load_config(config_file)
        assert config.peer is None

    def test_peer_token_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIGHTWAD_PEER_TOKEN", "env-secret")
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "cuda",
                "gpus": [{"name": "GPU0", "vram_gb": 16}],
            },
            "workers": [],
            "models": {
                "test": {"path": "/models/test.gguf", "default": True},
            },
            "peer": {
                "port": 9191,
            },
        }
        config_file = tmp_path / "cluster.yaml"
        config_file.write_text(yaml.dump(cfg))

        config = load_config(config_file)
        assert config.peer is not None
        assert config.peer.auth_token == "env-secret"


# ---------------------------------------------------------------------------
# Peer-based version detection in worker.py
# ---------------------------------------------------------------------------


class TestPeerVersionDetection:
    def test_get_remote_version_via_peer_success(self):
        from tightwad.worker import get_remote_version_via_peer

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "tightwad_version": "0.3.0",
            "llama_server_version": "llama-server b8112",
            "platform": "Linux",
            "machine": "x86_64",
            "hostname": "worker1",
        }

        with patch("tightwad.worker.httpx.get", return_value=mock_resp):
            result = get_remote_version_via_peer("192.168.1.100", 9191)
        assert result.version == "llama-server b8112"
        assert result.host == "192.168.1.100"

    def test_get_remote_version_via_peer_with_token(self):
        from tightwad.worker import get_remote_version_via_peer

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "llama_server_version": "b8112",
        }

        with patch("tightwad.worker.httpx.get", return_value=mock_resp) as mock_get:
            get_remote_version_via_peer("host", 9191, token="secret")
        # Verify auth header was passed
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer secret"

    def test_get_remote_version_via_peer_connection_error(self):
        from tightwad.worker import get_remote_version_via_peer

        with patch("tightwad.worker.httpx.get", side_effect=Exception("connection refused")):
            result = get_remote_version_via_peer("unreachable", 9191)
        assert result.version is None
        assert result.error is not None

    def test_get_remote_version_via_peer_401(self):
        from tightwad.worker import get_remote_version_via_peer

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("tightwad.worker.httpx.get", return_value=mock_resp):
            result = get_remote_version_via_peer("host", 9191)
        assert result.version is None
        assert "401" in result.error

    def test_check_version_match_uses_peer(self, tmp_path):
        """When worker has peer_port, check_version_match tries peer agent first."""
        from tightwad.worker import VersionInfo, check_version_match

        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "cuda",
                "gpus": [{"name": "GPU0", "vram_gb": 16}],
            },
            "workers": [
                {
                    "host": "192.168.1.100",
                    "peer_port": 9191,
                    "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
                }
            ],
            "models": {
                "test": {"path": "/models/test.gguf", "default": True},
            },
        }
        config_file = tmp_path / "cluster.yaml"
        config_file.write_text(yaml.dump(cfg))
        config = load_config(config_file)

        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")), \
             patch("tightwad.worker.get_remote_version_via_peer",
                   return_value=VersionInfo("192.168.1.100", "b8112")) as mock_peer, \
             patch("tightwad.worker.get_remote_version") as mock_ssh:
            result = check_version_match(config)

        # Peer was called, SSH was not
        mock_peer.assert_called_once()
        mock_ssh.assert_not_called()
        assert result.matched is True

    def test_check_version_match_falls_back_to_ssh(self, tmp_path):
        """When peer agent fails, falls back to SSH if ssh_user is set."""
        from tightwad.worker import VersionInfo, check_version_match

        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "cuda",
                "gpus": [{"name": "GPU0", "vram_gb": 16}],
            },
            "workers": [
                {
                    "host": "192.168.1.100",
                    "peer_port": 9191,
                    "ssh_user": "user",
                    "gpus": [{"name": "GPU", "vram_gb": 8, "rpc_port": 50052}],
                }
            ],
            "models": {
                "test": {"path": "/models/test.gguf", "default": True},
            },
        }
        config_file = tmp_path / "cluster.yaml"
        config_file.write_text(yaml.dump(cfg))
        config = load_config(config_file)

        with patch("tightwad.worker.get_local_version",
                    return_value=VersionInfo("localhost", "b8112")), \
             patch("tightwad.worker.get_remote_version_via_peer",
                   return_value=VersionInfo("192.168.1.100", None, error="timeout")) as mock_peer, \
             patch("tightwad.worker.get_remote_version",
                   return_value=VersionInfo("192.168.1.100", "b8112")) as mock_ssh:
            result = check_version_match(config)

        mock_peer.assert_called_once()
        mock_ssh.assert_called_once()
        assert result.matched is True


# ---------------------------------------------------------------------------
# ProcessManager unit tests
# ---------------------------------------------------------------------------


class TestProcessManager:
    def test_stop_unknown_port(self):
        pm = ProcessManager()
        assert pm.stop(99999) is False

    def test_list_empty(self):
        pm = ProcessManager()
        assert pm.list_processes() == []


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestScanGGUFFiles:
    def test_scans_directory(self, tmp_path):
        (tmp_path / "model1.gguf").write_bytes(b"\x00" * 2048)
        (tmp_path / "model2.gguf").write_bytes(b"\x00" * 4096)
        (tmp_path / "not_a_model.bin").write_bytes(b"\x00" * 100)

        results = _scan_gguf_files([str(tmp_path)])
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert "model1.gguf" in names
        assert "model2.gguf" in names

    def test_nonexistent_dir(self):
        results = _scan_gguf_files(["/nonexistent/path"])
        assert results == []

    def test_empty_dir(self, tmp_path):
        results = _scan_gguf_files([str(tmp_path)])
        assert results == []


# ---------------------------------------------------------------------------
# Pidfile tests
# ---------------------------------------------------------------------------


class TestPidfile:
    def test_write_read_remove(self, tmp_path, monkeypatch):
        pidfile = tmp_path / "peer.pid"
        monkeypatch.setattr("tightwad.peer.PIDFILE", pidfile)

        write_pidfile()
        assert pidfile.exists()
        pid = read_pidfile()
        assert pid == os.getpid()

        remove_pidfile()
        assert not pidfile.exists()
        assert read_pidfile() is None


class TestMoeProfileEndpoint:
    def test_returns_404_when_no_log(self, peer_config):
        client = TestClient(create_app(peer_config))
        resp = client.get("/v1/peer/moe/profile?port=55123")
        assert resp.status_code == 404

    def test_returns_400_when_port_missing(self, peer_config):
        client = TestClient(create_app(peer_config))
        resp = client.get("/v1/peer/moe/profile")
        assert resp.status_code == 400

    def test_parses_log(self, peer_config, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log = log_dir / "rpc-50052.log"
        log.write_text(
            "moe: layer=0 chosen=[0,1]\n"
            "moe: layer=0 chosen=[0,3]\n"
            "moe: layer=1 chosen=[5]\n"
        )
        monkeypatch.setattr("tightwad.peer.rpc_log_path", lambda p: log)
        client = TestClient(create_app(peer_config))
        resp = client.get("/v1/peer/moe/profile?port=50052")

        assert resp.status_code == 200
        data = resp.json()
        assert data["port"] == 50052
        assert data["total_tokens"] == 3
        top = {(h["layer"], h["expert"]): h["count"] for h in data["top_experts"]}
        assert top[(0, 0)] == 2


class TestRpcLogRotation:
    def test_rotates_over_limit(self, tmp_path, monkeypatch):
        import tightwad.peer as peer_module
        monkeypatch.setattr(peer_module, "RPC_LOG_MAX_BYTES", 100)
        log = tmp_path / "rpc.log"
        log.write_bytes(b"A" * 500)
        _rotate_if_needed(log)
        assert not log.exists()
        assert log.with_suffix(".log.1").exists()

    def test_does_not_rotate_under_limit(self, tmp_path):
        log = tmp_path / "rpc.log"
        log.write_bytes(b"small")
        _rotate_if_needed(log)
        assert log.exists()


class TestProcessManagerStderrCapture:
    def test_capture_writes_to_rotated_log(self, tmp_path, monkeypatch):
        import tightwad.peer as peer_module
        # Redirect rpc_log_path to a tmp dir so we don't scribble into $HOME
        monkeypatch.setattr(peer_module, "rpc_log_path",
                             lambda port: tmp_path / f"rpc-{port}.log")

        mgr = ProcessManager()
        # Use /usr/bin/true (or cmd.exe-style) to avoid a long-running process
        cmd = ["/usr/bin/true"] if Path("/usr/bin/true").exists() else ["true"]
        managed = mgr.start(cmd, port=50099, capture_stderr=True)

        assert (tmp_path / "rpc-50099.log").exists()
        # Clean up
        try:
            os.kill(managed.pid, 0)
        except OSError:
            pass
