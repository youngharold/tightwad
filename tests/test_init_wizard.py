"""Tests for init wizard and env var config fallback."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from click.testing import CliRunner

from tightwad.cli import cli
from tightwad.config import load_config, load_proxy_from_env
from tightwad.init_wizard import (
    DiscoveredServer,
    detect_backend,
    generate_cluster_yaml,
    generate_local_yaml,
    identify_server,
)


# --- Env var config fallback ---


def test_load_proxy_from_env_both_urls(monkeypatch):
    monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://192.168.1.10:11434")
    monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://192.168.1.20:11434")
    proxy = load_proxy_from_env()
    assert proxy is not None
    assert proxy.draft.url == "http://192.168.1.10:11434"
    assert proxy.target.url == "http://192.168.1.20:11434"
    assert proxy.draft.model_name == "draft"
    assert proxy.draft.backend == "ollama"
    assert proxy.target.model_name == "target"
    assert proxy.target.backend == "ollama"
    assert proxy.host == "0.0.0.0"
    assert proxy.port == 8088
    assert proxy.max_draft_tokens == 8


def test_load_proxy_from_env_custom_values(monkeypatch):
    monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://10.0.0.1:8081")
    monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://10.0.0.2:8080")
    monkeypatch.setenv("TIGHTWAD_DRAFT_MODEL", "qwen3:8b")
    monkeypatch.setenv("TIGHTWAD_TARGET_MODEL", "qwen3:32b")
    monkeypatch.setenv("TIGHTWAD_DRAFT_BACKEND", "llamacpp")
    monkeypatch.setenv("TIGHTWAD_TARGET_BACKEND", "llamacpp")
    monkeypatch.setenv("TIGHTWAD_PORT", "9090")
    monkeypatch.setenv("TIGHTWAD_HOST", "127.0.0.1")
    monkeypatch.setenv("TIGHTWAD_MAX_DRAFT_TOKENS", "64")
    proxy = load_proxy_from_env()
    assert proxy is not None
    assert proxy.draft.model_name == "qwen3:8b"
    assert proxy.target.model_name == "qwen3:32b"
    assert proxy.draft.backend == "llamacpp"
    assert proxy.port == 9090
    assert proxy.host == "127.0.0.1"
    assert proxy.max_draft_tokens == 64


def test_load_proxy_from_env_missing_urls():
    # Ensure the env vars are not set
    env = os.environ.copy()
    env.pop("TIGHTWAD_DRAFT_URL", None)
    env.pop("TIGHTWAD_TARGET_URL", None)
    with patch.dict(os.environ, env, clear=True):
        proxy = load_proxy_from_env()
        assert proxy is None


def test_load_config_falls_back_to_env(monkeypatch, tmp_path):
    """load_config with nonexistent YAML falls back to env vars."""
    monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://host1:11434")
    monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://host2:11434")
    nonexistent = tmp_path / "does_not_exist.yaml"
    config = load_config(nonexistent)
    assert config.proxy is not None
    assert config.proxy.draft.url == "http://host1:11434"
    assert config.proxy.target.url == "http://host2:11434"
    # Coordinator should have empty defaults
    assert config.coordinator_gpus == []
    assert config.workers == []
    assert config.models == {}


def test_load_config_no_yaml_no_env(tmp_path):
    """load_config with no YAML and no env vars raises FileNotFoundError."""
    nonexistent = tmp_path / "does_not_exist.yaml"
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(FileNotFoundError):
            load_config(nonexistent)


# --- YAML generation ---


def test_generate_cluster_yaml():
    """Generated YAML should be parseable by load_config."""
    draft = DiscoveredServer(host="192.168.1.10", port=11434, backend="ollama", models=["qwen3:8b"])
    target = DiscoveredServer(host="192.168.1.20", port=8080, backend="llamacpp", models=["qwen3:32b"])

    yaml_str = generate_cluster_yaml(draft, "qwen3:8b", target, "qwen3:32b")

    # Write to file and parse
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        f.flush()
        config = load_config(f.name)

    assert config.proxy is not None
    assert config.proxy.draft.url == "http://192.168.1.10:11434"
    assert config.proxy.draft.model_name == "qwen3:8b"
    assert config.proxy.draft.backend == "ollama"
    assert config.proxy.target.url == "http://192.168.1.20:8080"
    assert config.proxy.target.model_name == "qwen3:32b"
    assert config.proxy.target.backend == "llamacpp"
    assert config.proxy.max_draft_tokens == 32
    assert config.proxy.port == 8088

    os.unlink(f.name)


# --- Server identification ---


class FakeResponse:
    """Minimal fake httpx.Response for testing."""
    def __init__(self, text="", status_code=200, json_data=None):
        self.text = text
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


class FakeAsyncClient:
    """Fake httpx.AsyncClient that routes GET requests to a handler."""
    def __init__(self, handler):
        self._handler = handler

    async def get(self, url, **kwargs):
        return self._handler(url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_identify_server_ollama():
    """Mock Ollama detection: GET / returns 'Ollama is running', GET /api/tags returns models."""
    def handler(url):
        if url.endswith("/"):
            return FakeResponse(text="Ollama is running")
        if url.endswith("/api/tags"):
            return FakeResponse(json_data={"models": [{"name": "qwen3:8b"}, {"name": "qwen3:32b"}]})
        return FakeResponse(status_code=404)

    with patch("tightwad.init_wizard.httpx.AsyncClient", lambda **kw: FakeAsyncClient(handler)):
        server = await identify_server("192.168.1.10", 11434)

    assert server is not None
    assert server.backend == "ollama"
    assert server.models == ["qwen3:8b", "qwen3:32b"]
    assert server.status == "healthy"


@pytest.mark.asyncio
async def test_identify_server_llamacpp():
    """Mock llama-server detection: GET /health returns 200, GET /v1/models returns models."""
    def handler(url):
        if url.endswith("/"):
            return FakeResponse(text="llama-server")  # Not Ollama
        if url.endswith("/health"):
            return FakeResponse(json_data={"status": "ok"})
        if url.endswith("/v1/models"):
            return FakeResponse(json_data={"data": [{"id": "qwen3-32b"}]})
        return FakeResponse(status_code=404)

    with patch("tightwad.init_wizard.httpx.AsyncClient", lambda **kw: FakeAsyncClient(handler)):
        server = await identify_server("192.168.1.20", 8080)

    assert server is not None
    assert server.backend == "llamacpp"
    assert server.models == ["qwen3-32b"]
    assert server.status == "healthy"


# --- detect_backend ---


def test_detect_backend_ollama():
    assert detect_backend("http://192.168.1.10:11434") == "ollama"


def test_detect_backend_llamacpp():
    assert detect_backend("http://192.168.1.10:8080") == "llamacpp"


def test_detect_backend_no_port():
    assert detect_backend("http://192.168.1.10") == "llamacpp"


# --- Non-interactive init CLI ---


def test_non_interactive_init(tmp_path):
    """--draft-url + --target-url generates config without prompts."""
    output = tmp_path / "cluster.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "init",
        "--draft-url", "http://192.168.1.101:11434",
        "--draft-model", "qwen3:8b",
        "--target-url", "http://192.168.1.100:8080",
        "--target-model", "qwen3:32b",
        "-o", str(output),
        "-y",
    ])
    assert result.exit_code == 0, result.output
    assert output.exists()

    config = load_config(str(output))
    assert config.proxy is not None
    assert config.proxy.draft.url == "http://192.168.1.101:11434"
    assert config.proxy.draft.model_name == "qwen3:8b"
    assert config.proxy.draft.backend == "ollama"
    assert config.proxy.target.url == "http://192.168.1.100:8080"
    assert config.proxy.target.model_name == "qwen3:32b"
    assert config.proxy.target.backend == "llamacpp"


def test_non_interactive_explicit_backends(tmp_path):
    """Explicit --draft-backend and --target-backend override auto-detect."""
    output = tmp_path / "cluster.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "init",
        "--draft-url", "http://192.168.1.101:8081",
        "--draft-model", "qwen3-8b",
        "--draft-backend", "llamacpp",
        "--target-url", "http://192.168.1.100:8080",
        "--target-model", "qwen3-32b",
        "--target-backend", "llamacpp",
        "--max-draft-tokens", "64",
        "-o", str(output),
        "-y",
    ])
    assert result.exit_code == 0, result.output

    config = load_config(str(output))
    assert config.proxy.draft.backend == "llamacpp"
    assert config.proxy.target.backend == "llamacpp"
    assert config.proxy.max_draft_tokens == 64


def test_non_interactive_missing_model():
    """--draft-url without --draft-model should error."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "init",
        "--draft-url", "http://192.168.1.101:11434",
        "--target-url", "http://192.168.1.100:11434",
        "--target-model", "qwen3:32b",
    ])
    assert result.exit_code != 0
    assert "draft-model" in result.output.lower() or "draft-model" in str(result.exception).lower()


def test_yes_flag_overwrites(tmp_path):
    """-y overwrites existing file without prompt."""
    output = tmp_path / "cluster.yaml"
    output.write_text("old content")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "init",
        "--draft-url", "http://192.168.1.101:11434",
        "--draft-model", "qwen3:8b",
        "--target-url", "http://192.168.1.100:11434",
        "--target-model", "qwen3:32b",
        "-o", str(output),
        "-y",
    ])
    assert result.exit_code == 0, result.output
    assert "old content" not in output.read_text()
    assert "proxy" in output.read_text()


# --- Local mode ---


def test_generate_local_yaml_single_gpu():
    """generate_local_yaml produces valid YAML with GPU info."""
    from tightwad.gpu_detect import DetectedGPU

    gpus = [DetectedGPU(name="RTX 4070", vram_mb=16384, backend="cuda")]
    yaml_str = generate_local_yaml(gpus, binary="/usr/local/bin/llama-server", model_path="/models/test.gguf")

    import yaml as _yaml
    data = _yaml.safe_load(yaml_str)

    assert data["coordinator"]["backend"] == "cuda"
    assert data["coordinator"]["gpus"][0]["name"] == "RTX 4070"
    assert data["coordinator"]["gpus"][0]["vram_gb"] == 16
    assert data["models"]["default"]["path"] == "/models/test.gguf"
    assert data["binaries"]["coordinator"] == "/usr/local/bin/llama-server"


def test_generate_local_yaml_no_model():
    """generate_local_yaml with no model_path produces empty models dict."""
    from tightwad.gpu_detect import DetectedGPU

    gpus = [DetectedGPU(name="Apple M4", vram_mb=16384, backend="metal")]
    yaml_str = generate_local_yaml(gpus, binary=None)

    import yaml as _yaml
    data = _yaml.safe_load(yaml_str)

    assert data["models"] == {}
    assert "binaries" not in data


def test_local_init_cli(tmp_path):
    """--local flag generates coordinator-only config."""
    from tightwad.gpu_detect import DetectedGPU

    output = tmp_path / "cluster.yaml"
    runner = CliRunner()

    fake_gpus = [DetectedGPU(name="RTX 4070", vram_mb=16384, backend="cuda")]
    with patch("tightwad.cli.detect_gpus", return_value=fake_gpus) if False else \
         patch("tightwad.gpu_detect.detect_gpus", return_value=fake_gpus), \
         patch("tightwad.gpu_detect.detect_binary", return_value="/usr/bin/llama-server"):
        result = runner.invoke(cli, [
            "init", "--local",
            "--model-path", "/models/test.gguf",
            "-o", str(output),
            "-y",
        ])

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "cuda" in output.read_text()


def test_no_yes_flag_existing_file_errors(tmp_path):
    """Without -y, existing file should error in non-interactive mode."""
    output = tmp_path / "cluster.yaml"
    output.write_text("old content")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "init",
        "--draft-url", "http://192.168.1.101:11434",
        "--draft-model", "qwen3:8b",
        "--target-url", "http://192.168.1.100:11434",
        "--target-model", "qwen3:32b",
        "-o", str(output),
    ])
    assert result.exit_code != 0
    assert "already exists" in result.output
