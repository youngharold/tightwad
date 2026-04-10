"""Tests for model hub module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tightwad.model_hub import (
    MODEL_REGISTRY,
    ResolvedModel,
    resolve_model,
    list_models,
    download_model,
    validate_download,
)


def test_resolve_model_registry():
    """Registry specs resolve to HuggingFace URLs."""
    resolved = resolve_model("llama3.3:70b-q4_k_m")
    assert "huggingface.co" in resolved.hf_url
    assert resolved.filename.endswith(".gguf")
    assert "Q4_K_M" in resolved.filename


def test_resolve_model_case_insensitive():
    resolved = resolve_model("LLAMA3.3:70B-Q4_K_M")
    assert resolved.filename.endswith(".gguf")


def test_resolve_model_direct_url():
    url = "https://huggingface.co/some/repo/resolve/main/model.gguf"
    resolved = resolve_model(url)
    assert resolved.hf_url == url
    assert resolved.filename == "model.gguf"


def test_resolve_model_hf_repo_path():
    resolved = resolve_model("bartowski/MyModel-GGUF/MyModel-Q4_K_M.gguf")
    assert "huggingface.co/bartowski/MyModel-GGUF" in resolved.hf_url
    assert resolved.filename == "MyModel-Q4_K_M.gguf"


def test_resolve_model_unknown():
    with pytest.raises(ValueError, match="Unknown model spec"):
        resolve_model("nonexistent:model")


def test_list_models():
    models = list_models()
    assert len(models) == len(MODEL_REGISTRY)
    for spec, repo, filename in models:
        assert spec in MODEL_REGISTRY
        assert filename.endswith(".gguf")


def test_validate_download_valid(tmp_path):
    """A file with GGUF magic is valid."""
    path = tmp_path / "test.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 100)
    assert validate_download(path) is True


def test_validate_download_invalid(tmp_path):
    """A file without GGUF magic is invalid."""
    path = tmp_path / "test.gguf"
    path.write_bytes(b"NOT_GGUF" + b"\x00" * 100)
    assert validate_download(path) is False


def test_validate_download_missing():
    """A non-existent file is invalid."""
    assert validate_download(Path("/nonexistent/model.gguf")) is False


def test_download_model_mock(tmp_path):
    """Mock httpx to test download logic."""
    class FakeResponse:
        status_code = 200
        headers = {"content-length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=None):
            yield b"GGUF" + b"\x00" * 96

    progress_calls = []

    def on_progress(downloaded, total):
        progress_calls.append((downloaded, total))

    with patch("tightwad.model_hub.httpx.stream", return_value=FakeResponse()):
        path = download_model(
            "https://example.com/model.gguf",
            dest_dir=tmp_path,
            filename="test.gguf",
            progress_callback=on_progress,
        )

    assert path.exists()
    assert path.stat().st_size == 100
    assert len(progress_calls) == 1
    assert validate_download(path)
