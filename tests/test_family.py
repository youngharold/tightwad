"""Tests for model family detection and compatibility validation."""

import struct
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tightwad.family import (
    FamilyCheckResult,
    ModelFamily,
    arch_to_family,
    check_compatibility,
    check_proxy_families,
    detect_gguf_family,
    detect_ollama_family,
    detect_llamacpp_family,
)


# ---------------------------------------------------------------------------
# arch_to_family mapping
# ---------------------------------------------------------------------------


class TestArchToFamily:
    def test_llama_variants(self):
        assert arch_to_family("llama") == "llama"
        assert arch_to_family("Llama") == "llama"
        assert arch_to_family("LLAMA") == "llama"

    def test_qwen_variants(self):
        assert arch_to_family("qwen") == "qwen"
        assert arch_to_family("qwen2") == "qwen"
        assert arch_to_family("qwen2_moe") == "qwen"
        assert arch_to_family("qwen2moe") == "qwen"
        assert arch_to_family("qwen3") == "qwen"
        assert arch_to_family("qwen3moe") == "qwen"

    def test_mistral_mixtral(self):
        assert arch_to_family("mistral") == "mistral"
        assert arch_to_family("mixtral") == "mistral"

    def test_gemma(self):
        assert arch_to_family("gemma") == "gemma"
        assert arch_to_family("gemma2") == "gemma"
        assert arch_to_family("gemma3") == "gemma"

    def test_phi(self):
        assert arch_to_family("phi2") == "phi"
        assert arch_to_family("phi3") == "phi"
        assert arch_to_family("phi4") == "phi"

    def test_deepseek(self):
        assert arch_to_family("deepseek") == "deepseek"
        assert arch_to_family("deepseek2") == "deepseek"

    def test_chatglm(self):
        assert arch_to_family("chatglm") == "chatglm"
        assert arch_to_family("glm4") == "chatglm"

    def test_unknown_returns_self(self):
        assert arch_to_family("some_new_arch") == "some_new_arch"
        assert arch_to_family("FutureModel") == "futuremodel"

    def test_hyphen_normalization(self):
        assert arch_to_family("command-r") == "command-r"

    def test_whitespace_stripped(self):
        assert arch_to_family("  llama  ") == "llama"


# ---------------------------------------------------------------------------
# check_compatibility
# ---------------------------------------------------------------------------


def _family(arch: str, model: str = "test-model", source: str = "ollama") -> ModelFamily:
    return ModelFamily(
        arch=arch,
        family=arch_to_family(arch),
        model_name=model,
        source=source,
    )


class TestCheckCompatibility:
    def test_same_family_compatible(self):
        result = check_compatibility(
            _family("llama", "llama-3.2-3b"),
            _family("llama", "llama-3.3-70b"),
        )
        assert result.compatible is True
        assert result.draft is not None
        assert result.target is not None

    def test_qwen_variants_compatible(self):
        result = check_compatibility(
            _family("qwen2", "qwen-2.5-3b"),
            _family("qwen3", "qwen-3-32b"),
        )
        assert result.compatible is True

    def test_different_families_incompatible(self):
        result = check_compatibility(
            _family("llama", "llama-3.2-3b"),
            _family("qwen2", "qwen-2.5-72b"),
        )
        assert result.compatible is False
        assert "INCOMPATIBLE" in result.message
        assert "<5%" in result.message

    def test_mistral_vs_llama_incompatible(self):
        result = check_compatibility(
            _family("mistral", "mistral-7b"),
            _family("llama", "llama-3.3-70b"),
        )
        assert result.compatible is False

    def test_both_none_skips(self):
        result = check_compatibility(None, None)
        assert result.compatible is True
        assert "Could not detect" in result.message

    def test_draft_none_skips(self):
        result = check_compatibility(None, _family("llama"))
        assert result.compatible is True
        assert "Cannot verify" in result.message

    def test_target_none_skips(self):
        result = check_compatibility(_family("llama"), None)
        assert result.compatible is True
        assert "Cannot verify" in result.message

    def test_incompatible_message_suggests_fix(self):
        result = check_compatibility(
            _family("gemma", "gemma-2b"),
            _family("llama", "llama-70b"),
        )
        assert result.compatible is False
        assert "llama" in result.message.lower()
        assert "Use a llama-family model" in result.message

    def test_mixtral_and_mistral_compatible(self):
        """Mixtral and Mistral share the mistral family."""
        result = check_compatibility(
            _family("mistral", "mistral-7b"),
            _family("mixtral", "mixtral-8x22b"),
        )
        assert result.compatible is True


# ---------------------------------------------------------------------------
# Ollama detection (mocked HTTP)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Minimal async client mock that returns canned responses."""

    def __init__(self, handler):
        self._handler = handler

    async def post(self, url, **kwargs):
        return self._handler(url, kwargs)

    async def get(self, url, **kwargs):
        return self._handler(url, kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestDetectOllamaFamily:
    @pytest.mark.asyncio
    async def test_detects_from_general_architecture(self):
        def handler(url, kwargs):
            return _FakeResponse(json_data={
                "model_info": {
                    "general.architecture": "llama",
                    "llama.block_count": 32,
                }
            })

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await detect_ollama_family("http://localhost:11434", "llama3.2:3b")

        assert result is not None
        assert result.arch == "llama"
        assert result.family == "llama"
        assert result.source == "ollama"

    @pytest.mark.asyncio
    async def test_detects_qwen_from_architecture(self):
        def handler(url, kwargs):
            return _FakeResponse(json_data={
                "model_info": {
                    "general.architecture": "qwen2",
                    "qwen2.block_count": 40,
                }
            })

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await detect_ollama_family("http://localhost:11434", "qwen2.5:32b")

        assert result is not None
        assert result.family == "qwen"

    @pytest.mark.asyncio
    async def test_fallback_to_key_prefix(self):
        def handler(url, kwargs):
            return _FakeResponse(json_data={
                "model_info": {
                    "mistral.block_count": 32,
                    "mistral.attention.head_count": 32,
                }
            })

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await detect_ollama_family("http://localhost:11434", "mistral:7b")

        assert result is not None
        assert result.family == "mistral"

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self):
        def handler(url, kwargs):
            return _FakeResponse(status_code=404)

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await detect_ollama_family("http://localhost:11434", "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self):
        result = await detect_ollama_family("http://127.0.0.1:1", "model", timeout=0.5)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_model_info(self):
        def handler(url, kwargs):
            return _FakeResponse(json_data={"model_info": {}})

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await detect_ollama_family("http://localhost:11434", "mystery")

        assert result is None


# ---------------------------------------------------------------------------
# llama-server detection (mocked HTTP)
# ---------------------------------------------------------------------------


class TestDetectLlamacppFamily:
    @pytest.mark.asyncio
    async def test_detects_from_props(self):
        def handler(url, kwargs):
            if "/props" in url:
                return _FakeResponse(json_data={"general.architecture": "gemma2"})
            return _FakeResponse(status_code=404)

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await detect_llamacpp_family("http://localhost:8080", "gemma2:9b")

        assert result is not None
        assert result.family == "gemma"
        assert result.source == "llamacpp"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_arch_in_props(self):
        def handler(url, kwargs):
            return _FakeResponse(json_data={"model": "some-model"})

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await detect_llamacpp_family("http://localhost:8080", "model")

        assert result is None


# ---------------------------------------------------------------------------
# GGUF detection
# ---------------------------------------------------------------------------


def _make_gguf(path, arch="llama"):
    """Write a minimal valid GGUF v3 file with a general.architecture KV."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # magic
        f.write(struct.pack("<I", 3))            # version
        f.write(struct.pack("<Q", 0))            # tensor count
        f.write(struct.pack("<Q", 1))            # kv count
        key = b"general.architecture"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 8))            # string type
        value = arch.encode()
        f.write(struct.pack("<Q", len(value)))
        f.write(value)


class TestDetectGgufFamily:
    def test_detects_from_gguf_file(self, tmp_path):
        gguf_path = tmp_path / "test.gguf"
        _make_gguf(gguf_path, "llama")

        result = detect_gguf_family(str(gguf_path), "test-model")
        assert result is not None
        assert result.arch == "llama"
        assert result.family == "llama"
        assert result.source == "gguf"

    def test_detects_qwen_from_gguf(self, tmp_path):
        gguf_path = tmp_path / "qwen.gguf"
        _make_gguf(gguf_path, "qwen2")

        result = detect_gguf_family(str(gguf_path))
        assert result is not None
        assert result.family == "qwen"

    def test_returns_none_for_missing_file(self):
        result = detect_gguf_family("/nonexistent/path.gguf")
        assert result is None

    def test_returns_none_for_invalid_file(self, tmp_path):
        bad_file = tmp_path / "not_gguf.bin"
        bad_file.write_bytes(b"not a gguf file at all")
        result = detect_gguf_family(str(bad_file))
        assert result is None


# ---------------------------------------------------------------------------
# Full proxy check (mocked)
# ---------------------------------------------------------------------------


class TestCheckProxyFamilies:
    @pytest.mark.asyncio
    async def test_compatible_ollama_pair(self):
        call_count = {"n": 0}

        def handler(url, kwargs):
            call_count["n"] += 1
            return _FakeResponse(json_data={
                "model_info": {"general.architecture": "llama"}
            })

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await check_proxy_families(
                draft_url="http://draft:11434",
                draft_model="llama3.2:3b",
                draft_backend="ollama",
                target_url="http://target:11434",
                target_model="llama3.3:70b",
                target_backend="ollama",
            )

        assert result.compatible is True

    @pytest.mark.asyncio
    async def test_incompatible_ollama_pair(self):
        responses = iter([
            _FakeResponse(json_data={"model_info": {"general.architecture": "llama"}}),
            _FakeResponse(json_data={"model_info": {"general.architecture": "qwen2"}}),
        ])

        def handler(url, kwargs):
            return next(responses)

        with patch("tightwad.family.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await check_proxy_families(
                draft_url="http://draft:11434",
                draft_model="llama3.2:3b",
                draft_backend="ollama",
                target_url="http://target:11434",
                target_model="qwen2.5:72b",
                target_backend="ollama",
            )

        assert result.compatible is False
        assert "INCOMPATIBLE" in result.message

    @pytest.mark.asyncio
    async def test_unreachable_servers_skips_gracefully(self):
        result = await check_proxy_families(
            draft_url="http://127.0.0.1:1",
            draft_model="model",
            draft_backend="ollama",
            target_url="http://127.0.0.1:2",
            target_model="model",
            target_backend="ollama",
        )
        assert result.compatible is True
        assert "Could not detect" in result.message
