"""Tests for the speculative decoding proxy."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from starlette.testclient import TestClient

from tightwad.config import ProxyConfig, ServerEndpoint
from tightwad.proxy import SpeculativeProxy, apply_chat_template, create_app
from tightwad.speculation import DraftToken


@pytest.fixture
def proxy_config():
    return ProxyConfig(
        draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b"),
        target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b"),
        host="0.0.0.0",
        port=8088,
        max_draft_tokens=8,
        fallback_on_draft_failure=True,
    )


class TestChatTemplate:
    def test_basic_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt, stop = apply_chat_template(messages)
        assert "<|im_start|>user\nHello<|im_end|>" in prompt
        assert "<|im_start|>assistant\n" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")
        assert "<|im_end|>" in stop

    def test_system_message(self):
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hi"},
        ]
        prompt, stop = apply_chat_template(messages)
        assert "<|im_start|>system\nYou are a pirate.<|im_end|>" in prompt
        # System should not appear as a regular message
        assert prompt.count("<|im_start|>system") == 1

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        prompt, stop = apply_chat_template(messages)
        assert "<|im_start|>user\nWhat is 2+2?<|im_end|>" in prompt
        assert "<|im_start|>assistant\n4<|im_end|>" in prompt
        assert "<|im_start|>user\nAnd 3+3?<|im_end|>" in prompt

    def test_default_system_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        prompt, stop = apply_chat_template(messages)
        assert "You are a helpful assistant." in prompt


class TestProxyApp:
    def test_models_endpoint(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "qwen3-32b"
        assert data["data"][1]["id"] == "qwen3-8b"

    def test_status_endpoint_servers_down(self, proxy_config):
        """When draft/target are unreachable, status shows not alive."""
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.get("/v1/tightwad/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft"]["model"] == "qwen3-8b"
        assert data["target"]["model"] == "qwen3-32b"
        assert data["draft"]["health"]["alive"] is False
        assert data["target"]["health"]["alive"] is False
        assert data["stats"]["total_rounds"] == 0
        assert data["stats"]["acceptance_rate"] == 0.0

    def test_completion_draft_unavailable_fallback(self, proxy_config):
        """When both servers are down, request should fail gracefully."""
        app = create_app(proxy_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/completions", json={
            "prompt": "Hello",
            "max_tokens": 10,
        })
        # Should get a 500 since target is also unreachable
        assert resp.status_code == 500


class TestLogprobsVerification:
    @pytest.fixture
    def llamacpp_config(self):
        return ProxyConfig(
            draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b", backend="llamacpp"),
            target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b", backend="llamacpp"),
            host="0.0.0.0",
            port=8088,
            max_draft_tokens=4,
        )

    def test_can_use_logprobs_llamacpp(self, llamacpp_config):
        proxy = SpeculativeProxy(llamacpp_config)
        assert proxy._can_use_logprobs() is True

    def test_cannot_use_logprobs_ollama(self):
        config = ProxyConfig(
            draft=ServerEndpoint(url="http://draft:11434", model_name="qwen3-8b", backend="ollama"),
            target=ServerEndpoint(url="http://target:11434", model_name="qwen3-32b", backend="ollama"),
        )
        proxy = SpeculativeProxy(config)
        assert proxy._can_use_logprobs() is False

    def test_cannot_use_logprobs_mixed(self):
        config = ProxyConfig(
            draft=ServerEndpoint(url="http://draft:11434", model_name="qwen3-8b", backend="ollama"),
            target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b", backend="llamacpp"),
        )
        proxy = SpeculativeProxy(config)
        assert proxy._can_use_logprobs() is False

    @pytest.mark.asyncio
    async def test_verify_with_logprobs_all_accepted(self, llamacpp_config):
        """When target agrees with all draft tokens, get accepted + bonus."""
        proxy = SpeculativeProxy(llamacpp_config)

        draft_tokens = [
            DraftToken(token_id=100, logprob=-0.1, text="The"),
            DraftToken(token_id=200, logprob=-0.2, text=" answer"),
            DraftToken(token_id=300, logprob=-0.15, text=" is"),
        ]

        # Target generates bonus token after verifying draft via prompt-append
        completion_resp = MagicMock()
        completion_resp.status_code = 200
        completion_resp.raise_for_status = MagicMock()
        completion_resp.json.return_value = {
            "choices": [{"text": " 42", "logprobs": {"content": [
                {"id": 400, "token": " 42", "logprob": -0.3},
            ]}}]
        }

        # Both tokenizers return matching IDs (same-family models)
        tokenize_resp = MagicMock()
        tokenize_resp.status_code = 200
        tokenize_resp.raise_for_status = MagicMock()
        tokenize_resp.json.return_value = {"tokens": [100, 200, 300]}

        async def target_side_effect(url, **kwargs):
            if "/tokenize" in url:
                return tokenize_resp
            return completion_resp

        proxy.target_client.post = AsyncMock(side_effect=target_side_effect)
        proxy.draft_client.post = AsyncMock(return_value=tokenize_resp)

        result = await proxy.verify_with_logprobs("prompt text", draft_tokens, temperature=0.0)

        assert result.accepted_count == 3
        assert result.rejected_at is None
        assert result.bonus_token is not None
        assert result.bonus_token.token_id == 400

        await proxy.close()

    @pytest.mark.asyncio
    async def test_verify_with_logprobs_partial_accept(self, llamacpp_config):
        """When target disagrees at position 2, accept first 2 and resample."""
        proxy = SpeculativeProxy(llamacpp_config)

        draft_tokens = [
            DraftToken(token_id=100, logprob=-0.1, text="The"),
            DraftToken(token_id=200, logprob=-0.2, text=" answer"),
            DraftToken(token_id=300, logprob=-0.15, text=" is"),
        ]

        # Prompt-append completion response (used first)
        completion_resp = MagicMock()
        completion_resp.status_code = 200
        completion_resp.raise_for_status = MagicMock()
        completion_resp.json.return_value = {
            "choices": [{"text": "", "logprobs": {"content": []}}]
        }

        # Target tokenizer returns different IDs — triggers fallback path
        target_tok_resp = MagicMock()
        target_tok_resp.status_code = 200
        target_tok_resp.raise_for_status = MagicMock()
        target_tok_resp.json.return_value = {"tokens": [100, 200, 999]}

        draft_tok_resp = MagicMock()
        draft_tok_resp.status_code = 200
        draft_tok_resp.raise_for_status = MagicMock()
        draft_tok_resp.json.return_value = {"tokens": [100, 200, 300]}

        # Fallback: target generates N+1 tokens from base prompt
        fallback_resp = MagicMock()
        fallback_resp.status_code = 200
        fallback_resp.raise_for_status = MagicMock()
        fallback_resp.json.return_value = {
            "choices": [{"logprobs": {"content": [
                {"id": 100, "token": "The", "logprob": -0.05},
                {"id": 200, "token": " answer", "logprob": -0.1},
                {"id": 999, "token": " was", "logprob": -0.08},
                {"id": 400, "token": " 42", "logprob": -0.3},
            ]}}]
        }

        # target_client.post is called 3 times:
        # 1) /v1/completions (prompt-append), 2) /tokenize, 3) /v1/completions (fallback)
        target_calls = iter([completion_resp, target_tok_resp, fallback_resp])

        async def target_side_effect(url, **kwargs):
            return next(target_calls)

        proxy.target_client.post = AsyncMock(side_effect=target_side_effect)
        proxy.draft_client.post = AsyncMock(return_value=draft_tok_resp)

        result = await proxy.verify_with_logprobs("prompt", draft_tokens, temperature=0.0)

        assert result.accepted_count == 2
        assert result.rejected_at == 2
        assert result.resample_token is not None
        assert result.resample_token.token_id == 999
        assert result.resample_token.text == " was"

        await proxy.close()


class TestSSEFormat:
    def test_sse_chunk_format(self):
        """SSE data lines should be valid JSON."""
        line = 'data: {"id":"cmpl-tightwad-1","object":"text_completion","choices":[{"index":0,"text":"hello","finish_reason":null}]}'
        assert line.startswith("data: ")
        payload = json.loads(line[6:])
        assert payload["object"] == "text_completion"
        assert payload["choices"][0]["text"] == "hello"

    def test_sse_done_marker(self):
        line = "data: [DONE]"
        assert line == "data: [DONE]"
