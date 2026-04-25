"""Integration tests for SpeculativeProxy.speculation_round() and generate_completion()."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tightwad.config import ProxyConfig, ServerEndpoint
from tightwad.proxy import SpeculativeProxy
from tightwad.speculation import DraftToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int = 200, json_data: dict | None = None):
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data or {}
    return resp


def _llamacpp_config(**overrides) -> ProxyConfig:
    """ProxyConfig with llamacpp backends."""
    defaults = dict(
        draft=ServerEndpoint(url="http://draft:8081", model_name="draft-model", backend="llamacpp"),
        target=ServerEndpoint(url="http://target:8080", model_name="target-model", backend="llamacpp"),
        host="127.0.0.1",
        port=8088,
        max_draft_tokens=4,
        fallback_on_draft_failure=True,
    )
    defaults.update(overrides)
    return ProxyConfig(**defaults)


def _ollama_config(**overrides) -> ProxyConfig:
    """ProxyConfig with Ollama backends."""
    defaults = dict(
        draft=ServerEndpoint(url="http://draft:11434", model_name="draft-model", backend="ollama"),
        target=ServerEndpoint(url="http://target:11434", model_name="target-model", backend="ollama"),
        host="127.0.0.1",
        port=8088,
        max_draft_tokens=4,
        fallback_on_draft_failure=True,
    )
    defaults.update(overrides)
    return ProxyConfig(**defaults)


# ---------------------------------------------------------------------------
# Logprobs-based speculation_round (llamacpp + llamacpp)
# ---------------------------------------------------------------------------


class TestSpeculationRoundLogprobs:
    """Tests for speculation_round() when both backends support logprobs."""

    @pytest.mark.asyncio
    async def test_speculation_round_logprobs_all_accepted(self):
        """Draft returns 3 tokens, target argmax matches all -> 3 accepted + bonus."""
        proxy = SpeculativeProxy(_llamacpp_config())
        proxy._same_family = True

        # Draft response: 3 tokens with logprobs
        draft_resp = _make_response(json_data={
            "choices": [{"text": "The answer is", "logprobs": {"content": [
                {"id": 100, "token": "The", "logprob": -0.1},
                {"id": 200, "token": " answer", "logprob": -0.2},
                {"id": 300, "token": " is", "logprob": -0.15},
            ]}}],
        })
        proxy.draft_client.post = AsyncMock(return_value=draft_resp)

        # Per-position verify: target generates N+1 tokens; argmax matches draft
        target_resp = _make_response(json_data={
            "choices": [{"logprobs": {"content": [
                {"id": 100, "token": "The", "logprob": -0.05},
                {"id": 200, "token": " answer", "logprob": -0.10},
                {"id": 300, "token": " is", "logprob": -0.12},
                {"id": 400, "token": " 42", "logprob": -0.30},
            ]}}],
        })
        proxy.target_client.post = AsyncMock(return_value=target_resp)

        accepted_text, is_done, draft_ms, verify_ms = await proxy.speculation_round(
            "What is the meaning of life?", temperature=0.0,
        )

        assert "The" in accepted_text
        assert " answer" in accepted_text
        assert " is" in accepted_text
        assert " 42" in accepted_text
        assert is_done is False
        assert proxy.stats.total_rounds == 1
        assert proxy.stats.total_accepted == 3
        assert proxy.stats.total_bonus == 1

        await proxy.close()

    @pytest.mark.asyncio
    async def test_speculation_round_logprobs_partial_reject(self):
        """Draft returns 3 tokens, target argmax disagrees at position 2 -> accept 2 + resample."""
        proxy = SpeculativeProxy(_llamacpp_config())
        proxy._same_family = True

        # Draft response: 3 tokens
        draft_resp = _make_response(json_data={
            "choices": [{"text": "The answer is", "logprobs": {"content": [
                {"id": 100, "token": "The", "logprob": -0.1},
                {"id": 200, "token": " answer", "logprob": -0.2},
                {"id": 300, "token": " is", "logprob": -0.15},
            ]}}],
        })
        proxy.draft_client.post = AsyncMock(return_value=draft_resp)

        # Target argmax disagrees at position 2 (999 instead of 300).
        target_resp = _make_response(json_data={
            "choices": [{"logprobs": {"content": [
                {"id": 100, "token": "The", "logprob": -0.05},
                {"id": 200, "token": " answer", "logprob": -0.1},
                {"id": 999, "token": " was", "logprob": -0.08},
                {"id": 400, "token": " 42", "logprob": -0.3},
            ]}}],
        })
        proxy.target_client.post = AsyncMock(return_value=target_resp)

        accepted_text, is_done, draft_ms, verify_ms = await proxy.speculation_round(
            "What is the meaning of life?", temperature=0.0,
        )

        assert "The" in accepted_text
        assert " answer" in accepted_text
        assert " was" in accepted_text  # resampled token
        assert is_done is False
        assert proxy.stats.total_rounds == 1
        assert proxy.stats.total_accepted == 2
        assert proxy.stats.total_resampled == 1

        await proxy.close()


# ---------------------------------------------------------------------------
# Text-match speculation_round (Ollama backends)
# ---------------------------------------------------------------------------


class TestSpeculationRoundTextMatch:
    """Tests for speculation_round() with Ollama text-match verification."""

    @pytest.mark.asyncio
    async def test_speculation_round_text_match(self):
        """Draft and target produce matching text -> accepted via text-match."""
        proxy = SpeculativeProxy(_ollama_config())

        # Draft returns a text blob (Ollama style - single DraftToken)
        draft_resp = _make_response(json_data={
            "response": "The answer is 42",
        })

        # Target returns matching text (same prefix verifies)
        target_resp = _make_response(json_data={
            "response": "The answer is 42",
        })

        proxy.draft_client.post = AsyncMock(return_value=draft_resp)
        proxy.target_client.post = AsyncMock(return_value=target_resp)

        accepted_text, is_done, draft_ms, verify_ms = await proxy.speculation_round(
            "prompt", temperature=0.0,
        )

        # Text-match path returns target_text
        assert accepted_text == "The answer is 42"
        assert is_done is False
        assert proxy.stats.total_rounds == 1
        # In text-match mode, draft_len is char count, accepted is match_len
        assert proxy.stats.total_drafted > 0
        assert proxy.stats.total_accepted > 0

        await proxy.close()


# ---------------------------------------------------------------------------
# Draft failure fallback
# ---------------------------------------------------------------------------


class TestSpeculationRoundDraftFailure:

    @pytest.mark.asyncio
    async def test_speculation_round_draft_failure_fallback(self):
        """Draft raises exception -> falls back to target when fallback_on_draft_failure=True."""
        proxy = SpeculativeProxy(_ollama_config(fallback_on_draft_failure=True))

        # Draft raises an error
        proxy.draft_client.post = AsyncMock(
            side_effect=Exception("draft server connection refused"),
        )

        # Target works fine
        target_resp = _make_response(json_data={
            "response": "Direct target response",
        })
        proxy.target_client.post = AsyncMock(return_value=target_resp)

        accepted_text, is_done, draft_ms, verify_ms = await proxy.speculation_round(
            "prompt", temperature=0.0,
        )

        assert accepted_text == "Direct target response"
        assert is_done is False
        # Draft failed, so draft_ms should be 0
        assert draft_ms == 0.0
        assert verify_ms == 0.0

        await proxy.close()


# ---------------------------------------------------------------------------
# Consensus mode tests
# ---------------------------------------------------------------------------


class TestSpeculationRoundConsensus:

    @pytest.mark.asyncio
    async def test_speculation_round_consensus_all_agree(self):
        """Two drafters agree on all tokens -> accepted without target call."""
        config = _ollama_config(
            consensus_mode="strict",
            drafters=[
                ServerEndpoint(url="http://drafter1:11434", model_name="d1", backend="ollama"),
                ServerEndpoint(url="http://drafter2:11434", model_name="d2", backend="ollama"),
            ],
        )
        proxy = SpeculativeProxy(config)

        # Both drafters return the same text
        drafter_resp = _make_response(json_data={"response": "Hello world"})
        for _, client in proxy.draft_clients:
            client.post = AsyncMock(return_value=drafter_resp)

        # Target should NOT be called
        proxy.target_client.post = AsyncMock(
            side_effect=AssertionError("Target should not be called in consensus mode"),
        )

        accepted_text, is_done, draft_ms, verify_ms = await proxy.speculation_round(
            "prompt", temperature=0.0,
        )

        assert accepted_text == "Hello world"
        assert is_done is False
        assert verify_ms == 0.0  # no target verification needed
        assert proxy.stats.consensus_accepted == 1
        assert proxy.stats.consensus_fallback == 0
        proxy.target_client.post.assert_not_called()

        await proxy.close()

    @pytest.mark.asyncio
    async def test_speculation_round_consensus_disagree(self):
        """Two drafters disagree -> falls through to target verification.

        Uses llamacpp backends so drafters return per-token logprobs with
        distinct token_ids.  Ollama drafters always return token_id=0
        which would cause false consensus.
        """
        config = _llamacpp_config(
            consensus_mode="strict",
            drafters=[
                ServerEndpoint(url="http://drafter1:8081", model_name="d1", backend="llamacpp"),
                ServerEndpoint(url="http://drafter2:8082", model_name="d2", backend="llamacpp"),
            ],
        )
        proxy = SpeculativeProxy(config)

        # Drafter 1 returns token_id 100
        drafter1_resp = _make_response(json_data={
            "choices": [{"text": "Hello", "logprobs": {"content": [
                {"id": 100, "token": "Hello", "logprob": -0.1},
            ]}}],
        })
        # Drafter 2 returns token_id 200 (different!)
        drafter2_resp = _make_response(json_data={
            "choices": [{"text": "Goodbye", "logprobs": {"content": [
                {"id": 200, "token": "Goodbye", "logprob": -0.2},
            ]}}],
        })

        proxy.draft_clients[0][1].post = AsyncMock(return_value=drafter1_resp)
        proxy.draft_clients[1][1].post = AsyncMock(return_value=drafter2_resp)

        # Target is called for verification since consensus failed.
        # The proxy picks the longest drafter output and verifies via logprobs.
        # Since both have 1 token, it picks whichever max() returns.
        # We need the target to return a prompt-append response and tokenize matches.
        proxy._same_family = True  # skip tokenize round trips
        target_resp = _make_response(json_data={
            "choices": [{"text": " world", "logprobs": {"content": [
                {"id": 300, "token": " world", "logprob": -0.1},
            ]}}],
        })
        proxy.target_client.post = AsyncMock(return_value=target_resp)

        accepted_text, is_done, draft_ms, verify_ms = await proxy.speculation_round(
            "prompt", temperature=0.0,
        )

        # Should fall through to target verification
        assert accepted_text is not None
        assert proxy.stats.consensus_fallback == 1
        assert proxy.target_client.post.called

        await proxy.close()


# ---------------------------------------------------------------------------
# generate_completion (non-streaming)
# ---------------------------------------------------------------------------


class TestGenerateCompletion:

    @pytest.mark.asyncio
    async def test_generate_completion_non_streaming(self):
        """Full generate_completion() returns valid response with mocked backends."""
        proxy = SpeculativeProxy(_ollama_config())

        # First round: draft + target both return some text
        draft_resp = _make_response(json_data={"response": "Hello"})
        target_resp = _make_response(json_data={"response": "Hello"})

        # Second round: return empty to signal EOS
        draft_resp_empty = _make_response(json_data={"response": ""})

        draft_call_count = {"n": 0}

        async def draft_side_effect(url, **kwargs):
            draft_call_count["n"] += 1
            if draft_call_count["n"] <= 1:
                return draft_resp
            return draft_resp_empty

        async def target_side_effect(url, **kwargs):
            return target_resp

        proxy.draft_client.post = AsyncMock(side_effect=draft_side_effect)
        proxy.target_client.post = AsyncMock(side_effect=target_side_effect)

        result = await proxy.generate_completion(
            prompt="Say hello",
            max_tokens=10,
            temperature=0.0,
            stream=False,
        )

        assert isinstance(result, dict)
        assert result["object"] == "text_completion"
        assert len(result["choices"]) == 1
        assert isinstance(result["choices"][0]["text"], str)
        assert result["choices"][0]["text"] != ""
        assert result["choices"][0]["finish_reason"] in ("stop", "length")

        await proxy.close()
