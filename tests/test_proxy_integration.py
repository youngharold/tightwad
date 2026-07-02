"""Integration tests for SpeculativeProxy.speculation_round() and generate_completion()."""

from __future__ import annotations

import asyncio
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

        accepted_text, is_done, draft_ms, verify_ms, r_drafted, r_accepted = await proxy.speculation_round(
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

        accepted_text, is_done, draft_ms, verify_ms, r_drafted, r_accepted = await proxy.speculation_round(
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

        accepted_text, is_done, draft_ms, verify_ms, r_drafted, r_accepted = await proxy.speculation_round(
            "prompt", temperature=0.0,
        )

        # Text-match path returns target_text
        assert accepted_text == "The answer is 42"
        assert is_done is False
        assert proxy.stats.total_rounds == 1
        # In text-match mode the counters get token ESTIMATES derived from
        # char counts (~4 chars/token), not raw chars — see _est_tokens.
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

        accepted_text, is_done, draft_ms, verify_ms, r_drafted, r_accepted = await proxy.speculation_round(
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

        accepted_text, is_done, draft_ms, verify_ms, r_drafted, r_accepted = await proxy.speculation_round(
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

        accepted_text, is_done, draft_ms, verify_ms, r_drafted, r_accepted = await proxy.speculation_round(
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


# ---------------------------------------------------------------------------
# Streaming stop-sequence handling
# ---------------------------------------------------------------------------


class TestStreamingStopSequences:
    """Streamed output must match non-streamed output when a stop sequence
    lands mid-chunk (regression: the final round's pre-stop text was
    dropped from the SSE stream)."""

    def _mock_rounds(self, proxy, rounds):
        seq = list(rounds)

        async def fake_round(prompt, temperature=0.0):
            if seq:
                return seq.pop(0), False, 0.0, 0.0, 2, 2
            return "", True, 0.0, 0.0, 0, 0

        proxy.speculation_round = fake_round

    async def _collect_sse_text(self, sse_gen):
        text = ""
        async for event in sse_gen:
            for line in event.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    payload = json.loads(line[len("data: "):])
                    text += payload["choices"][0]["text"]
        return text

    @pytest.mark.asyncio
    async def test_streaming_emits_pre_stop_text(self):
        rounds = ["Hello wor", "ld STOP tail"]

        proxy = SpeculativeProxy(_ollama_config())
        self._mock_rounds(proxy, rounds)
        streamed = await self._collect_sse_text(
            await proxy.generate_completion(
                prompt="p", max_tokens=100, temperature=0.0,
                stream=True, stop=["STOP"],
            )
        )
        await proxy.close()

        proxy = SpeculativeProxy(_ollama_config())
        self._mock_rounds(proxy, rounds)
        result = await proxy.generate_completion(
            prompt="p", max_tokens=100, temperature=0.0,
            stream=False, stop=["STOP"],
        )
        await proxy.close()

        assert result["choices"][0]["text"] == "Hello world "
        assert streamed == "Hello world "

    @pytest.mark.asyncio
    async def test_streaming_stop_spanning_chunk_boundary(self):
        """Stop match starting in an already-streamed chunk must not raise
        and must not emit anything past the stop."""
        rounds = ["Hello ST", "OP tail"]

        proxy = SpeculativeProxy(_ollama_config())
        self._mock_rounds(proxy, rounds)
        streamed = await self._collect_sse_text(
            await proxy.generate_completion(
                prompt="p", max_tokens=100, temperature=0.0,
                stream=True, stop=["STOP"],
            )
        )
        await proxy.close()

        assert "OP tail" not in streamed
        assert streamed.startswith("Hello ")


# ---------------------------------------------------------------------------
# draft_tokens_parallel edge cases
# ---------------------------------------------------------------------------


class TestDraftTokensParallel:
    """Regression: asyncio.wait() was called on an empty pending set when
    every drafter resolved in the first wait (single drafter, fast
    failures, or same-cycle completion), raising ValueError."""

    def _proxy_with_drafters(self, endpoints):
        cfg = _llamacpp_config(drafters=list(endpoints))
        return SpeculativeProxy(cfg)

    @pytest.mark.asyncio
    async def test_single_drafter_does_not_crash(self):
        ep = ServerEndpoint(
            url="http://d1:8081", model_name="d1", backend="llamacpp",
        )
        proxy = self._proxy_with_drafters([ep])
        tokens = [DraftToken(token_id=1, logprob=-0.1, text="hi")]

        async def fake_draft(endpoint, client, prompt, n, temperature):
            return tokens

        proxy._draft_from_endpoint = fake_draft
        result = await proxy.draft_tokens_parallel("p", 4)
        await proxy.close()
        assert result == tokens

    @pytest.mark.asyncio
    async def test_one_drafter_down_does_not_crash(self):
        eps = [
            ServerEndpoint(url="http://d1:8081", model_name="d1", backend="llamacpp"),
            ServerEndpoint(url="http://d2:8081", model_name="d2", backend="llamacpp"),
        ]
        proxy = self._proxy_with_drafters(eps)
        tokens = [DraftToken(token_id=1, logprob=-0.1, text="hi")]

        async def fake_draft(endpoint, client, prompt, n, temperature):
            if endpoint.model_name == "d1":
                raise ConnectionError("connection refused")
            return tokens

        proxy._draft_from_endpoint = fake_draft
        result = await proxy.draft_tokens_parallel("p", 4)
        await proxy.close()
        assert result == tokens

    @pytest.mark.asyncio
    async def test_all_drafters_down_returns_empty(self):
        eps = [
            ServerEndpoint(url="http://d1:8081", model_name="d1", backend="llamacpp"),
        ]
        proxy = self._proxy_with_drafters(eps)

        async def fake_draft(endpoint, client, prompt, n, temperature):
            raise ConnectionError("connection refused")

        proxy._draft_from_endpoint = fake_draft
        result = await proxy.draft_tokens_parallel("p", 4)
        await proxy.close()
        assert result == []


# ---------------------------------------------------------------------------
# Chat streaming SSE format
# ---------------------------------------------------------------------------


class TestChatStreamAdapter:
    """Regression: chat streaming re-emitted raw text_completion events, so
    OpenAI-compatible clients reading choices[0].delta.content saw nothing."""

    @pytest.mark.asyncio
    async def test_adapter_emits_chat_completion_chunks(self):
        from tightwad.proxy import _adapt_chat_stream

        proxy = SpeculativeProxy(_ollama_config())
        seq = ["Hello", " world"]

        async def fake_round(prompt, temperature=0.0):
            if seq:
                return seq.pop(0), False, 0.0, 0.0, 2, 2
            return "", True, 0.0, 0.0, 0, 0

        proxy.speculation_round = fake_round
        sse = await proxy.generate_completion(
            prompt="p", max_tokens=100, temperature=0.0, stream=True,
        )

        events = [e async for e in _adapt_chat_stream(sse, "target-model")]
        await proxy.close()

        assert events[-1] == "data: [DONE]\n\n"
        payloads = [json.loads(e[len("data: "):]) for e in events[:-1]]
        assert all(p["object"] == "chat.completion.chunk" for p in payloads)
        assert all(p["model"] == "target-model" for p in payloads)
        assert all(p["id"].startswith("chatcmpl-") for p in payloads)
        # Role delta on the first chunk only
        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        assert all("role" not in p["choices"][0]["delta"] for p in payloads[1:])
        # Content deltas reassemble the full text
        text = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
        assert text == "Hello world"
        # Final chunk: empty delta + finish_reason, no earlier finish_reason
        assert payloads[-1]["choices"][0]["delta"] == {}
        assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
        assert all(p["choices"][0]["finish_reason"] is None for p in payloads[:-1])


# ---------------------------------------------------------------------------
# Generation progress guarantee
# ---------------------------------------------------------------------------


class TestProgressGuarantee:
    """Regression: whitespace-only chunks split() to [], contributing 0 to
    tokens_generated, so a model stuck on newlines looped forever."""

    def _newline_proxy(self):
        proxy = SpeculativeProxy(_ollama_config())

        async def newline_round(prompt, temperature=0.0):
            return "\n", False, 0.0, 0.0, 1, 1

        proxy.speculation_round = newline_round
        return proxy

    @pytest.mark.asyncio
    async def test_non_streaming_terminates_on_whitespace_rounds(self):
        proxy = self._newline_proxy()
        result = await proxy.generate_completion(
            prompt="p", max_tokens=5, temperature=0.0, stream=False,
        )
        await proxy.close()

        assert result["choices"][0]["text"] == "\n" * 5
        assert result["choices"][0]["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_streaming_terminates_on_whitespace_rounds(self):
        proxy = self._newline_proxy()
        sse = await proxy.generate_completion(
            prompt="p", max_tokens=5, temperature=0.0, stream=True,
        )
        events = [e async for e in sse]
        await proxy.close()

        assert events[-1] == "data: [DONE]\n\n"
        # 5 content chunks + final finish_reason chunk + [DONE]
        assert len(events) == 7

    @pytest.mark.asyncio
    async def test_pipelined_terminates_on_whitespace_rounds(self):
        proxy = SpeculativeProxy(_ollama_config())

        async def fake_draft(prompt, n, temperature=0.0):
            return [DraftToken(token_id=0, logprob=0.0, text="\n")]

        async def fake_verify(prompt, draft_text, temperature=0.0):
            return "\n", 1, 1

        proxy.draft_tokens = fake_draft
        proxy.verify_text_match = fake_verify
        generated = await proxy.pipelined_generate("p", max_tokens=5)
        await proxy.close()

        assert generated == "\n" * 5


# ---------------------------------------------------------------------------
# Per-request stats isolation
# ---------------------------------------------------------------------------


class TestPerRequestStats:
    """Regression: per-request drafted/accepted were computed by diffing the
    shared ProxyStats totals, so concurrent requests contaminated each
    other's RequestRecord history."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_do_not_cross_contaminate(self):
        proxy = SpeculativeProxy(_ollama_config())
        rounds_done = {"A": 0, "B": 0}
        per_round = {"A": (10, 8, "aaaa "), "B": (2, 1, "bb ")}

        async def fake_round(prompt, temperature=0.0):
            key = prompt[0]
            # Yield to the event loop so the two requests interleave
            await asyncio.sleep(0)
            if rounds_done[key] >= 3:
                return "", True, 0.0, 0.0, 0, 0
            rounds_done[key] += 1
            drafted, accepted, text = per_round[key]
            # The real speculation_round also bumps the shared totals
            proxy.stats.total_drafted += drafted
            proxy.stats.total_accepted += accepted
            return text, False, 0.0, 0.0, drafted, accepted

        proxy.speculation_round = fake_round
        await asyncio.gather(
            proxy.generate_completion(prompt="A", max_tokens=100, stream=False),
            proxy.generate_completion(prompt="B", max_tokens=100, stream=False),
        )
        await proxy.close()

        by_drafted = {r.drafted: r for r in proxy.stats.request_history}
        assert set(by_drafted) == {30, 6}, (
            "Per-request counts must be request-local, not diffs of the "
            "shared ProxyStats totals."
        )
        assert by_drafted[30].accepted == 24
        assert by_drafted[6].accepted == 3


# ---------------------------------------------------------------------------
# Text-match token-estimate counters
# ---------------------------------------------------------------------------


class TestTextMatchTokenEstimates:
    """Regression: text-match paths added raw CHAR counts to the token
    counters, inflating /metrics ~4-5x on Ollama deployments."""

    @pytest.mark.asyncio
    async def test_counters_use_token_estimates_not_chars(self):
        proxy = SpeculativeProxy(_ollama_config())
        # 16 chars of matching text ≈ 4 tokens
        draft_resp = _make_response(json_data={"response": "The answer is 42"})
        target_resp = _make_response(json_data={"response": "The answer is 42"})
        proxy.draft_client.post = AsyncMock(return_value=draft_resp)
        proxy.target_client.post = AsyncMock(return_value=target_resp)

        text, done, d_ms, v_ms, r_drafted, r_accepted = await proxy.speculation_round(
            "prompt", temperature=0.0,
        )
        await proxy.close()

        assert proxy.stats.total_drafted == 4
        assert proxy.stats.total_accepted == 4
        assert proxy.stats.total_tokens_output == 4
        assert (r_drafted, r_accepted) == (4, 4)


# ---------------------------------------------------------------------------
# Consensus fallback draft selection
# ---------------------------------------------------------------------------


class TestConsensusFallbackDraftSelection:
    """Regression: after a consensus disagreement the fallback verified the
    LONGEST draft even when its prefix lost the vote, so the remaining
    tokens were conditioned on a different prefix than the trusted one."""

    def _consensus_proxy(self):
        config = _ollama_config(
            consensus_mode="majority",
            drafters=[
                ServerEndpoint(url="http://d1:11434", model_name="d1", backend="ollama"),
                ServerEndpoint(url="http://d2:11434", model_name="d2", backend="ollama"),
                ServerEndpoint(url="http://d3:11434", model_name="d3", backend="ollama"),
            ],
        )
        return SpeculativeProxy(config)

    @staticmethod
    def _tok(token_id, text):
        return DraftToken(token_id=token_id, logprob=-0.1, text=text)

    @pytest.mark.asyncio
    async def test_picks_longest_prefix_consistent_draft(self):
        proxy = self._consensus_proxy()
        # Majority accepts [x, y] (from a/b); c is longest but lost at pos 0.
        a = [self._tok(1, "x"), self._tok(2, "y"), self._tok(3, "a")]
        b = [self._tok(1, "x"), self._tok(2, "y"), self._tok(4, "b")]
        c = [self._tok(9, "q"), self._tok(8, "z"),
             self._tok(7, "w"), self._tok(7, "w"), self._tok(7, "w")]

        async def fake_all(prompt, n, temperature=0.0):
            return [a, b, c]

        captured = {}

        async def fake_verify(prompt, draft_text, temperature=0.0):
            captured["prompt"] = prompt
            captured["draft_text"] = draft_text
            return draft_text, len(draft_text), len(draft_text)

        proxy.draft_tokens_all = fake_all
        proxy.verify_text_match = fake_verify

        await proxy.speculation_round("PROMPT", temperature=0.0)
        await proxy.close()

        # Remaining draft must come from a prefix-consistent draft (a or b),
        # never from c's post-"qz" tail.
        assert captured["prompt"] == "PROMPT" + "xy"
        assert captured["draft_text"] in ("a", "b")

    @pytest.mark.asyncio
    async def test_no_prefix_consistent_draft_falls_back_to_full_verify(self):
        proxy = self._consensus_proxy()
        # Majority prefix zigzags: pos0 from (a,b), pos1 from (b,c),
        # pos2 from (a,c) — no single draft contains [x, y, z].
        a = [self._tok(1, "x"), self._tok(8, "q"), self._tok(3, "z"), self._tok(4, "1")]
        b = [self._tok(1, "x"), self._tok(2, "y"), self._tok(7, "w"), self._tok(5, "2")]
        c = [self._tok(9, "m"), self._tok(2, "y"), self._tok(3, "z"), self._tok(6, "3")]

        async def fake_all(prompt, n, temperature=0.0):
            return [a, b, c]

        captured = {}

        async def fake_verify(prompt, draft_text, temperature=0.0):
            captured["prompt"] = prompt
            captured["draft_text"] = draft_text
            return draft_text, len(draft_text), len(draft_text)

        proxy.draft_tokens_all = fake_all
        proxy.verify_text_match = fake_verify

        await proxy.speculation_round("PROMPT", temperature=0.0)
        await proxy.close()

        # Prefix distrusted — full draft verified from the ORIGINAL prompt.
        assert captured["prompt"] == "PROMPT"
        assert captured["draft_text"] in ("xqz1", "xyw2", "myz3")
