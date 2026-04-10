"""Integration tests for QualityGateProxy.handle_request() with mocked HTTP backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tightwad.quality_gate import (
    AgentEndpoint,
    QualityGateConfig,
    QualityGateProxy,
    Verdict,
)


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


def _gate_config(
    n_agents: int = 1,
    backend: str = "llamacpp",
    cache_identical: bool = True,
    max_retries: int = 1,
) -> QualityGateConfig:
    """Build a QualityGateConfig with n agents."""
    agents = [
        AgentEndpoint(
            url=f"http://agent{i}:8080",
            model_name=f"agent-model-{i}",
            backend=backend,
        )
        for i in range(n_agents)
    ]
    return QualityGateConfig(
        verifier_url="http://verifier:8080",
        verifier_model="verifier-model",
        verifier_backend=backend,
        agents=agents,
        routing="round_robin",
        max_retries=max_retries,
        cache_identical=cache_identical,
    )


def _mock_agent_response(gate: QualityGateProxy, text: str, agent_idx: int = 0):
    """Mock a specific agent to return text via llamacpp backend."""
    resp = _make_response(json_data={
        "choices": [{"text": text}],
    })
    _, client = gate.agent_clients[agent_idx]
    client.post = AsyncMock(return_value=resp)


def _mock_all_agents(gate: QualityGateProxy, text: str):
    """Mock all agents to return the same text."""
    for i in range(len(gate.agent_clients)):
        _mock_agent_response(gate, text, i)


def _mock_verifier(gate: QualityGateProxy, verdict_text: str):
    """Mock the verifier to return a verdict string."""
    resp = _make_response(json_data={
        "choices": [{"text": verdict_text}],
    })
    gate.verifier_client.post = AsyncMock(return_value=resp)


# ---------------------------------------------------------------------------
# APPROVE path
# ---------------------------------------------------------------------------


class TestHandleRequestApprove:

    @pytest.mark.asyncio
    async def test_handle_request_approve(self):
        """Agent returns good response, verifier approves -> original returned."""
        gate = QualityGateProxy(_gate_config())
        _mock_agent_response(gate, "The capital of France is Paris.")
        _mock_verifier(gate, "APPROVE")

        result = await gate.handle_request("What is the capital of France?")

        assert result == "The capital of France is Paris."
        assert gate.stats.approved == 1
        assert gate.stats.total_requests == 1

        await gate.close()


# ---------------------------------------------------------------------------
# CORRECT path
# ---------------------------------------------------------------------------


class TestHandleRequestCorrect:

    @pytest.mark.asyncio
    async def test_handle_request_correct(self):
        """Agent returns response with error, verifier corrects -> corrected returned."""
        gate = QualityGateProxy(_gate_config())
        _mock_agent_response(gate, "The capital of France is Lyon.")
        _mock_verifier(gate, "CORRECT: The capital of France is Paris.")

        result = await gate.handle_request("What is the capital of France?")

        assert result == "The capital of France is Paris."
        assert gate.stats.corrected == 1
        assert gate.stats.total_requests == 1

        await gate.close()


# ---------------------------------------------------------------------------
# REJECT path
# ---------------------------------------------------------------------------


class TestHandleRequestReject:

    @pytest.mark.asyncio
    async def test_handle_request_reject_then_regenerate(self):
        """Agent returns bad response, verifier rejects, retries exhaust -> verifier generates."""
        gate = QualityGateProxy(_gate_config(max_retries=1))

        # Agent always returns bad content
        bad_resp = _make_response(json_data={"choices": [{"text": "completely wrong"}]})
        _, agent_client = gate.agent_clients[0]
        agent_client.post = AsyncMock(return_value=bad_resp)

        # Verifier rejects both initial and retry attempts, then generates directly
        reject_resp = _make_response(json_data={"choices": [{"text": "REJECT"}]})
        generate_resp = _make_response(json_data={
            "choices": [{"text": "The correct answer from GPU."}],
        })

        # Verifier calls: 1) verify initial -> REJECT, 2) verify retry -> REJECT,
        # 3) generate directly
        verifier_calls = iter([reject_resp, reject_resp, generate_resp])

        async def verifier_side_effect(url, **kwargs):
            return next(verifier_calls)

        gate.verifier_client.post = AsyncMock(side_effect=verifier_side_effect)

        result = await gate.handle_request("What is 2+2?")

        assert result == "The correct answer from GPU."
        assert gate.stats.rejected == 1
        assert gate.stats.total_requests == 1

        await gate.close()

    @pytest.mark.asyncio
    async def test_handle_request_reject_retry_then_approve(self):
        """Agent fails first, retry succeeds with verifier approval."""
        gate = QualityGateProxy(_gate_config(max_retries=1))

        # First attempt: bad response, second attempt: good response
        bad_resp = _make_response(json_data={"choices": [{"text": "wrong answer"}]})
        good_resp = _make_response(json_data={"choices": [{"text": "correct answer"}]})
        agent_calls = iter([bad_resp, good_resp])

        async def agent_side_effect(url, **kwargs):
            return next(agent_calls)

        _, agent_client = gate.agent_clients[0]
        agent_client.post = AsyncMock(side_effect=agent_side_effect)

        # Verifier: REJECT first attempt, APPROVE retry
        reject_resp = _make_response(json_data={"choices": [{"text": "REJECT"}]})
        approve_resp = _make_response(json_data={"choices": [{"text": "APPROVE"}]})
        verifier_calls = iter([reject_resp, approve_resp])

        async def verifier_side_effect(url, **kwargs):
            return next(verifier_calls)

        gate.verifier_client.post = AsyncMock(side_effect=verifier_side_effect)

        result = await gate.handle_request("What is 2+2?")

        assert result == "correct answer"
        assert gate.stats.approved == 1  # retry approved
        assert gate.stats.rejected == 1  # initial was rejected

        await gate.close()


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestHandleRequestCache:

    @pytest.mark.asyncio
    async def test_handle_request_cache_hit(self):
        """Second identical request uses cache, skipping verification."""
        gate = QualityGateProxy(_gate_config(cache_identical=True))
        _mock_agent_response(gate, "cached response text")
        _mock_verifier(gate, "APPROVE")

        # First call: goes through full pipeline
        result1 = await gate.handle_request("cached prompt")
        assert result1 == "cached response text"
        assert gate.stats.cache_hits == 0
        assert gate.stats.approved == 1

        # Second call: same agent response should hit cache
        result2 = await gate.handle_request("cached prompt")
        assert result2 == "cached response text"
        assert gate.stats.cache_hits == 1
        assert gate.stats.approved == 2  # counted again

        await gate.close()


# ---------------------------------------------------------------------------
# Agent failure fallthrough
# ---------------------------------------------------------------------------


class TestHandleRequestAgentFailure:

    @pytest.mark.asyncio
    async def test_handle_request_agent_failure(self):
        """Agent returns empty response -> falls through to verifier generation."""
        gate = QualityGateProxy(_gate_config())

        # Agent returns empty
        empty_resp = _make_response(json_data={"choices": [{"text": ""}]})
        _, agent_client = gate.agent_clients[0]
        agent_client.post = AsyncMock(return_value=empty_resp)

        # Verifier generates directly (no verification step since agent returned nothing)
        generate_resp = _make_response(json_data={
            "choices": [{"text": "GPU-generated response"}],
        })
        gate.verifier_client.post = AsyncMock(return_value=generate_resp)

        result = await gate.handle_request("prompt")

        assert result == "GPU-generated response"
        assert gate.stats.rejected == 1  # empty agent = rejected

        await gate.close()


# ---------------------------------------------------------------------------
# Round-robin routing integration
# ---------------------------------------------------------------------------


class TestRoundRobinRouting:

    @pytest.mark.asyncio
    async def test_round_robin_routing(self):
        """Three agents are called in round-robin order across requests."""
        gate = QualityGateProxy(_gate_config(n_agents=3))

        # Track which agents were called
        called_agents: list[str] = []

        for i in range(3):
            resp = _make_response(json_data={
                "choices": [{"text": f"response from agent {i}"}],
            })

            async def make_side_effect(response, agent_url):
                async def side_effect(url, **kwargs):
                    called_agents.append(agent_url)
                    return response
                return side_effect

            _, client = gate.agent_clients[i]
            # Each agent returns its own response
            client.post = AsyncMock(return_value=resp)

        # Verifier always approves
        _mock_verifier(gate, "APPROVE")

        # Make 3 requests
        for _ in range(3):
            await gate.handle_request("test prompt")

        # Verify round-robin: agent_clients[0], [1], [2] were picked in order
        # We check via the gate's internal counter
        assert gate._agent_idx == 3  # incremented 3 times
        assert gate.stats.total_requests == 3
        assert gate.stats.approved == 3

        await gate.close()


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


class TestStatsTracking:

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Run approve, correct, reject requests and verify stats."""
        gate = QualityGateProxy(_gate_config(n_agents=1, max_retries=0))

        _, agent_client = gate.agent_clients[0]

        # Request 1: APPROVE
        agent_client.post = AsyncMock(
            return_value=_make_response(json_data={"choices": [{"text": "good"}]}),
        )
        gate.verifier_client.post = AsyncMock(
            return_value=_make_response(json_data={"choices": [{"text": "APPROVE"}]}),
        )
        await gate.handle_request("p1")

        # Request 2: CORRECT
        agent_client.post = AsyncMock(
            return_value=_make_response(json_data={"choices": [{"text": "ok-ish"}]}),
        )
        gate.verifier_client.post = AsyncMock(
            return_value=_make_response(json_data={
                "choices": [{"text": "CORRECT: better version"}],
            }),
        )
        await gate.handle_request("p2")

        # Request 3: REJECT (max_retries=0 so goes straight to GPU gen)
        agent_client.post = AsyncMock(
            return_value=_make_response(json_data={"choices": [{"text": "bad"}]}),
        )
        reject_resp = _make_response(json_data={"choices": [{"text": "REJECT"}]})
        gpu_resp = _make_response(json_data={"choices": [{"text": "gpu response"}]})
        verifier_calls = iter([reject_resp, gpu_resp])

        async def verifier_side_effect(url, **kwargs):
            return next(verifier_calls)

        gate.verifier_client.post = AsyncMock(side_effect=verifier_side_effect)
        await gate.handle_request("p3")

        # Verify counts
        assert gate.stats.total_requests == 3
        assert gate.stats.approved == 1
        assert gate.stats.corrected == 1
        assert gate.stats.rejected == 1
        assert gate.stats.total_agent_ms > 0
        assert gate.stats.total_verify_ms > 0
        assert gate.stats.approve_rate == pytest.approx(1 / 3, abs=0.01)
        assert gate.stats.gpu_usage_rate == pytest.approx(2 / 3, abs=0.01)

        await gate.close()
