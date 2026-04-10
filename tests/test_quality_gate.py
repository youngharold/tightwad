"""Tests for the quality gate module."""

import pytest

from tightwad.quality_gate import (
    AgentEndpoint,
    GateStats,
    QualityGateConfig,
    QualityGateProxy,
    ResponseCache,
    Verdict,
    VerificationResult,
    parse_verdict,
)


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_approve(self):
        v, c = parse_verdict("APPROVE")
        assert v == Verdict.APPROVE
        assert c is None

    def test_approve_with_trailing(self):
        v, c = parse_verdict("APPROVE\nThe response looks good.")
        assert v == Verdict.APPROVE

    def test_approve_case_insensitive(self):
        v, c = parse_verdict("approve")
        assert v == Verdict.APPROVE

    def test_correct(self):
        v, c = parse_verdict("CORRECT: The capital of France is Paris, not Lyon.")
        assert v == Verdict.CORRECT
        assert c == "The capital of France is Paris, not Lyon."

    def test_correct_multiline(self):
        v, c = parse_verdict("CORRECT: Line one.\nLine two.\nLine three.")
        assert v == Verdict.CORRECT
        assert "Line one" in c
        assert "Line three" in c

    def test_correct_empty_body_becomes_approve(self):
        v, c = parse_verdict("CORRECT: ")
        assert v == Verdict.APPROVE

    def test_reject(self):
        v, c = parse_verdict("REJECT")
        assert v == Verdict.REJECT
        assert c is None

    def test_reject_case_insensitive(self):
        v, c = parse_verdict("reject")
        assert v == Verdict.REJECT

    def test_ambiguous_defaults_to_approve(self):
        v, c = parse_verdict("I think the response is mostly fine.")
        assert v == Verdict.APPROVE

    def test_contains_approve_keyword(self):
        v, c = parse_verdict("I would APPROVE this response.")
        assert v == Verdict.APPROVE

    def test_contains_reject_keyword(self):
        v, c = parse_verdict("I must REJECT this response due to errors.")
        assert v == Verdict.REJECT

    def test_whitespace_stripped(self):
        v, c = parse_verdict("  \n  APPROVE  \n  ")
        assert v == Verdict.APPROVE


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class TestResponseCache:
    def test_miss(self):
        cache = ResponseCache()
        assert cache.get("prompt", "response") is None

    def test_hit(self):
        cache = ResponseCache()
        cache.put("prompt", "response", Verdict.APPROVE)
        assert cache.get("prompt", "response") == Verdict.APPROVE

    def test_different_prompt_misses(self):
        cache = ResponseCache()
        cache.put("prompt1", "response", Verdict.APPROVE)
        assert cache.get("prompt2", "response") is None

    def test_different_response_misses(self):
        cache = ResponseCache()
        cache.put("prompt", "response1", Verdict.APPROVE)
        assert cache.get("prompt", "response2") is None

    def test_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.put("p1", "r1", Verdict.APPROVE)
        cache.put("p2", "r2", Verdict.CORRECT)
        cache.put("p3", "r3", Verdict.REJECT)
        # First entry should be evicted
        assert cache.get("p1", "r1") is None
        assert cache.get("p3", "r3") == Verdict.REJECT

    def test_lru_ordering(self):
        cache = ResponseCache(max_size=2)
        cache.put("p1", "r1", Verdict.APPROVE)
        cache.put("p2", "r2", Verdict.CORRECT)
        # Access p1 to make it recently used
        cache.get("p1", "r1")
        # Add p3 — should evict p2 (least recently used), not p1
        cache.put("p3", "r3", Verdict.REJECT)
        assert cache.get("p1", "r1") == Verdict.APPROVE
        assert cache.get("p2", "r2") is None


# ---------------------------------------------------------------------------
# GateStats
# ---------------------------------------------------------------------------


class TestGateStats:
    def test_approve_rate(self):
        s = GateStats(total_requests=10, approved=8)
        assert abs(s.approve_rate - 0.8) < 0.01

    def test_approve_rate_zero(self):
        s = GateStats()
        assert s.approve_rate == 0.0

    def test_gpu_usage_rate(self):
        s = GateStats(total_requests=10, corrected=2, rejected=1)
        assert abs(s.gpu_usage_rate - 0.3) < 0.01

    def test_gpu_usage_rate_zero(self):
        s = GateStats()
        assert s.gpu_usage_rate == 0.0


# ---------------------------------------------------------------------------
# QualityGateConfig
# ---------------------------------------------------------------------------


class TestQualityGateConfig:
    def test_defaults(self):
        cfg = QualityGateConfig(
            verifier_url="http://gpu:8080",
            verifier_model="llama-405b",
        )
        assert cfg.routing == "round_robin"
        assert cfg.max_retries == 1
        assert cfg.cache_identical is True
        assert cfg.port == 8088

    def test_with_agents(self):
        cfg = QualityGateConfig(
            verifier_url="http://gpu:8080",
            verifier_model="llama-405b",
            agents=[
                AgentEndpoint(url="http://cpu1:11434", model_name="qwen3:1.7b"),
                AgentEndpoint(url="http://cpu2:11434", model_name="qwen3:1.7b"),
            ],
        )
        assert len(cfg.agents) == 2


# ---------------------------------------------------------------------------
# QualityGateProxy agent selection
# ---------------------------------------------------------------------------


class TestAgentRouting:
    def test_round_robin(self):
        cfg = QualityGateConfig(
            verifier_url="http://gpu:8080",
            verifier_model="llama-405b",
            agents=[
                AgentEndpoint(url="http://cpu1:11434", model_name="m1"),
                AgentEndpoint(url="http://cpu2:11434", model_name="m2"),
                AgentEndpoint(url="http://cpu3:11434", model_name="m3"),
            ],
        )
        proxy = QualityGateProxy(cfg)
        urls = []
        for _ in range(6):
            agent, _ = proxy._pick_agent()
            urls.append(agent.url)

        # Should cycle: cpu1, cpu2, cpu3, cpu1, cpu2, cpu3
        assert urls == [
            "http://cpu1:11434", "http://cpu2:11434", "http://cpu3:11434",
            "http://cpu1:11434", "http://cpu2:11434", "http://cpu3:11434",
        ]

    def test_no_agents_raises(self):
        cfg = QualityGateConfig(
            verifier_url="http://gpu:8080",
            verifier_model="llama-405b",
            agents=[],
        )
        proxy = QualityGateProxy(cfg)
        with pytest.raises(RuntimeError, match="No agents"):
            proxy._pick_agent()
