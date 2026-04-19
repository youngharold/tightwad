"""Tests for A/B benchmark module."""

import pytest

from tightwad.bench import (
    BenchmarkResult,
    RunResult,
    _median,
    _percentile,
    format_report,
    DEFAULT_PROMPTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestMedian:
    def test_odd(self):
        assert _median([1, 2, 3]) == 2

    def test_even(self):
        assert _median([1, 2, 3, 4]) == 2.5

    def test_single(self):
        assert _median([5]) == 5

    def test_empty(self):
        assert _median([]) == 0.0

    def test_unsorted(self):
        assert _median([3, 1, 2]) == 2


class TestPercentile:
    def test_p95(self):
        values = list(range(1, 101))  # 1..100
        assert _percentile(values, 95) == 96

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([42], 95) == 42


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


def _run(idx: int, tokens: int, elapsed: float) -> RunResult:
    return RunResult(
        prompt_index=idx,
        tokens=tokens,
        elapsed_s=elapsed,
        tok_per_s=tokens / elapsed if elapsed > 0 else 0,
    )


class TestBenchmarkResult:
    def test_speedup(self):
        result = BenchmarkResult(
            proxy_results=[_run(0, 100, 5.0), _run(1, 100, 5.0)],   # 20 tok/s
            direct_results=[_run(0, 100, 10.0), _run(1, 100, 10.0)],  # 10 tok/s
        )
        assert abs(result.speedup - 2.0) < 0.01

    def test_no_speedup(self):
        result = BenchmarkResult(
            proxy_results=[_run(0, 100, 10.0)],
            direct_results=[_run(0, 100, 10.0)],
        )
        assert abs(result.speedup - 1.0) < 0.01

    def test_slower_proxy(self):
        result = BenchmarkResult(
            proxy_results=[_run(0, 100, 20.0)],   # 5 tok/s
            direct_results=[_run(0, 100, 10.0)],   # 10 tok/s
        )
        assert result.speedup < 1.0

    def test_empty(self):
        result = BenchmarkResult()
        assert result.proxy_avg_tps == 0.0
        assert result.direct_avg_tps == 0.0
        assert result.speedup == 0.0

    def test_avg_tps(self):
        result = BenchmarkResult(
            proxy_results=[_run(0, 50, 5.0), _run(1, 150, 5.0)],  # 10, 30 → avg 20
        )
        assert abs(result.proxy_avg_tps - 20.0) < 0.01

    def test_to_dict(self):
        result = BenchmarkResult(
            proxy_results=[_run(0, 100, 5.0)],
            direct_results=[_run(0, 100, 10.0)],
        )
        d = result.to_dict()
        assert "proxy" in d
        assert "direct" in d
        assert "speedup" in d
        assert d["proxy"]["avg_tok_s"] == 20.0
        assert d["direct"]["avg_tok_s"] == 10.0
        assert d["speedup"] == 2.0


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_produces_output(self):
        result = BenchmarkResult(
            proxy_results=[_run(0, 100, 5.0)],
            direct_results=[_run(0, 100, 10.0)],
        )
        output = format_report(result)
        assert "A/B Benchmark" in output
        assert "2.00x" in output

    def test_with_proxy_stats(self):
        result = BenchmarkResult(
            proxy_results=[_run(0, 100, 5.0)],
            direct_results=[_run(0, 100, 10.0)],
            proxy_stats={"acceptance_rate": 0.73, "total_rounds": 42},
        )
        output = format_report(result)
        assert "73.0%" in output


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_prompts_not_empty(self):
        assert len(DEFAULT_PROMPTS) >= 5

    def test_prompts_are_strings(self):
        for p in DEFAULT_PROMPTS:
            assert isinstance(p, str)
            assert len(p) > 10


# ---------------------------------------------------------------------------
# MoE benchmark (async streaming)
# ---------------------------------------------------------------------------


class TestRunMoeBenchmark:
    def test_schema_and_streaming(self, monkeypatch):
        import asyncio
        import tightwad.bench as bench_module

        async def fake_stream(client, url, model, prompt, max_tokens, auth_token=None):
            # proxy URL → faster; direct URL → slower
            if "proxy" in url:
                return (100, 0.05, 1.0, "proxy says hi")
            return (100, 0.08, 2.0, "direct says hi")

        async def fake_get(self, url, timeout=10.0):
            class R:
                status_code = 200
                def json(self):
                    return {"stats": {"acceptance_rate": 0.72, "total_rounds": 42}}
            return R()

        monkeypatch.setattr(bench_module, "_chat_completion_stream", fake_stream)
        monkeypatch.setattr("httpx.AsyncClient.get", fake_get)

        updates: list[dict] = []

        async def run():
            return await bench_module.run_moe_benchmark(
                proxy_url="http://proxy:8088",
                target_url="http://target:1234",
                target_model="test-model",
                prompts=[{"name": "a", "text": "hi"},
                          {"name": "b", "text": "yo"}],
                max_tokens=32, warmup=0,
                on_update=updates.append,
            )

        result = asyncio.run(run())

        assert result["target"] == "test-model"
        assert result["avg_speedup"] == 2.0
        assert len(result["prompts"]) == 2
        first = result["prompts"][0]
        assert first["proxy"]["tokens"] == 100
        assert first["proxy"]["ttft_ms"] == 50.0
        assert first["direct"]["tok_s"] == 50.0
        assert result["moe"]["acceptance_rate"] == 0.72

        assert len(updates) == 2
        assert updates[0]["name"] == "a"
        assert updates[0]["speedup"] == 2.0
