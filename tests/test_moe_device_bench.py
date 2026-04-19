"""Tests for auto-measured device scores."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tightwad.config import GPU, Worker, ClusterConfig, ModelConfig
from tightwad.moe_device_bench import (
    LOCAL_BASELINE_SCORE,
    measure_device_scores,
)


def _config(coord_gpus: list[int], workers: list[tuple[str, int, int]]) -> ClusterConfig:
    return ClusterConfig(
        coordinator_host="0.0.0.0", coordinator_port=8090,
        coordinator_backend="cuda",
        coordinator_gpus=[GPU(name=f"GPU{i}", vram_gb=gb) for i, gb in enumerate(coord_gpus)],
        workers=[
            Worker(host=host, gpus=[GPU(name="g", vram_gb=gb, rpc_port=port)])
            for host, port, gb in workers
        ],
        models={"m": ModelConfig(name="m", path="/m.gguf")},
        coordinator_binary="llama-server", rpc_server_binary="rpc-server",
    )


def test_coordinator_gpus_get_local_baseline(tmp_path, monkeypatch):
    config = _config(coord_gpus=[16, 12], workers=[])
    scores = measure_device_scores(config, force=True, cache_path=tmp_path / "scores.json")

    assert scores["CUDA0"] == LOCAL_BASELINE_SCORE
    assert scores["CUDA1"] == LOCAL_BASELINE_SCORE


def test_unreachable_worker_scores_zero(tmp_path):
    # A port that nothing is listening on — 55555 chosen to be almost always free
    config = _config(coord_gpus=[16], workers=[("127.0.0.1", 55555, 8)])
    scores = measure_device_scores(
        config, force=True, cache_path=tmp_path / "scores.json",
    )
    assert scores["CUDA0"] == LOCAL_BASELINE_SCORE
    assert scores["RPC[127.0.0.1:55555]"] == 0.0


def test_reachable_worker_scores_finite(tmp_path, monkeypatch):
    import tightwad.moe_device_bench as dm
    monkeypatch.setattr(dm, "_measure_tcp_rtt",
                        lambda host, port, iterations=5, timeout=2.0: 5.0)

    config = _config(coord_gpus=[], workers=[("10.0.0.1", 50052, 8)])
    scores = measure_device_scores(
        config, force=True, cache_path=tmp_path / "scores.json",
    )
    assert scores["RPC[10.0.0.1:50052]"] == LOCAL_BASELINE_SCORE / 5.0


def test_cache_hit_skips_measurement(tmp_path, monkeypatch):
    import tightwad.moe_device_bench as dm

    cache = tmp_path / "scores.json"
    cache.write_text(json.dumps({
        "captured_at": time.time(),
        "scores": {"CUDA0": 42.0},
        "detail": {"CUDA0": {"rtt_ms": 0.0, "source": "local"}},
    }))

    called = {"n": 0}

    def boom(*args, **kwargs):
        called["n"] += 1
        return 0.0

    monkeypatch.setattr(dm, "_measure_tcp_rtt", boom)

    config = _config(coord_gpus=[16], workers=[("10.0.0.1", 50052, 8)])
    scores = measure_device_scores(config, cache_path=cache)

    assert scores == {"CUDA0": 42.0}
    assert called["n"] == 0


def test_force_rebypasses_cache(tmp_path):
    cache = tmp_path / "scores.json"
    cache.write_text(json.dumps({
        "captured_at": time.time(),
        "scores": {"CUDA0": 42.0},
        "detail": {},
    }))
    config = _config(coord_gpus=[16], workers=[])

    scores = measure_device_scores(config, force=True, cache_path=cache)
    assert scores["CUDA0"] == LOCAL_BASELINE_SCORE


def test_cache_ttl_expires(tmp_path):
    cache = tmp_path / "scores.json"
    cache.write_text(json.dumps({
        "captured_at": 0,  # 1970, definitely stale
        "scores": {"CUDA0": 42.0},
        "detail": {},
    }))
    config = _config(coord_gpus=[16], workers=[])
    scores = measure_device_scores(config, cache_path=cache)
    assert scores["CUDA0"] == LOCAL_BASELINE_SCORE
