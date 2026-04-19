"""Auto-measure device latency to score placement decisions.

Coordinator GPUs get a high baseline (local VRAM is always faster than any
network path). RPC workers are scored from TCP connect + small HTTP round-trip
timings. Results cache at ``~/.tightwad/device-scores.json`` for 24 hours.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("tightwad.moe_device_bench")

CACHE_PATH = Path.home() / ".tightwad" / "device-scores.json"
TTL_SECONDS = 24 * 60 * 60
LOCAL_BASELINE_SCORE = 1000.0


@dataclass
class DeviceScore:
    ot_device: str
    score: float
    rtt_ms: float
    source: str  # "local" | "tcp" | "cache"


def measure_device_scores(
    config,
    force: bool = False,
    cache_path: Path = CACHE_PATH,
) -> dict[str, float]:
    """Return ``{ot_device: score}`` for every DeviceSlot derived from config.

    Higher score = faster device. Cached to disk with a 24h TTL; pass
    ``force=True`` to bypass the cache.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = _load_cache(cache_path)
        if cached is not None:
            logger.debug("moe_device_bench: using cached scores from %s", cache_path)
            return cached

    scores: dict[str, DeviceScore] = {}

    for i, _gpu in enumerate(config.coordinator_gpus):
        ot = f"CUDA{i}"
        scores[ot] = DeviceScore(ot_device=ot, score=LOCAL_BASELINE_SCORE,
                                   rtt_ms=0.0, source="local")

    for worker in config.workers:
        for gpu in worker.gpus:
            port = gpu.rpc_port or 50052
            ot = f"RPC[{worker.host}:{port}]"
            rtt = _measure_tcp_rtt(worker.host, port)
            if rtt is None:
                scores[ot] = DeviceScore(ot_device=ot, score=0.0,
                                           rtt_ms=float("inf"), source="tcp")
            else:
                scores[ot] = DeviceScore(ot_device=ot,
                                           score=LOCAL_BASELINE_SCORE / max(rtt, 0.1),
                                           rtt_ms=rtt, source="tcp")

    result = {d.ot_device: d.score for d in scores.values()}
    _save_cache(cache_path, scores)
    return result


def _measure_tcp_rtt(host: str, port: int, iterations: int = 5,
                     timeout: float = 2.0) -> float | None:
    """Average TCP connect RTT in milliseconds across ``iterations`` attempts."""
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except OSError as exc:
            logger.debug("moe_device_bench: tcp connect %s:%d failed: %s",
                         host, port, exc)
            return None
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    # Trim the slowest sample to reduce jitter
    return sum(samples[:-1]) / max(1, len(samples) - 1) if len(samples) > 1 else samples[0]


def _load_cache(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - payload.get("captured_at", 0) > TTL_SECONDS:
        return None
    return {k: float(v) for k, v in payload.get("scores", {}).items()}


def _save_cache(path: Path, scores: dict[str, DeviceScore]) -> None:
    payload = {
        "captured_at": time.time(),
        "scores": {d.ot_device: d.score for d in scores.values()},
        "detail": {
            d.ot_device: {"rtt_ms": d.rtt_ms, "source": d.source}
            for d in scores.values()
        },
    }
    path.write_text(json.dumps(payload, indent=2))
