"""A/B benchmark: proxy vs direct target comparison.

Runs the same prompts through the speculation proxy and directly against
the target, producing a side-by-side speedup report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("tightwad.bench")


# ---------------------------------------------------------------------------
# Default prompt set — mix of code, chat, reasoning
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome.",
    "Explain the difference between TCP and UDP in simple terms.",
    "What are the three laws of thermodynamics?",
    "Write a bash one-liner that finds the 10 largest files in the current directory.",
    "Translate this to French: The weather is beautiful today and I want to go for a walk.",
    "What causes a rainbow to appear after rain?",
    "Write a SQL query to find duplicate email addresses in a users table.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Explain how a hash table works and its time complexity.",
    "What is the capital of every country in South America?",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single prompt completion."""

    prompt_index: int
    tokens: int
    elapsed_s: float
    tok_per_s: float
    text: str = ""


@dataclass
class BenchmarkResult:
    """Side-by-side comparison of proxy vs direct target."""

    proxy_results: list[RunResult] = field(default_factory=list)
    direct_results: list[RunResult] = field(default_factory=list)
    proxy_stats: dict | None = None  # from /v1/tightwad/status

    @property
    def proxy_avg_tps(self) -> float:
        if not self.proxy_results:
            return 0.0
        return sum(r.tok_per_s for r in self.proxy_results) / len(self.proxy_results)

    @property
    def direct_avg_tps(self) -> float:
        if not self.direct_results:
            return 0.0
        return sum(r.tok_per_s for r in self.direct_results) / len(self.direct_results)

    @property
    def speedup(self) -> float:
        if self.direct_avg_tps <= 0:
            return 0.0
        return self.proxy_avg_tps / self.direct_avg_tps

    @property
    def proxy_median_tps(self) -> float:
        return _median([r.tok_per_s for r in self.proxy_results])

    @property
    def direct_median_tps(self) -> float:
        return _median([r.tok_per_s for r in self.direct_results])

    @property
    def proxy_p95_latency(self) -> float:
        return _percentile([r.elapsed_s for r in self.proxy_results], 95)

    @property
    def direct_p95_latency(self) -> float:
        return _percentile([r.elapsed_s for r in self.direct_results], 95)

    def to_dict(self) -> dict:
        return {
            "proxy": {
                "avg_tok_s": round(self.proxy_avg_tps, 1),
                "median_tok_s": round(self.proxy_median_tps, 1),
                "p95_latency_s": round(self.proxy_p95_latency, 2),
                "runs": len(self.proxy_results),
            },
            "direct": {
                "avg_tok_s": round(self.direct_avg_tps, 1),
                "median_tok_s": round(self.direct_median_tps, 1),
                "p95_latency_s": round(self.direct_p95_latency, 2),
                "runs": len(self.direct_results),
            },
            "speedup": round(self.speedup, 2),
            "proxy_stats": self.proxy_stats,
        }


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def _complete(
    url: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    auth_token: str | None = None,
) -> tuple[int, float, str]:
    """Send a completion request and return (tokens, elapsed_s, text)."""
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    t0 = time.monotonic()
    resp = httpx.post(
        f"{url}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        headers=headers,
        timeout=120.0,
    )
    elapsed = time.monotonic() - t0

    if resp.status_code != 200:
        logger.warning("Completion failed: HTTP %d from %s", resp.status_code, url)
        return 0, elapsed, ""

    data = resp.json()
    usage = data.get("usage", {})
    tokens = usage.get("completion_tokens", 0)
    text = ""
    choices = data.get("choices", [])
    if choices:
        text = choices[0].get("text", "")

    # If no usage data, estimate from text length
    if tokens == 0 and text:
        tokens = max(1, len(text.split()))

    return tokens, elapsed, text


def run_benchmark(
    proxy_url: str,
    target_url: str,
    prompts: list[str] | None = None,
    max_tokens: int = 128,
    warmup: int = 1,
    proxy_token: str | None = None,
) -> BenchmarkResult:
    """Run A/B comparison: same prompts through proxy and direct target.

    Parameters
    ----------
    proxy_url:
        Tightwad proxy URL (e.g. http://localhost:8088)
    target_url:
        Direct target model URL (e.g. http://localhost:11434 for Ollama)
    prompts:
        List of prompts to test. Uses DEFAULT_PROMPTS if None.
    max_tokens:
        Max tokens per completion.
    warmup:
        Number of warmup runs (not counted in results).
    proxy_token:
        Bearer token for authenticated proxy.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    result = BenchmarkResult()

    # Warmup
    for i in range(warmup):
        logger.debug("Warmup %d/%d", i + 1, warmup)
        _complete(proxy_url, prompts[0], max_tokens=32, auth_token=proxy_token)
        _complete(target_url, prompts[0], max_tokens=32)

    # Run through proxy
    for i, prompt in enumerate(prompts):
        tokens, elapsed, text = _complete(
            proxy_url, prompt, max_tokens, auth_token=proxy_token
        )
        tps = tokens / elapsed if elapsed > 0 else 0
        result.proxy_results.append(RunResult(
            prompt_index=i, tokens=tokens, elapsed_s=elapsed,
            tok_per_s=tps, text=text,
        ))
        logger.debug("Proxy [%d/%d]: %d tok in %.1fs (%.1f tok/s)",
                      i + 1, len(prompts), tokens, elapsed, tps)

    # Run through target directly
    for i, prompt in enumerate(prompts):
        tokens, elapsed, text = _complete(target_url, prompt, max_tokens)
        tps = tokens / elapsed if elapsed > 0 else 0
        result.direct_results.append(RunResult(
            prompt_index=i, tokens=tokens, elapsed_s=elapsed,
            tok_per_s=tps, text=text,
        ))
        logger.debug("Direct [%d/%d]: %d tok in %.1fs (%.1f tok/s)",
                      i + 1, len(prompts), tokens, elapsed, tps)

    # Fetch proxy stats
    try:
        headers = {}
        if proxy_token:
            headers["Authorization"] = f"Bearer {proxy_token}"
        resp = httpx.get(
            f"{proxy_url}/v1/tightwad/status", headers=headers, timeout=10.0
        )
        if resp.status_code == 200:
            result.proxy_stats = resp.json().get("stats")
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(result: BenchmarkResult) -> str:
    """Format benchmark results as a Rich-compatible string."""
    from rich.console import Console
    from rich.table import Table
    import io

    console = Console(record=True, width=80, file=io.StringIO())

    console.print("\n[bold]A/B Benchmark Results[/bold]")
    console.print(f"  Prompts: {len(result.proxy_results)}")

    table = Table()
    table.add_column("Metric")
    table.add_column("Proxy", justify="right")
    table.add_column("Direct", justify="right")
    table.add_column("", justify="center")

    table.add_row(
        "Avg tok/s",
        f"{result.proxy_avg_tps:.1f}",
        f"{result.direct_avg_tps:.1f}",
        f"[bold green]{result.speedup:.2f}x[/bold green]" if result.speedup >= 1.0
        else f"[bold red]{result.speedup:.2f}x[/bold red]",
    )
    table.add_row(
        "Median tok/s",
        f"{result.proxy_median_tps:.1f}",
        f"{result.direct_median_tps:.1f}",
        "",
    )
    table.add_row(
        "p95 latency",
        f"{result.proxy_p95_latency:.2f}s",
        f"{result.direct_p95_latency:.2f}s",
        "",
    )

    console.print(table)

    # Proxy stats
    if result.proxy_stats:
        stats = result.proxy_stats
        acceptance = stats.get("acceptance_rate", 0)
        rounds = stats.get("total_rounds", 0)
        console.print(f"\n[bold]Proxy Stats[/bold]")
        console.print(f"  Acceptance rate: {acceptance * 100:.1f}%")
        console.print(f"  Total rounds:    {rounds}")
        consensus_accepted = stats.get("consensus_accepted", 0)
        if consensus_accepted:
            console.print(f"  Consensus skips: {consensus_accepted}")

    # Per-prompt breakdown
    console.print(f"\n[bold]Per-Prompt Breakdown[/bold]")
    detail = Table()
    detail.add_column("#", justify="right")
    detail.add_column("Proxy tok/s", justify="right")
    detail.add_column("Direct tok/s", justify="right")
    detail.add_column("Speedup", justify="right")

    for p, d in zip(result.proxy_results, result.direct_results):
        sp = p.tok_per_s / d.tok_per_s if d.tok_per_s > 0 else 0
        style = "green" if sp >= 1.0 else "red"
        detail.add_row(
            str(p.prompt_index + 1),
            f"{p.tok_per_s:.1f}",
            f"{d.tok_per_s:.1f}",
            f"[{style}]{sp:.2f}x[/{style}]",
        )
    console.print(detail)

    return console.export_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]


# ---------------------------------------------------------------------------
# MoE benchmark — async, streams per-prompt updates
# ---------------------------------------------------------------------------


async def _chat_completion_stream(
    client, url: str, model: str, prompt: str, max_tokens: int,
    auth_token: str | None = None,
) -> tuple[int, float, float, str]:
    """Return (tokens, ttft_s, total_s, text) from an OpenAI-compat chat endpoint.

    Uses SSE streaming to measure time-to-first-token. Falls back to a
    non-streaming call if the server doesn't return an SSE body.
    """
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.monotonic()
    ttft = None
    text = ""
    tokens = 0
    try:
        async with client.stream(
            "POST", f"{url}/v1/chat/completions",
            json=payload, headers=headers, timeout=300.0,
        ) as resp:
            if resp.status_code != 200:
                logger.warning("chat stream HTTP %d from %s", resp.status_code, url)
                return 0, 0.0, time.monotonic() - start, ""
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_s = line[5:].strip()
                if data_s == "[DONE]":
                    break
                try:
                    import json as _json
                    chunk = _json.loads(data_s)
                except Exception:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - start
                    text += content
                    tokens += 1
    except httpx.HTTPError as exc:
        logger.warning("chat stream error %s", exc)
        return 0, 0.0, time.monotonic() - start, ""
    total = time.monotonic() - start
    if tokens == 0:
        tokens = max(1, len(text.split()))
    return tokens, ttft if ttft is not None else total, total, text


async def run_moe_benchmark(
    proxy_url: str,
    target_url: str,
    target_model: str,
    prompts: list[dict],
    max_tokens: int = 256,
    warmup: int = 2,
    proxy_token: str | None = None,
    live: bool = True,
    on_update=None,
) -> dict:
    """Run proxy vs direct comparison against an OpenAI-compatible target.

    Streams per-prompt progress to ``on_update(dict)`` when provided. Returns a
    JSON-serializable dict matching the ``benchmarks/*-moe-*.json`` schema.

    ``prompts`` is a list of ``{"name": str, "text": str}`` entries.
    """
    results: list[dict] = []

    async with httpx.AsyncClient() as client:
        for _ in range(warmup):
            await _chat_completion_stream(
                client, proxy_url, target_model, prompts[0]["text"],
                max_tokens=32, auth_token=proxy_token,
            )

        for p in prompts:
            name, text_prompt = p["name"], p["text"]
            p_tokens, p_ttft, p_total, p_text = await _chat_completion_stream(
                client, proxy_url, target_model, text_prompt, max_tokens,
                auth_token=proxy_token,
            )
            d_tokens, d_ttft, d_total, d_text = await _chat_completion_stream(
                client, target_url, target_model, text_prompt, max_tokens,
            )
            p_tps = p_tokens / p_total if p_total > 0 else 0
            d_tps = d_tokens / d_total if d_total > 0 else 0
            speedup = p_tps / d_tps if d_tps > 0 else 0

            entry = {
                "name": name,
                "prompt": text_prompt,
                "proxy": {"tokens": p_tokens, "ttft_ms": round(p_ttft * 1000, 1),
                           "elapsed_s": round(p_total, 3), "tok_s": round(p_tps, 1),
                           "text": p_text[:200]},
                "direct": {"tokens": d_tokens, "ttft_ms": round(d_ttft * 1000, 1),
                            "elapsed_s": round(d_total, 3), "tok_s": round(d_tps, 1),
                            "text": d_text[:200]},
                "speedup": round(speedup, 2),
            }
            results.append(entry)
            if on_update is not None:
                on_update({
                    "name": name,
                    "ttft_ms": entry["proxy"]["ttft_ms"],
                    "direct_tps": d_tps,
                    "proxy_tps": p_tps,
                    "speedup": speedup,
                    "acceptance_rate": None,
                })

        proxy_stats = None
        try:
            resp = await client.get(f"{proxy_url}/v1/tightwad/status",
                                      timeout=10.0)
            if resp.status_code == 200:
                proxy_stats = resp.json().get("stats")
        except httpx.HTTPError:
            pass

    avg_speedup = (
        sum(r["speedup"] for r in results) / len(results)
        if results else 0.0
    )
    return {
        "config": f"{target_model} via {target_url}",
        "target": target_model,
        "proxy_url": proxy_url,
        "max_tokens": max_tokens,
        "warmup": warmup,
        "avg_speedup": round(avg_speedup, 2),
        "prompts": results,
        "proxy_stats": proxy_stats,
        "moe": {
            "acceptance_rate": proxy_stats.get("acceptance_rate") if proxy_stats else None,
            "total_rounds": proxy_stats.get("total_rounds") if proxy_stats else None,
        },
    }
