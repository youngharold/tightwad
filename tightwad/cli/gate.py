"""Quality gate subcommand group: start, status."""

from __future__ import annotations

import json
import sys

import click

from . import cli, console, _load


@cli.group()
def gate():
    """Quality gate proxy commands (CPU fleet + GPU verifier)."""
    pass


@gate.command("start")
@click.pass_context
def gate_start(ctx):
    """Start the quality gate proxy server."""
    config = _load(ctx)
    if not config.quality_gate:
        console.print("[red]No quality_gate section in config. Add verifier + agents.[/red]")
        sys.exit(1)

    from ..quality_gate import QualityGateProxy

    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    qg = config.quality_gate
    proxy = QualityGateProxy(qg)

    async def handle_completion(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.0)

        text = await proxy.handle_request(prompt, max_tokens, temperature)

        return JSONResponse({
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "model": qg.verifier_model,
        })

    async def handle_status(request: Request):
        s = proxy.stats
        return JSONResponse({
            "mode": "quality_gate",
            "stats": {
                "total_requests": s.total_requests,
                "approved": s.approved,
                "corrected": s.corrected,
                "rejected": s.rejected,
                "errors": s.errors,
                "approve_rate": round(s.approve_rate, 3),
                "gpu_usage_rate": round(s.gpu_usage_rate, 3),
                "cache_hits": s.cache_hits,
                "avg_agent_ms": round(s.total_agent_ms / max(s.total_requests, 1), 1),
                "avg_verify_ms": round(s.total_verify_ms / max(s.total_requests, 1), 1),
                "uptime_seconds": round(s.uptime_seconds, 1),
            },
            "agents": [a.url for a in qg.agents],
            "verifier": qg.verifier_url,
        })

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        yield
        await proxy.close()

    app = Starlette(
        routes=[
            Route("/v1/completions", handle_completion, methods=["POST"]),
            Route("/v1/tightwad/status", handle_status, methods=["GET"]),
        ],
        lifespan=lifespan,
    )

    console.print("[bold]Starting quality gate proxy...[/bold]")
    console.print(f"  Verifier: {qg.verifier_url} ({qg.verifier_model})")
    console.print(f"  Agents:   {len(qg.agents)}")
    for a in qg.agents:
        console.print(f"    - {a.url} ({a.model_name})")
    console.print(f"  Routing:  {qg.routing}")
    console.print(f"  Cache:    {'on' if qg.cache_identical else 'off'}")

    uvicorn.run(app, host=qg.host, port=qg.port, log_level="info")


@gate.command("status")
@click.pass_context
def gate_status(ctx):
    """Check quality gate stats."""
    config = _load(ctx)
    if not config.quality_gate:
        console.print("[red]No quality_gate configured.[/red]")
        sys.exit(1)

    import httpx

    qg = config.quality_gate
    try:
        resp = httpx.get(f"http://127.0.0.1:{qg.port}/v1/tightwad/status", timeout=5.0)
        if resp.status_code == 200:
            console.print(json.dumps(resp.json(), indent=2))
        else:
            console.print(f"[red]HTTP {resp.status_code}[/red]")
    except Exception as e:
        console.print(f"[red]Quality gate not running: {e}[/red]")
