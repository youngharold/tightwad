"""Proxy subcommand group: start, stop, status."""

from __future__ import annotations

import os
import sys

import click
from rich.table import Table

from .. import proxy as proxy_mod
from ..coordinator import LOGDIR
from . import cli, console, _load, PROXY_LOG


@cli.group()
def proxy():
    """Speculative decoding proxy commands."""
    pass


@proxy.command("start")
@click.pass_context
def proxy_start(ctx):
    """Start the speculative decoding proxy server."""
    config = _load(ctx)
    if config.proxy is None:
        console.print("[red]No proxy section in config. Add it to cluster.yaml.[/red]")
        sys.exit(1)

    existing_pid = proxy_mod.read_pidfile()
    if existing_pid is not None:
        try:
            os.kill(existing_pid, 0)
            console.print(
                f"[yellow]Proxy already running (PID {existing_pid}). "
                f"Stop it first with: tightwad proxy stop[/yellow]"
            )
            sys.exit(1)
        except ProcessLookupError:
            proxy_mod.remove_pidfile()

    pc = config.proxy
    console.print(f"[bold]Starting speculative decoding proxy...[/bold]")
    if pc.drafters:
        console.print(f"  [bold]Drafters ({len(pc.drafters)}):[/bold]")
        for d in pc.drafters:
            console.print(f"    - {d.model_name} @ {d.url} ({d.backend})")
    else:
        console.print(f"  Draft:  {pc.draft.model_name} @ {pc.draft.url}")
    console.print(f"  Target: {pc.target.model_name} @ {pc.target.url}")
    console.print(f"  Max draft tokens: {pc.max_draft_tokens}")
    console.print(f"  Listening on: {pc.host}:{pc.port}")
    console.print(f"  Dashboard: http://127.0.0.1:{pc.port}/dashboard")

    import uvicorn
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "fmt": "%(asctime)s %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(PROXY_LOG),
                "mode": "a",
                "formatter": "default",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console", "file"], "level": "INFO"},
            "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO"},
            "uvicorn.access": {"handlers": ["console", "file"], "level": "INFO"},
        },
    }
    app = proxy_mod.create_app(pc)
    proxy_mod.write_pidfile()
    try:
        uvicorn.run(app, host=pc.host, port=pc.port, log_level="info", log_config=log_config)
    finally:
        proxy_mod.remove_pidfile()


@proxy.command("stop")
def proxy_stop():
    """Stop the speculative decoding proxy."""
    if proxy_mod.stop_proxy():
        console.print("[green]Proxy stopped.[/green]")
    else:
        console.print("[yellow]Proxy was not running.[/yellow]")


@proxy.command("status")
@click.pass_context
def proxy_status(ctx):
    """Show proxy health and acceptance rate stats."""
    config = _load(ctx)
    if config.proxy is None:
        console.print("[red]No proxy section in config.[/red]")
        sys.exit(1)

    import httpx

    pc = config.proxy
    url = f"http://127.0.0.1:{pc.port}/v1/tightwad/status"

    try:
        resp = httpx.get(url, timeout=5.0)
        data = resp.json()
    except Exception:
        console.print("[dim]○ Proxy not running[/dim]")
        return

    # Drafters or single draft health
    if "drafters" in data:
        console.print("  [bold]Drafters:[/bold]")
        drafter_table = Table()
        drafter_table.add_column("Model")
        drafter_table.add_column("URL")
        drafter_table.add_column("Backend")
        drafter_table.add_column("Health")
        drafter_table.add_column("Wins")
        for d in data["drafters"]:
            alive = d["health"].get("alive", False)
            health_str = "[green]alive[/green]" if alive else "[red]down[/red]"
            drafter_table.add_row(
                d["model"], d["url"], d["backend"],
                health_str, str(d["wins"]),
            )
        console.print(drafter_table)
    elif "draft" in data:
        info = data["draft"]
        alive = info["health"].get("alive", False)
        icon = "[green]●[/green]" if alive else "[red]●[/red]"
        console.print(f"  {icon} Draft: {info['model']} @ {info['url']}")

    # Target health
    target = data["target"]
    t_alive = target["health"].get("alive", False)
    t_icon = "[green]●[/green]" if t_alive else "[red]●[/red]"
    console.print(f"  {t_icon} Target: {target['model']} @ {target['url']}")

    # Stats
    stats = data.get("stats", {})
    if stats.get("total_rounds", 0) > 0:
        console.print()
        table = Table(title="Speculation Stats")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Rounds", str(stats["total_rounds"]))
        table.add_row("Drafted", str(stats["total_drafted"]))
        table.add_row("Accepted", str(stats["total_accepted"]))
        table.add_row("Acceptance rate", f"{stats['acceptance_rate']:.1%}")
        table.add_row("Tokens/round", f"{stats['effective_tokens_per_round']:.1f}")
        table.add_row("Uptime", f"{stats['uptime_seconds']:.0f}s")
        console.print(table)
    else:
        console.print("\n[dim]No speculation rounds yet.[/dim]")
