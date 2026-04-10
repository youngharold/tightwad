"""Peer agent subcommand group: start, stop, status."""

from __future__ import annotations

import os
import sys

import click

from ..coordinator import LOGDIR
from . import cli, console, _load


@cli.group()
def peer():
    """Manage the local peer agent daemon."""
    pass


@peer.command("start")
@click.option("--port", default=9191, type=int, help="Peer agent port (default: 9191)")
@click.pass_context
def peer_start(ctx, port):
    """Start the peer agent daemon."""
    from .. import peer as peer_mod

    # Try loading config for PeerConfig; fall back to defaults
    peer_config = None
    try:
        config = _load(ctx)
        peer_config = config.peer
    except SystemExit:
        pass

    if peer_config is None:
        from ..config import PeerConfig
        peer_config = PeerConfig(port=port)
    elif port != 9191:
        # CLI flag overrides config file
        peer_config.port = port

    existing_pid = peer_mod.read_pidfile()
    if existing_pid is not None:
        try:
            os.kill(existing_pid, 0)
            console.print(
                f"[yellow]Peer agent already running (PID {existing_pid}). "
                f"Stop it first with: tightwad peer stop[/yellow]"
            )
            sys.exit(1)
        except ProcessLookupError:
            peer_mod.remove_pidfile()

    console.print(f"[bold]Starting peer agent...[/bold]")
    console.print(f"  Listening on: {peer_config.host}:{peer_config.port}")
    if peer_config.model_dirs:
        console.print(f"  Model dirs: {', '.join(peer_config.model_dirs)}")
    if not peer_config.auth_token:
        console.print("  [yellow]Warning: no auth token configured[/yellow]")

    import uvicorn
    LOGDIR.mkdir(parents=True, exist_ok=True)
    peer_log = LOGDIR / "peer.log"
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
                "filename": str(peer_log),
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
    app = peer_mod.create_app(peer_config)
    peer_mod.write_pidfile()
    try:
        uvicorn.run(
            app, host=peer_config.host, port=peer_config.port,
            log_level="info", log_config=log_config,
        )
    finally:
        peer_mod.remove_pidfile()


@peer.command("stop")
def peer_stop():
    """Stop the peer agent daemon."""
    from .. import peer as peer_mod

    if peer_mod.stop_peer():
        console.print("[green]Peer agent stopped.[/green]")
    else:
        console.print("[yellow]Peer agent was not running.[/yellow]")


@peer.command("status")
def peer_status():
    """Check if the local peer agent is running."""
    from .. import peer as peer_mod

    pid = peer_mod.read_pidfile()
    if pid is None:
        console.print("[dim]Peer agent not running.[/dim]")
        return

    try:
        os.kill(pid, 0)
        console.print(f"[green bold]Peer agent running[/green bold] (PID {pid})")
    except ProcessLookupError:
        console.print("[yellow]Stale PID file (process not running).[/yellow]")
        peer_mod.remove_pidfile()
