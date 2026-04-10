"""Cluster lifecycle commands: start, stop, status, swap, benchmark, bench."""

from __future__ import annotations

import sys

import click
from rich.table import Table

from .. import coordinator, worker
from . import cli, console, _load


@cli.command()
@click.option("-m", "--model", default=None, help="Model name from config")
@click.option("--ram-reclaim", type=click.Choice(["off", "on", "auto"]), default=None,
              help="RAM reclaim mode (default: from config, usually 'auto')")
@click.option("--skip-version-check", is_flag=True, default=False,
              help="Skip llama.cpp version matching between coordinator and workers")
@click.pass_context
def start(ctx, model, ram_reclaim, skip_version_check):
    """Start the coordinator llama-server with RPC workers."""
    config = _load(ctx)

    console.print("[bold]Checking RPC workers...[/bold]")
    statuses = worker.check_all_workers(config)
    for s in statuses:
        icon = "[green]●[/green]" if s.alive else "[red]●[/red]"
        latency = f" ({s.latency_ms}ms)" if s.latency_ms else ""
        console.print(f"  {icon} {s.host}:{s.port}{latency}")

    dead = [s for s in statuses if not s.alive]
    if dead:
        console.print(
            "\n[red bold]Cannot start — RPC workers unreachable.[/red bold]"
        )
        console.print("Start rpc-server on the worker machine first.")
        sys.exit(1)

    model_cfg = (
        config.models.get(model) if model else config.default_model()
    )
    if not model_cfg:
        console.print("[red]No model specified and no default configured.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Starting coordinator with {model_cfg.name}...[/bold]")
    console.print(f"  Tensor split: {config.tensor_split()}")
    console.print(f"  Total VRAM: {config.total_vram_gb} GB across {len(config.all_gpus)} GPUs")

    mode = ram_reclaim or config.ram_reclaim

    try:
        if mode == "off":
            pid = coordinator.start(config, model, skip_version_check=skip_version_check)
            console.print(f"\n[green bold]Coordinator started (PID {pid})[/green bold]")
            console.print(f"  API: http://localhost:{config.coordinator_port}/v1")
        else:
            console.print(f"  RAM reclaim: {mode}")
            pid, result = coordinator.start_and_reclaim(
                config, model, ram_reclaim=mode,
                skip_version_check=skip_version_check,
            )
            console.print(f"\n[green bold]Coordinator started (PID {pid})[/green bold]")
            console.print(f"  API: http://localhost:{config.coordinator_port}/v1")
            if result:
                if result.method == "failed":
                    console.print(f"  RAM reclaim: [yellow]failed[/yellow] ({result.error})")
                elif result.method == "skipped":
                    console.print(f"  RAM reclaim: skipped ({result.error or 'not needed'})")
                else:
                    console.print(
                        f"  [green]Reclaimed {result.reclaimed_mb:,.0f} MB RAM "
                        f"({result.method})[/green]"
                    )
            elif mode == "auto":
                console.print("  RAM reclaim: skipped (sufficient RAM)")
    except RuntimeError as e:
        console.print(f"\n[red]{e}[/red]")
        sys.exit(1)


@cli.command()
def stop():
    """Stop the coordinator llama-server."""
    if coordinator.stop():
        console.print("[green]Coordinator stopped.[/green]")
    else:
        console.print("[yellow]Coordinator was not running.[/yellow]")


@cli.command()
@click.pass_context
def status(ctx):
    """Show cluster status."""
    # Try loading config, fall back to config-less mode
    config = None
    try:
        config = _load(ctx)
    except SystemExit:
        # _load calls sys.exit on FileNotFoundError — catch it for config-less mode
        pass

    st = coordinator.status(config)

    # Coordinator
    coord = st["coordinator"]
    if coord["running"]:
        console.print(
            f"[green bold]● Coordinator[/green bold] "
            f"PID {coord['pid']} on :{coord['port']}"
        )
        if coord["health"] and coord["health"].get("alive"):
            console.print("  Health: [green]OK[/green]")
        elif coord["health"]:
            console.print(f"  Health: [red]{coord['health'].get('error', 'unhealthy')}[/red]")
    else:
        console.print("[dim]○ Coordinator not running[/dim]")

    # Workers
    if st["workers"]:
        console.print()
        table = Table(title="RPC Workers")
        table.add_column("Address")
        table.add_column("Status")
        table.add_column("Latency")
        for w in st["workers"]:
            status_str = "[green]alive[/green]" if w["alive"] else f"[red]down[/red]"
            latency_str = f"{w['latency_ms']}ms" if w["latency_ms"] else "-"
            table.add_row(w["address"], status_str, latency_str)
        console.print(table)

    # Config summary
    cfg = st.get("config", {})
    if "total_vram_gb" in cfg:
        console.print(f"\nTotal VRAM: [bold]{cfg['total_vram_gb']} GB[/bold] across {cfg['gpu_count']} GPUs")
        console.print(f"Tensor split: {cfg['tensor_split']}")
        console.print(f"Models: {', '.join(cfg['models'])}")
    elif "model" in cfg:
        console.print(f"\n  Model: {cfg['model'] or 'unknown'}")
        if cfg.get("started"):
            import datetime
            started = datetime.datetime.fromtimestamp(cfg["started"])
            console.print(f"  Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")


@cli.command()
@click.argument("model_name")
@click.pass_context
def swap(ctx, model_name):
    """Hot-swap to a different model (restarts coordinator, keeps RPC workers)."""
    config = _load(ctx)
    try:
        pid = coordinator.swap_model(config, model_name)
        console.print(
            f"[green bold]Swapped to {model_name} (PID {pid})[/green bold]"
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def benchmark(ctx):
    """Run a quick benchmark against the running coordinator."""
    config = _load(ctx)
    health = worker.check_coordinator_health("127.0.0.1", config.coordinator_port)
    if not health.get("alive"):
        console.print("[red]Coordinator not running. Start it first.[/red]")
        sys.exit(1)

    import httpx
    import time

    base = f"http://127.0.0.1:{config.coordinator_port}"

    # Prompt processing benchmark
    console.print("[bold]Running benchmark...[/bold]\n")
    prompt = "Explain quantum computing in detail. " * 64  # ~512 tokens

    start_time = time.monotonic()
    resp = httpx.post(
        f"{base}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    elapsed = time.monotonic() - start_time

    if resp.status_code != 200:
        console.print(f"[red]Server returned {resp.status_code}[/red]")
        sys.exit(1)

    data = resp.json()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    pp_speed = prompt_tokens / elapsed if elapsed > 0 else 0
    tg_speed = completion_tokens / elapsed if elapsed > 0 else 0

    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Prompt tokens", str(prompt_tokens))
    table.add_row("Completion tokens", str(completion_tokens))
    table.add_row("Total time", f"{elapsed:.1f}s")
    table.add_row("Prompt processing", f"~{pp_speed:.0f} tok/s")
    table.add_row("Generation", f"~{tg_speed:.1f} tok/s")
    console.print(table)


@cli.command("bench")
@click.option("--prompts", type=click.Path(exists=True), default=None,
              help="File with one prompt per line (default: built-in set of 10)")
@click.option("--max-tokens", default=128, type=int, help="Max tokens per completion (default: 128)")
@click.option("--warmup", default=1, type=int, help="Warmup runs (default: 1)")
@click.option("--json", "as_json", is_flag=True, help="Output JSON instead of table")
@click.pass_context
def bench(ctx, prompts, max_tokens, warmup, as_json):
    """A/B benchmark: compare proxy speed vs direct target.

    Runs the same prompts through the Tightwad proxy and directly against
    the target model, reporting speedup, tok/s, and latency side-by-side.
    """
    config = _load(ctx)
    if not config.proxy:
        console.print("[red]No proxy configured. bench requires a proxy section in config.[/red]")
        sys.exit(1)

    proxy_url = f"http://127.0.0.1:{config.proxy.port}"
    target_url = config.proxy.target.url

    # Load prompts from file or use defaults
    prompt_list = None
    if prompts:
        prompt_list = [line.strip() for line in open(prompts) if line.strip()]
        console.print(f"[dim]Loaded {len(prompt_list)} prompts from {prompts}[/dim]")

    console.print("[bold]Running A/B benchmark...[/bold]")
    console.print(f"  Proxy:  {proxy_url}")
    console.print(f"  Target: {target_url}")
    console.print(f"  Prompts: {len(prompt_list) if prompt_list else 10}")
    console.print(f"  Max tokens: {max_tokens}")
    console.print()

    from ..bench import run_benchmark, format_report

    result = run_benchmark(
        proxy_url=proxy_url,
        target_url=target_url,
        prompts=prompt_list,
        max_tokens=max_tokens,
        warmup=warmup,
        proxy_token=config.proxy.auth_token,
    )

    if as_json:
        import json
        console.print(json.dumps(result.to_dict(), indent=2))
    else:
        console.print(format_report(result))
