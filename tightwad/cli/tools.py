"""Miscellaneous tool commands: doctor, init, inspect, distribute, deploy, tune,
pull, logs, reclaim, load, chat, manifest, swarm."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from .. import coordinator, worker
from .. import doctor as doctor_mod
from .. import proxy as proxy_mod
from .. import manifest as manifest_mod
from .. import swarm_transfer as swarm_mod
from .. import init_wizard
from ..coordinator import LOGDIR, COORDINATOR_LOG
from . import cli, console, _load, PROXY_LOG


@cli.command()
@click.option("--subnet", default=None, help="Subnet to scan (e.g. 192.168.1.0/24, auto-detected if omitted)")
@click.option("--port", "extra_ports", multiple=True, type=int, help="Additional ports to scan (repeatable)")
@click.option("-o", "--output", default="configs/cluster.yaml", type=click.Path(), help="Output config path")
@click.option("--draft-url", default=None, help="Draft server URL (non-interactive mode)")
@click.option("--draft-model", default=None, help="Draft model name (required with --draft-url)")
@click.option("--draft-backend", default=None, help="Draft backend: ollama or llamacpp (auto-detected from port)")
@click.option("--target-url", default=None, help="Target server URL (non-interactive mode)")
@click.option("--target-model", default=None, help="Target model name (required with --target-url)")
@click.option("--target-backend", default=None, help="Target backend: ollama or llamacpp (auto-detected from port)")
@click.option("--max-draft-tokens", default=32, type=int, help="Max tokens per draft round (default: 32)")
@click.option("-y", "--yes", is_flag=True, help="Overwrite existing config without prompting")
@click.option("--local", is_flag=True, help="Auto-detect local GPUs and generate coordinator-only config")
@click.option("--model-path", default=None, type=click.Path(), help="Path to GGUF model file (used with --local)")
@click.option("--port", "local_port", default=8080, type=int, help="Coordinator port (used with --local, default: 8080)")
def init(subnet, extra_ports, output, draft_url, draft_model, draft_backend,
         target_url, target_model, target_backend, max_draft_tokens, yes,
         local, model_path, local_port):
    """Auto-discover LAN inference servers and generate cluster.yaml."""
    import asyncio
    from urllib.parse import urlparse

    output_path = Path(output)

    # Local mode: auto-detect GPUs
    if local:
        from ..gpu_detect import detect_gpus, detect_binary

        console.print("[bold]Detecting GPUs...[/bold]")
        gpus = detect_gpus()
        if not gpus:
            console.print("[red]No GPUs detected.[/red]")
            console.print("Ensure nvidia-smi, rocm-smi, or system_profiler is available.")
            sys.exit(1)

        for gpu in gpus:
            vram_gb = gpu.vram_mb / 1024
            console.print(f"  [green]●[/green] {gpu.name} ({vram_gb:.0f} GB, {gpu.backend})")

        binary = detect_binary()
        if binary:
            console.print(f"\n  Binary: {binary}")
        else:
            console.print("\n  [yellow]llama-server not found in PATH[/yellow]")
            console.print("  Install llama.cpp: https://github.com/ggml-org/llama.cpp")

        yaml_str = init_wizard.generate_local_yaml(
            gpus=gpus, binary=binary, model_path=model_path, port=local_port,
        )

        if output_path.exists() and not yes:
            overwrite = input(f"{output_path} already exists. Overwrite? [y/N] ").strip().lower()
            if overwrite != "y":
                console.print("[dim]Cancelled.[/dim]")
                return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_str)
        console.print(f"\n[green]✓[/green] Config written to {output_path}")
        if not model_path:
            console.print("\n[dim]Tip: add a model with --model-path or edit the config directly.[/dim]")
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"  tightwad -c {output_path} start")
        return

    # Non-interactive mode: --draft-url + --target-url
    if draft_url and target_url:
        if not draft_model:
            raise click.UsageError("--draft-model is required when using --draft-url")
        if not target_model:
            raise click.UsageError("--target-model is required when using --target-url")

        draft_parsed = urlparse(draft_url)
        target_parsed = urlparse(target_url)
        d_backend = draft_backend or init_wizard.detect_backend(draft_url)
        t_backend = target_backend or init_wizard.detect_backend(target_url)

        draft_server = init_wizard.DiscoveredServer(
            host=draft_parsed.hostname,
            port=draft_parsed.port or 80,
            backend=d_backend,
            models=[draft_model],
        )
        target_server = init_wizard.DiscoveredServer(
            host=target_parsed.hostname,
            port=target_parsed.port or 80,
            backend=t_backend,
            models=[target_model],
        )

        yaml_str = init_wizard.generate_cluster_yaml(
            draft_server=draft_server,
            draft_model=draft_model,
            target_server=target_server,
            target_model=target_model,
            max_draft_tokens=max_draft_tokens,
        )

        if output_path.exists() and not yes:
            overwrite = input(f"{output_path} already exists. Overwrite? [y/N] ").strip().lower()
            if overwrite != "y":
                console.print("[dim]Cancelled.[/dim]")
                return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_str)
        console.print(f"[green]✓[/green] Config written to {output_path}")
        return

    # Interactive mode: scan LAN
    if output_path.exists() and not yes:
        overwrite = input(f"{output_path} already exists. Overwrite? [y/N] ").strip().lower()
        if overwrite != "y":
            console.print("[dim]Cancelled.[/dim]")
            return

    console.print("[bold]Scanning LAN for inference servers...[/bold]\n")
    result = asyncio.run(init_wizard.scan_lan(
        subnet=subnet,
        extra_ports=list(extra_ports) if extra_ports else None,
    ))

    init_wizard.run_wizard(console, result, output_path)


@cli.command()
@click.argument("service", default="coordinator", type=click.Choice(["coordinator", "proxy"]))
@click.option("-f", "--follow", is_flag=True, help="Live-tail the log (like tail -f)")
@click.option("--clear", is_flag=True, help="Truncate log files")
@click.option("-n", "--lines", default=50, help="Number of lines to show (default: 50)")
def logs(service, follow, clear, lines):
    """View coordinator or proxy logs."""
    log_file = COORDINATOR_LOG if service == "coordinator" else PROXY_LOG

    if clear:
        for lf in [COORDINATOR_LOG, PROXY_LOG]:
            if lf.exists():
                lf.write_text("")
        console.print("[green]Logs cleared.[/green]")
        return

    if not log_file.exists():
        console.print(f"[dim]No log file yet: {log_file}[/dim]")
        console.print(f"[dim]Start the {service} to generate logs.[/dim]")
        return

    if follow:
        import subprocess as sp
        try:
            sp.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
        return

    # Show last N lines
    text = log_file.read_text()
    tail = text.splitlines()[-lines:]
    if not tail:
        console.print("[dim]Log file is empty.[/dim]")
    else:
        for line in tail:
            console.print(line, highlight=False)


@cli.command()
@click.option("--fix", is_flag=True, help="Show suggested fix commands for failures")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON report")
@click.pass_context
def doctor(ctx, fix, as_json):
    """Diagnose cluster configuration, connectivity, and version issues."""
    report = doctor_mod.run_doctor(ctx.obj.get("config_path"))

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        doctor_mod.render_report(console, report, show_fix=fix)

    if not report.passed:
        sys.exit(1)


@cli.command()
@click.option("--pid", "target_pid", type=int, default=None,
              help="PID of llama-server process (auto-detected from pidfile)")
@click.option("--model-path", type=click.Path(), default=None,
              help="Path to GGUF model file (auto-detected on Linux)")
def reclaim(target_pid, model_path):
    """Reclaim RAM from a running llama-server after model loading.

    Tells the OS to release file-backed pages from a llama-server process.
    On Linux uses posix_fadvise(DONTNEED), on Windows trims the working set.
    On macOS this is a no-op (unified memory).

    The coordinator PID is auto-detected from the pidfile if --pid is not given.
    On Linux, the model path is auto-detected from /proc/{pid}/maps.
    """
    from ..reclaim import reclaim_ram
    from ..coordinator import PIDFILE

    pid = target_pid
    if pid is None:
        from ..coordinator import _read_pidfile
        pidfile_data = _read_pidfile()
        if pidfile_data is None:
            console.print("[red]No coordinator running and no --pid specified.[/red]")
            console.print("Start the coordinator first, or provide --pid.")
            sys.exit(1)
        pid = pidfile_data["pid"]
        # Verify it's alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            console.print(f"[red]PID {pid} from pidfile is not running.[/red]")
            sys.exit(1)
        except PermissionError:
            pass  # process exists but owned by another user — proceed

    console.print(f"[bold]Reclaiming RAM from PID {pid}...[/bold]")
    result = reclaim_ram(pid, model_path)

    if result.method == "failed":
        console.print(f"  RAM reclaim: [yellow]failed[/yellow] ({result.error})")
    elif result.method == "skipped":
        msg = result.error or "not applicable on this platform"
        console.print(f"  RAM reclaim: [dim]skipped[/dim] ({msg})")
    else:
        console.print(f"  RSS before: {result.rss_before_mb:,.1f} MB")
        console.print(f"  RSS after:  {result.rss_after_mb:,.1f} MB")
        console.print(
            f"  [green]Reclaimed {result.reclaimed_mb:,.0f} MB "
            f"({result.method})[/green]"
        )


@cli.command()
@click.option("--model", "model_path", type=click.Path(exists=True), default=None,
              help="Model file to check RAM sufficiency against")
def tune(model_path):
    """Diagnose system RAM/swap and recommend tuning for large models."""
    from ..tune import diagnose, recommend

    info = diagnose()

    console.print("[bold]System Resources:[/bold]")
    console.print(f"  RAM:        {info.total_ram_gb:.1f} GB ({info.available_ram_gb:.1f} GB available)")
    console.print(f"  Swap:       {info.swap_total_gb:.1f} GB ({info.swap_used_gb:.1f} GB used)")
    if info.vm_swappiness is not None:
        console.print(f"  Swappiness: {info.vm_swappiness}")
    if info.swap_on_nvme is not None:
        nvme_str = "[green]yes[/green]" if info.swap_on_nvme else "[yellow]no[/yellow]"
        console.print(f"  Swap NVMe:  {nvme_str}")

    model_size_gb = None
    if model_path:
        model_size = Path(model_path).stat().st_size
        model_size_gb = model_size / (1024**3)
        console.print(f"\n[bold]Model:[/bold] {Path(model_path).name} ({model_size_gb:.1f} GB)")

    recs = recommend(info, model_size_gb)
    console.print()

    severity_icons = {
        "critical": "[red bold][!] CRITICAL:[/red bold]",
        "warn": "[yellow][!] WARNING:[/yellow]",
        "info": "[dim][i][/dim]",
    }

    for rec in recs:
        icon = severity_icons.get(rec.severity, "[dim][i][/dim]")
        console.print(f"  {icon} {rec.message}")
        if rec.commands:
            console.print()
            for cmd in rec.commands:
                console.print(f"      {cmd}")
            console.print()


@cli.command("load")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mem-limit", default=None, type=str,
              help="Memory limit (e.g. '2G', '4096M'). Reserved for v0.1.5 --force-constrain.")
@click.option("--no-prewarm", is_flag=True, help="Skip sequential pre-warming")
@click.option("--ram-reclaim", type=click.Choice(["off", "on", "auto"]), default=None,
              help="RAM reclaim mode after loading (default: auto)")
@click.option("--timeout", default=300.0, type=float, help="Health check timeout in seconds")
@click.pass_context
def load_cmd(ctx, model_path, mem_limit, no_prewarm, ram_reclaim, timeout):
    """Load a GGUF model with pre-warming and memory-aware startup.

    Pre-warms the page cache sequentially before llama-server mmaps the file,
    then reclaims RAM after the model loads to VRAM. Use this for standalone
    loading of any GGUF file.

    For models configured in cluster.yaml, use 'tightwad start' instead —
    it integrates pre-warming automatically when ram_reclaim is 'auto' or 'on'.
    """
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from ..loader import (
        load_model, needs_streaming_load, prewarm_sequential,
    )
    from ..reclaim import get_available_ram_bytes
    from ..gguf_reader import read_header, model_summary

    model_path = Path(model_path)
    file_size = model_path.stat().st_size
    model_size_gb = file_size / (1024**3)

    if mem_limit is not None:
        console.print(
            "[yellow]--mem-limit is reserved for v0.1.5 (--force-constrain). "
            "Ignored in this release.[/yellow]"
        )

    # Parse GGUF header for display
    gguf_info = None
    try:
        header = read_header(model_path)
        gguf_info = model_summary(header)
    except Exception:
        pass

    console.print(f"\n[bold]Model:[/bold] {model_path.name}")
    if gguf_info:
        parts = []
        if gguf_info.get("arch"):
            parts.append(gguf_info["arch"])
        if gguf_info.get("layers"):
            parts.append(f"{gguf_info['layers']} layers")
        if gguf_info.get("quant"):
            parts.append(gguf_info["quant"])
        parts.append(f"{model_size_gb:.1f} GB")
        console.print(f"  {', '.join(parts)}")
    else:
        console.print(f"  Size: {model_size_gb:.1f} GB")

    available = get_available_ram_bytes()
    available_gb = available / (1024**3)
    console.print(f"\n[bold]System:[/bold] {available_gb:.1f} GB RAM available")

    needs_prewarm = needs_streaming_load(file_size, available)
    if needs_prewarm and not no_prewarm:
        console.print(f"  Strategy: pre-warm + reclaim (model > 80% of available RAM)")
    elif no_prewarm:
        console.print(f"  Strategy: skip pre-warm (--no-prewarm)")
    else:
        console.print(f"  Strategy: direct load (model fits in RAM)")

    # Pre-warm with progress bar
    if needs_prewarm and not no_prewarm:
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed:.1f}/{task.total:.1f} GB"),
            console=console,
        ) as progress:
            task = progress.add_task("Pre-warming...", total=model_size_gb)

            def on_progress(bytes_read, total):
                progress.update(task, completed=bytes_read / (1024**3))

            elapsed = prewarm_sequential(
                model_path, file_size, progress_callback=on_progress,
            )
            throughput = model_size_gb / elapsed if elapsed > 0 else 0

        console.print(f"  Pre-warm: {elapsed:.1f}s ({throughput:.2f} GB/s)")

    # Start coordinator via config
    config = _load(ctx)
    mode = ram_reclaim or config.ram_reclaim

    console.print(f"\n[bold]Starting coordinator...[/bold]")
    try:
        result = load_model(
            config,
            model_name=str(model_path),
            prewarm=False,  # already pre-warmed above
            ram_reclaim=mode,
            wait_timeout=timeout,
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if result.healthy:
        console.print(f"  Health check: [green]OK[/green] ({result.load_time_seconds:.1f}s)")
    else:
        console.print(f"  Health check: [yellow]timeout[/yellow] ({timeout:.0f}s)")

    if result.reclaim_result:
        r = result.reclaim_result
        if r.method != "skipped":
            console.print(
                f"  [green]Reclaimed {r.reclaimed_mb:,.0f} MB RAM ({r.method})[/green]"
            )

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  PID:       {result.pid}")
    console.print(f"  Peak RAM:  {result.peak_rss_mb:.1f} MB")
    console.print(f"  Model:     {result.model_size_gb:.1f} GB")
    console.print(f"  Load time: {result.load_time_seconds:.1f}s")


@cli.command("deploy")
@click.argument("host")
@click.option("--ssh-user", default="", help="SSH username (default: current user)")
@click.option("--config", "deploy_config", default=None, type=click.Path(exists=True),
              help="Config file to deploy (copied to remote)")
def deploy_cmd(host, ssh_user, deploy_config):
    """Deploy tightwad to a remote host via SSH.

    Installs tightwad, copies config, starts the coordinator, and verifies health.
    """
    from ..deploy import deploy

    console.print(f"[bold]Deploying to {host}...[/bold]")

    result = deploy(host, ssh_user=ssh_user, config_path=deploy_config)

    for step in result.steps_completed:
        console.print(f"  [green]✓[/green] {step}")

    if result.success:
        console.print(f"\n[green bold]Deployment complete![/green bold]")
        console.print(f"  API: http://{host}:8080/v1")
    else:
        console.print(f"\n[red]{result.message}[/red]")
        sys.exit(1)


@cli.command("pull")
@click.argument("model_spec", default="")
@click.option("--dir", "model_dir", default=None, type=click.Path(),
              help="Download directory (default: ~/.tightwad/models/)")
@click.option("--list", "list_models", is_flag=True, help="List available models")
def pull(model_spec, model_dir, list_models):
    """Download a GGUF model from HuggingFace.

    MODEL_SPEC can be a registry name (e.g. llama3.3:70b-q4_k_m), a direct URL,
    or a HuggingFace repo path (e.g. org/repo/file.gguf).
    """
    from ..model_hub import (
        resolve_model,
        download_model,
        validate_download,
        list_models as hub_list_models,
    )

    if list_models:
        from rich.table import Table
        table = Table(title="Available Models")
        table.add_column("Spec", style="bold")
        table.add_column("Repository")
        table.add_column("Filename")
        for spec, repo, filename in hub_list_models():
            table.add_row(spec, repo, filename)
        console.print(table)
        return

    if not model_spec:
        console.print("[red]Provide a model spec or use --list to see available models.[/red]")
        console.print("  tightwad pull llama3.3:70b-q4_k_m")
        console.print("  tightwad pull --list")
        sys.exit(1)

    try:
        resolved = resolve_model(model_spec)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    dest_dir = Path(model_dir) if model_dir else None
    console.print(f"[bold]Downloading {resolved.filename}...[/bold]")
    console.print(f"  URL: {resolved.hf_url}")

    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, DownloadColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=None)

        def on_progress(downloaded, total):
            if total and total > 0:
                progress.update(task, total=total, completed=downloaded)

        try:
            path = download_model(
                resolved.hf_url,
                dest_dir=dest_dir,
                filename=resolved.filename,
                progress_callback=on_progress,
            )
        except Exception as e:
            console.print(f"\n[red]Download failed: {e}[/red]")
            sys.exit(1)

    # Validate
    if validate_download(path):
        size_gb = path.stat().st_size / (1024 ** 3)
        console.print(f"\n[green]✓[/green] Downloaded: {path} ({size_gb:.1f} GB)")
    else:
        console.print(f"\n[yellow]Warning: {path} does not appear to be a valid GGUF file[/yellow]")


@cli.command("inspect")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--plan", "show_plan", is_flag=True, help="Show distribution plan for current cluster")
@click.pass_context
def inspect_cmd(ctx, model_path, show_plan):
    """Inspect a GGUF model file: metadata, tensors, distribution plan."""
    try:
        from ..gguf_inspect import inspect_model, plan_distribution, format_report
    except ImportError:
        console.print("[red]Missing gguf package. Install with: pip install tightwad[inspect][/red]")
        sys.exit(1)

    model_info = inspect_model(model_path)
    plan = None
    if show_plan:
        config = _load(ctx)
        plan = plan_distribution(model_info, config)

    output = format_report(model_info, plan)
    console.print(output)


@cli.command("distribute")
@click.argument("model_name")
@click.option("-t", "--target", "specific_target", default=None, help="Specific target as host:/path")
@click.option("--method", type=click.Choice(["auto", "rsync", "swarm"]), default="auto",
              help="Transfer method (default: auto-select by file size)")
@click.option("--token", default=None, help="Bearer token for swarm auth")
@click.option("--dry-run", is_flag=True, help="Preview transfers without executing")
@click.pass_context
def distribute_cmd(ctx, model_name, specific_target, method, token, dry_run):
    """Distribute a model to worker machines via rsync/scp or swarm P2P."""
    from ..distribute import (
        resolve_targets, distribute, distribute_swarm,
        format_dry_run, auto_select_method,
    )

    config = _load(ctx)
    try:
        local_path, targets = resolve_targets(config, model_name, specific_target)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if not targets:
        console.print("[yellow]No targets with model_dir configured. "
                       "Add model_dir to workers in config or use -t host:/path.[/yellow]")
        sys.exit(1)

    # Resolve auto method
    if method == "auto":
        if local_path.exists():
            method = auto_select_method(local_path)
        else:
            method = "rsync"  # can't check size, fall back
        console.print(f"[dim]Auto-selected method: {method}[/dim]")

    if dry_run:
        console.print(format_dry_run(local_path, targets, method=method, token=token))
        return

    if not local_path.exists():
        console.print(f"[red]Model file not found: {local_path}[/red]")
        sys.exit(1)

    console.print(f"[bold]Distributing {local_path.name} to {len(targets)} target(s) via {method}...[/bold]\n")

    if method == "swarm":
        results = distribute_swarm(local_path, targets, console, token=token)
    else:
        results = distribute(local_path, targets, console)

    failed = [r for r in results if not r.success]
    if failed:
        console.print(f"\n[red]{len(failed)} transfer(s) failed:[/red]")
        for r in failed:
            console.print(f"  {r.target.worker_name}: {r.message}")
        sys.exit(1)
    else:
        console.print(f"\n[green]All {len(results)} transfer(s) complete.[/green]")


@cli.command()
@click.option("--direct", is_flag=True, help="Chat directly with target (no speculation, for comparison)")
@click.pass_context
def chat(ctx, direct):
    """Interactive chat with the speculative decoding proxy."""
    import httpx
    config = _load(ctx)
    if config.proxy is None:
        console.print("[red]No proxy section in config. Add it to cluster.yaml.[/red]")
        sys.exit(1)
    pc = config.proxy
    if direct:
        base_url = pc.target.url
        console.print(f"\n[bold]Direct mode:[/bold] {pc.target.model_name} @ {pc.target.url}")
    else:
        base_url = f"http://127.0.0.1:{pc.port}"
        try:
            httpx.get(f"{base_url}/v1/tightwad/status", timeout=3.0)
        except Exception:
            console.print("[red]Proxy not running. Start it first:[/red]")
            console.print("  tightwad proxy start")
            sys.exit(1)
        console.print(f"\n[bold]Speculative mode:[/bold] proxy @ :{pc.port} -> {pc.target.model_name}")
    console.print("[dim]Type your message and press Enter. Ctrl+C to quit.[/dim]\n")
    messages: list[dict] = []
    prev_stats: dict | None = None
    status_url = f"{base_url}/v1/tightwad/status" if not direct else None
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})
        try:
            import time as _time
            url = f"{base_url}/v1/chat/completions"
            body = {"messages": messages, "max_tokens": 1024, "temperature": 0.0, "stream": False}
            t0 = _time.monotonic()
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
            elapsed = _time.monotonic() - t0
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not text:
                text = data.get("choices", [{}])[0].get("text", "")
            console.print(f"[bold cyan]AI:[/bold cyan] {text}")
            messages.append({"role": "assistant", "content": text})

            # Inline speculation stats (speculative mode only)
            if status_url:
                try:
                    sr = httpx.get(status_url, timeout=3.0)
                    cur = sr.json().get("stats", {})
                    if prev_stats and cur.get("total_rounds", 0) > prev_stats.get("total_rounds", 0):
                        dr = cur["total_rounds"] - prev_stats["total_rounds"]
                        dd = cur["total_drafted"] - prev_stats["total_drafted"]
                        da = cur["total_accepted"] - prev_stats["total_accepted"]
                        rate = da / dd * 100 if dd > 0 else 0
                        tpr = da / dr if dr > 0 else 0
                        tok_s = da / elapsed if elapsed > 0 else 0
                        console.print(
                            f"  [dim]↳ {dr} round{'s' if dr != 1 else ''}, "
                            f"{dd} drafted, {da} accepted ({rate:.1f}%), "
                            f"{tpr:.1f} tok/round, {tok_s:.1f} tok/s[/dim]"
                        )
                    prev_stats = cur
                except Exception:
                    pass

            console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]\n")
            messages.pop()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            messages.pop()


# --- Manifest subcommand group ---


@cli.group()
def manifest():
    """Swarm manifest commands."""
    pass


@manifest.command("create")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--piece-size", default=64, type=int, help="Piece size in MB (default: 64)")
@click.option("--no-inspect", is_flag=True, help="Skip GGUF metadata inspection")
@click.option("-o", "--output", "output_path", default=None, type=click.Path(), help="Output manifest path")
def manifest_create(model_path, piece_size, no_inspect, output_path):
    """Create a swarm manifest for a GGUF model file."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    model_path = Path(model_path)
    piece_bytes = piece_size * 1024 * 1024
    total_size = model_path.stat().st_size
    est_pieces = (total_size + piece_bytes - 1) // piece_bytes

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} pieces"),
        console=console,
    ) as progress:
        task = progress.add_task("Hashing pieces...", total=est_pieces)

        def on_progress(done, _total):
            progress.update(task, completed=done)

        m = manifest_mod.create_manifest(
            model_path,
            piece_size=piece_bytes,
            use_gguf_inspect=not no_inspect,
            progress_callback=on_progress,
        )

    if output_path is None:
        output_path = model_path.parent / f"{model_path.name}.tightwad.manifest"
    else:
        output_path = Path(output_path)

    m.save(output_path)
    console.print(f"\n[green]Manifest created:[/green] {output_path}")
    console.print(f"  Model:    {m.model}")
    console.print(f"  Size:     {m.total_size / (1024**3):.2f} GB")
    console.print(f"  Pieces:   {m.num_pieces} x {piece_size} MB")
    if m.metadata:
        console.print(f"  Metadata: {json.dumps(m.metadata)}")


# --- Swarm subcommand group ---


@cli.group()
def swarm():
    """Swarm P2P transfer commands."""
    pass


@swarm.command("seed")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--port", default=9080, type=int, help="Seeder port (default: 9080)")
@click.option("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
@click.option("--token", default=None, help="Require Bearer token for all requests")
@click.option("--allowed-ips", default=None, multiple=True, help="Restrict access to IP/CIDR (repeatable)")
def swarm_seed(model_path, port, host, token, allowed_ips):
    """Start a swarm seeder for a model file."""

    model_path = Path(model_path)

    # Load or create manifest
    m = manifest_mod.SwarmManifest.find_for_model(model_path)
    if m is None:
        console.print("[yellow]No manifest found. Creating one...[/yellow]")
        m = manifest_mod.create_manifest(model_path)
        manifest_file = model_path.parent / f"{model_path.name}.tightwad.manifest"
        m.save(manifest_file)
        console.print(f"[green]Manifest saved:[/green] {manifest_file}")

    # Build full bitfield (we have the complete file)
    bf = manifest_mod.PieceBitfield.load_or_create(
        model_path.parent / f"{model_path.name}.tightwad.pieces",
        m.num_pieces,
    )
    # Verify we have all pieces if bitfield is empty
    if not bf.have_all():
        for piece in m.pieces:
            if manifest_mod.verify_piece(model_path, piece):
                bf.mark_have(piece.index)
        bf.save()

    console.print(f"\n[bold]Starting swarm seeder...[/bold]")
    console.print(f"  Model:  {m.model} ({m.filename})")
    console.print(f"  Pieces: {len(bf.have)}/{m.num_pieces} ({bf.completion_pct():.0f}%)")
    console.print(f"  Listen: {host}:{port}")
    if token:
        console.print(f"  Auth:   Bearer token required")
    if allowed_ips:
        console.print(f"  IPs:    {', '.join(allowed_ips)}")

    swarm_mod.run_seeder(
        model_path, m, bf, host=host, port=port,
        token=token, allowed_ips=list(allowed_ips) if allowed_ips else None,
    )


@swarm.command("pull")
@click.argument("dest_path", type=click.Path())
@click.option("--manifest", "manifest_source", required=True, help="Path or URL to manifest")
@click.option("--peer", "peers", multiple=True, required=True, help="Peer URL (repeatable)")
@click.option("--parallel", default=4, type=int, help="Max concurrent downloads (default: 4)")
@click.option("--token", default=None, help="Bearer token for authenticated peers")
def swarm_pull(dest_path, manifest_source, peers, parallel, token):
    """Pull a model from swarm peers."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    dest_path = Path(dest_path)

    # Load manifest from file or URL
    if manifest_source.startswith("http://") or manifest_source.startswith("https://"):
        import httpx
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        console.print(f"Fetching manifest from {manifest_source}...")
        resp = httpx.get(manifest_source, timeout=30.0, headers=headers)
        resp.raise_for_status()
        m = manifest_mod.SwarmManifest.from_dict(resp.json())
    else:
        m = manifest_mod.SwarmManifest.load(manifest_source)

    # Load or create bitfield for destination
    bf = manifest_mod.PieceBitfield.load_or_create(
        dest_path.parent / f"{dest_path.name}.tightwad.pieces",
        m.num_pieces,
    )

    missing = bf.missing_pieces()
    console.print(f"\n[bold]Pulling {m.filename}[/bold]")
    console.print(f"  Pieces: {m.num_pieces} total, {len(missing)} to download")
    console.print(f"  Peers:  {len(peers)}")
    console.print(f"  Parallel: {parallel}")

    if not missing:
        console.print("\n[green]Already complete![/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading pieces...", total=len(missing))

        def on_progress(completed, total, piece_idx):
            progress.update(task, completed=completed, description=f"Piece {piece_idx}")

        ok = swarm_mod.run_puller(
            dest_path, m, bf, list(peers),
            max_concurrent=parallel,
            progress_callback=on_progress,
            token=token,
        )

    if ok:
        console.print(f"\n[green]Download complete:[/green] {dest_path}")
    else:
        console.print(f"\n[yellow]Download incomplete. Re-run to resume.[/yellow]")
        sys.exit(1)


@swarm.command("status")
@click.argument("model_path", type=click.Path(exists=True))
def swarm_status(model_path):
    """Show swarm status for a model file."""

    model_path = Path(model_path)

    m = manifest_mod.SwarmManifest.find_for_model(model_path)
    if m is None:
        console.print("[dim]No manifest found. Create one with:[/dim]")
        console.print(f"  tightwad manifest create {model_path}")
        return

    bf = manifest_mod.PieceBitfield.load_or_create(
        model_path.parent / f"{model_path.name}.tightwad.pieces",
        m.num_pieces,
    )

    # Check if we have the full file but empty bitfield
    if not bf.have and model_path.exists():
        console.print("[dim]Verifying pieces...[/dim]")
        for piece in m.pieces:
            if manifest_mod.verify_piece(model_path, piece):
                bf.mark_have(piece.index)
        bf.save()

    pct = bf.completion_pct()
    missing = bf.missing_pieces()

    console.print(f"\n[bold]Swarm Status:[/bold] {m.filename}")
    console.print(f"  Model:      {m.model}")
    console.print(f"  Size:       {m.total_size / (1024**3):.2f} GB")
    console.print(f"  Pieces:     {m.num_pieces} x {m.piece_size // (1024*1024)} MB")
    console.print(f"  Have:       {len(bf.have)}/{m.num_pieces} ({pct:.0f}%)")
    if missing:
        console.print(f"  Missing:    {len(missing)} pieces")
    else:
        console.print(f"  [green]Complete![/green]")
    if m.metadata:
        console.print(f"  Metadata:   {json.dumps(m.metadata)}")

    # Check for running seeder
    pid = swarm_mod.read_seeder_pidfile(m.model)
    if pid is not None:
        try:
            os.kill(pid, 0)
            console.print(f"  Seeder:     [green]running[/green] (PID {pid})")
        except ProcessLookupError:
            console.print(f"  Seeder:     [dim]not running[/dim]")
