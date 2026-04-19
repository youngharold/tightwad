"""MoE commands: plan, profile, summary, bench, defuse."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.table import Table

from . import cli, console, _load


@cli.group()
def moe():
    """Mixture-of-Experts placement, profiling, defusion, and benchmarking."""


# ---------------------------------------------------------------------------
# moe plan
# ---------------------------------------------------------------------------


@moe.command("plan")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--strategy", type=click.Choice(["balanced", "profile-guided"]),
              default="balanced", show_default=True)
@click.option("--hot-profile", type=click.Path(exists=True), default=None,
              help="JSON profile from `tightwad moe profile`. Required for profile-guided.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
@click.option("--emit-ot", is_flag=True,
              help="Print only the --override-tensor flags (shell-paste ready).")
@click.pass_context
def plan_cmd(ctx, model_path, strategy, hot_profile, as_json, emit_ot):
    """Generate an expert-aware placement plan for the current cluster."""
    from ..gguf_inspect import inspect_model
    from ..moe_placement import build_slots, plan_expert_placement
    from ..moe_device_bench import measure_device_scores

    config = _load(ctx)
    info = inspect_model(model_path)
    if not info.is_moe:
        console.print("[red]Model is not MoE — no expert placement to plan.[/red]")
        sys.exit(1)

    slots = build_slots(config)
    if not slots:
        console.print("[red]No GPUs in cluster config.[/red]")
        sys.exit(1)

    hot = None
    if strategy == "profile-guided":
        if not hot_profile:
            console.print("[red]--hot-profile is required for --strategy profile-guided[/red]")
            sys.exit(1)
        from ..moe_profile import HotExpertProfile
        hot = HotExpertProfile.load(hot_profile).frequency()

    scores = measure_device_scores(config) if strategy == "profile-guided" else None
    plan = plan_expert_placement(
        info, slots, hot_experts=hot, device_scores=scores, strategy=strategy,
    )

    if emit_ot:
        for flag in plan.to_cli_flags():
            click.echo(f"--override-tensor {flag}")
        return

    if as_json:
        click.echo(json.dumps(plan.to_dict(), indent=2))
        return

    _render_plan_table(plan, info, slots)


def _render_plan_table(plan, info, slots):
    if plan.fused_fallback:
        console.print(
            "[yellow]Fused expert tensors detected. Run "
            "`tightwad moe defuse` to enable per-expert placement.[/yellow]"
        )
        for w in plan.warnings:
            console.print(f"[yellow]  {w}[/yellow]")
        return

    table = Table(title=f"Expert placement — {info.moe.n_expert} experts × {info.n_layers} layers")
    table.add_column("Device")
    table.add_column("Host")
    table.add_column("Assigned experts")
    table.add_column("GB", justify="right")

    by_device: dict[str, dict] = {}
    for a in plan.assignments:
        entry = by_device.setdefault(a.device.ot_device, {
            "host": a.device.host, "count": 0,
        })
        entry["count"] += 1
    for slot in slots:
        entry = by_device.get(slot.ot_device, {"host": slot.host, "count": 0})
        gb = plan.per_device_bytes.get(slot.ot_device, 0) / (1024 ** 3)
        table.add_row(slot.ot_device, entry["host"], str(entry["count"]), f"{gb:.2f}")
    console.print(table)
    console.print(f"\n[bold]{len(plan.override_tensor_args)}[/bold] --override-tensor flags will be emitted.")


# ---------------------------------------------------------------------------
# moe profile
# ---------------------------------------------------------------------------


@moe.command("profile")
@click.option("--from-log", type=click.Path(exists=True), default=None,
              help="Parse an existing log file (skips --follow-*).")
@click.option("--follow-coord", is_flag=True,
              help="Tail ~/.tightwad/logs/coordinator.log.")
@click.option("--follow-peer", default=None, metavar="HOST:PORT",
              help="Pull aggregated counts from a peer agent's /v1/peer/moe/profile.")
@click.option("--rpc-port", type=int, default=None,
              help="When using --follow-peer, the rpc-server port to query.")
@click.option("--duration", type=int, default=60, show_default=True,
              help="Capture window in seconds (applies to live follow).")
@click.option("-o", "--output", type=click.Path(), default="~/.tightwad/moe-profile.json",
              show_default=True)
def profile_cmd(from_log, follow_coord, follow_peer, rpc_port, duration, output):
    """Capture per-expert routing counts from a running cluster."""
    import httpx
    from ..moe_profile import HotExpertProfile, parse_log_file
    from ..coordinator import COORDINATOR_LOG

    if from_log:
        profile = parse_log_file(from_log)
    elif follow_coord:
        profile = parse_log_file(COORDINATOR_LOG)
    elif follow_peer:
        if not rpc_port:
            console.print("[red]--rpc-port is required with --follow-peer[/red]")
            sys.exit(1)
        url = f"http://{follow_peer}/v1/peer/moe/profile?port={rpc_port}"
        try:
            resp = httpx.get(url, timeout=10)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            console.print(f"[red]Failed to query peer: {exc}[/red]")
            sys.exit(1)
        data = resp.json()
        profile = HotExpertProfile(total_tokens=data.get("total_tokens", 0),
                                    source=f"peer {follow_peer}")
        for h in data.get("top_experts", []):
            profile.hits[(h["layer"], h["expert"])] = h["count"]
    else:
        console.print(
            "[red]Specify one of --from-log / --follow-coord / --follow-peer[/red]"
        )
        sys.exit(1)

    out = Path(output).expanduser()
    profile.save(out)
    console.print(f"[green]Saved profile to {out}[/green]")
    console.print(
        f"  tokens: {profile.total_tokens}   unique experts: {len(profile.hits)}"
    )


# ---------------------------------------------------------------------------
# moe summary
# ---------------------------------------------------------------------------


@moe.command("summary")
@click.argument("profile", type=click.Path(exists=True))
@click.option("--top", type=int, default=20, show_default=True)
def summary_cmd(profile, top):
    """Show top hot experts and per-layer concentration from a captured profile."""
    from ..moe_profile import HotExpertProfile

    p = HotExpertProfile.load(profile)
    console.print(f"[bold]Profile:[/bold] {profile}")
    console.print(f"  tokens: {p.total_tokens}   unique experts: {len(p.hits)}")

    table = Table(title=f"Top {top} hot experts")
    table.add_column("Layer", justify="right")
    table.add_column("Expert", justify="right")
    table.add_column("Hits", justify="right")
    for h in p.top_n(top):
        table.add_row(str(h.layer), str(h.expert), str(h.count))
    console.print(table)

    skew = p.per_layer_skew()
    if skew:
        avg_skew = sum(skew.values()) / len(skew)
        console.print(f"\nAverage top-10% skew: {avg_skew:.2%}")


# ---------------------------------------------------------------------------
# moe bench
# ---------------------------------------------------------------------------


@moe.command("bench")
@click.option("--target-url", required=True, help="OpenAI-compatible target URL (e.g. LM Studio on :1234).")
@click.option("--target-model", required=True, help="Model name to send in the request body.")
@click.option("--prompts", type=click.Path(exists=True), default=None,
              help="JSON or text prompt file. Default: built-in MoE prompt set.")
@click.option("--max-tokens", type=int, default=256, show_default=True)
@click.option("--warmup", type=int, default=2, show_default=True)
@click.option("--json", "json_out", type=click.Path(), default=None,
              help="Write results JSON to this path.")
@click.option("--live/--no-live", default=True, show_default=True,
              help="Stream per-prompt results as they arrive.")
@click.pass_context
def bench_cmd(ctx, target_url, target_model, prompts, max_tokens, warmup,
              json_out, live):
    """MoE benchmark — compares proxy vs direct target with streaming progress."""
    import asyncio
    from ..bench import run_moe_benchmark

    config = _load(ctx)
    if config.proxy is None:
        console.print("[red]No proxy configured in cluster.yaml[/red]")
        sys.exit(1)
    proxy_url = f"http://{config.proxy.host}:{config.proxy.port}"

    prompt_list = _load_prompts(prompts)

    try:
        result = asyncio.run(run_moe_benchmark(
            proxy_url=proxy_url,
            target_url=target_url,
            target_model=target_model,
            prompts=prompt_list,
            max_tokens=max_tokens,
            warmup=warmup,
            live=live,
            on_update=(_render_live_row if live else None),
        ))
    except Exception as exc:
        console.print(f"[red]Benchmark failed: {exc}[/red]")
        sys.exit(1)

    if json_out:
        Path(json_out).expanduser().write_text(json.dumps(result, indent=2))
        console.print(f"[green]Wrote results to {json_out}[/green]")
    else:
        console.print(json.dumps(result, indent=2))


_DEFAULT_MOE_PROMPTS = [
    {"name": "reasoning", "text": "Solve step by step: a train leaves Chicago at 9 AM traveling 60 mph. Another leaves St. Louis at 10 AM traveling 75 mph toward Chicago. The cities are 295 miles apart. When do they meet?"},
    {"name": "code", "text": "Write a Python function that returns the longest palindromic substring of its input."},
    {"name": "factual", "text": "Summarize the Treaty of Westphalia in three sentences."},
    {"name": "translate", "text": "Translate to German: 'The rain in Spain falls mainly on the plain.'"},
    {"name": "multilingual", "text": "Explain recursion in English, Spanish, and Japanese, one sentence each."},
]


def _load_prompts(path: str | None) -> list[dict]:
    if not path:
        return _DEFAULT_MOE_PROMPTS
    text = Path(path).read_text()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return [{"name": f"p{i}", "text": line}
            for i, line in enumerate(text.splitlines()) if line.strip()]


def _render_live_row(update: dict) -> None:
    name = update.get("name", "?")
    ttft = update.get("ttft_ms")
    direct = update.get("direct_tps")
    proxy = update.get("proxy_tps")
    accept = update.get("acceptance_rate")
    speedup = update.get("speedup")
    console.print(
        f"[cyan]{name:<14}[/cyan]  "
        f"TTFT {ttft:>6.0f}ms  "
        f"direct {direct or 0:>6.1f} tok/s  "
        f"proxy {proxy or 0:>6.1f} tok/s  "
        f"accept {accept or 0:>5.1%}  "
        f"speedup {speedup or 0:>4.2f}x"
    )


# ---------------------------------------------------------------------------
# moe defuse
# ---------------------------------------------------------------------------


@moe.command("defuse")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def defuse_cmd(input_path, output_path):
    """Rewrite fused expert tensors in a GGUF to indexed form."""
    from ..moe_defuse import defuse_gguf

    console.print(f"Defusing {input_path} → {output_path}")
    try:
        summary = defuse_gguf(input_path, output_path)
    except Exception as exc:
        console.print(f"[red]Defusion failed: {exc}[/red]")
        sys.exit(1)
    console.print(
        f"[green]Done. n_expert={summary['n_expert']}, "
        f"layers_defused={summary['layers_defused']}, "
        f"tensors_written={summary['tensors_written']}[/green]"
    )
