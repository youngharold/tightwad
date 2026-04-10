"""CLI interface for Tightwad cluster management."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from ..config import load_config
from ..coordinator import LOGDIR

PROXY_LOG = LOGDIR / "proxy.log"

console = Console()


@click.group()
@click.version_option(package_name="tightwad")
@click.option(
    "-c", "--config",
    envvar="TIGHTWAD_CONFIG",
    default=None,
    help="Path to cluster.yaml config file",
)
@click.pass_context
def cli(ctx, config):
    """Tightwad — Mixed-vendor GPU inference cluster manager with speculative decoding."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


def _load(ctx) -> "ClusterConfig":
    try:
        return load_config(ctx.obj.get("config_path"))
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _parse_size(s: str) -> int:
    """Parse a human-readable size string to bytes. E.g. '2G', '4096M', '2147483648'."""
    s = s.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s and s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


# Import subcommand modules so they register their commands with the cli group.
# These imports MUST come after the cli group is defined.
from . import cluster  # noqa: E402, F401
from . import proxy  # noqa: E402, F401
from . import gate  # noqa: E402, F401
from . import peer  # noqa: E402, F401
from . import tools  # noqa: E402, F401
from . import service  # noqa: E402, F401
