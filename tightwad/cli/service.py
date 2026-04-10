"""Service subcommand group: install, uninstall, status."""

from __future__ import annotations

import sys

import click

from . import cli, console


@cli.group("service")
def service_group():
    """Manage tightwad as a system service (systemd/launchd)."""
    pass


@service_group.command("install")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Path to tightwad config YAML")
@click.option("--user/--system", default=True, help="Install as user service (default) or system service")
def service_install(config_path, user):
    """Install tightwad as a system service."""
    from ..service import install_service

    try:
        plat, path = install_service(config_path, user=user)
        console.print(f"[green]✓[/green] Installed {plat} service: {path}")
        if plat == "systemd":
            console.print(f"\n  Start:  systemctl {'--user ' if user else ''}start tightwad")
            console.print(f"  Status: systemctl {'--user ' if user else ''}status tightwad")
        elif plat == "launchd":
            console.print(f"\n  The service will start automatically on login.")
            console.print(f"  Status: launchctl list com.tightwad.server")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@service_group.command("uninstall")
def service_uninstall():
    """Uninstall the tightwad system service."""
    from ..service import uninstall_service

    try:
        plat, was_installed = uninstall_service()
        if was_installed:
            console.print(f"[green]✓[/green] Uninstalled {plat} service")
        else:
            console.print(f"[yellow]No {plat} service found to uninstall[/yellow]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@service_group.command("status")
def service_status_cmd():
    """Check if tightwad service is installed and running."""
    from ..service import service_status

    st = service_status()
    if st["installed"]:
        running_str = "[green]running[/green]" if st["running"] else "[yellow]stopped[/yellow]"
        console.print(f"  Service: {running_str} ({st['platform']})")
        console.print(f"  Path:    {st['path']}")
    else:
        console.print(f"  [dim]Not installed ({st['platform']})[/dim]")
        console.print(f"  Install with: tightwad service install --config /path/to/config.yaml")
