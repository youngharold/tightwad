"""LAN auto-discovery wizard for generating cluster.yaml."""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import httpx
import yaml
from rich.console import Console
from rich.table import Table


# Default ports to scan
DEFAULT_PORTS = [11434, 8080, 8081]

# Concurrency limit for scanning
MAX_CONCURRENT = 128

# TCP connect timeout per host:port
SCAN_TIMEOUT = 1.5


@dataclass
class DiscoveredServer:
    host: str
    port: int
    backend: str  # "ollama" or "llamacpp"
    models: list[str] = field(default_factory=list)
    status: str = "unknown"


@dataclass
class ScanResult:
    servers: list[DiscoveredServer] = field(default_factory=list)
    subnet: str = ""
    scan_time_seconds: float = 0.0


def detect_subnet() -> str:
    """Detect the local subnet (assumes /24)."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except socket.gaierror:
        infos = []

    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip = sockaddr[0]
        # Skip loopback, link-local, Docker bridge
        if ip.startswith("127.") or ip.startswith("169.254.") or ip.startswith("172.17."):
            continue
        parts = ip.split(".")
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

    # Fallback: try connecting to a public DNS to find our IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        parts = ip.split(".")
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    except Exception:
        return "192.168.1.0/24"


def _subnet_hosts(subnet: str) -> list[str]:
    """Generate all 254 host IPs for a /24 subnet."""
    base = subnet.rsplit(".", 1)[0]  # e.g. "192.168.1" from "192.168.1.0/24"
    if "/" in base:
        base = base.split("/")[0]
    # Strip trailing .0 if present from "192.168.1.0"
    if base.endswith(".0"):
        base = base[:-2]
    return [f"{base}.{i}" for i in range(1, 255)]


async def _check_port(host: str, port: int, sem: asyncio.Semaphore) -> tuple[str, int] | None:
    """Try TCP connect to host:port. Returns (host, port) if open, else None."""
    async with sem:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=SCAN_TIMEOUT,
            )
            writer.close()
            await writer.wait_closed()
            return (host, port)
        except (OSError, asyncio.TimeoutError):
            return None


async def _scan_subnet(subnet: str, ports: list[int]) -> list[tuple[str, int]]:
    """Scan all hosts in a /24 subnet on given ports. Returns list of open (host, port)."""
    hosts = _subnet_hosts(subnet)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for host in hosts:
        for port in ports:
            tasks.append(_check_port(host, port, sem))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def identify_server(host: str, port: int) -> DiscoveredServer | None:
    """Identify what's running on a host:port (Ollama or llama-server)."""
    base = f"http://{host}:{port}"
    async with httpx.AsyncClient(timeout=3.0) as client:
        # Try Ollama first: GET / returns "Ollama is running"
        try:
            resp = await client.get(f"{base}/")
            if "ollama" in resp.text.lower():
                models = []
                try:
                    tags_resp = await client.get(f"{base}/api/tags")
                    if tags_resp.status_code == 200:
                        data = tags_resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                except Exception:
                    pass
                return DiscoveredServer(
                    host=host, port=port, backend="ollama",
                    models=models, status="healthy",
                )
        except Exception:
            pass

        # Try llama-server: GET /health returns 200 + JSON
        try:
            resp = await client.get(f"{base}/health")
            if resp.status_code == 200:
                models = []
                try:
                    models_resp = await client.get(f"{base}/v1/models")
                    if models_resp.status_code == 200:
                        data = models_resp.json()
                        models = [m["id"] for m in data.get("data", [])]
                except Exception:
                    pass
                return DiscoveredServer(
                    host=host, port=port, backend="llamacpp",
                    models=models, status="healthy",
                )
        except Exception:
            pass

    return None


async def scan_lan(subnet: str | None = None, extra_ports: list[int] | None = None) -> ScanResult:
    """Scan the LAN for inference servers."""
    if subnet is None:
        subnet = detect_subnet()

    ports = list(DEFAULT_PORTS)
    if extra_ports:
        for p in extra_ports:
            if p not in ports:
                ports.append(p)

    t0 = monotonic()
    open_endpoints = await _scan_subnet(subnet, ports)

    # Identify each open endpoint
    identify_tasks = [identify_server(host, port) for host, port in open_endpoints]
    identified = await asyncio.gather(*identify_tasks)

    servers = [s for s in identified if s is not None]
    elapsed = monotonic() - t0

    return ScanResult(servers=servers, subnet=subnet, scan_time_seconds=elapsed)


def display_servers(console: Console, result: ScanResult) -> None:
    """Display discovered servers in a Rich table."""
    if not result.servers:
        console.print("[yellow]No inference servers found on {result.subnet}[/yellow]")
        return

    table = Table(title=f"Discovered Servers ({result.subnet})")
    table.add_column("#", style="bold")
    table.add_column("Host")
    table.add_column("Port")
    table.add_column("Backend")
    table.add_column("Models")
    table.add_column("Status")

    for i, s in enumerate(result.servers, 1):
        models_str = ", ".join(s.models) if s.models else "[dim]none loaded[/dim]"
        status_str = "[green]healthy[/green]" if s.status == "healthy" else f"[red]{s.status}[/red]"
        table.add_row(str(i), s.host, str(s.port), s.backend, models_str, status_str)

    console.print(table)
    console.print(f"[dim]Scan completed in {result.scan_time_seconds:.1f}s[/dim]\n")


def _pick_server(console: Console, servers: list[DiscoveredServer], role: str) -> tuple[DiscoveredServer, str]:
    """Interactive prompt to pick a server and model for a role (target/draft)."""
    console.print(f"[bold]Select {role} server:[/bold]")
    for i, s in enumerate(servers, 1):
        console.print(f"  {i}) {s.host}:{s.port} ({s.backend})")

    while True:
        try:
            choice = int(input(f"  Enter number (1-{len(servers)}): "))
            if 1 <= choice <= len(servers):
                break
        except (ValueError, EOFError):
            pass
        console.print(f"  [red]Please enter a number between 1 and {len(servers)}[/red]")

    server = servers[choice - 1]

    # Pick model if multiple available
    model_name = server.models[0] if len(server.models) == 1 else ""
    if len(server.models) > 1:
        console.print(f"\n  [bold]Models on {server.host}:{server.port}:[/bold]")
        for i, m in enumerate(server.models, 1):
            console.print(f"    {i}) {m}")
        while True:
            try:
                mc = int(input(f"  Pick model (1-{len(server.models)}): "))
                if 1 <= mc <= len(server.models):
                    model_name = server.models[mc - 1]
                    break
            except (ValueError, EOFError):
                pass
    elif not model_name:
        model_name = input("  Model name (e.g. qwen3:32b): ").strip() or server.backend

    return server, model_name


def detect_backend(url: str) -> str:
    """Guess backend from URL port. 11434 = ollama, else llamacpp."""
    from urllib.parse import urlparse
    port = urlparse(url).port or 80
    return "ollama" if port == 11434 else "llamacpp"


def generate_cluster_yaml(
    draft_server: DiscoveredServer,
    draft_model: str,
    target_server: DiscoveredServer,
    target_model: str,
    port: int = 8088,
    max_draft_tokens: int = 32,
) -> str:
    """Generate a cluster.yaml config string from selected servers."""
    config = {
        "proxy": {
            "host": "0.0.0.0",
            "port": port,
            "max_draft_tokens": max_draft_tokens,
            "fallback_on_draft_failure": True,
            "draft": {
                "url": f"http://{draft_server.host}:{draft_server.port}",
                "model_name": draft_model,
                "backend": draft_server.backend,
            },
            "target": {
                "url": f"http://{target_server.host}:{target_server.port}",
                "model_name": target_model,
                "backend": target_server.backend,
            },
        },
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "gpus": [],
        },
        "models": {},
    }
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def generate_local_yaml(
    gpus: list,
    binary: str | None,
    model_path: str | None = None,
    port: int = 8080,
) -> str:
    """Generate a coordinator-only config YAML from detected GPUs.

    Parameters
    ----------
    gpus:
        List of DetectedGPU objects from gpu_detect.detect_gpus().
    binary:
        Path to llama-server binary, or None to use default.
    model_path:
        Optional path to a GGUF model file.
    port:
        Coordinator port (default 8080).
    """
    # Determine backend from the first GPU
    backend = gpus[0].backend if gpus else "cuda"

    coordinator_gpus = []
    for gpu in gpus:
        coordinator_gpus.append({
            "name": gpu.name,
            "vram_gb": max(1, gpu.vram_mb // 1024),
        })

    models = {}
    if model_path:
        models["default"] = {
            "path": model_path,
            "ctx_size": 8192,
            "predict": 4096,
            "flash_attn": True,
            "default": True,
        }

    config = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": port,
            "backend": backend,
            "gpus": coordinator_gpus,
        },
        "models": models,
    }

    if binary:
        config["binaries"] = {"coordinator": binary}

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def run_wizard(console: Console, result: ScanResult, output: Path, proxy_port: int = 8088) -> bool:
    """Run the interactive wizard. Returns True if config was written."""
    display_servers(console, result)

    if not result.servers:
        console.print("[red]No servers found. Start Ollama or llama-server on your LAN first.[/red]")
        console.print("  Ollama: OLLAMA_HOST=0.0.0.0 ollama serve")
        console.print("  llama-server: llama-server -m model.gguf --host 0.0.0.0 --port 8080")
        return False

    if len(result.servers) < 2:
        console.print("[yellow]Only 1 server found. Speculative decoding needs at least 2 (draft + target).[/yellow]")
        console.print("[yellow]Start a second inference server and re-run tightwad init.[/yellow]")
        return False

    # Pick target (big model)
    console.print()
    target_server, target_model = _pick_server(console, result.servers, "TARGET (big model)")

    # Pick draft (small fast model)
    console.print()
    draft_server, draft_model = _pick_server(console, result.servers, "DRAFT (small fast model)")

    # Generate and preview
    yaml_str = generate_cluster_yaml(
        draft_server, draft_model,
        target_server, target_model,
        port=proxy_port,
    )

    console.print("\n[bold]Generated config:[/bold]")
    console.print(f"[dim]{yaml_str}[/dim]")

    # Confirm write
    confirm = input(f"Write to {output}? [Y/n] ").strip().lower()
    if confirm and confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml_str)
    console.print(f"\n[green]Config written to {output}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"  tightwad doctor -c {output}")
    console.print(f"  tightwad proxy start -c {output}")
    return True
