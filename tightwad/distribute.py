"""Distribute GGUF models to worker machines via rsync/scp or swarm P2P."""

from __future__ import annotations

import asyncio
import logging
import shutil
import socket
import threading
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from .config import ClusterConfig, Worker

logger = logging.getLogger("tightwad.distribute")

# Models under this size use rsync (swarm overhead not worth it)
SWARM_SIZE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1 GB


@dataclass
class TransferTarget:
    host: str
    ssh_user: str | None
    remote_path: str
    worker_name: str  # for display


@dataclass
class TransferResult:
    target: TransferTarget
    success: bool
    message: str


def resolve_targets(
    config: ClusterConfig,
    model_name: str,
    specific_target: str | None = None,
) -> tuple[Path, list[TransferTarget]]:
    """Resolve model path and transfer targets from config.

    Args:
        config: Cluster config with workers and models.
        model_name: Model name key from config.
        specific_target: Optional 'host:/path' override.

    Returns:
        Tuple of (local model path, list of transfer targets).
    """
    model_cfg = config.models.get(model_name)
    if not model_cfg:
        available = ", ".join(config.models.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    local_path = Path(model_cfg.path)

    if specific_target:
        host, _, remote_path = specific_target.partition(":")
        if not remote_path:
            raise ValueError(f"Target must be 'host:/path', got '{specific_target}'")
        return local_path, [TransferTarget(
            host=host,
            ssh_user=None,
            remote_path=remote_path,
            worker_name=host,
        )]

    targets: list[TransferTarget] = []
    for w in config.workers:
        if not w.model_dir:
            continue
        targets.append(TransferTarget(
            host=w.host,
            ssh_user=w.ssh_user,
            remote_path=str(Path(w.model_dir) / local_path.name),
            worker_name=f"{w.host} ({w.gpus[0].name})" if w.gpus else w.host,
        ))

    return local_path, targets


def build_transfer_cmd(local_path: Path, target: TransferTarget) -> list[str]:
    """Build rsync or scp command for a transfer."""
    dest_user = f"{target.ssh_user}@" if target.ssh_user else ""
    dest = f"{dest_user}{target.host}:{target.remote_path}"

    if shutil.which("rsync"):
        return [
            "rsync", "-avz", "--progress", "--partial",
            str(local_path), dest,
        ]
    else:
        return ["scp", "-C", str(local_path), dest]


async def transfer_one(
    local_path: Path,
    target: TransferTarget,
    progress: Progress,
    task_id: TaskID,
) -> TransferResult:
    """Transfer model to one target using rsync/scp."""
    cmd = build_transfer_cmd(local_path, target)

    progress.update(task_id, description=f"[cyan]{target.worker_name}[/cyan]: transferring...")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode == 0:
        progress.update(task_id, description=f"[green]{target.worker_name}[/green]: done", completed=100)
        return TransferResult(target=target, success=True, message="OK")
    else:
        err = stderr.decode().strip() or stdout.decode().strip()
        progress.update(task_id, description=f"[red]{target.worker_name}[/red]: failed")
        return TransferResult(target=target, success=False, message=err)


async def distribute_async(
    local_path: Path,
    targets: list[TransferTarget],
    console: Console,
) -> list[TransferResult]:
    """Transfer model to all targets in parallel with progress display."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        tasks = []
        for target in targets:
            task_id = progress.add_task(f"[dim]{target.worker_name}[/dim]: queued", total=100)
            tasks.append(transfer_one(local_path, target, progress, task_id))

        results = await asyncio.gather(*tasks)

    return list(results)


def distribute(
    local_path: Path,
    targets: list[TransferTarget],
    console: Console,
) -> list[TransferResult]:
    """Synchronous wrapper for distribute_async."""
    return asyncio.run(distribute_async(local_path, targets, console))


def format_dry_run(
    local_path: Path,
    targets: list[TransferTarget],
    method: str = "rsync",
    token: str | None = None,
) -> str:
    """Show what would be transferred without executing."""
    lines = [f"Model:  {local_path}"]
    if local_path.exists():
        size_gb = local_path.stat().st_size / (1024**3)
        lines.append(f"Size:   {size_gb:.2f} GB")
    else:
        lines.append(f"Size:   (file not found locally)")
    lines.append(f"Method: {method}")

    if method == "swarm":
        seeder_host = _get_local_ip()
        lines.append(f"\nSeeder: {seeder_host}:9080")
        lines.append(f"Targets ({len(targets)}):")
        for t in targets:
            cmd = _build_swarm_pull_ssh_cmd(t, seeder_host, 9080, token)
            lines.append(f"  {t.worker_name}:")
            lines.append(f"    {' '.join(cmd)}")
    else:
        lines.append(f"\nTransfers ({len(targets)}):")
        for t in targets:
            cmd = build_transfer_cmd(local_path, t)
            lines.append(f"  {t.worker_name}:")
            lines.append(f"    {' '.join(cmd)}")
    return "\n".join(lines)


def auto_select_method(local_path: Path) -> str:
    """Auto-select distribution method based on file size."""
    if local_path.exists() and local_path.stat().st_size >= SWARM_SIZE_THRESHOLD:
        return "swarm"
    return "rsync"


def _get_local_ip() -> str:
    """Get local LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _build_swarm_pull_ssh_cmd(
    target: TransferTarget,
    seeder_host: str,
    seeder_port: int,
    token: str | None = None,
) -> list[str]:
    """Build SSH command that runs tightwad swarm pull on a remote worker."""
    pull_cmd = (
        f"tightwad swarm pull {target.remote_path}"
        f" --manifest http://{seeder_host}:{seeder_port}/manifest"
        f" --peer http://{seeder_host}:{seeder_port}"
    )
    if token:
        pull_cmd += f" --token {token}"

    ssh_parts = ["ssh", "-o", "BatchMode=yes"]
    if target.ssh_user:
        ssh_parts.extend([f"{target.ssh_user}@{target.host}"])
    else:
        ssh_parts.append(target.host)
    ssh_parts.append(pull_cmd)
    return ssh_parts


async def _swarm_pull_worker(
    target: TransferTarget,
    seeder_host: str,
    seeder_port: int,
    progress: Progress,
    task_id: TaskID,
    token: str | None = None,
) -> TransferResult:
    """SSH into a worker and run tightwad swarm pull."""
    cmd = _build_swarm_pull_ssh_cmd(target, seeder_host, seeder_port, token)

    progress.update(task_id, description=f"[cyan]{target.worker_name}[/cyan]: pulling via swarm...")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode == 0:
        progress.update(task_id, description=f"[green]{target.worker_name}[/green]: done", completed=100)
        return TransferResult(target=target, success=True, message="OK")
    else:
        err = stderr.decode().strip() or stdout.decode().strip()
        progress.update(task_id, description=f"[red]{target.worker_name}[/red]: failed")
        return TransferResult(target=target, success=False, message=err)


async def distribute_swarm_async(
    local_path: Path,
    targets: list[TransferTarget],
    console: Console,
    seeder_port: int = 9080,
    token: str | None = None,
) -> list[TransferResult]:
    """Distribute a model via swarm: create manifest, seed, pull from all workers."""
    from .manifest import SwarmManifest, PieceBitfield, create_manifest
    from .swarm_transfer import create_seeder_app, write_seeder_pidfile, remove_seeder_pidfile

    # Step 1: Create manifest if needed
    manifest = SwarmManifest.find(local_path)
    if manifest is None:
        console.print("[dim]Creating swarm manifest...[/dim]")
        manifest = create_manifest(local_path)
        manifest.save()
        console.print(f"[dim]Manifest created: {manifest.num_pieces} pieces[/dim]")

    bitfield = PieceBitfield.load_or_create(manifest)
    # Source has all pieces
    if not bitfield.have_all():
        for i in range(manifest.num_pieces):
            bitfield.mark_have(i)
        bitfield.save()

    # Step 2: Start seeder in background thread
    seeder_host = _get_local_ip()
    console.print(
        f"[dim]Starting seeder on {seeder_host}:{seeder_port}...[/dim]"
    )

    import uvicorn

    app = create_seeder_app(local_path, manifest, bitfield, token=token)
    uvi_config = uvicorn.Config(app, host="0.0.0.0", port=seeder_port, log_level="warning")
    server = uvicorn.Server(uvi_config)

    seeder_thread = threading.Thread(target=server.run, daemon=True)
    seeder_thread.start()
    write_seeder_pidfile(manifest.model)

    # Give uvicorn a moment to start
    await asyncio.sleep(0.5)

    try:
        # Step 3: SSH pull on each worker in parallel
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            tasks = []
            for target in targets:
                tid = progress.add_task(f"[dim]{target.worker_name}[/dim]: queued", total=100)
                tasks.append(_swarm_pull_worker(
                    target, seeder_host, seeder_port, progress, tid, token=token,
                ))
            results = await asyncio.gather(*tasks)
    finally:
        # Step 4: Stop seeder
        server.should_exit = True
        seeder_thread.join(timeout=5)
        remove_seeder_pidfile(manifest.model)

    return list(results)


def distribute_swarm(
    local_path: Path,
    targets: list[TransferTarget],
    console: Console,
    seeder_port: int = 9080,
    token: str | None = None,
) -> list[TransferResult]:
    """Synchronous wrapper for distribute_swarm_async."""
    return asyncio.run(distribute_swarm_async(
        local_path, targets, console, seeder_port=seeder_port, token=token,
    ))
