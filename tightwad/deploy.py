"""Remote deployment: install and start tightwad on a remote host via SSH."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("tightwad.deploy")

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new",
]


@dataclass
class DeployResult:
    """Result of a deployment attempt."""
    host: str
    success: bool
    message: str
    steps_completed: list[str]


def _ssh_run(host: str, user: str, command: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run a command on a remote host via SSH.

    Returns (returncode, stdout, stderr).
    """
    target = f"{user}@{host}" if user else host
    cmd = ["ssh", *SSH_OPTS, target, command]
    logger.debug("SSH: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _scp(host: str, user: str, local_path: str, remote_path: str) -> bool:
    """Copy a file to a remote host via SCP."""
    target = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"
    cmd = ["scp", *SSH_OPTS, local_path, target]
    logger.debug("SCP: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def deploy(
    host: str,
    ssh_user: str = "",
    config_path: str | None = None,
    install_method: str = "pip",
) -> DeployResult:
    """Deploy tightwad to a remote host.

    Steps:
    1. Check SSH connectivity
    2. Check Python3 availability
    3. Install tightwad via pip
    4. Copy config file (if provided)
    5. Start tightwad
    6. Verify health from local machine

    Parameters
    ----------
    host:
        Remote hostname or IP.
    ssh_user:
        SSH username (default: current user).
    config_path:
        Local path to tightwad config YAML. Copied to remote ~/.tightwad/config.yaml.
    install_method:
        "pip" (default) — pip install tightwad.
    """
    steps: list[str] = []

    # Step 1: Check SSH connectivity
    try:
        rc, out, err = _ssh_run(host, ssh_user, "echo ok", timeout=15)
        if rc != 0:
            return DeployResult(host=host, success=False,
                                message=f"SSH connection failed: {err}", steps_completed=steps)
        steps.append("ssh_connect")
    except subprocess.TimeoutExpired:
        return DeployResult(host=host, success=False,
                            message="SSH connection timed out", steps_completed=steps)

    # Step 2: Check Python3
    rc, out, err = _ssh_run(host, ssh_user, "python3 --version")
    if rc != 0:
        return DeployResult(host=host, success=False,
                            message="python3 not found on remote host", steps_completed=steps)
    steps.append(f"python3: {out}")

    # Step 3: Install tightwad
    rc, out, err = _ssh_run(host, ssh_user,
                            "python3 -m pip install --user --upgrade tightwad",
                            timeout=300)
    if rc != 0:
        return DeployResult(host=host, success=False,
                            message=f"pip install failed: {err}", steps_completed=steps)
    steps.append("pip_install")

    # Step 4: Copy config
    if config_path:
        remote_config = "~/.tightwad/config.yaml"
        _ssh_run(host, ssh_user, "mkdir -p ~/.tightwad")
        if not _scp(host, ssh_user, config_path, remote_config):
            return DeployResult(host=host, success=False,
                                message="Failed to copy config file", steps_completed=steps)
        steps.append("config_copied")
    else:
        remote_config = None

    # Step 5: Start tightwad
    start_cmd = "python3 -m tightwad"
    if remote_config:
        start_cmd += f" -c {remote_config}"
    start_cmd += " start"

    rc, out, err = _ssh_run(host, ssh_user, start_cmd, timeout=60)
    if rc != 0:
        return DeployResult(host=host, success=False,
                            message=f"tightwad start failed: {err}", steps_completed=steps)
    steps.append("started")

    # Step 6: Verify health from local
    import httpx
    try:
        resp = httpx.get(f"http://{host}:8080/health", timeout=10.0)
        if resp.status_code == 200:
            steps.append("health_ok")
        else:
            steps.append(f"health_check: HTTP {resp.status_code}")
    except Exception as e:
        steps.append(f"health_check: {e}")

    return DeployResult(
        host=host,
        success=True,
        message="Deployment complete",
        steps_completed=steps,
    )
