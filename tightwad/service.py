"""Service management: install/uninstall tightwad as a system service."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path
from string import Template

logger = logging.getLogger("tightwad.service")

SERVICE_NAME = "tightwad"

SYSTEMD_TEMPLATE = Template("""\
[Unit]
Description=Tightwad inference cluster manager
After=network.target

[Service]
Type=simple
ExecStart=$exec_start
Environment=TIGHTWAD_CONFIG=$config_path
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
""")

LAUNCHD_TEMPLATE = Template("""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tightwad.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>$binary</string>
        <string>-c</string>
        <string>$config_path</string>
        <string>start</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>TIGHTWAD_CONFIG</key>
        <string>$config_path</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$log_dir/tightwad-service.log</string>
    <key>StandardErrorPath</key>
    <string>$log_dir/tightwad-service.log</string>
</dict>
</plist>
""")


def _find_tightwad_binary() -> str:
    """Find the tightwad CLI binary path."""
    import shutil
    found = shutil.which("tightwad")
    if found:
        return found
    # Fallback: try the current Python's scripts dir
    import sys
    scripts_dir = Path(sys.executable).parent
    candidate = scripts_dir / "tightwad"
    if candidate.exists():
        return str(candidate)
    return "tightwad"


def _systemd_unit_path(user: bool = True) -> Path:
    """Return the systemd unit file path."""
    if user:
        return Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
    return Path(f"/etc/systemd/system/{SERVICE_NAME}.service")


def _launchd_plist_path() -> Path:
    """Return the launchd plist path."""
    return Path.home() / "Library" / "LaunchAgents" / "com.tightwad.server.plist"


def install_service(config_path: str, user: bool = True) -> tuple[str, Path]:
    """Install tightwad as a system service.

    Parameters
    ----------
    config_path:
        Absolute path to the tightwad config YAML.
    user:
        If True, install as a user service (systemd --user or LaunchAgent).

    Returns
    -------
    (platform_name, service_file_path)
    """
    config_path = str(Path(config_path).resolve())
    binary = _find_tightwad_binary()
    system = platform.system()

    if system == "Linux":
        unit_path = _systemd_unit_path(user)
        unit_path.parent.mkdir(parents=True, exist_ok=True)

        content = SYSTEMD_TEMPLATE.substitute(
            exec_start=f"{binary} -c {config_path} start",
            config_path=config_path,
        )
        unit_path.write_text(content)

        # Enable the service
        scope = ["--user"] if user else []
        subprocess.run(
            ["systemctl", *scope, "daemon-reload"],
            check=False,
        )
        subprocess.run(
            ["systemctl", *scope, "enable", SERVICE_NAME],
            check=False,
        )
        return "systemd", unit_path

    elif system == "Darwin":
        plist_path = _launchd_plist_path()
        plist_path.parent.mkdir(parents=True, exist_ok=True)

        log_dir = Path.home() / ".tightwad" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        content = LAUNCHD_TEMPLATE.substitute(
            binary=binary,
            config_path=config_path,
            log_dir=str(log_dir),
        )
        plist_path.write_text(content)

        subprocess.run(
            ["launchctl", "load", str(plist_path)],
            check=False,
        )
        return "launchd", plist_path

    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def uninstall_service() -> tuple[str, bool]:
    """Stop and remove the tightwad service.

    Returns
    -------
    (platform_name, was_installed)
    """
    system = platform.system()

    if system == "Linux":
        unit_path = _systemd_unit_path(user=True)
        if not unit_path.exists():
            return "systemd", False

        subprocess.run(["systemctl", "--user", "stop", SERVICE_NAME], check=False)
        subprocess.run(["systemctl", "--user", "disable", SERVICE_NAME], check=False)
        unit_path.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        return "systemd", True

    elif system == "Darwin":
        plist_path = _launchd_plist_path()
        if not plist_path.exists():
            return "launchd", False

        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
        plist_path.unlink(missing_ok=True)
        return "launchd", True

    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def service_status() -> dict:
    """Check if tightwad service is installed and running.

    Returns
    -------
    dict with keys: installed, running, platform, path
    """
    system = platform.system()
    result = {"installed": False, "running": False, "platform": system, "path": None}

    if system == "Linux":
        unit_path = _systemd_unit_path(user=True)
        result["path"] = str(unit_path)
        if unit_path.exists():
            result["installed"] = True
            try:
                proc = subprocess.run(
                    ["systemctl", "--user", "is-active", SERVICE_NAME],
                    capture_output=True, text=True,
                )
                result["running"] = proc.stdout.strip() == "active"
            except (FileNotFoundError, OSError):
                pass

    elif system == "Darwin":
        plist_path = _launchd_plist_path()
        result["path"] = str(plist_path)
        if plist_path.exists():
            result["installed"] = True
            try:
                proc = subprocess.run(
                    ["launchctl", "list", "com.tightwad.server"],
                    capture_output=True, text=True,
                )
                result["running"] = proc.returncode == 0
            except (FileNotFoundError, OSError):
                pass

    return result
