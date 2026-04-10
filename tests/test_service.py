"""Tests for service management module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tightwad.service import (
    SYSTEMD_TEMPLATE,
    LAUNCHD_TEMPLATE,
    _systemd_unit_path,
    _launchd_plist_path,
    install_service,
    uninstall_service,
    service_status,
)


def test_systemd_template_renders():
    content = SYSTEMD_TEMPLATE.substitute(
        exec_start="/usr/bin/tightwad -c /etc/tightwad.yaml start",
        config_path="/etc/tightwad.yaml",
    )
    assert "ExecStart=/usr/bin/tightwad" in content
    assert "TIGHTWAD_CONFIG=/etc/tightwad.yaml" in content
    assert "Restart=on-failure" in content
    assert "[Install]" in content


def test_launchd_template_renders():
    content = LAUNCHD_TEMPLATE.substitute(
        binary="/usr/local/bin/tightwad",
        config_path="/Users/test/.tightwad/config.yaml",
        log_dir="/Users/test/.tightwad/logs",
    )
    assert "com.tightwad.server" in content
    assert "/usr/local/bin/tightwad" in content
    assert "RunAtLoad" in content
    assert "KeepAlive" in content


def test_install_service_linux(tmp_path):
    """Install creates systemd unit file on Linux."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("coordinator:\n  host: 0.0.0.0\n")

    unit_dir = tmp_path / ".config" / "systemd" / "user"

    with patch("tightwad.service.platform.system", return_value="Linux"), \
         patch("tightwad.service.Path.home", return_value=tmp_path), \
         patch("tightwad.service._find_tightwad_binary", return_value="/usr/bin/tightwad"), \
         patch("tightwad.service.subprocess.run") as mock_run:
        plat, path = install_service(str(config_file), user=True)

    assert plat == "systemd"
    assert path.exists()
    content = path.read_text()
    assert "ExecStart=/usr/bin/tightwad" in content
    assert str(config_file.resolve()) in content


def test_install_service_darwin(tmp_path):
    """Install creates launchd plist on macOS."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("coordinator:\n  host: 0.0.0.0\n")

    with patch("tightwad.service.platform.system", return_value="Darwin"), \
         patch("tightwad.service.Path.home", return_value=tmp_path), \
         patch("tightwad.service._find_tightwad_binary", return_value="/usr/local/bin/tightwad"), \
         patch("tightwad.service.subprocess.run") as mock_run:
        plat, path = install_service(str(config_file), user=True)

    assert plat == "launchd"
    assert path.exists()
    content = path.read_text()
    assert "com.tightwad.server" in content
    assert "/usr/local/bin/tightwad" in content


def test_uninstall_not_installed():
    """Uninstall returns False when no service is installed."""
    with patch("tightwad.service.platform.system", return_value="Linux"), \
         patch("tightwad.service._systemd_unit_path") as mock_path:
        mock_path.return_value = MagicMock(exists=MagicMock(return_value=False))
        plat, was_installed = uninstall_service()
    assert plat == "systemd"
    assert was_installed is False


def test_service_status_not_installed():
    """Status shows not installed when unit file doesn't exist."""
    with patch("tightwad.service.platform.system", return_value="Linux"), \
         patch("tightwad.service._systemd_unit_path") as mock_path:
        mock_path.return_value = Path("/tmp/nonexistent/tightwad.service")
        st = service_status()
    assert st["installed"] is False
    assert st["running"] is False
    assert st["platform"] == "Linux"


def test_install_unsupported_platform(tmp_path):
    """Install raises on unsupported platforms."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("coordinator:\n  host: 0.0.0.0\n")

    with patch("tightwad.service.platform.system", return_value="FreeBSD"), \
         pytest.raises(RuntimeError, match="Unsupported"):
        install_service(str(config_file))
