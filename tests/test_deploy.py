"""Tests for remote deployment module."""

from unittest.mock import patch, MagicMock

import pytest

from tightwad.deploy import deploy, _ssh_run, DeployResult, SSH_OPTS


def test_deploy_result_dataclass():
    r = DeployResult(host="10.0.0.1", success=True, message="ok", steps_completed=["ssh_connect"])
    assert r.host == "10.0.0.1"
    assert r.success is True
    assert r.steps_completed == ["ssh_connect"]


def test_deploy_ssh_fails():
    """Deploy fails cleanly when SSH connection fails."""
    with patch("tightwad.deploy._ssh_run", return_value=(1, "", "Connection refused")):
        result = deploy("10.0.0.1", ssh_user="test")
    assert result.success is False
    assert "SSH connection failed" in result.message
    assert result.steps_completed == []


def test_deploy_no_python():
    """Deploy fails when python3 is not found on remote."""
    def mock_ssh(host, user, cmd, timeout=120):
        if "echo ok" in cmd:
            return (0, "ok", "")
        if "python3 --version" in cmd:
            return (1, "", "python3: not found")
        return (0, "", "")

    with patch("tightwad.deploy._ssh_run", side_effect=mock_ssh):
        result = deploy("10.0.0.1", ssh_user="test")
    assert result.success is False
    assert "python3 not found" in result.message


def test_deploy_full_success():
    """Full deploy succeeds when all SSH commands return 0."""
    def mock_ssh(host, user, cmd, timeout=120):
        if "echo ok" in cmd:
            return (0, "ok", "")
        if "python3 --version" in cmd:
            return (0, "Python 3.11.0", "")
        if "pip install" in cmd:
            return (0, "", "")
        if "mkdir" in cmd:
            return (0, "", "")
        if "start" in cmd:
            return (0, "", "")
        return (0, "", "")

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    import httpx as httpx_mod
    with patch("tightwad.deploy._ssh_run", side_effect=mock_ssh), \
         patch("tightwad.deploy._scp", return_value=True), \
         patch.object(httpx_mod, "get", return_value=mock_resp):
        result = deploy("10.0.0.1", ssh_user="test", config_path="/tmp/config.yaml")

    assert result.success is True
    assert "ssh_connect" in result.steps_completed
    assert "pip_install" in result.steps_completed
    assert "config_copied" in result.steps_completed
    assert "started" in result.steps_completed


def test_ssh_opts_contain_batch_mode():
    """SSH options include BatchMode for non-interactive operation."""
    assert "BatchMode=yes" in " ".join(SSH_OPTS)
