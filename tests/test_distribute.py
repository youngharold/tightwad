"""Tests for model distribution to workers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tightwad.distribute import (
    TransferTarget,
    build_transfer_cmd,
    resolve_targets,
    format_dry_run,
    auto_select_method,
    _build_swarm_pull_ssh_cmd,
    SWARM_SIZE_THRESHOLD,
)
from tightwad.config import (
    GPU,
    Worker,
    ClusterConfig,
    ModelConfig,
)


@pytest.fixture
def cluster_config():
    return ClusterConfig(
        coordinator_host="0.0.0.0",
        coordinator_port=8080,
        coordinator_backend="cuda",
        coordinator_gpus=[GPU(name="P400", vram_gb=0)],
        workers=[
            Worker(
                host="192.168.1.100",
                gpus=[GPU(name="4070", vram_gb=16, rpc_port=50052)],
                ssh_user="youruser",
                model_dir="/models",
            ),
            Worker(
                host="192.168.1.200",
                gpus=[GPU(name="2070", vram_gb=8, rpc_port=50052)],
                ssh_user=None,
                model_dir="/data/models",
            ),
            Worker(
                host="192.168.1.300",
                gpus=[GPU(name="M2", vram_gb=16, rpc_port=50052)],
                # No model_dir — should be skipped
            ),
        ],
        models={
            "llama-3.3-70b": ModelConfig(
                name="llama-3.3-70b",
                path="/mnt/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
                default=True,
            ),
        },
        coordinator_binary="llama-server",
        rpc_server_binary="rpc-server",
    )


class TestResolveTargets:
    def test_resolve_from_config(self, cluster_config):
        local_path, targets = resolve_targets(cluster_config, "llama-3.3-70b")
        assert local_path == Path("/mnt/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf")
        # Only 2 workers have model_dir set
        assert len(targets) == 2
        assert targets[0].host == "192.168.1.100"
        assert targets[0].ssh_user == "youruser"
        assert targets[0].remote_path == "/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
        assert targets[1].host == "192.168.1.200"
        assert targets[1].ssh_user is None

    def test_resolve_specific_target(self, cluster_config):
        local_path, targets = resolve_targets(
            cluster_config, "llama-3.3-70b",
            specific_target="10.0.0.1:/data/model.gguf",
        )
        assert len(targets) == 1
        assert targets[0].host == "10.0.0.1"
        assert targets[0].remote_path == "/data/model.gguf"

    def test_unknown_model_raises(self, cluster_config):
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_targets(cluster_config, "nonexistent")

    def test_bad_specific_target_raises(self, cluster_config):
        with pytest.raises(ValueError, match="host:/path"):
            resolve_targets(cluster_config, "llama-3.3-70b", specific_target="nopath")


class TestBuildTransferCmd:
    def test_rsync_preferred(self):
        target = TransferTarget(
            host="192.168.1.100",
            ssh_user="youruser",
            remote_path="/models/model.gguf",
            worker_name="desktop",
        )
        with patch("shutil.which", return_value="/usr/bin/rsync"):
            cmd = build_transfer_cmd(Path("/local/model.gguf"), target)
        assert cmd[0] == "rsync"
        assert "youruser@192.168.1.100:/models/model.gguf" in cmd
        assert "--partial" in cmd

    def test_scp_fallback(self):
        target = TransferTarget(
            host="192.168.1.100",
            ssh_user=None,
            remote_path="/models/model.gguf",
            worker_name="desktop",
        )
        with patch("shutil.which", return_value=None):
            cmd = build_transfer_cmd(Path("/local/model.gguf"), target)
        assert cmd[0] == "scp"
        assert "192.168.1.100:/models/model.gguf" in cmd

    def test_no_user_prefix_when_none(self):
        target = TransferTarget(
            host="10.0.0.1",
            ssh_user=None,
            remote_path="/models/m.gguf",
            worker_name="test",
        )
        with patch("shutil.which", return_value="/usr/bin/rsync"):
            cmd = build_transfer_cmd(Path("/local/m.gguf"), target)
        dest = cmd[-1]
        assert dest == "10.0.0.1:/models/m.gguf"
        assert "@" not in dest


class TestFormatDryRun:
    def test_dry_run_output(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        targets = [
            TransferTarget(
                host="192.168.1.100",
                ssh_user="youruser",
                remote_path="/models/model.gguf",
                worker_name="desktop (4070)",
            ),
        ]
        output = format_dry_run(model_file, targets)
        assert "model.gguf" in output
        assert "desktop (4070)" in output
        assert "192.168.1.100" in output

    def test_dry_run_missing_file(self):
        targets = [
            TransferTarget(
                host="10.0.0.1",
                ssh_user=None,
                remote_path="/m.gguf",
                worker_name="test",
            ),
        ]
        output = format_dry_run(Path("/nonexistent/model.gguf"), targets)
        assert "not found" in output

    def test_dry_run_swarm_method(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        targets = [
            TransferTarget(
                host="192.168.1.100",
                ssh_user="youruser",
                remote_path="/models/model.gguf",
                worker_name="desktop (4070)",
            ),
        ]
        output = format_dry_run(model_file, targets, method="swarm")
        assert "swarm" in output.lower()
        assert "Seeder:" in output
        assert "tightwad swarm pull" in output
        assert "desktop (4070)" in output

    def test_dry_run_swarm_with_token(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        targets = [
            TransferTarget(
                host="192.168.1.100",
                ssh_user="youruser",
                remote_path="/models/model.gguf",
                worker_name="desktop",
            ),
        ]
        output = format_dry_run(model_file, targets, method="swarm", token="secret123")
        assert "--token secret123" in output


class TestAutoSelectMethod:
    def test_small_file_uses_rsync(self, tmp_path):
        small = tmp_path / "small.gguf"
        small.write_bytes(b"\x00" * 1024)
        assert auto_select_method(small) == "rsync"

    def test_large_file_uses_swarm(self, tmp_path):
        large = tmp_path / "large.gguf"
        # Create a file just at the threshold
        large.write_bytes(b"\x00" * SWARM_SIZE_THRESHOLD)
        assert auto_select_method(large) == "swarm"

    def test_nonexistent_file_uses_rsync(self):
        assert auto_select_method(Path("/does/not/exist.gguf")) == "rsync"


class TestSwarmPullSshCmd:
    def test_basic_cmd(self):
        target = TransferTarget(
            host="192.168.1.100",
            ssh_user="youruser",
            remote_path="/models/model.gguf",
            worker_name="desktop",
        )
        cmd = _build_swarm_pull_ssh_cmd(target, "192.168.1.1", 9080)
        assert cmd[0] == "ssh"
        assert "BatchMode=yes" in cmd
        assert "youruser@192.168.1.100" in cmd
        pull_part = cmd[-1]
        assert "tightwad swarm pull /models/model.gguf" in pull_part
        assert "--manifest http://192.168.1.1:9080/manifest" in pull_part
        assert "--peer http://192.168.1.1:9080" in pull_part

    def test_no_ssh_user(self):
        target = TransferTarget(
            host="10.0.0.5",
            ssh_user=None,
            remote_path="/m.gguf",
            worker_name="test",
        )
        cmd = _build_swarm_pull_ssh_cmd(target, "10.0.0.1", 9080)
        assert "10.0.0.5" in cmd
        assert not any("@" in part for part in cmd if part != cmd[-1])

    def test_with_token(self):
        target = TransferTarget(
            host="192.168.1.100",
            ssh_user=None,
            remote_path="/models/model.gguf",
            worker_name="desktop",
        )
        cmd = _build_swarm_pull_ssh_cmd(target, "192.168.1.1", 9080, token="mysecret")
        assert "--token mysecret" in cmd[-1]
