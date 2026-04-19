"""Smoke tests for `tightwad moe` subcommands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner


gguf = pytest.importorskip("gguf")

from tightwad.cli import cli  # noqa: E402


def _cluster_yaml_path(tmp_path: Path) -> Path:
    cfg = {
        "coordinator": {
            "host": "0.0.0.0", "port": 8090, "backend": "cuda",
            "gpus": [{"name": "RTX 4070", "vram_gb": 16},
                      {"name": "RTX 3060", "vram_gb": 12}],
        },
        "workers": [
            {"host": "10.0.0.1", "gpus": [{"name": "RTX 2070", "vram_gb": 8, "rpc_port": 50052}]},
        ],
        "models": {"m": {"path": "/fake/m.gguf", "default": True}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _indexed_gguf(tmp_path: Path, n_expert: int = 4, n_layers: int = 2) -> Path:
    """Write a tiny indexed-MoE GGUF to ``tmp_path/model.gguf``."""
    path = tmp_path / "model.gguf"
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_expert_count(n_expert)
    w.add_expert_used_count(2)
    w.add_block_count(n_layers)
    rng = np.random.default_rng(seed=0)
    for layer in range(n_layers):
        for part in ("ffn_gate", "ffn_up", "ffn_down"):
            for e in range(n_expert):
                w.add_tensor(
                    f"blk.{layer}.{part}.{e}.weight",
                    rng.random((16, 16), dtype=np.float32),
                )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _fused_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "fused.gguf"
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_expert_count(4)
    w.add_expert_used_count(2)
    w.add_block_count(1)
    rng = np.random.default_rng(seed=0)
    for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        w.add_tensor(f"blk.0.{part}.weight",
                     rng.random((4, 8, 16), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


# ---------------------------------------------------------------------------
# moe plan
# ---------------------------------------------------------------------------


def test_plan_indexed_emits_ot(tmp_path, monkeypatch):
    cfg = _cluster_yaml_path(tmp_path)
    model = _indexed_gguf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["-c", str(cfg), "moe", "plan",
                                  str(model), "--emit-ot"])
    assert result.exit_code == 0, result.output
    assert "--override-tensor" in result.output
    assert "^blk" in result.output


def test_plan_json_mode(tmp_path):
    cfg = _cluster_yaml_path(tmp_path)
    model = _indexed_gguf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["-c", str(cfg), "moe", "plan",
                                  str(model), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "assignments" in payload
    assert "override_tensor_args" in payload
    assert len(payload["assignments"]) > 0


def test_plan_fused_warns(tmp_path):
    cfg = _cluster_yaml_path(tmp_path)
    model = _fused_gguf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["-c", str(cfg), "moe", "plan", str(model)])
    assert result.exit_code == 0
    assert "defuse" in result.output.lower()


def test_plan_profile_guided_requires_hot_profile(tmp_path):
    cfg = _cluster_yaml_path(tmp_path)
    model = _indexed_gguf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["-c", str(cfg), "moe", "plan",
                                  str(model), "--strategy", "profile-guided"])
    assert result.exit_code != 0
    assert "hot-profile" in result.output


# ---------------------------------------------------------------------------
# moe profile / summary
# ---------------------------------------------------------------------------


def test_profile_from_log(tmp_path):
    log = tmp_path / "stderr.log"
    log.write_text(
        "moe: layer=0 chosen=[0,1]\n"
        "moe: layer=0 chosen=[0,2]\n"
        "moe: layer=1 chosen=[5]\n"
    )
    out = tmp_path / "profile.json"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "moe", "profile",
        "--from-log", str(log),
        "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["total_tokens"] == 3


def test_summary_prints_top(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "total_tokens": 10, "source": "test",
        "hits": [{"layer": 0, "expert": 0, "count": 5},
                  {"layer": 0, "expert": 1, "count": 3}],
    }))
    runner = CliRunner()
    result = runner.invoke(cli, ["moe", "summary", str(profile), "--top", "2"])
    assert result.exit_code == 0, result.output
    assert "Top 2 hot experts" in result.output


# ---------------------------------------------------------------------------
# moe defuse
# ---------------------------------------------------------------------------


def test_defuse_writes_indexed(tmp_path):
    fused = _fused_gguf(tmp_path)
    out = tmp_path / "indexed.gguf"
    runner = CliRunner()
    result = runner.invoke(cli, ["moe", "defuse", str(fused), str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "n_expert=4" in result.output
