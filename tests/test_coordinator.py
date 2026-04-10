"""Tests for coordinator command building."""

import json

import pytest
import yaml

from tightwad.config import load_config
from tightwad.coordinator import build_server_args, _read_pidfile, _write_pidfile, PIDFILE


@pytest.fixture
def config(tmp_path):
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "hip",
            "gpus": [
                {"name": "XTX 0", "vram_gb": 24},
                {"name": "XTX 1", "vram_gb": 24},
            ],
        },
        "workers": [
            {
                "host": "192.168.1.100",
                "gpus": [
                    {"name": "4070", "vram_gb": 16, "rpc_port": 50052},
                    {"name": "3060", "vram_gb": 12, "rpc_port": 50053},
                ],
            }
        ],
        "models": {
            "qwen3-32b": {
                "path": "/models/qwen3-32b.gguf",
                "ctx_size": 8192,
                "predict": 4096,
                "flash_attn": True,
                "default": True,
            }
        },
        "binaries": {"coordinator": "llama-server"},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    return load_config(p)


def test_build_args_basic(config):
    model = config.default_model()
    args = build_server_args(config, model)

    assert args[0] == "llama-server"
    assert "-m" in args
    assert args[args.index("-m") + 1] == "/models/qwen3-32b.gguf"
    assert "-ngl" in args
    assert "999" in args
    assert "--flash-attn" in args


def test_build_args_rpc(config):
    model = config.default_model()
    args = build_server_args(config, model)

    assert "--rpc" in args
    rpc_val = args[args.index("--rpc") + 1]
    assert "192.168.1.100:50052" in rpc_val
    assert "192.168.1.100:50053" in rpc_val


def test_build_args_flash_attn_value(config):
    """--flash-attn must be followed by 'on' (llama.cpp b8112+ format)."""
    model = config.default_model()
    args = build_server_args(config, model)
    idx = args.index("--flash-attn")
    assert args[idx + 1] == "on", (
        f"--flash-attn should be followed by 'on', got {args[idx + 1]!r}"
    )


def test_build_args_tensor_split(config):
    model = config.default_model()
    args = build_server_args(config, model)

    assert "--tensor-split" in args
    split_val = args[args.index("--tensor-split") + 1]
    parts = split_val.split(",")
    assert len(parts) == 4


# -- Pidfile JSON format --

def test_pidfile_json_roundtrip(tmp_path, monkeypatch):
    """JSON pidfile writes and reads back correctly."""
    pidfile = tmp_path / "coordinator.pid"
    monkeypatch.setattr("tightwad.coordinator.PIDFILE", pidfile)

    _write_pidfile(pid=12345, port=8080, model_name="test-model")
    data = _read_pidfile()

    assert data is not None
    assert data["pid"] == 12345
    assert data["port"] == 8080
    assert data["model"] == "test-model"
    assert "started" in data


def test_pidfile_legacy_int(tmp_path, monkeypatch):
    """Legacy plain-int pidfile is read correctly."""
    pidfile = tmp_path / "coordinator.pid"
    pidfile.write_text("54321")
    monkeypatch.setattr("tightwad.coordinator.PIDFILE", pidfile)

    data = _read_pidfile()
    assert data is not None
    assert data["pid"] == 54321
    assert "port" not in data  # legacy format has no port


def test_pidfile_missing(tmp_path, monkeypatch):
    """Missing pidfile returns None."""
    pidfile = tmp_path / "nonexistent.pid"
    monkeypatch.setattr("tightwad.coordinator.PIDFILE", pidfile)
    assert _read_pidfile() is None
