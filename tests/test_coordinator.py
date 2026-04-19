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


# -- MoE placement → --override-tensor flags --


def _stub_model_info(is_moe: bool, fused: bool = False):
    """Shape the minimal inspect_model() return used by the coordinator."""
    from pathlib import Path
    from tightwad.gguf_inspect import ModelInfo, MoEInfo, TensorInfo

    tensors = []
    if is_moe:
        n_expert, n_layers = 4, 2
        for layer in range(n_layers):
            if fused:
                for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                    tensors.append(TensorInfo(
                        name=f"blk.{layer}.{part}.weight",
                        shape=[16, 16, n_expert], dtype="F32",
                        n_bytes=n_expert * 16 * 16 * 4,
                    ))
            else:
                for expert in range(n_expert):
                    for part in ("ffn_gate", "ffn_up", "ffn_down"):
                        tensors.append(TensorInfo(
                            name=f"blk.{layer}.{part}.{expert}.weight",
                            shape=[16, 16], dtype="F32",
                            n_bytes=16 * 16 * 4,
                        ))
        moe = MoEInfo(
            n_expert=n_expert, n_expert_used=2,
            routing_overhead_bytes=int(0.5 * 1024 ** 3),
            expert_tensor_names=[t.name for t in tensors],
        )
    else:
        moe = None
    return ModelInfo(
        path=Path("/fake/m.gguf"), arch="llama", n_params=None,
        n_layers=2, quantization="F32", context_length=8192,
        total_size=sum(t.n_bytes for t in tensors), tensors=tensors, moe=moe,
    )


def test_build_args_ot_emitted_for_indexed_moe(config, monkeypatch):
    """moe_placement: balanced on an indexed-form MoE emits --override-tensor flags."""
    import tightwad.coordinator as coord

    model = config.default_model()
    model.moe_placement = "balanced"

    monkeypatch.setattr(coord, "_moe_override_tensor_flags",
                        lambda cfg, m: ["^blk\\.0\\.ffn_(gate|up|down)\\.(0|1)\\.weight$=CUDA0"])
    args = build_server_args(config, model)

    assert "--override-tensor" in args
    idx = args.index("--override-tensor")
    assert "CUDA0" in args[idx + 1]


def test_build_args_no_ot_when_placement_off(config, monkeypatch):
    import tightwad.coordinator as coord

    model = config.default_model()
    model.moe_placement = None
    monkeypatch.setattr(coord, "_moe_override_tensor_flags",
                        lambda cfg, m: ["SHOULD_NOT_BE_CALLED"])
    args = build_server_args(config, model)
    assert "--override-tensor" not in args


def test_build_args_no_ot_for_dense_model(config, monkeypatch):
    """moe_placement on a dense model silently no-ops (empty flag list)."""
    import tightwad.coordinator as coord

    model = config.default_model()
    model.moe_placement = "balanced"

    def fake_inspect(path):
        return _stub_model_info(is_moe=False)

    monkeypatch.setattr("tightwad.gguf_inspect.inspect_model", fake_inspect)
    args = build_server_args(config, model)
    assert "--override-tensor" not in args


def test_build_args_warns_on_fused_moe(config, monkeypatch, caplog):
    """Fused MoE falls back to layer-split but logs a warning."""
    import tightwad.coordinator as coord

    model = config.default_model()
    model.moe_placement = "balanced"

    def fake_inspect(path):
        return _stub_model_info(is_moe=True, fused=True)

    monkeypatch.setattr("tightwad.gguf_inspect.inspect_model", fake_inspect)

    import logging
    with caplog.at_level(logging.WARNING, logger="tightwad.coordinator"):
        args = build_server_args(config, model)
    assert "--override-tensor" not in args
    assert any("fused" in rec.message.lower() or "defuse" in rec.message.lower()
               for rec in caplog.records)


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
