"""Tests for cluster config loading."""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tightwad.config import ClusterConfig, ModelConfig, ProxyConfig, backend_presets, load_config, _resolve_config_path
from tightwad.coordinator import build_server_args


@pytest.fixture
def config_file(tmp_path):
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
            "test-model": {
                "path": "/models/test.gguf",
                "ctx_size": 4096,
                "default": True,
            }
        },
        "binaries": {
            "coordinator": "/usr/local/bin/llama-server",
            "rpc_server": "rpc-server.exe",
        },
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_load_config(config_file):
    config = load_config(config_file)
    assert config.coordinator_port == 8080
    assert config.coordinator_backend == "hip"
    assert len(config.coordinator_gpus) == 2
    assert len(config.workers) == 1
    assert len(config.workers[0].gpus) == 2


def test_total_vram(config_file):
    config = load_config(config_file)
    assert config.total_vram_gb == 76  # 24+24+16+12


def test_tensor_split(config_file):
    config = load_config(config_file)
    split = config.tensor_split()
    assert len(split) == 4
    assert sum(split) == pytest.approx(1.0, abs=0.05)
    # Order: RPC workers first (4070, 3060), then coordinator locals (XTX 0, XTX 1)
    assert split[2] == split[3]  # Two XTXs equal
    assert split[2] > split[0]  # XTX > 4070
    assert split[0] > split[1]  # 4070 > 3060


def test_rpc_addresses(config_file):
    config = load_config(config_file)
    addrs = config.rpc_addresses
    assert addrs == ["192.168.1.100:50052", "192.168.1.100:50053"]


def test_default_model(config_file):
    config = load_config(config_file)
    model = config.default_model()
    assert model is not None
    assert model.name == "test-model"
    assert model.default is True


def test_no_proxy_section(config_file):
    """Backward compat: configs without proxy section work fine."""
    config = load_config(config_file)
    assert config.proxy is None


def test_proxy_section_parsed(tmp_path):
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "hip",
            "gpus": [{"name": "XTX", "vram_gb": 24}],
        },
        "models": {
            "test": {"path": "/test.gguf", "default": True},
        },
        "proxy": {
            "host": "127.0.0.1",
            "port": 9999,
            "max_draft_tokens": 4,
            "fallback_on_draft_failure": False,
            "draft": {
                "url": "http://192.168.1.1:8081",
                "model_name": "small-model",
            },
            "target": {
                "url": "http://192.168.1.2:8080",
                "model_name": "big-model",
            },
        },
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)

    assert config.proxy is not None
    assert isinstance(config.proxy, ProxyConfig)
    assert config.proxy.host == "127.0.0.1"
    assert config.proxy.port == 9999
    assert config.proxy.max_draft_tokens == 4
    assert config.proxy.fallback_on_draft_failure is False
    assert config.proxy.draft.url == "http://192.168.1.1:8081"
    assert config.proxy.draft.model_name == "small-model"
    assert config.proxy.target.url == "http://192.168.1.2:8080"
    assert config.proxy.target.model_name == "big-model"


# -- Minimal ClusterConfig helper for build_server_args tests --

def _minimal_config(**overrides) -> ClusterConfig:
    defaults = dict(
        coordinator_host="0.0.0.0",
        coordinator_port=8080,
        coordinator_backend="cuda",
        coordinator_gpus=[],
        workers=[],
        models={},
        coordinator_binary="llama-server",
        rpc_server_binary="rpc-server",
    )
    defaults.update(overrides)
    return ClusterConfig(**defaults)


# -- flash_attn in build_server_args --

def test_flash_attn_true_with_value():
    """flash_attn=True emits --flash-attn on (b8112+ format)."""
    model = ModelConfig(name="m", path="/m.gguf", flash_attn=True)
    args = build_server_args(_minimal_config(), model)
    idx = args.index("--flash-attn")
    assert args[idx + 1] == "on"


def test_flash_attn_false_omitted():
    model = ModelConfig(name="m", path="/m.gguf", flash_attn=False)
    args = build_server_args(_minimal_config(), model)
    assert "--flash-attn" not in args


def test_flash_attn_legacy_string_coerced(tmp_path):
    """Legacy YAML with flash_attn: 'on' or 'auto' is coerced to True."""
    cfg = {
        "coordinator": {"host": "0.0.0.0", "port": 8080, "backend": "cuda", "gpus": []},
        "models": {"m": {"path": "/m.gguf", "default": True, "flash_attn": "on"}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)
    model = config.default_model()
    assert model.flash_attn is True


def test_flash_attn_legacy_off_string_coerced(tmp_path):
    """Legacy YAML with flash_attn: 'off' is coerced to False."""
    cfg = {
        "coordinator": {"host": "0.0.0.0", "port": 8080, "backend": "cuda", "gpus": []},
        "models": {"m": {"path": "/m.gguf", "default": True, "flash_attn": "off"}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)
    model = config.default_model()
    assert model.flash_attn is False


# -- MoE placement config --

def test_moe_placement_defaults_to_none(tmp_path):
    cfg = {
        "coordinator": {"host": "0.0.0.0", "port": 8080, "backend": "cuda", "gpus": []},
        "models": {"m": {"path": "/m.gguf", "default": True}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)
    model = config.default_model()
    assert model.moe_placement is None
    assert model.moe_hot_profile is None


def test_moe_placement_balanced_parsed(tmp_path):
    cfg = {
        "coordinator": {"host": "0.0.0.0", "port": 8080, "backend": "cuda", "gpus": []},
        "models": {
            "minimax": {
                "path": "/minimax.gguf", "default": True,
                "moe_placement": "balanced",
            }
        },
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    model = load_config(p).default_model()
    assert model.moe_placement == "balanced"


def test_moe_placement_profile_guided_parsed(tmp_path):
    cfg = {
        "coordinator": {"host": "0.0.0.0", "port": 8080, "backend": "cuda", "gpus": []},
        "models": {
            "minimax": {
                "path": "/minimax.gguf", "default": True,
                "moe_placement": "profile-guided",
                "moe_hot_profile": "~/.tightwad/moe-profile.json",
            }
        },
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    model = load_config(p).default_model()
    assert model.moe_placement == "profile-guided"
    assert model.moe_hot_profile == "~/.tightwad/moe-profile.json"


# -- extra_args passthrough --

def test_extra_args_appended():
    cfg = _minimal_config(extra_args=["--no-mmap", "--no-warmup"])
    model = ModelConfig(name="m", path="/m.gguf", flash_attn=False)
    args = build_server_args(cfg, model)
    assert args[-2:] == ["--no-mmap", "--no-warmup"]


def test_extra_args_parsed_from_yaml(tmp_path):
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "cuda",
            "gpus": [{"name": "GPU", "vram_gb": 24}],
            "extra_args": ["--no-mmap", "--no-warmup"],
        },
        "models": {"m": {"path": "/m.gguf", "default": True}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)
    assert config.extra_args == ["--no-mmap", "--no-warmup"]


# -- env passthrough --

def test_env_parsed_from_yaml(tmp_path):
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "cuda",
            "gpus": [{"name": "GPU", "vram_gb": 24}],
            "env": {"MY_VAR": "hello"},
        },
        "models": {"m": {"path": "/m.gguf", "default": True}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)
    assert config.env == {"MY_VAR": "hello"}


# -- backend_presets --

def test_backend_presets_hip_multi_gpu():
    presets = backend_presets("hip", 2)
    assert presets["env"]["HSA_ENABLE_SDMA"] == "0"
    assert presets["env"]["GPU_MAX_HW_QUEUES"] == "1"


def test_backend_presets_hip_single_gpu():
    presets = backend_presets("hip", 1)
    assert presets["env"] == {}


def test_backend_presets_cuda_no_env():
    presets = backend_presets("cuda", 4)
    assert presets["env"] == {}


def test_backend_presets_explicit_env_overrides(tmp_path):
    """User-specified env values override backend presets."""
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "hip",
            "gpus": [
                {"name": "XTX 0", "vram_gb": 24},
                {"name": "XTX 1", "vram_gb": 24},
            ],
            "env": {"HSA_ENABLE_SDMA": "1"},  # user override
        },
        "models": {"m": {"path": "/m.gguf", "default": True}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)
    # User's value wins over preset
    assert config.env["HSA_ENABLE_SDMA"] == "1"
    # Preset still fills in GPU_MAX_HW_QUEUES
    assert config.env["GPU_MAX_HW_QUEUES"] == "1"


def test_backend_presets_auto_injected(tmp_path):
    """hip + 2 GPUs auto-injects env without explicit env in YAML."""
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
        "models": {"m": {"path": "/m.gguf", "default": True}},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    config = load_config(p)
    assert config.env["HSA_ENABLE_SDMA"] == "0"
    assert config.env["GPU_MAX_HW_QUEUES"] == "1"


# -- Config auto-discovery --

def test_config_autodiscovery_cwd(tmp_path, monkeypatch):
    """Auto-discovers tightwad.yaml in current working directory."""
    cfg = {
        "coordinator": {"host": "0.0.0.0", "port": 8080, "backend": "cuda", "gpus": []},
        "models": {"m": {"path": "/m.gguf", "default": True}},
    }
    (tmp_path / "tightwad.yaml").write_text(yaml.dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TIGHTWAD_CONFIG", raising=False)

    config = load_config(None)
    assert config.coordinator_port == 8080


def test_config_autodiscovery_home(tmp_path, monkeypatch):
    """Auto-discovers ~/.tightwad/config.yaml."""
    cfg = {
        "coordinator": {"host": "0.0.0.0", "port": 9090, "backend": "cuda", "gpus": []},
        "models": {"m": {"path": "/m.gguf", "default": True}},
    }
    tightwad_dir = tmp_path / ".tightwad"
    tightwad_dir.mkdir()
    (tightwad_dir / "config.yaml").write_text(yaml.dump(cfg))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("TIGHTWAD_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_config(None)
    assert config.coordinator_port == 9090


def test_config_autodiscovery_raises_with_searched_paths(tmp_path, monkeypatch):
    """FileNotFoundError lists searched paths when no config found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    monkeypatch.delenv("TIGHTWAD_CONFIG", raising=False)
    monkeypatch.delenv("TIGHTWAD_DRAFT_URL", raising=False)
    monkeypatch.delenv("TIGHTWAD_TARGET_URL", raising=False)
    # Ensure DEFAULT_CONFIG doesn't exist by pointing home() somewhere empty
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Also patch DEFAULT_CONFIG to a non-existent path
    monkeypatch.setattr("tightwad.config.DEFAULT_CONFIG", tmp_path / "nonexistent" / "cluster.yaml")
    with pytest.raises(FileNotFoundError, match="tightwad init"):
        load_config(None)
