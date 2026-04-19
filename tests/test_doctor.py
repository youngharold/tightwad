"""Tests for tightwad doctor diagnostic checks."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from tightwad.doctor import (
    CheckResult,
    DoctorReport,
    Section,
    Status,
    check_config,
    check_models,
    run_doctor,
    _is_cross_platform_path,
)


def test_report_passed_all_pass():
    report = DoctorReport(sections=[
        Section(title="Test", results=[
            CheckResult(name="a", status=Status.PASS),
            CheckResult(name="b", status=Status.WARN),
            CheckResult(name="c", status=Status.SKIP),
        ]),
    ])
    assert report.passed is True


def test_report_failed_on_fail():
    report = DoctorReport(sections=[
        Section(title="Test", results=[
            CheckResult(name="a", status=Status.PASS),
            CheckResult(name="b", status=Status.FAIL, detail="broken"),
        ]),
    ])
    assert report.passed is False


def test_report_to_dict():
    report = DoctorReport(sections=[
        Section(title="Config", results=[
            CheckResult(name="file", status=Status.PASS, detail="/some/path"),
        ]),
    ])
    d = report.to_dict()
    assert d["passed"] is True
    assert len(d["sections"]) == 1
    assert d["sections"][0]["title"] == "Config"
    assert d["sections"][0]["results"][0]["status"] == "pass"
    # Verify JSON-serializable
    json.dumps(d)


def test_check_config_missing_file(tmp_path):
    section, config = check_config(tmp_path / "nonexistent.yaml")
    assert config is None
    assert section.results[0].status == Status.FAIL
    assert "Not found" in section.results[0].detail


def test_check_config_valid(tmp_path):
    config_file = tmp_path / "cluster.yaml"
    config_file.write_text(textwrap.dedent("""\
        coordinator:
          host: 0.0.0.0
          port: 8080
          gpus:
            - name: RTX 4070
              vram_gb: 12
        workers: []
        models:
          test-model:
            path: /tmp/model.gguf
            default: true
        binaries:
          coordinator: llama-server
          rpc_server: rpc-server
    """))
    section, config = check_config(config_file)
    assert config is not None
    # Config file found + YAML parses + models defined
    assert section.results[0].status == Status.PASS  # file exists
    assert section.results[1].status == Status.PASS  # YAML parses
    assert section.results[2].status == Status.PASS  # models defined


def test_check_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("{{invalid yaml")
    section, config = check_config(config_file)
    assert config is None
    assert any(r.status == Status.FAIL and "YAML" in r.name for r in section.results)


def test_is_cross_platform_path():
    with patch("tightwad.doctor.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        assert _is_cross_platform_path("C:/Users/youruser/models/foo.gguf") is True
        assert _is_cross_platform_path("/usr/local/bin/llama-server") is False

        mock_platform.system.return_value = "Windows"
        assert _is_cross_platform_path("C:/Users/youruser/models/foo.gguf") is False


def test_check_models_cross_platform_skip(tmp_path):
    """Windows model paths on macOS should SKIP, not FAIL."""
    from tightwad.config import ClusterConfig, GPU, ModelConfig

    config = ClusterConfig(
        coordinator_host="0.0.0.0",
        coordinator_port=8080,
        coordinator_backend="cuda",
        coordinator_gpus=[GPU(name="Test", vram_gb=16)],
        workers=[],
        models={"test": ModelConfig(name="test", path="C:/Users/youruser/models/test.gguf")},
        coordinator_binary="llama-server",
        rpc_server_binary="rpc-server",
    )

    with patch("tightwad.doctor.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        section = check_models(config)

    model_result = section.results[0]
    assert model_result.status == Status.SKIP
    assert "remote path" in model_result.detail.lower() or "cross-platform" in model_result.detail.lower()


def test_run_doctor_missing_config(tmp_path):
    report = run_doctor(tmp_path / "nope.yaml")
    assert report.passed is False
    # Should short-circuit after config failure (only 1 section)
    assert len(report.sections) == 1


# ---------------------------------------------------------------------------
# MoE placement checks
# ---------------------------------------------------------------------------


def _moe_model_info(is_moe: bool, fused: bool):
    from pathlib import Path
    from tightwad.gguf_inspect import ModelInfo, MoEInfo, TensorInfo

    tensors = []
    if is_moe and fused:
        tensors.append(TensorInfo(name="blk.0.ffn_gate_exps.weight",
                                    shape=[16, 16, 4], dtype="F32",
                                    n_bytes=16 * 16 * 4 * 4))
    elif is_moe:
        for e in range(4):
            tensors.append(TensorInfo(name=f"blk.0.ffn_gate.{e}.weight",
                                        shape=[16, 16], dtype="F32",
                                        n_bytes=16 * 16 * 4))
    else:
        tensors.append(TensorInfo(name="blk.0.ffn_gate.weight",
                                    shape=[16, 16], dtype="F32", n_bytes=1024))
    moe = MoEInfo(n_expert=4, n_expert_used=2, routing_overhead_bytes=1024,
                   expert_tensor_names=[t.name for t in tensors]) if is_moe else None
    return ModelInfo(path=Path("/fake/m.gguf"), arch="llama", n_params=None,
                      n_layers=1, quantization="F32", context_length=8192,
                      total_size=sum(t.n_bytes for t in tensors),
                      tensors=tensors, moe=moe)


def _make_config_with_moe(tmp_path, placement, hot_profile=None, env=None):
    from tightwad.config import ClusterConfig, GPU, ModelConfig

    model_path = tmp_path / "m.gguf"
    model_path.write_text("fake")
    return ClusterConfig(
        coordinator_host="0.0.0.0", coordinator_port=8080,
        coordinator_backend="cuda",
        coordinator_gpus=[GPU(name="RTX 4070", vram_gb=16)],
        workers=[], models={
            "m": ModelConfig(
                name="m", path=str(model_path), default=True,
                moe_placement=placement, moe_hot_profile=hot_profile,
            ),
        },
        coordinator_binary="llama-server", rpc_server_binary="rpc-server",
        env=env or {},
    )


def test_moe_placement_warns_on_fused(tmp_path):
    config = _make_config_with_moe(tmp_path, "balanced")
    with patch("tightwad.gguf_inspect.inspect_model",
                 return_value=_moe_model_info(is_moe=True, fused=True)):
        section = check_models(config)

    warns = [r for r in section.results
             if r.name.startswith("MoE placement") and r.status == Status.WARN]
    assert any("defuse" in w.fix.lower() for w in warns)


def test_moe_placement_warns_on_dense_model(tmp_path):
    config = _make_config_with_moe(tmp_path, "balanced")
    with patch("tightwad.gguf_inspect.inspect_model",
                 return_value=_moe_model_info(is_moe=False, fused=False)):
        section = check_models(config)
    warns = [r for r in section.results
             if r.name.startswith("MoE placement") and r.status == Status.WARN]
    assert any("dense" in w.detail.lower() for w in warns)


def test_moe_placement_profile_guided_missing_hot_profile(tmp_path):
    config = _make_config_with_moe(tmp_path, "profile-guided")
    with patch("tightwad.gguf_inspect.inspect_model",
                 return_value=_moe_model_info(is_moe=True, fused=False)):
        section = check_models(config)
    warns = [r for r in section.results
             if r.name.startswith("MoE placement") and r.status == Status.WARN]
    assert any("moe_hot_profile" in w.detail for w in warns)


def test_moe_placement_profile_guided_missing_env_var(tmp_path):
    hot_profile = tmp_path / "profile.json"
    hot_profile.write_text('{"total_tokens":1,"source":"t","hits":[]}')
    config = _make_config_with_moe(tmp_path, "profile-guided",
                                     hot_profile=str(hot_profile))
    with patch("tightwad.gguf_inspect.inspect_model",
                 return_value=_moe_model_info(is_moe=True, fused=False)):
        section = check_models(config)
    warns = [r for r in section.results
             if r.name.startswith("MoE placement") and r.status == Status.WARN]
    assert any("LLAMA_LOG_MOE" in w.detail for w in warns)


def test_moe_placement_profile_guided_all_good(tmp_path):
    hot_profile = tmp_path / "profile.json"
    hot_profile.write_text('{"total_tokens":1,"source":"t","hits":[]}')
    config = _make_config_with_moe(
        tmp_path, "profile-guided",
        hot_profile=str(hot_profile),
        env={"LLAMA_LOG_MOE": "1"},
    )
    with patch("tightwad.gguf_inspect.inspect_model",
                 return_value=_moe_model_info(is_moe=True, fused=False)):
        section = check_models(config)
    placement_results = [r for r in section.results if r.name.startswith("MoE placement")]
    # Last result should be the PASS (after env check)
    assert any(r.status == Status.PASS for r in placement_results)
