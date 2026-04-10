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
