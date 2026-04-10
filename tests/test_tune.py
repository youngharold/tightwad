"""Tests for system tuning diagnostics."""

from unittest.mock import patch

import pytest

from tightwad.tune import SystemInfo, Recommendation, diagnose, recommend


def test_system_info_dataclass():
    info = SystemInfo(
        platform="linux",
        total_ram_gb=16.0,
        available_ram_gb=12.3,
        swap_total_gb=0.0,
        swap_used_gb=0.0,
        vm_swappiness=60,
    )
    assert info.total_ram_gb == 16.0
    assert info.swap_on_nvme is None


def test_recommendation_dataclass():
    rec = Recommendation(
        severity="critical",
        message="No swap configured.",
        commands=["sudo swapon /swapfile"],
    )
    assert rec.severity == "critical"
    assert len(rec.commands) == 1


def test_diagnose_returns_system_info():
    """diagnose() returns a SystemInfo with populated fields."""
    info = diagnose()
    assert isinstance(info, SystemInfo)
    assert info.total_ram_gb > 0
    assert info.available_ram_gb >= 0
    assert info.platform in ("linux", "windows", "darwin")


def test_recommend_no_swap_critical():
    """Model > RAM with no swap should produce a critical recommendation."""
    info = SystemInfo(
        platform="linux",
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        swap_total_gb=0.0,
        swap_used_gb=0.0,
        vm_swappiness=60,
    )
    recs = recommend(info, model_size_gb=20.0)
    assert any(r.severity == "critical" for r in recs)
    critical = [r for r in recs if r.severity == "critical"][0]
    assert "swap" in critical.message.lower() or "No swap" in critical.message
    assert len(critical.commands) > 0


def test_recommend_no_swap_windows():
    """Windows with no pagefile and large model should produce critical."""
    info = SystemInfo(
        platform="windows",
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        swap_total_gb=0.0,
        swap_used_gb=0.0,
    )
    recs = recommend(info, model_size_gb=20.0)
    assert any(r.severity == "critical" for r in recs)


def test_recommend_sufficient():
    """Model that fits in RAM should produce info-only recommendations."""
    info = SystemInfo(
        platform="linux",
        total_ram_gb=64.0,
        available_ram_gb=50.0,
        swap_total_gb=32.0,
        swap_used_gb=0.0,
        vm_swappiness=60,
    )
    recs = recommend(info, model_size_gb=5.0)
    assert all(r.severity == "info" for r in recs)


def test_recommend_low_swappiness():
    """Low swappiness should produce a recommendation."""
    info = SystemInfo(
        platform="linux",
        total_ram_gb=64.0,
        available_ram_gb=50.0,
        swap_total_gb=32.0,
        swap_used_gb=0.0,
        vm_swappiness=10,
    )
    recs = recommend(info, model_size_gb=40.0)
    swappiness_recs = [r for r in recs if "swappiness" in r.message.lower()]
    assert len(swappiness_recs) > 0


def test_recommend_darwin_no_swap():
    """macOS should say swap is managed automatically."""
    info = SystemInfo(
        platform="darwin",
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        swap_total_gb=0.0,
        swap_used_gb=0.0,
    )
    recs = recommend(info, model_size_gb=20.0)
    assert any("managed automatically" in r.message for r in recs)


def test_recommend_no_model():
    """Without a model, should return general system info."""
    info = SystemInfo(
        platform="linux",
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        swap_total_gb=8.0,
        swap_used_gb=0.0,
        vm_swappiness=60,
    )
    recs = recommend(info)
    assert any("Specify --model" in r.message for r in recs)


def test_recommend_insufficient_swap():
    """Model > RAM with insufficient swap should warn."""
    info = SystemInfo(
        platform="linux",
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        swap_total_gb=4.0,
        swap_used_gb=0.0,
        vm_swappiness=60,
    )
    recs = recommend(info, model_size_gb=20.0)
    assert any(r.severity == "warn" for r in recs)
