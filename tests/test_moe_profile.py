"""Tests for hot-expert profile capture from llama.cpp stderr."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from tightwad.moe_profile import (
    HotExpertProfile,
    parse_log_file,
    parse_stderr_stream,
)


# ---------------------------------------------------------------------------
# Patched format
# ---------------------------------------------------------------------------


def test_parse_patched_stream():
    lines = [
        "moe: layer=0 chosen=[0,1,2,3]",
        "moe: layer=0 chosen=[0,4,5,6]",
        "moe: layer=1 chosen=[7,8,9,10]",
    ]
    profile = parse_stderr_stream(io.StringIO("\n".join(lines)))

    assert profile.total_tokens == 3
    assert profile.hits[(0, 0)] == 2
    assert profile.hits[(0, 1)] == 1
    assert profile.hits[(1, 7)] == 1


def test_handles_whitespace_in_chosen():
    line = "moe: layer=5 chosen=[ 1 , 2, 3 ]"
    profile = parse_stderr_stream(io.StringIO(line))
    assert profile.hits[(5, 1)] == 1
    assert profile.hits[(5, 2)] == 1


def test_skips_empty_chosen():
    line = "moe: layer=0 chosen=[]"
    profile = parse_stderr_stream(io.StringIO(line))
    assert profile.total_tokens == 0
    assert not profile.hits


# ---------------------------------------------------------------------------
# Unpatched fallback
# ---------------------------------------------------------------------------


def test_fallback_populates_total_tokens_without_hits():
    lines = [
        "some unrelated log line",
        "slot finished n_expert_used=4",
        "slot finished n_expert_used=4",
        "moe: layer=3 chosen=[7]",
    ]
    profile = parse_stderr_stream(io.StringIO("\n".join(lines)))
    assert profile.total_tokens == 3
    assert profile.hits == {(3, 7): 1}


# ---------------------------------------------------------------------------
# Frequency / top_n / skew
# ---------------------------------------------------------------------------


def test_frequency_normalizes_counts():
    p = HotExpertProfile()
    p.record(0, 1, 10)
    p.record(0, 2, 5)
    p.record(1, 1, 5)
    freq = p.frequency()
    assert abs(freq[(0, 1)] - 0.5) < 1e-6
    assert abs(freq[(0, 2)] - 0.25) < 1e-6


def test_top_n_orders_by_count():
    p = HotExpertProfile()
    p.record(0, 1, 100)
    p.record(0, 2, 5)
    p.record(1, 1, 50)
    top = p.top_n(2)
    assert [(h.layer, h.expert) for h in top] == [(0, 1), (1, 1)]


def test_per_layer_skew_detects_concentration():
    p = HotExpertProfile()
    # Layer 0: one expert dominates
    p.record(0, 0, 100)
    for i in range(1, 10):
        p.record(0, i, 1)
    # Layer 1: even distribution
    for i in range(10):
        p.record(1, i, 10)
    skew = p.per_layer_skew(top_fraction=0.1)
    assert skew[0] > skew[1]


# ---------------------------------------------------------------------------
# save/load/merge
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path):
    p = HotExpertProfile(total_tokens=42, source="test")
    p.record(3, 7, 4)
    p.record(3, 8, 1)

    out = tmp_path / "profile.json"
    p.save(out)
    loaded = HotExpertProfile.load(out)
    assert loaded.total_tokens == 42
    assert loaded.hits == {(3, 7): 4, (3, 8): 1}


def test_merge_sums_hits():
    a = HotExpertProfile(total_tokens=10)
    a.record(0, 0, 3)
    b = HotExpertProfile(total_tokens=5)
    b.record(0, 0, 2)
    b.record(1, 1, 7)
    merged = HotExpertProfile.merge([a, b])
    assert merged.total_tokens == 15
    assert merged.hits == {(0, 0): 5, (1, 1): 7}


# ---------------------------------------------------------------------------
# parse_log_file
# ---------------------------------------------------------------------------


def test_parse_log_file(tmp_path: Path):
    log = tmp_path / "rpc.log"
    log.write_text("moe: layer=0 chosen=[0,1]\nmoe: layer=0 chosen=[0,2]\n")
    p = parse_log_file(log)
    assert p.hits[(0, 0)] == 2
    assert p.hits[(0, 1)] == 1
    assert p.source == str(log)
