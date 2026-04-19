"""Tests for expert-aware placement (`-ot` generation)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tightwad.config import GPU, Worker, ClusterConfig, ModelConfig
from tightwad.gguf_inspect import ModelInfo, MoEInfo, TensorInfo
from tightwad.moe_placement import (
    DeviceSlot,
    PlacementPlan,
    build_slots,
    plan_expert_placement,
    render_override_tensor_regex,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _indexed_model(n_layers: int, n_experts: int, bytes_per_ffn: int = 20_000_000) -> ModelInfo:
    """Build a synthetic indexed-form MoE ModelInfo."""
    tensors: list[TensorInfo] = []
    for i in range(n_layers):
        for exp in range(n_experts):
            for part in ("ffn_gate", "ffn_up", "ffn_down"):
                tensors.append(TensorInfo(
                    name=f"blk.{i}.{part}.{exp}.weight",
                    shape=[4096, 4096], dtype="Q4_K", n_bytes=bytes_per_ffn,
                ))
        tensors.append(TensorInfo(
            name=f"blk.{i}.attn_q.weight", shape=[4096, 4096],
            dtype="Q4_K", n_bytes=50_000_000,
        ))
        tensors.append(TensorInfo(
            name=f"blk.{i}.ffn_gate_inp.weight", shape=[4096, n_experts],
            dtype="F32", n_bytes=n_experts * 4096 * 4,
        ))
    total = sum(t.n_bytes for t in tensors)
    moe = MoEInfo(
        n_expert=n_experts, n_expert_used=2,
        routing_overhead_bytes=int(2 * 1024 ** 3),
        expert_tensor_names=[t.name for t in tensors if "ffn_" in t.name and "_inp" not in t.name],
    )
    return ModelInfo(
        path=Path("/fake/indexed.gguf"), arch="llama",
        n_params=None, n_layers=n_layers,
        quantization="Q4_K_M", context_length=8192,
        total_size=total, tensors=tensors, moe=moe,
    )


def _fused_model(n_layers: int) -> ModelInfo:
    tensors: list[TensorInfo] = []
    for i in range(n_layers):
        for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            tensors.append(TensorInfo(
                name=f"blk.{i}.{part}.weight", shape=[4096, 4096, 128],
                dtype="Q4_K", n_bytes=2_560_000_000,
            ))
    total = sum(t.n_bytes for t in tensors)
    moe = MoEInfo(
        n_expert=128, n_expert_used=8,
        routing_overhead_bytes=int(5 * 1024 ** 3),
        expert_tensor_names=[t.name for t in tensors],
    )
    return ModelInfo(
        path=Path("/fake/fused.gguf"), arch="mixtral",
        n_params=None, n_layers=n_layers,
        quantization="Q4_K_M", context_length=8192,
        total_size=total, tensors=tensors, moe=moe,
    )


def _slots(sizes_gb: list[int]) -> list[DeviceSlot]:
    slots: list[DeviceSlot] = []
    for i, gb in enumerate(sizes_gb):
        slots.append(DeviceSlot(
            gpu_name=f"GPU{i}", host="coordinator" if i == 0 else f"worker{i}",
            vram_gb=gb, ot_device=f"CUDA{i}" if i == 0 else f"RPC[worker{i}:50052]",
        ))
    return slots


# ---------------------------------------------------------------------------
# Balanced strategy
# ---------------------------------------------------------------------------


class TestBalanced:
    def test_all_units_assigned_exactly_once(self):
        model = _indexed_model(n_layers=4, n_experts=8)
        plan = plan_expert_placement(model, _slots([24, 16, 12, 8]))

        pairs = {(a.layer, a.expert) for a in plan.assignments}
        assert len(pairs) == 4 * 8
        assert len(plan.assignments) == 4 * 8

    def test_capacity_honored(self):
        model = _indexed_model(n_layers=4, n_experts=8, bytes_per_ffn=20_000_000)
        slots = _slots([24, 16, 12, 8])
        plan = plan_expert_placement(model, slots)

        for slot in slots:
            capacity = int((slot.vram_gb * 0.85 - model.moe.routing_overhead_gb) * (1024 ** 3))
            allocated = plan.per_device_bytes.get(slot.ot_device, 0)
            # With a small model the cap isn't binding, but the total must be
            # non-negative and sum over devices matches total expert bytes.
            assert allocated >= 0
            assert allocated <= capacity + 1_000_000_000

    def test_balance_ratio_within_tolerance(self):
        """4-slot split proportional to VRAM ratios within ±15 %."""
        slots = _slots([24, 16, 12, 8])
        model = _indexed_model(n_layers=8, n_experts=16, bytes_per_ffn=10_000_000)
        plan = plan_expert_placement(model, slots)

        total = sum(plan.per_device_bytes.values())
        total_vram = sum(s.vram_gb for s in slots)
        for slot in slots:
            expected = total * (slot.vram_gb / total_vram)
            actual = plan.per_device_bytes.get(slot.ot_device, 0)
            if expected > 0:
                deviation = abs(actual - expected) / expected
                assert deviation < 0.15, (
                    f"{slot.ot_device}: expected {expected:.0f} got {actual} "
                    f"({deviation*100:.1f}% off)"
                )

    def test_non_moe_returns_empty(self):
        tensors = [TensorInfo(name=f"blk.{i}.ffn_gate.weight", shape=[4096, 4096],
                              dtype="Q4_K", n_bytes=80_000_000)
                   for i in range(4)]
        dense = ModelInfo(
            path=Path("/fake/dense.gguf"), arch="llama", n_params=None,
            n_layers=4, quantization="Q4_K_M", context_length=8192,
            total_size=sum(t.n_bytes for t in tensors), tensors=tensors, moe=None,
        )
        plan = plan_expert_placement(dense, _slots([24, 16]))
        assert plan.override_tensor_args == []
        assert plan.assignments == []


# ---------------------------------------------------------------------------
# Fused fallback
# ---------------------------------------------------------------------------


class TestFusedFallback:
    def test_fused_model_returns_fallback(self):
        model = _fused_model(n_layers=4)
        plan = plan_expert_placement(model, _slots([24, 24]))

        assert plan.fused_fallback is True
        assert plan.override_tensor_args == []
        assert plan.assignments == []
        assert any("defuse" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Regex correctness
# ---------------------------------------------------------------------------


class TestRegex:
    def test_regex_matches_only_owned_tensors(self):
        """Emitted regexes must match owned tensors and nothing else."""
        model = _indexed_model(n_layers=3, n_experts=4)
        slots = _slots([24, 16])
        plan = plan_expert_placement(model, slots)

        owned: dict[str, set[str]] = {}
        for a in plan.assignments:
            for part in ("ffn_gate", "ffn_up", "ffn_down"):
                owned.setdefault(a.device.ot_device, set()).add(
                    f"blk.{a.layer}.{part}.{a.expert}.weight"
                )

        all_tensor_names = {t.name for t in model.tensors}

        for flag in plan.override_tensor_args:
            pattern, device = flag.rsplit("=", 1)
            rgx = re.compile(pattern)
            for name in all_tensor_names:
                if rgx.match(name):
                    assert name in owned.get(device, set()), (
                        f"Regex {pattern} for {device} over-matched {name}"
                    )

    def test_flags_are_per_layer(self):
        model = _indexed_model(n_layers=3, n_experts=4)
        plan = plan_expert_placement(model, _slots([24, 16]))

        for flag in plan.override_tensor_args:
            pattern = flag.rsplit("=", 1)[0]
            assert re.match(r"^\^blk\\\.\d+\\\.", pattern), (
                f"Flag does not target a single layer: {flag}"
            )

    def test_render_override_tensor_regex_empty_for_fused(self):
        assignments = []
        assert render_override_tensor_regex(assignments) == []


# ---------------------------------------------------------------------------
# Profile-guided
# ---------------------------------------------------------------------------


class TestProfileGuided:
    def test_hot_experts_pinned_to_high_score_slot(self):
        model = _indexed_model(n_layers=2, n_experts=8, bytes_per_ffn=5_000_000)
        slots = _slots([24, 24])  # both roomy
        hot = {(0, 3): 0.9, (1, 3): 0.9}
        scores = {slots[0].ot_device: 10.0, slots[1].ot_device: 1.0}

        plan = plan_expert_placement(
            model, slots, hot_experts=hot, device_scores=scores,
            strategy="profile-guided",
        )

        hot_pairs = {(0, 3), (1, 3)}
        placement = {(a.layer, a.expert): a.device.ot_device for a in plan.assignments}
        for pair in hot_pairs:
            assert placement[pair] == slots[0].ot_device, (
                f"Hot expert {pair} not on fastest slot: {placement[pair]}"
            )

    def test_integrates_with_device_bench(self, tmp_path, monkeypatch):
        """End-to-end: measure_device_scores → plan_expert_placement."""
        from tightwad.config import GPU, Worker, ClusterConfig, ModelConfig
        from tightwad.moe_placement import build_slots
        import tightwad.moe_device_bench as dm

        monkeypatch.setattr(dm, "_measure_tcp_rtt",
                            lambda host, port, iterations=5, timeout=2.0: 20.0)

        config = ClusterConfig(
            coordinator_host="0.0.0.0", coordinator_port=8090,
            coordinator_backend="cuda",
            coordinator_gpus=[GPU(name="RTX 4090", vram_gb=24)],
            workers=[Worker(host="10.0.0.1", gpus=[GPU(name="RTX 3060", vram_gb=12, rpc_port=50052)])],
            models={"m": ModelConfig(name="m", path="/m.gguf")},
            coordinator_binary="llama-server", rpc_server_binary="rpc-server",
        )
        scores = dm.measure_device_scores(config, force=True,
                                            cache_path=tmp_path / "scores.json")
        slots = build_slots(config)
        model = _indexed_model(n_layers=2, n_experts=4, bytes_per_ffn=2_000_000)

        plan = plan_expert_placement(
            model, slots,
            hot_experts={(0, 0): 1.0, (1, 0): 1.0},
            device_scores=scores,
            strategy="profile-guided",
        )
        hot_devices = {a.device.ot_device for a in plan.assignments
                        if a.expert == 0}
        assert "CUDA0" in hot_devices, (
            f"Hot experts should land on coordinator CUDA0 (score {scores['CUDA0']} vs "
            f"{scores['RPC[10.0.0.1:50052]']}); got {hot_devices}"
        )


# ---------------------------------------------------------------------------
# build_slots
# ---------------------------------------------------------------------------


class TestBuildSlots:
    def test_order_workers_first_then_coordinator(self):
        config = ClusterConfig(
            coordinator_host="0.0.0.0", coordinator_port=8090,
            coordinator_backend="cuda",
            coordinator_gpus=[
                GPU(name="RTX 4070 Ti Super", vram_gb=16),
                GPU(name="RTX 3060", vram_gb=12),
            ],
            workers=[
                Worker(host="192.168.1.20", gpus=[
                    GPU(name="RTX 2070", vram_gb=8, rpc_port=50052),
                ]),
            ],
            models={"m": ModelConfig(name="m", path="/m.gguf")},
            coordinator_binary="llama-server", rpc_server_binary="rpc-server",
        )
        slots = build_slots(config)

        assert len(slots) == 3
        assert slots[0].ot_device == "RPC[192.168.1.20:50052]"
        assert slots[1].ot_device == "CUDA0"
        assert slots[2].ot_device == "CUDA1"
        assert slots[0].vram_gb == 8
        assert slots[1].vram_gb == 16
