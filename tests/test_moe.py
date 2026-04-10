"""Tests for MoE detection, VRAM estimation, and warnings."""

import pytest

from tightwad.gguf_inspect import (
    MoEInfo,
    ModelInfo,
    TensorInfo,
    _detect_moe,
    check_moe_vram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tensors(n_layers=4, n_experts=0, include_expert_tensors=True):
    """Create a realistic set of tensors, optionally with MoE expert layers."""
    tensors = []
    # Embedding + output head
    tensors.append(TensorInfo(
        name="token_embd.weight", shape=[4096, 32000], dtype="Q4_K",
        n_bytes=70_000_000,
    ))
    tensors.append(TensorInfo(
        name="output.weight", shape=[32000, 4096], dtype="Q6_K",
        n_bytes=90_000_000,
    ))

    for i in range(n_layers):
        # Shared attention tensors (replicated to every GPU)
        for name, size in [
            ("attn_q.weight", 50_000_000),
            ("attn_k.weight", 10_000_000),
            ("attn_v.weight", 10_000_000),
            ("attn_output.weight", 50_000_000),
            ("attn_norm.weight", 16_384),
            ("ffn_norm.weight", 16_384),
        ]:
            tensors.append(TensorInfo(
                name=f"blk.{i}.{name}", shape=[4096, 4096],
                dtype="Q4_K", n_bytes=size,
            ))

        if n_experts > 0 and include_expert_tensors:
            # MoE expert FFN tensors (split across GPUs)
            for exp in range(n_experts):
                for name, size in [
                    ("ffn_gate", 20_000_000),
                    ("ffn_up", 20_000_000),
                    ("ffn_down", 20_000_000),
                ]:
                    tensors.append(TensorInfo(
                        name=f"blk.{i}.{name}.{exp}.weight",
                        shape=[4096, 4096], dtype="Q4_K", n_bytes=size,
                    ))
            # Routing gate (shared)
            tensors.append(TensorInfo(
                name=f"blk.{i}.ffn_gate_inp.weight",
                shape=[4096, n_experts], dtype="F32", n_bytes=n_experts * 4096 * 4,
            ))
        else:
            # Dense FFN tensors
            for name, size in [
                ("ffn_gate.weight", 80_000_000),
                ("ffn_up.weight", 80_000_000),
                ("ffn_down.weight", 80_000_000),
            ]:
                tensors.append(TensorInfo(
                    name=f"blk.{i}.{name}", shape=[4096, 4096],
                    dtype="Q4_K", n_bytes=size,
                ))

    return tensors


# ---------------------------------------------------------------------------
# _detect_moe
# ---------------------------------------------------------------------------


class TestDetectMoE:
    def test_detects_from_metadata_key(self):
        meta = {"llama.expert_count": 8, "llama.expert_used_count": 2}
        tensors = _make_tensors(n_layers=2, n_experts=8)
        result = _detect_moe(meta, "llama", tensors)

        assert result is not None
        assert result.n_expert == 8
        assert result.n_expert_used == 2

    def test_detects_from_general_key(self):
        meta = {"general.expert_count": 64}
        tensors = _make_tensors(n_layers=2, n_experts=64)
        result = _detect_moe(meta, "mixtral", tensors)

        assert result is not None
        assert result.n_expert == 64

    def test_no_moe_returns_none(self):
        meta = {}
        tensors = _make_tensors(n_layers=4, n_experts=0)
        result = _detect_moe(meta, "llama", tensors)

        assert result is None

    def test_single_expert_returns_none(self):
        meta = {"llama.expert_count": 1}
        tensors = _make_tensors(n_layers=2)
        result = _detect_moe(meta, "llama", tensors)

        assert result is None

    def test_fallback_tensor_name_detection(self):
        """Detect MoE from tensor names when metadata key is missing."""
        meta = {}
        tensors = _make_tensors(n_layers=2, n_experts=8)
        result = _detect_moe(meta, "mixtral", tensors)

        assert result is not None
        assert result.n_expert == 8

    def test_expert_used_count_optional(self):
        meta = {"llama.expert_count": 16}
        tensors = _make_tensors(n_layers=2, n_experts=16)
        result = _detect_moe(meta, "llama", tensors)

        assert result is not None
        assert result.n_expert_used is None

    def test_routing_overhead_excludes_expert_tensors(self):
        meta = {"llama.expert_count": 8}
        tensors = _make_tensors(n_layers=2, n_experts=8)
        result = _detect_moe(meta, "llama", tensors)

        assert result is not None
        # Expert tensors should NOT be in routing overhead
        assert result.routing_overhead_bytes > 0
        expert_total = sum(t.n_bytes for t in tensors if "expert" in t.name.lower())
        # Routing overhead should be much less than expert total
        assert result.routing_overhead_bytes < sum(t.n_bytes for t in tensors)

    def test_expert_tensor_names_collected(self):
        meta = {"llama.expert_count": 4}
        tensors = _make_tensors(n_layers=1, n_experts=4)
        result = _detect_moe(meta, "llama", tensors)

        assert result is not None
        assert len(result.expert_tensor_names) > 0
        assert all("expert" in name.lower() or "ffn_" in name.lower()
                    for name in result.expert_tensor_names
                    if "." in name and any(c.isdigit() for c in name.split("ffn_")[-1][:2]))


# ---------------------------------------------------------------------------
# MoEInfo
# ---------------------------------------------------------------------------


class TestMoEInfo:
    def test_routing_overhead_gb(self):
        moe = MoEInfo(
            n_expert=8, n_expert_used=2,
            routing_overhead_bytes=int(5 * 1024**3),
        )
        assert abs(moe.routing_overhead_gb - 5.0) < 0.01

    def test_min_vram_gb(self):
        moe = MoEInfo(
            n_expert=128, n_expert_used=4,
            routing_overhead_bytes=int(20 * 1024**3),  # 20GB overhead
        )
        assert moe.min_vram_gb() == 22  # 20 + 2 headroom

    def test_min_vram_gb_floor(self):
        moe = MoEInfo(
            n_expert=4, n_expert_used=2,
            routing_overhead_bytes=int(0.5 * 1024**3),
        )
        assert moe.min_vram_gb() >= 4  # minimum floor


# ---------------------------------------------------------------------------
# check_moe_vram
# ---------------------------------------------------------------------------


class TestCheckMoeVram:
    def _model_info(self, n_experts=8, overhead_gb=5.0):
        tensors = _make_tensors(n_layers=2, n_experts=n_experts)
        total = sum(t.n_bytes for t in tensors)
        return ModelInfo(
            path=__import__("pathlib").Path("/fake/model.gguf"),
            arch="mixtral",
            n_params=None,
            n_layers=2,
            quantization="Q4_K_M",
            context_length=8192,
            total_size=total,
            tensors=tensors,
            moe=MoEInfo(
                n_expert=n_experts,
                n_expert_used=2,
                routing_overhead_bytes=int(overhead_gb * 1024**3),
            ),
        )

    def test_warns_on_small_gpu(self):
        model = self._model_info(n_experts=128, overhead_gb=20.0)
        gpu_vram = {"RTX 3060": 12, "RTX 4070": 16}
        warnings = check_moe_vram(model, gpu_vram=gpu_vram)

        assert len(warnings) >= 2
        assert any("RTX 3060" in w and "OOM" in w for w in warnings)
        assert any("RTX 4070" in w and "OOM" in w for w in warnings)

    def test_no_warning_for_big_gpus(self):
        model = self._model_info(n_experts=8, overhead_gb=3.0)
        gpu_vram = {"A100": 80, "RTX 4090": 24}
        warnings = check_moe_vram(model, gpu_vram=gpu_vram)

        # Should only have the general info message, no OOM warnings
        assert not any("OOM" in w for w in warnings)

    def test_no_warnings_for_dense_model(self):
        tensors = _make_tensors(n_layers=4, n_experts=0)
        total = sum(t.n_bytes for t in tensors)
        model = ModelInfo(
            path=__import__("pathlib").Path("/fake/model.gguf"),
            arch="llama",
            n_params=None,
            n_layers=4,
            quantization="Q4_K_M",
            context_length=8192,
            total_size=total,
            tensors=tensors,
        )
        warnings = check_moe_vram(model, gpu_vram={"RTX 3060": 12})
        assert warnings == []

    def test_general_info_when_no_gpu_vram_issues(self):
        model = self._model_info(n_experts=8, overhead_gb=3.0)
        gpu_vram = {"A100": 80}
        warnings = check_moe_vram(model, gpu_vram=gpu_vram)

        assert len(warnings) == 1
        assert "8 experts" in warnings[0]
        assert "OOM" not in warnings[0]
