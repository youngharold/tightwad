"""Tests for GGUF inspection and distribution planning."""

import pytest

from tightwad.gguf_inspect import (
    TensorInfo,
    ModelInfo,
    GPUAssignment,
    DistributionPlan,
    plan_distribution,
    format_report,
    _file_type_to_quant,
    _guess_quant,
    _human_params,
)
from tightwad.config import GPU, Worker, ClusterConfig, ModelConfig


@pytest.fixture
def sample_tensors():
    """Create a realistic set of tensors for a 4-layer model."""
    tensors = []
    # Embedding (no layer index)
    tensors.append(TensorInfo(name="token_embd.weight", shape=[4096, 32000], dtype="Q4_K", n_bytes=70_000_000))
    # Output head
    tensors.append(TensorInfo(name="output.weight", shape=[32000, 4096], dtype="Q6_K", n_bytes=90_000_000))
    tensors.append(TensorInfo(name="output_norm.weight", shape=[4096], dtype="F32", n_bytes=16_384))
    # 4 transformer blocks
    for i in range(4):
        for name, size in [
            ("attn_q.weight", 50_000_000),
            ("attn_k.weight", 10_000_000),
            ("attn_v.weight", 10_000_000),
            ("attn_output.weight", 50_000_000),
            ("ffn_gate.weight", 80_000_000),
            ("ffn_up.weight", 80_000_000),
            ("ffn_down.weight", 80_000_000),
            ("attn_norm.weight", 16_384),
            ("ffn_norm.weight", 16_384),
        ]:
            tensors.append(TensorInfo(
                name=f"blk.{i}.{name}",
                shape=[4096, 4096],
                dtype="Q4_K",
                n_bytes=size,
            ))
    return tensors


@pytest.fixture
def sample_model(tmp_path, sample_tensors):
    total = sum(t.n_bytes for t in sample_tensors)
    return ModelInfo(
        path=tmp_path / "test-model-Q4_K_M.gguf",
        arch="llama",
        n_params=7_000_000_000,
        n_layers=4,
        quantization="Q4_K_M",
        context_length=8192,
        total_size=total,
        tensors=sample_tensors,
    )


@pytest.fixture
def sample_config():
    return ClusterConfig(
        coordinator_host="0.0.0.0",
        coordinator_port=8080,
        coordinator_backend="cuda",
        coordinator_gpus=[GPU(name="P400", vram_gb=0)],
        workers=[
            Worker(
                host="192.168.1.100",
                gpus=[
                    GPU(name="4070", vram_gb=16, rpc_port=50052),
                    GPU(name="3060", vram_gb=12, rpc_port=50053),
                ],
            ),
            Worker(
                host="192.168.1.200",
                gpus=[GPU(name="2070", vram_gb=8, rpc_port=50052)],
            ),
        ],
        models={"test": ModelConfig(name="test", path="/test.gguf", default=True)},
        coordinator_binary="llama-server",
        rpc_server_binary="rpc-server",
    )


class TestTensorInfo:
    def test_layer_index_from_block(self):
        t = TensorInfo(name="blk.42.attn_q.weight", shape=[], dtype="Q4_K", n_bytes=100)
        assert t.layer_index == 42

    def test_layer_index_none_for_embedding(self):
        t = TensorInfo(name="token_embd.weight", shape=[], dtype="Q4_K", n_bytes=100)
        assert t.layer_index is None

    def test_layer_index_none_for_output(self):
        t = TensorInfo(name="output.weight", shape=[], dtype="Q4_K", n_bytes=100)
        assert t.layer_index is None


class TestModelInfo:
    def test_size_gb(self, sample_model):
        assert sample_model.size_gb > 0

    def test_layer_sizes(self, sample_model):
        sizes = sample_model.layer_sizes()
        assert len(sizes) == 4  # 4 layers
        # Each layer should be the same size
        values = list(sizes.values())
        assert all(v == values[0] for v in values)

    def test_non_layer_size(self, sample_model):
        non_layer = sample_model.non_layer_size()
        # Embedding + output + output_norm
        assert non_layer == 70_000_000 + 90_000_000 + 16_384


class TestDistributionPlan:
    def test_plan_assigns_all_layers(self, sample_model, sample_config):
        plan = plan_distribution(sample_model, sample_config)
        total_layers = sum(a.n_layers for a in plan.assignments)
        assert total_layers == sample_model.n_layers

    def test_plan_no_gaps(self, sample_model, sample_config):
        plan = plan_distribution(sample_model, sample_config)
        # Assignments should be contiguous
        for i in range(1, len(plan.assignments)):
            assert plan.assignments[i].layer_start == plan.assignments[i - 1].layer_end

    def test_zero_vram_gpu_gets_zero_layers(self, sample_model, sample_config):
        plan = plan_distribution(sample_model, sample_config)
        # Tensor split order: workers first (4070, 3060, 2070), then coordinator (P400)
        p400 = plan.assignments[-1]  # Last GPU is P400 (coordinator, vram_gb=0)
        assert p400.gpu_name == "P400"
        assert p400.n_layers == 0

    def test_coordinator_ram_equals_total_size(self, sample_model, sample_config):
        plan = plan_distribution(sample_model, sample_config)
        assert plan.coordinator_ram_required == sample_model.total_size

    def test_plan_empty_cluster(self, sample_model):
        config = ClusterConfig(
            coordinator_host="0.0.0.0",
            coordinator_port=8080,
            coordinator_backend="cuda",
            coordinator_gpus=[],
            workers=[],
            models={},
            coordinator_binary="llama-server",
            rpc_server_binary="rpc-server",
        )
        plan = plan_distribution(sample_model, config)
        assert len(plan.assignments) == 0


class TestFormatReport:
    def test_report_contains_model_name(self, sample_model):
        report = format_report(sample_model)
        assert "test-model-Q4_K_M.gguf" in report

    def test_report_contains_arch(self, sample_model):
        report = format_report(sample_model)
        assert "llama" in report

    def test_report_with_plan(self, sample_model, sample_config):
        plan = plan_distribution(sample_model, sample_config)
        report = format_report(sample_model, plan)
        assert "Distribution Plan" in report
        assert "P400" in report
        assert "4070" in report


class TestHelpers:
    def test_file_type_to_quant(self):
        assert _file_type_to_quant(15) == "Q4_K_M"
        assert _file_type_to_quant(7) == "Q8_0"
        assert _file_type_to_quant(0) == "F32"
        assert _file_type_to_quant(999) == "type_999"

    def test_guess_quant(self):
        assert _guess_quant("Qwen2.5-72B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
        assert _guess_quant("model-Q8_0.gguf") == "Q8_0"
        assert _guess_quant("model-IQ2_XS.gguf") == "IQ2_XS"
        assert _guess_quant("model.gguf") == "unknown"

    def test_human_params(self):
        assert _human_params(7_000_000_000) == "7.0B"
        assert _human_params(32_000_000_000) == "32.0B"
        assert _human_params(500_000_000) == "500.0M"
        assert _human_params(1234) == "1234"
