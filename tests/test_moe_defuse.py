"""Tests for GGUF defusion (fused → indexed expert tensors)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


gguf = pytest.importorskip("gguf")
from gguf import GGUFReader, GGUFWriter


def _write_fused_gguf(path: Path, n_expert: int = 4, n_layers: int = 2,
                      n_embd: int = 16, n_ff: int = 8) -> dict[str, np.ndarray]:
    """Write a synthetic fused-MoE GGUF and return the original expert data
    so tests can assert round-trip equality."""
    w = GGUFWriter(str(path), "llama")
    w.add_expert_count(n_expert)
    w.add_expert_used_count(2)
    w.add_block_count(n_layers)

    originals: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed=0)
    for layer in range(n_layers):
        for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            arr = rng.random((n_expert, n_ff, n_embd), dtype=np.float32)
            name = f"blk.{layer}.{part}.weight"
            originals[name] = arr
            w.add_tensor(name, arr)
        non_expert = rng.random((n_embd, n_embd), dtype=np.float32)
        w.add_tensor(f"blk.{layer}.attn_q.weight", non_expert)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return originals


def test_defuse_round_trip_preserves_weights():
    from tightwad.moe_defuse import defuse_gguf

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "fused.gguf"
        out_path = Path(td) / "indexed.gguf"
        originals = _write_fused_gguf(in_path, n_expert=4, n_layers=2)

        summary = defuse_gguf(in_path, out_path)
        assert summary["n_expert"] == 4
        assert summary["layers_defused"] == 2 * 3  # gate + up + down per layer
        # 2 layers × (3 fused → 12 indexed) + 2 non-expert tensors = 26
        assert summary["tensors_written"] == 2 * 12 + 2

        r = GGUFReader(str(out_path))
        indexed = {t.name: t.data for t in r.tensors}

        # Every original fused block must produce n_expert indexed tensors
        # whose data equals the corresponding slice of the fused source.
        for fused_name, fused_data in originals.items():
            # "blk.0.ffn_gate_exps.weight" → part = "gate"
            part = fused_name.split(".")[2].replace("_exps", "")
            layer = fused_name.split(".")[1]
            for e in range(fused_data.shape[0]):
                indexed_name = f"blk.{layer}.{part}.{e}.weight"
                assert indexed_name in indexed, f"Missing {indexed_name}"
                np.testing.assert_array_equal(
                    indexed[indexed_name],
                    fused_data[e],
                    err_msg=f"Mismatch for expert {e} in {fused_name}",
                )


def test_defuse_preserves_non_expert_tensors():
    from tightwad.moe_defuse import defuse_gguf

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "fused.gguf"
        out_path = Path(td) / "indexed.gguf"
        _write_fused_gguf(in_path, n_expert=4, n_layers=1)

        defuse_gguf(in_path, out_path)
        r_in = GGUFReader(str(in_path))
        r_out = GGUFReader(str(out_path))

        in_non_expert = {t.name: t.data for t in r_in.tensors
                          if not t.name.endswith("_exps.weight")}
        out_non_expert = {t.name: t.data for t in r_out.tensors
                           if "ffn_gate" not in t.name and "ffn_up" not in t.name
                           and "ffn_down" not in t.name}

        assert set(in_non_expert.keys()) == set(out_non_expert.keys())
        for name in in_non_expert:
            np.testing.assert_array_equal(in_non_expert[name], out_non_expert[name])


def test_defuse_preserves_expert_count_metadata():
    from tightwad.moe_defuse import defuse_gguf

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "fused.gguf"
        out_path = Path(td) / "indexed.gguf"
        _write_fused_gguf(in_path, n_expert=8, n_layers=1)

        defuse_gguf(in_path, out_path)
        r = GGUFReader(str(out_path))
        f = r.fields.get("llama.expert_count")
        assert f is not None
        val = f.parts[f.data[0]].tolist()
        assert (val[0] if isinstance(val, list) else val) == 8


def test_defuse_rejects_dense_model():
    from tightwad.moe_defuse import defuse_gguf

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "dense.gguf"
        out_path = Path(td) / "out.gguf"

        w = GGUFWriter(str(in_path), "llama")
        w.add_block_count(2)
        arr = np.random.rand(16, 16).astype(np.float32)
        w.add_tensor("blk.0.attn_q.weight", arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        with pytest.raises(ValueError, match="MoE"):
            defuse_gguf(in_path, out_path)
