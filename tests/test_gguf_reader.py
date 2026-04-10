"""Tests for pure-Python GGUF binary parser."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest

from tightwad.gguf_reader import (
    GGUF_MAGIC,
    GGUF_TYPES,
    GGUFHeader,
    GGUFTensorEntry,
    _compute_tensor_bytes,
    _align,
    read_header,
    tensor_data_range,
    model_summary,
)


def _build_gguf(
    *,
    version: int = 3,
    kv_pairs: list[tuple[str, int, bytes]] | None = None,
    tensors: list[tuple[str, list[int], int, int]] | None = None,
) -> bytes:
    """Build a minimal GGUF binary for testing.

    Parameters
    ----------
    version:
        GGUF version (2 or 3).
    kv_pairs:
        List of (key, value_type, packed_value_bytes).
    tensors:
        List of (name, dims, dtype, offset_in_data_block).

    Returns
    -------
    Raw bytes of a valid GGUF file.
    """
    if kv_pairs is None:
        kv_pairs = []
    if tensors is None:
        tensors = []

    buf = bytearray()
    # Magic + version
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", version)
    # Counts (uint64)
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", len(kv_pairs))

    # KV pairs
    for key, vtype, value_bytes in kv_pairs:
        key_enc = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_enc))
        buf += key_enc
        buf += struct.pack("<I", vtype)
        buf += value_bytes

    # Tensor info entries
    for name, dims, dtype, offset in tensors:
        name_enc = name.encode("utf-8")
        buf += struct.pack("<Q", len(name_enc))
        buf += name_enc
        buf += struct.pack("<I", len(dims))
        for d in dims:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<I", dtype)
        buf += struct.pack("<Q", offset)

    # Pad to default alignment (32 bytes)
    aligned = _align(len(buf), 32)
    buf += b"\x00" * (aligned - len(buf))

    # Add some dummy tensor data
    buf += b"\x00" * 256

    return bytes(buf)


def _kv_string(value: str) -> tuple[int, bytes]:
    """Build a STRING KV value (type 8)."""
    enc = value.encode("utf-8")
    return 8, struct.pack("<Q", len(enc)) + enc


def _kv_uint32(value: int) -> tuple[int, bytes]:
    """Build a UINT32 KV value (type 4)."""
    return 4, struct.pack("<I", value)


def _kv_bool(value: bool) -> tuple[int, bytes]:
    """Build a BOOL KV value (type 7)."""
    return 7, struct.pack("<B", int(value))


def _kv_array_uint32(values: list[int]) -> tuple[int, bytes]:
    """Build an ARRAY of UINT32 values (type 9, elem type 4)."""
    data = struct.pack("<I", 4)  # element type
    data += struct.pack("<Q", len(values))
    for v in values:
        data += struct.pack("<I", v)
    return 9, data


def _write_gguf(data: bytes) -> Path:
    """Write GGUF bytes to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
    f.write(data)
    f.close()
    return Path(f.name)


class TestReadHeaderMagicValidation:
    def test_non_gguf_raises(self, tmp_path):
        bad_file = tmp_path / "not_a_model.bin"
        bad_file.write_bytes(b"NOT_GGUF_DATA" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a GGUF file"):
            read_header(bad_file)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.gguf"
        empty.write_bytes(b"")
        with pytest.raises(ValueError):
            read_header(empty)

    def test_truncated_file_raises(self, tmp_path):
        short = tmp_path / "short.gguf"
        short.write_bytes(struct.pack("<I", GGUF_MAGIC))  # magic only
        with pytest.raises(ValueError):
            read_header(short)


class TestReadHeaderV3:
    def test_basic_header(self):
        data = _build_gguf(version=3)
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.version == 3
            assert header.tensor_count == 0
            assert header.kv_count == 0
            assert header.alignment == 32  # default
            assert header.file_size == len(data)
        finally:
            path.unlink()

    def test_uint64_counts(self):
        data = _build_gguf(version=3, tensors=[
            ("tensor.0", [32, 32], 0, 0),  # F32
        ])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.tensor_count == 1
        finally:
            path.unlink()


class TestReadKVTypes:
    def test_string(self):
        vtype, vbytes = _kv_string("hello world")
        data = _build_gguf(kv_pairs=[("test.key", vtype, vbytes)])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.metadata["test.key"] == "hello world"
        finally:
            path.unlink()

    def test_uint32(self):
        vtype, vbytes = _kv_uint32(42)
        data = _build_gguf(kv_pairs=[("test.count", vtype, vbytes)])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.metadata["test.count"] == 42
        finally:
            path.unlink()

    def test_bool(self):
        vtype, vbytes = _kv_bool(True)
        data = _build_gguf(kv_pairs=[("test.flag", vtype, vbytes)])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.metadata["test.flag"] is True
        finally:
            path.unlink()

    def test_array(self):
        vtype, vbytes = _kv_array_uint32([10, 20, 30])
        data = _build_gguf(kv_pairs=[("test.dims", vtype, vbytes)])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.metadata["test.dims"] == [10, 20, 30]
        finally:
            path.unlink()


class TestTensorInfoParsing:
    def test_single_tensor(self):
        data = _build_gguf(tensors=[
            ("blk.0.attn_q.weight", [4096, 4096], 1, 0),  # F16
        ])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert len(header.tensors) == 1
            t = header.tensors[0]
            assert t.name == "blk.0.attn_q.weight"
            assert t.n_dims == 2
            assert t.dims == [4096, 4096]
            assert t.dtype == 1  # F16
            assert t.dtype_name == "F16"
            assert t.offset == 0
        finally:
            path.unlink()

    def test_multiple_tensors(self):
        data = _build_gguf(tensors=[
            ("embed", [32000, 4096], 1, 0),
            ("blk.0.weight", [4096, 4096], 12, 33554432),
        ])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert len(header.tensors) == 2
            assert header.tensors[0].name == "embed"
            assert header.tensors[1].name == "blk.0.weight"
            assert header.tensors[1].dtype == 12  # Q4_K
        finally:
            path.unlink()


class TestComputeTensorBytes:
    def test_q4_0(self):
        # Q4_0: 18 bytes per block of 32 elements
        # 1024 elements = 32 blocks * 18 = 576
        assert _compute_tensor_bytes([1024], 2) == 576

    def test_f16(self):
        # F16: 2 bytes per element
        assert _compute_tensor_bytes([4096, 4096], 1) == 4096 * 4096 * 2

    def test_f32(self):
        # F32: 4 bytes per element
        assert _compute_tensor_bytes([100], 0) == 400

    def test_q4_k(self):
        # Q4_K: 144 bytes per block of 256 elements
        # 4096 elements = 16 blocks * 144 = 2304
        assert _compute_tensor_bytes([4096], 12) == 2304

    def test_unknown_type_returns_minus_one(self):
        assert _compute_tensor_bytes([100], 999) == -1

    def test_multidimensional(self):
        # F16, 2D: 32 * 64 = 2048 elements * 2 bytes = 4096
        assert _compute_tensor_bytes([32, 64], 1) == 4096


class TestAlignment:
    def test_default_32(self):
        data = _build_gguf()
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.alignment == 32
        finally:
            path.unlink()

    def test_custom_alignment(self):
        vtype, vbytes = _kv_uint32(64)
        data = _build_gguf(kv_pairs=[("general.alignment", vtype, vbytes)])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            assert header.alignment == 64
        finally:
            path.unlink()


class TestDataOffset:
    def test_data_offset_aligned(self):
        data = _build_gguf()
        path = _write_gguf(data)
        try:
            header = read_header(path)
            # Data offset must be aligned to alignment boundary
            assert header.data_offset % header.alignment == 0
        finally:
            path.unlink()

    def test_data_offset_after_header(self):
        data = _build_gguf(tensors=[
            ("t1", [100], 0, 0),
        ])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            # data_offset must be > 0 and past the header
            assert header.data_offset > 0
            # Must be within file
            assert header.data_offset <= header.file_size
        finally:
            path.unlink()


class TestTensorDataRange:
    def test_correct_positions(self):
        data = _build_gguf(tensors=[
            ("t1", [1024], 1, 0),       # F16: 2048 bytes at offset 0
            ("t2", [1024], 1, 2048),    # F16: 2048 bytes at offset 2048
        ])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            t1_start, t1_end = tensor_data_range(header, header.tensors[0])
            t2_start, t2_end = tensor_data_range(header, header.tensors[1])

            # t1 starts at data_offset + 0
            assert t1_start == header.data_offset
            assert t1_end == header.data_offset + 2048

            # t2 starts at data_offset + 2048
            assert t2_start == header.data_offset + 2048
            assert t2_end == header.data_offset + 2048 + 2048
        finally:
            path.unlink()

    def test_unknown_type_returns_zero_range(self):
        data = _build_gguf(tensors=[
            ("t1", [100], 999, 0),  # unknown type
        ])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            start, end = tensor_data_range(header, header.tensors[0])
            assert start == header.data_offset
            assert end == start  # zero-length range
        finally:
            path.unlink()


class TestModelSummary:
    def test_extracts_metadata(self):
        kv_pairs = [
            ("general.architecture", *_kv_string("qwen3")),
            ("qwen3.block_count", *_kv_uint32(64)),
            ("qwen3.context_length", *_kv_uint32(32768)),
            ("general.file_type", *_kv_uint32(15)),  # Q4_K_M
        ]
        data = _build_gguf(kv_pairs=kv_pairs, tensors=[
            ("blk.0.weight", [4096, 4096], 12, 0),
        ])
        path = _write_gguf(data)
        try:
            header = read_header(path)
            summary = model_summary(header)
            assert summary["arch"] == "qwen3"
            assert summary["layers"] == 64
            assert summary["context_length"] == 32768
            assert summary["quant"] == "Q4_K_M"
            assert summary["total_size"] > 0
        finally:
            path.unlink()

    def test_missing_metadata(self):
        data = _build_gguf()
        path = _write_gguf(data)
        try:
            header = read_header(path)
            summary = model_summary(header)
            assert summary["arch"] == "unknown"
            assert summary["layers"] is None
            assert summary["quant"] is None
        finally:
            path.unlink()


class TestAlignHelper:
    def test_already_aligned(self):
        assert _align(32, 32) == 32
        assert _align(64, 32) == 64

    def test_not_aligned(self):
        assert _align(33, 32) == 64
        assert _align(1, 32) == 32
        assert _align(63, 32) == 64
