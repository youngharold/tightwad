"""Pure-Python GGUF binary parser (zero dependencies).

Parses GGUF v2/v3 headers, KV metadata, and tensor info without reading
tensor data.  This module exists so that ``pip install tightwad`` (without
extras) can still perform model family detection, loader pre-warming, and
basic metadata extraction — no ``gguf`` package required.

For full model inspection (tensor inventory, distribution planning), use
``pip install tightwad[inspect]`` which pulls in the official ``gguf``
package and enables :mod:`tightwad.gguf_inspect`.

**Decision (issue #38):** Evaluated dropping this in favor of the official
package.  Kept it because family detection at proxy startup and model
loading must work with zero optional dependencies.  The 336-line parser
covers v2/v3 headers which is sufficient for metadata extraction.  The
official package handles advanced features (tensor inventory, distribution
plans) via the ``[inspect]`` extra.
"""

from __future__ import annotations

import logging
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("tightwad.gguf_reader")

# ---------------------------------------------------------------------------
# GGUF type tables (from ggml-common.h)
# ---------------------------------------------------------------------------

# Maps GGUF tensor type enum → (type_size_bytes, block_size)
GGUF_TYPES: dict[int, tuple[int, int]] = {
    0: (4, 1),        # F32:  4 bytes per element
    1: (2, 1),        # F16:  2 bytes per element
    2: (18, 32),      # Q4_0: 18 bytes per block of 32
    3: (20, 32),      # Q4_1
    6: (22, 32),      # Q5_0
    7: (24, 32),      # Q5_1
    8: (34, 32),      # Q8_0
    9: (36, 32),      # Q8_1
    10: (54, 256),    # Q2_K
    11: (110, 256),   # Q3_K
    12: (144, 256),   # Q4_K
    13: (176, 256),   # Q5_K
    14: (210, 256),   # Q6_K
    15: (292, 256),   # Q8_K
    16: (10, 32),     # IQ2_XXS
    17: (12, 32),     # IQ2_XS
    18: (28, 256),    # IQ3_XXS
    19: (12, 32),     # IQ1_S
    20: (20, 32),     # IQ4_NL
    21: (44, 256),    # IQ3_S
    22: (4, 1),       # IQ2_S  (placeholder)
    23: (88, 256),    # IQ4_XS
    24: (1, 1),       # I8
    25: (2, 1),       # I16
    26: (4, 1),       # I32
    27: (8, 1),       # I64
    28: (8, 1),       # F64
    29: (10, 32),     # IQ1_M
    30: (2, 1),       # BF16
}

GGUF_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}

# KV value type enum
_KV_UINT8 = 0
_KV_INT8 = 1
_KV_UINT16 = 2
_KV_INT16 = 3
_KV_UINT32 = 4
_KV_INT32 = 5
_KV_FLOAT32 = 6
_KV_BOOL = 7
_KV_STRING = 8
_KV_ARRAY = 9
_KV_UINT64 = 10
_KV_INT64 = 11
_KV_FLOAT64 = 12

GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian uint32

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GGUFTensorEntry:
    name: str
    n_dims: int
    dims: list[int]
    dtype: int              # raw GGUF type enum
    dtype_name: str         # "Q4_K", "F16", etc.
    offset: int             # byte offset within tensor data block
    n_bytes: int            # computed from dims + dtype (-1 if unknown type)


@dataclass
class GGUFHeader:
    version: int
    tensor_count: int
    kv_count: int
    metadata: dict[str, object]
    tensors: list[GGUFTensorEntry]
    alignment: int          # from general.alignment KV or default 32
    data_offset: int        # byte position where tensor data starts
    file_size: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_string(f) -> str:
    """Read a GGUF string: uint64 length + raw bytes."""
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_kv_value(f, vtype: int) -> object:
    """Read a single KV value by type enum."""
    if vtype == _KV_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == _KV_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == _KV_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == _KV_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == _KV_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == _KV_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == _KV_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == _KV_BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == _KV_STRING:
        return _read_string(f)
    if vtype == _KV_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [_read_kv_value(f, elem_type) for _ in range(count)]
    if vtype == _KV_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == _KV_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == _KV_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    # Unknown type — skip is impossible without knowing size
    warnings.warn(f"Unknown GGUF KV type {vtype}", stacklevel=3)
    return None


def _compute_tensor_bytes(dims: list[int], dtype: int) -> int:
    """Compute tensor size in bytes from dimensions and type.

    Returns -1 if dtype is unknown.
    """
    if dtype not in GGUF_TYPES:
        return -1
    type_size, block_size = GGUF_TYPES[dtype]
    n_elements = 1
    for d in dims:
        n_elements *= d
    # Quantized types pack elements in blocks
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * type_size


def _align(offset: int, alignment: int) -> int:
    """Round up offset to the next alignment boundary."""
    return (offset + alignment - 1) & ~(alignment - 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_header(path: str | Path) -> GGUFHeader:
    """Parse GGUF file header, KV pairs, and tensor info.

    Does NOT read tensor data — only the header section.

    Parameters
    ----------
    path:
        Path to a GGUF file.

    Returns
    -------
    GGUFHeader with metadata, tensor entries, and data offset.

    Raises
    ------
    ValueError:
        If the file is not a valid GGUF file or has an unsupported version.
    """
    path = Path(path)
    file_size = path.stat().st_size

    try:
        with open(path, "rb") as f:
            # Magic
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                raise ValueError(f"File too small to be GGUF: {path}")
            (magic,) = struct.unpack("<I", magic_bytes)
            if magic != GGUF_MAGIC:
                raise ValueError(
                    f"Not a GGUF file (magic: 0x{magic:08X}, "
                    f"expected 0x{GGUF_MAGIC:08X}): {path}"
                )

            # Version
            (version,) = struct.unpack("<I", f.read(4))
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version {version}: {path}")

            # Counts — v2 uses uint64, v3 uses uint64
            (tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))

            # KV pairs
            metadata: dict[str, object] = {}
            for _ in range(kv_count):
                key = _read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                value = _read_kv_value(f, vtype)
                if value is not None:
                    metadata[key] = value

            alignment = int(metadata.get("general.alignment", 32))

            # Tensor info entries
            tensors: list[GGUFTensorEntry] = []
            for _ in range(tensor_count):
                name = _read_string(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                (dtype,) = struct.unpack("<I", f.read(4))
                (offset,) = struct.unpack("<Q", f.read(8))

                dtype_name = GGUF_TYPE_NAMES.get(dtype, f"type_{dtype}")
                n_bytes = _compute_tensor_bytes(dims, dtype)

                tensors.append(GGUFTensorEntry(
                    name=name,
                    n_dims=n_dims,
                    dims=dims,
                    dtype=dtype,
                    dtype_name=dtype_name,
                    offset=offset,
                    n_bytes=n_bytes,
                ))

            # Data offset: current position aligned to alignment boundary
            data_offset = _align(f.tell(), alignment)

    except (struct.error, EOFError) as e:
        raise ValueError(f"Malformed GGUF file: {e}") from e

    return GGUFHeader(
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        metadata=metadata,
        tensors=tensors,
        alignment=alignment,
        data_offset=data_offset,
        file_size=file_size,
    )


def tensor_data_range(
    header: GGUFHeader, tensor: GGUFTensorEntry
) -> tuple[int, int]:
    """Return (start_byte, end_byte) absolute file positions for a tensor.

    The tensor's ``offset`` field is relative to the data block start.
    This function adds ``header.data_offset`` to produce absolute positions.

    Returns
    -------
    (start, end) where end = start + tensor.n_bytes.
    Returns (start, start) if n_bytes is unknown (-1).
    """
    start = header.data_offset + tensor.offset
    if tensor.n_bytes < 0:
        return start, start
    return start, start + tensor.n_bytes


def model_summary(header: GGUFHeader) -> dict[str, object]:
    """Extract common model metadata from a parsed GGUF header.

    Returns a dict with keys: arch, params, layers, quant, context_length,
    total_size (bytes).
    """
    meta = header.metadata
    arch = str(meta.get("general.architecture", "unknown"))
    n_layers = meta.get(f"{arch}.block_count", meta.get("general.block_count"))
    context_length = meta.get(
        f"{arch}.context_length",
        meta.get("general.context_length"),
    )
    n_params = meta.get("general.parameter_count")

    # Quantization from file_type
    file_type = meta.get("general.file_type")
    quant = None
    if file_type is not None:
        _ft_map = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
            7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
            10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
            14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
            18: "Q6_K",
        }
        quant = _ft_map.get(int(file_type), f"type_{file_type}")

    # Total tensor data size
    total_size = sum(t.n_bytes for t in header.tensors if t.n_bytes > 0)

    return {
        "arch": arch,
        "params": int(n_params) if n_params is not None else None,
        "layers": int(n_layers) if n_layers is not None else None,
        "quant": quant,
        "context_length": int(context_length) if context_length is not None else None,
        "total_size": total_size,
    }
