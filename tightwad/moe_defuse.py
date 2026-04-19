"""Rewrite fused MoE expert tensors to indexed form for per-expert ``-ot`` placement."""

from __future__ import annotations

import re
from pathlib import Path

FUSED_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def is_fused_expert(name: str) -> bool:
    return bool(FUSED_RE.match(name))


def split_fused_name(name: str) -> tuple[int, str] | None:
    m = FUSED_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def defuse_gguf(in_path: str | Path, out_path: str | Path) -> dict:
    """Rewrite fused expert tensors (``blk.L.ffn_*_exps.weight``) into indexed
    form (``blk.L.ffn_*.E.weight``). All other tensors and all KV metadata are
    copied unchanged.

    Returns a summary ``{'n_expert': int, 'layers_defused': int, 'tensors_written': int}``.

    Layout assumption: fused tensor numpy data has ``shape[0] == n_expert``
    (GGUF stores shapes reversed, so the stored shape shows n_expert last).
    Slicing along axis 0 preserves contiguous block structure for quantized
    tensors when ``prod(shape[1:]) % elements_per_block == 0``, which is the
    case for every MoE architecture llama.cpp ships today.
    """
    from gguf import GGUFReader, GGUFWriter
    from gguf.constants import GGUFValueType

    in_path = Path(in_path)
    out_path = Path(out_path)
    reader = GGUFReader(str(in_path))

    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        raise ValueError("GGUF missing general.architecture")
    arch = _scalar_str(arch_field)

    n_expert = _find_expert_count(reader, arch)
    if n_expert is None or n_expert <= 1:
        raise ValueError(
            f"GGUF at {in_path} has no MoE expert_count or n_expert <= 1. "
            "Defusion only applies to MoE models."
        )

    writer = GGUFWriter(str(out_path), arch)

    for name, field in reader.fields.items():
        if name in _HEADER_KEYS:
            continue
        _copy_kv(writer, name, field, GGUFValueType)

    layers_defused = 0
    tensors_written = 0
    for tensor in reader.tensors:
        if is_fused_expert(tensor.name):
            layer_part = split_fused_name(tensor.name)
            if layer_part is None:
                continue
            layer, part = layer_part
            data = tensor.data
            if data.shape[0] != n_expert:
                raise ValueError(
                    f"Expected fused tensor {tensor.name} axis 0 to equal "
                    f"n_expert={n_expert}, got {data.shape[0]}"
                )
            for e in range(n_expert):
                slice_data = data[e].copy()
                indexed_name = f"blk.{layer}.ffn_{part}.{e}.weight"
                writer.add_tensor(indexed_name, slice_data, raw_dtype=tensor.tensor_type)
                tensors_written += 1
            layers_defused += 1
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            tensors_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "n_expert": n_expert,
        "layers_defused": layers_defused,
        "tensors_written": tensors_written,
    }


_HEADER_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}


def _scalar_str(field) -> str:
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def _find_expert_count(reader, arch: str) -> int | None:
    for key in (f"{arch}.expert_count", "general.expert_count"):
        f = reader.fields.get(key)
        if f is None:
            continue
        part = f.parts[f.data[0]]
        if hasattr(part, "tolist"):
            val = part.tolist()
            return int(val[0] if isinstance(val, list) else val)
        return int(part)
    return None


def _copy_kv(writer, name: str, field, value_type_enum) -> None:
    """Copy a KV field from reader to writer, preserving type.

    gguf's Writer has dozens of typed add_* helpers; the generic fallback is
    ``add_key_value(key, value, type)`` plus ``add_array`` for lists.
    """
    if not field.types:
        return
    primary = field.types[0]
    value = _field_value(reader_field=field, primary=primary, value_type_enum=value_type_enum)
    if value is None:
        return
    if primary == value_type_enum.ARRAY:
        writer.add_array(name, value)
    else:
        writer.add_key_value(name, value, primary)


def _field_value(reader_field, primary, value_type_enum):
    if primary == value_type_enum.STRING:
        return bytes(reader_field.parts[reader_field.data[0]]).decode("utf-8")
    if primary == value_type_enum.ARRAY:
        items = []
        if len(reader_field.types) < 2:
            return items
        inner = reader_field.types[1]
        for idx in reader_field.data:
            part = reader_field.parts[idx]
            if inner == value_type_enum.STRING:
                items.append(bytes(part).decode("utf-8"))
            else:
                items.append(part.tolist()[0] if hasattr(part, "tolist") else part)
        return items
    part = reader_field.parts[reader_field.data[0]]
    if hasattr(part, "tolist"):
        val = part.tolist()
        return val[0] if isinstance(val, list) and len(val) == 1 else val
    return part


