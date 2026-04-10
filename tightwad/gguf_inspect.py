"""GGUF model inspection and cluster distribution planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    n_bytes: int

    @property
    def layer_index(self) -> int | None:
        """Extract layer index from tensor name like 'blk.42.attn_q.weight'."""
        for part in self.name.split("."):
            if part.isdigit():
                return int(part)
        return None


@dataclass
class MoEInfo:
    """Mixture-of-Experts model metadata."""

    n_expert: int  # total expert count (e.g. 128 for GPT-OSS 120B)
    n_expert_used: int | None  # active experts per token (e.g. 4)
    routing_overhead_bytes: int  # estimated per-device routing table size
    expert_tensor_names: list[str] = field(default_factory=list)

    @property
    def routing_overhead_gb(self) -> float:
        return self.routing_overhead_bytes / (1024**3)

    def min_vram_gb(self) -> int:
        """Minimum GPU VRAM recommended for RPC workers with this MoE model."""
        overhead_gb = self.routing_overhead_gb
        # Add 2GB headroom for KV cache, activations, etc.
        return max(4, int(overhead_gb + 2))


@dataclass
class ModelInfo:
    path: Path
    arch: str
    n_params: int | None
    n_layers: int
    quantization: str
    context_length: int | None
    total_size: int
    tensors: list[TensorInfo]
    metadata: dict[str, object] = field(default_factory=dict)
    moe: MoEInfo | None = None

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    @property
    def size_gb(self) -> float:
        return self.total_size / (1024**3)

    def layer_sizes(self) -> dict[int, int]:
        """Sum tensor bytes per layer index."""
        sizes: dict[int, int] = {}
        for t in self.tensors:
            idx = t.layer_index
            if idx is not None:
                sizes[idx] = sizes.get(idx, 0) + t.n_bytes
        return sizes

    def non_layer_size(self) -> int:
        """Bytes for embedding + output head (not assigned to any block)."""
        return sum(t.n_bytes for t in self.tensors if t.layer_index is None)


@dataclass
class GPUAssignment:
    gpu_name: str
    host: str
    layer_start: int
    layer_end: int  # exclusive
    est_vram_bytes: int

    @property
    def est_vram_gb(self) -> float:
        return self.est_vram_bytes / (1024**3)

    @property
    def n_layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass
class DistributionPlan:
    assignments: list[GPUAssignment]
    coordinator_ram_required: int  # bytes — full mmap
    non_layer_overhead: int  # bytes — embedding + output head

    @property
    def coordinator_ram_gb(self) -> float:
        return self.coordinator_ram_required / (1024**3)


def inspect_model(path: str | Path) -> ModelInfo:
    """Parse a GGUF file and return model metadata + tensor inventory."""
    from gguf import GGUFReader

    path = Path(path)
    reader = GGUFReader(str(path))

    # Extract metadata — kv.data holds indices into kv.parts
    from gguf import GGUFValueType

    meta: dict[str, object] = {}
    for kv in reader.fields.values():
        if len(kv.data) != 1 or len(kv.types) != 1:
            continue
        part = kv.parts[kv.data[0]]
        vtype = kv.types[0]
        if vtype == GGUFValueType.STRING:
            meta[kv.name] = bytes(part).decode("utf-8", errors="replace")
        elif vtype in (GGUFValueType.UINT32, GGUFValueType.INT32,
                       GGUFValueType.UINT64, GGUFValueType.INT64,
                       GGUFValueType.UINT16, GGUFValueType.INT16,
                       GGUFValueType.UINT8, GGUFValueType.INT8):
            meta[kv.name] = int(part[0])
        elif vtype in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
            meta[kv.name] = float(part[0])
        elif vtype == GGUFValueType.BOOL:
            meta[kv.name] = bool(part[0])

    arch = str(meta.get("general.architecture", "unknown"))
    n_layers = int(meta.get(f"{arch}.block_count", meta.get("general.block_count", 0)))
    context_length = meta.get(f"{arch}.context_length", None)
    if context_length is not None:
        context_length = int(context_length)

    # Quantization from general.file_type or filename heuristic
    file_type = meta.get("general.file_type", None)
    quantization = _file_type_to_quant(int(file_type)) if file_type is not None else _guess_quant(path.name)

    # Tensor inventory
    tensors: list[TensorInfo] = []
    total_size = 0
    for tensor in reader.tensors:
        n_bytes = int(tensor.n_bytes)
        total_size += n_bytes
        tensors.append(TensorInfo(
            name=tensor.name,
            shape=list(tensor.shape),
            dtype=str(tensor.tensor_type).split(".")[-1],
            n_bytes=n_bytes,
        ))

    # Parameter count estimate
    n_params = meta.get("general.parameter_count", None)
    if n_params is not None:
        n_params = int(n_params)

    # MoE detection: check for expert_count in metadata
    moe = _detect_moe(meta, arch, tensors)

    return ModelInfo(
        path=path,
        arch=arch,
        n_params=n_params,
        n_layers=n_layers,
        quantization=quantization,
        context_length=context_length,
        total_size=total_size,
        tensors=tensors,
        metadata=meta,
        moe=moe,
    )


def _detect_moe(
    meta: dict[str, object],
    arch: str,
    tensors: list[TensorInfo],
) -> MoEInfo | None:
    """Detect MoE metadata from GGUF KV pairs and tensor names.

    Checks ``{arch}.expert_count`` (standard GGUF key) and falls back to
    scanning tensor names for ``ffn_.*expert`` patterns.
    """
    # Primary: GGUF metadata key
    n_expert = meta.get(f"{arch}.expert_count")
    if n_expert is None:
        # Fallback aliases used by some models
        n_expert = meta.get("general.expert_count")
    if n_expert is not None:
        n_expert = int(n_expert)

    # Identify expert tensors: either "expert" in name or indexed FFN pattern
    # Common patterns:
    #   "blk.0.ffn_gate_exps.weight" (has "expert")
    #   "blk.0.ffn_gate.0.weight" (indexed by expert number)
    import re
    expert_tensors: list[str] = []
    expert_indices: set[int] = set()
    for t in tensors:
        if "expert" in t.name.lower():
            expert_tensors.append(t.name)
            continue
        # Indexed FFN: blk.N.ffn_{gate,up,down}.K.weight
        m = re.search(r"\.ffn_\w+\.(\d+)\.", t.name)
        if m:
            expert_tensors.append(t.name)
            expert_indices.add(int(m.group(1)))

    if n_expert is None and expert_indices:
        n_expert = max(expert_indices) + 1

    if n_expert is None or n_expert <= 1:
        return None

    n_expert_used = meta.get(f"{arch}.expert_used_count")
    if n_expert_used is None:
        n_expert_used = meta.get("general.expert_used_count")
    if n_expert_used is not None:
        n_expert_used = int(n_expert_used)

    # Estimate routing overhead: tensors that get replicated to every GPU.
    # In llama.cpp RPC, non-expert shared tensors (attention, norm, routing
    # gates) are replicated. Expert FFN tensors are split.
    # Heuristic: sum of all non-expert block tensors = shared per device.
    expert_name_set = set(expert_tensors)
    shared_bytes = 0
    for t in tensors:
        if t.name in expert_name_set:
            continue
        # Non-expert tensors (block or non-block) are shared
        shared_bytes += t.n_bytes

    return MoEInfo(
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        routing_overhead_bytes=shared_bytes,
        expert_tensor_names=expert_tensors,
    )


def plan_distribution(model_info: ModelInfo, config: "ClusterConfig") -> DistributionPlan:
    """Map model layers to GPUs based on tensor-split ratios from cluster config."""
    from .config import ClusterConfig

    split = config.tensor_split()
    # GPU order must match tensor_split(): workers first, then coordinator
    all_gpus: list = []
    for w in config.workers:
        all_gpus.extend(w.gpus)
    all_gpus.extend(config.coordinator_gpus)
    layer_sizes = model_info.layer_sizes()
    n_layers = model_info.n_layers
    non_layer = model_info.non_layer_size()

    if not all_gpus:
        return DistributionPlan(
            assignments=[],
            coordinator_ram_required=model_info.total_size,
            non_layer_overhead=non_layer,
        )

    # Distribute layers proportionally to VRAM split
    assignments: list[GPUAssignment] = []
    layer_cursor = 0

    for i, (gpu, ratio) in enumerate(zip(all_gpus, split)):
        # Last GPU gets remaining layers to avoid rounding gaps
        # (unless it has zero VRAM, in which case it gets nothing)
        if i == len(all_gpus) - 1 and ratio > 0:
            n = n_layers - layer_cursor
        else:
            n = round(n_layers * ratio)

        if n < 0:
            n = 0

        layer_start = layer_cursor
        layer_end = min(layer_cursor + n, n_layers)

        # Estimate VRAM: sum of layer tensor sizes in this range + share of non-layer
        est_bytes = sum(layer_sizes.get(l, 0) for l in range(layer_start, layer_end))
        if n > 0:
            est_bytes += int(non_layer * ratio)

        # Determine host
        host = "coordinator"
        for worker in config.workers:
            if gpu in worker.gpus:
                host = worker.host
                break

        assignments.append(GPUAssignment(
            gpu_name=gpu.name,
            host=host,
            layer_start=layer_start,
            layer_end=layer_end,
            est_vram_bytes=est_bytes,
        ))

        layer_cursor = layer_end

    return DistributionPlan(
        assignments=assignments,
        coordinator_ram_required=model_info.total_size,
        non_layer_overhead=non_layer,
    )


def check_moe_vram(
    model_info: ModelInfo,
    plan: DistributionPlan | None = None,
    gpu_vram: dict[str, int] | None = None,
) -> list[str]:
    """Check if MoE model's shared overhead will fit on each GPU.

    Parameters
    ----------
    model_info:
        Parsed model info (must have moe set).
    plan:
        Optional distribution plan (for GPU names from cluster config).
    gpu_vram:
        Optional dict of gpu_name → vram_gb. If not provided, falls back
        to checking plan assignments against overhead only.

    Returns a list of warning strings (empty if all GPUs are safe).
    """
    if not model_info.is_moe or model_info.moe is None:
        return []

    moe = model_info.moe
    warnings: list[str] = []
    min_vram = moe.min_vram_gb()

    if gpu_vram:
        for gpu_name, vram_gb in gpu_vram.items():
            if vram_gb < min_vram:
                warnings.append(
                    f"GPU {gpu_name} ({vram_gb}GB) likely OOM — "
                    f"MoE shared overhead is ~{moe.routing_overhead_gb:.1f}GB/device, "
                    f"minimum recommended: {min_vram}GB+"
                )
    elif plan:
        for a in plan.assignments:
            # Plan estimates don't include MoE replication overhead —
            # the estimated VRAM from tensor split + the shared overhead
            total_est_gb = a.est_vram_gb + moe.routing_overhead_gb
            # We don't have GPU VRAM capacity in the plan, so just warn
            # that the estimate is higher than it looks
            pass  # Warnings handled via gpu_vram dict from config

    if not warnings and min_vram > 0:
        warnings.append(
            f"MoE model with {moe.n_expert} experts — "
            f"~{moe.routing_overhead_gb:.1f}GB shared overhead replicated to every GPU. "
            f"Minimum {min_vram}GB VRAM per RPC worker recommended."
        )

    return warnings


def format_report(model_info: ModelInfo, plan: DistributionPlan | None = None) -> str:
    """Format model info (and optional distribution plan) as Rich-compatible output."""
    from rich.console import Console
    from rich.table import Table

    import io
    console = Console(record=True, width=100, file=io.StringIO())

    # Model summary
    console.print(f"\n[bold]Model:[/bold] {model_info.path.name}")
    if model_info.is_moe:
        moe = model_info.moe
        active = f", {moe.n_expert_used} active" if moe.n_expert_used else ""
        console.print(f"  Architecture: {model_info.arch} [bold yellow](MoE: {moe.n_expert} experts{active})[/bold yellow]")
    else:
        console.print(f"  Architecture: {model_info.arch}")
    if model_info.n_params:
        console.print(f"  Parameters:   {_human_params(model_info.n_params)}")
    console.print(f"  Layers:       {model_info.n_layers}")
    console.print(f"  Quantization: {model_info.quantization}")
    if model_info.context_length:
        console.print(f"  Context:      {model_info.context_length:,}")
    console.print(f"  Total size:   {model_info.size_gb:.2f} GB ({model_info.total_size:,} bytes)")
    console.print(f"  Tensors:      {len(model_info.tensors)}")
    console.print(f"  Mmap RAM:     {model_info.size_gb:.2f} GB")

    # MoE details
    if model_info.is_moe:
        moe = model_info.moe
        console.print(f"\n[bold yellow]MoE Details:[/bold yellow]")
        console.print(f"  Expert count:          {moe.n_expert}")
        if moe.n_expert_used:
            console.print(f"  Active per token:      {moe.n_expert_used}")
        console.print(f"  Shared overhead/GPU:   {moe.routing_overhead_gb:.2f} GB")
        console.print(f"  Min GPU VRAM for RPC:  {moe.min_vram_gb()} GB+")
        console.print(f"  Expert tensors:        {len(moe.expert_tensor_names)}")

    # Layer size table
    layer_sizes = model_info.layer_sizes()
    if layer_sizes:
        non_layer = model_info.non_layer_size()
        console.print(f"\n  Non-layer tensors (embed+output): {non_layer / (1024**2):.1f} MB")

        # Show range summary instead of every layer
        sizes = list(layer_sizes.values())
        if sizes:
            avg = sum(sizes) / len(sizes)
            console.print(f"  Per-layer avg: {avg / (1024**2):.1f} MB ({len(sizes)} layers)")

    # Distribution plan
    if plan:
        console.print(f"\n[bold]Distribution Plan[/bold]")
        console.print(f"  Coordinator RAM required: {plan.coordinator_ram_gb:.2f} GB")

        table = Table()
        table.add_column("GPU")
        table.add_column("Host")
        table.add_column("Layers")
        table.add_column("Est. VRAM")

        for a in plan.assignments:
            if a.n_layers == 0:
                layers = "(none)"
            else:
                layers = f"{a.layer_start}-{a.layer_end - 1} ({a.n_layers})"
            table.add_row(
                a.gpu_name,
                a.host,
                layers,
                f"{a.est_vram_gb:.2f} GB",
            )
        console.print(table)

        # MoE VRAM warnings
        if model_info.is_moe:
            warnings = check_moe_vram(model_info, plan)
            for w in warnings:
                console.print(f"  [bold red]WARNING:[/bold red] {w}")

    return console.export_text()


def _file_type_to_quant(ft: int) -> str:
    """Map GGUF file_type enum to human-readable quantization string."""
    mapping = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
        7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
        10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
        14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
        18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
    }
    return mapping.get(ft, f"type_{ft}")


def _guess_quant(filename: str) -> str:
    """Guess quantization from filename patterns like Q4_K_M."""
    import re
    m = re.search(r"((?:IQ|Q)\d+[_\w]*)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else "unknown"


def _human_params(n: int) -> str:
    """Format parameter count: 32000000000 -> '32.0B'."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)
