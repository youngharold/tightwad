"""Expert-aware placement: map MoE expert tensors to specific GPUs via ``-ot``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


FUSED_RE = re.compile(r"^blk\.(\d+)\.ffn_\w+_exps\.weight$")
INDEXED_RE = re.compile(r"^blk\.(\d+)\.ffn_(\w+)\.(\d+)\.weight$")


@dataclass(frozen=True)
class DeviceSlot:
    gpu_name: str
    host: str
    vram_gb: int
    ot_device: str


@dataclass
class ExpertAssignment:
    layer: int
    expert: int | str  # int for indexed, "fused" for fused
    device: DeviceSlot
    n_bytes: int


@dataclass
class PlacementPlan:
    assignments: list[ExpertAssignment]
    override_tensor_args: list[str]
    per_device_bytes: dict[str, int] = field(default_factory=dict)
    fused_fallback: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_cli_flags(self) -> list[str]:
        return list(self.override_tensor_args)

    def to_dict(self) -> dict:
        return {
            "assignments": [
                {
                    "layer": a.layer,
                    "expert": a.expert,
                    "device": a.device.ot_device,
                    "host": a.device.host,
                    "n_bytes": a.n_bytes,
                }
                for a in self.assignments
            ],
            "override_tensor_args": self.override_tensor_args,
            "per_device_bytes": self.per_device_bytes,
            "fused_fallback": self.fused_fallback,
            "warnings": self.warnings,
        }


def build_slots(config) -> list[DeviceSlot]:
    """Derive DeviceSlots from a ClusterConfig.

    Order matches ``tensor_split()``: RPC worker GPUs first, then coordinator
    locals. Each worker GPU becomes ``RPC[host:port]``; each coordinator GPU
    becomes ``CUDA{index}`` (index within the coordinator host).
    """
    slots: list[DeviceSlot] = []
    for worker in config.workers:
        for gpu in worker.gpus:
            port = gpu.rpc_port or 50052
            slots.append(DeviceSlot(
                gpu_name=gpu.name,
                host=worker.host,
                vram_gb=gpu.vram_gb,
                ot_device=f"RPC[{worker.host}:{port}]",
            ))
    for i, gpu in enumerate(config.coordinator_gpus):
        slots.append(DeviceSlot(
            gpu_name=gpu.name,
            host="coordinator",
            vram_gb=gpu.vram_gb,
            ot_device=f"CUDA{i}",
        ))
    return slots


def _enumerate_units(model_info) -> tuple[list[tuple[int, int | str, int]], bool]:
    """Return list of (layer, expert_id, bytes) tuples and a fused flag.

    - Indexed form ``blk.L.ffn_X.E.weight``: one unit per (layer, expert),
      aggregated across gate/up/down tensors.
    - Fused form ``blk.L.ffn_X_exps.weight``: one unit per (layer, "fused"),
      aggregated across gate/up/down. Caller may refuse to split.
    """
    fused_sizes: dict[int, int] = {}
    indexed_sizes: dict[tuple[int, int], int] = {}
    for t in model_info.tensors:
        m = INDEXED_RE.match(t.name)
        if m:
            layer = int(m.group(1))
            expert = int(m.group(3))
            indexed_sizes[(layer, expert)] = indexed_sizes.get((layer, expert), 0) + t.n_bytes
            continue
        m = FUSED_RE.match(t.name)
        if m:
            layer = int(m.group(1))
            fused_sizes[layer] = fused_sizes.get(layer, 0) + t.n_bytes

    if indexed_sizes:
        units = [(layer, expert, sz) for (layer, expert), sz in indexed_sizes.items()]
        return units, False
    units = [(layer, "fused", sz) for layer, sz in fused_sizes.items()]
    return units, bool(fused_sizes)


def plan_expert_placement(
    model_info,
    slots: list[DeviceSlot],
    hot_experts: dict[tuple[int, int], float] | None = None,
    device_scores: dict[str, float] | None = None,
    strategy: str = "balanced",
) -> PlacementPlan:
    """Assign (layer, expert) units to DeviceSlots using a bin-packing strategy.

    For fused-expert GGUFs (``ffn_*_exps.weight``) the caller should run
    ``tightwad moe defuse`` first. This function returns an empty plan with
    ``fused_fallback=True`` rather than producing bad ``-ot`` output.
    """
    if not model_info.is_moe or not slots:
        return PlacementPlan(assignments=[], override_tensor_args=[])

    units, fused = _enumerate_units(model_info)
    if fused:
        return PlacementPlan(
            assignments=[],
            override_tensor_args=[],
            fused_fallback=True,
            warnings=[
                "Model uses fused expert tensors (ffn_*_exps.weight). "
                "Run `tightwad moe defuse` to enable per-expert placement."
            ],
        )
    if not units:
        return PlacementPlan(assignments=[], override_tensor_args=[])

    shared_overhead_gb = model_info.moe.routing_overhead_gb
    capacities: dict[str, float] = {}
    for slot in slots:
        capacities[slot.ot_device] = max(
            0.0,
            (slot.vram_gb * 0.85 - shared_overhead_gb) * (1024 ** 3),
        )

    total_unit_bytes = sum(sz for _, _, sz in units)
    total_vram_gb = sum(s.vram_gb for s in slots) or 1
    targets: dict[str, float] = {
        s.ot_device: total_unit_bytes * (s.vram_gb / total_vram_gb) for s in slots
    }

    scores = device_scores or {slot.ot_device: 1.0 for slot in slots}

    weighted: list[tuple[float, int, int | str, int]] = []
    for layer, expert, sz in units:
        freq = 0.0
        if strategy == "profile-guided" and hot_experts:
            freq = hot_experts.get((layer, expert if isinstance(expert, int) else -1), 0.0)
        weight = sz * (1.0 + 3.0 * freq)
        weighted.append((weight, layer, expert, sz))
    weighted.sort(key=lambda x: x[0], reverse=True)

    allocated: dict[str, float] = {s.ot_device: 0.0 for s in slots}
    assignments: list[ExpertAssignment] = []
    slot_by_device = {s.ot_device: s for s in slots}

    def deficit(device: str) -> float:
        return targets[device] - allocated[device]

    def has_capacity(device: str, sz: int) -> bool:
        return capacities[device] - allocated[device] >= sz

    for weight, layer, expert, sz in weighted:
        freq = 0.0
        if strategy == "profile-guided" and hot_experts:
            freq = hot_experts.get((layer, expert if isinstance(expert, int) else -1), 0.0)

        candidates = [s for s in slots if has_capacity(s.ot_device, sz)]
        if not candidates:
            candidates = list(slots)

        if strategy == "profile-guided" and freq > 0:
            candidates.sort(
                key=lambda s: (scores.get(s.ot_device, 0.0), deficit(s.ot_device)),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda s: deficit(s.ot_device), reverse=True)

        chosen = candidates[0]
        allocated[chosen.ot_device] += sz
        assignments.append(ExpertAssignment(
            layer=layer, expert=expert, device=chosen, n_bytes=sz,
        ))

    override_args = render_override_tensor_regex(assignments)
    per_device: dict[str, int] = {}
    for a in assignments:
        per_device[a.device.ot_device] = per_device.get(a.device.ot_device, 0) + a.n_bytes

    return PlacementPlan(
        assignments=assignments,
        override_tensor_args=override_args,
        per_device_bytes=per_device,
    )


def render_override_tensor_regex(assignments: list[ExpertAssignment]) -> list[str]:
    """Group (layer, device) pairs into a regex per pair.

    Emits: ``^blk\\.L\\.ffn_(gate|up|down)\\.(E1|E2|...)\\.weight$=DEVICE``.
    llama.cpp ``-ot`` applies flags in order without override, so we emit
    narrowest-first (per-layer). Per-layer scope prevents cross-layer
    over-matching when two layers share expert indices.
    """
    grouped: dict[tuple[int, str], list[int]] = {}
    for a in assignments:
        if not isinstance(a.expert, int):
            continue
        key = (a.layer, a.device.ot_device)
        grouped.setdefault(key, []).append(a.expert)

    flags: list[str] = []
    for (layer, device), experts in sorted(grouped.items()):
        experts_sorted = sorted(set(experts))
        expert_alt = "|".join(str(e) for e in experts_sorted)
        pattern = fr"^blk\.{layer}\.ffn_(gate|up|down)\.({expert_alt})\.weight$"
        flags.append(f"{pattern}={device}")
    return flags
