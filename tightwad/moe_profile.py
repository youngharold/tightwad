"""Capture per-expert routing counts from llama.cpp stderr.

Primary format (requires the ``scripts/patches/llamacpp-moe-log.patch``
instrumentation and ``LLAMA_LOG_MOE=1``):

    moe: layer=12 chosen=[47,88,12,3]

Fallback for unpatched builds: count any ``n_expert_used=N`` token-event line
as a routing event, which populates ``total_tokens`` without per-expert hits.
``profile-guided`` placement then degrades to ``balanced``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable


MOE_LINE_RE = re.compile(
    r"moe:\s*layer\s*=\s*(\d+)\s+chosen\s*=\s*\[([0-9,\s]*)\]"
)
FALLBACK_LINE_RE = re.compile(r"n_expert_used\s*[=:]\s*(\d+)")


@dataclass
class ExpertHit:
    layer: int
    expert: int
    count: int


@dataclass
class HotExpertProfile:
    hits: dict[tuple[int, int], int] = field(default_factory=dict)
    total_tokens: int = 0
    source: str = "unknown"

    def record(self, layer: int, expert: int, count: int = 1) -> None:
        key = (layer, expert)
        self.hits[key] = self.hits.get(key, 0) + count

    def frequency(self) -> dict[tuple[int, int], float]:
        if not self.hits:
            return {}
        total = sum(self.hits.values())
        return {k: v / total for k, v in self.hits.items()}

    def top_n(self, n: int = 32) -> list[ExpertHit]:
        items = sorted(self.hits.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [ExpertHit(layer=k[0], expert=k[1], count=v) for k, v in items]

    def per_layer_skew(self, top_fraction: float = 0.1) -> dict[int, float]:
        by_layer: dict[int, list[int]] = {}
        for (layer, _expert), count in self.hits.items():
            by_layer.setdefault(layer, []).append(count)
        skew: dict[int, float] = {}
        for layer, counts in by_layer.items():
            counts.sort(reverse=True)
            k = max(1, int(len(counts) * top_fraction))
            skew[layer] = sum(counts[:k]) / max(1, sum(counts))
        return skew

    def save(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_tokens": self.total_tokens,
            "source": self.source,
            "hits": [
                {"layer": layer, "expert": expert, "count": count}
                for (layer, expert), count in self.hits.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HotExpertProfile":
        path = Path(path).expanduser()
        data = json.loads(path.read_text())
        p = cls(total_tokens=data.get("total_tokens", 0),
                source=data.get("source", "unknown"))
        for h in data.get("hits", []):
            p.hits[(int(h["layer"]), int(h["expert"]))] = int(h["count"])
        return p

    @classmethod
    def merge(cls, profiles: Iterable["HotExpertProfile"]) -> "HotExpertProfile":
        out = cls()
        for p in profiles:
            out.total_tokens += p.total_tokens
            for key, count in p.hits.items():
                out.hits[key] = out.hits.get(key, 0) + count
            if p.source != "unknown":
                out.source = p.source
        return out


def parse_stderr_stream(stream: IO[str]) -> HotExpertProfile:
    profile = HotExpertProfile(source="stderr")
    for line in stream:
        _consume_line(profile, line)
    return profile


def parse_log_file(path: str | Path) -> HotExpertProfile:
    path = Path(path).expanduser()
    profile = HotExpertProfile(source=str(path))
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            _consume_line(profile, line)
    return profile


def _consume_line(profile: HotExpertProfile, line: str) -> None:
    m = MOE_LINE_RE.search(line)
    if m:
        layer = int(m.group(1))
        experts_s = m.group(2).strip()
        if not experts_s:
            return
        for e in experts_s.split(","):
            e = e.strip()
            if e.isdigit():
                profile.record(layer, int(e))
        profile.total_tokens += 1
        return
    m = FALLBACK_LINE_RE.search(line)
    if m:
        profile.total_tokens += 1
