"""Swarm manifest generation and bitfield tracking for P2P model distribution."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

DEFAULT_PIECE_SIZE = 64 * 1024 * 1024  # 64 MB


@dataclass
class PieceInfo:
    index: int
    offset: int
    size: int
    sha256: str


@dataclass
class SwarmManifest:
    model: str
    filename: str
    total_size: int
    piece_size: int
    pieces: list[PieceInfo]
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def num_pieces(self) -> int:
        return len(self.pieces)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "filename": self.filename,
            "total_size": self.total_size,
            "piece_size": self.piece_size,
            "pieces": [
                {"index": p.index, "offset": p.offset, "size": p.size, "sha256": p.sha256}
                for p in self.pieces
            ],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SwarmManifest:
        pieces = [
            PieceInfo(index=p["index"], offset=p["offset"], size=p["size"], sha256=p["sha256"])
            for p in d["pieces"]
        ]
        return cls(
            model=d["model"],
            filename=d["filename"],
            total_size=d["total_size"],
            piece_size=d["piece_size"],
            pieces=pieces,
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", ""),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> SwarmManifest:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def find_for_model(cls, model_path: Path | str) -> SwarmManifest | None:
        model_path = Path(model_path)
        manifest_path = model_path.parent / f"{model_path.name}.tightwad.manifest"
        if manifest_path.exists():
            return cls.load(manifest_path)
        return None


@dataclass
class PieceBitfield:
    pieces_file: Path
    have: set[int] = field(default_factory=set)
    _total: int = 0

    @classmethod
    def load_or_create(cls, pieces_file: Path | str, total_pieces: int) -> PieceBitfield:
        pieces_file = Path(pieces_file)
        bf = cls(pieces_file=pieces_file, _total=total_pieces)
        if pieces_file.exists():
            try:
                data = json.loads(pieces_file.read_text())
                bf.have = set(data.get("have", []))
            except (json.JSONDecodeError, KeyError):
                bf.have = set()
        return bf

    def mark_have(self, index: int) -> None:
        self.have.add(index)

    def mark_missing(self, index: int) -> None:
        self.have.discard(index)

    def save(self) -> None:
        self.pieces_file.write_text(json.dumps({"have": sorted(self.have)}))

    def completion_pct(self) -> float:
        if self._total == 0:
            return 100.0
        return len(self.have) / self._total * 100.0

    def missing_pieces(self) -> list[int]:
        return sorted(set(range(self._total)) - self.have)

    def have_all(self) -> bool:
        return len(self.have) >= self._total


def create_manifest(
    model_path: Path | str,
    piece_size: int = DEFAULT_PIECE_SIZE,
    use_gguf_inspect: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SwarmManifest:
    """Create a swarm manifest for a model file.

    Reads the file in chunks, SHA256-hashes each piece.
    Optionally extracts GGUF metadata via gguf_inspect.
    """
    model_path = Path(model_path)
    total_size = model_path.stat().st_size

    metadata: dict = {}
    if use_gguf_inspect:
        try:
            from .gguf_inspect import inspect_model, _human_params
            info = inspect_model(model_path)
            metadata["arch"] = info.arch
            metadata["quantization"] = info.quantization
            metadata["n_layers"] = info.n_layers
            if info.n_params:
                metadata["params"] = _human_params(info.n_params)
            if info.context_length:
                metadata["context_length"] = info.context_length
        except Exception:
            pass

    pieces: list[PieceInfo] = []
    offset = 0
    index = 0

    with open(model_path, "rb") as f:
        while offset < total_size:
            chunk = f.read(piece_size)
            if not chunk:
                break
            sha = hashlib.sha256(chunk).hexdigest()
            pieces.append(PieceInfo(
                index=index,
                offset=offset,
                size=len(chunk),
                sha256=sha,
            ))
            offset += len(chunk)
            index += 1
            if progress_callback:
                progress_callback(index, -1)  # total unknown until done

    # Model name from filename (strip extension)
    model_name = model_path.stem
    if model_name.endswith(".gguf"):
        model_name = model_name[:-5]

    return SwarmManifest(
        model=model_name,
        filename=model_path.name,
        total_size=total_size,
        piece_size=piece_size,
        pieces=pieces,
        metadata=metadata,
    )


def verify_piece(model_path: Path | str, piece: PieceInfo) -> bool:
    """Read a piece from disk and verify its SHA256 hash."""
    model_path = Path(model_path)
    with open(model_path, "rb") as f:
        f.seek(piece.offset)
        data = f.read(piece.size)
    return hashlib.sha256(data).hexdigest() == piece.sha256
