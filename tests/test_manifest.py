"""Tests for manifest.py: SwarmManifest generation, serialization, and piece verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tightwad.manifest import (
    PieceBitfield,
    PieceInfo,
    SwarmManifest,
    create_manifest,
    verify_piece,
    DEFAULT_PIECE_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(path: Path, size: int, pattern: bytes = b"\xAB") -> None:
    """Write a file of the given size filled with a repeating pattern."""
    with open(path, "wb") as f:
        written = 0
        chunk_size = min(size, 4096)
        chunk = (pattern * chunk_size)[:chunk_size]
        while written < size:
            to_write = min(chunk_size, size - written)
            f.write(chunk[:to_write])
            written += to_write


# ---------------------------------------------------------------------------
# create_manifest
# ---------------------------------------------------------------------------


class TestCreateManifest:

    def test_create_manifest(self, tmp_path):
        """Create a manifest for a known file, verify pieces have correct sizes and SHA256."""
        piece_size = 64  # small for testing
        file_size = 200  # should produce ceil(200/64) = 4 pieces
        file_path = tmp_path / "model.gguf"
        _write_file(file_path, file_size)

        manifest = create_manifest(file_path, piece_size=piece_size, use_gguf_inspect=False)

        assert manifest.model == "model"
        assert manifest.filename == "model.gguf"
        assert manifest.total_size == file_size
        assert manifest.piece_size == piece_size
        assert len(manifest.pieces) == 4  # 64+64+64+8

        # Verify piece sizes
        assert manifest.pieces[0].size == 64
        assert manifest.pieces[1].size == 64
        assert manifest.pieces[2].size == 64
        assert manifest.pieces[3].size == 8  # remaining

        # Verify offsets
        assert manifest.pieces[0].offset == 0
        assert manifest.pieces[1].offset == 64
        assert manifest.pieces[2].offset == 128
        assert manifest.pieces[3].offset == 192

        # Verify SHA256 hashes are correct
        with open(file_path, "rb") as f:
            for piece in manifest.pieces:
                f.seek(piece.offset)
                data = f.read(piece.size)
                expected_hash = hashlib.sha256(data).hexdigest()
                assert piece.sha256 == expected_hash

        # Verify indices
        for i, piece in enumerate(manifest.pieces):
            assert piece.index == i


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestManifestJsonRoundtrip:

    def test_manifest_json_roundtrip(self, tmp_path):
        """Serialize manifest to JSON, deserialize, verify equality."""
        file_path = tmp_path / "test_model.bin"
        _write_file(file_path, 150)

        original = create_manifest(file_path, piece_size=50, use_gguf_inspect=False)

        # Serialize to dict and JSON
        d = original.to_dict()
        json_str = json.dumps(d)

        # Deserialize
        restored = SwarmManifest.from_dict(json.loads(json_str))

        assert restored.model == original.model
        assert restored.filename == original.filename
        assert restored.total_size == original.total_size
        assert restored.piece_size == original.piece_size
        assert restored.num_pieces == original.num_pieces

        for orig_piece, rest_piece in zip(original.pieces, restored.pieces):
            assert rest_piece.index == orig_piece.index
            assert rest_piece.offset == orig_piece.offset
            assert rest_piece.size == orig_piece.size
            assert rest_piece.sha256 == orig_piece.sha256

    def test_manifest_save_and_load(self, tmp_path):
        """Save manifest to file, load it back, verify fields match."""
        file_path = tmp_path / "model.bin"
        _write_file(file_path, 100)
        manifest_path = tmp_path / "model.manifest"

        original = create_manifest(file_path, piece_size=40, use_gguf_inspect=False)
        original.save(manifest_path)

        loaded = SwarmManifest.load(manifest_path)

        assert loaded.model == original.model
        assert loaded.total_size == original.total_size
        assert loaded.num_pieces == original.num_pieces
        for i in range(loaded.num_pieces):
            assert loaded.pieces[i].sha256 == original.pieces[i].sha256


# ---------------------------------------------------------------------------
# verify_piece
# ---------------------------------------------------------------------------


class TestVerifyPiece:

    def test_verify_piece_valid(self, tmp_path):
        """Each piece in a freshly created manifest should pass verification."""
        file_path = tmp_path / "valid.bin"
        _write_file(file_path, 256)

        manifest = create_manifest(file_path, piece_size=100, use_gguf_inspect=False)

        for piece in manifest.pieces:
            assert verify_piece(file_path, piece) is True

    def test_verify_piece_corrupted(self, tmp_path):
        """Corrupting a byte in a piece should cause verification to fail."""
        file_path = tmp_path / "corrupt.bin"
        _write_file(file_path, 256)

        manifest = create_manifest(file_path, piece_size=100, use_gguf_inspect=False)

        # Corrupt a byte in the second piece (offset 100)
        with open(file_path, "r+b") as f:
            f.seek(100)
            original_byte = f.read(1)
            f.seek(100)
            # Write a different byte
            corrupted = bytes([(original_byte[0] + 1) % 256])
            f.write(corrupted)

        # First piece should still verify
        assert verify_piece(file_path, manifest.pieces[0]) is True

        # Second piece should fail
        assert verify_piece(file_path, manifest.pieces[1]) is False

        # Third piece should still verify
        assert verify_piece(file_path, manifest.pieces[2]) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestManifestEdgeCases:

    def test_manifest_empty_file(self, tmp_path):
        """Empty file produces a manifest with zero pieces."""
        file_path = tmp_path / "empty.bin"
        file_path.write_bytes(b"")

        manifest = create_manifest(file_path, piece_size=64, use_gguf_inspect=False)

        assert manifest.total_size == 0
        assert manifest.num_pieces == 0
        assert manifest.pieces == []

    def test_piece_count(self, tmp_path):
        """Verify correct number of pieces for various file sizes and piece sizes."""
        test_cases = [
            (100, 100, 1),   # exactly one piece
            (100, 50, 2),    # exactly two pieces
            (100, 33, 4),    # 33+33+33+1 = 4 pieces
            (1, 100, 1),     # file smaller than piece size
            (64, 64, 1),     # exact fit
            (65, 64, 2),     # just one byte over
        ]

        for file_size, piece_size, expected_pieces in test_cases:
            file_path = tmp_path / f"test_{file_size}_{piece_size}.bin"
            _write_file(file_path, file_size)

            manifest = create_manifest(
                file_path, piece_size=piece_size, use_gguf_inspect=False,
            )

            assert manifest.num_pieces == expected_pieces, (
                f"file_size={file_size}, piece_size={piece_size}: "
                f"expected {expected_pieces} pieces, got {manifest.num_pieces}"
            )

    def test_num_pieces_property(self, tmp_path):
        """The num_pieces property should match len(pieces)."""
        file_path = tmp_path / "prop.bin"
        _write_file(file_path, 200)

        manifest = create_manifest(file_path, piece_size=50, use_gguf_inspect=False)
        assert manifest.num_pieces == len(manifest.pieces) == 4

    def test_find_for_model_not_found(self, tmp_path):
        """find_for_model returns None when no manifest file exists."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"data")

        assert SwarmManifest.find_for_model(model_path) is None

    def test_find_for_model_found(self, tmp_path):
        """find_for_model returns manifest when .tightwad.manifest file exists."""
        model_path = tmp_path / "model.gguf"
        _write_file(model_path, 100)

        manifest = create_manifest(model_path, piece_size=50, use_gguf_inspect=False)
        manifest_path = tmp_path / "model.gguf.tightwad.manifest"
        manifest.save(manifest_path)

        found = SwarmManifest.find_for_model(model_path)
        assert found is not None
        assert found.model == manifest.model
        assert found.num_pieces == manifest.num_pieces

    def test_progress_callback(self, tmp_path):
        """create_manifest calls progress_callback for each piece."""
        file_path = tmp_path / "progress.bin"
        _write_file(file_path, 200)

        calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            calls.append((current, total))

        manifest = create_manifest(
            file_path, piece_size=50, use_gguf_inspect=False,
            progress_callback=callback,
        )

        assert len(calls) == manifest.num_pieces
        # Each call should have incrementing current index
        for i, (current, total) in enumerate(calls):
            assert current == i + 1
            assert total == -1  # total is unknown during streaming


# ---------------------------------------------------------------------------
# PieceBitfield
# ---------------------------------------------------------------------------


class TestPieceBitfield:

    def test_bitfield_mark_and_completion(self, tmp_path):
        """Mark pieces as having and check completion percentage."""
        bf_path = tmp_path / "pieces.json"
        bf = PieceBitfield.load_or_create(bf_path, total_pieces=4)

        assert bf.completion_pct() == 0.0
        assert bf.have_all() is False
        assert bf.missing_pieces() == [0, 1, 2, 3]

        bf.mark_have(0)
        bf.mark_have(2)
        assert bf.completion_pct() == 50.0
        assert bf.missing_pieces() == [1, 3]

        bf.mark_have(1)
        bf.mark_have(3)
        assert bf.completion_pct() == 100.0
        assert bf.have_all() is True
        assert bf.missing_pieces() == []

    def test_bitfield_save_and_load(self, tmp_path):
        """Save bitfield, load it back."""
        bf_path = tmp_path / "pieces.json"
        bf = PieceBitfield.load_or_create(bf_path, total_pieces=4)
        bf.mark_have(0)
        bf.mark_have(3)
        bf.save()

        bf2 = PieceBitfield.load_or_create(bf_path, total_pieces=4)
        assert bf2.have == {0, 3}
        assert bf2.missing_pieces() == [1, 2]

    def test_bitfield_mark_missing(self, tmp_path):
        """mark_missing removes a piece from the have set."""
        bf_path = tmp_path / "pieces.json"
        bf = PieceBitfield.load_or_create(bf_path, total_pieces=3)
        bf.mark_have(0)
        bf.mark_have(1)
        bf.mark_have(2)
        assert bf.have_all() is True

        bf.mark_missing(1)
        assert bf.have_all() is False
        assert bf.missing_pieces() == [1]
