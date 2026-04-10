"""Swarm P2P transfer: seeder HTTP server + async puller engine."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import signal
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .manifest import PieceBitfield, PieceInfo, SwarmManifest, verify_piece

logger = logging.getLogger("tightwad.swarm")

SWARM_DIR = Path.home() / ".tightwad"


class TokenAuthMiddleware:
    """Starlette middleware that requires a Bearer token on all requests."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth, f"Bearer {self.token}"):
            response = Response(status_code=401, content="Unauthorized")
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class IPFilterMiddleware:
    """Starlette middleware that restricts access to allowed IP networks."""

    def __init__(self, app: ASGIApp, allowed_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network]) -> None:
        self.app = app
        self.allowed_networks = allowed_networks

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        if client:
            try:
                client_ip = ipaddress.ip_address(client[0])
            except ValueError:
                # Non-IP client (e.g. test client) — deny by default
                response = Response(status_code=403, content="Forbidden")
                await response(scope, receive, send)
                return
            if not any(client_ip in net for net in self.allowed_networks):
                response = Response(status_code=403, content="Forbidden")
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


def _pidfile_for(model: str) -> Path:
    safe = model.replace("/", "_").replace(" ", "_")
    return SWARM_DIR / f"swarm-{safe}.pid"


# --- Seeder ---

_seeder_manifest: SwarmManifest | None = None
_seeder_bitfield: PieceBitfield | None = None
_seeder_model_path: Path | None = None
_seeder_start_time: float = 0.0


async def handle_manifest(request: Request) -> JSONResponse:
    if _seeder_manifest is None:
        return Response(status_code=503, content="Seeder not initialized")
    return JSONResponse(_seeder_manifest.to_dict())


async def handle_bitfield(request: Request) -> JSONResponse:
    if _seeder_bitfield is None:
        return Response(status_code=503, content="Seeder not initialized")
    return JSONResponse({"have": sorted(_seeder_bitfield.have)})


async def handle_piece(request: Request) -> Response:
    if _seeder_manifest is None or _seeder_bitfield is None or _seeder_model_path is None:
        return Response(status_code=503, content="Seeder not initialized")

    index = int(request.path_params["index"])

    if index < 0 or index >= _seeder_manifest.num_pieces:
        return Response(status_code=404, content="piece index out of range")

    if index not in _seeder_bitfield.have:
        return Response(status_code=404, content="piece not available")

    piece = _seeder_manifest.pieces[index]
    with open(_seeder_model_path, "rb") as f:
        f.seek(piece.offset)
        data = f.read(piece.size)

    return Response(content=data, media_type="application/octet-stream")


async def handle_health(request: Request) -> JSONResponse:
    if _seeder_manifest is None or _seeder_bitfield is None:
        return Response(status_code=503, content="Seeder not initialized")
    return JSONResponse({
        "status": "ok",
        "model": _seeder_manifest.model,
        "filename": _seeder_manifest.filename,
        "total_pieces": _seeder_manifest.num_pieces,
        "have_pieces": len(_seeder_bitfield.have),
        "completion_pct": round(_seeder_bitfield.completion_pct(), 1),
        "uptime_seconds": round(time.monotonic() - _seeder_start_time, 1),
    })


def reset_seeder_state() -> None:
    """Reset all module-level seeder globals to their uninitialized values.

    Call this between tests or before re-initializing the seeder to prevent
    stale state from a previous ``create_seeder_app()`` call leaking into
    subsequent requests.
    """
    global _seeder_manifest, _seeder_bitfield, _seeder_model_path, _seeder_start_time
    _seeder_manifest = None
    _seeder_bitfield = None
    _seeder_model_path = None
    _seeder_start_time = 0.0


def create_seeder_app(
    model_path: Path,
    manifest: SwarmManifest,
    bitfield: PieceBitfield,
    token: str | None = None,
    allowed_ips: list[str] | None = None,
) -> Starlette:
    """Create and configure the seeder ASGI application.

    .. note::
        This function populates module-level globals (``_seeder_manifest``,
        ``_seeder_bitfield``, etc.) that are shared by all request handlers.
        Only one seeder instance per process is supported.  If you need to
        reset state between test runs, call :func:`reset_seeder_state` first.
    """
    global _seeder_manifest, _seeder_bitfield, _seeder_model_path, _seeder_start_time
    _seeder_manifest = manifest
    _seeder_bitfield = bitfield
    _seeder_model_path = model_path
    _seeder_start_time = time.monotonic()

    app = Starlette(routes=[
        Route("/manifest", handle_manifest, methods=["GET"]),
        Route("/bitfield", handle_bitfield, methods=["GET"]),
        Route("/pieces/{index:int}", handle_piece, methods=["GET"]),
        Route("/health", handle_health, methods=["GET"]),
    ])

    # Wrap with auth middleware (token checked before IP filter)
    if allowed_ips:
        networks = [ipaddress.ip_network(ip, strict=False) for ip in allowed_ips]
        app = IPFilterMiddleware(app, networks)

    if token:
        app = TokenAuthMiddleware(app, token)

    return app


def write_seeder_pidfile(model: str) -> None:
    SWARM_DIR.mkdir(parents=True, exist_ok=True)
    _pidfile_for(model).write_text(str(os.getpid()))


def remove_seeder_pidfile(model: str) -> None:
    _pidfile_for(model).unlink(missing_ok=True)


def read_seeder_pidfile(model: str) -> int | None:
    pf = _pidfile_for(model)
    if pf.exists():
        try:
            return int(pf.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


# --- Puller ---

@dataclass
class PeerState:
    url: str
    have: set[int] = field(default_factory=set)
    active: int = 0  # current in-flight downloads from this peer


class SwarmPuller:
    def __init__(
        self,
        model_path: Path,
        manifest: SwarmManifest,
        bitfield: PieceBitfield,
        peers: list[str],
        max_concurrent: int = 4,
        token: str | None = None,
    ):
        self.model_path = model_path
        self.manifest = manifest
        self.bitfield = bitfield
        self.peers = [PeerState(url=url.rstrip("/")) for url in peers]
        self.max_concurrent = max_concurrent
        self.token = token
        self._file_preallocated = False

    @property
    def _auth_headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    async def discover_peer_bitfields(self) -> None:
        async with httpx.AsyncClient(timeout=10.0, headers=self._auth_headers) as client:
            tasks = [self._fetch_bitfield(client, peer) for peer in self.peers]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_bitfield(self, client, peer: PeerState) -> None:
        try:
            resp = await client.get(f"{peer.url}/bitfield")
            resp.raise_for_status()
            data = resp.json()
            peer.have = set(data.get("have", []))
            logger.info("Peer %s has %d pieces", peer.url, len(peer.have))
        except Exception as e:
            logger.warning("Failed to get bitfield from %s: %s", peer.url, e)

    def _select_piece_order(
        self,
        peers: list[PeerState],
        missing: list[int],
    ) -> list[tuple[int, PeerState]]:
        """Rarest-first piece selection with load balancing across peers."""
        if not missing or not peers:
            return []

        # Count availability per piece
        availability: Counter[int] = Counter()
        for piece_idx in missing:
            for peer in peers:
                if piece_idx in peer.have:
                    availability[piece_idx] += 1

        # Sort by rarity (fewest sources first)
        sorted_pieces = sorted(missing, key=lambda idx: (availability.get(idx, 0), idx))

        result: list[tuple[int, PeerState]] = []
        for piece_idx in sorted_pieces:
            # Find peers that have this piece, prefer least loaded
            candidates = [p for p in peers if piece_idx in p.have]
            if not candidates:
                logger.warning("Piece %d not available from any peer", piece_idx)
                continue
            # Pick peer with fewest active downloads
            best = min(candidates, key=lambda p: p.active)
            result.append((piece_idx, best))

        return result

    async def download_piece(
        self,
        piece: PieceInfo,
        peer: PeerState,
        sem: asyncio.Semaphore,
    ) -> bool:
        async with sem:
            peer.active += 1
            try:
                async with httpx.AsyncClient(timeout=120.0, headers=self._auth_headers) as client:
                    resp = await client.get(f"{peer.url}/pieces/{piece.index}")
                    resp.raise_for_status()
                    data = resp.content

                # Verify SHA256
                actual_hash = hashlib.sha256(data).hexdigest()
                if actual_hash != piece.sha256:
                    logger.error(
                        "Piece %d SHA256 mismatch from %s: expected %s, got %s",
                        piece.index, peer.url, piece.sha256, actual_hash,
                    )
                    return False

                # Write to correct offset
                self._write_piece(piece, data)
                self.bitfield.mark_have(piece.index)
                self.bitfield.save()
                return True
            except Exception as e:
                logger.error("Failed to download piece %d from %s: %s", piece.index, peer.url, e)
                return False
            finally:
                peer.active -= 1

    def _preallocate(self) -> None:
        """Pre-allocate sparse file on first run."""
        if self._file_preallocated:
            return
        if not self.model_path.exists():
            with open(self.model_path, "wb") as f:
                f.seek(self.manifest.total_size - 1)
                f.write(b"\0")
        self._file_preallocated = True

    def _write_piece(self, piece: PieceInfo, data: bytes) -> None:
        with open(self.model_path, "r+b") as f:
            f.seek(piece.offset)
            f.write(data)

    async def run(
        self,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> bool:
        """Full pull loop: discover, select, download all missing pieces.

        progress_callback(completed, total, piece_index) called after each piece.
        Returns True if all pieces downloaded successfully.
        """
        self._preallocate()
        await self.discover_peer_bitfields()

        missing = self.bitfield.missing_pieces()
        if not missing:
            logger.info("Already have all pieces")
            return True

        total = len(missing)
        plan = self._select_piece_order(self.peers, missing)

        if not plan:
            logger.error("No peers have any of the %d missing pieces", total)
            return False

        sem = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        failed = 0

        async def _do_piece(piece_idx: int, peer: PeerState):
            nonlocal completed, failed
            piece = self.manifest.pieces[piece_idx]
            ok = await self.download_piece(piece, peer, sem)
            if ok:
                completed += 1
            else:
                failed += 1
            if progress_callback:
                progress_callback(completed, total, piece_idx)

        tasks = [_do_piece(idx, peer) for idx, peer in plan]
        await asyncio.gather(*tasks)

        if failed > 0:
            logger.warning("%d pieces failed to download", failed)
            return False

        return self.bitfield.have_all()


# --- Sync wrappers for CLI ---

def run_seeder(
    model_path: Path,
    manifest: SwarmManifest,
    bitfield: PieceBitfield,
    host: str = "0.0.0.0",
    port: int = 9080,
    token: str | None = None,
    allowed_ips: list[str] | None = None,
) -> None:
    """Start seeder HTTP server (blocking)."""
    import uvicorn
    app = create_seeder_app(model_path, manifest, bitfield, token=token, allowed_ips=allowed_ips)
    write_seeder_pidfile(manifest.model)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        remove_seeder_pidfile(manifest.model)


def run_puller(
    model_path: Path,
    manifest: SwarmManifest,
    bitfield: PieceBitfield,
    peers: list[str],
    max_concurrent: int = 4,
    progress_callback: Callable[[int, int, int], None] | None = None,
    token: str | None = None,
) -> bool:
    """Run the puller (blocking). Returns True if successful."""
    puller = SwarmPuller(model_path, manifest, bitfield, peers, max_concurrent, token=token)
    return asyncio.run(puller.run(progress_callback))
