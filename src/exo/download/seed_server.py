"""HTTP server that serves this node's local model files to peers.

When explicitly enabled, an exo node can seed weight files to peers over a
trusted cluster network. This avoids one HuggingFace download per node and
lets nodes with small disks store only the shard of the model they actually
run.

The protocol is intentionally unauthenticated, so it is disabled by default
and binds to loopback unless the operator selects a cluster interface. Only
regular, non-symlink files under a configured model directory are served.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final

from aiohttp import web
from anyio import to_thread
from loguru import logger

from exo.shared.constants import EXO_MODELS_DIRS, EXO_MODELS_READ_ONLY_DIRS

# Marker written next to model files recording which shard layers were fetched.
# Never served to peers (hidden file) and excluded from download patterns.
SHARD_MARKER_FILENAME = ".exo_shard.json"

API_PREFIX = "/v1/models"
MAX_PEER_FILE_BYTES: Final = 1 << 40
MAX_PEER_FILE_COUNT: Final = 100_000
MAX_PEER_FILE_LIST_BYTES: Final = 16 << 20
MAX_PEER_PATH_BYTES: Final = 4096
_NORMALIZED_MODEL_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")


class PeerProtocolError(ValueError):
    """An unsafe or malformed value at the peer-seeding trust boundary."""


def validate_normalized_model_id(value: object) -> str:
    """Validate one path-segment model identifier used by the seed protocol."""

    if not isinstance(value, str) or not _NORMALIZED_MODEL_ID_RE.fullmatch(value):
        raise PeerProtocolError("invalid normalized model identifier")
    if value in {".", ".."}:
        raise PeerProtocolError("invalid normalized model identifier")
    return value


def validate_relative_file_path(value: object) -> str:
    """Return a canonical, traversal-free POSIX model-file path."""

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or "\\" in value
        or len(value.encode("utf-8")) > MAX_PEER_PATH_BYTES
    ):
        raise PeerProtocolError("invalid peer file path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise PeerProtocolError("peer file path must be canonical and relative")
    return value


def validate_peer_file_size(value: object) -> int:
    """Validate a peer-advertised file size before disk accounting or I/O."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_PEER_FILE_BYTES
    ):
        raise PeerProtocolError(
            f"peer file size must be between 0 and {MAX_PEER_FILE_BYTES} bytes"
        )
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _model_dirs(normalized_id: str) -> list[Path]:
    """All local directories that may hold files for the given model."""

    normalized_id = validate_normalized_model_id(normalized_id)
    dirs: list[Path] = []
    for root in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS):
        try:
            resolved_root = root.expanduser().resolve(strict=True)
            candidate = resolved_root / normalized_id
            if candidate.is_symlink():
                continue
            resolved_candidate = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved_candidate.is_dir() and _is_relative_to(
            resolved_candidate, resolved_root
        ):
            dirs.append(resolved_candidate)
    return dirs


def _visible_file(rel: Path) -> bool:
    # Skip hidden files/dirs (.exo_shard.json, .cache HF metadata, .DS_Store...).
    return not any(part.startswith(".") for part in rel.parts)


def _scan_files(normalized_id: str) -> dict[str, tuple[int, Path]]:
    """Map relative path -> (size, absolute path) for all complete local files."""

    normalized_id = validate_normalized_model_id(normalized_id)
    files: dict[str, tuple[int, Path]] = {}
    for model_dir in _model_dirs(normalized_id):
        for dirpath, dirnames, filenames in os.walk(model_dir, followlinks=False):
            current_dir = Path(dirpath)
            dirnames[:] = [
                name
                for name in dirnames
                if not name.startswith(".") and not (current_dir / name).is_symlink()
            ]
            for name in filenames:
                if name.endswith(".partial"):
                    continue
                full = current_dir / name
                if full.is_symlink():
                    continue
                rel = full.relative_to(model_dir)
                if not _visible_file(rel):
                    continue
                try:
                    rel_str = validate_relative_file_path(rel.as_posix())
                    resolved = full.resolve(strict=True)
                    if not resolved.is_file() or not _is_relative_to(
                        resolved, model_dir
                    ):
                        continue
                    size = validate_peer_file_size(resolved.stat().st_size)
                except (OSError, PeerProtocolError):
                    continue
                # Read-only dirs win; first occurrence is kept.
                if rel_str not in files:
                    files[rel_str] = (size, resolved)
    return files


@dataclass
class SeedServer:
    """Serves local model weight files to peers over HTTP."""

    port: int
    host: str = "127.0.0.1"
    _runner: web.AppRunner | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("seed-server port must be between 1 and 65535")
        if (
            not self.host
            or "://" in self.host
            or "/" in self.host
            or "\\" in self.host
            or "\x00" in self.host
        ):
            raise ValueError("seed-server host must be a hostname or IP address")

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _handle_files(self, request: web.Request) -> web.Response:
        try:
            normalized_id = validate_normalized_model_id(
                request.match_info["normalized_id"]
            )
        except PeerProtocolError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        files = await to_thread.run_sync(_scan_files, normalized_id)
        if not files:
            return web.json_response({"error": "model not found"}, status=404)
        return web.json_response(
            {
                "files": [
                    {"path": path, "size": size} for path, (size, _) in files.items()
                ]
            }
        )

    async def _handle_file(self, request: web.Request) -> web.StreamResponse:
        try:
            normalized_id = validate_normalized_model_id(
                request.match_info["normalized_id"]
            )
            rel_path = validate_relative_file_path(request.match_info["path"])
        except PeerProtocolError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        files = await to_thread.run_sync(_scan_files, normalized_id)
        entry = files.get(rel_path)
        if entry is None:
            return web.json_response({"error": "file not found"}, status=404)
        _, full_path = entry
        # FileResponse supports Range requests out of the box, enabling
        # resumable peer downloads.
        return web.FileResponse(full_path)

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/v1/health", self._handle_health)
        app.router.add_get(f"{API_PREFIX}/{{normalized_id}}/files", self._handle_files)
        app.router.add_get(
            f"{API_PREFIX}/{{normalized_id}}/file/{{path:.*}}", self._handle_file
        )
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        if self.host in {"0.0.0.0", "::"}:
            logger.warning(
                "Unauthenticated seed server is listening on every interface; "
                "use a dedicated trusted-cluster address when possible"
            )
        logger.info(f"Seed server listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
