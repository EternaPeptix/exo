"""HTTP server that serves this node's local model files to peers.

Every exo node runs a seed server so that nodes which already hold (part of)
a model can seed weight files to peers over the LAN. This avoids one
HuggingFace download per node and lets nodes with small disks store only the
shard of the model they actually run.

The trust model matches the rest of exo: the cluster LAN is trusted, no auth.
Only files under known model directories are served, and paths are looked up
in a scanned index rather than resolved from user input, so path traversal is
not possible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from aiohttp import web
from loguru import logger

from exo.shared.constants import EXO_MODELS_DIRS, EXO_MODELS_READ_ONLY_DIRS

# Marker written next to model files recording which shard layers were fetched.
# Never served to peers (hidden file) and excluded from download patterns.
SHARD_MARKER_FILENAME = ".exo_shard.json"

API_PREFIX = "/v1/models"


def _model_dirs(normalized_id: str) -> list[Path]:
    """All local directories that may hold files for the given model."""
    dirs = []
    for root in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS):
        candidate = root / normalized_id
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def _visible_file(rel: Path) -> bool:
    # Skip hidden files/dirs (.exo_shard.json, .cache HF metadata, .DS_Store...).
    return not any(part.startswith(".") for part in rel.parts)


def _scan_files(normalized_id: str) -> dict[str, tuple[int, Path]]:
    """Map relative path -> (size, absolute path) for all complete local files."""
    files: dict[str, tuple[int, Path]] = {}
    for model_dir in _model_dirs(normalized_id):
        for dirpath, _, filenames in os.walk(model_dir):
            for name in filenames:
                if name.endswith(".partial"):
                    continue
                full = Path(dirpath) / name
                rel = full.relative_to(model_dir)
                if not _visible_file(rel):
                    continue
                rel_str = str(rel)
                # Read-only dirs win; first occurrence is kept.
                if rel_str not in files:
                    try:
                        files[rel_str] = (full.stat().st_size, full)
                    except OSError:
                        continue
    return files


@dataclass
class SeedServer:
    """Serves local model weight files to peers over HTTP."""

    port: int
    _runner: web.AppRunner | None = field(init=False, default=None)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _handle_files(self, request: web.Request) -> web.Response:
        normalized_id = request.match_info["normalized_id"]
        files = await anyio.to_thread.run_sync(_scan_files, normalized_id)
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
        normalized_id = request.match_info["normalized_id"]
        rel_path = request.match_info["path"]
        files = await anyio.to_thread.run_sync(_scan_files, normalized_id)
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
        app.router.add_get(
            f"{API_PREFIX}/{{normalized_id}}/files", self._handle_files
        )
        app.router.add_get(
            f"{API_PREFIX}/{{normalized_id}}/file/{{path:.*}}", self._handle_file
        )
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        logger.info(f"Seed server listening on 0.0.0.0:{self.port}")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
