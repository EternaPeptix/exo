"""Client side of peer weight seeding.

Fetches model files from a peer node's seed server (see
:mod:`exo.download.seed_server`) instead of HuggingFace. Files are written
with the same ``.partial`` + rename convention as the HF downloader, so
partial results resume cleanly and interrupted transfers never look complete.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Callable

import aiofiles
import aiofiles.os as aios
import aiohttp
from loguru import logger

from exo.download.seed_server import API_PREFIX


class PeerDownloadError(Exception):
    """Raised when a file cannot be fetched from a peer (caller should fall back)."""


async def fetch_peer_file_list(
    base_url: str, normalized_id: str, timeout_s: float = 5.0
) -> dict[str, int] | None:
    """Return {relative_path: size} for files the peer holds, or None if the
    peer is unreachable / does not have the model."""
    url = f"{base_url}{API_PREFIX}/{normalized_id}/files"
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(url) as resp,
        ):
            if resp.status != 200:
                return None
            data = await resp.json()
            return {entry["path"]: int(entry["size"]) for entry in data["files"]}
    except Exception as e:
        logger.debug(f"Peer {base_url} file list unavailable: {e}")
        return None


async def check_peer_health(base_url: str, timeout_s: float = 3.0) -> bool:
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(f"{base_url}/v1/health") as resp,
        ):
            return resp.status == 200
    except Exception:
        return False


async def _validate_safetensors_header(path: Path) -> None:
    """Cheap integrity check: the 8-byte length prefix + JSON header must parse."""
    async with aiofiles.open(path, "rb") as f:
        prefix = await f.read(8)
        if len(prefix) != 8:
            raise PeerDownloadError(f"{path.name}: too short to be safetensors")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > 256 * 1024 * 1024:
            raise PeerDownloadError(
                f"{path.name}: implausible safetensors header length {header_len}"
            )
        header = await f.read(header_len)
        if len(header) != header_len:
            raise PeerDownloadError(f"{path.name}: truncated safetensors header")
        try:
            parsed = json.loads(header)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise PeerDownloadError(
                f"{path.name}: corrupt safetensors header: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise PeerDownloadError(f"{path.name}: invalid safetensors header")


async def download_file_from_peer(
    base_url: str,
    normalized_id: str,
    path: str,
    target_dir: Path,
    expected_size: int | None,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    timeout_s: float = 1800.0,
) -> Path:
    """Download ``path`` from a peer into ``target_dir`` (with .partial + rename).

    Raises PeerDownloadError on any failure; the caller should try the next
    source. ``expected_size`` (from the repo file list) is verified when known.
    """
    url = f"{base_url}{API_PREFIX}/{normalized_id}/file/{path}"
    target_path = target_dir / path
    partial_path = target_dir / f"{path}.partial"

    resume_byte_pos = (
        (await aios.stat(partial_path)).st_size
        if await aios.path.exists(partial_path)
        else 0
    )
    if expected_size is None or resume_byte_pos != expected_size:
        headers = {}
        if resume_byte_pos:
            headers["Range"] = f"bytes={resume_byte_pos}-"
        timeout = aiohttp.ClientTimeout(
            total=timeout_s, sock_read=120.0, sock_connect=10.0
        )
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(url, headers=headers) as resp,
            ):
                if resp.status == 404:
                    raise PeerDownloadError(f"peer {base_url} lacks {path}")
                if resp.status not in (200, 206):
                    raise PeerDownloadError(
                        f"peer {base_url} returned {resp.status} for {path}"
                    )
                if resume_byte_pos and resp.status == 200:
                    # Peer ignored the Range header; restart from scratch.
                    resume_byte_pos = 0
                await aios.makedirs(target_path.parent, exist_ok=True)
                n_read = resume_byte_pos
                total = expected_size or resp.content_length or 0
                if resp.status == 206 and resp.content_length:
                    total = resume_byte_pos + resp.content_length
                async with aiofiles.open(
                    partial_path, "ab" if resume_byte_pos else "wb"
                ) as f:
                    async for chunk in resp.content.iter_chunked(8 * 1024 * 1024):
                            await f.write(chunk)
                            n_read += len(chunk)
                            on_progress(n_read, total, False)
        except aiohttp.ClientError as e:
            raise PeerDownloadError(f"peer {base_url} transfer failed: {e}") from e

    final_size = (await aios.stat(partial_path)).st_size
    if expected_size is not None and final_size != expected_size:
        raise PeerDownloadError(
            f"{path}: size mismatch after peer download "
            f"(got {final_size}, expected {expected_size})"
        )
    if path.endswith(".safetensors"):
        await _validate_safetensors_header(partial_path)
    await aios.rename(partial_path, target_path)
    on_progress(final_size, final_size, True)
    return target_path
