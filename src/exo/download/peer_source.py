"""Client side of peer weight seeding.

Fetches model files from a peer node's seed server (see
:mod:`exo.download.seed_server`) instead of HuggingFace. Files are written
with the same ``.partial`` + rename convention as the HF downloader, so
partial results resume cleanly and interrupted transfers never look complete.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Callable, cast
from urllib.parse import quote, urlsplit, urlunsplit

import aiofiles
import aiofiles.os as aios
import aiohttp
from loguru import logger

from exo.download.seed_server import (
    API_PREFIX,
    MAX_PEER_FILE_BYTES,
    MAX_PEER_FILE_COUNT,
    MAX_PEER_FILE_LIST_BYTES,
    PeerProtocolError,
    validate_normalized_model_id,
    validate_peer_file_size,
    validate_relative_file_path,
)


class PeerDownloadError(Exception):
    """Raised when a file cannot be fetched from a peer (caller should fall back)."""


def _canonical_peer_base_url(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise PeerDownloadError("invalid peer seed URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is None
        or not 1 <= port <= 65535
    ):
        raise PeerDownloadError("invalid peer seed URL")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _peer_model_url(base_url: str, normalized_id: str, suffix: str) -> str:
    base = _canonical_peer_base_url(base_url)
    model = quote(validate_normalized_model_id(normalized_id), safe="")
    return f"{base}{API_PREFIX}/{model}/{suffix}"


def _peer_file_url(base_url: str, normalized_id: str, path: str) -> str:
    canonical_path = validate_relative_file_path(path)
    encoded_path = "/".join(
        quote(part, safe="") for part in PurePosixPath(canonical_path).parts
    )
    return _peer_model_url(base_url, normalized_id, f"file/{encoded_path}")


def _parse_peer_file_list(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise PeerDownloadError("peer file list has an invalid schema")
    mapping = cast(dict[object, object], value)
    if set(mapping) != {"files"}:
        raise PeerDownloadError("peer file list has an invalid schema")
    entries_value = mapping["files"]
    if not isinstance(entries_value, list):
        raise PeerDownloadError("peer file list has an invalid file count")
    entries = cast(list[object], entries_value)
    if len(entries) > MAX_PEER_FILE_COUNT:
        raise PeerDownloadError("peer file list has an invalid file count")
    result: dict[str, int] = {}
    try:
        for entry in entries:
            if not isinstance(entry, dict):
                raise PeerProtocolError("invalid peer file entry")
            entry_mapping = cast(dict[object, object], entry)
            if set(entry_mapping) != {"path", "size"}:
                raise PeerProtocolError("invalid peer file entry")
            path = validate_relative_file_path(entry_mapping["path"])
            size = validate_peer_file_size(entry_mapping["size"])
            if path in result:
                raise PeerProtocolError("duplicate peer file path")
            result[path] = size
    except PeerProtocolError as error:
        raise PeerDownloadError(str(error)) from error
    return result


def _safe_target_paths(target_dir: Path, path: str) -> tuple[Path, Path, Path]:
    canonical_path = validate_relative_file_path(path)
    root = target_dir.expanduser().resolve()
    target = root.joinpath(*PurePosixPath(canonical_path).parts)
    partial = target.with_name(f"{target.name}.partial")
    try:
        target.parent.resolve().relative_to(root)
    except ValueError as error:
        raise PeerDownloadError(
            "peer target path escapes the model directory"
        ) from error
    for candidate in (target, partial):
        if candidate.is_symlink():
            raise PeerDownloadError(f"refusing peer-download symlink: {candidate}")
        if candidate.exists() and not candidate.is_file():
            raise PeerDownloadError(
                f"peer-download target is not a regular file: {candidate}"
            )
    return root, target, partial


def _validate_content_range(response: aiohttp.ClientResponse, offset: int) -> None:
    if response.status != 206:
        return
    value = response.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)", value)
    if match is None or int(match.group(1)) != offset:
        raise PeerDownloadError("peer returned an invalid Content-Range")


async def fetch_peer_file_list(
    base_url: str, normalized_id: str, timeout_s: float = 5.0
) -> dict[str, int] | None:
    """Return {relative_path: size} for files the peer holds, or None if the
    peer is unreachable / does not have the model."""
    try:
        url = _peer_model_url(base_url, normalized_id, "files")
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(url) as resp,
        ):
            if resp.status != 200:
                return None
            payload = await resp.content.read(MAX_PEER_FILE_LIST_BYTES + 1)
            if len(payload) > MAX_PEER_FILE_LIST_BYTES:
                raise PeerDownloadError("peer file list exceeds the response limit")
            try:
                data = cast(object, json.loads(payload))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PeerDownloadError("peer file list is not valid JSON") from error
            return _parse_peer_file_list(data)
    except Exception as e:
        logger.debug(f"Peer {base_url} file list unavailable: {e}")
        return None


async def check_peer_health(base_url: str, timeout_s: float = 3.0) -> bool:
    try:
        base_url = _canonical_peer_base_url(base_url)
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
        header_len = int.from_bytes(prefix, byteorder="little", signed=False)
        if header_len > 256 * 1024 * 1024:
            raise PeerDownloadError(
                f"{path.name}: implausible safetensors header length {header_len}"
            )
        header = await f.read(header_len)
        if len(header) != header_len:
            raise PeerDownloadError(f"{path.name}: truncated safetensors header")
        try:
            parsed = cast(object, json.loads(header))
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
    try:
        url = _peer_file_url(base_url, normalized_id, path)
        if expected_size is not None:
            expected_size = validate_peer_file_size(expected_size)
        root, target_path, partial_path = _safe_target_paths(target_dir, path)
    except PeerProtocolError as error:
        raise PeerDownloadError(str(error)) from error

    partial_exists = await aios.path.exists(partial_path)
    resume_byte_pos = (await aios.stat(partial_path)).st_size if partial_exists else 0
    byte_limit = expected_size if expected_size is not None else MAX_PEER_FILE_BYTES
    if not 0 <= resume_byte_pos <= byte_limit:
        raise PeerDownloadError(f"{path}: partial file exceeds the trusted size limit")
    if not partial_exists or expected_size is None or resume_byte_pos != expected_size:
        headers: dict[str, str] = {}
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
                _validate_content_range(resp, resume_byte_pos)
                if resume_byte_pos and resp.status == 200:
                    # Peer ignored the Range header; restart from scratch.
                    resume_byte_pos = 0
                if (
                    resp.content_length is not None
                    and resp.content_length > byte_limit - resume_byte_pos
                ):
                    raise PeerDownloadError(
                        f"peer {base_url} response exceeds the trusted size for {path}"
                    )
                await aios.makedirs(target_path.parent, exist_ok=True)
                # Re-resolve after directory creation so a pre-existing parent
                # symlink cannot redirect the write outside the model root.
                checked_root, checked_target, checked_partial = _safe_target_paths(
                    root, path
                )
                if (
                    checked_root != root
                    or checked_target != target_path
                    or checked_partial != partial_path
                ):
                    raise PeerDownloadError("peer target path changed before write")
                n_read = resume_byte_pos
                total = expected_size or resp.content_length or 0
                if resp.status == 206 and resp.content_length:
                    total = resume_byte_pos + resp.content_length
                async with aiofiles.open(
                    partial_path, "ab" if resume_byte_pos else "wb"
                ) as f:
                    async for chunk in resp.content.iter_chunked(8 * 1024 * 1024):
                        if len(chunk) > byte_limit - n_read:
                            raise PeerDownloadError(
                                f"peer {base_url} streamed too many bytes for {path}"
                            )
                        await f.write(chunk)
                        n_read += len(chunk)
                        on_progress(n_read, total, False)
        except aiohttp.ClientError as e:
            raise PeerDownloadError(f"peer {base_url} transfer failed: {e}") from e
        except OSError as e:
            raise PeerDownloadError(f"cannot stage peer file {path}: {e}") from e

    final_size = (await aios.stat(partial_path)).st_size
    validate_peer_file_size(final_size)
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
