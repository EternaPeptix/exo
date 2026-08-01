import asyncio
import hashlib
import json
import os
import random
import shutil
import ssl
import time
import traceback
from collections.abc import Awaitable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Callable, Literal, cast
from urllib.parse import urljoin

import aiofiles
import aiofiles.os as aios
import aiohttp
import certifi
from huggingface_hub import (
    snapshot_download,  # pyright: ignore[reportUnknownVariableType]
)
from loguru import logger
from pydantic import (
    TypeAdapter,
)

from exo.download.huggingface_utils import (
    filter_repo_objects,
    get_allow_patterns,
    get_auth_headers,
    get_hf_endpoint,
    get_hf_token,
)
from exo.download.seed_server import SHARD_MARKER_FILENAME
from exo.shared.constants import (
    EXO_DEFAULT_MODELS_DIR,
    EXO_MODELS_DIRS,
    EXO_MODELS_READ_ONLY_DIRS,
)
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import SeedSource
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadProgressData,
    FileListEntry,
    ModelSafetensorsIndex,
    RepoDownloadProgress,
    RepoFileDownloadProgress,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.rank_local_checkpoint import (
    PINNED_RANK_LOCAL_METADATA_FILES,
    SUPPORTED_LOADER_SCHEMA,
    SUPPORTED_MLX_LM_COMMIT,
    SUPPORTED_MLX_LM_KIMI_K3_SHA256,
    SUPPORTED_SOURCE_CONFIG_SHA256,
    SUPPORTED_SOURCE_INDEX_SHA256,
    SUPPORTED_SOURCE_REVISION,
    SUPPORTED_TP_CONTRACT,
    SUPPORTED_TP_CONTRACT_DIGEST,
    RankLocalConfigurationError,
    preflight_rank_local_runtime,
    resolve_configured_rank_local_checkpoint_path,
)


class HuggingFaceAuthenticationError(Exception):
    """Raised when HuggingFace returns 401/403 for a model download."""


class HuggingFaceRateLimitError(Exception):
    """429 Huggingface code"""

    def __init__(self, msg: str, retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Parse seconds-to-reset from HF's RateLimit header.

    HF sends e.g. ``ratelimit: "api";r=0;t=52`` on 429s; ``t`` is the wait.
    Returns ``None`` if the header is missing or has no ``t`` field.
    """
    raw = headers.get("RateLimit") or headers.get("ratelimit")
    if raw is None:
        return None
    for part in raw.split(";"):
        key, _, val = part.strip().partition("=")
        if key == "t":
            try:
                return float(val)
            except ValueError:
                return None
    return None


# reset window is 5 min
_RATE_LIMIT_MAX_SLEEP_SECS = 300.0

# 24h. Manually clear the cache (or `delete_model`) to force a refresh.
_FILE_LIST_CACHE_TTL_SECS = 24 * 60 * 60


async def _build_auth_error_message(status_code: int, model_id: ModelId) -> str:
    token = await get_hf_token()
    if status_code == 401 and token is None:
        return (
            f"Model '{model_id}' requires authentication. "
            f"Set HF_TOKEN in the app's Advanced settings, set the HF_TOKEN environment variable, or run `hf auth login`. "
            f"Get a token at https://huggingface.co/settings/tokens"
        )
    elif status_code == 403:
        return (
            f"Access denied to '{model_id}'. "
            f"Please accept the model terms at https://huggingface.co/{model_id}"
        )
    else:
        return f"Authentication failed for '{model_id}' (HTTP {status_code})"


def trim_etag(etag: str) -> str:
    if (etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'"):
        return etag[1:-1]
    return etag


def map_repo_file_download_progress_to_download_progress_data(
    repo_file_download_progress: RepoFileDownloadProgress,
) -> DownloadProgressData:
    return DownloadProgressData(
        downloaded=repo_file_download_progress.downloaded,
        downloaded_this_session=repo_file_download_progress.downloaded_this_session,
        total=repo_file_download_progress.total,
        completed_files=1 if repo_file_download_progress.status == "complete" else 0,
        total_files=1,
        speed=repo_file_download_progress.speed,
        eta_ms=int(repo_file_download_progress.eta.total_seconds() * 1000),
        files={},
    )


def map_repo_download_progress_to_download_progress_data(
    repo_download_progress: RepoDownloadProgress,
) -> DownloadProgressData:
    return DownloadProgressData(
        total=repo_download_progress.total,
        downloaded=repo_download_progress.downloaded,
        downloaded_this_session=repo_download_progress.downloaded_this_session,
        completed_files=repo_download_progress.completed_files,
        total_files=repo_download_progress.total_files,
        speed=repo_download_progress.overall_speed,
        eta_ms=int(repo_download_progress.overall_eta.total_seconds() * 1000),
        files={
            file_path: map_repo_file_download_progress_to_download_progress_data(
                file_progress
            )
            for file_path, file_progress in repo_download_progress.file_progress.items()
        },
    )


class InsufficientDiskSpaceError(Exception):
    """Raised when no writable model directory has enough free space."""


def resolve_existing_model(
    model_id: ModelId, card: ModelCard | None = None
) -> Path | None:
    """Search all model directories for a complete, pre-existing model.

    Checks read-only directories first, then writable directories.
    A candidate is only returned if ``is_model_directory_complete`` confirms
    all weight files are present.
    """
    normalized = model_id.normalize()
    for search_dir in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS):
        candidate = search_dir / normalized
        if candidate.is_dir() and is_model_directory_complete(candidate, card):
            return candidate
    return None


def is_read_only_model_dir(model_dir: Path) -> bool:
    """Check if a model directory lives under a read-only models root."""
    return any(model_dir.is_relative_to(d) for d in EXO_MODELS_READ_ONLY_DIRS)


def build_model_path(model_id: ModelId) -> Path:
    found = resolve_existing_model(model_id)
    if found is not None:
        return found
    return EXO_DEFAULT_MODELS_DIR / model_id.normalize()


def select_download_dir(required_bytes: int) -> Path:
    """Pick the first writable model directory with enough free space.

    Raises ``InsufficientDiskSpaceError`` if none have enough space.
    """
    for candidate_dir in EXO_MODELS_DIRS:
        if not candidate_dir.exists():
            continue
        try:
            usage = shutil.disk_usage(candidate_dir)
            if usage.free >= required_bytes:
                return candidate_dir
        except OSError:
            continue
    raise InsufficientDiskSpaceError(
        f"No writable model directory has {required_bytes / (1024**3):.1f} GiB free. "
        f"Checked: {[str(d) for d in EXO_MODELS_DIRS]}"
    )


async def select_download_dir_for_shard(
    model_id: ModelId,
    filtered_file_list: list[FileListEntry],
    total_size: int,
) -> Path:
    for candidate_dir in EXO_MODELS_DIRS:
        if not candidate_dir.exists():
            continue
        sub = candidate_dir / model_id.normalize()
        if not await aios.path.isdir(sub):
            continue
        existing_bytes = 0
        for file_entry in filtered_file_list:
            existing_bytes += await get_downloaded_size(sub / file_entry.path)
        remaining = max(total_size - existing_bytes, 0)
        try:
            if shutil.disk_usage(candidate_dir).free >= remaining:
                return candidate_dir
        except OSError:
            continue
    return select_download_dir(total_size)


async def resolve_model_dir(model_id: ModelId) -> Path:
    """Return the directory for a model's files, creating it if needed.

    Checks all model directories for an existing complete model first,
    then falls back to the default models directory.
    """
    target = await asyncio.to_thread(build_model_path, model_id)
    await aios.makedirs(target, exist_ok=True)
    return target


async def ensure_cache_dir(model_id: ModelId) -> Path:
    """Return the cache directory for a model's metadata, creating it if needed."""
    target = EXO_DEFAULT_MODELS_DIR / "caches" / model_id.normalize()
    await aios.makedirs(target, exist_ok=True)
    return target


async def delete_model(model_id: ModelId) -> bool:
    """Delete a model from writable directories. Skips read-only dirs."""
    normalized = model_id.normalize()
    deleted = False
    for models_dir in EXO_MODELS_DIRS:
        model_dir = models_dir / normalized
        if await aios.path.exists(model_dir):
            await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
            deleted = True

    # Clear cache from default dir
    cache_dir = EXO_DEFAULT_MODELS_DIR / "caches" / normalized
    if await aios.path.exists(cache_dir):
        await asyncio.to_thread(shutil.rmtree, cache_dir, ignore_errors=False)

    return deleted


async def seed_models(seed_dir: str | Path):
    """Move models from resources folder to the default models directory."""
    source_dir = Path(seed_dir)
    await aios.makedirs(EXO_DEFAULT_MODELS_DIR, exist_ok=True)
    dest_dir = EXO_DEFAULT_MODELS_DIR
    for path in source_dir.iterdir():
        if path.is_dir() and path.name.startswith("models--"):
            dest_path = dest_dir / path.name
            if await aios.path.exists(dest_path):
                logger.info("Skipping moving model to .cache directory")
            else:
                try:
                    await aios.rename(str(path), str(dest_path))
                except Exception:
                    logger.error(f"Error seeding model {path} to {dest_path}")
                    logger.error(traceback.format_exc())


def _scan_model_directory(
    model_dir: Path, recursive: bool = False
) -> list[FileListEntry] | None:
    """Scan a local model directory and build a file list.

    Requires at least one ``*.safetensors.index.json``.  Every weight file
    referenced by the index that is missing on disk gets ``size=None``.
    """
    index_files = list(model_dir.glob("**/*.safetensors.index.json"))
    if not index_files:
        return None

    entries_by_path: dict[str, FileListEntry] = {}

    if recursive:
        for dirpath, _, filenames in os.walk(model_dir):
            for filename in filenames:
                if filename.endswith(".partial"):
                    continue
                full_path = Path(dirpath) / filename
                rel_path = str(full_path.relative_to(model_dir))
                entries_by_path[rel_path] = FileListEntry(
                    type="file",
                    path=rel_path,
                    size=full_path.stat().st_size,
                )
    else:
        for item in model_dir.iterdir():
            if item.is_file() and not item.name.endswith(".partial"):
                entries_by_path[item.name] = FileListEntry(
                    type="file",
                    path=item.name,
                    size=item.stat().st_size,
                )

    # Add expected weight files from index that haven't been downloaded yet
    for index_file in index_files:
        try:
            index_data = ModelSafetensorsIndex.model_validate_json(
                index_file.read_text()
            )
            relative_dir = index_file.parent.relative_to(model_dir)
            for filename in set(index_data.weight_map.values()):
                rel_path = (
                    str(relative_dir / filename)
                    if relative_dir != Path(".")
                    else filename
                )
                if rel_path not in entries_by_path:
                    entries_by_path[rel_path] = FileListEntry(
                        type="file",
                        path=rel_path,
                        size=None,
                    )
        except Exception:
            continue

    return list(entries_by_path.values())


def is_model_directory_complete(model_dir: Path, card: ModelCard | None = None) -> bool:
    """Check if a model directory contains all required weight files.
    Also checks for sibling weights repo.
    """
    file_list = _scan_model_directory(model_dir, recursive=True)
    if file_list is None or not all(f.size is not None for f in file_list):
        return False
    if (
        card is not None
        and card.vision is not None
        and card.vision.weights_repo != str(card.model_id)
    ):
        vision_id = ModelId(card.vision.weights_repo)
        normalized = vision_id.normalize()
        for search_dir in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS):
            candidate = search_dir / normalized
            if candidate.is_dir() and is_model_directory_complete(candidate):
                return True
        return False
    return True


async def _build_file_list_from_local_directory(
    model_id: ModelId,
    recursive: bool = False,
) -> list[FileListEntry] | None:
    """Build a file list from locally existing model files.

    We can only figure out the files we need from safetensors index, so
    a local directory must contain a *.safetensors.index.json and
    safetensors listed there.
    """
    normalized = model_id.normalize()
    for search_dir in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS):
        model_dir = search_dir / normalized
        if await aios.path.exists(model_dir):
            file_list = await asyncio.to_thread(
                _scan_model_directory, model_dir, recursive
            )
            if file_list:
                return file_list
    return None


async def fetch_file_list_with_cache(
    model_id: ModelId,
    revision: str = "main",
    recursive: bool = False,
    skip_internet: bool = False,
    on_connection_lost: Callable[[], None] = lambda: None,
) -> list[FileListEntry]:
    target_dir = await ensure_cache_dir(model_id)
    cache_file = target_dir / f"{model_id.normalize()}--{revision}--file_list.json"

    # cache survives process restarts so cold starts don't re-burst HF
    if await aios.path.exists(cache_file):
        try:
            cache_age = time.time() - (await aios.stat(cache_file)).st_mtime
        except OSError:
            cache_age = float("inf")
        if cache_age < _FILE_LIST_CACHE_TTL_SECS:
            async with aiofiles.open(cache_file, "r") as f:
                return TypeAdapter(list[FileListEntry]).validate_json(await f.read())

    if skip_internet:
        if await aios.path.exists(cache_file):
            async with aiofiles.open(cache_file, "r") as f:
                return TypeAdapter(list[FileListEntry]).validate_json(await f.read())
        local_file_list = await _build_file_list_from_local_directory(
            model_id, recursive
        )
        if local_file_list is not None:
            logger.warning(
                f"No internet and no cached file list for {model_id} - using local file list"
            )
            return local_file_list
        raise FileNotFoundError(
            f"No internet connection and no cached file list for {model_id}"
        )

    try:
        file_list = await fetch_file_list_with_retry(
            model_id,
            revision,
            recursive=recursive,
            on_connection_lost=on_connection_lost,
        )
        async with aiofiles.open(cache_file, "w") as f:
            await f.write(
                TypeAdapter(list[FileListEntry]).dump_json(file_list).decode()
            )
        return file_list
    except Exception as e:
        logger.opt(exception=e).warning(
            "Ran into exception when fetching file list from HF."
        )

        if await aios.path.exists(cache_file):
            logger.warning(
                f"No cached file list for {model_id} - using local file list"
            )
            async with aiofiles.open(cache_file, "r") as f:
                return TypeAdapter(list[FileListEntry]).validate_json(await f.read())
        local_file_list = await _build_file_list_from_local_directory(
            model_id, recursive
        )
        if local_file_list is not None:
            logger.warning(
                f"Failed to fetch file list for {model_id} and no cache exists, using local file list"
            )
            return local_file_list
        raise FileNotFoundError(f"Failed to fetch file list for {model_id}: {e}") from e


async def fetch_file_list_with_retry(
    model_id: ModelId,
    revision: str = "main",
    path: str = "",
    recursive: bool = False,
    on_connection_lost: Callable[[], None] = lambda: None,
) -> list[FileListEntry]:
    n_attempts = 5
    for attempt in range(n_attempts):
        try:
            return await _fetch_file_list(model_id, revision, path, recursive)
        except HuggingFaceAuthenticationError:
            raise
        except HuggingFaceRateLimitError as e:
            if attempt == n_attempts - 1:
                raise
            sleep_for = e.retry_after if e.retry_after is not None else 2.0**attempt
            sleep_for = min(sleep_for, _RATE_LIMIT_MAX_SLEEP_SECS) + random.uniform(
                0, 1
            )
            logger.warning(
                f"Rate limited by HuggingFace fetching file list for {model_id}; "
                f"sleeping {sleep_for:.1f}s before retry {attempt + 2}/{n_attempts}"
            )
            await asyncio.sleep(sleep_for)
        except Exception as e:
            on_connection_lost()
            if attempt == n_attempts - 1:
                raise e
            await asyncio.sleep(2.0**attempt + random.uniform(0, 1))
    raise Exception(
        f"Failed to fetch file list for {model_id=} {revision=} {path=} {recursive=}"
    )


async def _fetch_file_list(
    model_id: ModelId, revision: str = "main", path: str = "", recursive: bool = False
) -> list[FileListEntry]:
    api_url = f"{get_hf_endpoint()}/api/models/{model_id}/tree/{revision}"
    url = f"{api_url}/{path}" if path else api_url
    # ?recursive=true returns the whole subtree in one request
    if recursive:
        url = f"{url}?recursive=true"

    headers = await get_download_headers()
    async with (
        create_http_session(timeout_profile="short") as session,
        session.get(url, headers=headers) as response,
    ):
        if response.status in [401, 403]:
            msg = await _build_auth_error_message(response.status, model_id)
            raise HuggingFaceAuthenticationError(msg)
        elif response.status == 429:
            raise HuggingFaceRateLimitError(
                f"HuggingFace rate limit hit fetching file list for {model_id}",
                retry_after=_parse_retry_after(response.headers),
            )
        elif response.status == 200:
            data_json = await response.text()
            data = TypeAdapter(list[FileListEntry]).validate_json(data_json)
            files: list[FileListEntry] = []
            for item in data:
                if item.type == "file":
                    files.append(FileListEntry.model_validate(item))
                elif item.type == "directory" and recursive:
                    # already inlined by ?recursive=true
                    continue
            if recursive and len(data) >= 1000:
                # HF tree endpoint paginates at 1000; we don't follow cursors
                logger.warning(
                    f"File list for {model_id} hit the 1000-entry page cap "
                    "and may be truncated; cursor pagination is not implemented"
                )
            return files
        else:
            raise Exception(f"Failed to fetch file list: {response.status}")


async def get_download_headers() -> dict[str, str]:
    return {**(await get_auth_headers()), "Accept-Encoding": "identity"}


def create_http_session(
    auto_decompress: bool = False,
    timeout_profile: Literal["short", "long"] = "long",
) -> aiohttp.ClientSession:
    if timeout_profile == "short":
        total_timeout = 30
        connect_timeout = 10
        sock_read_timeout = 30
        sock_connect_timeout = 10
    else:
        total_timeout = 1800
        connect_timeout = 60
        sock_read_timeout = 60
        sock_connect_timeout = 60

    ssl_context = ssl.create_default_context(
        cafile=os.getenv("SSL_CERT_FILE") or certifi.where()
    )
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    return aiohttp.ClientSession(
        auto_decompress=auto_decompress,
        connector=connector,
        proxy=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None,
        timeout=aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_read=sock_read_timeout,
            sock_connect=sock_connect_timeout,
        ),
    )


async def calc_hash(path: Path, hash_type: Literal["sha1", "sha256"] = "sha1") -> str:
    hasher = hashlib.sha1() if hash_type == "sha1" else hashlib.sha256()
    if hash_type == "sha1":
        header = f"blob {(await aios.stat(path)).st_size}\0".encode()
        hasher.update(header)
    async with aiofiles.open(path, "rb") as f:
        while chunk := await f.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


async def file_meta(
    model_id: ModelId, revision: str, path: str, redirected_location: str | None = None
) -> tuple[int, str]:
    url = (
        urljoin(f"{get_hf_endpoint()}/{model_id}/resolve/{revision}/", path)
        if redirected_location is None
        else f"{get_hf_endpoint()}{redirected_location}"
    )
    headers = await get_download_headers()
    async with (
        create_http_session(timeout_profile="short") as session,
        session.head(url, headers=headers) as r,
    ):
        if r.status == 307:
            # On redirect, only trust Hugging Face's x-linked-* headers.
            x_linked_size = r.headers.get("x-linked-size")
            x_linked_etag = r.headers.get("x-linked-etag")
            if x_linked_size and x_linked_etag:
                content_length = int(x_linked_size)
                etag = trim_etag(x_linked_etag)
                return content_length, etag
            # Otherwise, follow the redirect to get authoritative size/hash
            redirected_location = r.headers.get("location")
            return await file_meta(model_id, revision, path, redirected_location)
        if r.status in [401, 403]:
            msg = await _build_auth_error_message(r.status, model_id)
            raise HuggingFaceAuthenticationError(msg)
        if r.status == 429:
            raise HuggingFaceRateLimitError(
                f"HuggingFace rate limit hit fetching metadata for {model_id}/{path}",
                retry_after=_parse_retry_after(r.headers),
            )
        content_length = int(
            r.headers.get("x-linked-size") or r.headers.get("content-length") or 0
        )
        etag = r.headers.get("x-linked-etag") or r.headers.get("etag")
        assert content_length > 0, f"No content length for {url}"
        assert etag is not None, f"No remote hash for {url}"
        etag = trim_etag(etag)
        return content_length, etag


async def download_file_with_retry(
    model_id: ModelId,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    on_connection_lost: Callable[[], None] = lambda: None,
    skip_internet: bool = False,
) -> Path:
    n_attempts = 5
    for attempt in range(n_attempts):
        try:
            return await _download_file(
                model_id, revision, path, target_dir, on_progress, skip_internet
            )
        except HuggingFaceAuthenticationError:
            raise
        except FileNotFoundError:
            raise
        except HuggingFaceRateLimitError as e:
            if attempt == n_attempts - 1:
                raise
            sleep_for = e.retry_after if e.retry_after is not None else 2.0**attempt
            sleep_for = min(sleep_for, _RATE_LIMIT_MAX_SLEEP_SECS) + random.uniform(
                0, 1
            )
            logger.warning(
                f"Rate limited by HuggingFace downloading {model_id}/{path}; "
                f"sleeping {sleep_for:.1f}s before retry {attempt + 2}/{n_attempts}"
            )
            await asyncio.sleep(sleep_for)
        except Exception as e:
            if attempt == n_attempts - 1:
                on_connection_lost()
                raise e
            logger.error(
                f"Download error on attempt {attempt + 1}/{n_attempts} for {model_id=} {revision=} {path=} {target_dir=}"
            )
            logger.error(traceback.format_exc())
            await asyncio.sleep(2.0**attempt + random.uniform(0, 1))
    raise Exception(
        f"Failed to download file {model_id=} {revision=} {path=} {target_dir=}"
    )


async def _download_file(
    model_id: ModelId,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    skip_internet: bool = False,
) -> Path:
    target_path = target_dir / path

    if await aios.path.exists(target_path):
        if skip_internet:
            return target_path

        local_size = (await aios.stat(target_path)).st_size

        # Try to verify against remote, but allow offline operation
        try:
            remote_size, _ = await file_meta(model_id, revision, path)
            if local_size != remote_size:
                logger.info(
                    f"File {path} size mismatch (local={local_size}, remote={remote_size}), re-downloading"
                )
                await aios.remove(target_path)
            else:
                return target_path
        except Exception as e:
            # Offline or network error - trust local file
            logger.debug(
                f"Could not verify {path} against remote (offline?): {e}, using local file"
            )
            return target_path

    if skip_internet:
        raise FileNotFoundError(
            f"File {path} not found locally and cannot download in offline mode"
        )

    await aios.makedirs((target_dir / path).parent, exist_ok=True)
    length, etag = await file_meta(model_id, revision, path)
    remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
    partial_path = target_dir / f"{path}.partial"
    resume_byte_pos = (
        (await aios.stat(partial_path)).st_size
        if (await aios.path.exists(partial_path))
        else None
    )
    if resume_byte_pos != length:
        url = urljoin(f"{get_hf_endpoint()}/{model_id}/resolve/{revision}/", path)
        headers = await get_download_headers()
        if resume_byte_pos:
            headers["Range"] = f"bytes={resume_byte_pos}-"
        n_read = resume_byte_pos or 0
        async with (
            create_http_session(timeout_profile="long") as session,
            session.get(url, headers=headers) as r,
        ):
            if r.status == 404:
                raise FileNotFoundError(f"File not found: {url}")
            if r.status in [401, 403]:
                msg = await _build_auth_error_message(r.status, model_id)
                raise HuggingFaceAuthenticationError(msg)
            if r.status == 429:
                raise HuggingFaceRateLimitError(
                    f"HuggingFace rate limit hit downloading {model_id}/{path}",
                    retry_after=_parse_retry_after(r.headers),
                )
            assert r.status in [200, 206], (
                f"Failed to download {path} from {url}: {r.status}"
            )
            async with aiofiles.open(
                partial_path, "ab" if resume_byte_pos else "wb"
            ) as f:
                while chunk := await r.content.read(8 * 1024 * 1024):
                    n_read = n_read + (await f.write(chunk))
                    on_progress(n_read, length, False)

    final_hash = await calc_hash(
        partial_path, hash_type="sha256" if len(remote_hash) == 64 else "sha1"
    )
    integrity = final_hash == remote_hash
    if not integrity:
        try:
            await aios.remove(partial_path)
        except Exception as e:
            logger.error(f"Error removing partial file {partial_path}: {e}")
        raise Exception(
            f"Downloaded file {target_dir / path} has hash {final_hash} but remote hash is {remote_hash}"
        )
    await aios.rename(partial_path, target_dir / path)
    on_progress(length, length, True)
    return target_dir / path


def calculate_repo_progress(
    shard: ShardMetadata,
    model_id: ModelId,
    revision: str,
    file_progress: dict[str, RepoFileDownloadProgress],
    all_start_time: float,
) -> RepoDownloadProgress:
    all_total = sum((p.total for p in file_progress.values()), Memory.from_bytes(0))
    all_downloaded = sum(
        (p.downloaded for p in file_progress.values()), Memory.from_bytes(0)
    )
    all_downloaded_this_session = sum(
        (p.downloaded_this_session for p in file_progress.values()),
        Memory.from_bytes(0),
    )
    elapsed_time = time.time() - all_start_time
    all_speed = (
        all_downloaded_this_session.in_bytes / elapsed_time if elapsed_time > 0 else 0
    )
    all_eta = (
        timedelta(seconds=(all_total - all_downloaded).in_bytes / all_speed)
        if all_speed > 0
        else timedelta(seconds=0)
    )
    status = (
        "complete"
        if all(p.status == "complete" for p in file_progress.values())
        else "in_progress"
        if any(p.status == "in_progress" for p in file_progress.values())
        else "not_started"
    )
    return RepoDownloadProgress(
        repo_id=model_id,
        repo_revision=revision,
        shard=shard,
        completed_files=len(
            [p for p in file_progress.values() if p.downloaded == p.total]
        ),
        total_files=len(file_progress),
        downloaded=all_downloaded,
        downloaded_this_session=all_downloaded_this_session,
        total=all_total,
        overall_speed=all_speed,
        overall_eta=all_eta,
        status=status,
        file_progress=file_progress,
    )


def _parse_weight_map_from_dir(model_dir: Path) -> dict[str, str]:
    """Parse all *.safetensors.index.json files under a local directory."""
    index_files = list(model_dir.glob("**/*.safetensors.index.json"))

    weight_map: dict[str, str] = {}

    for index_file in index_files:
        relative_dir = index_file.parent.relative_to(model_dir)
        index_data = ModelSafetensorsIndex.model_validate_json(index_file.read_text())

        if relative_dir != Path("."):
            prefixed_weight_map = {
                f"{relative_dir}/{key}": str(relative_dir / value)
                for key, value in index_data.weight_map.items()
            }
            weight_map = weight_map | prefixed_weight_map
        else:
            weight_map = weight_map | index_data.weight_map

    return weight_map


def get_local_weight_map(model_id: ModelId) -> dict[str, str]:
    """Parse the weight map from index files already on disk (any model dir)."""
    normalized = model_id.normalize()
    for search_dir in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS):
        candidate = search_dir / normalized
        if candidate.is_dir():
            weight_map = _parse_weight_map_from_dir(candidate)
            if weight_map:
                return weight_map
    return {}


async def get_weight_map(model_id: ModelId, revision: str = "main") -> dict[str, str]:
    # Prefer index files that are already on disk (e.g. seeded by a peer or
    # fetched on a previous run) to avoid a HuggingFace round-trip.
    local = await asyncio.to_thread(get_local_weight_map, model_id)
    if local:
        return local

    target_dir = await resolve_model_dir(model_id)

    index_files_dir = snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        allow_patterns="*.safetensors.index.json",
    )

    return await asyncio.to_thread(_parse_weight_map_from_dir, Path(index_files_dir))


async def resolve_allow_patterns(shard: ShardMetadata) -> list[str]:
    # 'Smart' (shard-scoped) downloads are only valid for pipeline-sharded
    # text models: tensor parallel reads every tensor, and image models keep
    # weights in component subdirectories with their own layout.
    if not isinstance(shard, PipelineShardMetadata) or is_image_model(shard):
        return ["*"]
    try:
        weight_map = await get_weight_map(shard.model_card.model_id)
        if not weight_map:
            return ["*"]
        return get_allow_patterns(weight_map, shard)
    except Exception:
        logger.error(f"Error getting weight map for {shard.model_card.model_id=}")
        logger.error(traceback.format_exc())
        return ["*"]


def is_image_model(shard: ShardMetadata) -> bool:
    tasks = shard.model_card.tasks
    return ModelTask.TextToImage in tasks or ModelTask.ImageToImage in tasks


async def get_downloaded_size(path: Path) -> int:
    partial_path = path.with_suffix(path.suffix + ".partial")
    if await aios.path.exists(path):
        return (await aios.stat(path)).st_size
    if await aios.path.exists(partial_path):
        return (await aios.stat(partial_path)).st_size
    return 0


async def _build_peer_file_map(
    seed_sources: list[SeedSource], normalized_id: str
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Map relative file path -> seed base URLs holding it, plus file sizes."""
    from exo.download.peer_source import fetch_peer_file_list

    urls = [url for source in seed_sources for url in source.base_urls]
    peer_files: dict[str, list[str]] = {}
    peer_sizes: dict[str, int] = {}
    if not urls:
        return peer_files, peer_sizes

    async def _fetch(url: str) -> tuple[str, dict[str, int] | None]:
        return url, await fetch_peer_file_list(url, normalized_id)

    results = await asyncio.gather(*(_fetch(url) for url in urls))
    for url, files in results:
        if not files:
            continue
        for path, size in files.items():
            peer_files.setdefault(path, []).append(url)
            peer_sizes.setdefault(path, size)
    return peer_files, peer_sizes


async def download_shard(
    shard: ShardMetadata,
    on_progress: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    max_parallel_downloads: int = 8,
    skip_download: bool = False,
    skip_internet: bool = False,
    allow_patterns: list[str] | None = None,
    on_connection_lost: Callable[[], None] = lambda: None,
    seed_sources: list[SeedSource] | None = None,
) -> tuple[Path, RepoDownloadProgress]:
    from exo.download.peer_source import (
        PeerDownloadError,
        download_file_from_peer,
    )

    if not skip_download:
        logger.debug(f"Downloading {shard.model_card.model_id=}")

    model_id = shard.model_card.model_id
    normalized_id = model_id.normalize()
    revision = "main"

    # Discover which files each seed peer holds (parallel, failure-tolerant).
    peer_files: dict[str, list[str]] = {}
    peer_sizes: dict[str, int] = {}
    if seed_sources and not skip_download:
        peer_files, peer_sizes = await _build_peer_file_map(seed_sources, normalized_id)
        if peer_files:
            logger.info(
                f"Peer seeding available for {model_id}: "
                f"{len(peer_files)} files offered by {len(seed_sources)} peer(s)"
            )

    if not allow_patterns:
        allow_patterns = await resolve_allow_patterns(shard)

    if not skip_download:
        logger.debug(f"Downloading {model_id=} with {allow_patterns=}")

    all_start_time = time.time()
    try:
        file_list = await fetch_file_list_with_cache(
            model_id,
            revision,
            recursive=True,
            skip_internet=skip_internet,
            on_connection_lost=on_connection_lost,
        )
    except FileNotFoundError:
        # HuggingFace metadata unavailable (offline/air-gapped). If seed peers
        # hold the model we can still proceed from their file lists.
        if peer_files:
            logger.warning(
                f"No HuggingFace file list for {model_id}; using peer file lists"
            )
            file_list = [
                FileListEntry(type="file", path=path, size=peer_sizes.get(path))
                for path in peer_files
            ]
        else:
            not_started_progress = RepoDownloadProgress(
                repo_id=str(model_id),
                repo_revision=revision,
                shard=shard,
                completed_files=0,
                total_files=0,
                downloaded=Memory.from_bytes(0),
                downloaded_this_session=Memory.from_bytes(0),
                total=Memory.from_bytes(0),
                overall_speed=0.0,
                overall_eta=timedelta(0),
                status="not_started",
                file_progress={},
            )
            return EXO_DEFAULT_MODELS_DIR / normalized_id, not_started_progress
    filtered_file_list = list(
        filter_repo_objects(
            file_list,
            allow_patterns=allow_patterns,
            ignore_patterns=["original/*", "metal/*", SHARD_MARKER_FILENAME],
            key=lambda x: x.path,
        )
    )

    # For image models, skip root-level safetensors files since weights
    # are stored in component subdirectories (e.g., transformer/, vae/)
    if is_image_model(shard):
        filtered_file_list = [
            f
            for f in filtered_file_list
            if "/" in f.path or not f.path.endswith(".safetensors")
        ]

    # Pick a writable directory with enough free space.
    total_size = sum(f.size or 0 for f in filtered_file_list)
    if skip_download:
        existing = resolve_existing_model(model_id)
        target_dir = (
            existing
            if existing is not None
            else EXO_DEFAULT_MODELS_DIR / model_id.normalize()
        )
    else:
        models_dir = await select_download_dir_for_shard(
            model_id, filtered_file_list, total_size
        )
        target_dir = models_dir / model_id.normalize()
        await aios.makedirs(target_dir, exist_ok=True)
    file_progress: dict[str, RepoFileDownloadProgress] = {}

    async def on_progress_wrapper(
        file: FileListEntry, curr_bytes: int, total_bytes: int, is_renamed: bool
    ) -> None:
        previous_progress = file_progress.get(file.path)

        # Detect re-download: curr_bytes < previous downloaded means file was deleted and restarted
        is_redownload = (
            previous_progress is not None
            and curr_bytes < previous_progress.downloaded.in_bytes
        )

        if is_redownload or previous_progress is None:
            # Fresh download or re-download: reset tracking
            start_time = time.time()
            downloaded_this_session = curr_bytes
        else:
            # Continuing download: accumulate
            start_time = previous_progress.start_time
            downloaded_this_session = (
                previous_progress.downloaded_this_session.in_bytes
                + (curr_bytes - previous_progress.downloaded.in_bytes)
            )

        speed = (
            downloaded_this_session / (time.time() - start_time)
            if time.time() - start_time > 0
            else 0
        )
        eta = (
            timedelta(seconds=(total_bytes - curr_bytes) / speed)
            if speed > 0
            else timedelta(seconds=0)
        )
        file_progress[file.path] = RepoFileDownloadProgress(
            repo_id=model_id,
            repo_revision=revision,
            file_path=file.path,
            downloaded=Memory.from_bytes(curr_bytes),
            downloaded_this_session=Memory.from_bytes(downloaded_this_session),
            total=Memory.from_bytes(total_bytes),
            speed=speed,
            eta=eta,
            status="complete"
            if curr_bytes == total_bytes and is_renamed
            else "in_progress",
            start_time=start_time,
        )
        await on_progress(
            shard,
            calculate_repo_progress(
                shard,
                shard.model_card.model_id,
                revision,
                file_progress,
                all_start_time,
            ),
        )

    for file in filtered_file_list:
        downloaded_bytes = await get_downloaded_size(target_dir / file.path)
        final_file_exists = await aios.path.exists(target_dir / file.path)
        file_progress[file.path] = RepoFileDownloadProgress(
            repo_id=model_id,
            repo_revision=revision,
            file_path=file.path,
            downloaded=Memory.from_bytes(downloaded_bytes),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_bytes(file.size or 0),
            speed=0,
            eta=timedelta(0),
            status="complete"
            if final_file_exists and downloaded_bytes == file.size
            else "not_started",
            start_time=time.time(),
        )

    semaphore = asyncio.Semaphore(max_parallel_downloads)

    def schedule_progress(
        file: FileListEntry, curr_bytes: int, total_bytes: int, is_renamed: bool
    ) -> None:
        asyncio.create_task(
            on_progress_wrapper(file, curr_bytes, total_bytes, is_renamed)
        )

    async def download_with_semaphore(file: FileListEntry) -> None:
        async with semaphore:
            target_path = target_dir / file.path

            def on_file_progress(curr_bytes: int, total_bytes: int, is_renamed: bool):
                schedule_progress(file, curr_bytes, total_bytes, is_renamed)

            # Skip files already fully present locally. This check uses the
            # repo/peer file list size and avoids one HTTP HEAD per file,
            # which keeps restarts fast and works without internet.
            if (
                file.size is not None
                and await aios.path.exists(target_path)
                and (await aios.stat(target_path)).st_size == file.size
            ):
                on_file_progress(file.size, file.size, True)
                return

            # Seed peers first: the LAN is typically orders of magnitude
            # faster than HuggingFace, and it works without internet.
            for base_url in peer_files.get(file.path, []):
                try:
                    await download_file_from_peer(
                        base_url,
                        normalized_id,
                        file.path,
                        target_dir,
                        file.size,
                        on_progress=on_file_progress,
                    )
                    logger.info(f"Seeded {file.path} from peer {base_url}")
                    return
                except PeerDownloadError as e:
                    logger.warning(
                        f"Peer seeding of {file.path} from {base_url} failed: {e}"
                    )

            await download_file_with_retry(
                model_id,
                revision,
                file.path,
                target_dir,
                lambda curr_bytes, total_bytes, is_renamed: schedule_progress(
                    file, curr_bytes, total_bytes, is_renamed
                ),
                on_connection_lost=on_connection_lost,
                skip_internet=skip_internet,
            )

    if not skip_download:
        await asyncio.gather(
            *[download_with_semaphore(file) for file in filtered_file_list]
        )
        await _write_shard_marker(target_dir, shard, filtered_file_list)
    final_repo_progress = calculate_repo_progress(
        shard, model_id, revision, file_progress, all_start_time
    )
    await on_progress(shard, final_repo_progress)
    if gguf := next((f for f in filtered_file_list if f.path.endswith(".gguf")), None):
        return target_dir / gguf.path, final_repo_progress
    else:
        return target_dir, final_repo_progress


async def _write_shard_marker(
    target_dir: Path, shard: ShardMetadata, files: list[FileListEntry]
) -> None:
    """Record which shard layers were fetched into this directory.

    The marker lets future runs recognise a shard-scoped (partial) model
    directory as loadable without re-downloading, and tells other nodes
    exactly which files this directory can seed.
    """
    match shard:
        case PipelineShardMetadata():
            start_layer, end_layer, n_layers = (
                shard.start_layer,
                shard.end_layer,
                shard.n_layers,
            )
        case _:
            # Tensor/CFG shards always fetch the full repository.
            n_layers = shard.n_layers
            start_layer, end_layer = 0, n_layers
    marker = {
        "model_id": str(shard.model_card.model_id),
        "start_layer": start_layer,
        "end_layer": end_layer,
        "n_layers": n_layers,
        "files": {f.path: f.size for f in files if f.size is not None},
    }
    marker_path = target_dir / SHARD_MARKER_FILENAME
    async with aiofiles.open(marker_path, "w") as f:
        await f.write(json.dumps(marker))


def _read_shard_marker(model_dir: Path) -> dict[str, object] | None:
    marker_path = model_dir / SHARD_MARKER_FILENAME
    if not marker_path.is_file():
        return None
    try:
        raw_marker = cast(object, json.loads(marker_path.read_text(encoding="utf-8")))
        if not isinstance(raw_marker, Mapping):
            return None
        marker: dict[str, object] = {}
        for key, value in cast(Mapping[object, object], raw_marker).items():
            if not isinstance(key, str):
                return None
            marker[key] = value
        if "files" not in marker:
            return None
        return marker
    except (OSError, ValueError):
        return None


def marker_covers_shard(marker: Mapping[str, object], shard: ShardMetadata) -> bool:
    """True if a directory with this marker can serve ``shard`` without
    fetching any more files."""
    if marker.get("n_layers") != shard.n_layers:
        return False
    m_start, m_end = marker.get("start_layer"), marker.get("end_layer")
    if not isinstance(m_start, int) or not isinstance(m_end, int):
        return False
    if m_start == 0 and m_end == shard.n_layers:
        return True
    if not isinstance(shard, PipelineShardMetadata):
        # Non-pipeline shards need every file: only a full marker covers them.
        return False
    return m_start <= shard.start_layer and m_end >= shard.end_layer


def _marker_files(marker: Mapping[str, object]) -> dict[str, int] | None:
    raw_files = marker.get("files")
    if not isinstance(raw_files, Mapping):
        return None
    files: dict[str, int] = {}
    for raw_path, raw_size in cast(Mapping[object, object], raw_files).items():
        if (
            not isinstance(raw_path, str)
            or isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or raw_size < 0
        ):
            return None
        files[raw_path] = raw_size
    return files


def _rank_local_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RankLocalConfigurationError(f"{label} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise RankLocalConfigurationError(f"{label} contains a non-string key")
        result[key] = item
    return result


def _read_rank_local_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RankLocalConfigurationError(
            f"rank-local checkpoint is missing regular {label}"
        )
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise RankLocalConfigurationError(
            f"rank-local checkpoint has invalid {label}"
        ) from exc
    return _rank_local_object(value, label)


def _safe_rank_local_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RankLocalConfigurationError(
            f"unsafe rank-local checkpoint filename {value!r}"
        )
    relative = Path(value)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise RankLocalConfigurationError(
            f"unsafe rank-local checkpoint filename {value!r}"
        )
    return relative.as_posix()


def _rank_local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_local_metadata_contract_sha256(
    metadata_files: Mapping[str, Mapping[str, object]],
) -> str:
    return hashlib.sha256(
        json.dumps(
            metadata_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_rank_local_metadata(
    checkpoint: Path,
    manifest: Mapping[str, object],
    source: Mapping[str, object],
    weight_filenames: set[str],
) -> None:
    raw_metadata = _rank_local_object(
        manifest.get("metadata_files"), "manifest metadata_files"
    )
    if not raw_metadata:
        raise RankLocalConfigurationError(
            "rank-local checkpoint manifest lists no metadata files"
        )

    normalized: dict[str, dict[str, object]] = {}
    for filename, raw_record in raw_metadata.items():
        if filename not in PINNED_RANK_LOCAL_METADATA_FILES:
            raise RankLocalConfigurationError(
                f"rank-local checkpoint metadata is not allowlisted: {filename}"
            )
        safe_name = _safe_rank_local_relative_path(filename)
        if safe_name != filename or "/" in filename:
            raise RankLocalConfigurationError(
                f"non-canonical rank-local metadata filename {filename!r}"
            )
        record = _rank_local_object(raw_record, f"manifest metadata record {filename}")
        if set(record) != {"bytes", "sha256"}:
            raise RankLocalConfigurationError(
                f"malformed rank-local metadata record for {filename}"
            )
        byte_count = record.get("bytes")
        checksum = record.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise RankLocalConfigurationError(
                f"malformed rank-local metadata record for {filename}"
            )
        path = checkpoint / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size != byte_count:
            raise RankLocalConfigurationError(
                f"rank-local metadata is missing or truncated: {path}"
            )
        if _rank_local_sha256(path) != checksum:
            raise RankLocalConfigurationError(
                f"rank-local metadata checksum mismatch: {path}"
            )
        normalized[filename] = {"bytes": byte_count, "sha256": checksum}

    if normalized != PINNED_RANK_LOCAL_METADATA_FILES:
        raise RankLocalConfigurationError(
            "rank-local checkpoint metadata differs from the authenticated "
            "source contract"
        )
    if normalized["config.json"]["sha256"] != source.get("config_sha256"):
        raise RankLocalConfigurationError(
            "rank-local metadata config hash differs from the source contract"
        )

    contract_digest = manifest.get("metadata_contract_sha256")
    actual_digest = _rank_local_metadata_contract_sha256(normalized)
    if (
        not isinstance(contract_digest, str)
        or len(contract_digest) != 64
        or any(character not in "0123456789abcdef" for character in contract_digest)
        or contract_digest != actual_digest
    ):
        raise RankLocalConfigurationError(
            "rank-local metadata contract digest mismatch"
        )

    expected_entries = (
        set(normalized)
        | weight_filenames
        | {"model.safetensors.index.json", "tp_manifest.json"}
    )
    actual_entries: set[str] = set()
    for child in checkpoint.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RankLocalConfigurationError(
                f"rank-local checkpoint contains a non-regular entry: {child}"
            )
        actual_entries.add(child.name)
    if actual_entries != expected_entries:
        extra = sorted(actual_entries - expected_entries)
        missing_entries = sorted(expected_entries - actual_entries)
        raise RankLocalConfigurationError(
            "rank-local checkpoint file inventory differs from its manifest: "
            f"extra={extra}, missing={missing_entries}"
        )


def _validate_rank_local_checkpoint(
    checkpoint: Path,
    model_id: ModelId,
    shard: TensorShardMetadata,
) -> None:
    """Validate rank-local readiness without reading weight payloads.

    The manifest, config, and rank-local index are small and authenticated.
    Weight shards are checked by safe filename, regular-file identity, and
    exact byte size; hashing hundreds of GiB here would duplicate the loader's
    optional full-integrity pass and delay every readiness check.
    """

    manifest = _read_rank_local_json(
        checkpoint / "tp_manifest.json", "tp_manifest.json"
    )
    if manifest.get("schema") != SUPPORTED_LOADER_SCHEMA or (
        manifest.get("complete") is not True
    ):
        raise RankLocalConfigurationError(
            "rank-local checkpoint manifest is incomplete or unsupported"
        )

    source = _rank_local_object(manifest.get("source"), "manifest source")
    if (
        source.get("repo") != str(model_id)
        or source.get("revision") != SUPPORTED_SOURCE_REVISION
        or source.get("config_sha256") != SUPPORTED_SOURCE_CONFIG_SHA256
        or source.get("index_sha256") != SUPPORTED_SOURCE_INDEX_SHA256
    ):
        raise RankLocalConfigurationError(
            "rank-local checkpoint source contract is not the audited checkpoint"
        )
    config_sha256 = source.get("config_sha256")
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise RankLocalConfigurationError(
            "rank-local checkpoint manifest has an invalid config hash"
        )

    runtime = _rank_local_object(manifest.get("runtime"), "manifest runtime")
    if (
        runtime.get("mlx_lm_commit") != SUPPORTED_MLX_LM_COMMIT
        or runtime.get("mlx_lm_kimi_k3_sha256") != SUPPORTED_MLX_LM_KIMI_K3_SHA256
    ):
        raise RankLocalConfigurationError(
            "rank-local checkpoint runtime contract is not audited"
        )

    tp = _rank_local_object(manifest.get("tp"), "manifest tp")
    if (
        tp.get("rank") != shard.device_rank
        or tp.get("world_size") != shard.world_size
        or tp.get("contract") != SUPPORTED_TP_CONTRACT
        or tp.get("contract_digest") != SUPPORTED_TP_CONTRACT_DIGEST
    ):
        raise RankLocalConfigurationError(
            "rank-local checkpoint TP contract does not match TensorShardMetadata"
        )

    rank_data_bytes = manifest.get("rank_data_bytes")
    if (
        isinstance(rank_data_bytes, bool)
        or not isinstance(rank_data_bytes, int)
        or rank_data_bytes <= 0
    ):
        raise RankLocalConfigurationError(
            "rank-local checkpoint manifest has invalid rank_data_bytes"
        )

    files = _rank_local_object(manifest.get("files"), "manifest files")
    if not files:
        raise RankLocalConfigurationError(
            "rank-local checkpoint manifest lists no weight files"
        )
    manifest_filenames: set[str] = set()
    for filename, raw_record in files.items():
        safe_name = _safe_rank_local_relative_path(filename)
        if safe_name != filename:
            raise RankLocalConfigurationError(
                f"non-canonical rank-local checkpoint filename {filename!r}"
            )
        record = _rank_local_object(raw_record, f"manifest file record {filename}")
        if record.get("name") != filename:
            raise RankLocalConfigurationError(
                f"rank-local checkpoint file record/name mismatch for {filename}"
            )
        byte_count = record.get("bytes")
        checksum = record.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise RankLocalConfigurationError(
                f"malformed rank-local checkpoint file record for {filename}"
            )
        weight_path = checkpoint / filename
        try:
            resolved_weight = weight_path.resolve(strict=True)
        except OSError as exc:
            raise RankLocalConfigurationError(
                f"rank-local checkpoint file is missing: {weight_path}"
            ) from exc
        if (
            weight_path.is_symlink()
            or not resolved_weight.is_file()
            or not resolved_weight.is_relative_to(checkpoint)
            or resolved_weight.stat().st_size != byte_count
        ):
            raise RankLocalConfigurationError(
                f"rank-local checkpoint file is missing or truncated: {weight_path}"
            )
        manifest_filenames.add(filename)

    _validate_rank_local_metadata(
        checkpoint,
        manifest,
        source,
        manifest_filenames,
    )

    index = _read_rank_local_json(
        checkpoint / "model.safetensors.index.json",
        "model.safetensors.index.json",
    )
    metadata = _rank_local_object(index.get("metadata"), "rank-local metadata")
    if metadata.get("total_size") != rank_data_bytes:
        raise RankLocalConfigurationError(
            "rank-local checkpoint index and manifest byte totals differ"
        )
    weight_map = _rank_local_object(index.get("weight_map"), "rank-local weight_map")
    if not weight_map:
        raise RankLocalConfigurationError("rank-local checkpoint index has no weights")
    index_tensor_files: dict[str, str] = {}
    for tensor_name, raw_filename in weight_map.items():
        if not tensor_name:
            raise RankLocalConfigurationError(
                "rank-local checkpoint index has an empty tensor name"
            )
        index_tensor_files[tensor_name] = _safe_rank_local_relative_path(raw_filename)
    index_filenames = set(index_tensor_files.values())
    if index_filenames != manifest_filenames:
        raise RankLocalConfigurationError(
            "rank-local checkpoint index and manifest list different files"
        )

    tensors = _rank_local_object(manifest.get("tensors"), "manifest tensors")
    if set(tensors) != set(index_tensor_files):
        raise RankLocalConfigurationError(
            "rank-local checkpoint index and manifest list different tensors"
        )
    tensor_bytes = 0
    for tensor_name, raw_record in tensors.items():
        record = _rank_local_object(raw_record, f"manifest tensor record {tensor_name}")
        source_file = _safe_rank_local_relative_path(record.get("source_file"))
        byte_count = record.get("bytes")
        rank_shape = record.get("rank_shape")
        if (
            source_file != index_tensor_files[tensor_name]
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(rank_shape, list)
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in cast(list[object], rank_shape)
            )
        ):
            raise RankLocalConfigurationError(
                f"malformed rank-local checkpoint tensor record for {tensor_name}"
            )
        tensor_bytes += byte_count
    if tensor_bytes != rank_data_bytes:
        raise RankLocalConfigurationError(
            "rank-local checkpoint tensor and manifest byte totals differ"
        )


def _resolve_configured_rank_local_model_for_shard(
    model_id: ModelId, shard: ShardMetadata
) -> Path | None:
    checkpoint = resolve_configured_rank_local_checkpoint_path(shard)
    if checkpoint is None:
        return None
    if not isinstance(shard, TensorShardMetadata):
        raise AssertionError("rank-local path resolver returned a non-tensor shard")
    if model_id != shard.model_card.model_id:
        raise RankLocalConfigurationError(
            "requested model ID differs from TensorShardMetadata"
        )
    _validate_rank_local_checkpoint(checkpoint, model_id, shard)
    preflight_rank_local_runtime()
    return checkpoint


def resolve_existing_model_for_shard(
    model_id: ModelId, shard: ShardMetadata
) -> Path | None:
    """Find a local directory that already holds everything ``shard`` needs.

    Unlike :func:`resolve_existing_model` this accepts shard-scoped (partial)
    model directories recorded by a shard marker, so a node can restart and
    reload its pipeline shard without re-downloading.
    """
    rank_local = _resolve_configured_rank_local_model_for_shard(model_id, shard)
    if rank_local is not None:
        return rank_local

    normalized = model_id.normalize()
    for search_dir in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS):
        candidate = search_dir / normalized
        if not candidate.is_dir():
            continue
        marker = _read_shard_marker(candidate)
        if marker is None or not marker_covers_shard(marker, shard):
            continue
        marker_files = _marker_files(marker)
        if marker_files is None:
            continue
        # Verify every recorded file is still present with its recorded size.
        if all(
            (candidate / path).is_file() and (candidate / path).stat().st_size == size
            for path, size in marker_files.items()
        ):
            return candidate
    return None


def build_model_path_for_shard(model_id: ModelId, shard: ShardMetadata) -> Path:
    """Resolve the on-disk path to load ``shard`` from (full or partial dir)."""
    # An explicit tensor-rank checkpoint wins over a legacy full checkpoint;
    # invalid opt-in state raises instead of silently falling back and risking
    # a full-checkpoint load or duplicate download.
    found = resolve_existing_model_for_shard(model_id, shard)
    if found is not None:
        return found
    found = resolve_existing_model(model_id)
    if found is not None:
        return found
    return EXO_DEFAULT_MODELS_DIR / model_id.normalize()
