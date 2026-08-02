"""Opt-in bridge to an externally audited rank-local MLX checkpoint loader.

Rank-local checkpoints contain tensors that have already been sliced for a
specific tensor-parallel rank.  They must be loaded into an empty model after
its structure is sharded; routing them through ``mlx_lm.utils.load_model``
would materialize the full checkpoint first and can exhaust unified memory.

This module deliberately contains only configuration and contract checks.  The
checkpoint-specific validation and load order remain in the external
``rank_local_loader.py`` used to create and smoke-test the Kimi K3 TP2 files.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Protocol, cast

from exo.shared.types.worker.shards import ShardMetadata, TensorShardMetadata

RANK_LOCAL_CHECKPOINT_ENV = "EXO_MLX_RANK_LOCAL_CHECKPOINT"
RANK_LOCAL_LOADER_ENV = "EXO_MLX_RANK_LOCAL_LOADER"
RANK_LOCAL_VERIFY_HASHES_ENV = "EXO_MLX_RANK_LOCAL_VERIFY_HASHES"
RANK_LOCAL_VOCAB_PARALLEL_HEAD_ENV = "EXO_MLX_K3_VOCAB_PARALLEL_HEAD"

SUPPORTED_MODEL_ID = "kernelpool/Kimi-K3-2bit-UVMAX"
SUPPORTED_LOADER_SCHEMA = "k3-rank-local-tp/v2"
SUPPORTED_SOURCE_REVISION = "edb5113218df612f4a92f95145680f3f8eacd375"
SUPPORTED_SOURCE_CONFIG_SHA256 = (
    "d041003554810a367bb600d18733976bdd21041bb46e75cc1e27c7b15fe034d0"
)
SUPPORTED_SOURCE_INDEX_SHA256 = (
    "ac65bcb3cd9e07cab3e7942ff455dde33879a9e02211bae40938e22fc204ae09"
)
SUPPORTED_MLX_LM_COMMIT = "7d505c285b801108a52c23353c7fb6af07204717"
SUPPORTED_MLX_LM_KIMI_K3_SHA256 = (
    "3dd2e9db585190bca118d5812bcb5b103d1e7c6ec12187b20351992fed7e63cc"
)
SUPPORTED_TP_CONTRACT = "mlx-lm-kimi-k3-shard@7d505c2"
SUPPORTED_TP_CONTRACT_DIGEST = (
    "1b7fdf1b28433fb08fff7e0e26a7bccc2ca0fcd51f29498ab611892c9fc48da5"
)
SUPPORTED_LOADER_SHA256 = (
    "cd3407fa7724e2fe0faca020a4b5fd327a010dde447285c3094e868d209b9f9f"
)
# Exact non-weight inventory from SOURCE_REPO@SUPPORTED_SOURCE_REVISION.  Some
# of these files execute under ``trust_remote_code=True``.  Pinning only their
# names would authorize substituted Python, so the runtime authenticates the
# complete inventory, size, and digest before importing the loader.
PINNED_RANK_LOCAL_METADATA_FILES: dict[str, dict[str, int | str]] = {
    "README.md": {
        "bytes": 1923,
        "sha256": "d0d7a4d1a5af37c542594449d2ce893b9e3c33ccb031afb71cf13e4f23a5349d",
    },
    "added_tokens.json": {
        "bytes": 200,
        "sha256": "27373c2f39a52c87e674caf7e9604ec6756c68c5f8d5f140657299048b6ab8ba",
    },
    "config.json": {"bytes": 459349, "sha256": SUPPORTED_SOURCE_CONFIG_SHA256},
    "configuration_kimi_k3.py": {
        "bytes": 11343,
        "sha256": "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
    },
    "encoding_k3.py": {
        "bytes": 22827,
        "sha256": "b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333",
    },
    "generation_config.json": {
        "bytes": 53,
        "sha256": "c6648c25e9705af7fba8847e243840d21b5cc63ddeb6297f750a7ddbb6a02836",
    },
    "kimi_k3_processor.py": {
        "bytes": 7660,
        "sha256": "ec9f7e86d2ab0eee07a8e7e7c037046e77ac3c25a710ad1298ec13be3b585b54",
    },
    "kimi_k3_vision_processing.py": {
        "bytes": 6686,
        "sha256": "d122b30bfd3a51a6f05d4bfcfda1e657827322b1353f7caefeebc2835d7736b5",
    },
    "media_utils.py": {
        "bytes": 13844,
        "sha256": "78403540328f9847d6b7ebc5c44eb2e6a752863de0afb7d0710728bb161dc60d",
    },
    "modeling_kimi_k3.py": {
        "bytes": 53444,
        "sha256": "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2",
    },
    "modeling_kimi_linear.py": {
        "bytes": 51506,
        "sha256": "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    },
    "preprocessor_config.json": {
        "bytes": 1011,
        "sha256": "4be333605990c53a816e586dee9d5dd545afb7a59947c17f8f7ef26b4782668e",
    },
    "tiktoken.model": {
        "bytes": 2795286,
        "sha256": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    },
    "tokenization_kimi.py": {
        "bytes": 16145,
        "sha256": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    },
    "tokenizer_config.json": {
        "bytes": 4790,
        "sha256": "d06a6e8a2ef0a09d62031591d0ea2b7c5128fd28a17ea693984bf85eafade1df",
    },
}
_ALLOWED_CHECKPOINT_TEMPLATE_FIELDS = frozenset({"rank", "world_size"})


class RankLocalConfigurationError(RuntimeError):
    """The rank-local opt-in is present but cannot be used safely."""


class DistributedGroup(Protocol):
    def rank(self) -> int: ...

    def size(self) -> int: ...


@dataclass(frozen=True)
class RankLocalLoad:
    model: object
    checkpoint_path: Path
    config: Mapping[str, object]


@dataclass(frozen=True)
class RankLocalRuntimePreflight:
    loader_path: Path
    verify_file_hashes: bool


class RankLocalModelLoader(Protocol):
    def __call__(
        self,
        model_dir: Path,
        tensor_group: DistributedGroup,
        *,
        verify_file_hashes: bool,
    ) -> object: ...


@dataclass(frozen=True)
class PreparedRankLocalLoad:
    checkpoint_path: Path
    checkpoint_template: str
    runtime: RankLocalRuntimePreflight
    vocab_parallel_head: bool
    load_model: RankLocalModelLoader


def parse_rank_local_verify_hashes() -> bool:
    """Parse the opt-in weight-integrity flag without importing MLX."""

    value = os.environ.get(RANK_LOCAL_VERIFY_HASHES_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise RankLocalConfigurationError(
        f"{RANK_LOCAL_VERIFY_HASHES_ENV} must be exactly 0 or 1"
    )


def _render_checkpoint_path(template: str, *, rank: int, world_size: int) -> Path:
    if not template:
        raise RankLocalConfigurationError(
            f"{RANK_LOCAL_CHECKPOINT_ENV} cannot be empty"
        )
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise RankLocalConfigurationError(
            f"{RANK_LOCAL_CHECKPOINT_ENV} is not a valid path template"
        ) from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if (
            field_name not in _ALLOWED_CHECKPOINT_TEMPLATE_FIELDS
            or format_spec
            or conversion
        ):
            raise RankLocalConfigurationError(
                f"{RANK_LOCAL_CHECKPOINT_ENV} may only use {{rank}} and "
                "{world_size} placeholders without conversions or format specs"
            )
    rendered = template.format(rank=rank, world_size=world_size)
    path = Path(rendered)
    if not path.is_absolute():
        raise RankLocalConfigurationError(
            f"{RANK_LOCAL_CHECKPOINT_ENV} must resolve to an absolute path"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RankLocalConfigurationError(
            f"rank-local checkpoint does not exist: {path}"
        ) from exc
    if not resolved.is_dir():
        raise RankLocalConfigurationError(
            f"rank-local checkpoint is not a directory: {resolved}"
        )
    return resolved


def resolve_configured_rank_local_checkpoint_path(
    shard_metadata: ShardMetadata,
) -> Path | None:
    """Resolve this node's explicit rank checkpoint without loading weights.

    Download/readiness code uses the same template and assignment contract as
    the MLX loader. Pipeline and CFG shards deliberately ignore this tensor-
    only opt-in so their existing full/partial checkpoint behavior is unchanged.
    """

    checkpoint_template = os.environ.get(RANK_LOCAL_CHECKPOINT_ENV)
    if checkpoint_template is None or not isinstance(
        shard_metadata, TensorShardMetadata
    ):
        return None
    if str(shard_metadata.model_card.model_id).casefold() != (
        SUPPORTED_MODEL_ID.casefold()
    ):
        raise RankLocalConfigurationError(
            "rank-local checkpoint opt-in is pinned to "
            f"{SUPPORTED_MODEL_ID}, not {shard_metadata.model_card.model_id}"
        )
    if (
        shard_metadata.start_layer != 0
        or shard_metadata.end_layer != shard_metadata.n_layers
    ):
        raise RankLocalConfigurationError(
            "tensor-parallel rank-local checkpoints must cover every model layer"
        )
    return _render_checkpoint_path(
        checkpoint_template,
        rank=shard_metadata.device_rank,
        world_size=shard_metadata.world_size,
    )


def _load_external_module(loader_path: Path) -> object:
    module_name = "_exo_external_rank_local_loader"
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise RankLocalConfigurationError(
            f"cannot import rank-local loader from {loader_path}"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RankLocalConfigurationError(
            f"failed to import rank-local loader from {loader_path}"
        ) from exc
    return module


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_rank_local_loader_path() -> Path:
    """Resolve and authenticate the external loader without importing it."""

    configured = os.environ.get(RANK_LOCAL_LOADER_ENV)
    if configured is None or not configured:
        raise RankLocalConfigurationError(
            f"{RANK_LOCAL_LOADER_ENV} must name the audited rank_local_loader.py"
        )
    path = Path(configured)
    if not path.is_absolute():
        raise RankLocalConfigurationError(
            f"{RANK_LOCAL_LOADER_ENV} must be an absolute path"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RankLocalConfigurationError(
            f"rank-local loader does not exist: {path}"
        ) from exc
    if not resolved.is_file():
        raise RankLocalConfigurationError(
            f"rank-local loader is not a regular file: {resolved}"
        )
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != SUPPORTED_LOADER_SHA256:
        raise RankLocalConfigurationError(
            "rank-local loader SHA-256 does not match the audited loader: "
            f"expected {SUPPORTED_LOADER_SHA256}, got {actual_sha256}"
        )
    return resolved


def preflight_rank_local_runtime() -> RankLocalRuntimePreflight:
    """Validate every loader-side setting used after checkpoint readiness."""

    return RankLocalRuntimePreflight(
        loader_path=resolve_rank_local_loader_path(),
        verify_file_hashes=parse_rank_local_verify_hashes(),
    )


def preflight_configured_rank_local_model(
    shard_metadata: ShardMetadata,
    group: DistributedGroup,
) -> PreparedRankLocalLoad | None:
    """Finish every local check before the external loader's collectives."""

    checkpoint_template = os.environ.get(RANK_LOCAL_CHECKPOINT_ENV)
    if checkpoint_template is None or not isinstance(
        shard_metadata, TensorShardMetadata
    ):
        return None

    rank = group.rank()
    world_size = group.size()
    if rank != shard_metadata.device_rank:
        raise RankLocalConfigurationError(
            f"MLX group rank {rank} differs from assigned rank "
            f"{shard_metadata.device_rank}"
        )
    if world_size != shard_metadata.world_size:
        raise RankLocalConfigurationError(
            f"MLX group size {world_size} differs from assigned world size "
            f"{shard_metadata.world_size}"
        )
    checkpoint_path = resolve_configured_rank_local_checkpoint_path(shard_metadata)
    assert checkpoint_path is not None
    runtime = preflight_rank_local_runtime()
    vocab_raw = os.environ.get(RANK_LOCAL_VOCAB_PARALLEL_HEAD_ENV, "0")
    if vocab_raw not in {"0", "1"}:
        raise RankLocalConfigurationError(
            f"{RANK_LOCAL_VOCAB_PARALLEL_HEAD_ENV} must be exactly 0 or 1"
        )
    module = _load_external_module(runtime.loader_path)
    if cast(object, getattr(module, "SCHEMA", None)) != SUPPORTED_LOADER_SCHEMA:
        raise RankLocalConfigurationError(
            "external rank-local loader has an unsupported checkpoint schema"
        )
    if cast(object, getattr(module, "SOURCE_REPO", None)) != SUPPORTED_MODEL_ID:
        raise RankLocalConfigurationError(
            "external rank-local loader is pinned to a different source model"
        )
    load_model_value = cast(object, getattr(module, "load_rank_local_model", None))
    if not callable(load_model_value):
        raise RankLocalConfigurationError(
            "external rank-local loader has no callable load_rank_local_model"
        )
    return PreparedRankLocalLoad(
        checkpoint_path=checkpoint_path,
        checkpoint_template=checkpoint_template,
        runtime=runtime,
        vocab_parallel_head=vocab_raw == "1",
        load_model=cast(RankLocalModelLoader, load_model_value),
    )


def load_preflighted_rank_local_model(
    prepared: PreparedRankLocalLoad,
    group: DistributedGroup,
) -> RankLocalLoad:
    """Enter the external loader only after every rank agreed on preflight."""

    loaded: object = prepared.load_model(
        prepared.checkpoint_path,
        tensor_group=group,
        verify_file_hashes=prepared.runtime.verify_file_hashes,
    )
    if not isinstance(loaded, tuple):
        raise RankLocalConfigurationError(
            "external load_rank_local_model must return (model, config)"
        )
    loaded_tuple = cast(tuple[object, ...], loaded)
    if len(loaded_tuple) != 2:
        raise RankLocalConfigurationError(
            "external load_rank_local_model must return (model, config)"
        )
    model: object = loaded_tuple[0]
    raw_config: object = loaded_tuple[1]
    if not isinstance(raw_config, Mapping):
        raise RankLocalConfigurationError(
            "external load_rank_local_model returned a non-mapping config"
        )
    config: dict[str, object] = {}
    for key, value in cast(Mapping[object, object], raw_config).items():
        if not isinstance(key, str):
            raise RankLocalConfigurationError(
                "external rank-local config contains a non-string key"
            )
        config[key] = value
    if config.get("model_type") != "kimi_k3":
        raise RankLocalConfigurationError(
            "external rank-local loader returned a non-Kimi-K3 config"
        )
    if not isinstance(config.get("_rank_local_compatibility_transform"), Mapping):
        raise RankLocalConfigurationError(
            "external rank-local loader did not attest its compatibility transform"
        )
    return RankLocalLoad(
        model=model,
        checkpoint_path=prepared.checkpoint_path,
        config=config,
    )


def load_configured_rank_local_model(
    shard_metadata: ShardMetadata,
    group: DistributedGroup,
) -> RankLocalLoad | None:
    """Load the configured rank checkpoint, or return ``None`` when disabled.

    For tensor shards, once ``EXO_MLX_RANK_LOCAL_CHECKPOINT`` is set, every
    validation or loader error is fatal. Pipeline and CFG shards ignore this
    tensor-only opt-in, matching the path resolver.
    """

    prepared = preflight_configured_rank_local_model(shard_metadata, group)
    if prepared is None:
        return None
    return load_preflighted_rank_local_model(prepared, group)
