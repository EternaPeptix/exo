"""MLX-LM loader for checkpoints pre-sliced by ``k3_tp_checkpoint.py``.

Stock ``mlx_lm.utils.sharded_load`` downloads and loads the full checkpoint,
then calls ``model.shard``.  A rank-local checkpoint must reverse the last two
operations:

1. instantiate and quantize the full model structure;
2. call the pinned Kimi K3 ``Model.shard``;
3. load the already-sliced local weights.

This module is intended to be copied beside the deployment launcher and
imported under the mlx-lm PR #1626 environment.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional

SOURCE_REPO = "kernelpool/Kimi-K3-2bit-UVMAX"
SOURCE_REVISION = "edb5113218df612f4a92f95145680f3f8eacd375"
EFFECTIVE_SOURCE_REVISION = "2f7de449f18498c47fd32485566a611e66ba80ae"
MLX_LM_COMMIT = "7d505c285b801108a52c23353c7fb6af07204717"
CHECKPOINT_MLX_LM_KIMI_K3_SHA256 = (
    "3dd2e9db585190bca118d5812bcb5b103d1e7c6ec12187b20351992fed7e63cc"
)
RUNTIME_MLX_LM_COMMIT = "591e11093b55b3b03f7cbc5018cd3b7d47abba4f"
RUNTIME_MLX_LM_DSPARK_0731_COMMIT = "e041c6ac9296bd9a73bd4c8b1f4f0d4b55b62e35"
MLX_LM_KIMI_K3_SHA256 = (
    "c20fe4020bc2a830403b98a2eacac7f1fa90cc2c5406a6ed3cfd2f1ff7a9107f"
)
MLX_LM_CACHE_SHA256 = "a83a454942864d6430b0d8f28c716593be4d9286c2a683083b6708df97e04e78"
MLX_LM_GENERATE_SHA256 = (
    "096f24553953a90f8e331cb7c7415a75636db4eec8a5bd159b6ad3e93cc4e369"
)
MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE_SHA256 = (
    "beaadba191fb762d70c2fe2d9d41f53f06984b857db02b8b7fefcd93ee15a925"
)
MLX_LM_KIMI_K3_DSPARK_SHA256 = (
    "be221a4dde09ec97011a961a4f2d7de1f5f0327af706395967166f710f968021"
)
MLX_LM_KIMI_K3_DSPARK_0731_SHA256 = (
    "5aed25bdb2e5971d89dc264cab85e1fb4c97a05d5be41e6db26eddbf6152c1b1"
)
MLX_LM_GATED_DELTA_SHA256 = (
    "44aef2791ed0cd5cfb84e31ef00cb4df3d40ae184e6c6852b7f0dba7406d2f78"
)
MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256 = (
    "d9025e323239a27dfb5869530444fbe428d6f704eac45063d184ad006a4161c5"
)
MLX_LM_SWITCH_LAYERS_SHA256 = (
    "793679ed80ab4051858ac76ab32b9b9abcabe21eb02bac7c3fde75b6c15de840"
)
MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256 = (
    "584b4c94a623b583dae470a62851a8c882580dc7cb1519916a4019a631e7a3e5"
)
MLX_LM_KIMI_K3_WIDTH4_FUSED_EXPERT_SHA256 = (
    "5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832"
)
MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256 = (
    "9deca87fa4deeae2a320abe4fb50ef49ab245ae070fb123110d0c9243ed9cb85"
)
MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256 = (
    "2401fa7c58ad095e0b48bd617387df8fd30403b4e11416c56620879b0044c7b7"
)
MLX_LM_KIMI_K3_FUSED_ROUTER_SHA256 = (
    "14a7cb54eb3585c992504c109db8bed45eb6ec521ae37e680eb601c419ed8ed1"
)
MLX_LM_KIMI_K3_TUNED_GATHER_QMV_SHA256 = (
    "ec702446d3b3fe72cd95b0ee24497de54bcfde1ea2de9f78bad96e2f98ac4de2"
)
MLX_LM_KIMI_K3_PACKED_MOE_FRONT_SHA256 = (
    "82076bf9c0098f2fc72a0434a5482e72a6e022fc982574f05867dcce32645435"
)
SOURCE_CONFIG_SHA256 = (
    "d041003554810a367bb600d18733976bdd21041bb46e75cc1e27c7b15fe034d0"
)
SOURCE_INDEX_SHA256 = "ac65bcb3cd9e07cab3e7942ff455dde33879a9e02211bae40938e22fc204ae09"
SCHEMA = "k3-rank-local-tp/v2"
CONTRACT_VERSION = "mlx-lm-kimi-k3-shard@7d505c2"
CONTRACT_DIGEST = "1b7fdf1b28433fb08fff7e0e26a7bccc2ca0fcd51f29498ab611892c9fc48da5"
DTYPE_FIX_CONTRACT = "kernelpool-k3-fp32-norm-conv-to-bf16/v1"
DTYPE_FIX_TENSOR_COUNT = 138
DTYPE_FIX_KDA_LAYER_COUNT = 69
DTYPE_FIX_TP2_RANK_ELEMENTS = 5_096_064

# Exact non-weight inventory from SOURCE_REPO@SOURCE_REVISION.  This must stay
# aligned with ``k3_tp_checkpoint.py`` and EXO readiness validation.  Several
# entries execute under ``trust_remote_code=True``; names alone are not an
# authentication boundary.
PINNED_METADATA_FILES: dict[str, dict[str, int | str]] = {
    "README.md": {
        "bytes": 1923,
        "sha256": "d0d7a4d1a5af37c542594449d2ce893b9e3c33ccb031afb71cf13e4f23a5349d",
    },
    "added_tokens.json": {
        "bytes": 200,
        "sha256": "27373c2f39a52c87e674caf7e9604ec6756c68c5f8d5f140657299048b6ab8ba",
    },
    "config.json": {"bytes": 459349, "sha256": SOURCE_CONFIG_SHA256},
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
ALLOWED_METADATA_FILENAMES = frozenset(PINNED_METADATA_FILES)
INTERNAL_CHECKPOINT_FILENAMES = frozenset(
    {"model.safetensors.index.json", "tp_manifest.json"}
)


class RankLocalLoadError(RuntimeError):
    pass


def _strict_env_flag(name: str) -> bool:
    raw = os.environ.get(name, "0")
    if raw not in {"0", "1"}:
        raise RankLocalLoadError(f"{name} must be 0 or 1")
    return raw == "1"


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _safe_checkpoint_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RankLocalLoadError(f"unsafe checkpoint filename {value!r}")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise RankLocalLoadError(f"unsafe checkpoint filename {value!r}")
    return path.as_posix()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise RankLocalLoadError(f"{path}: expected a JSON object")
    return value


def _canonical_metadata_contract(
    metadata_files: Mapping[str, Mapping[str, int | str]],
) -> str:
    return hashlib.sha256(
        json.dumps(
            metadata_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _verify_metadata_contract(
    model_dir: Path,
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    weight_filenames: set[str],
) -> str:
    raw_metadata = manifest.get("metadata_files")
    if not isinstance(raw_metadata, Mapping) or not raw_metadata:
        raise RankLocalLoadError("rank-local manifest has no metadata_files")

    normalized: dict[str, dict[str, int | str]] = {}
    for raw_name, raw_record in raw_metadata.items():
        if not isinstance(raw_name, str) or raw_name not in ALLOWED_METADATA_FILENAMES:
            raise RankLocalLoadError(
                f"rank-local manifest contains unallowlisted metadata {raw_name!r}"
            )
        name = _safe_checkpoint_relative(raw_name)
        if name != raw_name or "/" in name or not isinstance(raw_record, Mapping):
            raise RankLocalLoadError(
                f"malformed rank-local metadata record for {raw_name!r}"
            )
        if set(raw_record) != {"bytes", "sha256"}:
            raise RankLocalLoadError(f"malformed rank-local metadata record for {name}")
        byte_count = raw_record.get("bytes")
        checksum = raw_record.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(checksum, str)
            or SHA256_RE.fullmatch(checksum) is None
        ):
            raise RankLocalLoadError(f"malformed rank-local metadata record for {name}")
        path = model_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != byte_count:
            raise RankLocalLoadError(f"missing or truncated metadata file {path}")
        if _sha256_file(path) != checksum:
            raise RankLocalLoadError(f"metadata checksum mismatch: {path}")
        normalized[name] = {"bytes": byte_count, "sha256": checksum}

    if normalized != PINNED_METADATA_FILES:
        raise RankLocalLoadError(
            "rank-local metadata differs from the authenticated source contract"
        )
    if normalized["config.json"]["sha256"] != source.get("config_sha256"):
        raise RankLocalLoadError(
            "metadata config hash differs from the source manifest"
        )

    expected_digest = manifest.get("metadata_contract_sha256")
    actual_digest = _canonical_metadata_contract(normalized)
    if (
        not isinstance(expected_digest, str)
        or SHA256_RE.fullmatch(expected_digest) is None
        or expected_digest != actual_digest
    ):
        raise RankLocalLoadError("rank-local metadata contract digest mismatch")

    expected_entries = (
        set(normalized) | weight_filenames | set(INTERNAL_CHECKPOINT_FILENAMES)
    )
    actual_entries: set[str] = set()
    for child in model_dir.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RankLocalLoadError(
                f"rank-local checkpoint contains a non-regular entry: {child}"
            )
        actual_entries.add(child.name)
    if actual_entries != expected_entries:
        extra = sorted(actual_entries - expected_entries)
        missing_entries = sorted(expected_entries - actual_entries)
        raise RankLocalLoadError(
            "rank-local checkpoint file inventory differs from the signed "
            f"manifest: extra={extra}, missing={missing_entries}"
        )
    return actual_digest


def _error_fingerprint(error: BaseException | None) -> int:
    if error is None:
        return 0
    digest = hashlib.sha256(
        f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _all_gather_rows(mx: Any, group: Any, row: tuple[int, ...]) -> list[list[int]]:
    gathered = mx.distributed.all_gather(
        mx.array(row, dtype=mx.int32),
        group=group,
    )
    mx.eval(gathered)
    values = list(gathered.tolist())
    expected = group.size() * len(row)
    if len(values) != expected:
        raise RankLocalLoadError(
            f"rank agreement returned {len(values)} values, expected {expected}"
        )
    return [
        [int(value) for value in values[offset : offset + len(row)]]
        for offset in range(0, len(values), len(row))
    ]


def _agree_local_validation(
    mx: Any,
    group: Any,
    label: str,
    local_error: BaseException | None,
) -> None:
    rows = _all_gather_rows(
        mx,
        group,
        (int(local_error is None), _error_fingerprint(local_error)),
    )
    if all(row[0] == 1 and row[1] == 0 for row in rows):
        return
    if local_error is not None:
        raise RankLocalLoadError(
            f"{label} failed locally: {local_error}"
        ) from local_error
    failures = [
        f"rank {rank} fingerprint {row[1]}"
        for rank, row in enumerate(rows)
        if row[0] != 1 or row[1] != 0
    ]
    raise RankLocalLoadError(f"{label} failed on peer: {', '.join(failures)}")


def _agree_sha256_contract(
    mx: Any,
    group: Any,
    digest: str,
    *,
    label: str,
) -> None:
    if SHA256_RE.fullmatch(digest) is None:
        raise RankLocalLoadError(f"invalid local {label} digest")
    local_bytes = bytes.fromhex(digest)
    rows = _all_gather_rows(mx, group, tuple(local_bytes))
    if any(row != rows[0] for row in rows[1:]):
        raise RankLocalLoadError(f"{label} differs across tensor-parallel ranks")


def _agree_metadata_contract(mx: Any, group: Any, digest: str) -> None:
    _agree_sha256_contract(
        mx,
        group,
        digest,
        label="rank-local metadata contract",
    )


def _agree_dspark_runtime_source(mx: Any, group: Any, digest: str) -> None:
    _agree_sha256_contract(
        mx,
        group,
        digest,
        label="DSpark execution runtime source",
    )


def _verify_runtime_source() -> str:
    """Fail closed if the pinned execution-time K3 source is not imported."""

    from mlx_lm import generate
    from mlx_lm.models import (
        cache,
        gated_delta,
        kimi_k3,
        kimi_k3_derived_bias,
        kimi_k3_dspark,
        kimi_k3_fused_down_reduce,
        kimi_k3_fused_expert,
        kimi_k3_fused_router,
        kimi_k3_fused_switch_glu,
        kimi_k3_packed_moe_front,
        kimi_k3_prefill_route_combine,
        kimi_k3_tuned_gather_qmv,
        kimi_k3_width4_fused_expert,
        switch_layers,
    )

    pinned_sources = (
        (kimi_k3, "models/kimi_k3.py", MLX_LM_KIMI_K3_SHA256),
        (cache, "models/cache.py", MLX_LM_CACHE_SHA256),
        (generate, "generate.py", MLX_LM_GENERATE_SHA256),
        (gated_delta, "gated_delta.py", MLX_LM_GATED_DELTA_SHA256),
        (
            kimi_k3_derived_bias,
            "kimi_k3_derived_bias.py",
            MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256,
        ),
        (switch_layers, "switch_layers.py", MLX_LM_SWITCH_LAYERS_SHA256),
        (
            kimi_k3_fused_expert,
            "kimi_k3_fused_expert.py",
            MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256,
        ),
        (
            kimi_k3_width4_fused_expert,
            "kimi_k3_width4_fused_expert.py",
            MLX_LM_KIMI_K3_WIDTH4_FUSED_EXPERT_SHA256,
        ),
        (
            kimi_k3_fused_switch_glu,
            "kimi_k3_fused_switch_glu.py",
            MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256,
        ),
        (
            kimi_k3_fused_down_reduce,
            "kimi_k3_fused_down_reduce.py",
            MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256,
        ),
        (
            kimi_k3_fused_router,
            "kimi_k3_fused_router.py",
            MLX_LM_KIMI_K3_FUSED_ROUTER_SHA256,
        ),
        (
            kimi_k3_tuned_gather_qmv,
            "kimi_k3_tuned_gather_qmv.py",
            MLX_LM_KIMI_K3_TUNED_GATHER_QMV_SHA256,
        ),
        (
            kimi_k3_packed_moe_front,
            "kimi_k3_packed_moe_front.py",
            MLX_LM_KIMI_K3_PACKED_MOE_FRONT_SHA256,
        ),
        (
            kimi_k3_prefill_route_combine,
            "kimi_k3_prefill_route_combine.py",
            MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE_SHA256,
        ),
    )
    for module, filename, expected in pinned_sources:
        source = Path(inspect.getfile(module)).resolve()
        actual = _sha256_file(source)
        if actual != expected:
            raise RankLocalLoadError(
                f"mlx_lm/{filename} does not match the execution runtime "
                f"pin: expected {expected}, got {actual} at {source}. "
                f"Stage mlx-lm runtime {RUNTIME_MLX_LM_COMMIT} or audit and "
                "update the execution runtime pin."
            )

    dspark_source = Path(inspect.getfile(kimi_k3_dspark)).resolve()
    dspark_actual = _sha256_file(dspark_source)
    dspark_allowed = {
        MLX_LM_KIMI_K3_DSPARK_SHA256,
        MLX_LM_KIMI_K3_DSPARK_0731_SHA256,
    }
    if dspark_actual not in dspark_allowed:
        raise RankLocalLoadError(
            "mlx_lm/kimi_k3_dspark.py does not match either audited execution "
            f"runtime pin: got {dspark_actual} at {dspark_source}. Stage "
            f"mlx-lm runtime {RUNTIME_MLX_LM_COMMIT} or "
            f"{RUNTIME_MLX_LM_DSPARK_0731_COMMIT}."
        )
    return dspark_actual


def _prepare_execution_runtime(model_dir: str | Path, mlx_lm_utils: Any) -> tuple:
    """Perform every fallible local operation before the first agreement."""

    from mlx.utils import tree_flatten

    checkpoint_path = Path(model_dir)
    load_config = getattr(mlx_lm_utils, "load_config", None)
    if not callable(load_config):
        raise RankLocalLoadError("installed mlx_lm.utils exposes no load_config")
    get_model_classes = getattr(mlx_lm_utils, "get_model_classes", None)
    if get_model_classes is None:
        get_model_classes = getattr(mlx_lm_utils, "_get_classes", None)
    if not callable(get_model_classes):
        raise RankLocalLoadError(
            "installed mlx_lm.utils exposes neither get_model_classes nor _get_classes"
        )
    if not callable(tree_flatten):
        raise RankLocalLoadError("installed mlx.utils exposes no tree_flatten")
    vocab_parallel_head = _strict_env_flag("EXO_MLX_K3_VOCAB_PARALLEL_HEAD")
    dspark_runtime_sha256 = _verify_runtime_source()
    return (
        checkpoint_path,
        load_config,
        get_model_classes,
        tree_flatten,
        vocab_parallel_head,
        dspark_runtime_sha256,
    )


def _strict_load_release_and_evaluate(
    model: Any,
    weights: dict[str, Any],
    mx_module: Any,
) -> None:
    """Strictly install weights, then release raw checkpoint allocations.

    ``load_weights`` must complete before any references are released. Once it
    succeeds, the model owns its parameters, so retaining the rank-local shard
    dictionary only duplicates references and delays allocator reclamation
    during the first full-model evaluation.
    """

    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    mx_module.clear_cache()
    model.eval()
    mx_module.eval(model.parameters())


def _verify_manifest(
    model_dir: Path,
    group: Any,
    *,
    verify_file_hashes: bool,
    test_only_allow_unpinned: bool,
) -> dict:
    manifest_path = model_dir / "tp_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RankLocalLoadError("missing regular tp_manifest.json")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise RankLocalLoadError("rank-local checkpoint manifest is incomplete")
    source = manifest.get("source", {})
    runtime = manifest.get("runtime", {})
    tp = manifest.get("tp", {})
    if not test_only_allow_unpinned:
        expected = {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
        }
        if any(source.get(k) != v for k, v in expected.items()):
            raise RankLocalLoadError("manifest source does not match pinned checkpoint")
        if source.get("config_sha256") != SOURCE_CONFIG_SHA256:
            raise RankLocalLoadError("manifest config hash is not the pinned config")
        if source.get("index_sha256") != SOURCE_INDEX_SHA256:
            raise RankLocalLoadError("manifest source index hash is not pinned")
    if runtime.get("mlx_lm_commit") != MLX_LM_COMMIT:
        raise RankLocalLoadError(
            "manifest mlx-lm commit does not match the checkpoint converter"
        )
    if runtime.get("mlx_lm_kimi_k3_sha256") != CHECKPOINT_MLX_LM_KIMI_K3_SHA256:
        raise RankLocalLoadError(
            "manifest Kimi K3 hash does not match the checkpoint converter"
        )
    if int(tp.get("world_size", -1)) != group.size():
        raise RankLocalLoadError(
            f"manifest TP{tp.get('world_size')} loaded with TP{group.size()}"
        )
    if int(tp.get("rank", -1)) != group.rank():
        raise RankLocalLoadError(
            f"manifest rank {tp.get('rank')} loaded on group rank {group.rank()}"
        )
    if not test_only_allow_unpinned:
        if tp.get("contract") != CONTRACT_VERSION:
            raise RankLocalLoadError("manifest uses a different sharding contract")
        if tp.get("contract_digest") != CONTRACT_DIGEST:
            raise RankLocalLoadError("manifest sharding contract digest is not pinned")
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise RankLocalLoadError("missing regular config.json")
    if _sha256_file(config_path) != source.get("config_sha256"):
        raise RankLocalLoadError("config.json hash differs from source manifest")
    if index_path.is_symlink() or not index_path.is_file():
        raise RankLocalLoadError(
            "missing regular rank-local model.safetensors.index.json"
        )
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    files = manifest.get("files")
    tensors = manifest.get("tensors")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RankLocalLoadError("rank-local index has no weight_map")
    if not isinstance(files, dict) or not files:
        raise RankLocalLoadError("rank-local manifest has no files")
    if not isinstance(tensors, dict) or not tensors:
        raise RankLocalLoadError("rank-local manifest has no tensors")
    if not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise RankLocalLoadError("rank-local weight_map must contain strings")
    if set(weight_map) != set(tensors):
        raise RankLocalLoadError("rank-local index and manifest tensor sets differ")

    normalized_files: dict[str, Mapping[str, Any]] = {}
    for source_file, record in files.items():
        relative = _safe_checkpoint_relative(source_file)
        if relative != source_file or not isinstance(record, Mapping):
            raise RankLocalLoadError("malformed rank-local file record")
        record_name = _safe_checkpoint_relative(record.get("name"))
        if record_name != source_file:
            raise RankLocalLoadError(
                f"rank file record/name mismatch for {source_file}"
            )
        byte_count = record.get("bytes")
        tensor_count = record.get("tensor_count")
        checksum = record.get("sha256")
        file_tensors = record.get("tensors")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or isinstance(tensor_count, bool)
            or not isinstance(tensor_count, int)
            or tensor_count <= 0
            or not isinstance(checksum, str)
            or SHA256_RE.fullmatch(checksum) is None
            or not isinstance(file_tensors, dict)
        ):
            raise RankLocalLoadError(
                f"malformed rank-local file record for {source_file}"
            )
        expected_names = {
            name for name, filename in weight_map.items() if filename == source_file
        }
        if (
            set(file_tensors) != expected_names
            or tensor_count != len(expected_names)
            or any(file_tensors[name] != tensors[name] for name in expected_names)
        ):
            raise RankLocalLoadError(
                f"rank file tensor inventory mismatch for {source_file}"
            )
        normalized_files[source_file] = record

    index_files = {
        _safe_checkpoint_relative(filename) for filename in weight_map.values()
    }
    if index_files != set(normalized_files):
        raise RankLocalLoadError("rank-local index and manifest file sets differ")

    rank_data_bytes = 0
    for tensor_name, record in tensors.items():
        if not isinstance(tensor_name, str) or not isinstance(record, Mapping):
            raise RankLocalLoadError("malformed rank-local tensor record")
        source_file = _safe_checkpoint_relative(record.get("source_file"))
        byte_count = record.get("bytes")
        rank_shape = record.get("rank_shape")
        if (
            source_file != weight_map[tensor_name]
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(rank_shape, list)
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in rank_shape
            )
        ):
            raise RankLocalLoadError(
                f"malformed rank-local tensor record for {tensor_name}"
            )
        rank_data_bytes += byte_count
    index_metadata = index.get("metadata")
    if (
        not isinstance(index_metadata, dict)
        or index_metadata.get("total_size") != rank_data_bytes
        or manifest.get("rank_data_bytes") != rank_data_bytes
    ):
        raise RankLocalLoadError(
            "rank-local byte totals disagree across index and manifest"
        )

    for source_file, record in normalized_files.items():
        path = model_dir / source_file
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
        ):
            raise RankLocalLoadError(f"missing or truncated rank file {path}")
        if verify_file_hashes and _sha256_file(path) != record["sha256"]:
            raise RankLocalLoadError(f"rank file checksum mismatch: {path}")
    _verify_metadata_contract(
        model_dir,
        manifest,
        source,
        set(normalized_files),
    )
    return manifest


def _quantize_structure(model: Any, config: Mapping, weight_names: set[str]) -> None:
    """Mirror the fine-grained quantization branch in mlx_lm.utils.load_model."""

    import mlx.nn as nn

    quantization = config.get("quantization")
    if quantization is None:
        raise RankLocalLoadError("pinned UVMAX config has no quantization map")

    def class_predicate(path: str, module: Any):
        if path in quantization:
            return quantization[path]
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weight_names

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def _dtype_fix_tensor_names(config: Mapping[str, Any]) -> list[str]:
    """Return the exact tensors changed by upstream revision ``2f7de44``."""

    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise RankLocalLoadError("K3 dtype fix requires text_config")
    linear_config = text_config.get("linear_attn_config")
    if not isinstance(linear_config, Mapping):
        raise RankLocalLoadError("K3 dtype fix requires linear_attn_config")
    kda_layers = linear_config.get("kda_layers")
    if (
        not isinstance(kda_layers, list)
        or len(kda_layers) != DTYPE_FIX_KDA_LAYER_COUNT
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer <= 0
            for layer in kda_layers
        )
        or len(set(kda_layers)) != len(kda_layers)
    ):
        raise RankLocalLoadError(
            "K3 dtype fix expected exactly 69 unique one-based KDA layers"
        )

    names: list[str] = []
    for one_based_layer in sorted(kda_layers):
        prefix = f"language_model.model.layers.{one_based_layer - 1}.self_attn"
        names.extend(
            [
                f"{prefix}.o_norm.weight",
                f"{prefix}.qkv_conv.conv.weight",
            ]
        )
    if len(names) != DTYPE_FIX_TENSOR_COUNT:
        raise RankLocalLoadError(
            f"K3 dtype fix expected {DTYPE_FIX_TENSOR_COUNT} tensors"
        )
    return sorted(names)


def _apply_upstream_dtype_fix(
    weights: dict[str, Any],
    config: Mapping[str, Any],
    *,
    float32_dtype: Any,
    bfloat16_dtype: Any,
) -> dict[str, Any]:
    """Reproduce KernelPool's F32→BF16 correction without redownloading shards.

    The pinned local rank checkpoints were produced from ``edb5113``, whose
    KDA output norms and short-convolution weights were accidentally F32.
    Upstream revision ``2f7de44`` changes only these 138 tensor dtypes.  Cast
    the already rank-sliced arrays before loading them into the model and fail
    closed if the pinned checkpoint does not have the exact expected surface.
    """

    names = _dtype_fix_tensor_names(config)
    missing = [name for name in names if name not in weights]
    if missing:
        raise RankLocalLoadError(f"K3 dtype fix tensors are missing: {missing[:5]}")
    wrong_dtype = [name for name in names if weights[name].dtype != float32_dtype]
    if wrong_dtype:
        raise RankLocalLoadError(
            f"K3 dtype fix expected source tensors to be float32: {wrong_dtype[:5]}"
        )

    source_elements = 0
    category_counts = {"o_norm.weight": 0, "qkv_conv.conv.weight": 0}
    for name in names:
        value = weights[name]
        source_elements += int(value.size)
        if name.endswith(".o_norm.weight"):
            category_counts["o_norm.weight"] += 1
        elif name.endswith(".qkv_conv.conv.weight"):
            category_counts["qkv_conv.conv.weight"] += 1
        weights[name] = value.astype(bfloat16_dtype)
    if source_elements != DTYPE_FIX_TP2_RANK_ELEMENTS:
        raise RankLocalLoadError(
            "K3 TP2 dtype fix element count differs from the audited contract: "
            f"expected {DTYPE_FIX_TP2_RANK_ELEMENTS}, got {source_elements}"
        )

    names_sha256 = hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()
    return {
        "contract": DTYPE_FIX_CONTRACT,
        "source_revision": SOURCE_REVISION,
        "effective_source_revision": EFFECTIVE_SOURCE_REVISION,
        "operation": "astype(bfloat16)",
        "source_dtype": "float32",
        "target_dtype": "bfloat16",
        "tensor_count": len(names),
        "kda_layer_count": DTYPE_FIX_KDA_LAYER_COUNT,
        "category_counts": category_counts,
        "rank_elements_cast": source_elements,
        "tensor_names_sha256": names_sha256,
    }


def load_rank_local_model(
    model_dir: str | Path,
    tensor_group: Optional[Any] = None,
    *,
    verify_file_hashes: bool = False,
    test_only_allow_unpinned: bool = False,
):
    """Load a rank-local K3 TP checkpoint without materializing full weights.

    Weight checksums are optional because hashing ~400 GB adds a complete disk
    pass.  Every small metadata file is always hashed and agreed across ranks;
    weight sizes, manifest/config pins, rank/world, model source hash, and
    strict parameter names/shapes are always checked.
    """

    import mlx.core as mx
    import mlx_lm.utils as mlx_lm_utils

    group = tensor_group or mx.distributed.init()
    execution_runtime: tuple | None = None
    runtime_error: Exception | None = None
    try:
        execution_runtime = _prepare_execution_runtime(model_dir, mlx_lm_utils)
    except Exception as exc:
        runtime_error = exc
    _agree_local_validation(mx, group, "execution runtime validation", runtime_error)
    if execution_runtime is None:
        raise RankLocalLoadError("runtime validation produced no execution runtime")
    (
        model_dir,
        load_config,
        get_model_classes,
        tree_flatten,
        vocab_parallel_head,
        dspark_runtime_sha256,
    ) = execution_runtime
    _agree_dspark_runtime_source(mx, group, dspark_runtime_sha256)

    manifest: dict[str, Any] | None = None
    manifest_error: Exception | None = None
    try:
        manifest = _verify_manifest(
            model_dir,
            group,
            verify_file_hashes=verify_file_hashes,
            test_only_allow_unpinned=test_only_allow_unpinned,
        )
    except Exception as exc:
        manifest_error = exc
    _agree_local_validation(mx, group, "checkpoint validation", manifest_error)
    if manifest is None:
        raise RankLocalLoadError("checkpoint validation produced no manifest")
    metadata_digest = manifest.get("metadata_contract_sha256")
    if not isinstance(metadata_digest, str):
        raise RankLocalLoadError("checkpoint has no metadata contract digest")
    _agree_metadata_contract(mx, group, metadata_digest)

    config = load_config(model_dir)
    if config.get("model_type") != "kimi_k3":
        raise RankLocalLoadError("rank-local config is not Kimi K3")

    model_class, args_class = get_model_classes(config=config)
    model = model_class(args_class.from_dict(config))
    expected_names = set(manifest["tensors"])
    _quantize_structure(model, config, expected_names)

    # This is the essential inversion relative to stock sharded_load: shard
    # the empty model structure before loading pre-sliced arrays.
    model.shard(group)
    parameter_shapes = {
        name: tuple(value.shape) for name, value in tree_flatten(model.parameters())
    }
    expected_shapes = {
        name: tuple(record["rank_shape"])
        for name, record in manifest["tensors"].items()
    }
    missing = sorted(set(parameter_shapes) - set(expected_shapes))
    extra = sorted(set(expected_shapes) - set(parameter_shapes))
    mismatched = sorted(
        name
        for name in set(parameter_shapes) & set(expected_shapes)
        if parameter_shapes[name] != expected_shapes[name]
    )
    if missing or extra or mismatched:
        raise RankLocalLoadError(
            "rank-local parameter contract mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}, "
            f"shape_mismatch={mismatched[:5]}"
        )

    index = _load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    if set(weight_map) != expected_names:
        raise RankLocalLoadError("rank-local index and manifest tensor sets differ")

    weights: dict[str, Any] = {}
    for filename in sorted(set(weight_map.values())):
        loaded = mx.load(str(model_dir / filename))
        overlap = set(weights) & set(loaded)
        if overlap:
            raise RankLocalLoadError(
                f"duplicate tensors across rank files: {sorted(overlap)[:5]}"
            )
        weights.update(loaded)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    dtype_fix_audit = _apply_upstream_dtype_fix(
        weights,
        config,
        float32_dtype=mx.float32,
        bfloat16_dtype=mx.bfloat16,
    )

    _strict_load_release_and_evaluate(model, weights, mx)
    if vocab_parallel_head:
        shard_vocab_head = getattr(model, "shard_vocab_head", None)
        if shard_vocab_head is None:
            raise RankLocalLoadError("K3 runtime does not expose shard_vocab_head")
        shard_vocab_head(group)
        mx.eval(model.parameters())
        mx.clear_cache()
    config["_rank_local_vocab_parallel_head"] = {
        "enabled": vocab_parallel_head,
        "world_size": group.size(),
    }
    config["_rank_local_compatibility_transform"] = dtype_fix_audit
    return model, config


def load_rank_local(
    model_dir: str | Path,
    tensor_group: Optional[Any] = None,
    *,
    tokenizer_config: Optional[dict] = None,
    verify_file_hashes: bool = False,
    return_config: bool = False,
):
    """Load model plus tokenizer with the same shape as mlx_lm.sharded_load."""

    from mlx_lm.utils import load_tokenizer

    model, config = load_rank_local_model(
        model_dir,
        tensor_group,
        verify_file_hashes=verify_file_hashes,
    )
    tokenizer = load_tokenizer(
        Path(model_dir),
        tokenizer_config or {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id"),
    )
    if return_config:
        return model, tokenizer, config
    return model, tokenizer
