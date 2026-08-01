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
RUNTIME_MLX_LM_COMMIT = "c72ba9ce3d813f2ccfee028cf623a9cfc17243e5"
MLX_LM_KIMI_K3_SHA256 = (
    "4e550e1a1ae1801a86b472a0c02918f467806664ed3b4b3b860f7d3157141ce9"
)
MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256 = (
    "56792d995c2e7b29e7bbf5de42b9eb94fa7ecec16d22318e10073a8bb980b515"
)
MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256 = (
    "a4cc3b031052bd65679153ecfdd4f8197920a0196aa433c15963a3fa9803c7f3"
)
MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256 = (
    "ba62f6a1d375cb873c0e8fee8a08cc87554f45953c2c49f4fb869824ab1f6fa6"
)
MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256 = (
    "038a79648ff9bb27faf8c54c551e0ffcc0c19b129105d1affad1ed3c4b48486f"
)
MLX_LM_KIMI_K3_TUNED_GATHER_QMV_SHA256 = (
    "ec702446d3b3fe72cd95b0ee24497de54bcfde1ea2de9f78bad96e2f98ac4de2"
)
SOURCE_CONFIG_SHA256 = (
    "d041003554810a367bb600d18733976bdd21041bb46e75cc1e27c7b15fe034d0"
)
SOURCE_INDEX_SHA256 = "ac65bcb3cd9e07cab3e7942ff455dde33879a9e02211bae40938e22fc204ae09"
SCHEMA = "k3-rank-local-tp/v1"
CONTRACT_VERSION = "mlx-lm-kimi-k3-shard@7d505c2"
CONTRACT_DIGEST = "1b7fdf1b28433fb08fff7e0e26a7bccc2ca0fcd51f29498ab611892c9fc48da5"
DTYPE_FIX_CONTRACT = "kernelpool-k3-fp32-norm-conv-to-bf16/v1"
DTYPE_FIX_TENSOR_COUNT = 138
DTYPE_FIX_KDA_LAYER_COUNT = 69
DTYPE_FIX_TP2_RANK_ELEMENTS = 5_096_064


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


def _verify_runtime_source() -> None:
    """Fail closed if the pinned execution-time K3 source is not imported."""

    from mlx_lm.models import (
        kimi_k3,
        kimi_k3_derived_bias,
        kimi_k3_fused_down_reduce,
        kimi_k3_fused_expert,
        kimi_k3_fused_switch_glu,
        kimi_k3_tuned_gather_qmv,
    )

    pinned_sources = (
        (kimi_k3, "kimi_k3.py", MLX_LM_KIMI_K3_SHA256),
        (
            kimi_k3_derived_bias,
            "kimi_k3_derived_bias.py",
            MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256,
        ),
        (
            kimi_k3_fused_expert,
            "kimi_k3_fused_expert.py",
            MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256,
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
            kimi_k3_tuned_gather_qmv,
            "kimi_k3_tuned_gather_qmv.py",
            MLX_LM_KIMI_K3_TUNED_GATHER_QMV_SHA256,
        ),
    )
    for module, filename, expected in pinned_sources:
        source = Path(inspect.getfile(module)).resolve()
        actual = _sha256_file(source)
        if actual != expected:
            raise RankLocalLoadError(
                f"mlx_lm.models.{filename} does not match the execution runtime "
                f"pin: expected {expected}, got {actual} at {source}. "
                f"Install mlx-lm commit {RUNTIME_MLX_LM_COMMIT} or audit and "
                "update the execution runtime pin."
            )


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

    File checksums are optional because hashing ~400 GB adds a complete disk
    pass.  File sizes, manifest/config pins, rank/world, model source hash, and
    strict parameter names/shapes are always checked.
    """

    vocab_parallel_head = _strict_env_flag("EXO_MLX_K3_VOCAB_PARALLEL_HEAD")

    import mlx.core as mx
    import mlx_lm.utils as mlx_lm_utils
    from mlx.utils import tree_flatten

    load_config = mlx_lm_utils.load_config
    get_model_classes = getattr(mlx_lm_utils, "get_model_classes", None)
    if get_model_classes is None:
        get_model_classes = getattr(mlx_lm_utils, "_get_classes", None)
    if get_model_classes is None:
        raise RankLocalLoadError(
            "installed mlx_lm.utils exposes neither get_model_classes nor _get_classes"
        )

    model_dir = Path(model_dir).resolve()
    group = tensor_group or mx.distributed.init()
    _verify_runtime_source()
    manifest = _verify_manifest(
        model_dir,
        group,
        verify_file_hashes=verify_file_hashes,
        test_only_allow_unpinned=test_only_allow_unpinned,
    )
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

    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
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
