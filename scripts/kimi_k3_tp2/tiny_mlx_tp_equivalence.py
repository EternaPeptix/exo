#!/usr/bin/env python3
"""Two-rank MLX equivalence test for K3 rank-local TP checkpoints.

Run in two phases:

1. single-process ``prepare`` creates a tiny mixed-quantized K3 checkpoint and
   converts it with the production safetensors slicing engine;
2. ``mlx.launch -n 2 --backend ring`` runs ``distributed`` to compare normal
   full-checkpoint ``Model.shard`` against the rank-local loader for parameters,
   prefill logits/cache, and multiple cached decode steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from k3_tp_checkpoint import (
    CONTRACT_VERSION,
    MLX_LM_COMMIT,
    MLX_LM_KIMI_K3_SHA256,
    SCHEMA,
    KimiK3ShardingContract,
    SafeTensorFile,
    _tensor_manifest,
    write_rank_shard,
)
from rank_local_loader import _quantize_structure, load_rank_local_model


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Mapping) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def get_model_classes(config: dict):
    import mlx_lm.utils as utils

    getter = getattr(utils, "get_model_classes", None)
    if getter is None:
        getter = getattr(utils, "_get_classes", None)
    if getter is None:
        raise RuntimeError("mlx_lm.utils has no model-class lookup API")
    return getter(config=config)


def tiny_config() -> dict:
    # This is the upstream PR #1626 test configuration. Dimensions are all
    # divisible by TP2; group-size-32 quantization is selectively applied.
    return {
        "_synthetic_test": True,
        "model_type": "kimi_k3",
        "vocab_size": 1000,
        "num_hidden_layers": 4,
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            # TP2 + affine group-32 requires every sharded input to contain
            # an even number of complete quantization groups.
            "intermediate_size": 128,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [4],
                "num_heads": 2,
                "head_dim": 32,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
            "num_experts": 8,
            "moe_intermediate_size": 64,
            "q_lora_rank": 32,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 8,
            "v_head_dim": 32,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "routed_expert_hidden_size": 32,
            "latent_moe_use_norm": True,
            "attn_res_block_size": 2,
            "tie_word_embeddings": False,
        },
    }


def apply_tiny_mixed_quantization(model: Any, config: dict) -> None:
    """Quantize enough module families to exercise every packed TP axis."""

    quantization: dict[str, Any] = {
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
    }

    def predicate(path: str, module: Any):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith("res_proj"):
            return False
        if module.weight.shape[-1] % 32:
            return False
        if "switch_mlp" in path:
            bits = 2
        elif ".self_attn." in path:
            bits = 6
        elif path.endswith("mlp.gate"):
            bits = 8
        else:
            bits = 4
        params = {"group_size": 32, "bits": bits, "mode": "affine"}
        quantization[path] = params
        return params

    nn.quantize(
        model,
        group_size=32,
        bits=4,
        mode="affine",
        class_predicate=predicate,
    )
    config["quantization"] = quantization
    config["quantization_config"] = quantization


def prepare(work_dir: Path) -> None:
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty test dir: {work_dir}")
    full_dir = work_dir / "full"
    rank_dirs = [work_dir / "rank0", work_dir / "rank1"]
    full_dir.mkdir(parents=True, exist_ok=True)
    for path in rank_dirs:
        path.mkdir(parents=True, exist_ok=True)

    config = tiny_config()
    model_class, args_class = get_model_classes(config)
    mx.random.seed(20260729)
    model = model_class(args_class.from_dict(config))
    apply_tiny_mixed_quantization(model, config)
    model.eval()
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    full_weights = full_dir / "model.safetensors"
    mx.save_safetensors(str(full_weights), weights, metadata={"format": "mlx"})
    config_path = full_dir / "config.json"
    save_json(config_path, config)

    source_index = {
        "metadata": {
            "total_size": sum(int(value.nbytes) for value in weights.values()),
            "total_parameters": sum(int(value.size) for value in weights.values()),
        },
        "weight_map": {name: full_weights.name for name in sorted(weights)},
    }
    source_index_path = full_dir / "model.safetensors.index.json"
    save_json(source_index_path, source_index)
    config_sha = sha256_file(config_path)
    source_index_sha = sha256_file(source_index_path)

    contract = KimiK3ShardingContract(config, world_size=2)
    with SafeTensorFile(full_weights) as source:
        if set(source.tensors) != set(weights):
            raise RuntimeError("MLX save and parameter trees differ")
        for rank, rank_dir in enumerate(rank_dirs):
            plans = [contract.plan(desc, rank) for desc in source.tensors.values()]
            if any(plan is None for plan in plans):
                raise RuntimeError("tiny text model unexpectedly had excluded tensors")
            plans = [plan for plan in plans if plan is not None]
            output_path = rank_dir / "model.safetensors"
            record = write_rank_shard(
                source,
                output_path,
                plans,
                rank=rank,
                world_size=2,
                max_buffer_bytes=1 << 20,
            )
            shutil.copy2(config_path, rank_dir / "config.json")
            rank_index = {
                "metadata": {
                    "total_size": sum(plan.output_nbytes for plan in plans),
                    "tp_rank": rank,
                    "tp_world_size": 2,
                },
                "weight_map": {
                    plan.source.name: output_path.name for plan in plans
                },
            }
            save_json(rank_dir / "model.safetensors.index.json", rank_index)
            tensors = {
                plan.source.name: _tensor_manifest(plan, full_weights.name)
                for plan in plans
            }
            # Mirror the converter: the loader requires the per-file record to
            # carry its own tensors mapping (rank_local_loader._verify_manifest).
            record["tensors"] = tensors
            manifest = {
                "schema": SCHEMA,
                "complete": True,
                "source": {
                    "repo": "synthetic/tiny-kimi-k3",
                    "revision": "mlx-logits-equivalence-v1",
                    "config_sha256": config_sha,
                    "index_sha256": source_index_sha,
                },
                "runtime": {
                    "mlx_lm_commit": MLX_LM_COMMIT,
                    "mlx_lm_kimi_k3_sha256": MLX_LM_KIMI_K3_SHA256,
                },
                "tp": {
                    "rank": rank,
                    "world_size": 2,
                    "contract": CONTRACT_VERSION,
                    "contract_digest": contract.contract_digest(),
                },
                "rank_data_bytes": sum(plan.output_nbytes for plan in plans),
                "files": {full_weights.name: record},
                "tensors": tensors,
            }
            save_json(rank_dir / "tp_manifest.json", manifest)

    report = {
        "work_dir": str(work_dir),
        "full_file_bytes": full_weights.stat().st_size,
        "full_tensor_count": len(weights),
        "quantized_module_count": len(config["quantization"]) - 3,
        "rank_file_bytes": [
            (rank_dir / "model.safetensors").stat().st_size
            for rank_dir in rank_dirs
        ],
        "rank_tensor_counts": [
            len(json.loads((rank_dir / "tp_manifest.json").read_text())["tensors"])
            for rank_dir in rank_dirs
        ],
    }
    save_json(work_dir / "prepare-result.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def instantiate_full_then_shard(
    full_dir: Path, group: Any
) -> tuple[Any, dict]:
    config = json.loads((full_dir / "config.json").read_text())
    model_class, args_class = get_model_classes(config)
    model = model_class(args_class.from_dict(config))
    weights = mx.load(str(full_dir / "model.safetensors"))
    _quantize_structure(model, config, set(weights))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    model.shard(group)
    mx.eval(model.parameters())
    return model, config


def array_max_abs(left: mx.array, right: mx.array) -> float:
    left = left.astype(mx.float32)
    right = right.astype(mx.float32)
    return float(mx.max(mx.abs(left - right)).item())


def compare_parameter_trees(left: Any, right: Any) -> dict:
    left_params = dict(tree_flatten(left.parameters()))
    right_params = dict(tree_flatten(right.parameters()))
    if set(left_params) != set(right_params):
        raise AssertionError("parameter name sets differ")
    max_diff = 0.0
    mismatches: list[str] = []
    for name in sorted(left_params):
        a, b = left_params[name], right_params[name]
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            mismatches.append(name)
            continue
        mx.eval(a, b)
        diff = array_max_abs(a, b)
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            mismatches.append(name)
    if mismatches:
        raise AssertionError(
            f"parameter mismatch ({len(mismatches)}): {mismatches[:5]}"
        )
    return {"tensor_count": len(left_params), "max_abs_diff": max_diff}


def cache_snapshot(cache: list[Any]) -> tuple[list[tuple[str, mx.array]], list]:
    arrays: list[tuple[str, mx.array]] = []
    metadata: list = []
    for layer_index, layer_cache in enumerate(cache):
        state = layer_cache.state
        if not isinstance(state, (list, tuple)):
            state = [state]
        for state_index, value in enumerate(state):
            if value is not None:
                arrays.append((f"{layer_index}.{state_index}", value))
        metadata.append(
            {
                "class": type(layer_cache).__name__,
                "offset": getattr(layer_cache, "offset", None),
                "size": int(layer_cache.size()),
                "empty": bool(layer_cache.empty()),
            }
        )
    return arrays, metadata


def compare_caches(left: list[Any], right: list[Any]) -> dict:
    left_arrays, left_meta = cache_snapshot(left)
    right_arrays, right_meta = cache_snapshot(right)
    if left_meta != right_meta:
        raise AssertionError(
            f"cache metadata differs: left={left_meta}, right={right_meta}"
        )
    if [name for name, _ in left_arrays] != [name for name, _ in right_arrays]:
        raise AssertionError("cache state array sets differ")
    max_diff = 0.0
    for (name, a), (_, b) in zip(left_arrays, right_arrays):
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            raise AssertionError(f"cache shape/dtype differs at {name}")
        mx.eval(a, b)
        diff = array_max_abs(a, b)
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            raise AssertionError(f"cache differs at {name}: max_abs={diff}")
    return {
        "array_count": len(left_arrays),
        "max_abs_diff": max_diff,
        "metadata": left_meta,
    }


def compare_logits(label: str, left: mx.array, right: mx.array) -> dict:
    mx.eval(left)
    mx.eval(right)
    diff = array_max_abs(left, right)
    close = bool(mx.allclose(left, right, rtol=1e-5, atol=1e-5).item())
    if not close:
        raise AssertionError(f"{label} logits differ: max_abs={diff}")
    return {
        "label": label,
        "shape": list(left.shape),
        "max_abs_diff": diff,
        "allclose_rtol": 1e-5,
        "allclose_atol": 1e-5,
    }


def distributed(work_dir: Path) -> None:
    group = mx.distributed.init()
    rank = group.rank()
    if group.size() != 2:
        raise RuntimeError(f"test requires TP2, got {group.size()}")

    full_model, _config = instantiate_full_then_shard(work_dir / "full", group)
    rank_model, _ = load_rank_local_model(
        work_dir / f"rank{rank}",
        tensor_group=group,
        verify_file_hashes=True,
        test_only_allow_unpinned=True,
    )
    parameter_result = compare_parameter_trees(full_model, rank_model)

    full_cache = full_model.make_cache()
    rank_cache = rank_model.make_cache()
    prompt = mx.array([[1, 7, 11, 19, 23, 29]], dtype=mx.int32)
    full_logits = full_model(prompt, cache=full_cache)
    mx.eval(full_logits)
    rank_logits = rank_model(prompt, cache=rank_cache)
    mx.eval(rank_logits)
    logits_results = [compare_logits("prefill", full_logits, rank_logits)]
    cache_results = [{"label": "prefill", **compare_caches(full_cache, rank_cache)}]

    for step, token in enumerate((31, 37, 41), 1):
        token_array = mx.array([[token]], dtype=mx.int32)
        full_logits = full_model(token_array, cache=full_cache)
        mx.eval(full_logits)
        rank_logits = rank_model(token_array, cache=rank_cache)
        mx.eval(rank_logits)
        logits_results.append(
            compare_logits(f"decode-{step}", full_logits, rank_logits)
        )
        cache_results.append(
            {
                "label": f"decode-{step}",
                **compare_caches(full_cache, rank_cache),
            }
        )

    # A final collective ensures both ranks completed every comparison before
    # either process exits.
    barrier = mx.distributed.all_sum(mx.array(1, dtype=mx.int32), group=group)
    mx.eval(barrier)
    if int(barrier.item()) != 2:
        raise AssertionError("distributed completion barrier failed")

    report = {
        "rank": rank,
        "world_size": group.size(),
        "parameters": parameter_result,
        "logits": logits_results,
        "caches": cache_results,
        "status": "PASS",
    }
    save_json(work_dir / f"rank{rank}-result.json", report)
    print(f"RANK {rank} PASS {json.dumps(report, sort_keys=True)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "distributed"))
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.work_dir.resolve())
    else:
        distributed(args.work_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
