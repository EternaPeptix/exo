#!/usr/bin/env python3
"""Localize Kimi K3 TP2 multi-token target-verification divergence.

This is a companion diagnostic, not an alternative correctness gate.  It
does not change the ``k3-tp2-target-verification/v3`` PASS/FAIL contract.
Instead, for one selected T>1 width it records:

* target-versus-sequential logits at every token position;
* final normalized hidden states and every decoder-layer output;
* target-final versus sequential-final mixed KDA/MLA cache state;
* one subsequent T=1 rollout from each final cache, followed by another cache
  comparison.

Completed artifacts use non-promotional ``COMPLETE_*`` statuses.  Any setup,
capture, topology, runtime, or cache-layout inconsistency fails closed without
writing a successful-looking artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import k3_target_verify as target
import tp2_benchmark as base

DIAGNOSTIC_SCHEMA = "k3-tp2-target-divergence-diagnostic/v1"
DIAGNOSTIC_WIDTHS = tuple(width for width in target.VERIFY_WIDTHS if width > 1)
COMPLETE_DIVERGENCE = "COMPLETE_DIVERGENCE"
COMPLETE_NO_DIVERGENCE = "COMPLETE_NO_DIVERGENCE"


class DiagnosticError(target.VerificationError):
    """A fail-closed target-divergence diagnostic error."""


@dataclass
class ForwardCapture:
    layer_outputs: list[tuple[int, Any]] = field(default_factory=list)
    final_hidden_states: list[Any] = field(default_factory=list)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shape(value: Any) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def _dtype(value: Any) -> str:
    return str(value.dtype)


def _active_k3_parts(model: Any) -> tuple[Any, list[Any]]:
    language_model = getattr(model, "language_model", None)
    text_model = getattr(language_model, "model", None)
    layers = list(getattr(model, "layers", ()))
    if language_model is None or text_model is None:
        raise DiagnosticError("model does not expose the pinned K3 language model")
    if len(layers) != target.EXPECTED_LAYER_COUNT:
        raise DiagnosticError(
            f"expected {target.EXPECTED_LAYER_COUNT} active K3 layers, got "
            f"{len(layers)}"
        )
    if not layers or any(layer is None for layer in layers):
        raise DiagnosticError("active K3 layer list contains an empty entry")
    layer_type = type(layers[0])
    if any(type(layer) is not layer_type for layer in layers):
        raise DiagnosticError("K3 decoder layers do not share one capture type")
    return text_model, layers


@contextlib.contextmanager
def capture_k3_forward(model: Any) -> Iterator[ForwardCapture]:
    """Capture decoder outputs without changing the model's numerical path."""

    text_model, layers = _active_k3_parts(model)
    layer_type = type(layers[0])
    text_model_type = type(text_model)
    layer_indices = {id(layer): index for index, layer in enumerate(layers)}
    original_layer_call = layer_type.__call__
    original_text_model_call = text_model_type.__call__
    capture = ForwardCapture()

    def captured_layer_call(layer_self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_layer_call(layer_self, *args, **kwargs)
        layer_index = layer_indices.get(id(layer_self))
        if layer_index is not None:
            if not isinstance(result, tuple) or not result:
                raise DiagnosticError(
                    f"layer {layer_index} did not return the pinned K3 tuple"
                )
            capture.layer_outputs.append((layer_index, result[0]))
        return result

    def captured_text_model_call(
        text_model_self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_text_model_call(text_model_self, *args, **kwargs)
        if text_model_self is text_model:
            capture.final_hidden_states.append(result)
        return result

    setattr(layer_type, "__call__", captured_layer_call)
    setattr(text_model_type, "__call__", captured_text_model_call)
    try:
        yield capture
    finally:
        setattr(layer_type, "__call__", original_layer_call)
        setattr(text_model_type, "__call__", original_text_model_call)


def validate_capture(
    capture: ForwardCapture,
    *,
    layer_count: int,
    forward_calls: int,
) -> None:
    expected_layers = layer_count * forward_calls
    if len(capture.layer_outputs) != expected_layers:
        raise DiagnosticError(
            f"captured {len(capture.layer_outputs)} layer outputs, expected "
            f"{expected_layers}"
        )
    if len(capture.final_hidden_states) != forward_calls:
        raise DiagnosticError(
            f"captured {len(capture.final_hidden_states)} final hidden states, "
            f"expected {forward_calls}"
        )
    expected_indices = list(range(layer_count)) * forward_calls
    actual_indices = [index for index, _ in capture.layer_outputs]
    if actual_indices != expected_indices:
        raise DiagnosticError(
            "captured layer order differs from the pinned K3 forward order"
        )


def captured_layer_sequences(
    capture: ForwardCapture,
    *,
    mx: Any,
    layer_count: int,
    forward_calls: int,
) -> list[Any]:
    validate_capture(
        capture,
        layer_count=layer_count,
        forward_calls=forward_calls,
    )
    if forward_calls == 1:
        return [value for _, value in capture.layer_outputs]
    return [
        mx.concatenate(
            [
                capture.layer_outputs[call * layer_count + layer_index][1]
                for call in range(forward_calls)
            ],
            axis=1,
        )
        for layer_index in range(layer_count)
    ]


def captured_hidden_sequence(
    capture: ForwardCapture,
    *,
    mx: Any,
    forward_calls: int,
) -> Any:
    if len(capture.final_hidden_states) != forward_calls:
        raise DiagnosticError("final hidden-state capture count is inconsistent")
    if forward_calls == 1:
        return capture.final_hidden_states[0]
    return mx.concatenate(capture.final_hidden_states, axis=1)


def _forward_with_capture(
    *,
    mx: Any,
    model: Any,
    base_cache: Sequence[Any],
    token_ids: Sequence[int],
    sequential: bool,
) -> tuple[Any, list[Any], ForwardCapture, float]:
    cache = target.clone_k3_cache(base_cache, mx)
    started = time.perf_counter()
    with capture_k3_forward(model) as capture:
        if sequential:
            parts = []
            for token in token_ids:
                logits = model(
                    mx.array([[int(token)]], dtype=mx.uint32),
                    cache=cache,
                )
                mx.eval(logits)
                parts.append(logits)
            result = mx.concatenate(parts, axis=1)
        else:
            inputs = mx.array([list(token_ids)], dtype=mx.uint32)
            mx.eval(inputs)
            result = model(inputs, cache=cache)
        mx.eval(
            result,
            *(value for _, value in capture.layer_outputs),
            *capture.final_hidden_states,
        )
        mx.synchronize()
    elapsed = time.perf_counter() - started
    layer_count = len(_active_k3_parts(model)[1])
    validate_capture(
        capture,
        layer_count=layer_count,
        forward_calls=len(token_ids) if sequential else 1,
    )
    return result, cache, capture, elapsed


def cosine_from_sums(
    numerator: float,
    left_sum_squares: float,
    right_sum_squares: float,
    *,
    exact: bool,
) -> float:
    if min(left_sum_squares, right_sum_squares) < 0:
        raise DiagnosticError("negative sum of squares in cosine calculation")
    denominator = math.sqrt(left_sum_squares * right_sum_squares)
    if denominator == 0:
        return 1.0 if exact else 0.0
    value = numerator / denominator
    if not math.isfinite(value):
        raise DiagnosticError("nonfinite cosine similarity")
    return max(-1.0, min(1.0, value))


def _local_array_metrics(mx: Any, left: Any, right: Any) -> dict[str, Any]:
    if _shape(left) != _shape(right):
        raise DiagnosticError(
            f"comparison shape mismatch: {_shape(left)} vs {_shape(right)}"
        )
    left_float = left.astype(mx.float32)
    right_float = right.astype(mx.float32)
    difference = left_float - right_float
    finite = mx.all(mx.isfinite(left_float)) & mx.all(mx.isfinite(right_float))
    exact = mx.array_equal(left, right)
    maximum = mx.max(mx.abs(difference))
    numerator = mx.sum(left_float * right_float)
    left_sum_squares = mx.sum(left_float * left_float)
    right_sum_squares = mx.sum(right_float * right_float)
    mx.eval(
        finite,
        exact,
        maximum,
        numerator,
        left_sum_squares,
        right_sum_squares,
    )
    exact_value = bool(exact.item())
    result = {
        "finite": bool(finite.item()),
        "exact": exact_value,
        "max_abs_error": float(maximum.item()),
        "cosine_similarity": cosine_from_sums(
            float(numerator.item()),
            float(left_sum_squares.item()),
            float(right_sum_squares.item()),
            exact=exact_value,
        ),
        "shape": _shape(left),
        "left_dtype": _dtype(left),
        "right_dtype": _dtype(right),
    }
    del left_float, right_float, difference
    return result


def _gather_named_array_metrics(
    *,
    mx: Any,
    group: Any,
    names: Sequence[str],
    left: Sequence[Any],
    right: Sequence[Any],
    limits: target.EquivalenceLimits,
) -> list[dict[str, Any]]:
    if not names or len(names) != len(left) or len(left) != len(right):
        raise DiagnosticError("named comparison arrays have inconsistent lengths")
    local = [
        _local_array_metrics(mx, left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    ]
    metadata = [
        {
            "name": name,
            "shape": record["shape"],
            "left_dtype": record["left_dtype"],
            "right_dtype": record["right_dtype"],
        }
        for name, record in zip(names, local, strict=True)
    ]
    metadata_digest = _sha256_json(metadata)
    base.require_shared_digest(mx, group, metadata_digest, "comparison metadata")
    packed = []
    for record in local:
        packed.extend(
            [
                float(record["finite"]),
                float(record["exact"]),
                record["max_abs_error"],
                record["cosine_similarity"],
            ]
        )
    gathered = base.gather_floats(mx, group, packed)
    result = []
    for item_index, (name, metadata_record) in enumerate(
        zip(names, metadata, strict=True)
    ):
        offset = item_index * 4
        per_rank = [
            {
                "rank": rank,
                "finite": bool(round(row[offset])),
                "exact": bool(round(row[offset + 1])),
                "max_abs_error": row[offset + 2],
                "cosine_similarity": row[offset + 3],
            }
            for rank, row in enumerate(gathered)
        ]
        finite_all = all(row["finite"] for row in per_rank)
        exact_all = all(row["exact"] for row in per_rank)
        worst_maximum = max(row["max_abs_error"] for row in per_rank)
        worst_cosine = min(row["cosine_similarity"] for row in per_rank)
        result.append(
            {
                **metadata_record,
                "name": name,
                "finite_all_ranks": finite_all,
                "exact_all_ranks": exact_all,
                "worst_max_abs_error": worst_maximum,
                "worst_cosine_similarity": worst_cosine,
                "within_v3_numeric_limits_all_ranks": (
                    finite_all
                    and worst_maximum <= limits.max_abs_error
                    and worst_cosine >= limits.min_cosine
                ),
                "per_rank": per_rank,
            }
        )
    return result


def compare_position_logits(
    *,
    mx: Any,
    group: Any,
    target_logits: Any,
    sequential_logits: Any,
    expected_ids: Sequence[int],
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    if _shape(target_logits) != _shape(sequential_logits):
        raise DiagnosticError("target and sequential logit shapes differ")
    if len(target_logits.shape) != 3 or int(target_logits.shape[0]) != 1:
        raise DiagnosticError("expected [1, width, vocabulary] logits")
    width = int(target_logits.shape[1])
    if len(expected_ids) != width:
        raise DiagnosticError("expected token count differs from logit width")
    names = [f"position.{position}" for position in range(width)]
    records = _gather_named_array_metrics(
        mx=mx,
        group=group,
        names=names,
        left=[target_logits[:, position, :] for position in range(width)],
        right=[sequential_logits[:, position, :] for position in range(width)],
        limits=limits,
    )
    local_ids: list[float] = []
    for position, expected in enumerate(expected_ids):
        target_id = mx.argmax(target_logits[0, position], axis=-1)
        sequential_id = mx.argmax(sequential_logits[0, position], axis=-1)
        mx.eval(target_id, sequential_id)
        local_ids.extend(
            [
                float(target_id.item()),
                float(sequential_id.item()),
                float(expected),
            ]
        )
    gathered_ids = base.gather_floats(mx, group, local_ids)
    for position, record in enumerate(records):
        offset = position * 3
        per_rank_ids = [
            {
                "rank": rank,
                "target_top1_id": int(round(row[offset])),
                "sequential_top1_id": int(round(row[offset + 1])),
                "expected_id": int(round(row[offset + 2])),
            }
            for rank, row in enumerate(gathered_ids)
        ]
        record["per_rank_top1"] = per_rank_ids
        record["top1_equal_all_ranks"] = all(
            row["target_top1_id"] == row["sequential_top1_id"]
            for row in per_rank_ids
        )
        record["known_continuation_equal_all_ranks"] = all(
            row["target_top1_id"]
            == row["sequential_top1_id"]
            == row["expected_id"]
            for row in per_rank_ids
        )
    return {
        "positions": records,
        "finite_all_positions_all_ranks": all(
            record["finite_all_ranks"] for record in records
        ),
        "exact_all_positions_all_ranks": all(
            record["exact_all_ranks"] for record in records
        ),
        "within_v3_numeric_limits_all_positions_all_ranks": all(
            record["within_v3_numeric_limits_all_ranks"] for record in records
        ),
        "top1_equal_all_positions_all_ranks": all(
            record["top1_equal_all_ranks"] for record in records
        ),
        "known_continuation_equal_all_positions_all_ranks": all(
            record["known_continuation_equal_all_ranks"] for record in records
        ),
    }


def compare_hidden_positions(
    *,
    mx: Any,
    group: Any,
    target_hidden: Any,
    sequential_hidden: Any,
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    if _shape(target_hidden) != _shape(sequential_hidden):
        raise DiagnosticError("target and sequential final-hidden shapes differ")
    width = int(target_hidden.shape[1])
    records = _gather_named_array_metrics(
        mx=mx,
        group=group,
        names=[f"position.{position}" for position in range(width)],
        left=[target_hidden[:, position, :] for position in range(width)],
        right=[sequential_hidden[:, position, :] for position in range(width)],
        limits=limits,
    )
    return {
        "positions": records,
        "exact_all_positions_all_ranks": all(
            record["exact_all_ranks"] for record in records
        ),
        "within_v3_numeric_limits_all_positions_all_ranks": all(
            record["within_v3_numeric_limits_all_ranks"] for record in records
        ),
    }


def first_divergent_layer(
    records: Sequence[dict[str, Any]],
    *,
    criterion: str,
) -> int | None:
    if criterion not in {"exact", "v3_limits"}:
        raise DiagnosticError(f"unknown divergence criterion {criterion!r}")
    key = (
        "exact_all_ranks"
        if criterion == "exact"
        else "within_v3_numeric_limits_all_ranks"
    )
    for record in records:
        if not bool(record[key]):
            return int(record["layer"])
    return None


def compare_layer_outputs(
    *,
    mx: Any,
    group: Any,
    layers: Sequence[Any],
    target_outputs: Sequence[Any],
    sequential_outputs: Sequence[Any],
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    if len(layers) != len(target_outputs) or len(layers) != len(sequential_outputs):
        raise DiagnosticError("captured layer-output lengths differ")
    records = _gather_named_array_metrics(
        mx=mx,
        group=group,
        names=[f"layer.{index:03d}" for index in range(len(layers))],
        left=target_outputs,
        right=sequential_outputs,
        limits=limits,
    )
    for layer_index, (layer, record) in enumerate(zip(layers, records, strict=True)):
        record["layer"] = layer_index
        record["is_linear_attention"] = bool(getattr(layer, "is_linear", False))
        record["attention_class"] = type(getattr(layer, "self_attn", None)).__name__
    return {
        "layers": records,
        "first_exact_divergent_layer": first_divergent_layer(
            records,
            criterion="exact",
        ),
        "first_v3_limit_divergent_layer": first_divergent_layer(
            records,
            criterion="v3_limits",
        ),
    }


def cache_components(cache: Sequence[Any]) -> list[tuple[str, int, str, Any]]:
    components: list[tuple[str, int, str, Any]] = []
    for layer_index, layer_cache in enumerate(cache):
        class_name = type(layer_cache).__name__
        if class_name == "ArraysCache":
            for state_index, value in enumerate(layer_cache.cache):
                if value is not None:
                    components.append(
                        (
                            f"layer.{layer_index:03d}.state.{state_index}",
                            layer_index,
                            class_name,
                            value,
                        )
                    )
            for name in ("left_padding", "lengths"):
                value = getattr(layer_cache, name)
                if value is not None:
                    components.append(
                        (
                            f"layer.{layer_index:03d}.{name}",
                            layer_index,
                            class_name,
                            value,
                        )
                    )
        elif class_name == "KVCache":
            for name in ("keys", "values"):
                value = getattr(layer_cache, name)
                if value is not None:
                    components.append(
                        (
                            f"layer.{layer_index:03d}.{name}",
                            layer_index,
                            class_name,
                            value,
                        )
                    )
        else:
            raise DiagnosticError(
                f"layer {layer_index}: unsupported cache class {class_name!r}"
            )
    if not components:
        raise DiagnosticError("cache comparison has no populated components")
    return components


def _compact_cache_layout(cache: Sequence[Any]) -> dict[str, Any]:
    layout = target.cache_layout(cache)
    return {
        "layers": layout["layers"],
        "class_counts": layout["class_counts"],
        "total_nbytes": layout["total_nbytes"],
        "total_gb_decimal": layout["total_gb_decimal"],
        "kv_offset": layout["kv_offset"],
    }


def compare_caches(
    *,
    mx: Any,
    group: Any,
    target_cache: Sequence[Any],
    sequential_cache: Sequence[Any],
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    target_layout = target.cache_layout(target_cache)
    sequential_layout = target.cache_layout(sequential_cache)
    if target_layout != sequential_layout:
        raise DiagnosticError("target-final and sequential-final cache layouts differ")
    left_components = cache_components(target_cache)
    right_components = cache_components(sequential_cache)
    left_identity = [
        (name, layer, class_name)
        for name, layer, class_name, _ in left_components
    ]
    right_identity = [
        (name, layer, class_name)
        for name, layer, class_name, _ in right_components
    ]
    if left_identity != right_identity:
        raise DiagnosticError("target and sequential cache components differ")
    records = _gather_named_array_metrics(
        mx=mx,
        group=group,
        names=[name for name, _, _, _ in left_components],
        left=[value for _, _, _, value in left_components],
        right=[value for _, _, _, value in right_components],
        limits=limits,
    )
    for identity, record in zip(left_identity, records, strict=True):
        _, layer_index, class_name = identity
        record["layer"] = layer_index
        record["cache_class"] = class_name

    layer_records = []
    for layer_index in range(len(target_cache)):
        components = [
            record for record in records if int(record["layer"]) == layer_index
        ]
        if not components:
            raise DiagnosticError(f"cache layer {layer_index} has no components")
        layer_records.append(
            {
                "layer": layer_index,
                "cache_class": components[0]["cache_class"],
                "component_count": len(components),
                "finite_all_ranks": all(
                    record["finite_all_ranks"] for record in components
                ),
                "exact_all_ranks": all(
                    record["exact_all_ranks"] for record in components
                ),
                "worst_max_abs_error": max(
                    record["worst_max_abs_error"] for record in components
                ),
                "worst_cosine_similarity": min(
                    record["worst_cosine_similarity"] for record in components
                ),
            }
        )
    first_divergent = next(
        (
            int(record["layer"])
            for record in layer_records
            if not record["exact_all_ranks"]
        ),
        None,
    )
    return {
        "layout": _compact_cache_layout(target_cache),
        "component_count": len(records),
        "exact_all_components_all_ranks": all(
            record["exact_all_ranks"] for record in records
        ),
        "finite_all_components_all_ranks": all(
            record["finite_all_ranks"] for record in records
        ),
        "first_exact_divergent_layer": first_divergent,
        "layers": layer_records,
        "components": records,
    }


def localization_summary(
    *,
    logits: dict[str, Any],
    final_hidden: dict[str, Any],
    layers: dict[str, Any],
    final_cache: dict[str, Any],
    rollout_logits: dict[str, Any],
) -> dict[str, Any]:
    logits_within_limits = bool(
        logits["within_v3_numeric_limits_all_positions_all_ranks"]
    )
    hidden_within_limits = bool(
        final_hidden["within_v3_numeric_limits_all_positions_all_ranks"]
    )
    first_layer = layers["first_v3_limit_divergent_layer"]
    if logits_within_limits:
        first_logit_scope = "none_within_v3_limits"
    elif hidden_within_limits:
        first_logit_scope = "vocabulary_head_or_logit_projection"
    elif first_layer is not None:
        first_logit_scope = f"decoder_layer_{int(first_layer)}"
    else:
        first_logit_scope = "final_normalization"
    return {
        "inferred_first_logit_limit_divergence_scope": first_logit_scope,
        "first_exact_divergent_decoder_layer": layers[
            "first_exact_divergent_layer"
        ],
        "first_v3_limit_divergent_decoder_layer": first_layer,
        "final_hidden_within_v3_limits": hidden_within_limits,
        "logits_within_v3_limits": logits_within_limits,
        "final_cache_exact": final_cache["exact_all_components_all_ranks"],
        "first_exact_divergent_cache_layer": final_cache[
            "first_exact_divergent_layer"
        ],
        "subsequent_t1_logits_within_v3_limits": rollout_logits[
            "within_v3_numeric_limits_all_positions_all_ranks"
        ],
    }


def diagnostic_status(
    *,
    logits: dict[str, Any],
    final_hidden: dict[str, Any],
    layers: dict[str, Any],
    final_cache: dict[str, Any],
    rollout_logits: dict[str, Any],
    rollout_cache: dict[str, Any],
) -> str:
    diverged = bool(
        not logits["within_v3_numeric_limits_all_positions_all_ranks"]
        or not logits["top1_equal_all_positions_all_ranks"]
        or not logits["known_continuation_equal_all_positions_all_ranks"]
        or not final_hidden["exact_all_positions_all_ranks"]
        or layers["first_exact_divergent_layer"] is not None
        or not final_cache["exact_all_components_all_ranks"]
        or not rollout_logits["within_v3_numeric_limits_all_positions_all_ranks"]
        or not rollout_logits["top1_equal_all_positions_all_ranks"]
        or not rollout_logits[
            "known_continuation_equal_all_positions_all_ranks"
        ]
        or not rollout_cache["exact_all_components_all_ranks"]
    )
    return COMPLETE_DIVERGENCE if diverged else COMPLETE_NO_DIVERGENCE


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import wired_limit

    init_started = time.perf_counter()
    group = base.init_distributed(mx, backend="jaccl")
    rank = int(group.rank())
    world_size = int(group.size())
    if world_size != base.WORLD_SIZE:
        raise DiagnosticError(f"expected JACCL TP2, got TP{world_size}")
    init_seconds = time.perf_counter() - init_started

    transport = base.inspect_jaccl_ring_transport(rank=rank)
    matrix_digests = base.require_shared_digest(
        mx,
        group,
        transport["device_matrix_sha256"],
        "JACCL device matrix",
    )
    coordinator_digests = base.require_shared_digest(
        mx,
        group,
        transport["coordinator_sha256"],
        "JACCL coordinator",
    )
    contract_digests = base.require_shared_digest(
        mx,
        group,
        transport["transport_contract_sha256"],
        "JACCL transport contract",
    )
    model_dir = base.select_rank_checkpoint(
        args.rank_checkpoint,
        rank=rank,
        world_size=world_size,
    )
    manifest = base.inspect_pinned_manifest(model_dir, rank=rank)
    manifest_digests = base.gather_digests(
        mx,
        group,
        manifest["manifest_sha256"],
    )
    runtime_source = base.inspect_runtime_k3_source()
    runtime_digests = base.require_shared_digest(
        mx,
        group,
        runtime_source["sha256"],
        "K3 runtime source",
    )
    mlx_version = str(mx.__version__)
    mlx_version_digests = base.require_shared_digest(
        mx,
        group,
        hashlib.sha256(mlx_version.encode("utf-8")).hexdigest(),
        "MLX runtime version",
    )

    base._event(rank, "target_diagnostic_load_start", checkpoint=str(model_dir))
    base.barrier(mx, group)
    load_started = time.perf_counter()
    model, tokenizer, loaded_config = base.load_rank_local(
        model_dir,
        tensor_group=group,
        verify_file_hashes=args.verify_file_hashes,
        return_config=True,
    )
    runtime_attestation = target._attest_loaded_runtime(loaded_config)
    load_seconds = time.perf_counter() - load_started
    base.barrier(mx, group)
    load_rows = base.gather_floats(
        mx,
        group,
        [load_seconds, float(mx.get_peak_memory())],
    )
    _, layers = _active_k3_parts(model)
    base._event(rank, "target_diagnostic_load_complete", seconds=load_seconds)

    prompt_ids, prompt_mode = base.build_prompt_tokens(
        tokenizer,
        args.prompt,
        args.prompt_token_target,
    )
    prompt_digest = base.token_digest(prompt_ids)
    prompt_digests = base.require_shared_digest(
        mx,
        group,
        prompt_digest,
        "prompt token IDs",
    )
    prompt_input = mx.array([prompt_ids], dtype=mx.uint32)
    mx.eval(prompt_input)

    generation_context = (
        wired_limit(model) if args.wired_limit else contextlib.nullcontext()
    )
    with generation_context:
        base_cache = model.make_cache()
        base.barrier(mx, group)
        prefill_started = time.perf_counter()
        prefill_logits = model(prompt_input, cache=base_cache)
        mx.eval(prefill_logits)
        mx.synchronize()
        prefill_seconds = time.perf_counter() - prefill_started
        base.barrier(mx, group)
        prefill_rows = base.gather_floats(
            mx,
            group,
            [prefill_seconds, float(mx.get_peak_memory())],
        )
        populated_layout = target.cache_layout(base_cache)
        if populated_layout["kv_offset"] != len(prompt_ids):
            raise DiagnosticError(
                "MLA cache offset does not equal prompt length: "
                f"{populated_layout['kv_offset']} vs {len(prompt_ids)}"
            )
        first_token_array = mx.argmax(prefill_logits[0, -1], axis=-1)
        mx.eval(first_token_array)
        first_token = int(first_token_array.item())
        del prefill_logits

        immutable_guard = target.clone_k3_cache(base_cache, mx)
        target.assert_cache_value_equivalent(base_cache, immutable_guard, mx)
        known_ids = target._known_continuation(
            mx=mx,
            model=model,
            base_cache=base_cache,
            first_token=first_token,
            count=args.width + 2,
        )
        known_digest = base.token_digest(known_ids)
        known_digests = base.require_shared_digest(
            mx,
            group,
            known_digest,
            "diagnostic known continuation",
        )
        input_ids = known_ids[: args.width]
        expected_ids = known_ids[1 : args.width + 1]
        limits = target.EquivalenceLimits(
            min_cosine=args.min_logit_cosine,
            max_abs_error=args.max_logit_abs_error,
        )

        base.barrier(mx, group)
        (
            target_logits,
            target_final_cache,
            target_capture,
            target_elapsed,
        ) = _forward_with_capture(
            mx=mx,
            model=model,
            base_cache=base_cache,
            token_ids=input_ids,
            sequential=False,
        )
        base.barrier(mx, group)
        (
            sequential_logits,
            sequential_final_cache,
            sequential_capture,
            sequential_elapsed,
        ) = _forward_with_capture(
            mx=mx,
            model=model,
            base_cache=base_cache,
            token_ids=input_ids,
            sequential=True,
        )
        timing_rows = base.gather_floats(
            mx,
            group,
            [target_elapsed, sequential_elapsed],
        )

        target_layer_outputs = captured_layer_sequences(
            target_capture,
            mx=mx,
            layer_count=len(layers),
            forward_calls=1,
        )
        sequential_layer_outputs = captured_layer_sequences(
            sequential_capture,
            mx=mx,
            layer_count=len(layers),
            forward_calls=args.width,
        )
        target_hidden = captured_hidden_sequence(
            target_capture,
            mx=mx,
            forward_calls=1,
        )
        sequential_hidden = captured_hidden_sequence(
            sequential_capture,
            mx=mx,
            forward_calls=args.width,
        )

        position_logits = compare_position_logits(
            mx=mx,
            group=group,
            target_logits=target_logits,
            sequential_logits=sequential_logits,
            expected_ids=expected_ids,
            limits=limits,
        )
        final_hidden = compare_hidden_positions(
            mx=mx,
            group=group,
            target_hidden=target_hidden,
            sequential_hidden=sequential_hidden,
            limits=limits,
        )
        layer_comparison = compare_layer_outputs(
            mx=mx,
            group=group,
            layers=layers,
            target_outputs=target_layer_outputs,
            sequential_outputs=sequential_layer_outputs,
            limits=limits,
        )
        final_cache = compare_caches(
            mx=mx,
            group=group,
            target_cache=target_final_cache,
            sequential_cache=sequential_final_cache,
            limits=limits,
        )

        rollout_input_id = known_ids[args.width]
        rollout_expected_id = known_ids[args.width + 1]
        rollout_input = mx.array([[rollout_input_id]], dtype=mx.uint32)
        mx.eval(rollout_input)
        target_rollout_logits = model(
            rollout_input,
            cache=target_final_cache,
        )
        mx.eval(target_rollout_logits)
        sequential_rollout_logits = model(
            rollout_input,
            cache=sequential_final_cache,
        )
        mx.eval(sequential_rollout_logits)
        mx.synchronize()
        rollout_logits = compare_position_logits(
            mx=mx,
            group=group,
            target_logits=target_rollout_logits,
            sequential_logits=sequential_rollout_logits,
            expected_ids=[rollout_expected_id],
            limits=limits,
        )
        rollout_cache = compare_caches(
            mx=mx,
            group=group,
            target_cache=target_final_cache,
            sequential_cache=sequential_final_cache,
            limits=limits,
        )
        target.assert_cache_value_equivalent(immutable_guard, base_cache, mx)

    localization = localization_summary(
        logits=position_logits,
        final_hidden=final_hidden,
        layers=layer_comparison,
        final_cache=final_cache,
        rollout_logits=rollout_logits,
    )
    status = diagnostic_status(
        logits=position_logits,
        final_hidden=final_hidden,
        layers=layer_comparison,
        final_cache=final_cache,
        rollout_logits=rollout_logits,
        rollout_cache=rollout_cache,
    )
    artifact = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": status,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": {
            "authoritative_target_verification_schema": target.ARTIFACT_SCHEMA,
            "authoritative_target_verification_gate_unchanged": True,
            "diagnostic_status_is_non_promotional": True,
            "diagnostic_cannot_relabel_v3_fail_as_pass": True,
        },
        "source": {
            "repo": base.SOURCE_REPO,
            "revision": base.SOURCE_REVISION,
            "effective_revision": base.EFFECTIVE_SOURCE_REVISION,
            "config_sha256": base.SOURCE_CONFIG_SHA256,
            "index_sha256": base.SOURCE_INDEX_SHA256,
        },
        "runtime_contract": {
            **target.runtime_contract_record(
                runtime_source=runtime_source,
                runtime_digests=runtime_digests,
                attestation=runtime_attestation,
            ),
            "mlx_version": mlx_version,
            "mlx_version_sha256_by_rank": mlx_version_digests,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "distributed": {
            "backend": "jaccl-ring",
            "world_size": world_size,
            "strict_init_requested": True,
            "init_seconds_rank0": init_seconds,
            "checkpoint_argument_by_rank": [
                str(path) for path in args.rank_checkpoint
            ],
            "checkpoint_manifest_sha256_by_rank": manifest_digests,
            "load_seconds_by_rank": [row[0] for row in load_rows],
            "load_peak_memory_gb_by_rank": [
                row[1] / 1e9 for row in load_rows
            ],
            "transport": {
                "coordinator_sha256_by_rank": coordinator_digests,
                "device_matrix": transport["device_matrix"],
                "device_matrix_sha256_by_rank": matrix_digests,
                "transport_contract_sha256_by_rank": contract_digests,
            },
        },
        "diagnostic": {
            "prompt_mode": prompt_mode,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": prompt_digest,
            "prompt_token_sha256_by_rank": prompt_digests,
            "prefill_critical_path_seconds": max(row[0] for row in prefill_rows),
            "prefill_peak_memory_gb_by_rank": [
                row[1] / 1e9 for row in prefill_rows
            ],
            "width": args.width,
            "input_token_ids": input_ids,
            "expected_next_token_ids": expected_ids,
            "known_continuation_token_ids": known_ids,
            "known_continuation_sha256": known_digest,
            "known_continuation_sha256_by_rank": known_digests,
            "equivalence_limits": {
                "min_cosine_similarity": limits.min_cosine,
                "max_abs_logit_error": limits.max_abs_error,
            },
            "timing": {
                "target_seconds_by_rank": [row[0] for row in timing_rows],
                "sequential_seconds_by_rank": [row[1] for row in timing_rows],
                "capture_enabled": True,
                "not_a_performance_benchmark": True,
            },
            "base_cache": {
                **_compact_cache_layout(base_cache),
                "unchanged_after_diagnostic": True,
            },
            "localization": localization,
            "position_logits": position_logits,
            "final_hidden_states": final_hidden,
            "decoder_layers": layer_comparison,
            "final_cache": final_cache,
            "subsequent_t1_rollout": {
                "input_token_id": rollout_input_id,
                "expected_next_token_id": rollout_expected_id,
                "position_logits": rollout_logits,
                "post_rollout_cache": rollout_cache,
            },
        },
    }
    if rank == 0:
        base.atomic_write_json(args.artifact, artifact)
        compact = {
            "artifact": str(args.artifact.expanduser().resolve()),
            "status": status,
            "width": args.width,
            "first_exact_divergent_layer": layer_comparison[
                "first_exact_divergent_layer"
            ],
            "first_v3_limit_divergent_layer": layer_comparison[
                "first_v3_limit_divergent_layer"
            ],
            "inferred_scope": localization[
                "inferred_first_logit_limit_divergence_scope"
            ],
            "final_hidden_within_v3_limits": localization[
                "final_hidden_within_v3_limits"
            ],
            "logits_within_v3_limits": localization["logits_within_v3_limits"],
            "final_cache_exact": localization["final_cache_exact"],
            "rollout_logits_within_v3_limits": localization[
                "subsequent_t1_logits_within_v3_limits"
            ],
        }
        print(
            "K3_TARGET_DIAGNOSTIC_RESULT "
            + json.dumps(compact, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    del (
        target_logits,
        sequential_logits,
        target_rollout_logits,
        sequential_rollout_logits,
    )
    gc.collect()
    return {"status": status, "artifact": artifact if rank == 0 else None}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kimi K3 TP2 multi-token divergence diagnostic"
    )
    parser.add_argument(
        "--rank-checkpoint",
        action="append",
        required=True,
        metavar="PATH",
        help="pass exactly twice in TP-rank order",
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--prompt", default=base.DEFAULT_PROMPT)
    parser.add_argument("--prompt-token-target", default=128, type=int)
    parser.add_argument(
        "--width",
        default=2,
        type=int,
        choices=DIAGNOSTIC_WIDTHS,
    )
    parser.add_argument("--min-logit-cosine", default=0.999, type=float)
    parser.add_argument("--max-logit-abs-error", default=1.0, type=float)
    parser.add_argument(
        "--wired-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verify-file-hashes", action="store_true")
    args = parser.parse_args(argv)
    if len(args.rank_checkpoint) != base.WORLD_SIZE:
        parser.error(
            f"--rank-checkpoint must be supplied exactly {base.WORLD_SIZE} times"
        )
    if args.prompt_token_target <= 0:
        parser.error("--prompt-token-target must be positive")
    try:
        target.EquivalenceLimits(
            min_cosine=args.min_logit_cosine,
            max_abs_error=args.max_logit_abs_error,
        )
    except target.VerificationError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    rank_text = os.environ.get("MLX_RANK")
    try:
        run(parse_args(argv))
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        error: BaseException = DiagnosticError("interrupted")
    except BaseException as exc:
        error = exc
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "rank": int(rank_text) if rank_text and rank_text.isdigit() else None,
        "status": "ERROR",
    }
    print(
        "K3_TARGET_DIAGNOSTIC_ERROR "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )
    if os.environ.get("K3_TARGET_DIAGNOSTIC_TRACEBACK") == "1":
        traceback.print_exception(error, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
