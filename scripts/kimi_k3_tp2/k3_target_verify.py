#!/usr/bin/env python3
"""Direct TP2 target-verification benchmark for rank-local Kimi K3.

The benchmark answers one narrow question needed before implementing
speculative decoding: how long does the current target model take to verify
1, 2, 3, 4, or 7 known continuation tokens in one forward pass?

Every timed call starts from a newly materialized copy of the same post-prefill
cache.  Kimi K3's cache is heterogeneous: 69 KDA layers use ``ArraysCache``
and 24 MLA layers use ``KVCache``.  Generic ``deepcopy`` and cache trimming are
both intentionally avoided:

* KDA recurrent arrays, optional padding/length arrays, and MLA key/value
  backing buffers are copied as MLX arrays;
* MLA offsets and the full 256-token-capacity backing buffers are preserved,
  so a verification call is not charged an artificial cache reallocation;
* the immutable base cache is retained and compared exactly after all runs.

The one-pass logits are compared with token-by-token target forwards from the
same cache state.  A PASS artifact requires finite logits, exact top-1
agreement, exact agreement with the known greedy continuation, cosine
similarity above the configured floor, and maximum absolute logit error below
the configured ceiling on every TP rank.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import tp2_benchmark as base

ARTIFACT_SCHEMA = "k3-tp2-target-verification/v3"
VERIFY_WIDTHS = (1, 2, 3, 4, 7, 8)
EXPECTED_ARRAY_CACHE_COUNT = 69
EXPECTED_KV_CACHE_COUNT = 24
EXPECTED_LAYER_COUNT = EXPECTED_ARRAY_CACHE_COUNT + EXPECTED_KV_CACHE_COUNT


class VerificationError(base.BenchmarkError):
    """A fail-closed target-verification precondition failure."""


@dataclass(frozen=True)
class EquivalenceLimits:
    min_cosine: float
    max_abs_error: float

    def __post_init__(self) -> None:
        if not 0.0 < self.min_cosine <= 1.0:
            raise VerificationError("min cosine must be in (0, 1]")
        if not math.isfinite(self.max_abs_error) or self.max_abs_error < 0:
            raise VerificationError("max absolute error must be finite and nonnegative")


def _median(values: Iterable[float]) -> float:
    concrete = [float(value) for value in values]
    if not concrete:
        raise VerificationError("cannot take median of an empty sequence")
    return float(statistics.median(concrete))


def _percentile(values: Iterable[float], percentile: float) -> float:
    concrete = sorted(float(value) for value in values)
    if not concrete:
        raise VerificationError("cannot take percentile of an empty sequence")
    if not 0 <= percentile <= 1:
        raise VerificationError("percentile must be in [0, 1]")
    if len(concrete) == 1:
        return concrete[0]
    position = percentile * (len(concrete) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return concrete[lower] * (1 - fraction) + concrete[upper] * fraction


def _dtype_name(value: Any) -> str:
    return str(getattr(value, "dtype", type(value).__name__))


def _shape_list(value: Any) -> list[int]:
    return [int(dimension) for dimension in getattr(value, "shape", ())]


def _nbytes(value: Any) -> int:
    return int(getattr(value, "nbytes", 0))


def _clone_array(mx: Any, value: Any) -> Any:
    """Make an independently materialized MLX array.

    ``mx.array(existing_array)`` is the least surprising cross-version copy
    API.  The identity fallback handles a backend that elects to return the
    original immutable array object.
    """

    if value is None:
        return None
    # The explicit elementwise expression is deliberate.  Some array
    # constructors may return a second Python handle to immutable storage;
    # adding a same-shaped zero array forces a separately evaluated result
    # while preserving every bit of integer and floating-point state.
    copied = mx.array(value) + mx.zeros_like(value)
    if copied is value:
        raise VerificationError("MLX array copy returned the source object")
    return copied


def _arrays_in_cache(cache: Sequence[Any]) -> list[Any]:
    arrays: list[Any] = []
    for layer_cache in cache:
        class_name = type(layer_cache).__name__
        if class_name == "ArraysCache":
            arrays.extend(value for value in layer_cache.cache if value is not None)
            if layer_cache.left_padding is not None:
                arrays.append(layer_cache.left_padding)
            if layer_cache.lengths is not None:
                arrays.append(layer_cache.lengths)
        elif class_name == "KVCache":
            if layer_cache.keys is not None:
                arrays.append(layer_cache.keys)
            if layer_cache.values is not None:
                arrays.append(layer_cache.values)
        else:
            raise VerificationError(f"unsupported cache class {class_name!r}")
    return arrays


def cache_layout(
    cache: Sequence[Any], *, require_populated: bool = True
) -> dict[str, Any]:
    """Return and validate the exact K3 cache contract."""

    if len(cache) != EXPECTED_LAYER_COUNT:
        raise VerificationError(
            f"expected {EXPECTED_LAYER_COUNT} cache layers, got {len(cache)}"
        )
    counts = {"ArraysCache": 0, "KVCache": 0}
    layers: list[dict[str, Any]] = []
    total_bytes = 0
    kv_offsets: list[int] = []
    for layer_index, layer_cache in enumerate(cache):
        class_name = type(layer_cache).__name__
        if class_name not in counts:
            raise VerificationError(
                f"layer {layer_index}: unsupported cache class {class_name!r}"
            )
        counts[class_name] += 1
        if class_name == "ArraysCache":
            values = list(layer_cache.cache)
            if len(values) != 2:
                raise VerificationError(
                    f"layer {layer_index}: ArraysCache must contain two states"
                )
            if require_populated and any(value is None for value in values):
                raise VerificationError(
                    f"layer {layer_index}: KDA cache is not populated after prefill"
                )
            state = [
                None
                if value is None
                else {
                    "shape": _shape_list(value),
                    "dtype": _dtype_name(value),
                    "nbytes": _nbytes(value),
                }
                for value in values
            ]
            layer_bytes = sum(_nbytes(value) for value in values if value is not None)
            layers.append(
                {
                    "layer": layer_index,
                    "class": class_name,
                    "state": state,
                    "left_padding": layer_cache.left_padding is not None,
                    "lengths": layer_cache.lengths is not None,
                    "nbytes": layer_bytes,
                }
            )
        else:
            keys = layer_cache.keys
            values = layer_cache.values
            if require_populated and (keys is None or values is None):
                raise VerificationError(
                    f"layer {layer_index}: MLA KV cache is not populated after prefill"
                )
            offset = int(layer_cache.offset)
            kv_offsets.append(offset)
            layer_bytes = _nbytes(keys) + _nbytes(values)
            layers.append(
                {
                    "layer": layer_index,
                    "class": class_name,
                    "offset": offset,
                    "keys": (
                        None
                        if keys is None
                        else {
                            "shape": _shape_list(keys),
                            "dtype": _dtype_name(keys),
                            "nbytes": _nbytes(keys),
                        }
                    ),
                    "values": (
                        None
                        if values is None
                        else {
                            "shape": _shape_list(values),
                            "dtype": _dtype_name(values),
                            "nbytes": _nbytes(values),
                        }
                    ),
                    "nbytes": layer_bytes,
                }
            )
        total_bytes += layer_bytes

    if counts != {
        "ArraysCache": EXPECTED_ARRAY_CACHE_COUNT,
        "KVCache": EXPECTED_KV_CACHE_COUNT,
    }:
        raise VerificationError(f"wrong K3 cache-class counts: {counts}")
    if require_populated and len(set(kv_offsets)) != 1:
        raise VerificationError(f"MLA cache offsets disagree: {kv_offsets}")
    return {
        "layers": len(cache),
        "class_counts": counts,
        "total_nbytes": total_bytes,
        "total_gb_decimal": total_bytes / 1e9,
        "kv_offset": kv_offsets[0] if kv_offsets else None,
        "detail": layers,
    }


def clone_k3_cache(cache: Sequence[Any], mx: Any) -> list[Any]:
    """Clone the exact mixed K3 cache without trimming KV backing capacity."""

    cache_layout(cache)
    cloned: list[Any] = []
    pending: list[Any] = []
    for layer_cache in cache:
        class_name = type(layer_cache).__name__
        if class_name == "ArraysCache":
            copied = type(layer_cache)(size=len(layer_cache.cache))
            copied.cache = [_clone_array(mx, value) for value in layer_cache.cache]
            copied.left_padding = _clone_array(mx, layer_cache.left_padding)
            copied.lengths = _clone_array(mx, layer_cache.lengths)
            pending.extend(value for value in copied.cache if value is not None)
            if copied.left_padding is not None:
                pending.append(copied.left_padding)
            if copied.lengths is not None:
                pending.append(copied.lengths)
        elif class_name == "KVCache":
            copied = type(layer_cache)()
            copied.keys = _clone_array(mx, layer_cache.keys)
            copied.values = _clone_array(mx, layer_cache.values)
            copied.offset = int(layer_cache.offset)
            if copied.keys is not None:
                pending.append(copied.keys)
            if copied.values is not None:
                pending.append(copied.values)
        else:
            raise VerificationError(f"unsupported cache class {class_name!r}")
        cloned.append(copied)
    if pending:
        mx.eval(*pending)
    if cache_layout(cloned) != cache_layout(cache):
        raise VerificationError("cloned cache layout differs from source")
    return cloned


def assert_cache_value_equivalent(
    source: Sequence[Any],
    candidate: Sequence[Any],
    mx: Any,
) -> None:
    """Require exact metadata and elementwise state equality."""

    if cache_layout(source) != cache_layout(candidate):
        raise VerificationError("cache metadata/layout mismatch")
    source_arrays = _arrays_in_cache(source)
    candidate_arrays = _arrays_in_cache(candidate)
    if len(source_arrays) != len(candidate_arrays):
        raise VerificationError("cache array counts differ")
    checks = []
    for index, (left, right) in enumerate(
        zip(source_arrays, candidate_arrays, strict=True)
    ):
        if left is right:
            raise VerificationError(f"cache array {index} aliases source object")
        if _shape_list(left) != _shape_list(right):
            raise VerificationError(f"cache array {index} shape differs")
        if _dtype_name(left) != _dtype_name(right):
            raise VerificationError(f"cache array {index} dtype differs")
        checks.append(mx.array_equal(left, right))
    if checks:
        mx.eval(*checks)
    failed = [index for index, check in enumerate(checks) if not bool(check.item())]
    if failed:
        raise VerificationError(f"cache arrays differ at indices {failed}")


def _memory_bytes(mx: Any) -> dict[str, int]:
    required = ("get_active_memory", "get_peak_memory", "get_cache_memory")
    missing = [name for name in required if not hasattr(mx, name)]
    if missing:
        raise VerificationError(f"MLX memory APIs are missing: {missing}")
    return {
        "active": int(mx.get_active_memory()),
        "peak": int(mx.get_peak_memory()),
        "cache": int(mx.get_cache_memory()),
    }


def _timing_record(
    rows: Sequence[Sequence[float]],
    *,
    width: int,
) -> dict[str, Any]:
    if len(rows) != base.WORLD_SIZE or any(len(row) != 5 for row in rows):
        raise VerificationError("timing gather returned the wrong TP2 shape")
    critical_seconds = max(row[0] for row in rows)
    if critical_seconds <= 0:
        raise VerificationError("nonpositive critical-path time")
    return {
        "critical_path_seconds": critical_seconds,
        "verified_tokens_per_second": width / critical_seconds,
        "milliseconds_per_verified_token": 1000 * critical_seconds / width,
        "critical_peak_memory_gb": max(row[3] for row in rows) / 1e9,
        "critical_peak_delta_gb": max(max(row[3] - row[1], 0.0) for row in rows) / 1e9,
        "per_rank": [
            {
                "rank": rank,
                "seconds": row[0],
                "active_before_gb": row[1] / 1e9,
                "active_after_gb": row[2] / 1e9,
                "peak_gb": row[3] / 1e9,
                "cache_allocator_gb": row[4] / 1e9,
            }
            for rank, row in enumerate(rows)
        ],
    }


def _timed_target_forward(
    *,
    mx: Any,
    group: Any,
    model: Any,
    base_cache: Sequence[Any],
    token_ids: Sequence[int],
) -> dict[str, Any]:
    snapshot = clone_k3_cache(base_cache, mx)
    inputs = mx.array([list(token_ids)], dtype=mx.uint32)
    mx.eval(inputs)
    mx.synchronize()
    base.barrier(mx, group)
    mx.reset_peak_memory()
    memory_before = _memory_bytes(mx)
    started = time.perf_counter()
    logits = model(inputs, cache=snapshot)
    mx.eval(logits)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    memory_after = _memory_bytes(mx)
    rows = base.gather_floats(
        mx,
        group,
        [
            elapsed,
            float(memory_before["active"]),
            float(memory_after["active"]),
            float(memory_after["peak"]),
            float(memory_after["cache"]),
        ],
    )
    del logits, inputs, snapshot
    gc.collect()
    return _timing_record(rows, width=len(token_ids))


def _timed_sequential_forward(
    *,
    mx: Any,
    group: Any,
    model: Any,
    base_cache: Sequence[Any],
    token_ids: Sequence[int],
) -> dict[str, Any]:
    snapshot = clone_k3_cache(base_cache, mx)
    inputs = [mx.array([[int(token)]], dtype=mx.uint32) for token in token_ids]
    mx.eval(*inputs)
    mx.synchronize()
    base.barrier(mx, group)
    mx.reset_peak_memory()
    memory_before = _memory_bytes(mx)
    started = time.perf_counter()
    logits = None
    for token_input in inputs:
        logits = model(token_input, cache=snapshot)
        mx.eval(logits)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    memory_after = _memory_bytes(mx)
    rows = base.gather_floats(
        mx,
        group,
        [
            elapsed,
            float(memory_before["active"]),
            float(memory_after["active"]),
            float(memory_after["peak"]),
            float(memory_after["cache"]),
        ],
    )
    del logits, inputs, snapshot
    gc.collect()
    return _timing_record(rows, width=len(token_ids))


def _forward_logits(
    *,
    mx: Any,
    model: Any,
    base_cache: Sequence[Any],
    token_ids: Sequence[int],
    sequential: bool,
) -> Any:
    snapshot = clone_k3_cache(base_cache, mx)
    if sequential:
        parts = []
        for token in token_ids:
            logits = model(
                mx.array([[int(token)]], dtype=mx.uint32),
                cache=snapshot,
            )
            mx.eval(logits)
            parts.append(logits)
        result = mx.concatenate(parts, axis=1)
    else:
        result = model(
            mx.array([list(token_ids)], dtype=mx.uint32),
            cache=snapshot,
        )
    mx.eval(result)
    del snapshot
    return result


def equivalence_pass(
    record: dict[str, Any],
    limits: EquivalenceLimits,
) -> bool:
    return bool(
        record.get("finite")
        and record.get("top1_equal")
        and record.get("known_continuation_equal")
        and float(record.get("cosine_similarity", float("-inf"))) >= limits.min_cosine
        and float(record.get("max_abs_error", float("inf"))) <= limits.max_abs_error
    )


def _equivalence_metrics(
    *,
    mx: Any,
    group: Any,
    model: Any,
    base_cache: Sequence[Any],
    token_ids: Sequence[int],
    expected_next_ids: Sequence[int],
    limits: EquivalenceLimits,
) -> dict[str, Any]:
    target = _forward_logits(
        mx=mx,
        model=model,
        base_cache=base_cache,
        token_ids=token_ids,
        sequential=False,
    ).astype(mx.float32)
    sequential = _forward_logits(
        mx=mx,
        model=model,
        base_cache=base_cache,
        token_ids=token_ids,
        sequential=True,
    ).astype(mx.float32)
    if target.shape != sequential.shape:
        raise VerificationError(
            f"target/sequential logit shapes differ: {target.shape} vs "
            f"{sequential.shape}"
        )
    difference = target - sequential
    target_flat = target.reshape(-1)
    sequential_flat = sequential.reshape(-1)
    finite = mx.all(mx.isfinite(target)) & mx.all(mx.isfinite(sequential))
    max_abs = mx.max(mx.abs(difference))
    numerator = mx.sum(target_flat * sequential_flat)
    denominator = mx.sqrt(
        mx.sum(target_flat * target_flat) * mx.sum(sequential_flat * sequential_flat)
    )
    cosine = numerator / denominator
    target_top1 = mx.argmax(target, axis=-1).reshape(-1)
    sequential_top1 = mx.argmax(sequential, axis=-1).reshape(-1)
    top1_equal = mx.all(target_top1 == sequential_top1)
    mx.eval(
        finite,
        max_abs,
        cosine,
        target_top1,
        sequential_top1,
        top1_equal,
    )
    target_ids = [int(value) for value in target_top1.tolist()]
    sequential_ids = [int(value) for value in sequential_top1.tolist()]
    expected = [int(value) for value in expected_next_ids]
    local = {
        "finite": bool(finite.item()),
        "max_abs_error": float(max_abs.item()),
        "cosine_similarity": float(cosine.item()),
        "top1_equal": bool(top1_equal.item()),
        "known_continuation_equal": (
            target_ids == expected and sequential_ids == expected
        ),
        "target_top1_ids": target_ids,
        "sequential_top1_ids": sequential_ids,
        "expected_next_ids": expected,
    }
    local["pass"] = equivalence_pass(local, limits)
    numeric_rows = base.gather_floats(
        mx,
        group,
        [
            float(local["finite"]),
            local["max_abs_error"],
            local["cosine_similarity"],
            float(local["top1_equal"]),
            float(local["known_continuation_equal"]),
            float(local["pass"]),
        ],
    )
    target_digest = base.token_digest(target_ids)
    sequential_digest = base.token_digest(sequential_ids)
    target_digests = base.gather_digests(mx, group, target_digest)
    sequential_digests = base.gather_digests(mx, group, sequential_digest)
    del target, sequential, difference
    per_rank = [
        {
            "rank": rank,
            "finite": bool(round(row[0])),
            "max_abs_error": row[1],
            "cosine_similarity": row[2],
            "top1_equal": bool(round(row[3])),
            "known_continuation_equal": bool(round(row[4])),
            "pass": bool(round(row[5])),
            "target_top1_sha256": target_digests[rank],
            "sequential_top1_sha256": sequential_digests[rank],
        }
        for rank, row in enumerate(numeric_rows)
    ]
    return {
        **local,
        "pass_all_ranks": all(row["pass"] for row in per_rank),
        "worst_max_abs_error": max(row["max_abs_error"] for row in per_rank),
        "worst_cosine_similarity": min(row["cosine_similarity"] for row in per_rank),
        "per_rank": per_rank,
    }


def summarize_width(
    *,
    width: int,
    target_runs: Sequence[dict[str, Any]],
    sequential_runs: Sequence[dict[str, Any]],
    equivalence: dict[str, Any],
) -> dict[str, Any]:
    if not target_runs or not sequential_runs:
        raise VerificationError("each width requires target and sequential runs")
    target_seconds = [row["critical_path_seconds"] for row in target_runs]
    sequential_seconds = [row["critical_path_seconds"] for row in sequential_runs]
    target_median = _median(target_seconds)
    sequential_median = _median(sequential_seconds)
    return {
        "width": width,
        "equivalence": equivalence,
        "target_forward": {
            "median_critical_path_seconds": target_median,
            "p90_critical_path_seconds": _percentile(target_seconds, 0.9),
            "median_verified_tokens_per_second": width / target_median,
            "median_milliseconds_per_verified_token": (1000 * target_median / width),
            "runs": list(target_runs),
        },
        "sequential_forward": {
            "median_critical_path_seconds": sequential_median,
            "p90_critical_path_seconds": _percentile(sequential_seconds, 0.9),
            "median_tokens_per_second": width / sequential_median,
            "median_milliseconds_per_token": 1000 * sequential_median / width,
            "runs": list(sequential_runs),
        },
        "target_speedup_vs_sequential": sequential_median / target_median,
    }


def artifact_status(width_records: Sequence[dict[str, Any]]) -> str:
    if [int(record["width"]) for record in width_records] != list(VERIFY_WIDTHS):
        raise VerificationError(
            f"artifact widths must be exactly {list(VERIFY_WIDTHS)}"
        )
    return (
        "PASS"
        if all(record["equivalence"]["pass_all_ranks"] for record in width_records)
        else "FAIL"
    )


def _known_continuation(
    *,
    mx: Any,
    model: Any,
    base_cache: Sequence[Any],
    first_token: int,
    count: int,
) -> list[int]:
    if count <= 0:
        raise VerificationError("known continuation count must be positive")
    working = clone_k3_cache(base_cache, mx)
    result = [int(first_token)]
    while len(result) < count:
        logits = model(
            mx.array([[result[-1]]], dtype=mx.uint32),
            cache=working,
        )
        next_token = mx.argmax(logits[0, -1], axis=-1)
        mx.eval(next_token)
        result.append(int(next_token.item()))
    del working
    return result


def _attest_loaded_runtime(config: dict[str, Any]) -> dict[str, Any]:
    compatibility = config.get("_rank_local_compatibility_transform")
    if (
        not isinstance(compatibility, dict)
        or compatibility.get("contract") != base.DTYPE_FIX_CONTRACT
        or compatibility.get("effective_source_revision")
        != base.EFFECTIVE_SOURCE_REVISION
    ):
        raise VerificationError("rank-local K3 dtype correction was not attested")
    vocab = config.get("_rank_local_vocab_parallel_head")
    if not isinstance(vocab, dict) or vocab.get("enabled") is not True:
        raise VerificationError(
            "current benchmark requires EXO_MLX_K3_VOCAB_PARALLEL_HEAD=1"
        )
    if int(vocab.get("world_size", 0)) != base.WORLD_SIZE:
        raise VerificationError("vocabulary-parallel head is not TP2")
    rejected = []
    for key in (
        "_rank_local_routed_latent_requant",
        "_rank_local_attention_qkvg_requant",
    ):
        value = config.get(key)
        if isinstance(value, dict) and value.get("enabled"):
            rejected.append(key)
    if rejected:
        raise VerificationError(
            f"lossy experimental requantization is forbidden: {rejected}"
        )
    return {
        "compatibility_transform": compatibility,
        "vocab_parallel_head": vocab,
        "lossy_requantization_enabled": False,
    }


def runtime_contract_record(
    *,
    runtime_source: dict[str, str],
    runtime_digests: Sequence[str],
    attestation: dict[str, Any],
) -> dict[str, Any]:
    """Record checkpoint-converter and execution-runtime provenance separately."""

    return {
        "checkpoint_converter_mlx_lm_commit": base.MLX_LM_COMMIT,
        "checkpoint_mlx_lm_kimi_k3_sha256": (base.CHECKPOINT_MLX_LM_KIMI_K3_SHA256),
        "execution_runtime_mlx_lm_commit": base.RUNTIME_MLX_LM_COMMIT,
        "execution_runtime_kimi_k3_sha256": base.MLX_LM_KIMI_K3_SHA256,
        "imported_kimi_k3_path_rank0": runtime_source["path"],
        "imported_kimi_k3_sha256_by_rank": list(runtime_digests),
        "tp_contract": base.CONTRACT_VERSION,
        "tp_contract_digest": base.CONTRACT_DIGEST,
        "attestation": attestation,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import wired_limit

    init_started = time.perf_counter()
    group = base.init_distributed(mx, backend="jaccl")
    rank = int(group.rank())
    world_size = int(group.size())
    if world_size != base.WORLD_SIZE:
        raise VerificationError(f"expected JACCL TP2, got TP{world_size}")
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

    base._event(rank, "target_verify_load_start", checkpoint=str(model_dir))
    base.barrier(mx, group)
    load_started = time.perf_counter()
    model, tokenizer, loaded_config = base.load_rank_local(
        model_dir,
        tensor_group=group,
        verify_file_hashes=args.verify_file_hashes,
        return_config=True,
    )
    runtime_attestation = _attest_loaded_runtime(loaded_config)
    load_seconds = time.perf_counter() - load_started
    base.barrier(mx, group)
    load_rows = base.gather_floats(
        mx,
        group,
        [load_seconds, float(mx.get_peak_memory())],
    )
    base._event(rank, "target_verify_load_complete", seconds=load_seconds)

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
        initial_layout = cache_layout(base_cache, require_populated=False)
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
        populated_layout = cache_layout(base_cache)
        if populated_layout["kv_offset"] != len(prompt_ids):
            raise VerificationError(
                "MLA cache offset does not equal the fixed prompt length: "
                f"{populated_layout['kv_offset']} vs {len(prompt_ids)}"
            )
        first_token_array = mx.argmax(prefill_logits[0, -1], axis=-1)
        mx.eval(first_token_array)
        first_token = int(first_token_array.item())
        del prefill_logits

        preflight_clone = clone_k3_cache(base_cache, mx)
        assert_cache_value_equivalent(base_cache, preflight_clone, mx)
        immutable_guard = preflight_clone

        known_ids = _known_continuation(
            mx=mx,
            model=model,
            base_cache=base_cache,
            first_token=first_token,
            count=max(VERIFY_WIDTHS) + 1,
        )
        known_digest = base.token_digest(known_ids)
        known_digests = base.require_shared_digest(
            mx,
            group,
            known_digest,
            "known greedy continuation",
        )
        try:
            known_text = tokenizer.decode(
                known_ids,
                skip_special_tokens=False,
            )
        except TypeError:
            known_text = tokenizer.decode(known_ids)

        limits = EquivalenceLimits(
            min_cosine=args.min_logit_cosine,
            max_abs_error=args.max_logit_abs_error,
        )
        width_records = []
        for width in VERIFY_WIDTHS:
            input_ids = known_ids[:width]
            expected_next = known_ids[1 : width + 1]
            base._event(rank, "target_verify_width_start", width=width)

            for _ in range(args.warmups):
                _timed_target_forward(
                    mx=mx,
                    group=group,
                    model=model,
                    base_cache=base_cache,
                    token_ids=input_ids,
                )
                _timed_sequential_forward(
                    mx=mx,
                    group=group,
                    model=model,
                    base_cache=base_cache,
                    token_ids=input_ids,
                )

            equivalence = _equivalence_metrics(
                mx=mx,
                group=group,
                model=model,
                base_cache=base_cache,
                token_ids=input_ids,
                expected_next_ids=expected_next,
                limits=limits,
            )
            target_runs = []
            sequential_runs = []
            for _ in range(args.runs):
                target_runs.append(
                    _timed_target_forward(
                        mx=mx,
                        group=group,
                        model=model,
                        base_cache=base_cache,
                        token_ids=input_ids,
                    )
                )
                sequential_runs.append(
                    _timed_sequential_forward(
                        mx=mx,
                        group=group,
                        model=model,
                        base_cache=base_cache,
                        token_ids=input_ids,
                    )
                )
            record = summarize_width(
                width=width,
                target_runs=target_runs,
                sequential_runs=sequential_runs,
                equivalence=equivalence,
            )
            width_records.append(record)
            base._event(
                rank,
                "target_verify_width_complete",
                width=width,
                pass_all_ranks=equivalence["pass_all_ranks"],
                target_seconds=record["target_forward"]["median_critical_path_seconds"],
            )

        assert_cache_value_equivalent(immutable_guard, base_cache, mx)

    status = artifact_status(width_records)
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "status": status,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "repo": base.SOURCE_REPO,
            "revision": base.SOURCE_REVISION,
            "effective_revision": base.EFFECTIVE_SOURCE_REVISION,
            "config_sha256": base.SOURCE_CONFIG_SHA256,
            "index_sha256": base.SOURCE_INDEX_SHA256,
        },
        "runtime_contract": runtime_contract_record(
            runtime_source=runtime_source,
            runtime_digests=runtime_digests,
            attestation=runtime_attestation,
        ),
        "distributed": {
            "backend": "jaccl-ring",
            "world_size": world_size,
            "strict_init_requested": True,
            "init_seconds_rank0": init_seconds,
            "checkpoint_argument_by_rank": [str(path) for path in args.rank_checkpoint],
            "checkpoint_manifest_sha256_by_rank": manifest_digests,
            "load_seconds_by_rank": [row[0] for row in load_rows],
            "load_peak_memory_gb_by_rank": [row[1] / 1e9 for row in load_rows],
            "transport": {
                "coordinator_sha256_by_rank": coordinator_digests,
                "device_matrix": transport["device_matrix"],
                "device_matrix_sha256_by_rank": matrix_digests,
                "transport_contract_sha256_by_rank": contract_digests,
            },
        },
        "benchmark": {
            "question": (
                "critical-path TP2 cost of one-pass target verification from "
                "an identical post-prefill Kimi K3 cache"
            ),
            "widths": list(VERIFY_WIDTHS),
            "runs": args.runs,
            "warmups": args.warmups,
            "prompt_mode": prompt_mode,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": prompt_digest,
            "prompt_token_sha256_by_rank": prompt_digests,
            "prefill_critical_path_seconds": max(row[0] for row in prefill_rows),
            "prefill_peak_memory_gb_by_rank": [row[1] / 1e9 for row in prefill_rows],
            "known_continuation_token_ids": known_ids,
            "known_continuation_sha256": known_digest,
            "known_continuation_sha256_by_rank": known_digests,
            "known_continuation_text": known_text,
            "cache_contract": {
                "expected_arrays_cache_layers": EXPECTED_ARRAY_CACHE_COUNT,
                "expected_kv_cache_layers": EXPECTED_KV_CACHE_COUNT,
                "initial": initial_layout,
                "post_prefill": populated_layout,
                "clone_preflight_exact": True,
                "base_cache_unchanged_after_all_runs": True,
                "kv_backing_capacity_preserved": True,
            },
            "equivalence_limits": {
                "min_cosine_similarity": limits.min_cosine,
                "max_abs_logit_error": limits.max_abs_error,
                "finite_required": True,
                "exact_top1_required": True,
                "known_continuation_required": True,
            },
            "timing_semantics": {
                "clone": "outside timed region; materialized before barrier",
                "start": "after TP2 barrier and local synchronization",
                "end": "after forward logits are evaluated and synchronized",
                "critical_path": "maximum elapsed time across the two ranks",
                "target": "one model forward containing width input tokens",
                "sequential": "width one-token forwards with evaluation per token",
                "cache": "fresh exact clone of one immutable post-prefill state",
            },
            "wired_memory": args.wired_limit,
            "width_results": width_records,
        },
    }
    if rank == 0:
        base.atomic_write_json(args.artifact, artifact)
        compact = {
            "artifact": str(args.artifact.expanduser().resolve()),
            "status": status,
            "widths": {
                str(row["width"]): {
                    "target_ms": 1000
                    * row["target_forward"]["median_critical_path_seconds"],
                    "sequential_ms": 1000
                    * row["sequential_forward"]["median_critical_path_seconds"],
                    "speedup": row["target_speedup_vs_sequential"],
                    "equivalent": row["equivalence"]["pass_all_ranks"],
                }
                for row in width_records
            },
        }
        print(
            "K3_TARGET_VERIFY_RESULT "
            + json.dumps(compact, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    return {"status": status, "artifact": artifact if rank == 0 else None}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kimi K3 direct TP2 target-verification width benchmark"
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
    parser.add_argument("--runs", default=3, type=int)
    parser.add_argument("--warmups", default=1, type=int)
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
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    try:
        EquivalenceLimits(
            min_cosine=args.min_logit_cosine,
            max_abs_error=args.max_logit_abs_error,
        )
    except VerificationError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    rank_text = os.environ.get("MLX_RANK")
    try:
        result = run(parse_args(argv))
        return 0 if result["status"] == "PASS" else 3
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        error: BaseException = VerificationError("interrupted")
    except BaseException as exc:
        error = exc
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "rank": int(rank_text) if rank_text and rank_text.isdigit() else None,
        "status": "ERROR",
    }
    print(
        "K3_TARGET_VERIFY_ERROR "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )
    if os.environ.get("K3_TARGET_VERIFY_TRACEBACK") == "1":
        traceback.print_exception(error, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
