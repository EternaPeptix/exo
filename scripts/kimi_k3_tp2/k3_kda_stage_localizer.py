#!/usr/bin/env python3
"""Fail-closed real-checkpoint localizer for Kimi K3 layer-0 KDA width two.

The authoritative target verifier found the first exact divergence at layer 0.
This one-load diagnostic narrows that result without changing the verifier's
PASS/FAIL contract.  It prefills exactly 128 prompt tokens, derives two known
continuation tokens, and makes both paths consume slices of one evaluated
layer-0 attention input:

* target: the speculative T=2 path, including state histories;
* reference: two accepted packed T=1 decode steps.

Every numerical boundary is evaluated before the other path can mutate cache
or initialize packed weights.  The tool also checks that its staged equations
reproduce the real attention endpoints.  An endpoint mismatch is an ERROR, not
a successful-looking localization artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import k3_target_diagnostic as diagnostic
import k3_target_verify as target
import tp2_benchmark as base

SCHEMA = "k3-tp2-kda-width2-stage-localizer/v1"
PROMPT_TOKEN_TARGET = 128
WIDTH = 2
COMPLETE_DIVERGENCE = "COMPLETE_DIVERGENCE"
COMPLETE_NO_DIVERGENCE = "COMPLETE_NO_DIVERGENCE"

REQUIRED_RUNTIME_ENV = {
    "EXO_MLX_JACCL_FORCE_MESH": "1",
    "EXO_MLX_K3_VOCAB_PARALLEL_HEAD": "1",
    "EXO_MLX_K3_REQUANT_ROUTED_LATENT_MXFP4": "0",
    "EXO_MLX_K3_REQUANT_ATTENTION_QKVG_MXFP4": "0",
    "MLX_LM_KIMI_K3_FUSED_EXPERTS": "1",
    "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE": "1",
    "MLX_LM_KIMI_K3_FUSED_ROUTER": "1",
    "MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS": "1",
    "MLX_LM_KIMI_K3_PACKED_KDA_SKINNY": "1",
    "MLX_LM_KIMI_K3_PACKED_KDA_WIDE": "1",
    "MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD": "1",
    "MLX_LM_KIMI_K3_FUSED_POST_KDA_RMS_SIGMOID_GATE": "0",
    "MLX_LM_KIMI_K3_COMPILED_DECODE": "0",
    "MLX_LM_KIMI_K3_PACKED_MOE_FRONT": "0",
    "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT": "1",
    "MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL": "1",
    "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES": "laguna8",
    "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE": "hidden",
}
CANDIDATE_EXACT_ENV = "MLX_LM_KIMI_K3_EXACT_SPECULATIVE_KDA"
CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV = "MLX_LM_KIMI_K3_EXACT_WIDE_SHORT_CONV"


class LocalizerError(diagnostic.DiagnosticError):
    """A fail-closed layer-0 stage-localization error."""


@dataclass(frozen=True)
class Stage:
    name: str
    target_values: tuple[Any, ...]
    reference_values: tuple[Any, ...]
    component_names: tuple[str, ...] = ("value",)

    def __post_init__(self) -> None:
        if not self.name:
            raise LocalizerError("stage name cannot be empty")
        if (
            not self.target_values
            or len(self.target_values) != len(self.reference_values)
            or len(self.target_values) != len(self.component_names)
        ):
            raise LocalizerError(f"stage {self.name!r} has inconsistent components")


@dataclass
class PathCapture:
    stages: list[Stage]
    output: Any
    final_conv_state: Any
    final_ssm_state: Any


def require_runtime_flags() -> dict[str, str]:
    actual = {name: os.environ.get(name, "") for name in REQUIRED_RUNTIME_ENV}
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in REQUIRED_RUNTIME_ENV.items()
        if actual[name] != expected
    }
    if mismatches:
        raise LocalizerError(
            "current accepted runtime flags are not pinned: "
            + json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
        )
    for candidate_env in (
        CANDIDATE_EXACT_ENV,
        CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV,
    ):
        candidate_value = os.environ.get(candidate_env, "0")
        if candidate_value not in {"0", "1"}:
            raise LocalizerError(f"{candidate_env} must be 0 or 1")
        actual[candidate_env] = candidate_value
    return actual


def _concat(mx: Any, values: Sequence[Any], *, axis: int = 1) -> Any:
    if len(values) != WIDTH:
        raise LocalizerError(f"expected {WIDTH} sequential values, got {len(values)}")
    return mx.concatenate(list(values), axis=axis)


def _eval_values(mx: Any, values: Iterable[Any]) -> None:
    flat = list(values)
    if not flat:
        raise LocalizerError("refusing to evaluate an empty stage set")
    mx.eval(*flat)
    mx.synchronize()


def _stage_values(stages: Sequence[Stage]) -> list[Any]:
    return [
        value
        for stage in stages
        for values in (stage.target_values, stage.reference_values)
        for value in values
    ]


def _split_qkv(attention: Any, qkv: Any) -> tuple[Any, Any, Any]:
    batch, tokens = int(qkv.shape[0]), int(qkv.shape[1])
    projection = int(attention.projection_dim)
    heads = int(attention.num_heads)
    head_dim = int(attention.head_dim)
    q = qkv[..., :projection].reshape(batch, tokens, heads, head_dim)
    k = qkv[..., projection : 2 * projection].reshape(batch, tokens, heads, head_dim)
    v = qkv[..., 2 * projection :].reshape(batch, tokens, heads, head_dim)
    return q, k, v


def _normalize_qk(mx: Any, attention: Any, q: Any, k: Any) -> tuple[Any, Any]:
    epsilon = 1e-6 / int(attention.head_dim)
    scale = float(attention.scale)
    return (
        (scale**2) * mx.fast.rms_norm(q, None, epsilon),
        scale * mx.fast.rms_norm(k, None, epsilon),
    )


def _compute_g(mx: Any, attention: Any, a_logits: Any) -> Any:
    from mlx_lm.models.gated_delta import compute_g, compute_g_safe

    a_log = attention.A_log.reshape(attention.num_heads, 1)
    dt_bias = attention.dt_bias.reshape(attention.num_heads, attention.head_dim)
    if attention.lower_bound is None:
        return compute_g(a_log, a_logits, dt_bias)
    return compute_g_safe(a_log, a_logits, dt_bias, attention.lower_bound)


def _target_path(
    *,
    mx: Any,
    attention: Any,
    shared_input: Any,
    initial_conv_state: Any,
    initial_ssm_state: Any,
) -> PathCapture:
    from mlx_lm.models.gated_delta import (
        gated_delta_kernel,
        gated_delta_update,
    )

    stages: list[Stage] = []
    projected_qkv = attention.qkv_proj(shared_input)
    gate = attention.g_proj(shared_input)
    f_a = attention.f_a_proj(shared_input)
    a_logits = attention.f_b_proj(f_a).reshape(
        1, WIDTH, attention.num_heads, attention.head_dim
    )
    b_logits = attention.b_proj(shared_input).reshape(1, WIDTH, attention.num_heads)
    _eval_values(mx, (projected_qkv, gate, f_a, a_logits, b_logits))

    qkv, conv_state, conv_history = attention.qkv_conv(
        projected_qkv,
        initial_conv_state,
        None,
        None,
        return_state_history=True,
    )
    _eval_values(mx, (qkv, conv_state, conv_history))
    q_raw, k_raw, v = _split_qkv(attention, qkv)
    gate_reshaped = gate.reshape(1, WIDTH, attention.num_heads, attention.head_dim)
    candidate_exact = os.environ.get(CANDIDATE_EXACT_ENV, "0") == "1"
    if candidate_exact:
        q_parts = []
        k_parts = []
        g_parts = []
        beta_parts = []
        gated_parts = []
        ssm_parts = []
        norm_parts = []
        sigmoid_parts = []
        gated_norm_parts = []
        ssm_state = initial_ssm_state
        for position in range(WIDTH):
            q_part, k_part = _normalize_qk(
                mx,
                attention,
                q_raw[:, position : position + 1],
                k_raw[:, position : position + 1],
            )
            beta_part = mx.sigmoid(b_logits[:, position : position + 1])
            g_part = _compute_g(mx, attention, a_logits[:, position : position + 1])
            _eval_values(mx, (q_part, k_part, beta_part, g_part))
            gated_part, ssm_state = gated_delta_kernel(
                q_part,
                k_part,
                v[:, position : position + 1],
                g_part,
                beta_part,
                ssm_state,
            )
            _eval_values(mx, (gated_part, ssm_state))
            reshaped = gated_part.reshape(1, 1, attention.num_heads, attention.head_dim)
            norm_part = attention.o_norm(reshaped)
            sigmoid_part = mx.sigmoid(gate_reshaped[:, position : position + 1])
            gated_norm_part = (norm_part * sigmoid_part).reshape(1, 1, -1)
            _eval_values(mx, (norm_part, sigmoid_part, gated_norm_part))
            q_parts.append(q_part)
            k_parts.append(k_part)
            g_parts.append(g_part)
            beta_parts.append(beta_part)
            gated_parts.append(gated_part)
            ssm_parts.append(ssm_state)
            norm_parts.append(norm_part)
            sigmoid_parts.append(sigmoid_part)
            gated_norm_parts.append(gated_norm_part)
        q = _concat(mx, q_parts)
        k = _concat(mx, k_parts)
        compute_g = _concat(mx, g_parts)
        beta = _concat(mx, beta_parts)
        gated_output = _concat(mx, gated_parts)
        ssm_history = mx.stack(ssm_parts, axis=2)
        o_norm = _concat(mx, norm_parts)
        gate_sigmoid = _concat(mx, sigmoid_parts)
        gated_norm = _concat(mx, gated_norm_parts)
        output = attention.o_proj(gated_norm)
    else:
        q, k = _normalize_qk(mx, attention, q_raw, k_raw)
        beta = mx.sigmoid(b_logits)
        compute_g = _compute_g(mx, attention, a_logits)
        _eval_values(mx, (q, k, v, beta, compute_g))
        gated_output, ssm_state, ssm_history = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            attention.A_log.reshape(attention.num_heads, 1),
            attention.dt_bias.reshape(attention.num_heads, attention.head_dim),
            state=initial_ssm_state,
            mask=None,
            use_kernel=True,
            lower_bound=attention.lower_bound,
            return_state_history=True,
        )
        _eval_values(mx, (gated_output, ssm_state, ssm_history))
        reshaped = gated_output.reshape(
            1, WIDTH, attention.num_heads, attention.head_dim
        )
        o_norm = attention.o_norm(reshaped)
        gate_sigmoid = mx.sigmoid(gate_reshaped)
        gated_norm = (o_norm * gate_sigmoid).reshape(1, WIDTH, -1)
        output = attention.o_proj(gated_norm)
    _eval_values(
        mx,
        (
            q,
            k,
            v,
            compute_g,
            beta,
            gated_output,
            ssm_state,
            ssm_history,
            o_norm,
            gate_reshaped,
            gate_sigmoid,
            gated_norm,
            output,
        ),
    )

    # Target-only records are converted to comparisons in _pair_paths.
    stages = [
        Stage("qkv_projection", (projected_qkv,), (projected_qkv,)),
        Stage("conv_output", (qkv,), (qkv,)),
        Stage("conv_state_history", (conv_history,), (conv_history,)),
        Stage("conv_final_state", (conv_state,), (conv_state,)),
        Stage("q_rms", (q,), (q,)),
        Stage("k_rms", (k,), (k,)),
        Stage("value", (v,), (v,)),
        Stage("f_a", (f_a,), (f_a,)),
        Stage("a_logits", (a_logits,), (a_logits,)),
        Stage("b_logits", (b_logits,), (b_logits,)),
        Stage("full_rank_gate", (gate_reshaped,), (gate_reshaped,)),
        Stage("compute_g", (compute_g,), (compute_g,)),
        Stage("beta", (beta,), (beta,)),
        Stage("gated_delta_output", (gated_output,), (gated_output,)),
        Stage("gated_delta_state_history", (ssm_history,), (ssm_history,)),
        Stage("gated_delta_final_state", (ssm_state,), (ssm_state,)),
        Stage("o_norm", (o_norm,), (o_norm,)),
        Stage("gate_sigmoid", (gate_sigmoid,), (gate_sigmoid,)),
        Stage("o_norm_times_gate", (gated_norm,), (gated_norm,)),
        Stage("o_proj", (output,), (output,)),
    ]
    return PathCapture(stages, output, conv_state, ssm_state)


def _reference_path(
    *,
    mx: Any,
    attention: Any,
    shared_input: Any,
    initial_conv_state: Any,
    initial_ssm_state: Any,
) -> PathCapture:
    from mlx_lm.models.gated_delta import gated_delta_update
    from mlx_lm.models.kimi_k3_packed_kda_projections import (
        maybe_authoritative_packed_k3_kda_skinny,
        maybe_authoritative_packed_k3_kda_wide,
    )

    values: dict[str, list[Any]] = {
        name: []
        for name in (
            "qkv_projection",
            "conv_output",
            "q_rms",
            "k_rms",
            "value",
            "f_a",
            "a_logits",
            "b_logits",
            "full_rank_gate",
            "compute_g",
            "beta",
            "gated_delta_output",
            "o_norm",
            "gate_sigmoid",
            "o_norm_times_gate",
            "o_proj",
        )
    }
    conv_history: list[Any] = []
    ssm_history: list[Any] = []
    conv_state = initial_conv_state
    ssm_state = initial_ssm_state

    for position in range(WIDTH):
        token_input = shared_input[:, position : position + 1, :]
        packed_wide = maybe_authoritative_packed_k3_kda_wide(attention, token_input)
        packed_skinny = maybe_authoritative_packed_k3_kda_skinny(attention, token_input)
        if packed_wide is None or len(packed_wide) != 2:
            raise LocalizerError("accepted packed KDA wide path was unavailable")
        if packed_skinny is None or len(packed_skinny) != 2:
            raise LocalizerError("accepted packed KDA skinny path was unavailable")
        projected_qkv, gate = packed_wide
        f_a, b_logits = packed_skinny
        a_logits = attention.f_b_proj(f_a).reshape(
            1, 1, attention.num_heads, attention.head_dim
        )
        b_logits = b_logits.reshape(1, 1, attention.num_heads)
        _eval_values(mx, (projected_qkv, gate, f_a, a_logits, b_logits))

        qkv, conv_state = attention.qkv_conv(
            projected_qkv,
            conv_state,
            None,
            None,
        )
        _eval_values(mx, (qkv, conv_state))
        q_raw, k_raw, v = _split_qkv(attention, qkv)
        q, k = _normalize_qk(mx, attention, q_raw, k_raw)
        beta = mx.sigmoid(b_logits)
        compute_g = _compute_g(mx, attention, a_logits)
        _eval_values(mx, (q, k, v, beta, compute_g))

        gated_output, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            attention.A_log.reshape(attention.num_heads, 1),
            attention.dt_bias.reshape(attention.num_heads, attention.head_dim),
            state=ssm_state,
            mask=None,
            use_kernel=True,
            lower_bound=attention.lower_bound,
            return_state_history=False,
        )
        _eval_values(mx, (gated_output, ssm_state))

        reshaped = gated_output.reshape(1, 1, attention.num_heads, attention.head_dim)
        o_norm = attention.o_norm(reshaped)
        gate_reshaped = gate.reshape(1, 1, attention.num_heads, attention.head_dim)
        gate_sigmoid = mx.sigmoid(gate_reshaped)
        gated_norm = (o_norm * gate_sigmoid).reshape(1, 1, -1)
        output = attention.o_proj(gated_norm)
        _eval_values(mx, (o_norm, gate_reshaped, gate_sigmoid, gated_norm, output))

        step = {
            "qkv_projection": projected_qkv,
            "conv_output": qkv,
            "q_rms": q,
            "k_rms": k,
            "value": v,
            "f_a": f_a,
            "a_logits": a_logits,
            "b_logits": b_logits,
            "full_rank_gate": gate_reshaped,
            "compute_g": compute_g,
            "beta": beta,
            "gated_delta_output": gated_output,
            "o_norm": o_norm,
            "gate_sigmoid": gate_sigmoid,
            "o_norm_times_gate": gated_norm,
            "o_proj": output,
        }
        for name, value in step.items():
            values[name].append(value)
        conv_history.append(conv_state)
        ssm_history.append(ssm_state)

    sequence_values = {name: _concat(mx, parts) for name, parts in values.items()}
    conv_state_history = mx.stack(conv_history, axis=1)
    ssm_state_history = mx.stack(ssm_history, axis=2)
    _eval_values(
        mx,
        (
            *sequence_values.values(),
            conv_state_history,
            ssm_state_history,
            conv_state,
            ssm_state,
        ),
    )
    stages = [
        Stage(
            "qkv_projection",
            (sequence_values["qkv_projection"],),
            (sequence_values["qkv_projection"],),
        ),
        Stage(
            "conv_output",
            (sequence_values["conv_output"],),
            (sequence_values["conv_output"],),
        ),
        Stage("conv_state_history", (conv_state_history,), (conv_state_history,)),
        Stage("conv_final_state", (conv_state,), (conv_state,)),
        Stage("q_rms", (sequence_values["q_rms"],), (sequence_values["q_rms"],)),
        Stage("k_rms", (sequence_values["k_rms"],), (sequence_values["k_rms"],)),
        Stage("value", (sequence_values["value"],), (sequence_values["value"],)),
        Stage("f_a", (sequence_values["f_a"],), (sequence_values["f_a"],)),
        Stage(
            "a_logits", (sequence_values["a_logits"],), (sequence_values["a_logits"],)
        ),
        Stage(
            "b_logits", (sequence_values["b_logits"],), (sequence_values["b_logits"],)
        ),
        Stage(
            "full_rank_gate",
            (sequence_values["full_rank_gate"],),
            (sequence_values["full_rank_gate"],),
        ),
        Stage(
            "compute_g",
            (sequence_values["compute_g"],),
            (sequence_values["compute_g"],),
        ),
        Stage("beta", (sequence_values["beta"],), (sequence_values["beta"],)),
        Stage(
            "gated_delta_output",
            (sequence_values["gated_delta_output"],),
            (sequence_values["gated_delta_output"],),
        ),
        Stage("gated_delta_state_history", (ssm_state_history,), (ssm_state_history,)),
        Stage("gated_delta_final_state", (ssm_state,), (ssm_state,)),
        Stage("o_norm", (sequence_values["o_norm"],), (sequence_values["o_norm"],)),
        Stage(
            "gate_sigmoid",
            (sequence_values["gate_sigmoid"],),
            (sequence_values["gate_sigmoid"],),
        ),
        Stage(
            "o_norm_times_gate",
            (sequence_values["o_norm_times_gate"],),
            (sequence_values["o_norm_times_gate"],),
        ),
        Stage("o_proj", (sequence_values["o_proj"],), (sequence_values["o_proj"],)),
    ]
    return PathCapture(stages, sequence_values["o_proj"], conv_state, ssm_state)


def pair_paths(target_path: PathCapture, reference_path: PathCapture) -> list[Stage]:
    if len(target_path.stages) != len(reference_path.stages):
        raise LocalizerError("target and reference stage counts differ")
    paired = []
    for target_stage, reference_stage in zip(
        target_path.stages, reference_path.stages, strict=True
    ):
        if (
            target_stage.name != reference_stage.name
            or target_stage.component_names != reference_stage.component_names
        ):
            raise LocalizerError("target and reference stage order differs")
        paired.append(
            Stage(
                target_stage.name,
                target_stage.target_values,
                reference_stage.target_values,
                target_stage.component_names,
            )
        )
    return paired


def compare_stages(
    *,
    mx: Any,
    group: Any,
    stages: Sequence[Stage],
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    names = [
        f"{stage.name}.{component}"
        for stage in stages
        for component in stage.component_names
    ]
    left = [value for stage in stages for value in stage.target_values]
    right = [value for stage in stages for value in stage.reference_values]
    records = diagnostic._gather_named_array_metrics(
        mx=mx,
        group=group,
        names=names,
        left=left,
        right=right,
        limits=limits,
    )
    grouped = []
    offset = 0
    for stage in stages:
        count = len(stage.component_names)
        components = records[offset : offset + count]
        offset += count
        grouped.append(
            {
                "stage": stage.name,
                "exact_all_components_all_ranks": all(
                    record["exact_all_ranks"] for record in components
                ),
                "finite_all_components_all_ranks": all(
                    record["finite_all_ranks"] for record in components
                ),
                "within_v3_numeric_limits_all_components_all_ranks": all(
                    record["within_v3_numeric_limits_all_ranks"]
                    for record in components
                ),
                "worst_max_abs_error": max(
                    record["worst_max_abs_error"] for record in components
                ),
                "worst_cosine_similarity": min(
                    record["worst_cosine_similarity"] for record in components
                ),
                "components": components,
            }
        )
    first = next(
        (
            record["stage"]
            for record in grouped
            if not record["exact_all_components_all_ranks"]
        ),
        None,
    )
    return {
        "first_exact_divergent_stage": first,
        "exact_all_stages_all_ranks": first is None,
        "finite_all_stages_all_ranks": all(
            record["finite_all_components_all_ranks"] for record in grouped
        ),
        "stages": grouped,
    }


def _actual_target_endpoint(
    *,
    mx: Any,
    model: Any,
    attention: Any,
    shared_input: Any,
    base_cache: Sequence[Any],
) -> tuple[Any, Any, Any]:
    working = target.clone_k3_cache(base_cache, mx)
    transaction = model.language_model.begin_speculative_cache(working, WIDTH)
    try:
        output = attention(shared_input, mask=None, cache=working[0])
        conv_state, ssm_state = working[0]
        _eval_values(mx, (output, conv_state, ssm_state))
    finally:
        model.language_model.cancel_speculative_cache(transaction)
    return output, conv_state, ssm_state


def _actual_reference_endpoint(
    *,
    mx: Any,
    attention: Any,
    shared_input: Any,
    initial_conv_state: Any,
    initial_ssm_state: Any,
) -> tuple[Any, Any, Any]:
    conv_state = initial_conv_state
    ssm_state = initial_ssm_state
    outputs = []
    if attention._step is None:
        attention._step = mx.compile(attention._decode_core)
    for position in range(WIDTH):
        output, conv_state, ssm_state = attention._step(
            shared_input[:, position : position + 1, :],
            conv_state,
            ssm_state,
        )
        _eval_values(mx, (output, conv_state, ssm_state))
        outputs.append(output)
    return _concat(mx, outputs), conv_state, ssm_state


def validate_endpoint_reproduction(
    *,
    mx: Any,
    group: Any,
    actual_target: tuple[Any, Any, Any],
    actual_reference: tuple[Any, Any, Any],
    staged_target: PathCapture,
    staged_reference: PathCapture,
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    stages = [
        Stage(
            "target_staged_reproduces_real",
            tuple(actual_target),
            (
                staged_target.output,
                staged_target.final_conv_state,
                staged_target.final_ssm_state,
            ),
            ("output", "conv_state", "ssm_state"),
        ),
        Stage(
            "reference_staged_reproduces_real",
            tuple(actual_reference),
            (
                staged_reference.output,
                staged_reference.final_conv_state,
                staged_reference.final_ssm_state,
            ),
            ("output", "conv_state", "ssm_state"),
        ),
    ]
    result = compare_stages(mx=mx, group=group, stages=stages, limits=limits)
    if not result["exact_all_stages_all_ranks"]:
        raise LocalizerError(
            "staged localizer does not exactly reproduce real attention endpoints"
        )
    return result


def _hidden_sha256(mx: Any, value: Any) -> str:
    import numpy as np

    raw = value.view(mx.uint8)
    mx.eval(raw)
    return hashlib.sha256(
        np.asarray(raw, dtype=np.uint8).tobytes(order="C")
    ).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.kimi_k3 import ResidualBlocks

    runtime_flags = require_runtime_flags()
    init_started = time.perf_counter()
    group = base.init_distributed(mx, backend="jaccl")
    rank = int(group.rank())
    world_size = int(group.size())
    if world_size != base.WORLD_SIZE:
        raise LocalizerError(f"expected JACCL TP2, got TP{world_size}")
    init_seconds = time.perf_counter() - init_started

    transport = base.inspect_jaccl_transport(rank=rank)
    mode_attestation = target.attest_shared_transport_mode(mx, group, transport)
    matrix_digests = base.require_shared_digest(
        mx, group, transport["device_matrix_sha256"], "JACCL device matrix"
    )
    contract_digests = base.require_shared_digest(
        mx,
        group,
        transport["transport_contract_sha256"],
        "JACCL transport contract",
    )
    model_dir = base.select_rank_checkpoint(
        args.rank_checkpoint, rank=rank, world_size=world_size
    )
    manifest = base.inspect_pinned_manifest(model_dir, rank=rank)
    manifest_digests = base.gather_digests(mx, group, manifest["manifest_sha256"])
    runtime_source = base.inspect_runtime_k3_source()
    runtime_digests = base.require_shared_digest(
        mx, group, runtime_source["sha256"], "K3 runtime source"
    )

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
        mx, group, [load_seconds, float(mx.get_peak_memory())]
    )

    text_model, layers = diagnostic._active_k3_parts(model)
    layer = layers[0]
    attention = layer.self_attn
    if not bool(layer.is_linear) or type(attention).__name__ != "KimiK3DeltaAttention":
        raise LocalizerError("layer 0 is not the pinned Kimi K3 KDA layer")
    if not bool(attention.use_full_rank_gate):
        raise LocalizerError("real checkpoint does not expose the full-rank KDA gate")

    prompt_ids, prompt_mode = base.build_prompt_tokens(
        tokenizer, base.DEFAULT_PROMPT, PROMPT_TOKEN_TARGET
    )
    if len(prompt_ids) != PROMPT_TOKEN_TARGET:
        raise LocalizerError(
            f"prompt builder produced {len(prompt_ids)} tokens, expected exactly "
            f"{PROMPT_TOKEN_TARGET}"
        )
    prompt_digest = base.token_digest(prompt_ids)
    prompt_digests = base.require_shared_digest(
        mx, group, prompt_digest, "prompt token IDs"
    )
    prompt_input = mx.array([prompt_ids], dtype=mx.uint32)
    mx.eval(prompt_input)

    with wired_limit(model):
        base_cache = model.make_cache()
        base.barrier(mx, group)
        prefill_started = time.perf_counter()
        prefill_logits = model(prompt_input, cache=base_cache)
        mx.eval(prefill_logits)
        mx.synchronize()
        prefill_seconds = time.perf_counter() - prefill_started
        prefill_rows = base.gather_floats(
            mx, group, [prefill_seconds, float(mx.get_peak_memory())]
        )
        layout = target.cache_layout(base_cache)
        if layout["kv_offset"] != PROMPT_TOKEN_TARGET:
            raise LocalizerError("post-prefill MLA offset does not equal 128")
        first_token_array = mx.argmax(prefill_logits[0, -1], axis=-1)
        mx.eval(first_token_array)
        first_token = int(first_token_array.item())
        del prefill_logits

        immutable_guard = target.clone_k3_cache(base_cache, mx)
        known_ids = target._known_continuation(
            mx=mx,
            model=model,
            base_cache=base_cache,
            first_token=first_token,
            count=WIDTH + 1,
        )
        known_digest = base.token_digest(known_ids)
        known_digests = base.require_shared_digest(
            mx, group, known_digest, "localizer known continuation"
        )

        token_input = mx.array([known_ids[:WIDTH]], dtype=mx.uint32)
        embedded = text_model.embed_tokens(token_input)
        blocks = ResidualBlocks(layer.eps)
        shared_input, _, _ = layer._prepare_attention(embedded, blocks)
        _eval_values(mx, (shared_input,))
        if list(shared_input.shape[:2]) != [1, WIDTH]:
            raise LocalizerError("shared layer-0 attention input has wrong shape")
        reconstructed = mx.concatenate(
            [shared_input[:, position : position + 1, :] for position in range(WIDTH)],
            axis=1,
        )
        mx.eval(reconstructed)
        if not bool(mx.array_equal(shared_input, reconstructed).item()):
            raise LocalizerError("T=1 references do not share the exact T=2 input")
        hidden_digest = _hidden_sha256(mx, shared_input)
        hidden_digests = base.gather_digests(mx, group, hidden_digest)

        target_initial = target.clone_k3_cache(base_cache, mx)
        reference_initial = target.clone_k3_cache(base_cache, mx)
        target.assert_cache_value_equivalent(target_initial, reference_initial, mx)
        target_conv, target_ssm = target_initial[0]
        reference_conv, reference_ssm = reference_initial[0]

        # Known-continuation generation has initialized the accepted T=1 packs.
        actual_reference = _actual_reference_endpoint(
            mx=mx,
            attention=attention,
            shared_input=shared_input,
            initial_conv_state=reference_conv,
            initial_ssm_state=reference_ssm,
        )
        actual_target = _actual_target_endpoint(
            mx=mx,
            model=model,
            attention=attention,
            shared_input=shared_input,
            base_cache=base_cache,
        )
        staged_target = _target_path(
            mx=mx,
            attention=attention,
            shared_input=shared_input,
            initial_conv_state=target_conv,
            initial_ssm_state=target_ssm,
        )
        staged_reference = _reference_path(
            mx=mx,
            attention=attention,
            shared_input=shared_input,
            initial_conv_state=reference_conv,
            initial_ssm_state=reference_ssm,
        )
        limits = target.EquivalenceLimits(
            min_cosine=args.min_cosine,
            max_abs_error=args.max_abs_error,
        )
        endpoint_reproduction = validate_endpoint_reproduction(
            mx=mx,
            group=group,
            actual_target=actual_target,
            actual_reference=actual_reference,
            staged_target=staged_target,
            staged_reference=staged_reference,
            limits=limits,
        )
        paired = pair_paths(staged_target, staged_reference)
        stage_comparison = compare_stages(
            mx=mx, group=group, stages=paired, limits=limits
        )
        target.assert_cache_value_equivalent(immutable_guard, base_cache, mx)

    status = (
        COMPLETE_NO_DIVERGENCE
        if stage_comparison["exact_all_stages_all_ranks"]
        else COMPLETE_DIVERGENCE
    )
    artifact = {
        "schema": SCHEMA,
        "status": status,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": {
            "authoritative_target_verification_schema": target.ARTIFACT_SCHEMA,
            "authoritative_gate_unchanged": True,
            "diagnostic_status_is_non_promotional": True,
            "one_model_load_per_rank": True,
        },
        "source": {
            "repo": base.SOURCE_REPO,
            "revision": base.SOURCE_REVISION,
            "effective_revision": base.EFFECTIVE_SOURCE_REVISION,
            "config_sha256": base.SOURCE_CONFIG_SHA256,
            "index_sha256": base.SOURCE_INDEX_SHA256,
        },
        "runtime": {
            **target.runtime_contract_record(
                runtime_source=runtime_source,
                runtime_digests=runtime_digests,
                attestation=runtime_attestation,
            ),
            "flags": runtime_flags,
            "mlx_version": str(mx.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "distributed": {
            "backend": mode_attestation["mode"],
            "world_size": world_size,
            "init_seconds_rank0": init_seconds,
            "checkpoint_manifest_sha256_by_rank": manifest_digests,
            "load_seconds_by_rank": [row[0] for row in load_rows],
            "load_peak_memory_gb_by_rank": [row[1] / 1e9 for row in load_rows],
            "device_matrix_sha256_by_rank": matrix_digests,
            "transport_contract_sha256_by_rank": contract_digests,
        },
        "diagnostic": {
            "layer": 0,
            "attention_class": type(attention).__name__,
            "full_rank_gate": True,
            "width": WIDTH,
            "prompt_mode": prompt_mode,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": prompt_digest,
            "prompt_token_sha256_by_rank": prompt_digests,
            "prefill_seconds_critical_path": max(row[0] for row in prefill_rows),
            "prefill_peak_memory_gb_by_rank": [row[1] / 1e9 for row in prefill_rows],
            "known_continuation_token_ids": known_ids,
            "known_continuation_sha256": known_digest,
            "known_continuation_sha256_by_rank": known_digests,
            "shared_post_prefill_hidden_input": {
                "shape": [int(value) for value in shared_input.shape],
                "dtype": str(shared_input.dtype),
                "sha256_by_rank": hidden_digests,
                "target_and_t1_slices_exact": True,
            },
            "base_cache_unchanged": True,
            "endpoint_reproduction": endpoint_reproduction,
            "stage_comparison": stage_comparison,
        },
    }
    if rank == 0:
        base.atomic_write_json(args.artifact, artifact)
        print(
            "K3_KDA_STAGE_LOCALIZER_RESULT "
            + json.dumps(
                {
                    "artifact": str(args.artifact.expanduser().resolve()),
                    "status": status,
                    "first_exact_divergent_stage": stage_comparison[
                        "first_exact_divergent_stage"
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    del model
    gc.collect()
    return {"status": status, "artifact": artifact if rank == 0 else None}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-checkpoint Kimi K3 layer-0 width-2 KDA localizer"
    )
    parser.add_argument(
        "--rank-checkpoint", action="append", required=True, metavar="PATH"
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--min-cosine", default=0.999, type=float)
    parser.add_argument("--max-abs-error", default=1.0, type=float)
    parser.add_argument("--verify-file-hashes", action="store_true")
    args = parser.parse_args(argv)
    if len(args.rank_checkpoint) != base.WORLD_SIZE:
        parser.error(
            f"--rank-checkpoint must be supplied exactly {base.WORLD_SIZE} times"
        )
    try:
        target.EquivalenceLimits(
            min_cosine=args.min_cosine,
            max_abs_error=args.max_abs_error,
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
        error: BaseException = LocalizerError("interrupted")
    except BaseException as exc:
        error = exc
    print(
        "K3_KDA_STAGE_LOCALIZER_ERROR "
        + json.dumps(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "rank": int(rank_text) if rank_text and rank_text.isdigit() else None,
                "status": "ERROR",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )
    if os.environ.get("K3_KDA_STAGE_LOCALIZER_TRACEBACK") == "1":
        traceback.print_exception(error, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
