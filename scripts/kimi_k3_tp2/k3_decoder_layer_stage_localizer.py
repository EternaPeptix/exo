#!/usr/bin/env python3
"""Fail-closed real-checkpoint Kimi K3 layer-0 width-two localizer.

The authoritative TP2 target diagnostic first diverges at decoder layer 0,
while the narrower KDA diagnostic is exact when both paths receive one shared
``_prepare_attention`` result.  That shared-input experiment deliberately
excluded the decoder wrapper.  This companion diagnostic closes that gap.

It performs one 128-token prefill, derives two known continuation tokens, and
compares a real T=2 layer-0 execution with two real T=1 executions at every
wrapper boundary: embedding, attention preparation and residual-block state,
KDA output/cache, AttnRes/RMS input to the MoE, routed/shared MoE stages, and
the final residual.  Controlled probes then hold the prepared KDA input or the
finish-attention input constant.  The KDA probe deliberately distinguishes the
ordinary non-checkpointed T=2 path used by the verifier from the speculative
history-producing T=2 path used by cache transactions; conflating those paths
was the blind spot in the earlier KDA localizer.  Finally, both staged layer
paths must exactly reproduce layer 0 captured from complete model forwards. A
mismatch in that reproduction is an ERROR and cannot produce a completed
artifact.

This is diagnostic only.  It does not change or satisfy the authoritative
``k3-tp2-target-verification/v5`` PASS/FAIL contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import k3_kda_stage_localizer as kda
import k3_target_diagnostic as diagnostic
import k3_target_verify as target
import tp2_benchmark as base

SCHEMA = "k3-tp2-decoder-layer-width2-stage-localizer/v1"
PROMPT_TOKEN_TARGET = 128
WIDTH = 2
LAYER_INDEX = 0
COMPLETE_DIVERGENCE = "COMPLETE_DIVERGENCE"
COMPLETE_NO_DIVERGENCE = "COMPLETE_NO_DIVERGENCE"


class LocalizerError(diagnostic.DiagnosticError):
    """A fail-closed decoder-layer stage-localization error."""


@dataclass(frozen=True)
class PathStage:
    """One ordered numerical boundary from one execution path.

    ``token_axes`` says how T=1 values are reassembled.  ``None`` means the
    last T=1 value is the final state and is compared directly with the T=2
    final state (for example, recurrent cache state).
    """

    name: str
    values: tuple[Any, ...]
    component_names: tuple[str, ...] = ("value",)
    token_axes: tuple[int | None, ...] = (1,)

    def __post_init__(self) -> None:
        if not self.name:
            raise LocalizerError("stage name cannot be empty")
        if (
            not self.values
            or len(self.values) != len(self.component_names)
            or len(self.values) != len(self.token_axes)
        ):
            raise LocalizerError(f"stage {self.name!r} has inconsistent components")
        if len(set(self.component_names)) != len(self.component_names):
            raise LocalizerError(f"stage {self.name!r} repeats a component name")


@dataclass
class FinishCapture:
    stages: list[PathStage]
    output: Any
    provenance: dict[str, str]
    residual_consumed: bool


@dataclass
class LayerCapture:
    stages: list[PathStage]
    output: Any
    attention_input: Any
    attention_output: Any
    blocks: Any
    final_conv_state: Any
    final_ssm_state: Any
    finish: FinishCapture
    provenance: dict[str, str]


def require_runtime_flags() -> dict[str, str]:
    """Require exactly the accepted v6 runtime, with no candidate rewrite."""

    actual = kda.require_runtime_flags()
    if actual[kda.CANDIDATE_EXACT_ENV] != "0":
        raise LocalizerError(
            f"{kda.CANDIDATE_EXACT_ENV} must be 0 for the accepted v6 control"
        )
    return actual


def _eval_values(mx: Any, values: Iterable[Any]) -> None:
    flat = list(values)
    if not flat:
        raise LocalizerError("refusing to evaluate an empty stage set")
    if any(value is None for value in flat):
        raise LocalizerError("a required stage value is missing")
    mx.eval(*flat)
    mx.synchronize()


def _add_stage(
    stages: list[PathStage],
    name: str,
    values: Any | Sequence[Any],
    *,
    component_names: Sequence[str] = ("value",),
    token_axes: Sequence[int | None] = (1,),
) -> None:
    if not isinstance(values, (tuple, list)):
        values = (values,)
    stage = PathStage(
        name=name,
        values=tuple(values),
        component_names=tuple(component_names),
        token_axes=tuple(token_axes),
    )
    _eval_values_for_stage(stage)
    stages.append(stage)


def _eval_values_for_stage(stage: PathStage) -> None:
    # Evaluation happens explicitly in the execution functions, where ``mx``
    # is available.  This helper only makes accidental None values fail before
    # a stage can be recorded.
    if any(value is None for value in stage.values):
        raise LocalizerError(f"stage {stage.name!r} has a missing value")


def combine_sequential_stages(
    mx: Any,
    paths: Sequence[Sequence[PathStage]],
) -> list[PathStage]:
    """Reassemble exactly WIDTH T=1 traces into one T=2-shaped trace."""

    if len(paths) != WIDTH:
        raise LocalizerError(f"expected {WIDTH} sequential traces, got {len(paths)}")
    if not paths[0]:
        raise LocalizerError("sequential trace has no stages")
    expected_count = len(paths[0])
    if any(len(path) != expected_count for path in paths):
        raise LocalizerError("sequential traces have different stage counts")

    combined: list[PathStage] = []
    for stage_index in range(expected_count):
        reference = paths[0][stage_index]
        peers = [path[stage_index] for path in paths]
        for peer in peers[1:]:
            if (
                peer.name != reference.name
                or peer.component_names != reference.component_names
                or peer.token_axes != reference.token_axes
            ):
                raise LocalizerError("sequential stage descriptors differ")
        values = []
        for component_index, axis in enumerate(reference.token_axes):
            parts = [peer.values[component_index] for peer in peers]
            values.append(
                parts[-1] if axis is None else mx.concatenate(parts, axis=axis)
            )
        combined.append(
            PathStage(
                reference.name,
                tuple(values),
                reference.component_names,
                reference.token_axes,
            )
        )
    _eval_values(mx, (value for stage in combined for value in stage.values))
    return combined


def pair_stages(
    target_stages: Sequence[PathStage],
    reference_stages: Sequence[PathStage],
) -> list[kda.Stage]:
    if len(target_stages) != len(reference_stages):
        raise LocalizerError("target and reference stage counts differ")
    paired = []
    for left, right in zip(target_stages, reference_stages, strict=True):
        if (
            left.name != right.name
            or left.component_names != right.component_names
            or left.token_axes != right.token_axes
        ):
            raise LocalizerError("target and reference stage descriptors differ")
        paired.append(
            kda.Stage(
                left.name,
                left.values,
                right.values,
                left.component_names,
            )
        )
    return paired


def _run_sparse_moe(
    *,
    mx: Any,
    kimi: Any,
    moe: Any,
    x: Any,
    residual: Any,
) -> tuple[list[PathStage], Any, bool, dict[str, str]]:
    """Replay the accepted SparseMoE call while exposing exact boundaries."""

    stages: list[PathStage] = []
    provenance: dict[str, str] = {}

    if moe.sharding_group is not None:
        x = kimi.sum_gradients(moe.sharding_group)(x)
        input_provider = "sum_gradients"
    else:
        input_provider = "identity"
    _eval_values(mx, (x,))
    _add_stage(stages, "moe.input", x)
    provenance["input"] = input_provider

    optimized_front = kimi.maybe_multibank_k3_moe_front(moe, x)
    front_provider = "multibank"
    if optimized_front is None:
        optimized_front = kimi.maybe_authoritative_packed_k3_moe_front(moe, x)
        front_provider = "authoritative_packed"
    if optimized_front is None:
        optimized_front = kimi.maybe_packed_k3_moe_front(moe, x)
        front_provider = "packed"
    if optimized_front is None:
        scores = moe.gate(x)
        routed_latent = (
            moe.routed_expert_down_proj(x) if moe.latent_size is not None else x
        )
        shared_gate = None
        shared_up = None
        front_provider = "fallback"
    else:
        if len(optimized_front) != 4:
            raise LocalizerError("optimized MoE front returned the wrong arity")
        shared_gate, shared_up, scores, routed_latent = optimized_front
    _eval_values(mx, (scores, routed_latent))
    provenance["front"] = front_provider

    routed = kimi.maybe_fused_k3_router(
        scores,
        moe.e_score_correction_bias,
        top_k=moe.args.num_experts_per_token,
        n_group=moe.args.num_expert_group,
        topk_group=moe.args.topk_group,
        routed_scaling_factor=moe.args.routed_scaling_factor,
        renormalize=moe.args.moe_renormalize,
        training=getattr(moe, "training", True),
    )
    if routed is None:
        inds, weights = kimi._group_expert_select(
            scores,
            moe.e_score_correction_bias,
            moe.args.num_experts_per_token,
            moe.args.num_expert_group,
            moe.args.topk_group,
            moe.args.routed_scaling_factor,
            moe.args.moe_renormalize,
        )
        router_provider = "fallback"
    else:
        inds, weights = routed
        router_provider = "fused"
    _eval_values(mx, (inds, weights))
    provenance["router"] = router_provider

    fused_reduced = kimi.maybe_fused_k3_switch_glu_reduce(
        moe.switch_mlp,
        routed_latent,
        inds,
        weights,
    )
    if fused_reduced is None:
        fused_experts = kimi.maybe_fused_k3_switch_glu(
            moe.switch_mlp,
            routed_latent,
            inds,
        )
        if fused_experts is None:
            expert_rows = moe.switch_mlp(routed_latent, inds)
            expert_provider = "fallback"
        else:
            expert_rows = fused_experts
            expert_provider = "fused_experts"
        routed_reduced = (expert_rows * weights[..., None]).sum(axis=-2)
    else:
        routed_reduced = fused_reduced
        expert_provider = "fused_reduce"
    _eval_values(mx, (routed_reduced,))
    provenance["experts"] = expert_provider

    if moe.shared_experts is None:
        raise LocalizerError("layer-0 checkpoint has no shared experts")
    if optimized_front is None:
        shared_gate = moe.shared_experts.gate_proj(x)
        shared_up = moe.shared_experts.up_proj(x)
    if shared_gate is None or shared_up is None:
        raise LocalizerError("MoE front did not expose shared projections")
    shared = moe.shared_experts.down_proj(
        kimi._situ(
            shared_up,
            shared_gate,
            moe.shared_experts.beta,
            moe.shared_experts.linear_beta,
        )
    )
    _eval_values(mx, (shared_gate, shared_up, shared))

    _add_stage(
        stages,
        "moe.front",
        (shared_gate, shared_up, scores, routed_latent),
        component_names=("shared_gate", "shared_up", "router_scores", "routed_latent"),
        token_axes=(1, 1, 1, 1),
    )
    _add_stage(
        stages,
        "moe.router",
        (inds, weights),
        component_names=("expert_indices", "expert_weights"),
        token_axes=(1, 1),
    )
    _add_stage(stages, "moe.routed_expert_reduce", routed_reduced)
    _add_stage(stages, "moe.shared_expert", shared)

    if moe.sharding_group is not None:
        split = routed_reduced.shape[-1]
        combined = mx.distributed.all_sum(
            mx.concatenate([routed_reduced, shared], axis=-1),
            group=moe.sharding_group,
        )
        routed_reduced, shared = mx.split(combined, [split], axis=-1)
        collective_provider = "concatenated_all_sum"
    else:
        collective_provider = "identity"
    _eval_values(mx, (routed_reduced, shared))
    provenance["collective"] = collective_provider
    _add_stage(
        stages,
        "moe.collective",
        (routed_reduced, shared),
        component_names=("routed", "shared"),
        token_axes=(1, 1),
    )

    if moe.routed_expert_norm is not None:
        routed_norm = moe.routed_expert_norm(routed_reduced)
        norm_provider = "rms_norm"
    else:
        routed_norm = routed_reduced
        norm_provider = "identity"
    _eval_values(mx, (routed_norm,))
    provenance["routed_norm"] = norm_provider
    _add_stage(stages, "moe.routed_norm", routed_norm)

    residual_consumed = False
    output = routed_norm
    if moe.latent_size is not None:
        fused_up_add = kimi.maybe_fused_k3_routed_up_add(
            moe,
            output,
            shared,
            residual,
        )
        if fused_up_add is None:
            output = moe.routed_expert_up_proj(output)
            up_provider = "routed_up_projection"
        else:
            output = fused_up_add
            shared = None
            residual_consumed = True
            up_provider = "fused_routed_up_shared_residual_add"
    else:
        up_provider = "identity"
    if shared is not None:
        output = output + shared
    _eval_values(mx, (output,))
    provenance["up_and_residual"] = up_provider
    _add_stage(stages, "moe.output", output)
    return stages, output, residual_consumed, provenance


def _direct_kda_path(
    *,
    mx: Any,
    attention: Any,
    shared_input: Any,
    initial_conv_state: Any,
    initial_ssm_state: Any,
) -> kda.PathCapture:
    """Replay the verifier's ordinary T=2 KDA path without history outputs."""

    from mlx_lm.models.gated_delta import gated_delta_update

    projected_qkv = attention.qkv_proj(shared_input)
    gate = attention.g_proj(shared_input)
    f_a = attention.f_a_proj(shared_input)
    a_logits = attention.f_b_proj(f_a).reshape(
        1, WIDTH, attention.num_heads, attention.head_dim
    )
    b_logits = attention.b_proj(shared_input).reshape(1, WIDTH, attention.num_heads)
    _eval_values(mx, (projected_qkv, gate, f_a, a_logits, b_logits))

    # This is intentionally return_state_history=False.  The previous KDA
    # localizer enabled a speculative cache transaction, which selects the
    # history-producing kernels and therefore did not reproduce the verifier's
    # direct model(T=2) call.
    qkv, conv_state = attention.qkv_conv(
        projected_qkv,
        initial_conv_state,
        None,
        None,
        return_state_history=False,
    )
    _eval_values(mx, (qkv, conv_state))
    q_raw, k_raw, value = kda._split_qkv(attention, qkv)
    q, key = kda._normalize_qk(mx, attention, q_raw, k_raw)
    beta = mx.sigmoid(b_logits)
    compute_g = kda._compute_g(mx, attention, a_logits)
    _eval_values(mx, (q, key, value, beta, compute_g))

    gated_output, ssm_state = gated_delta_update(
        q,
        key,
        value,
        a_logits,
        b_logits,
        attention.A_log.reshape(attention.num_heads, 1),
        attention.dt_bias.reshape(attention.num_heads, attention.head_dim),
        state=initial_ssm_state,
        mask=None,
        use_kernel=True,
        lower_bound=attention.lower_bound,
        return_state_history=False,
    )
    _eval_values(mx, (gated_output, ssm_state))
    reshaped = gated_output.reshape(1, WIDTH, attention.num_heads, attention.head_dim)
    o_norm = attention.o_norm(reshaped)
    gate_reshaped = gate.reshape(1, WIDTH, attention.num_heads, attention.head_dim)
    gate_sigmoid = mx.sigmoid(gate_reshaped)
    gated_norm = (o_norm * gate_sigmoid).reshape(1, WIDTH, -1)
    output = attention.o_proj(gated_norm)
    _eval_values(
        mx,
        (
            o_norm,
            gate_reshaped,
            gate_sigmoid,
            gated_norm,
            output,
        ),
    )

    values = {
        "qkv_projection": projected_qkv,
        "conv_output": qkv,
        "conv_final_state": conv_state,
        "q_rms": q,
        "k_rms": key,
        "value": value,
        "f_a": f_a,
        "a_logits": a_logits,
        "b_logits": b_logits,
        "full_rank_gate": gate_reshaped,
        "compute_g": compute_g,
        "beta": beta,
        "gated_delta_output": gated_output,
        "gated_delta_final_state": ssm_state,
        "o_norm": o_norm,
        "gate_sigmoid": gate_sigmoid,
        "o_norm_times_gate": gated_norm,
        "o_proj": output,
    }
    stages = [kda.Stage(name, (value,), (value,)) for name, value in values.items()]
    return kda.PathCapture(stages, output, conv_state, ssm_state)


def _actual_direct_attention_endpoint(
    *,
    mx: Any,
    attention: Any,
    shared_input: Any,
    base_cache: Sequence[Any],
) -> tuple[Any, Any, Any]:
    """Execute the exact non-transactional T=2 attention call under test."""

    working = target.clone_k3_cache(base_cache, mx)
    output = attention(shared_input, mask=None, cache=working[LAYER_INDEX])
    conv_state, ssm_state = working[LAYER_INDEX]
    _eval_values(mx, (output, conv_state, ssm_state))
    return output, conv_state, ssm_state


def _pair_common_kda_stages(
    target_path: kda.PathCapture,
    reference_path: kda.PathCapture,
) -> list[kda.Stage]:
    """Pair direct-path stages, excluding histories it does not produce."""

    reference = {stage.name: stage for stage in reference_path.stages}
    paired = []
    for stage in target_path.stages:
        peer = reference.get(stage.name)
        if peer is None:
            raise LocalizerError(f"reference KDA path lacks stage {stage.name!r}")
        if stage.component_names != peer.component_names:
            raise LocalizerError("KDA stage component contracts differ")
        paired.append(
            kda.Stage(
                stage.name,
                stage.target_values,
                peer.target_values,
                stage.component_names,
            )
        )
    return paired


def _validate_direct_kda_reproduction(
    *,
    mx: Any,
    group: Any,
    actual_target: tuple[Any, Any, Any],
    actual_reference: tuple[Any, Any, Any],
    staged_target: kda.PathCapture,
    staged_reference: kda.PathCapture,
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    result = kda.compare_stages(
        mx=mx,
        group=group,
        stages=[
            kda.Stage(
                "direct_target_staged_reproduces_real",
                tuple(actual_target),
                (
                    staged_target.output,
                    staged_target.final_conv_state,
                    staged_target.final_ssm_state,
                ),
                ("output", "conv_state", "ssm_state"),
            ),
            kda.Stage(
                "reference_staged_reproduces_real",
                tuple(actual_reference),
                (
                    staged_reference.output,
                    staged_reference.final_conv_state,
                    staged_reference.final_ssm_state,
                ),
                ("output", "conv_state", "ssm_state"),
            ),
        ],
        limits=limits,
    )
    if not result["exact_all_stages_all_ranks"]:
        raise LocalizerError(
            "staged direct KDA paths do not exactly reproduce real endpoints"
        )
    return result


def _run_finish(
    *,
    mx: Any,
    kimi: Any,
    layer: Any,
    partial_sum: Any | None,
    attention_output: Any,
    blocks: Any,
) -> FinishCapture:
    stages: list[PathStage] = []
    partial_after_attention = (
        attention_output if partial_sum is None else partial_sum + attention_output
    )
    _eval_values(mx, (partial_after_attention,))
    _add_stage(stages, "finish.partial_sum_after_attention", partial_after_attention)

    mlp_input = layer._mix_and_norm(
        blocks,
        partial_after_attention,
        layer._mlp_res_w_eff,
        layer.post_attention_layernorm,
    )
    _eval_values(mx, (mlp_input,))
    _add_stage(stages, "finish.mlp_input", mlp_input)

    if type(layer.mlp).__name__ != "KimiK3SparseMoE":
        raise LocalizerError("layer 0 does not expose the pinned KimiK3SparseMoE")
    moe_stages, mlp_output, residual_consumed, provenance = _run_sparse_moe(
        mx=mx,
        kimi=kimi,
        moe=layer.mlp,
        x=mlp_input,
        residual=partial_after_attention,
    )
    stages.extend(moe_stages)
    layer_output = (
        mlp_output if residual_consumed else partial_after_attention + mlp_output
    )
    _eval_values(mx, (layer_output,))
    _add_stage(stages, "finish.layer_output", layer_output)
    return FinishCapture(
        stages=stages,
        output=layer_output,
        provenance=provenance,
        residual_consumed=residual_consumed,
    )


def _run_layer(
    *,
    mx: Any,
    kimi: Any,
    text_model: Any,
    layer: Any,
    layer_cache: Any,
    token_input: Any,
) -> LayerCapture:
    stages: list[PathStage] = []
    embedded = text_model.embed_tokens(token_input)
    _eval_values(mx, (embedded,))
    _add_stage(stages, "embedding", embedded)

    blocks = kimi.ResidualBlocks(layer.eps)
    if not layer.use_attn_res or not layer.is_block_start or blocks.raw is not None:
        raise LocalizerError("layer 0 does not have the pinned empty AttnRes block")

    # With an empty block, _attn_res_mix is exactly the embedding.  Recording
    # this identity separately distinguishes RMSNorm from AttnRes mixing.
    _add_stage(stages, "prepare.attnres_mix", embedded)
    attention_input, partial_sum, blocks = layer._prepare_attention(embedded, blocks)
    if partial_sum is not None:
        raise LocalizerError("layer-0 block start retained a partial sum")
    if blocks.raw is None or blocks.inv_rms is None:
        raise LocalizerError("layer-0 prepare did not append its residual block")
    _eval_values(mx, (attention_input, blocks.raw, blocks.inv_rms))
    _add_stage(stages, "prepare.attention_input_rmsnorm", attention_input)
    _add_stage(
        stages,
        "prepare.blocks",
        (blocks.raw, blocks.inv_rms),
        component_names=("raw", "inv_rms"),
        token_axes=(2, 2),
    )

    attention_output = layer.self_attn(attention_input, mask=None, cache=layer_cache)
    conv_state, ssm_state = layer_cache
    _eval_values(mx, (attention_output, conv_state, ssm_state))
    _add_stage(stages, "attention.output", attention_output)
    _add_stage(
        stages,
        "attention.cache",
        (conv_state, ssm_state),
        component_names=("conv_state", "ssm_state"),
        token_axes=(None, None),
    )

    finish = _run_finish(
        mx=mx,
        kimi=kimi,
        layer=layer,
        partial_sum=partial_sum,
        attention_output=attention_output,
        blocks=blocks,
    )
    stages.extend(finish.stages)
    return LayerCapture(
        stages=stages,
        output=finish.output,
        attention_input=attention_input,
        attention_output=attention_output,
        blocks=blocks,
        final_conv_state=conv_state,
        final_ssm_state=ssm_state,
        finish=finish,
        provenance=finish.provenance,
    )


def _slice_blocks(kimi: Any, blocks: Any, position: int) -> Any:
    if blocks.raw is None or blocks.inv_rms is None:
        raise LocalizerError("cannot slice an empty residual block")
    sliced = kimi.ResidualBlocks(blocks.eps)
    sliced.raw = blocks.raw[:, :, position : position + 1, :]
    sliced.inv_rms = blocks.inv_rms[:, :, position : position + 1]
    return sliced


def _compare(
    *,
    mx: Any,
    group: Any,
    target_stages: Sequence[PathStage],
    reference_stages: Sequence[PathStage],
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    return kda.compare_stages(
        mx=mx,
        group=group,
        stages=pair_stages(target_stages, reference_stages),
        limits=limits,
    )


def infer_localization(
    *,
    natural: dict[str, Any],
    controlled_direct_attention: dict[str, Any],
    controlled_speculative_attention: dict[str, Any],
    controlled_finish: dict[str, Any],
) -> dict[str, Any]:
    """Turn measured stage results into a conservative localization label."""

    first = natural.get("first_exact_divergent_stage")
    direct_attention_exact = bool(
        controlled_direct_attention.get("exact_all_stages_all_ranks")
    )
    speculative_attention_exact = bool(
        controlled_speculative_attention.get("exact_all_stages_all_ranks")
    )
    finish_exact = bool(controlled_finish.get("exact_all_stages_all_ranks"))
    if first is None:
        scope = "none"
    elif first == "embedding":
        scope = "token_embedding_width_shape"
    elif first == "prepare.attnres_mix":
        scope = "layer0_attnres_prepare_mix"
    elif first == "prepare.attention_input_rmsnorm":
        scope = "layer0_input_rmsnorm_width_shape"
    elif first == "prepare.blocks":
        scope = "layer0_residual_block_inv_rms"
    elif first.startswith("attention."):
        if not direct_attention_exact and speculative_attention_exact:
            scope = "kda_direct_nonhistory_path"
        elif not direct_attention_exact:
            scope = "kda_attention_core"
        else:
            scope = "upstream_prepare_propagation_into_kda"
    elif first.startswith(("finish.", "moe.")):
        scope = (
            "finish_attention_or_moe_width_shape"
            if not finish_exact
            else "upstream_attention_propagation_into_finish"
        )
    else:
        scope = "unknown_measured_stage"
    return {
        "first_exact_divergent_stage": first,
        "inferred_scope": scope,
        "shared_prepared_input_direct_attention_exact": direct_attention_exact,
        "shared_prepared_input_speculative_attention_exact": (
            speculative_attention_exact
        ),
        "first_exact_divergent_direct_kda_stage": (
            controlled_direct_attention.get("first_exact_divergent_stage")
        ),
        "shared_finish_input_exact": finish_exact,
    }


def _full_model_endpoint_reproduction(
    *,
    mx: Any,
    group: Any,
    model: Any,
    base_cache: Sequence[Any],
    token_ids: Sequence[int],
    target_capture: LayerCapture,
    sequential_capture: LayerCapture,
    limits: target.EquivalenceLimits,
) -> dict[str, Any]:
    target_logits, target_cache, target_forward, _ = diagnostic._forward_with_capture(
        mx=mx,
        model=model,
        base_cache=base_cache,
        token_ids=token_ids,
        sequential=False,
    )
    sequential_logits, sequential_cache, sequential_forward, _ = (
        diagnostic._forward_with_capture(
            mx=mx,
            model=model,
            base_cache=base_cache,
            token_ids=token_ids,
            sequential=True,
        )
    )
    del target_logits, sequential_logits
    layer_count = len(diagnostic._active_k3_parts(model)[1])
    actual_target = diagnostic.captured_layer_sequences(
        target_forward,
        mx=mx,
        layer_count=layer_count,
        forward_calls=1,
    )[LAYER_INDEX]
    actual_reference = diagnostic.captured_layer_sequences(
        sequential_forward,
        mx=mx,
        layer_count=layer_count,
        forward_calls=WIDTH,
    )[LAYER_INDEX]
    actual_target_conv, actual_target_ssm = target_cache[LAYER_INDEX]
    actual_reference_conv, actual_reference_ssm = sequential_cache[LAYER_INDEX]
    _eval_values(
        mx,
        (
            actual_target,
            actual_reference,
            actual_target_conv,
            actual_target_ssm,
            actual_reference_conv,
            actual_reference_ssm,
        ),
    )
    comparison = kda.compare_stages(
        mx=mx,
        group=group,
        stages=[
            kda.Stage(
                "target_staged_reproduces_full_model_layer0",
                (actual_target, actual_target_conv, actual_target_ssm),
                (
                    target_capture.output,
                    target_capture.final_conv_state,
                    target_capture.final_ssm_state,
                ),
                ("output", "conv_state", "ssm_state"),
            ),
            kda.Stage(
                "reference_staged_reproduces_full_model_layer0",
                (actual_reference, actual_reference_conv, actual_reference_ssm),
                (
                    sequential_capture.output,
                    sequential_capture.final_conv_state,
                    sequential_capture.final_ssm_state,
                ),
                ("output", "conv_state", "ssm_state"),
            ),
        ],
        limits=limits,
    )
    if not comparison["exact_all_stages_all_ranks"]:
        raise LocalizerError(
            "staged decoder-layer paths do not exactly reproduce full-model layer 0"
        )
    return comparison


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx_lm.models.kimi_k3 as kimi
    from mlx_lm.generate import wired_limit

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
    if mode_attestation["mode"] != "mesh":
        raise LocalizerError("decoder-layer localizer requires accepted JACCL mesh")
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
    layer = layers[LAYER_INDEX]
    if not bool(layer.is_linear) or type(layer.self_attn).__name__ != (
        "KimiK3DeltaAttention"
    ):
        raise LocalizerError("layer 0 is not the pinned Kimi K3 KDA layer")
    if type(layer.mlp).__name__ != "KimiK3SparseMoE":
        raise LocalizerError("layer 0 is not the pinned sparse MoE layer")

    prompt_ids, prompt_mode = base.build_prompt_tokens(
        tokenizer, base.DEFAULT_PROMPT, PROMPT_TOKEN_TARGET
    )
    if len(prompt_ids) != PROMPT_TOKEN_TARGET:
        raise LocalizerError("prompt builder did not produce exactly 128 tokens")
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
        limits = target.EquivalenceLimits(
            min_cosine=args.min_cosine,
            max_abs_error=args.max_abs_error,
        )

        target_cache = target.clone_k3_cache(base_cache, mx)
        target_input = mx.array([known_ids[:WIDTH]], dtype=mx.uint32)
        target_capture = _run_layer(
            mx=mx,
            kimi=kimi,
            text_model=text_model,
            layer=layer,
            layer_cache=target_cache[LAYER_INDEX],
            token_input=target_input,
        )

        sequential_cache = target.clone_k3_cache(base_cache, mx)
        sequential_parts = []
        for token_id in known_ids[:WIDTH]:
            sequential_parts.append(
                _run_layer(
                    mx=mx,
                    kimi=kimi,
                    text_model=text_model,
                    layer=layer,
                    layer_cache=sequential_cache[LAYER_INDEX],
                    token_input=mx.array([[int(token_id)]], dtype=mx.uint32),
                )
            )
        sequential_stages = combine_sequential_stages(
            mx, [part.stages for part in sequential_parts]
        )
        sequential_output = mx.concatenate(
            [part.output for part in sequential_parts], axis=1
        )
        mx.eval(sequential_output)
        sequential_capture = LayerCapture(
            stages=sequential_stages,
            output=sequential_output,
            attention_input=mx.concatenate(
                [part.attention_input for part in sequential_parts], axis=1
            ),
            attention_output=mx.concatenate(
                [part.attention_output for part in sequential_parts], axis=1
            ),
            blocks=sequential_parts[-1].blocks,
            final_conv_state=sequential_parts[-1].final_conv_state,
            final_ssm_state=sequential_parts[-1].final_ssm_state,
            finish=sequential_parts[-1].finish,
            provenance=sequential_parts[-1].provenance,
        )

        natural = _compare(
            mx=mx,
            group=group,
            target_stages=target_capture.stages,
            reference_stages=sequential_stages,
            limits=limits,
        )

        # Hold the already-evaluated T=2 prepare result constant.  First replay
        # the verifier's non-transactional T=2 path, then separately replay the
        # speculative history-producing path used by the earlier localizer.
        initial_target_cache = target.clone_k3_cache(base_cache, mx)
        initial_target_conv, initial_target_ssm = initial_target_cache[LAYER_INDEX]
        staged_direct_attention = _direct_kda_path(
            mx=mx,
            attention=layer.self_attn,
            shared_input=target_capture.attention_input,
            initial_conv_state=initial_target_conv,
            initial_ssm_state=initial_target_ssm,
        )
        actual_direct_attention = _actual_direct_attention_endpoint(
            mx=mx,
            attention=layer.self_attn,
            shared_input=target_capture.attention_input,
            base_cache=base_cache,
        )
        initial_reference_cache = target.clone_k3_cache(base_cache, mx)
        initial_conv, initial_ssm = initial_reference_cache[LAYER_INDEX]
        staged_reference_attention = kda._reference_path(
            mx=mx,
            attention=layer.self_attn,
            shared_input=target_capture.attention_input,
            initial_conv_state=initial_conv,
            initial_ssm_state=initial_ssm,
        )
        actual_reference_attention = kda._actual_reference_endpoint(
            mx=mx,
            attention=layer.self_attn,
            shared_input=target_capture.attention_input,
            initial_conv_state=initial_conv,
            initial_ssm_state=initial_ssm,
        )
        direct_kda_endpoint_reproduction = _validate_direct_kda_reproduction(
            mx=mx,
            group=group,
            actual_target=actual_direct_attention,
            actual_reference=actual_reference_attention,
            staged_target=staged_direct_attention,
            staged_reference=staged_reference_attention,
            limits=limits,
        )
        controlled_direct_attention = kda.compare_stages(
            mx=mx,
            group=group,
            stages=_pair_common_kda_stages(
                staged_direct_attention,
                staged_reference_attention,
            ),
            limits=limits,
        )

        speculative_initial = target.clone_k3_cache(base_cache, mx)
        speculative_conv, speculative_ssm = speculative_initial[LAYER_INDEX]
        staged_speculative_attention = kda._target_path(
            mx=mx,
            attention=layer.self_attn,
            shared_input=target_capture.attention_input,
            initial_conv_state=speculative_conv,
            initial_ssm_state=speculative_ssm,
        )
        actual_speculative_attention = kda._actual_target_endpoint(
            mx=mx,
            model=model,
            attention=layer.self_attn,
            shared_input=target_capture.attention_input,
            base_cache=base_cache,
        )
        speculative_kda_endpoint_reproduction = kda.validate_endpoint_reproduction(
            mx=mx,
            group=group,
            actual_target=actual_speculative_attention,
            actual_reference=actual_reference_attention,
            staged_target=staged_speculative_attention,
            staged_reference=staged_reference_attention,
            limits=limits,
        )
        controlled_speculative_attention = kda.compare_stages(
            mx=mx,
            group=group,
            stages=kda.pair_paths(
                staged_speculative_attention,
                staged_reference_attention,
            ),
            limits=limits,
        )

        # Hold T=2 attention output and block state constant, then run the
        # complete finish-attention/MoE path at T=2 and as two T=1 slices.
        controlled_finish_parts = []
        for position in range(WIDTH):
            controlled_finish_parts.append(
                _run_finish(
                    mx=mx,
                    kimi=kimi,
                    layer=layer,
                    partial_sum=None,
                    attention_output=target_capture.attention_output[
                        :, position : position + 1, :
                    ],
                    blocks=_slice_blocks(kimi, target_capture.blocks, position),
                )
            )
        controlled_finish_reference = combine_sequential_stages(
            mx, [part.stages for part in controlled_finish_parts]
        )
        controlled_finish = _compare(
            mx=mx,
            group=group,
            target_stages=target_capture.finish.stages,
            reference_stages=controlled_finish_reference,
            limits=limits,
        )

        endpoint_reproduction = _full_model_endpoint_reproduction(
            mx=mx,
            group=group,
            model=model,
            base_cache=base_cache,
            token_ids=known_ids[:WIDTH],
            target_capture=target_capture,
            sequential_capture=sequential_capture,
            limits=limits,
        )
        target.assert_cache_value_equivalent(immutable_guard, base_cache, mx)

    localization = infer_localization(
        natural=natural,
        controlled_direct_attention=controlled_direct_attention,
        controlled_speculative_attention=controlled_speculative_attention,
        controlled_finish=controlled_finish,
    )
    reference_provenance = [part.provenance for part in sequential_parts]
    controlled_finish_provenance = [part.provenance for part in controlled_finish_parts]
    provenance_record = {
        "target": target_capture.provenance,
        "reference_by_token": reference_provenance,
        "controlled_finish_reference_by_token": controlled_finish_provenance,
        "target_residual_consumed": target_capture.finish.residual_consumed,
        "reference_residual_consumed_by_token": [
            part.finish.residual_consumed for part in sequential_parts
        ],
        "controlled_finish_residual_consumed_by_token": [
            part.residual_consumed for part in controlled_finish_parts
        ],
    }
    provenance_sha256, provenance_sha256_by_rank = base.require_shared_text(
        mx,
        group,
        json.dumps(provenance_record, sort_keys=True, separators=(",", ":")),
        "decoder-layer execution provenance",
    )
    status = (
        COMPLETE_NO_DIVERGENCE
        if natural["exact_all_stages_all_ranks"]
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
            "full_model_endpoint_reproduction_required": True,
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
            "layer": LAYER_INDEX,
            "attention_class": type(layer.self_attn).__name__,
            "mlp_class": type(layer.mlp).__name__,
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
            "base_cache_unchanged": True,
            "execution_provenance": provenance_record,
            "execution_provenance_sha256": provenance_sha256,
            "execution_provenance_sha256_by_rank": provenance_sha256_by_rank,
            "full_model_endpoint_reproduction": endpoint_reproduction,
            "direct_kda_endpoint_reproduction": direct_kda_endpoint_reproduction,
            "speculative_kda_endpoint_reproduction": (
                speculative_kda_endpoint_reproduction
            ),
            "natural_layer_stage_comparison": natural,
            "controlled_shared_prepared_input_direct_attention": (
                controlled_direct_attention
            ),
            "controlled_shared_prepared_input_speculative_attention": (
                controlled_speculative_attention
            ),
            "controlled_shared_finish_input": controlled_finish,
            "localization": localization,
        },
    }
    if rank == 0:
        base.atomic_write_json(args.artifact, artifact)
        print(
            "K3_DECODER_LAYER_STAGE_LOCALIZER_RESULT "
            + json.dumps(
                {
                    "artifact": str(args.artifact.expanduser().resolve()),
                    "status": status,
                    **localization,
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
        description="Real-checkpoint Kimi K3 layer-0 width-2 wrapper localizer"
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
        "K3_DECODER_LAYER_STAGE_LOCALIZER_ERROR "
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
    if os.environ.get("K3_DECODER_LAYER_STAGE_LOCALIZER_TRACEBACK") == "1":
        traceback.print_exception(error, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
