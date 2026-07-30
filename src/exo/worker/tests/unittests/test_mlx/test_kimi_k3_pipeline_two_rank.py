"""Two-rank numerical equivalence gate for the EXO Kimi K3 adapter.

Run only after copying ``kimi_k3_pipeline.py`` into EXO and applying
``auto_parallel_integration.patch``.  This deliberately uses a split whose
second rank needs rebased hybrid-cache indices.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.kimi_k3 import Model as KimiK3Model
from mlx_lm.models.kimi_k3 import ModelArgs

from exo.worker.engines.mlx.auto_parallel import (
    flush_prefill_sends,
    get_active_relay_context,
    patch_pipeline_model,
    relay_sampled_tokens,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
    set_pipeline_token_relay,
)
from exo.worker.engines.mlx.kimi_k3_pipeline import (
    KimiK3GraphCheckpointLayer,
    KimiK3PipelineLastLayer,
    configure_kimi_k3_local_cache_indices,
    wrap_kimi_k3_pipeline_layers,
)

_TINY_CONFIG = {
    "model_type": "kimi_k3",
    "text_config": {
        "model_type": "kimi_linear",
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 96,
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
        "moe_intermediate_size": 32,
        "q_lora_rank": 24,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
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


def _barrier(group: mx.distributed.Group) -> None:
    value = mx.distributed.all_sum(mx.array(1, dtype=mx.int32), group=group)
    mx.eval(value)


def _assert_close(actual: mx.array, expected: mx.array) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert bool(mx.allclose(actual, expected, rtol=1e-4, atol=1e-4).item())


def _assert_tree_close(actual: object, expected: object) -> None:
    if actual is None or expected is None:
        assert actual is expected
    elif isinstance(actual, mx.array) and isinstance(expected, mx.array):
        _assert_close(actual, expected)
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_close(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_local_cache_matches(
    local_cache: list[Any],
    reference_cache: list[Any],
    start_layer: int,
    end_layer: int,
) -> None:
    expected = reference_cache[start_layer:end_layer]
    assert len(local_cache) == len(expected)
    for local_entry, reference_entry in zip(local_cache, expected, strict=True):
        assert type(local_entry) is type(reference_entry)
        if hasattr(local_entry, "offset"):
            assert local_entry.offset == reference_entry.offset
        _assert_tree_close(local_entry.state, reference_entry.state)


def _make_models() -> tuple[KimiK3Model, KimiK3Model]:
    args = ModelArgs.from_dict(_TINY_CONFIG)
    mx.random.seed(17)
    reference = KimiK3Model(args)
    mx.eval(reference.parameters())

    sharded = KimiK3Model(args)
    sharded.load_weights(tree_flatten(reference.parameters()), strict=True)
    mx.eval(sharded.parameters())
    return reference, sharded


def _partition_for_test(
    model: KimiK3Model,
    group: mx.distributed.Group,
    rank: int,
) -> KimiK3Model:
    start_layer, end_layer = ((0, 2), (2, 4))[rank]
    inner = model.model
    local_layers = list(inner.layers[start_layer:end_layer])
    wrapped = wrap_kimi_k3_pipeline_layers(
        local_layers,
        start_layer=start_layer,
        end_layer=end_layer,
        total_layers=4,
        block_size=2,
        rank=rank,
        world_size=2,
        group=group,
    )
    inner.layers = wrapped
    configure_kimi_k3_local_cache_indices(inner, wrapped)
    return patch_pipeline_model(model, group)


def test_kimi_k3_pipeline_graph_checkpoints_follow_interval_and_final_layer() -> None:
    args = ModelArgs.from_dict(_TINY_CONFIG)
    model = KimiK3Model(args)
    layers = list(model.model.layers)
    previous_interval = os.environ.get("EXO_K3_GRAPH_CHECKPOINT_INTERVAL")
    os.environ["EXO_K3_GRAPH_CHECKPOINT_INTERVAL"] = "2"
    try:
        wrapped = wrap_kimi_k3_pipeline_layers(
            layers,
            start_layer=0,
            end_layer=4,
            total_layers=4,
            block_size=2,
            rank=0,
            world_size=1,
            group=None,  # type: ignore[arg-type]
        )
    finally:
        if previous_interval is None:
            os.environ.pop("EXO_K3_GRAPH_CHECKPOINT_INTERVAL", None)
        else:
            os.environ["EXO_K3_GRAPH_CHECKPOINT_INTERVAL"] = previous_interval

    assert isinstance(wrapped[1], KimiK3GraphCheckpointLayer)
    assert wrapped[1].checkpoint_after
    assert isinstance(wrapped[-1], KimiK3PipelineLastLayer)
    final_inner = wrapped[-1].original_layer
    assert isinstance(final_inner, KimiK3GraphCheckpointLayer)
    assert final_inner.checkpoint_after


def _run_rank(
    rank: int,
    hostfile: str,
    result_queue: Any,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        reference, pipeline = _make_models()
        pipeline = _partition_for_test(pipeline, group, rank)
        start_layer, end_layer = ((0, 2), (2, 4))[rank]

        # The split intentionally leaves rank 0 without MLA and rank 1 with a
        # rank-local MLA index of 1 (the unpatched global value is 3).
        if rank == 0:
            assert pipeline.model.ssm_idx == 0
            assert pipeline.model.attn_idx is None
        else:
            assert pipeline.model.ssm_idx == 0
            assert pipeline.model.attn_idx == 1

        reference_cache = reference.make_cache()
        local_cache = pipeline.make_cache()
        prompt = mx.array([[2, 7, 11, 5, 3]], dtype=mx.int32)

        # 1. Prefill, including EXO's queued-send path.  Only the last rank's
        # logits are meaningful during pipeline prefill; every local cache must
        # nevertheless match its corresponding full-model cache.
        reference_prefill = reference(prompt, cache=reference_cache)
        mx.eval(reference_prefill, [entry.state for entry in reference_cache])

        set_pipeline_prefill(pipeline, True)
        set_pipeline_queue_sends(pipeline, True)
        pipeline_prefill = pipeline(prompt, cache=local_cache)
        mx.eval([entry.state for entry in local_cache])
        flush_prefill_sends()
        set_pipeline_queue_sends(pipeline, False)
        set_pipeline_prefill(pipeline, False)
        if rank == 1:
            mx.eval(pipeline_prefill)
            _assert_close(pipeline_prefill, reference_prefill)
        _assert_local_cache_matches(
            local_cache, reference_cache, start_layer, end_layer
        )

        # 2. Token-relay decode.  Last-rank logits must match the full model;
        # the non-last rank must return detached zeros and receive the sampled
        # token from rank 1.
        decode_input = mx.array([[13]], dtype=mx.int32)
        reference_decode = reference(decode_input, cache=reference_cache)
        mx.eval(reference_decode, [entry.state for entry in reference_cache])

        set_pipeline_token_relay(pipeline, True)
        pipeline_decode = pipeline(decode_input, cache=local_cache)
        mx.eval(pipeline_decode, [entry.state for entry in local_cache])
        if rank == 0:
            assert bool(mx.all(pipeline_decode == 0).item())
        else:
            _assert_close(pipeline_decode, reference_decode)

        relay_context = get_active_relay_context(pipeline)
        assert relay_context is not None
        local_token = mx.argmax(pipeline_decode[:, -1, :], axis=-1)
        relayed_token = relay_sampled_tokens(local_token, relay_context)
        expected_token = mx.argmax(reference_decode[:, -1, :], axis=-1)
        mx.eval(relayed_token, expected_token)
        assert bool(mx.array_equal(relayed_token, expected_token).item())
        set_pipeline_token_relay(pipeline, False)
        _assert_local_cache_matches(
            local_cache, reference_cache, start_layer, end_layer
        )

        # 3. Legacy/logprobs decode.  This forces the reverse relay of the final
        # partial sum *and* all final ResidualBlocks, so the output AttnRes mix
        # and logits must be equal on both ranks.
        reference_legacy = reference(relayed_token[:, None], cache=reference_cache)
        pipeline_legacy = pipeline(relayed_token[:, None], cache=local_cache)
        mx.eval(
            reference_legacy,
            pipeline_legacy,
            [entry.state for entry in reference_cache],
            [entry.state for entry in local_cache],
        )
        _assert_close(pipeline_legacy, reference_legacy)
        _assert_local_cache_matches(
            local_cache, reference_cache, start_layer, end_layer
        )
        result_queue.put((rank, True, None))
    except Exception as error:
        result_queue.put((rank, False, repr(error)))


def test_kimi_k3_two_rank_prefill_decode_and_cache_equivalence() -> None:
    context = mp.get_context("spawn")
    hosts = ["127.0.0.1:29840", "127.0.0.1:29841"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        json.dump(hosts, file)
        hostfile = file.name

    try:
        result_queue: Any = context.Queue()
        processes = [
            context.Process(target=_run_rank, args=(rank, hostfile, result_queue))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

        results: dict[int, tuple[bool, str | None]] = {}
        while not result_queue.empty():
            rank, success, error = result_queue.get()
            results[rank] = (success, error)

        assert len(results) == 2, f"missing rank result(s): {results}"
        for rank in range(2):
            success, error = results[rank]
            assert success, f"rank {rank} failed: {error}"
    finally:
        os.unlink(hostfile)
