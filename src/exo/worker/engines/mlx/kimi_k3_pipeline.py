"""Kimi K3 pipeline-parallel transport for EXO's MLX backend.

This file is a production-shaped prototype.  Copy it to
``src/exo/worker/engines/mlx/kimi_k3_pipeline.py`` after applying the small
``auto_parallel.py`` integration patch beside this file.

Why a model-specific adapter is required
----------------------------------------
``KimiK3DecoderLayer`` returns ``(partial_sum, ResidualBlocks)``.  The
``ResidualBlocks`` object contains every AttnRes anchor accumulated before the
current layer:

* ``raw``: ``[num_blocks, batch, sequence, hidden]`` in the activation dtype.
* ``inv_rms``: ``[num_blocks, batch, sequence]`` in float32.

EXO's generic pipeline wrapper only sends the first activation tensor.  A
receiving K3 rank would otherwise start with an empty ``ResidualBlocks`` object
and produce different logits even when the split is aligned to a 12-layer
AttnRes boundary.

The wire format deliberately uses one message:

``concat([partial_sum[None], blocks.raw], axis=0)``

The receiver recomputes ``blocks.inv_rms`` from the received raw BF16 anchors
using the same expression as ``ResidualBlocks.append``.  This avoids
consecutive JACCL point-to-point operations with different shapes and dtypes,
which can hang in current MLX/JACCL releases.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from math import prod
from typing import Protocol, cast, final

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from exo.worker.engines.mlx.auto_parallel import (
    PIPELINE_DTYPE_BFLOAT16,
    PIPELINE_DTYPE_FLOAT16,
    PIPELINE_DTYPE_FLOAT32,
    PIPELINE_PHASE_DECODE,
    PIPELINE_PHASE_PREFILL,
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
    decode_timings,
    drain_prefill_sends,
    get_pipeline_tcp_transport,
    pipeline_receive_stream,
    queue_pipeline_ring_activation,
    queue_pipeline_send,
    queue_pipeline_tcp_activation,
    register_decode_send,
    send_pipeline_payloads,
)


class _LayerCallable(Protocol):
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class _ResidualBlocks(Protocol):
    eps: float
    raw: mx.array | None
    inv_rms: mx.array | None


class _CacheListLike(Protocol):
    def __getitem__(self, index: int) -> object: ...


class _KeyCache(Protocol):
    keys: mx.array | None


class _StateCache(Protocol):
    @property
    def state(self) -> object: ...


class _CacheIndexModel(Protocol):
    ssm_idx: int | None
    attn_idx: int | None


def residual_block_count(layer_exclusive: int, block_size: int) -> int:
    """Count AttnRes block starts in global layers ``[0, layer_exclusive)``."""
    if layer_exclusive < 0:
        raise ValueError("layer_exclusive must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return (layer_exclusive + block_size - 1) // block_size


def _find_call_argument(
    _layer: object,
    name: str,
    _x: mx.array,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object | None:
    """Find a K3 argument without trusting a wrapper's generic signature.

    ``PipelineKVProgressLayer`` has a generic ``*args, **kwargs`` signature, so
    signature binding at the outer layer loses positional ``cache``/``blocks``
    names. K3's stable concrete order is ``x, mask, cache, blocks`` and every
    EXO wrapper forwards the original ``args``/``kwargs`` unchanged.
    """
    if name in kwargs:
        return kwargs[name]
    positional_index = {"cache": 1, "blocks": 2}.get(name)
    return (
        args[positional_index]
        if positional_index is not None and positional_index < len(args)
        else None
    )


def _require_blocks(value: object | None) -> _ResidualBlocks:
    if value is None or not hasattr(value, "raw") or not hasattr(value, "inv_rms"):
        raise TypeError(
            "Kimi K3 pipeline transport requires the ResidualBlocks argument"
        )
    return cast(_ResidualBlocks, value)


def _require_layer_result(result: object) -> tuple[mx.array, _ResidualBlocks]:
    if not isinstance(result, tuple):
        raise TypeError(
            "Kimi K3 pipeline layer must return (partial_sum, ResidualBlocks)"
        )
    items = cast(tuple[object, ...], result)
    if len(items) != 2:
        raise TypeError("Kimi K3 pipeline layer must return exactly two tuple elements")
    partial_sum, blocks_value = items
    if not isinstance(partial_sum, mx.array):
        raise TypeError("Kimi K3 partial_sum must be an mx.array")
    return partial_sum, _require_blocks(blocks_value)


def _encode_wire(
    partial_sum: mx.array,
    blocks: _ResidualBlocks,
    expected_blocks: int,
) -> mx.array:
    raw = blocks.raw
    inv_rms = blocks.inv_rms
    if raw is None or inv_rms is None:
        raise ValueError("Kimi K3 produced an empty residual state at a boundary")

    expected_raw_shape = (expected_blocks, *partial_sum.shape)
    expected_inv_shape = (expected_blocks, *partial_sum.shape[:-1])
    if raw.shape != expected_raw_shape:
        raise ValueError(
            f"Kimi K3 raw residual shape {raw.shape} != {expected_raw_shape}"
        )
    if inv_rms.shape != expected_inv_shape:
        raise ValueError(
            f"Kimi K3 inv_rms shape {inv_rms.shape} != {expected_inv_shape}"
        )
    if raw.dtype != partial_sum.dtype:
        raise ValueError(
            f"Kimi K3 residual dtype {raw.dtype} != partial dtype {partial_sum.dtype}"
        )
    if inv_rms.dtype != mx.float32:
        raise ValueError(f"Kimi K3 inv_rms must be float32, got {inv_rms.dtype}")

    return mx.concatenate([partial_sum[None], raw], axis=0)


def _activation_dtype_code(dtype: mx.Dtype) -> int:
    if dtype == mx.bfloat16:
        return PIPELINE_DTYPE_BFLOAT16
    if dtype == mx.float16:
        return PIPELINE_DTYPE_FLOAT16
    if dtype == mx.float32:
        return PIPELINE_DTYPE_FLOAT32
    raise ValueError(f"unsupported Kimi-K3 activation dtype {dtype}")


def _activation_dtype_from_code(dtype_code: int) -> mx.Dtype:
    if dtype_code == PIPELINE_DTYPE_BFLOAT16:
        return mx.bfloat16
    if dtype_code == PIPELINE_DTYPE_FLOAT16:
        return mx.float16
    if dtype_code == PIPELINE_DTYPE_FLOAT32:
        return mx.float32
    raise ValueError(f"unsupported Kimi-K3 activation dtype code {dtype_code}")


def _pipeline_wire_dtype() -> mx.Dtype | None:
    """Resolve the K3 PP boundary precision.

    K3 accumulates its AttnRes partial sum in FP32 on the first pipeline
    stage, while a normal non-pipeline forward and EXO's original
    ``recv_like`` boundary continue from the model's BF16 activation dtype.
    Preserving the FP32 boundary over the host-backed TCP fallback therefore
    promotes every downstream activation on rank one and is dramatically
    slower.  BF16 is the production default; ``preserve`` remains available
    for numerical A/B checks.
    """
    configured = (
        os.environ.get(
            "EXO_K3_PIPELINE_WIRE_DTYPE",
            "bfloat16",
        )
        .strip()
        .lower()
    )
    if configured in {"bfloat16", "bf16"}:
        return mx.bfloat16
    if configured in {"preserve", "native"}:
        return None
    raise ValueError(
        "EXO_K3_PIPELINE_WIRE_DTYPE must be one of "
        "'bfloat16', 'bf16', 'preserve', or 'native'"
    )


def _prepare_activation(
    packed: mx.array,
    *,
    wire_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, int, tuple[int, int, int, int]]:
    """Materialize and describe a K3 boundary at the selected precision."""
    if wire_dtype is not None:
        if wire_dtype != mx.bfloat16:
            raise ValueError(f"unsupported Kimi-K3 wire dtype {wire_dtype}")
        packed = packed.astype(wire_dtype)
    mx.eval(packed)
    if packed.ndim != 4:
        raise ValueError(f"Kimi-K3 packed activation must be 4D, got {packed.shape}")
    dtype_code = _activation_dtype_code(packed.dtype)
    shape = cast(
        tuple[int, int, int, int],
        tuple(int(dimension) for dimension in packed.shape),
    )
    return packed, dtype_code, shape


def _activation_to_bytes(
    packed: mx.array,
    *,
    wire_dtype: mx.Dtype | None = None,
) -> tuple[int, tuple[int, int, int, int], memoryview]:
    """Serialize an evaluated K3 boundary at the selected precision."""
    packed, dtype_code, shape = _prepare_activation(
        packed,
        wire_dtype=wire_dtype,
    )
    if packed.dtype == mx.bfloat16:
        values = np.ascontiguousarray(np.asarray(packed.view(mx.uint16)))
    else:
        values = np.ascontiguousarray(np.asarray(packed))
    return dtype_code, shape, memoryview(values).cast("B")


def _activation_from_bytes(
    payload: bytes | bytearray | memoryview,
    *,
    shape: tuple[int, ...],
    dtype: mx.Dtype,
) -> mx.array:
    """Reconstruct a K3 boundary tensor, including exact BF16 bit patterns."""
    expected_count = prod(shape)
    dtype_code = _activation_dtype_code(dtype)
    item_size = (
        2
        if dtype_code
        in (
            PIPELINE_DTYPE_BFLOAT16,
            PIPELINE_DTYPE_FLOAT16,
        )
        else 4
    )
    expected_bytes = expected_count * item_size
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Kimi-K3 activation payload has {len(payload)} bytes, "
            f"expected {expected_bytes}"
        )

    if dtype == mx.bfloat16:
        values = np.frombuffer(payload, dtype=np.uint16).reshape(shape)
        return mx.array(values).view(mx.bfloat16)
    if dtype == mx.float16:
        values = np.frombuffer(payload, dtype=np.float16).reshape(shape)
        return mx.array(values)
    values = np.frombuffer(payload, dtype=np.float32).reshape(shape)
    return mx.array(values)


def _receive_wire(
    like: mx.array,
    blocks: _ResidualBlocks,
    expected_blocks: int,
    source: int,
    group: mx.distributed.Group,
) -> mx.array:
    # Match EXO's generic wrapper: materializing the local embedding before a
    # receive keeps the distributed operation off a GPU timeout path.
    mx.eval(like)
    packed_template = mx.zeros((expected_blocks + 1, *like.shape), dtype=like.dtype)
    packed = mx.distributed.recv_like(
        packed_template,
        source,
        group=group,
        stream=pipeline_receive_stream,
    )
    mx.eval(packed)

    raw = packed[1:]
    raw_fp32 = raw.astype(mx.float32)
    blocks.raw = raw
    blocks.inv_rms = mx.rsqrt((raw_fp32 * raw_fp32).mean(axis=-1) + blocks.eps)
    return packed[0]


def _receive_wire_tcp(
    like: mx.array,
    blocks: _ResidualBlocks,
    expected_blocks: int,
    *,
    rank: int,
    world_size: int,
    phase: int,
) -> mx.array:
    """Receive one exact K3 packed boundary over the selected PP2 transport."""
    mx.eval(like)
    shape = (expected_blocks + 1, *tuple(int(dim) for dim in like.shape))
    transport = get_pipeline_tcp_transport(rank, world_size)
    exact_shape = cast(tuple[int, int, int, int], shape)
    if transport.activation_transport == "ring":
        dtype_code, packed = transport.receive_activation_array(
            shape=exact_shape,
            phase=phase,
        )
        if _activation_dtype_code(packed.dtype) != dtype_code:
            raise RuntimeError(
                "Kimi-K3 ring activation dtype does not match its descriptor"
            )
    else:
        dtype_code, payload = transport.receive_activation(
            shape=exact_shape,
            phase=phase,
        )
        wire_dtype = _activation_dtype_from_code(dtype_code)
        packed = _activation_from_bytes(
            payload,
            shape=shape,
            dtype=wire_dtype,
        )

    raw = packed[1:]
    raw_fp32 = raw.astype(mx.float32)
    blocks.raw = raw
    blocks.inv_rms = mx.rsqrt((raw_fp32 * raw_fp32).mean(axis=-1) + blocks.eps)
    return packed[0]


def _set_cache_dependency(cache: object | None, dependency: mx.array) -> None:
    if cache is None:
        return
    inner = cast(_CacheListLike, cache)[0] if hasattr(cache, "caches") else cache
    if not hasattr(inner, "keys"):
        return
    key_cache = cast(_KeyCache, inner)
    keys = key_cache.keys
    if keys is not None:
        key_cache.keys = mx.depends(keys, dependency)


@final
class KimiK3PipelineFirstLayer(PipelineFirstLayer):
    """Receive both the partial sum and all preceding AttnRes anchors."""

    def __init__(
        self,
        original_layer: _LayerCallable,
        rank: int,
        world_size: int,
        incoming_block_count: int,
        group: mx.distributed.Group,
    ) -> None:
        super().__init__(original_layer, r=rank, group=group)
        self.world_size: int = world_size
        self.incoming_block_count: int = incoming_block_count
        self.uses_tcp_pipeline_transport = "MLX_JACCL_COORDINATOR" in os.environ
        if self.uses_tcp_pipeline_transport and world_size != 2:
            raise ValueError(
                "Kimi-K3 JACCL pipeline transport requires exactly two ranks"
            )

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if (
            self.uses_tcp_pipeline_transport
            and not self.is_prefill
            and not self.token_relay
        ):
            raise RuntimeError(
                "Kimi-K3 PP2 JACCL decode requires TCP token relay; "
                "logprobs are not supported"
            )
        blocks = _require_blocks(
            _find_call_argument(self.original_layer, "blocks", x, args, kwargs)
        )
        if self.r != 0:
            receive_start = time.perf_counter()
            if self.uses_tcp_pipeline_transport:
                x = _receive_wire_tcp(
                    x,
                    blocks,
                    expected_blocks=self.incoming_block_count,
                    rank=self.r,
                    world_size=self.world_size,
                    phase=(
                        PIPELINE_PHASE_PREFILL
                        if self.is_prefill
                        else PIPELINE_PHASE_DECODE
                    ),
                )
            else:
                x = _receive_wire(
                    x,
                    blocks,
                    expected_blocks=self.incoming_block_count,
                    source=self.r - 1,
                    group=self.group,
                )
            if not self.is_prefill:
                decode_timings.record_recv(time.perf_counter() - receive_start)
        return self.original_layer(x, *args, **kwargs)


@final
class KimiK3PipelineLastLayer(PipelineLastLayer):
    """Send K3 tuple state forward and, when needed, relay final state back."""

    def __init__(
        self,
        original_layer: _LayerCallable,
        rank: int,
        world_size: int,
        outgoing_block_count: int,
        final_block_count: int,
        group: mx.distributed.Group,
    ) -> None:
        super().__init__(
            original_layer,
            r=rank,
            s=world_size,
            group=group,
        )
        self.outgoing_block_count: int = outgoing_block_count
        self.final_block_count: int = final_block_count
        self.uses_tcp_pipeline_transport = "MLX_JACCL_COORDINATOR" in os.environ
        self.pipeline_wire_dtype = (
            _pipeline_wire_dtype() if self.uses_tcp_pipeline_transport else None
        )
        if self.uses_tcp_pipeline_transport and world_size != 2:
            raise ValueError(
                "Kimi-K3 JACCL pipeline transport requires exactly two ranks"
            )

    def _relay_final_state(
        self,
        partial_sum: mx.array,
        blocks: _ResidualBlocks,
    ) -> tuple[mx.array, _ResidualBlocks]:
        """Reverse-relay final state for the legacy/logprobs logits path.

        K3 applies one final AttnRes mix after its decoder loop.  Relaying only
        the last partial sum (the generic EXO behavior) leaves earlier ranks
        with incomplete ``ResidualBlocks`` and therefore different logits.
        Token-relay decode skips this transfer because only the last rank's
        logits are consumed.
        """
        if self.r != self.s - 1:
            partial_sum = _receive_wire(
                partial_sum,
                blocks,
                expected_blocks=self.final_block_count,
                source=self.r + 1,
                group=self.group,
            )

        if self.r > 0:
            packed = _encode_wire(partial_sum, blocks, self.final_block_count)
            mx.eval(packed)
            send_pipeline_payloads(
                (packed,),
                destination=self.r - 1,
                group=self.group,
                asynchronous=False,
            )
        return partial_sum, blocks

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if (
            self.uses_tcp_pipeline_transport
            and not self.is_prefill
            and not self.token_relay
        ):
            raise RuntimeError(
                "Kimi-K3 PP2 JACCL decode requires TCP token relay; "
                "logprobs are not supported"
            )
        cache = _find_call_argument(self.original_layer, "cache", x, args, kwargs)
        partial_sum, blocks = _require_layer_result(
            self.original_layer(x, *args, **kwargs)
        )
        forward_dependency: mx.array | None = None

        if self.r != self.s - 1:
            packed = _encode_wire(partial_sum, blocks, self.outgoing_block_count)
            # Split the lazy graph before communication, matching
            # PipelineLastLayer. The last rank skips this potentially large
            # packing operation unless legacy decode needs a reverse relay.
            mx.eval(packed)
            send_start = time.perf_counter()
            destination = self.r + 1
            if self.uses_tcp_pipeline_transport:
                transport = get_pipeline_tcp_transport(self.r, self.s)
                packed, dtype_code, shape = _prepare_activation(
                    packed,
                    wire_dtype=self.pipeline_wire_dtype,
                )
                phase = (
                    PIPELINE_PHASE_PREFILL if self.is_prefill else PIPELINE_PHASE_DECODE
                )
                if transport.activation_transport == "ring":
                    if self.queue_sends:
                        queue_pipeline_ring_activation(
                            transport,
                            packed,
                            dtype_code=dtype_code,
                            shape=shape,
                            phase=phase,
                        )
                    else:
                        transport.send_activation_array(
                            packed,
                            dtype_code=dtype_code,
                            shape=shape,
                            phase=phase,
                        )
                else:
                    _, _, payload = _activation_to_bytes(packed)
                    if self.queue_sends:
                        queue_pipeline_tcp_activation(
                            transport,
                            payload,
                            dtype_code=dtype_code,
                            shape=shape,
                            phase=phase,
                        )
                    else:
                        transport.send_activation(
                            payload,
                            dtype_code=dtype_code,
                            shape=shape,
                            phase=phase,
                        )
            else:
                if self.queue_sends:
                    queue_pipeline_send(
                        (packed,),
                        destination=destination,
                        group=self.group,
                    )
                    # Preserve the generic wrapper's cache/output dependency.
                    _set_cache_dependency(cache, packed)
                else:
                    # The ring fallback retains the existing asynchronous
                    # forward/synchronous reverse ordering.
                    asynchronous_forward = not self.is_prefill
                    dependency = send_pipeline_payloads(
                        (packed,),
                        destination=destination,
                        group=self.group,
                        asynchronous=asynchronous_forward,
                    )
                    if asynchronous_forward:
                        if self.token_relay:
                            register_decode_send(dependency)
                        else:
                            forward_dependency = dependency
                    else:
                        _set_cache_dependency(cache, dependency)
            if not self.is_prefill:
                decode_timings.record_send(time.perf_counter() - send_start)

        if not self.is_prefill and not self.token_relay:
            gather_start = time.perf_counter()
            partial_sum, blocks = self._relay_final_state(partial_sum, blocks)
            if forward_dependency is not None:
                # The reverse receive runs on a separate CPU stream, allowing
                # it to overlap the forward send. Once both directions finish,
                # the earlier asynchronous prefill tail is also ordered done.
                mx.eval(forward_dependency)
                drain_prefill_sends()
            decode_timings.record_gather_and_advance(time.perf_counter() - gather_start)

        # K3's runtime return is a tuple despite EXO's array-only protocol.
        return cast(mx.array, cast(object, (partial_sum, blocks)))


@final
class KimiK3GraphCheckpointLayer(CustomMlxLayer):
    """Bound one segment of a pipeline rank's lazy Metal graph.

    Tensor parallelism naturally segments K3 execution with a collective in
    every layer. A pipeline rank instead owns dozens of consecutive local
    layers, so without checkpoints the boundary ``mx.eval`` can submit one
    very large graph and trip Metal's GPU-hang watchdog. The shared segment
    state retains every cache-side array as well as the main tuple state;
    evaluating only the layer output does not detach KDA/KV cache graphs.
    """

    def __init__(
        self,
        original_layer: _LayerCallable,
        *,
        segment_state: _KimiK3GraphSegmentState,
        reset_before: bool,
        checkpoint_after: bool,
    ) -> None:
        super().__init__(original_layer)
        self.segment_state = segment_state
        self.reset_before = reset_before
        self.checkpoint_after = checkpoint_after

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.reset_before:
            self.segment_state.cache_arrays.clear()

        cache = _find_call_argument(self.original_layer, "cache", x, args, kwargs)
        partial_sum, blocks = _require_layer_result(
            self.original_layer(x, *args, **kwargs)
        )
        if cache is not None and hasattr(cache, "state"):
            _collect_mx_arrays(
                cast(_StateCache, cache).state,
                self.segment_state.cache_arrays,
            )

        if self.checkpoint_after:
            state = [partial_sum]
            if blocks.raw is not None:
                state.append(blocks.raw)
            if blocks.inv_rms is not None:
                state.append(blocks.inv_rms)
            state.extend(self.segment_state.cache_arrays)
            try:
                mx.eval(*state)
            finally:
                self.segment_state.cache_arrays.clear()
        return cast(mx.array, cast(object, (partial_sum, blocks)))


class _KimiK3GraphSegmentState:
    def __init__(self) -> None:
        self.cache_arrays: list[mx.array] = []


def _collect_mx_arrays(value: object, output: list[mx.array]) -> None:
    if isinstance(value, mx.array):
        output.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            _collect_mx_arrays(nested, output)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_mx_arrays(nested, output)


def _graph_checkpoint_interval() -> int:
    raw = os.environ.get("EXO_K3_GRAPH_CHECKPOINT_INTERVAL", "2")
    try:
        interval = int(raw)
    except ValueError as error:
        raise ValueError(
            "EXO_K3_GRAPH_CHECKPOINT_INTERVAL must be an integer"
        ) from error
    if interval <= 0:
        raise ValueError("EXO_K3_GRAPH_CHECKPOINT_INTERVAL must be positive")
    return interval


def wrap_kimi_k3_pipeline_layers(
    layers: Sequence[_LayerCallable],
    *,
    start_layer: int,
    end_layer: int,
    total_layers: int,
    block_size: int,
    rank: int,
    world_size: int,
    group: mx.distributed.Group,
) -> list[_LayerCallable]:
    """Install tuple-aware boundary wrappers around already-local K3 layers.

    Consecutive local layers are segmented by materialization checkpoints. The
    first and last local layers retain the model-specific pipeline transport
    wrappers outside those checkpoint wrappers.
    """
    if not layers:
        raise ValueError("A pipeline rank must own at least one Kimi K3 layer")
    if not 0 <= start_layer < end_layer <= total_layers:
        raise ValueError(
            f"invalid Kimi K3 interval [{start_layer}, {end_layer})/{total_layers}"
        )

    wrapped = list(layers)
    checkpoint_interval = _graph_checkpoint_interval()
    segment_state = _KimiK3GraphSegmentState()
    for local_index, layer in enumerate(wrapped):
        wrapped[local_index] = KimiK3GraphCheckpointLayer(
            layer,
            segment_state=segment_state,
            reset_before=local_index == 0,
            checkpoint_after=(
                (local_index + 1) % checkpoint_interval == 0
                or local_index == len(wrapped) - 1
            ),
        )
    wrapped[0] = KimiK3PipelineFirstLayer(
        wrapped[0],
        rank=rank,
        world_size=world_size,
        incoming_block_count=residual_block_count(start_layer, block_size),
        group=group,
    )
    wrapped[-1] = KimiK3PipelineLastLayer(
        wrapped[-1],
        rank=rank,
        world_size=world_size,
        outgoing_block_count=residual_block_count(end_layer, block_size),
        final_block_count=residual_block_count(total_layers, block_size),
        group=group,
    )
    return wrapped


def configure_kimi_k3_local_cache_indices(
    inner_model: nn.Module,
    layers: Sequence[_LayerCallable],
) -> None:
    """Rebase K3's hybrid-cache mask probes to the rank-local cache list."""
    linear_indices = [
        index
        for index, layer in enumerate(layers)
        if bool(getattr(layer, "is_linear", False))
    ]
    attention_indices = [
        index
        for index, layer in enumerate(layers)
        if not bool(getattr(layer, "is_linear", False))
    ]
    cache_index_model = cast(_CacheIndexModel, inner_model)
    cache_index_model.ssm_idx = linear_indices[0] if linear_indices else None
    cache_index_model.attn_idx = attention_indices[0] if attention_indices else None
