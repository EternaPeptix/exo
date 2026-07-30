from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from queue import Queue
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from exo.worker.engines.mlx import auto_parallel, kimi_k3_pipeline
from exo.worker.engines.mlx.auto_parallel import (
    PIPELINE_DTYPE_BFLOAT16,
    PIPELINE_DTYPE_FLOAT16,
    PIPELINE_DTYPE_FLOAT32,
    PIPELINE_PHASE_DECODE,
    PIPELINE_PHASE_PREFILL,
    PipelineDecodeTimings,
    _PipelineTcpTransport,
    clear_prefill_sends,
    discard_unsent_prefill_sends_after_agreed_cancel,
    flush_prefill_sends,
    queue_pipeline_ring_activation,
    queue_pipeline_send,
    queue_pipeline_tcp_activation,
)
from exo.worker.engines.mlx.kimi_k3_pipeline import (
    KimiK3PipelineFirstLayer,
    KimiK3PipelineLastLayer,
    _activation_from_bytes,
    _activation_to_bytes,
    _pipeline_wire_dtype,
)


def _reserve_loopback_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


@contextmanager
def _transport_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[_PipelineTcpTransport, _PipelineTcpTransport]]:
    relay_port = _reserve_loopback_port()
    coordinator_port = relay_port - 1 if relay_port > 1024 else relay_port + 1
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", f"127.0.0.1:{coordinator_port}")
    monkeypatch.setenv("EXO_PIPELINE_TOKEN_RELAY_PORT", str(relay_port))
    monkeypatch.setenv("EXO_PIPELINE_TCP_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("EXO_K3_PIPELINE_ACTIVATION_TRANSPORT", "tcp")
    rank_zero = _PipelineTcpTransport(rank=0, world_size=2)
    rank_one = _PipelineTcpTransport(rank=1, world_size=2)
    try:
        yield rank_zero, rank_one
    finally:
        rank_zero.close()
        rank_one.close()


@contextmanager
def _ring_array_transport_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[_PipelineTcpTransport, _PipelineTcpTransport, object]]:
    with _transport_pair(monkeypatch) as (rank_zero, rank_one):
        errors: list[BaseException] = []

        def connect_rank_one() -> None:
            try:
                rank_one.ensure_connected()
            except BaseException as error:
                errors.append(error)

        connector = threading.Thread(target=connect_rank_one, daemon=True)
        connector.start()
        rank_zero.ensure_connected()
        connector.join(timeout=10)
        if connector.is_alive() or errors:
            raise RuntimeError(f"unable to establish test control transport: {errors}")

        group = object()
        for transport in (rank_zero, rank_one):
            transport.activation_transport = "ring"
            transport.activation_transport_code = (
                auto_parallel._PIPELINE_ACTIVATION_TRANSPORT_RING
            )
            transport.secondary_ring_group = group  # type: ignore[assignment]
        try:
            yield rank_zero, rank_one, group
        finally:
            rank_zero.close()
            rank_one.close()


@pytest.fixture(autouse=True)
def _force_tcp_transport_for_existing_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unit-test hosts generally do not have the production en3-en6 link-local
    # fabric. Ring-specific tests inject an already-connected secondary group.
    monkeypatch.setenv("EXO_K3_PIPELINE_ACTIVATION_TRANSPORT", "tcp")


def _uint16_payload(start: int, shape: tuple[int, int, int, int]) -> memoryview:
    element_count = int(np.prod(shape))
    values = np.arange(start, start + element_count, dtype=np.uint16).reshape(shape)
    return memoryview(values).cast("B")


def test_activation_serialization_preserves_bfloat16_bits() -> None:
    bits = np.array(
        [
            0x0000,
            0x8000,
            0x3F80,
            0xBF80,
            0x7F80,
            0xFF80,
            0x7FC1,
            0x0001,
            0x7FFF,
            0x1234,
            0xABCD,
            0x4000,
            0x4049,
            0x0080,
            0x8080,
            0x3F00,
        ],
        dtype=np.uint16,
    ).reshape((2, 1, 2, 4))
    packed = mx.array(bits).view(mx.bfloat16)

    dtype_code, shape, payload = _activation_to_bytes(packed)
    restored = _activation_from_bytes(
        payload,
        shape=shape,
        dtype=mx.bfloat16,
    )
    mx.eval(restored)

    assert dtype_code == PIPELINE_DTYPE_BFLOAT16
    assert shape == bits.shape
    assert np.array_equal(np.asarray(restored.view(mx.uint16)), bits)


@pytest.mark.parametrize(
    ("mlx_dtype", "numpy_dtype", "dtype_code"),
    [
        (mx.float16, np.float16, PIPELINE_DTYPE_FLOAT16),
        (mx.float32, np.float32, PIPELINE_DTYPE_FLOAT32),
    ],
)
def test_activation_serialization_preserves_supported_float_payloads(
    mlx_dtype: mx.Dtype,
    numpy_dtype: np.dtype[np.generic],
    dtype_code: int,
) -> None:
    values = np.linspace(-3.0, 4.0, 16, dtype=numpy_dtype).reshape((2, 1, 2, 4))
    packed = mx.array(values, dtype=mlx_dtype)

    actual_code, shape, payload = _activation_to_bytes(packed)
    restored = _activation_from_bytes(payload, shape=shape, dtype=mlx_dtype)
    mx.eval(restored)

    assert actual_code == dtype_code
    assert np.array_equal(np.asarray(restored), values)


def test_activation_serialization_can_downcast_float32_boundary_to_bfloat16() -> None:
    values = np.linspace(-3.0, 4.0, 16, dtype=np.float32).reshape((2, 1, 2, 4))
    packed = mx.array(values, dtype=mx.float32)

    dtype_code, shape, payload = _activation_to_bytes(
        packed,
        wire_dtype=mx.bfloat16,
    )
    restored = _activation_from_bytes(payload, shape=shape, dtype=mx.bfloat16)
    expected = packed.astype(mx.bfloat16)
    mx.eval(restored, expected)

    assert dtype_code == PIPELINE_DTYPE_BFLOAT16
    assert len(payload) == packed.size * 2
    assert np.array_equal(
        np.asarray(restored.view(mx.uint16)),
        np.asarray(expected.view(mx.uint16)),
    )


def test_pipeline_wire_dtype_defaults_to_bfloat16_and_supports_preserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_K3_PIPELINE_WIRE_DTYPE", raising=False)
    assert _pipeline_wire_dtype() == mx.bfloat16

    monkeypatch.setenv("EXO_K3_PIPELINE_WIRE_DTYPE", "preserve")
    assert _pipeline_wire_dtype() is None

    monkeypatch.setenv("EXO_K3_PIPELINE_WIRE_DTYPE", "float32")
    with pytest.raises(ValueError, match="EXO_K3_PIPELINE_WIRE_DTYPE"):
        _pipeline_wire_dtype()


def test_pipeline_decode_timing_interval_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_PIPELINE_TIMING_LOG_EVERY", "8")
    assert PipelineDecodeTimings().log_every == 8


def test_ring_address_discovery_uses_ordered_link_local_en3_through_en6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_K3_PIPELINE_RING_ADDRESSES", raising=False)
    monkeypatch.setattr(
        auto_parallel.psutil,
        "net_if_addrs",
        lambda: {
            "en3": [
                SimpleNamespace(
                    family=socket.AF_INET,
                    address="169.254.10.3",
                )
            ],
            "en4": [
                SimpleNamespace(
                    family=socket.AF_INET6,
                    address="fe80::1",
                ),
                SimpleNamespace(
                    family=socket.AF_INET,
                    address="192.0.2.4",
                ),
                SimpleNamespace(
                    family=socket.AF_INET,
                    address="169.254.10.4",
                ),
            ],
            "en5": [
                SimpleNamespace(
                    family=socket.AF_INET,
                    address="169.254.10.5",
                )
            ],
            "en6": [
                SimpleNamespace(
                    family=socket.AF_INET,
                    address="169.254.10.6",
                )
            ],
        },
    )

    assert auto_parallel._discover_pipeline_ring_addresses() == (
        "169.254.10.3",
        "169.254.10.4",
        "169.254.10.5",
        "169.254.10.6",
    )


def test_rank_zero_exchanges_canonical_ring_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen_ports = (60001, 60002)
    choices: list[tuple[int, int, int]] = []

    def choose_ports(
        coordinator_port: int,
        relay_port: int,
        count: int,
    ) -> tuple[int, ...]:
        choices.append((coordinator_port, relay_port, count))
        return chosen_ports

    monkeypatch.setattr(auto_parallel, "_pipeline_ring_ports", choose_ports)
    with _transport_pair(monkeypatch) as (rank_zero, rank_one):
        rank_zero.local_ring_addresses = ("169.254.1.1", "169.254.1.2")
        rank_one.local_ring_addresses = ("169.254.2.1", "169.254.2.2")
        zero_socket, one_socket = socket.socketpair()
        rank_one_result: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

        def exchange_rank_one() -> None:
            rank_one_result.append(rank_one._exchange_ring_configuration(one_socket))

        receiver = threading.Thread(target=exchange_rank_one, daemon=True)
        receiver.start()
        rank_zero_result = rank_zero._exchange_ring_configuration(zero_socket)
        receiver.join(timeout=10)
        zero_socket.close()
        one_socket.close()

    assert not receiver.is_alive()
    assert choices == [(rank_zero.coordinator_port, rank_zero.relay_port, 2)]
    assert rank_zero_result == (
        ("169.254.2.1", "169.254.2.2"),
        chosen_ports,
    )
    assert rank_one_result == [(("169.254.1.1", "169.254.1.2"), chosen_ports)]
    assert auto_parallel._pipeline_ring_hosts(
        0,
        rank_zero.local_ring_addresses,
        rank_zero_result[0],
        chosen_ports,
    ) == [
        ["169.254.1.1:60001", "169.254.1.2:60002"],
        ["169.254.2.1:60001", "169.254.2.2:60002"],
    ]
    assert auto_parallel._pipeline_ring_hosts(
        1,
        rank_one.local_ring_addresses,
        rank_one_result[0][0],
        chosen_ports,
    ) == [
        ["169.254.1.1:60001", "169.254.1.2:60002"],
        ["169.254.2.1:60001", "169.254.2.2:60002"],
    ]


def test_handshake_rejects_stale_protocol_v2_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _transport_pair(monkeypatch) as (rank_zero, _):
        zero_socket, one_socket = socket.socketpair()
        stale_hello = auto_parallel._PIPELINE_TCP_HELLO.pack(
            auto_parallel._PIPELINE_TCP_MAGIC,
            2,
            1,
            2,
            auto_parallel._PIPELINE_ACTIVATION_TRANSPORT_TCP,
            rank_zero.coordinator_port,
        )
        one_socket.sendall(stale_hello)
        try:
            with pytest.raises(RuntimeError, match="version=2"):
                rank_zero._exchange_hello(zero_socket)
        finally:
            zero_socket.close()
            one_socket.close()


class _TestResidualBlocks:
    def __init__(self) -> None:
        self.eps = 1e-5
        self.raw: mx.array | None = None
        self.inv_rms: mx.array | None = None


def test_receive_wire_uses_validated_float32_sender_dtype_with_bfloat16_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 1, 2, 4)
    wire_values = np.linspace(-4.0, 5.0, 16, dtype=np.float32).reshape(shape)

    class _Float32Transport:
        activation_transport = "tcp"

        def receive_activation(
            self,
            *,
            shape: tuple[int, int, int, int],
            phase: int,
        ) -> tuple[int, bytearray]:
            assert shape == wire_values.shape
            assert phase == PIPELINE_PHASE_PREFILL
            return (
                PIPELINE_DTYPE_FLOAT32,
                bytearray(memoryview(wire_values).cast("B")),
            )

    transport = _Float32Transport()
    monkeypatch.setattr(
        kimi_k3_pipeline,
        "get_pipeline_tcp_transport",
        lambda rank, world_size: transport,
    )
    template = mx.zeros((1, 2, 4), dtype=mx.bfloat16)
    blocks = _TestResidualBlocks()

    partial = kimi_k3_pipeline._receive_wire_tcp(
        template,
        blocks,
        expected_blocks=1,
        rank=1,
        world_size=2,
        phase=PIPELINE_PHASE_PREFILL,
    )
    assert blocks.raw is not None
    assert blocks.inv_rms is not None
    mx.eval(partial, blocks.raw, blocks.inv_rms)

    assert partial.dtype == mx.float32
    assert blocks.raw.dtype == mx.float32
    assert np.array_equal(np.asarray(partial), wire_values[0])
    assert np.array_equal(np.asarray(blocks.raw), wire_values[1:])


def test_transport_orders_prefill_then_a0_a1_t0_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (3, 1, 2, 4)
    prefill = _uint16_payload(0, shape)
    activation_zero = _uint16_payload(100, shape)
    activation_one = _uint16_payload(200, shape)
    activation_two = _uint16_payload(300, shape)
    received: list[bytes] = []
    received_dtype_codes: list[int] = []
    errors: list[BaseException] = []

    with _transport_pair(monkeypatch) as (rank_zero, rank_one):

        def rank_one_loop() -> None:
            try:

                def receive(phase: int) -> None:
                    dtype_code, payload = rank_one.receive_activation(
                        shape=shape,
                        phase=phase,
                    )
                    received_dtype_codes.append(dtype_code)
                    received.append(bytes(payload))

                receive(PIPELINE_PHASE_PREFILL)
                for _ in range(2):
                    receive(PIPELINE_PHASE_DECODE)
                rank_one.send_tokens([17])
                receive(PIPELINE_PHASE_DECODE)
                rank_one.send_tokens([23])
            except BaseException as error:
                errors.append(error)

        receiver = threading.Thread(target=rank_one_loop, daemon=True)
        receiver.start()
        rank_zero.send_activation(
            prefill,
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=shape,
            phase=PIPELINE_PHASE_PREFILL,
        )
        rank_zero.send_activation(
            activation_zero,
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=shape,
            phase=PIPELINE_PHASE_DECODE,
        )
        rank_zero.send_activation(
            activation_one,
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=shape,
            phase=PIPELINE_PHASE_DECODE,
        )
        assert rank_zero.receive_tokens(element_count=1) == (17,)

        # A subsequent request/step reuses the same connection and continues
        # both sequence spaces rather than resetting them.
        rank_zero.send_activation(
            activation_two,
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=shape,
            phase=PIPELINE_PHASE_DECODE,
        )
        assert rank_zero.receive_tokens(element_count=1) == (23,)
        receiver.join(timeout=10)

        assert not receiver.is_alive()
        assert errors == []
        assert received == [
            bytes(prefill),
            bytes(activation_zero),
            bytes(activation_one),
            bytes(activation_two),
        ]
        assert received_dtype_codes == [PIPELINE_DTYPE_BFLOAT16] * 4
        assert rank_zero.prefill_activation_send_sequence == 1
        assert rank_one.prefill_activation_receive_sequence == 1
        assert rank_zero.decode_activation_send_sequence == 3
        assert rank_one.decode_activation_receive_sequence == 3
        assert rank_zero.token_receive_sequence == 2
        assert rank_one.token_send_sequence == 2


def test_secondary_ring_canary_is_synchronous_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire: Queue[mx.array] = Queue()

    with _ring_array_transport_pair(monkeypatch) as (
        rank_zero,
        rank_one,
        group,
    ):

        def send(
            activation: mx.array,
            destination: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            assert destination == 1
            assert group is rank_zero.secondary_ring_group
            assert stream is auto_parallel.pipeline_send_stream
            wire.put(activation)
            return activation

        def receive(
            shape: tuple[int, ...],
            dtype: mx.Dtype,
            source: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            assert shape == (1,)
            assert dtype == mx.int32
            assert source == 0
            assert group is rank_one.secondary_ring_group
            assert stream is auto_parallel.pipeline_receive_stream
            return wire.get(timeout=10)

        monkeypatch.setattr(mx.distributed, "send", send)
        monkeypatch.setattr(mx.distributed, "recv", receive)
        errors: list[BaseException] = []

        def run_rank_one() -> None:
            try:
                rank_one._run_secondary_ring_canary(rank_one.connection)  # type: ignore[arg-type]
            except BaseException as error:
                errors.append(error)

        receiver = threading.Thread(target=run_rank_one, daemon=True)
        receiver.start()
        rank_zero._run_secondary_ring_canary(rank_zero.connection)  # type: ignore[arg-type]
        receiver.join(timeout=10)

        assert not receiver.is_alive()
        assert errors == []
        assert group is rank_zero.secondary_ring_group


def test_first_rank_zero_ring_send_initializes_group_before_resolving_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (1, 1, 1, 4)
    activation = mx.arange(4, dtype=mx.float32).reshape(shape)
    expected_group = object()
    events: list[str] = []

    with _transport_pair(monkeypatch) as (rank_zero, _):
        rank_zero.activation_transport = "ring"
        rank_zero.activation_transport_code = (
            auto_parallel._PIPELINE_ACTIVATION_TRANSPORT_RING
        )
        assert rank_zero.secondary_ring_group is None

        def ensure_connected() -> object:
            events.append("ensure_connected")
            rank_zero.secondary_ring_group = expected_group  # type: ignore[assignment]
            return object()

        def send_header(**_: object) -> object:
            events.append("send_header")
            assert rank_zero.secondary_ring_group is expected_group
            return object()

        def receive_header(**_: object) -> tuple[object, int, int]:
            events.append("receive_header")
            return object(), auto_parallel.PIPELINE_DTYPE_NONE, 0

        def send(
            value: mx.array,
            destination: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            events.append("ring_send")
            assert value is activation
            assert destination == 1
            assert group is expected_group
            assert stream is auto_parallel.pipeline_send_stream
            return value

        monkeypatch.setattr(rank_zero, "ensure_connected", ensure_connected)
        monkeypatch.setattr(rank_zero, "_send_header", send_header)
        monkeypatch.setattr(rank_zero, "_receive_header", receive_header)
        monkeypatch.setattr(mx.distributed, "send", send)

        rank_zero.send_activation_array(
            activation,
            dtype_code=PIPELINE_DTYPE_FLOAT32,
            shape=shape,
            phase=PIPELINE_PHASE_DECODE,
        )

    assert events == [
        "ensure_connected",
        "send_header",
        "receive_header",
        "ring_send",
        "receive_header",
    ]


def test_ring_transport_orders_a0_a1_ack_then_t0_without_host_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 1, 2, 4)
    activations = [
        mx.array(
            np.arange(index * 16, (index + 1) * 16, dtype=np.float32).reshape(shape)
        )
        for index in range(2)
    ]
    receive_values: Queue[mx.array] = Queue()
    for activation in activations:
        receive_values.put(activation)
    scheduled: list[mx.array] = []
    received: list[np.ndarray] = []
    errors: list[BaseException] = []

    with _ring_array_transport_pair(monkeypatch) as (
        rank_zero,
        rank_one,
        expected_group,
    ):

        def send(
            activation: mx.array,
            destination: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            assert destination == 1
            assert group is expected_group
            assert stream is auto_parallel.pipeline_send_stream
            scheduled.append(activation)
            return activation

        def receive(
            requested_shape: tuple[int, ...],
            dtype: mx.Dtype,
            source: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            assert requested_shape == shape
            assert dtype == mx.float32
            assert source == 0
            assert group is expected_group
            assert stream is auto_parallel.pipeline_receive_stream
            # MLX constructs recv lazily before READY and executes it at eval.
            # Returning a prepared value keeps this Python fake nonblocking.
            return receive_values.get_nowait()

        monkeypatch.setattr(mx.distributed, "send", send)
        monkeypatch.setattr(mx.distributed, "recv", receive)

        def rank_one_loop() -> None:
            try:
                for _ in range(2):
                    dtype_code, activation = rank_one.receive_activation_array(
                        shape=shape,
                        phase=PIPELINE_PHASE_DECODE,
                    )
                    assert dtype_code == PIPELINE_DTYPE_FLOAT32
                    received.append(np.asarray(activation))
                rank_one.send_tokens([41])
            except BaseException as error:
                errors.append(error)

        receiver = threading.Thread(target=rank_one_loop, daemon=True)
        receiver.start()
        for activation in activations:
            rank_zero.send_activation_array(
                activation,
                dtype_code=PIPELINE_DTYPE_FLOAT32,
                shape=shape,
                phase=PIPELINE_PHASE_DECODE,
            )
        assert rank_zero.receive_tokens(element_count=1) == (41,)
        receiver.join(timeout=10)

        assert not receiver.is_alive()
        assert errors == []
        assert all(
            np.array_equal(actual, np.asarray(expected))
            for actual, expected in zip(received, activations, strict=True)
        )
        assert all(
            actual is expected
            for actual, expected in zip(scheduled, activations, strict=True)
        )
        assert rank_zero.decode_activation_send_sequence == 2
        assert rank_one.decode_activation_receive_sequence == 2
        assert rank_zero.token_receive_sequence == 1
        assert rank_one.token_send_sequence == 1


def test_ring_ready_preflight_rejects_bad_metadata_before_scheduling_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_shape = (2, 1, 2, 4)
    expected_shape = (2, 2, 1, 4)
    scheduled: list[mx.array] = []
    receiver_errors: list[BaseException] = []

    with _ring_array_transport_pair(monkeypatch) as (
        rank_zero,
        rank_one,
        _,
    ):

        def send(
            activation: mx.array,
            destination: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            scheduled.append(activation)
            return activation

        monkeypatch.setattr(mx.distributed, "send", send)

        def receive_bad_metadata() -> None:
            try:
                rank_one.receive_activation_array(
                    shape=expected_shape,
                    phase=PIPELINE_PHASE_DECODE,
                )
            except BaseException as error:
                receiver_errors.append(error)

        receiver = threading.Thread(target=receive_bad_metadata, daemon=True)
        receiver.start()
        with pytest.raises((ConnectionError, OSError, RuntimeError)):
            rank_zero.send_activation_array(
                mx.zeros(sent_shape, dtype=mx.float32),
                dtype_code=PIPELINE_DTYPE_FLOAT32,
                shape=sent_shape,
                phase=PIPELINE_PHASE_DECODE,
            )
        receiver.join(timeout=10)

        assert not receiver.is_alive()
        assert len(receiver_errors) == 1
        assert "shape" in str(receiver_errors[0])
        assert scheduled == []
        assert rank_zero._poisoned
        assert rank_one._poisoned


def test_queued_ring_prefill_retains_mlx_array_until_flush() -> None:
    clear_prefill_sends()
    shape = (1, 1, 1, 4)
    activation = mx.arange(4, dtype=mx.float32).reshape(shape)
    calls: list[mx.array] = []

    class _RecordingTransport:
        def send_activation_array(
            self,
            value: mx.array,
            *,
            dtype_code: int,
            shape: tuple[int, int, int, int],
            phase: int,
        ) -> None:
            assert dtype_code == PIPELINE_DTYPE_FLOAT32
            assert shape == (1, 1, 1, 4)
            assert phase == PIPELINE_PHASE_PREFILL
            calls.append(value)

        def abort(self, reason: str) -> None:
            raise AssertionError(reason)

    transport = _RecordingTransport()
    queue_pipeline_ring_activation(
        transport,  # type: ignore[arg-type]
        activation,
        dtype_code=PIPELINE_DTYPE_FLOAT32,
        shape=shape,
        phase=PIPELINE_PHASE_PREFILL,
    )
    assert auto_parallel._pending_ring_activations[0][1] is activation
    flush_prefill_sends()
    assert calls == [activation]
    assert auto_parallel._pending_ring_activations == []


def test_transport_rejects_same_count_with_wrong_exact_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_shape = (5, 1, 2, 4)
    expected_shape = (5, 2, 1, 4)
    receiver_errors: list[BaseException] = []

    with _transport_pair(monkeypatch) as (rank_zero, rank_one):

        def receive_wrong_shape() -> None:
            try:
                rank_one.receive_activation(
                    shape=expected_shape,
                    phase=PIPELINE_PHASE_DECODE,
                )
            except BaseException as error:
                receiver_errors.append(error)

        receiver = threading.Thread(target=receive_wrong_shape, daemon=True)
        receiver.start()
        # The receiver is allowed to close immediately after rejecting the
        # header, before the sender has finished the small payload write.
        with suppress(BrokenPipeError, ConnectionResetError):
            rank_zero.send_activation(
                _uint16_payload(0, sent_shape),
                dtype_code=PIPELINE_DTYPE_BFLOAT16,
                shape=sent_shape,
                phase=PIPELINE_PHASE_DECODE,
            )
        receiver.join(timeout=10)

        assert not receiver.is_alive()
        assert len(receiver_errors) == 1
        assert "shape" in str(receiver_errors[0])
        assert rank_one._poisoned


def test_transport_rejects_unknown_activation_dtype_from_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (1, 1, 1, 1)
    receiver_errors: list[BaseException] = []

    with _transport_pair(monkeypatch) as (rank_zero, rank_one):

        def receive_unknown_dtype() -> None:
            try:
                rank_one.receive_activation(
                    shape=shape,
                    phase=PIPELINE_PHASE_DECODE,
                )
            except BaseException as error:
                receiver_errors.append(error)

        receiver = threading.Thread(
            target=receive_unknown_dtype,
            daemon=True,
        )
        receiver.start()
        connection = rank_zero.ensure_connected()
        header = auto_parallel._PIPELINE_TCP_HEADER.pack(
            auto_parallel._PIPELINE_TCP_MAGIC,
            auto_parallel._PIPELINE_TCP_VERSION,
            auto_parallel._PIPELINE_TCP_ACTIVATION,
            PIPELINE_PHASE_DECODE,
            255,
            0,
            0,
            1,
            2,
            *shape,
        )
        connection.sendall(header)
        receiver.join(timeout=10)

        assert not receiver.is_alive()
        assert len(receiver_errors) == 1
        assert "unsupported" in str(receiver_errors[0])
        assert "dtype code 255" in str(receiver_errors[0])
        assert rank_one._poisoned


def test_transport_rejects_activation_above_configured_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay_port = _reserve_loopback_port()
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "127.0.0.1:49000")
    monkeypatch.setenv("EXO_PIPELINE_TOKEN_RELAY_PORT", str(relay_port))
    monkeypatch.setenv("EXO_PIPELINE_TCP_MAX_ACTIVATION_BYTES", "4")
    transport = _PipelineTcpTransport(rank=0, world_size=2)
    try:
        with pytest.raises(RuntimeError, match="exceeds bound"):
            transport.send_activation(
                memoryview(bytes(8)),
                dtype_code=PIPELINE_DTYPE_BFLOAT16,
                shape=(1, 1, 1, 4),
                phase=PIPELINE_PHASE_PREFILL,
            )
    finally:
        transport.close()


def test_discarding_queued_tcp_prefill_poisons_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_prefill_sends()
    relay_port = _reserve_loopback_port()
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "127.0.0.1:49000")
    monkeypatch.setenv("EXO_PIPELINE_TOKEN_RELAY_PORT", str(relay_port))
    transport = _PipelineTcpTransport(rank=0, world_size=2)
    try:
        queue_pipeline_tcp_activation(
            transport,
            memoryview(bytes(2)),
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=(1, 1, 1, 1),
            phase=PIPELINE_PHASE_PREFILL,
        )
        clear_prefill_sends()
        assert transport._poisoned
    finally:
        clear_prefill_sends()
        transport.close()


def test_agreed_cancel_discards_unsent_queues_and_transport_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_prefill_sends()
    shape = (1, 1, 1, 1)
    payload = _uint16_payload(9, shape)
    received: list[bytes] = []
    errors: list[BaseException] = []

    with _transport_pair(monkeypatch) as (rank_zero, rank_one):
        queue_pipeline_send(
            (mx.zeros(shape, dtype=mx.bfloat16),),
            destination=1,
            group=object(),  # type: ignore[arg-type]
        )
        queue_pipeline_tcp_activation(
            rank_zero,
            payload,
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=shape,
            phase=PIPELINE_PHASE_PREFILL,
        )
        queue_pipeline_ring_activation(
            rank_zero,
            mx.zeros(shape, dtype=mx.bfloat16),
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=shape,
            phase=PIPELINE_PHASE_PREFILL,
        )

        discard_unsent_prefill_sends_after_agreed_cancel()
        assert auto_parallel._pending_prefill_sends == []
        assert auto_parallel._pending_tcp_activations == []
        assert auto_parallel._pending_ring_activations == []
        assert not rank_zero._poisoned
        assert rank_zero.prefill_activation_send_sequence == 0

        def receive_after_cancel() -> None:
            try:
                _, actual = rank_one.receive_activation(
                    shape=shape,
                    phase=PIPELINE_PHASE_PREFILL,
                )
                received.append(bytes(actual))
            except BaseException as error:
                errors.append(error)

        receiver = threading.Thread(target=receive_after_cancel, daemon=True)
        receiver.start()
        rank_zero.send_activation(
            payload,
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=shape,
            phase=PIPELINE_PHASE_PREFILL,
        )
        receiver.join(timeout=10)

        assert not receiver.is_alive()
        assert errors == []
        assert received == [bytes(payload)]
        assert rank_zero.prefill_activation_send_sequence == 1
        assert rank_one.prefill_activation_receive_sequence == 1


def test_discarding_queued_ring_prefill_poisons_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_prefill_sends()
    relay_port = _reserve_loopback_port()
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "127.0.0.1:49000")
    monkeypatch.setenv("EXO_PIPELINE_TOKEN_RELAY_PORT", str(relay_port))
    transport = _PipelineTcpTransport(rank=0, world_size=2)
    transport.activation_transport = "ring"
    try:
        queue_pipeline_ring_activation(
            transport,
            mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
            dtype_code=PIPELINE_DTYPE_BFLOAT16,
            shape=(1, 1, 1, 1),
            phase=PIPELINE_PHASE_PREFILL,
        )
        clear_prefill_sends()
        assert transport._poisoned
    finally:
        clear_prefill_sends()
        transport.close()


class _IdentityLayer(nn.Module):
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x


def test_tcp_pipeline_decode_without_token_relay_fails_before_model_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "127.0.0.1:49000")
    first = KimiK3PipelineFirstLayer(
        _IdentityLayer(),
        rank=1,
        world_size=2,
        incoming_block_count=1,
        group=None,  # type: ignore[arg-type]
    )
    last = KimiK3PipelineLastLayer(
        _IdentityLayer(),
        rank=0,
        world_size=2,
        outgoing_block_count=1,
        final_block_count=1,
        group=None,  # type: ignore[arg-type]
    )
    x = mx.zeros((1, 1, 4))

    with pytest.raises(RuntimeError, match="logprobs are not supported"):
        first(x)
    with pytest.raises(RuntimeError, match="logprobs are not supported"):
        last(x)


def test_tcp_pipeline_requires_exactly_two_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "127.0.0.1:49000")
    with pytest.raises(ValueError, match="exactly two ranks"):
        KimiK3PipelineFirstLayer(
            _IdentityLayer(),
            rank=1,
            world_size=3,
            incoming_block_count=1,
            group=None,  # type: ignore[arg-type]
        )
