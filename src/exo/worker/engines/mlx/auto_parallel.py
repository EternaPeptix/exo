import atexit
import ipaddress
import json
import os
import socket
import struct
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Literal, Protocol, cast, final

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx.nn.layers.distributed import (
    shard_inplace,
    shard_linear,
    sum_gradients,
)
from mlx_lm.models.base import (
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3MLP
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.deepseek_v4 import DeepseekV4MoE, V4Attention
from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
from mlx_lm.models.deepseek_v32 import DeepseekV32MLP
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gemma4 import Model as Gemma4Model
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe import MoE
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer, Glm4MoeLiteMLP
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel
from mlx_lm.models.gpt_oss import GptOssMoeModel
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.kimi_k25 import Model as KimiK25Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.minimax import MiniMaxAttention
from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.ministral3 import Model as Ministral3Model
from mlx_lm.models.nemotron_h import Model as NemotronHModel
from mlx_lm.models.nemotron_h import (
    NemotronHAttention,
    NemotronHMamba2Mixer,
    NemotronHMoE,
)
from mlx_lm.models.nemotron_h import NemotronHModel as NemotronHInnerModel
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3TransformerBlock
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen3_5DecoderLayer
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel as Qwen3_5TextModelInner
from mlx_lm.models.qwen3_5 import SparseMoeBlock as Qwen3_5SparseMoeBlock
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from mlx_lm.models.qwen3_next import Qwen3NextModel as Qwen3NextInnerModel
from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_lm.models.step3p5 import Model as Step35Model
from mlx_lm.models.step3p5 import Step3p5MLP as Step35MLP
from mlx_lm.models.step3p5 import Step3p5Model as Step35InnerModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache


@final
class PipelineDecodeTimings:
    """Accumulates per-token pipeline communication timings during decode.

    Timings are collected inside ``PipelineFirstLayer`` / ``PipelineLastLayer``
    and logged as a rolling summary every ``log_every`` decode steps, so a
    ``-vv`` run attributes per-token latency to recv / send / gather phases.
    """

    def __init__(self, log_every: int | None = None) -> None:
        if log_every is None:
            # This source tree is dedicated to the K3 experiment. Keep the
            # diagnostic window short enough to attribute a bounded smoke run;
            # production deployments can restore 64 through the environment.
            log_every = int(os.environ.get("EXO_PIPELINE_TIMING_LOG_EVERY", "4"))
        if log_every <= 0:
            raise ValueError("pipeline timing log interval must be positive")
        self.log_every = log_every
        self.recv_seconds = 0.0
        self.send_seconds = 0.0
        self.gather_seconds = 0.0
        self.steps = 0

    def record_recv(self, seconds: float) -> None:
        self.recv_seconds += seconds

    def record_send(self, seconds: float) -> None:
        self.send_seconds += seconds

    def record_gather_and_advance(self, seconds: float) -> None:
        """The gather (or relay) phase runs once per decode step, so it also
        advances the step counter and emits the periodic summary."""
        self.gather_seconds += seconds
        self.steps += 1
        if self.steps % self.log_every == 0:
            per_step_ms = 1000.0 / self.log_every
            logger.debug(
                "pipeline decode comm (avg over "
                f"{self.log_every} tokens): "
                f"recv={self.recv_seconds * per_step_ms:.2f}ms "
                f"send={self.send_seconds * per_step_ms:.2f}ms "
                f"token_wait={self.gather_seconds * per_step_ms:.2f}ms"
            )
            self.recv_seconds = 0.0
            self.send_seconds = 0.0
            self.gather_seconds = 0.0


decode_timings = PipelineDecodeTimings()

pipeline_send_stream = mx.new_stream(mx.cpu)
pipeline_receive_stream = mx.new_stream(mx.cpu)


PipelineSendPayloads = tuple[mx.array, ...]
_pending_prefill_sends: list[
    tuple[PipelineSendPayloads, int, mx.distributed.Group]
] = []
_inflight_prefill_send: mx.array | None = None
_inflight_decode_sends: list[mx.array] = []
_pending_tcp_activations: list[
    tuple[
        "_PipelineTcpTransport",
        memoryview,
        int,
        tuple[int, int, int, int],
        int,
    ]
] = []
_pending_ring_activations: list[
    tuple[
        "_PipelineTcpTransport",
        mx.array,
        int,
        tuple[int, int, int, int],
        int,
    ]
] = []
_pipeline_tcp_transport: "_PipelineTcpTransport | None" = None

PIPELINE_DTYPE_NONE = 0
PIPELINE_DTYPE_BFLOAT16 = 1
PIPELINE_DTYPE_FLOAT16 = 2
PIPELINE_DTYPE_FLOAT32 = 3
PIPELINE_DTYPE_INT32 = 4

_PIPELINE_TCP_MAGIC = b"EXOK3PP2"
_PIPELINE_TCP_VERSION = 3
_PIPELINE_TCP_ACTIVATION = 1
_PIPELINE_TCP_TOKEN = 2
_PIPELINE_TCP_ACTIVATION_ACK = 3
_PIPELINE_TCP_ACTIVATION_READY = 4
PIPELINE_PHASE_PREFILL = 1
PIPELINE_PHASE_DECODE = 2
_PIPELINE_TCP_HEADER = struct.Struct("!8sBBBBQQQQQQQQ")
_PIPELINE_TCP_HELLO = struct.Struct("!8sBBBBQ")
_PIPELINE_TCP_ADDRESS_LENGTH = struct.Struct("!I")
_PIPELINE_TCP_MAX_ADDRESS_BYTES = 1024
_PIPELINE_TCP_RING_PORT_COUNT = struct.Struct("!B")
_PIPELINE_TCP_RING_PORT = struct.Struct("!H")
_PIPELINE_TCP_DEFAULT_MAX_ACTIVATION_BYTES = 512 * 1024 * 1024
_PIPELINE_TCP_MAX_TOKEN_BYTES = 1024 * 1024
_PIPELINE_DYNAMIC_PORT_START = 49152
_PIPELINE_DYNAMIC_PORT_COUNT = 16384
_PIPELINE_ACTIVATION_TRANSPORT_TCP = 1
_PIPELINE_ACTIVATION_TRANSPORT_RING = 2
_PIPELINE_RING_CANARY_VALUE = 0x4B335250
_PIPELINE_RING_CANARY_ACK = b"\xa5"
_PIPELINE_DTYPE_ITEMSIZE = {
    PIPELINE_DTYPE_BFLOAT16: 2,
    PIPELINE_DTYPE_FLOAT16: 2,
    PIPELINE_DTYPE_FLOAT32: 4,
    PIPELINE_DTYPE_INT32: 4,
}


def _pipeline_activation_transport() -> str:
    transport = (
        os.environ.get(
            "EXO_K3_PIPELINE_ACTIVATION_TRANSPORT",
            "ring",
        )
        .strip()
        .lower()
    )
    if transport not in ("ring", "tcp"):
        raise ValueError("EXO_K3_PIPELINE_ACTIVATION_TRANSPORT must be 'ring' or 'tcp'")
    return transport


def _validate_pipeline_ring_addresses(
    addresses: list[str],
    *,
    require_link_local: bool,
) -> tuple[str, ...]:
    if not 1 <= len(addresses) <= 4:
        raise RuntimeError(
            "Kimi-K3 secondary ring requires between one and four IPv4 addresses"
        )
    validated: list[str] = []
    for raw_address in addresses:
        try:
            address = ipaddress.IPv4Address(raw_address)
        except ipaddress.AddressValueError as error:
            raise RuntimeError(
                f"invalid Kimi-K3 secondary ring IPv4 address {raw_address!r}"
            ) from error
        if address.is_unspecified or address.is_multicast or address.is_loopback:
            raise RuntimeError(f"unsafe Kimi-K3 secondary ring IPv4 address {address}")
        if require_link_local and not address.is_link_local:
            raise RuntimeError(
                "auto-discovered Kimi-K3 secondary ring address is not "
                f"link-local: {address}"
            )
        canonical = str(address)
        if canonical in validated:
            raise RuntimeError(f"duplicate Kimi-K3 secondary ring address {canonical}")
        validated.append(canonical)
    return tuple(validated)


def _discover_pipeline_ring_addresses() -> tuple[str, ...]:
    override = os.environ.get("EXO_K3_PIPELINE_RING_ADDRESSES")
    if override is not None:
        addresses = [
            address.strip() for address in override.split(",") if address.strip()
        ]
        return _validate_pipeline_ring_addresses(
            addresses,
            require_link_local=False,
        )

    addresses: list[str] = []
    interfaces = psutil.net_if_addrs()
    for interface in ("en3", "en4", "en5", "en6"):
        for entry in interfaces.get(interface, ()):
            if entry.family != socket.AF_INET:
                continue
            address = ipaddress.IPv4Address(entry.address)
            if address.is_link_local:
                addresses.append(str(address))
    return _validate_pipeline_ring_addresses(
        addresses,
        require_link_local=True,
    )


def _pipeline_ring_ports(
    coordinator_port: int,
    relay_port: int,
    count: int,
) -> tuple[int, ...]:
    override = os.environ.get("EXO_K3_PIPELINE_RING_BASE_PORT")
    if override is None:
        offset = (
            coordinator_port
            - _PIPELINE_DYNAMIC_PORT_START
            + (_PIPELINE_DYNAMIC_PORT_COUNT // 4)
        ) % _PIPELINE_DYNAMIC_PORT_COUNT
        base_port = _PIPELINE_DYNAMIC_PORT_START + offset
    else:
        try:
            base_port = int(override)
        except ValueError as error:
            raise ValueError(
                "EXO_K3_PIPELINE_RING_BASE_PORT must be an integer"
            ) from error
        if not (
            _PIPELINE_DYNAMIC_PORT_START
            <= base_port
            < _PIPELINE_DYNAMIC_PORT_START + _PIPELINE_DYNAMIC_PORT_COUNT
        ):
            raise ValueError(
                "EXO_K3_PIPELINE_RING_BASE_PORT must be in the dynamic port range"
            )

    ports = tuple(
        _PIPELINE_DYNAMIC_PORT_START
        + (base_port - _PIPELINE_DYNAMIC_PORT_START + index)
        % _PIPELINE_DYNAMIC_PORT_COUNT
        for index in range(count)
    )
    if coordinator_port in ports or relay_port in ports:
        raise RuntimeError(
            "Kimi-K3 secondary ring port collides with its control transport"
        )
    return ports


def _pipeline_dtype_from_code(dtype_code: int) -> mx.Dtype:
    if dtype_code == PIPELINE_DTYPE_BFLOAT16:
        return mx.bfloat16
    if dtype_code == PIPELINE_DTYPE_FLOAT16:
        return mx.float16
    if dtype_code == PIPELINE_DTYPE_FLOAT32:
        return mx.float32
    raise RuntimeError(f"unsupported Kimi-K3 activation dtype code {dtype_code}")


def _pipeline_ring_hosts(
    rank: int,
    local_addresses: tuple[str, ...],
    remote_addresses: tuple[str, ...],
    ports: tuple[int, ...],
) -> list[list[str]]:
    if len(local_addresses) != len(remote_addresses):
        raise RuntimeError(
            "Kimi-K3 secondary ring peers discovered different rail counts: "
            f"{len(local_addresses)} != {len(remote_addresses)}"
        )
    if len(local_addresses) != len(ports):
        raise RuntimeError("Kimi-K3 secondary ring address/port count mismatch")
    local_hosts = [
        f"{address}:{port}"
        for address, port in zip(local_addresses, ports, strict=True)
    ]
    remote_hosts = [
        f"{address}:{port}"
        for address, port in zip(remote_addresses, ports, strict=True)
    ]
    return [local_hosts, remote_hosts] if rank == 0 else [remote_hosts, local_hosts]


@final
class _PipelineTcpTransport:
    """Ordered PP2 Kimi-K3 control and activation transport.

    JACCL's point-to-point completion semantics permit a receiver to return
    before the matching remote send has retired. Switching direction at that
    point can leave both striped rings in ``RingImpl::send``. K3 therefore uses
    one persistent TCP connection for metadata, acknowledgements, and reverse
    sampled tokens. Forward activation payloads use an independent MLX TCP ring
    by default; ``EXO_K3_PIPELINE_ACTIVATION_TRANSPORT=tcp`` retains the
    complete framed-TCP fallback. JACCL remains available for matched
    collectives.

    MLX caches distributed groups for the process lifetime and its ring
    operations do not expose a Python timeout. Any failed canary, ring
    operation, or control frame poisons this transport permanently: EXO must
    restart the runner subprocess rather than attempting an in-process
    reconnect.

    Rank zero sends activation frames and receives token frames. Rank one
    receives activations and sends tokens. The two directions have independent,
    process-lifetime sequence numbers so the first decode exchange is exactly
    ``A0, A1, T0`` and later requests continue rather than resetting counters.
    """

    def __init__(self, rank: int, world_size: int) -> None:
        if world_size != 2:
            raise ValueError(
                "Kimi-K3 TCP pipeline transport requires exactly two ranks"
            )
        if rank not in (0, 1):
            raise ValueError(f"invalid Kimi-K3 pipeline rank {rank}")
        coordinator = os.environ.get("MLX_JACCL_COORDINATOR")
        if coordinator is None:
            raise RuntimeError(
                "Kimi-K3 TCP pipeline transport requires the JACCL backend"
            )
        try:
            coordinator_host, coordinator_port_text = coordinator.rsplit(":", 1)
            coordinator_port = int(coordinator_port_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid MLX_JACCL_COORDINATOR {coordinator!r}"
            ) from error

        # Toggle one bit inside the IANA dynamic range. This is deterministic,
        # non-identity and bijective for EXO's 49152-65535 coordinator ports.
        default_port = coordinator_port ^ 0x2000
        relay_port = int(
            os.environ.get("EXO_PIPELINE_TOKEN_RELAY_PORT", str(default_port))
        )
        if not 1024 <= relay_port <= 65535:
            raise ValueError(f"invalid Kimi-K3 TCP pipeline port {relay_port}")

        timeout_seconds = float(
            os.environ.get("EXO_PIPELINE_TCP_TIMEOUT_SECONDS", "120")
        )
        if timeout_seconds <= 0:
            raise ValueError("EXO_PIPELINE_TCP_TIMEOUT_SECONDS must be positive")
        max_activation_bytes = int(
            os.environ.get(
                "EXO_PIPELINE_TCP_MAX_ACTIVATION_BYTES",
                str(_PIPELINE_TCP_DEFAULT_MAX_ACTIVATION_BYTES),
            )
        )
        if max_activation_bytes <= 0:
            raise ValueError("EXO_PIPELINE_TCP_MAX_ACTIVATION_BYTES must be positive")

        self.rank = rank
        self.world_size = world_size
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.relay_port = relay_port
        self.timeout_seconds = timeout_seconds
        self.max_activation_bytes = max_activation_bytes
        self.activation_transport = _pipeline_activation_transport()
        self.activation_transport_code = (
            _PIPELINE_ACTIVATION_TRANSPORT_RING
            if self.activation_transport == "ring"
            else _PIPELINE_ACTIVATION_TRANSPORT_TCP
        )
        self.local_ring_addresses = (
            _discover_pipeline_ring_addresses()
            if self.activation_transport == "ring"
            else ()
        )
        self.secondary_ring_group: mx.distributed.Group | None = None
        self.connection: socket.socket | None = None
        self.listener: socket.socket | None = None
        self.prefill_activation_send_sequence = 0
        self.prefill_activation_receive_sequence = 0
        self.decode_activation_send_sequence = 0
        self.decode_activation_receive_sequence = 0
        self.token_send_sequence = 0
        self.token_receive_sequence = 0
        self._ever_connected = False
        self._poisoned = False
        self._activation_telemetry_logged = False
        self._io_lock = threading.Lock()

        if rank == 0:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", relay_port))
            listener.listen(1)
            listener.settimeout(timeout_seconds)
            self.listener = listener

        atexit.register(self.close)
        logger.info(
            "Kimi-K3 pipeline control transport using TCP "
            f"rank={rank} endpoint={coordinator_host}:{relay_port} "
            f"activation_transport={self.activation_transport}"
        )

    def _configure(self, connection: socket.socket) -> socket.socket:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        connection.settimeout(self.timeout_seconds)
        return connection

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytearray:
        data = bytearray(size)
        view = memoryview(data)
        received = 0
        while received < size:
            count = connection.recv_into(view[received:])
            if count == 0:
                raise ConnectionError(
                    "Kimi-K3 TCP pipeline connection closed "
                    f"after {received}/{size} bytes"
                )
            received += count
        return data

    def _exchange_hello(self, connection: socket.socket) -> None:
        local = _PIPELINE_TCP_HELLO.pack(
            _PIPELINE_TCP_MAGIC,
            _PIPELINE_TCP_VERSION,
            self.rank,
            self.world_size,
            self.activation_transport_code,
            self.coordinator_port,
        )
        if self.rank == 1:
            connection.sendall(local)
            remote = self._recv_exact(connection, _PIPELINE_TCP_HELLO.size)
        else:
            remote = self._recv_exact(connection, _PIPELINE_TCP_HELLO.size)
            connection.sendall(local)

        (
            magic,
            version,
            rank,
            world_size,
            activation_transport,
            coordinator_port,
        ) = _PIPELINE_TCP_HELLO.unpack(remote)
        expected_rank = 1 - self.rank
        if (
            magic != _PIPELINE_TCP_MAGIC
            or version != _PIPELINE_TCP_VERSION
            or rank != expected_rank
            or world_size != self.world_size
            or activation_transport != self.activation_transport_code
            or coordinator_port != self.coordinator_port
        ):
            raise RuntimeError(
                "Kimi-K3 TCP pipeline handshake mismatch: "
                f"magic={magic!r} version={version} rank={rank} "
                f"world_size={world_size} "
                f"activation_transport={activation_transport} "
                f"coordinator_port={coordinator_port}"
            )

    @staticmethod
    def _send_address_list(
        connection: socket.socket,
        addresses: tuple[str, ...],
    ) -> None:
        payload = json.dumps(
            list(addresses),
            separators=(",", ":"),
        ).encode("ascii")
        if len(payload) > _PIPELINE_TCP_MAX_ADDRESS_BYTES:
            raise RuntimeError("Kimi-K3 secondary ring address list exceeds bound")
        connection.sendall(_PIPELINE_TCP_ADDRESS_LENGTH.pack(len(payload)))
        connection.sendall(payload)

    def _receive_address_list(
        self,
        connection: socket.socket,
    ) -> tuple[str, ...]:
        length_bytes = self._recv_exact(
            connection,
            _PIPELINE_TCP_ADDRESS_LENGTH.size,
        )
        (payload_size,) = _PIPELINE_TCP_ADDRESS_LENGTH.unpack(length_bytes)
        if payload_size > _PIPELINE_TCP_MAX_ADDRESS_BYTES:
            raise RuntimeError("Kimi-K3 secondary ring address list exceeds bound")
        payload = self._recv_exact(connection, payload_size)
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(
                "invalid Kimi-K3 secondary ring address payload"
            ) from error
        if not isinstance(decoded, list) or not all(
            isinstance(address, str) for address in decoded
        ):
            raise RuntimeError("Kimi-K3 secondary ring addresses must be a string list")
        return _validate_pipeline_ring_addresses(
            decoded,
            require_link_local=False,
        )

    @staticmethod
    def _send_ring_ports(
        connection: socket.socket,
        ports: tuple[int, ...],
    ) -> None:
        if not 1 <= len(ports) <= 4:
            raise RuntimeError(
                "Kimi-K3 secondary ring requires between one and four ports"
            )
        connection.sendall(_PIPELINE_TCP_RING_PORT_COUNT.pack(len(ports)))
        for port in ports:
            connection.sendall(_PIPELINE_TCP_RING_PORT.pack(port))

    def _receive_ring_ports(
        self,
        connection: socket.socket,
    ) -> tuple[int, ...]:
        count_bytes = self._recv_exact(
            connection,
            _PIPELINE_TCP_RING_PORT_COUNT.size,
        )
        (count,) = _PIPELINE_TCP_RING_PORT_COUNT.unpack(count_bytes)
        if count != len(self.local_ring_addresses):
            raise RuntimeError(
                "Kimi-K3 secondary ring port count does not match local rails: "
                f"{count} != {len(self.local_ring_addresses)}"
            )
        ports = tuple(
            _PIPELINE_TCP_RING_PORT.unpack(
                self._recv_exact(connection, _PIPELINE_TCP_RING_PORT.size)
            )[0]
            for _ in range(count)
        )
        if len(set(ports)) != len(ports):
            raise RuntimeError("Kimi-K3 secondary ring ports are not unique")
        if any(
            not (
                _PIPELINE_DYNAMIC_PORT_START
                <= port
                < _PIPELINE_DYNAMIC_PORT_START + _PIPELINE_DYNAMIC_PORT_COUNT
            )
            for port in ports
        ):
            raise RuntimeError(
                "Kimi-K3 secondary ring port is outside the dynamic range"
            )
        if self.coordinator_port in ports or self.relay_port in ports:
            raise RuntimeError(
                "Kimi-K3 secondary ring port collides with its control transport"
            )
        return ports

    def _exchange_ring_configuration(
        self,
        connection: socket.socket,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        if self.rank == 1:
            self._send_address_list(connection, self.local_ring_addresses)
            remote_addresses = self._receive_address_list(connection)
            ports = self._receive_ring_ports(connection)
            return remote_addresses, ports
        remote_addresses = self._receive_address_list(connection)
        ports = _pipeline_ring_ports(
            self.coordinator_port,
            self.relay_port,
            len(self.local_ring_addresses),
        )
        self._send_address_list(connection, self.local_ring_addresses)
        self._send_ring_ports(connection, ports)
        return remote_addresses, ports

    def _run_secondary_ring_canary(self, connection: socket.socket) -> None:
        group = self.secondary_ring_group
        if group is None:
            raise RuntimeError("Kimi-K3 secondary ring group is unavailable")
        if self.rank == 0:
            canary = mx.array([_PIPELINE_RING_CANARY_VALUE], dtype=mx.int32)
            dependency = mx.distributed.send(
                canary,
                1,
                group=group,
                stream=pipeline_send_stream,
            )
            mx.eval(dependency)
            acknowledgement = self._recv_exact(
                connection,
                len(_PIPELINE_RING_CANARY_ACK),
            )
            if bytes(acknowledgement) != _PIPELINE_RING_CANARY_ACK:
                raise RuntimeError(
                    "Kimi-K3 secondary ring canary acknowledgement mismatch"
                )
        else:
            canary = mx.distributed.recv(
                (1,),
                mx.int32,
                0,
                group=group,
                stream=pipeline_receive_stream,
            )
            mx.eval(canary)
            if int(canary.item()) != _PIPELINE_RING_CANARY_VALUE:
                raise RuntimeError("Kimi-K3 secondary ring canary payload mismatch")
            connection.sendall(_PIPELINE_RING_CANARY_ACK)

    def _initialize_secondary_ring(self, connection: socket.socket) -> None:
        if self.activation_transport != "ring":
            return
        if not mx.distributed.is_available("ring"):
            raise RuntimeError("Kimi-K3 secondary MLX ring backend is unavailable")
        remote_addresses, ports = self._exchange_ring_configuration(connection)
        hosts = _pipeline_ring_hosts(
            self.rank,
            self.local_ring_addresses,
            remote_addresses,
            ports,
        )

        previous_hostfile = os.environ.get("MLX_HOSTFILE")
        previous_rank = os.environ.get("MLX_RANK")
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                encoding="utf-8",
            ) as hostfile:
                json.dump(hosts, hostfile, separators=(",", ":"))
                hostfile.flush()
                os.environ["MLX_HOSTFILE"] = hostfile.name
                os.environ["MLX_RANK"] = str(self.rank)
                group = mx.distributed.init(backend="ring", strict=True)
        finally:
            if previous_hostfile is None:
                os.environ.pop("MLX_HOSTFILE", None)
            else:
                os.environ["MLX_HOSTFILE"] = previous_hostfile
            if previous_rank is None:
                os.environ.pop("MLX_RANK", None)
            else:
                os.environ["MLX_RANK"] = previous_rank

        if group.rank() != self.rank or group.size() != self.world_size:
            raise RuntimeError(
                "Kimi-K3 secondary ring returned stale rank/world: "
                f"rank={group.rank()} size={group.size()}, "
                f"expected rank={self.rank} size={self.world_size}; "
                "runner restart required"
            )
        self.secondary_ring_group = group
        self._run_secondary_ring_canary(connection)
        logger.info(
            "Kimi-K3 secondary MLX ring initialized "
            f"rank={self.rank} rails={len(self.local_ring_addresses)} "
            f"ports={ports}"
        )

    def ensure_connected(self) -> socket.socket:
        if self._poisoned:
            raise RuntimeError("Kimi-K3 TCP pipeline transport is poisoned")
        if self.connection is not None:
            return self.connection
        if self._ever_connected:
            raise RuntimeError(
                "Kimi-K3 TCP pipeline connection was lost; runner restart required"
            )

        try:
            if self.rank == 0:
                if self.listener is None:
                    raise RuntimeError("Kimi-K3 TCP pipeline listener is unavailable")
                connection, _ = self.listener.accept()
                connection = self._configure(connection)
                self.listener.close()
                self.listener = None
            else:
                deadline = time.monotonic() + self.timeout_seconds
                last_error: OSError | None = None
                while True:
                    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        connection.settimeout(min(5.0, self.timeout_seconds))
                        connection.connect((self.coordinator_host, self.relay_port))
                        connection = self._configure(connection)
                        break
                    except OSError as error:
                        last_error = error
                        connection.close()
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "unable to connect Kimi-K3 TCP pipeline transport"
                            ) from last_error
                        time.sleep(0.05)

            self._exchange_hello(connection)
            self._initialize_secondary_ring(connection)
            self.connection = connection
            self._ever_connected = True
            return connection
        except Exception:
            self._poisoned = True
            self.close()
            raise

    def _validate_frame_values(
        self,
        *,
        kind: int,
        phase: int,
        dtype_code: int,
        element_count: int,
        payload_bytes: int,
        shape: tuple[int, int, int, int],
        max_payload_bytes: int,
    ) -> None:
        if kind in (
            _PIPELINE_TCP_ACTIVATION_ACK,
            _PIPELINE_TCP_ACTIVATION_READY,
        ):
            control_name = "ACK" if kind == _PIPELINE_TCP_ACTIVATION_ACK else "READY"
            if phase not in (PIPELINE_PHASE_PREFILL, PIPELINE_PHASE_DECODE):
                raise RuntimeError(
                    f"invalid Kimi-K3 activation {control_name} phase {phase}"
                )
            if dtype_code != PIPELINE_DTYPE_NONE:
                raise RuntimeError(
                    f"invalid Kimi-K3 activation {control_name} dtype code {dtype_code}"
                )
            if element_count != 0 or payload_bytes != 0 or shape != (0, 0, 0, 0):
                raise RuntimeError(
                    f"invalid Kimi-K3 activation {control_name} payload descriptor"
                )
            return

        item_size = _PIPELINE_DTYPE_ITEMSIZE.get(dtype_code)
        if item_size is None:
            raise RuntimeError(
                f"unsupported Kimi-K3 TCP pipeline dtype code {dtype_code}"
            )
        if element_count < 0:
            raise RuntimeError("negative Kimi-K3 TCP pipeline element count")
        if payload_bytes != element_count * item_size:
            raise RuntimeError(
                "Kimi-K3 TCP pipeline frame size mismatch: "
                f"{payload_bytes} bytes for {element_count} elements "
                f"of dtype code {dtype_code}"
            )
        if payload_bytes > max_payload_bytes:
            raise RuntimeError(
                "Kimi-K3 TCP pipeline payload exceeds bound: "
                f"{payload_bytes} > {max_payload_bytes}"
            )
        if kind == _PIPELINE_TCP_ACTIVATION:
            if dtype_code not in (
                PIPELINE_DTYPE_BFLOAT16,
                PIPELINE_DTYPE_FLOAT16,
                PIPELINE_DTYPE_FLOAT32,
            ):
                raise RuntimeError(
                    f"unsupported Kimi-K3 TCP activation dtype code {dtype_code}"
                )
            if phase not in (PIPELINE_PHASE_PREFILL, PIPELINE_PHASE_DECODE):
                raise RuntimeError(f"invalid Kimi-K3 activation phase {phase}")
            if any(dimension <= 0 for dimension in shape):
                raise RuntimeError(f"invalid Kimi-K3 activation shape {shape}")
            shape_count = 1
            for dimension in shape:
                shape_count *= dimension
            if shape_count != element_count:
                raise RuntimeError(
                    f"Kimi-K3 activation shape {shape} has {shape_count} "
                    f"elements, frame declares {element_count}"
                )
        elif kind == _PIPELINE_TCP_TOKEN:
            if dtype_code != PIPELINE_DTYPE_INT32:
                raise RuntimeError(f"invalid Kimi-K3 token dtype code {dtype_code}")
            if phase != PIPELINE_PHASE_DECODE:
                raise RuntimeError(f"invalid Kimi-K3 token phase {phase}")
            if shape != (element_count, 0, 0, 0):
                raise RuntimeError(f"invalid Kimi-K3 token shape descriptor {shape}")
        else:
            raise RuntimeError(f"invalid Kimi-K3 TCP frame kind {kind}")

    def _send_header(
        self,
        *,
        kind: int,
        phase: int,
        dtype_code: int,
        sequence: int,
        correlation_sequence: int,
        element_count: int,
        shape: tuple[int, int, int, int],
        payload_bytes: int,
        max_payload_bytes: int,
    ) -> socket.socket:
        self._validate_frame_values(
            kind=kind,
            phase=phase,
            dtype_code=dtype_code,
            element_count=element_count,
            payload_bytes=payload_bytes,
            shape=shape,
            max_payload_bytes=max_payload_bytes,
        )
        header = _PIPELINE_TCP_HEADER.pack(
            _PIPELINE_TCP_MAGIC,
            _PIPELINE_TCP_VERSION,
            kind,
            phase,
            dtype_code,
            sequence,
            correlation_sequence,
            element_count,
            payload_bytes,
            *shape,
        )
        connection = self.ensure_connected()
        try:
            connection.sendall(header)
        except Exception:
            self._poisoned = True
            self.close()
            raise
        return connection

    def _receive_header(
        self,
        *,
        expected_kind: int,
        expected_phase: int,
        expected_dtype_code: int | None,
        expected_sequence: int,
        expected_correlation_sequence: int,
        expected_element_count: int,
        expected_shape: tuple[int, int, int, int],
        max_payload_bytes: int,
    ) -> tuple[socket.socket, int, int]:
        connection = self.ensure_connected()
        try:
            header = self._recv_exact(connection, _PIPELINE_TCP_HEADER.size)
            (
                magic,
                version,
                kind,
                phase,
                dtype_code,
                sequence,
                correlation_sequence,
                element_count,
                payload_bytes,
                shape_0,
                shape_1,
                shape_2,
                shape_3,
            ) = _PIPELINE_TCP_HEADER.unpack(header)
            shape = (shape_0, shape_1, shape_2, shape_3)
            if magic != _PIPELINE_TCP_MAGIC or version != _PIPELINE_TCP_VERSION:
                raise RuntimeError("invalid Kimi-K3 TCP pipeline frame magic/version")
            if kind != expected_kind:
                raise RuntimeError(
                    f"Kimi-K3 TCP pipeline frame kind {kind}, expected {expected_kind}"
                )
            if phase != expected_phase:
                raise RuntimeError(
                    f"Kimi-K3 TCP pipeline phase {phase}, expected {expected_phase}"
                )
            if expected_dtype_code is not None and dtype_code != expected_dtype_code:
                raise RuntimeError(
                    f"Kimi-K3 TCP pipeline dtype code {dtype_code}, "
                    f"expected {expected_dtype_code}"
                )
            if sequence != expected_sequence:
                raise RuntimeError(
                    f"Kimi-K3 TCP pipeline sequence {sequence}, "
                    f"expected {expected_sequence}"
                )
            if correlation_sequence != expected_correlation_sequence:
                raise RuntimeError(
                    "Kimi-K3 TCP pipeline correlation sequence "
                    f"{correlation_sequence}, "
                    f"expected {expected_correlation_sequence}"
                )
            if element_count != expected_element_count:
                raise RuntimeError(
                    f"Kimi-K3 TCP pipeline element count {element_count}, "
                    f"expected {expected_element_count}"
                )
            if shape != expected_shape:
                raise RuntimeError(
                    f"Kimi-K3 TCP pipeline shape {shape}, expected {expected_shape}"
                )
            self._validate_frame_values(
                kind=kind,
                phase=phase,
                dtype_code=dtype_code,
                element_count=element_count,
                payload_bytes=payload_bytes,
                shape=shape,
                max_payload_bytes=max_payload_bytes,
            )
            return connection, dtype_code, payload_bytes
        except Exception:
            self._poisoned = True
            self.close()
            raise

    def _send_frame(
        self,
        *,
        kind: int,
        phase: int,
        dtype_code: int,
        sequence: int,
        correlation_sequence: int,
        element_count: int,
        shape: tuple[int, int, int, int],
        payload: bytes | bytearray | memoryview,
        max_payload_bytes: int,
    ) -> None:
        connection = self._send_header(
            kind=kind,
            phase=phase,
            dtype_code=dtype_code,
            sequence=sequence,
            correlation_sequence=correlation_sequence,
            element_count=element_count,
            shape=shape,
            payload_bytes=len(payload),
            max_payload_bytes=max_payload_bytes,
        )
        try:
            connection.sendall(payload)
        except Exception:
            self._poisoned = True
            self.close()
            raise

    def _receive_frame(
        self,
        *,
        expected_kind: int,
        expected_phase: int,
        expected_dtype_code: int | None,
        expected_sequence: int,
        expected_correlation_sequence: int,
        expected_element_count: int,
        expected_shape: tuple[int, int, int, int],
        max_payload_bytes: int,
    ) -> tuple[int, bytearray]:
        connection, dtype_code, payload_bytes = self._receive_header(
            expected_kind=expected_kind,
            expected_phase=expected_phase,
            expected_dtype_code=expected_dtype_code,
            expected_sequence=expected_sequence,
            expected_correlation_sequence=expected_correlation_sequence,
            expected_element_count=expected_element_count,
            expected_shape=expected_shape,
            max_payload_bytes=max_payload_bytes,
        )
        try:
            return dtype_code, self._recv_exact(connection, payload_bytes)
        except Exception:
            self._poisoned = True
            self.close()
            raise

    def _log_activation_telemetry(
        self,
        *,
        dtype_code: int,
        shape: tuple[int, int, int, int],
        logical_bytes: int,
    ) -> None:
        if self._activation_telemetry_logged:
            return
        dtype = _pipeline_dtype_from_code(dtype_code)
        logger.info(
            "Kimi-K3 pipeline activation "
            f"rank={self.rank} transport={self.activation_transport} "
            f"dtype={dtype} shape={shape} logical_bytes={logical_bytes}"
        )
        self._activation_telemetry_logged = True

    def _activation_send_sequence(self, phase: int) -> int:
        if phase == PIPELINE_PHASE_PREFILL:
            return self.prefill_activation_send_sequence
        if phase == PIPELINE_PHASE_DECODE:
            return self.decode_activation_send_sequence
        raise RuntimeError(f"invalid Kimi-K3 activation phase {phase}")

    def _activation_receive_sequence(self, phase: int) -> int:
        if phase == PIPELINE_PHASE_PREFILL:
            return self.prefill_activation_receive_sequence
        if phase == PIPELINE_PHASE_DECODE:
            return self.decode_activation_receive_sequence
        raise RuntimeError(f"invalid Kimi-K3 activation phase {phase}")

    def send_activation(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        dtype_code: int,
        shape: tuple[int, int, int, int],
        phase: int,
    ) -> None:
        if self.rank != 0:
            raise RuntimeError("only rank zero may send Kimi-K3 activations")
        if self.activation_transport != "tcp":
            raise RuntimeError(
                "byte activation API requires EXO_K3_PIPELINE_ACTIVATION_TRANSPORT=tcp"
            )
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        with self._io_lock:
            sequence = self._activation_send_sequence(phase)
            self._send_frame(
                kind=_PIPELINE_TCP_ACTIVATION,
                phase=phase,
                dtype_code=dtype_code,
                sequence=sequence,
                correlation_sequence=sequence,
                element_count=element_count,
                shape=shape,
                payload=payload,
                max_payload_bytes=self.max_activation_bytes,
            )
            if phase == PIPELINE_PHASE_PREFILL:
                self.prefill_activation_send_sequence += 1
            else:
                self.decode_activation_send_sequence += 1
            self._log_activation_telemetry(
                dtype_code=dtype_code,
                shape=shape,
                logical_bytes=len(payload),
            )

    def receive_activation(
        self,
        *,
        shape: tuple[int, int, int, int],
        phase: int,
    ) -> tuple[int, bytearray]:
        if self.rank != 1:
            raise RuntimeError("only rank one may receive Kimi-K3 activations")
        if self.activation_transport != "tcp":
            raise RuntimeError(
                "byte activation API requires EXO_K3_PIPELINE_ACTIVATION_TRANSPORT=tcp"
            )
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        with self._io_lock:
            sequence = self._activation_receive_sequence(phase)
            payload = self._receive_frame(
                expected_kind=_PIPELINE_TCP_ACTIVATION,
                expected_phase=phase,
                expected_dtype_code=None,
                expected_sequence=sequence,
                expected_correlation_sequence=sequence,
                expected_element_count=element_count,
                expected_shape=shape,
                max_payload_bytes=self.max_activation_bytes,
            )
            if phase == PIPELINE_PHASE_PREFILL:
                self.prefill_activation_receive_sequence += 1
            else:
                self.decode_activation_receive_sequence += 1
            self._log_activation_telemetry(
                dtype_code=payload[0],
                shape=shape,
                logical_bytes=len(payload[1]),
            )
            return payload

    def send_activation_array(
        self,
        activation: mx.array,
        *,
        dtype_code: int,
        shape: tuple[int, int, int, int],
        phase: int,
    ) -> None:
        if self.rank != 0:
            raise RuntimeError("only rank zero may send Kimi-K3 activations")
        if self.activation_transport != "ring":
            raise RuntimeError("MLX activation API requires ring transport")
        if tuple(int(dimension) for dimension in activation.shape) != shape:
            raise RuntimeError(
                f"Kimi-K3 activation array shape {activation.shape} != {shape}"
            )
        if activation.dtype != _pipeline_dtype_from_code(dtype_code):
            raise RuntimeError(
                "Kimi-K3 activation array dtype does not match descriptor: "
                f"{activation.dtype} != {_pipeline_dtype_from_code(dtype_code)}"
            )
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        logical_bytes = element_count * _PIPELINE_DTYPE_ITEMSIZE[dtype_code]
        self._validate_frame_values(
            kind=_PIPELINE_TCP_ACTIVATION,
            phase=phase,
            dtype_code=dtype_code,
            element_count=element_count,
            payload_bytes=logical_bytes,
            shape=shape,
            max_payload_bytes=self.max_activation_bytes,
        )

        # The ring must see a materialized allocation before its exact logical
        # descriptor is published on TCP.
        mx.eval(activation)
        with self._io_lock:
            sequence = self._activation_send_sequence(phase)
            try:
                # The first control-channel connection also initializes the
                # secondary MLX ring.  Resolve the group only after that
                # initialization has completed.
                self.ensure_connected()
                group = self.secondary_ring_group
                if group is None:
                    raise RuntimeError("Kimi-K3 secondary ring group is unavailable")
                self._send_header(
                    kind=_PIPELINE_TCP_ACTIVATION,
                    phase=phase,
                    dtype_code=dtype_code,
                    sequence=sequence,
                    correlation_sequence=sequence,
                    element_count=element_count,
                    shape=shape,
                    payload_bytes=logical_bytes,
                    max_payload_bytes=self.max_activation_bytes,
                )
                _, _, ready_bytes = self._receive_header(
                    expected_kind=_PIPELINE_TCP_ACTIVATION_READY,
                    expected_phase=phase,
                    expected_dtype_code=PIPELINE_DTYPE_NONE,
                    expected_sequence=sequence,
                    expected_correlation_sequence=sequence,
                    expected_element_count=0,
                    expected_shape=(0, 0, 0, 0),
                    max_payload_bytes=0,
                )
                if ready_bytes != 0:
                    raise RuntimeError(
                        "Kimi-K3 activation READY unexpectedly has a payload"
                    )
                dependency = mx.distributed.send(
                    activation,
                    1,
                    group=group,
                    stream=pipeline_send_stream,
                )
                mx.eval(dependency)
                _, _, acknowledged_bytes = self._receive_header(
                    expected_kind=_PIPELINE_TCP_ACTIVATION_ACK,
                    expected_phase=phase,
                    expected_dtype_code=PIPELINE_DTYPE_NONE,
                    expected_sequence=sequence,
                    expected_correlation_sequence=sequence,
                    expected_element_count=0,
                    expected_shape=(0, 0, 0, 0),
                    max_payload_bytes=0,
                )
                if acknowledged_bytes != 0:
                    raise RuntimeError(
                        "Kimi-K3 activation ACK unexpectedly has a payload"
                    )
            except Exception:
                self.abort("secondary ring activation send failed")
                raise

            if phase == PIPELINE_PHASE_PREFILL:
                self.prefill_activation_send_sequence += 1
            else:
                self.decode_activation_send_sequence += 1
            self._log_activation_telemetry(
                dtype_code=dtype_code,
                shape=shape,
                logical_bytes=logical_bytes,
            )

    def receive_activation_array(
        self,
        *,
        shape: tuple[int, int, int, int],
        phase: int,
    ) -> tuple[int, mx.array]:
        if self.rank != 1:
            raise RuntimeError("only rank one may receive Kimi-K3 activations")
        if self.activation_transport != "ring":
            raise RuntimeError("MLX activation API requires ring transport")
        element_count = 1
        for dimension in shape:
            element_count *= dimension

        with self._io_lock:
            sequence = self._activation_receive_sequence(phase)
            try:
                _, dtype_code, logical_bytes = self._receive_header(
                    expected_kind=_PIPELINE_TCP_ACTIVATION,
                    expected_phase=phase,
                    expected_dtype_code=None,
                    expected_sequence=sequence,
                    expected_correlation_sequence=sequence,
                    expected_element_count=element_count,
                    expected_shape=shape,
                    max_payload_bytes=self.max_activation_bytes,
                )
                group = self.secondary_ring_group
                if group is None:
                    raise RuntimeError("Kimi-K3 secondary ring group is unavailable")
                activation = mx.distributed.recv(
                    shape,
                    _pipeline_dtype_from_code(dtype_code),
                    0,
                    group=group,
                    stream=pipeline_receive_stream,
                )
                self._send_header(
                    kind=_PIPELINE_TCP_ACTIVATION_READY,
                    phase=phase,
                    dtype_code=PIPELINE_DTYPE_NONE,
                    sequence=sequence,
                    correlation_sequence=sequence,
                    element_count=0,
                    shape=(0, 0, 0, 0),
                    payload_bytes=0,
                    max_payload_bytes=0,
                )
                mx.eval(activation)
                self._send_header(
                    kind=_PIPELINE_TCP_ACTIVATION_ACK,
                    phase=phase,
                    dtype_code=PIPELINE_DTYPE_NONE,
                    sequence=sequence,
                    correlation_sequence=sequence,
                    element_count=0,
                    shape=(0, 0, 0, 0),
                    payload_bytes=0,
                    max_payload_bytes=0,
                )
            except Exception:
                self.abort("secondary ring activation receive failed")
                raise

            if phase == PIPELINE_PHASE_PREFILL:
                self.prefill_activation_receive_sequence += 1
            else:
                self.decode_activation_receive_sequence += 1
            self._log_activation_telemetry(
                dtype_code=dtype_code,
                shape=shape,
                logical_bytes=logical_bytes,
            )
            return dtype_code, activation

    def send_tokens(self, values: list[int]) -> None:
        if self.rank != 1:
            raise RuntimeError("only rank one may send Kimi-K3 sampled tokens")
        payload = struct.pack(f"!{len(values)}i", *values)
        with self._io_lock:
            sequence = self.token_send_sequence
            if self.decode_activation_receive_sequence == 0:
                raise RuntimeError(
                    "cannot send a Kimi-K3 token before a decode activation"
                )
            correlation_sequence = self.decode_activation_receive_sequence - 1
            self._send_frame(
                kind=_PIPELINE_TCP_TOKEN,
                phase=PIPELINE_PHASE_DECODE,
                dtype_code=PIPELINE_DTYPE_INT32,
                sequence=sequence,
                correlation_sequence=correlation_sequence,
                element_count=len(values),
                shape=(len(values), 0, 0, 0),
                payload=payload,
                max_payload_bytes=_PIPELINE_TCP_MAX_TOKEN_BYTES,
            )
            self.token_send_sequence += 1

    def receive_tokens(self, *, element_count: int) -> tuple[int, ...]:
        if self.rank != 0:
            raise RuntimeError("only rank zero may receive Kimi-K3 sampled tokens")
        with self._io_lock:
            sequence = self.token_receive_sequence
            if self.decode_activation_send_sequence == 0:
                raise RuntimeError(
                    "cannot receive a Kimi-K3 token before a decode activation"
                )
            correlation_sequence = self.decode_activation_send_sequence - 1
            _, payload = self._receive_frame(
                expected_kind=_PIPELINE_TCP_TOKEN,
                expected_phase=PIPELINE_PHASE_DECODE,
                expected_dtype_code=PIPELINE_DTYPE_INT32,
                expected_sequence=sequence,
                expected_correlation_sequence=correlation_sequence,
                expected_element_count=element_count,
                expected_shape=(element_count, 0, 0, 0),
                max_payload_bytes=_PIPELINE_TCP_MAX_TOKEN_BYTES,
            )
            self.token_receive_sequence += 1
        return struct.unpack(f"!{element_count}i", payload)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.listener is not None:
            self.listener.close()
            self.listener = None

    def abort(self, reason: str) -> None:
        self._poisoned = True
        self.close()
        logger.error(
            f"Kimi-K3 pipeline transport aborted; runner restart required: {reason}"
        )


def get_pipeline_tcp_transport(rank: int, world_size: int) -> _PipelineTcpTransport:
    """Return the process-persistent PP2 Kimi-K3 TCP transport."""
    global _pipeline_tcp_transport
    if _pipeline_tcp_transport is None:
        _pipeline_tcp_transport = _PipelineTcpTransport(rank, world_size)
    elif (
        _pipeline_tcp_transport.rank != rank
        or _pipeline_tcp_transport.world_size != world_size
    ):
        raise RuntimeError(
            "Kimi-K3 TCP pipeline transport was initialized for a different rank/world"
        )
    return _pipeline_tcp_transport


def queue_pipeline_tcp_activation(
    transport: _PipelineTcpTransport,
    payload: memoryview,
    *,
    dtype_code: int,
    shape: tuple[int, int, int, int],
    phase: int,
) -> None:
    _pending_tcp_activations.append((transport, payload, dtype_code, shape, phase))


def queue_pipeline_ring_activation(
    transport: _PipelineTcpTransport,
    activation: mx.array,
    *,
    dtype_code: int,
    shape: tuple[int, int, int, int],
    phase: int,
) -> None:
    _pending_ring_activations.append((transport, activation, dtype_code, shape, phase))


def send_pipeline_payloads(
    payloads: PipelineSendPayloads,
    *,
    destination: int,
    group: mx.distributed.Group,
    asynchronous: bool,
) -> mx.array:
    """Send one logical pipeline frame as an ordered array sequence."""
    if not payloads:
        raise ValueError("A pipeline send must contain at least one payload")

    dependency: mx.array | None = None
    for payload in payloads:
        ordered = payload if dependency is None else mx.depends(payload, dependency)
        dependency = mx.distributed.send(
            ordered,
            destination,
            group=group,
            stream=pipeline_send_stream,
        )
    assert dependency is not None
    if asynchronous:
        mx.async_eval(dependency)
    else:
        global _inflight_prefill_send
        mx.eval(dependency)
        # A synchronous send on the same ordered CPU stream also fences every
        # asynchronous prefill send submitted before it.
        _inflight_prefill_send = None
    return dependency


def queue_pipeline_send(
    payloads: PipelineSendPayloads,
    *,
    destination: int,
    group: mx.distributed.Group,
) -> None:
    if not payloads:
        raise ValueError("A queued pipeline send must contain a payload")
    _pending_prefill_sends.append((payloads, destination, group))


def flush_prefill_sends() -> None:
    global _inflight_prefill_send
    for payloads, destination, group in _pending_prefill_sends:
        # All pipeline sends use the same ordered CPU stream. Retaining its
        # latest dependency is therefore sufficient to drain the entire tail,
        # without holding every large prefill activation until the phase ends.
        _inflight_prefill_send = send_pipeline_payloads(
            payloads,
            destination=destination,
            group=group,
            asynchronous=True,
        )
    _pending_prefill_sends.clear()
    for transport, payload, dtype_code, shape, phase in _pending_tcp_activations:
        transport.send_activation(
            payload,
            dtype_code=dtype_code,
            shape=shape,
            phase=phase,
        )
    _pending_tcp_activations.clear()
    for transport, activation, dtype_code, shape, phase in _pending_ring_activations:
        transport.send_activation_array(
            activation,
            dtype_code=dtype_code,
            shape=shape,
            phase=phase,
        )
    _pending_ring_activations.clear()


def drain_prefill_sends() -> None:
    """Wait for every queued prefill send before changing collective phases."""
    global _inflight_prefill_send
    if _inflight_prefill_send is not None:
        mx.eval(_inflight_prefill_send)
        _inflight_prefill_send = None


def register_decode_send(dependency: mx.array) -> None:
    """Retain ordered asynchronous activations until the next token relay.

    MLX-LM can issue more than one one-token model call before its first yield
    (the short prompt tail followed by the sampled step), so this is a queue
    rather than a single slot.
    """
    _inflight_decode_sends.append(dependency)


def advance_decode_sends() -> None:
    """Retire every forward activation except the newest one.

    On the striped JACCL ring a matching receive can finish before the remote
    send reports local completion.  In practice that newest send retires only
    after the receiver posts its next receive.  Keep exactly one such credit
    outstanding while the receiver immediately advances to that next receive.
    This bounds retained buffers without blocking the operation that makes the
    newest send complete.
    """
    if len(_inflight_decode_sends) > 1:
        mx.eval(_inflight_decode_sends[:-1])
        del _inflight_decode_sends[:-1]
        # The first retained decode send is ordered after the queued prefill
        # tail on the same CPU stream, so retiring it also retires that tail.
        drain_prefill_sends()


def clear_prefill_sends() -> None:
    # Unexpected teardown poisons persistent transports before discarding.
    _pending_prefill_sends.clear()
    if _pending_tcp_activations or _pending_ring_activations:
        transports = {
            item[0] for item in (*_pending_tcp_activations, *_pending_ring_activations)
        }
        for transport in transports:
            transport.abort("discarded an unsent prefill activation")
    _pending_tcp_activations.clear()
    _pending_ring_activations.clear()


def discard_unsent_prefill_sends_after_agreed_cancel() -> None:
    """Discard queues after every rank has agreed to cancel this prefill.

    This is intentionally narrower than ``clear_prefill_sends``. It is safe
    only while no peer has posted a receive for these still-unscheduled
    activations. An agreed cancellation leaves the persistent transport and
    its sequence counters reusable; an unexpected exception must use
    ``clear_prefill_sends`` and poison the transport instead.
    """
    _pending_prefill_sends.clear()
    _pending_tcp_activations.clear()
    _pending_ring_activations.clear()


class _LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class CustomMlxLayer(nn.Module):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _LayerCallable):
        super().__init__()
        dict.__setitem__(self, "_original_layer", original_layer)  # pyright: ignore[reportUnknownMemberType]

    @property
    def original_layer(self) -> _LayerCallable:
        return cast(_LayerCallable, self["_original_layer"])

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = cast(_LayerCallable, self["_original_layer"])
                return getattr(original_layer, name)


class PipelineFirstLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.group = group
        self.is_prefill: bool = False
        self.token_relay: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            # We want to avoid GPU timeout errors by evalling the distributed operation
            # so that it stays on CPU, which does not have a timeout.
            mx.eval(x)
            recv_start = time.perf_counter()
            x = mx.distributed.recv_like(
                x,
                (self.r - 1),
                group=self.group,
                stream=pipeline_receive_stream,
            )
            mx.eval(x)
            if not self.is_prefill:
                decode_timings.record_recv(time.perf_counter() - recv_start)
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group
        self.original_layer_signature = signature(self.original_layer.__call__)
        self.is_prefill: bool = False
        self.queue_sends: bool = False
        self.token_relay: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        cache = self.original_layer_signature.bind_partial(
            x, *args, **kwargs
        ).arguments.get("cache", None)

        output: mx.array = self.original_layer(x, *args, **kwargs)

        # Eval layer output to materialize it before send — this splits the graph
        # so the send is isolated and the receiving rank's recv can complete.
        mx.eval(output)

        if self.r != self.s - 1:
            send_start = time.perf_counter()
            if self.queue_sends:
                queue_pipeline_send(
                    (output,),
                    destination=(self.r + 1) % self.s,
                    group=self.group,
                )
            else:
                output = send_pipeline_payloads(
                    (output,),
                    destination=(self.r + 1) % self.s,
                    group=self.group,
                    asynchronous=False,
                )
            if cache is not None:
                # CacheList (used by MLA models like DeepSeekV32, GLM MoE DSA)
                # doesn't have .keys directly; access via first sub-cache.
                _cache = cache[0] if hasattr(cache, "caches") else cache  # type: ignore
                if hasattr(_cache, "keys"):  # pyright: ignore[reportAny]
                    _cache.keys = mx.depends(_cache.keys, output)  # type: ignore
            mx.eval(output)
            if cache is not None and hasattr(_cache, "keys"):  # type: ignore
                mx.eval(_cache.keys)  # type: ignore
            if not self.is_prefill:
                decode_timings.record_send(time.perf_counter() - send_start)

        if not self.is_prefill and not self.token_relay:
            # Legacy lockstep decode: every rank gathers the last stage's
            # activation so all ranks compute identical logits and sample the
            # same token. With token relay enabled the last rank samples alone
            # and circulates only the token id (see relay_sampled_tokens), so
            # this full-hidden-state collective is skipped.
            gather_start = time.perf_counter()
            output = mx.distributed.all_gather(output, group=self.group)[
                -output.shape[0] :
            ]
            mx.eval(output)
            decode_timings.record_gather_and_advance(time.perf_counter() - gather_start)

        return output


def set_pipeline_prefill(model: nn.Module, is_prefill: bool) -> None:
    for layer in model.layers:  # type: ignore
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            layer.is_prefill = is_prefill


def set_pipeline_queue_sends(model: nn.Module, queue_sends: bool) -> None:
    for layer in model.layers:  # type: ignore
        if isinstance(layer, PipelineLastLayer):
            layer.queue_sends = queue_sends


def set_pipeline_token_relay(model: nn.Module, token_relay: bool) -> None:
    """Toggle token-relay decode for a pipeline-parallel model.

    When enabled, decode skips the per-token all_gather of the final hidden
    state: only the last pipeline rank computes final norm / lm_head and
    samples, and the sampled token ids are circulated with
    ``relay_sampled_tokens``. Must only be enabled while every rank agrees to
    call ``relay_sampled_tokens`` after each decode step, otherwise ranks
    deadlock or diverge.
    """
    for layer in model.layers:  # type: ignore
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            layer.token_relay = token_relay


@final
@dataclass(frozen=True)
class PipelineRelayContext:
    """Rank/group facts needed to relay sampled tokens between pipeline ranks."""

    group: mx.distributed.Group
    device_rank: int
    world_size: int
    tcp_transport: _PipelineTcpTransport | None = None

    @property
    def is_last_rank(self) -> bool:
        return self.device_rank == self.world_size - 1


def get_active_relay_context(model: nn.Module) -> PipelineRelayContext | None:
    """Return the relay context when token-relay decode is currently enabled.

    Returns None for non-pipeline models and for pipeline models with the
    legacy all_gather decode active.
    """
    layers = cast(list[object], getattr(model, "layers", []))
    for layer in layers:
        if isinstance(layer, PipelineLastLayer):
            if not layer.token_relay or layer.is_prefill:
                return None
            tcp_transport = (
                get_pipeline_tcp_transport(layer.r, layer.s)
                if bool(getattr(layer, "uses_tcp_pipeline_transport", False))
                else None
            )
            return PipelineRelayContext(
                group=layer.group,
                device_rank=layer.r,
                world_size=layer.s,
                tcp_transport=tcp_transport,
            )
    return None


def relay_sampled_tokens(
    sampled: mx.array, relay_context: PipelineRelayContext
) -> mx.array:
    """Relay sampled token ids backward without changing to a collective."""
    gather_start = time.perf_counter()
    relayed = sampled.astype(mx.int32)
    mx.eval(relayed)
    if relay_context.tcp_transport is not None:
        transport = relay_context.tcp_transport
        flat = relayed.reshape(-1)
        if relay_context.is_last_rank:
            values = flat.tolist()
            if not isinstance(values, list):
                values = [values]
            transport.send_tokens([int(value) for value in values])
        else:
            values = transport.receive_tokens(element_count=int(flat.size))
            relayed = mx.array(values, dtype=mx.int32).reshape(relayed.shape)
    else:
        if not relay_context.is_last_rank:
            relayed = mx.distributed.recv_like(
                relayed,
                relay_context.device_rank + 1,
                group=relay_context.group,
                stream=pipeline_receive_stream,
            )
            mx.eval(relayed)
        if relay_context.device_rank > 0:
            send_pipeline_payloads(
                (relayed,),
                destination=relay_context.device_rank - 1,
                group=relay_context.group,
                asynchronous=False,
            )
        advance_decode_sends()
    decode_timings.record_gather_and_advance(time.perf_counter() - gather_start)
    return relayed


def get_inner_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "language_model", None)
    if isinstance(inner, nn.Module):
        inner_inner = getattr(inner, "model", None)
        if isinstance(inner_inner, nn.Module):
            return inner_inner

    inner = getattr(model, "backbone", None)
    if isinstance(inner, nn.Module):
        return inner

    raise ValueError(
        "Model must either have a 'model', 'transformer', or 'backbone' attribute"
    )


def get_layers(inner_model_instance: nn.Module) -> list[_LayerCallable]:
    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable]
    if hasattr(inner_model_instance, "layers"):
        layers = cast(list[_LayerCallable], inner_model_instance.layers)
    elif hasattr(inner_model_instance, "h"):
        layers = cast(list[_LayerCallable], inner_model_instance.h)
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")

    return layers


def _patch_hybrid_cache(
    model: Qwen3_5TextModel | Qwen3NextModel | NemotronHModel,
    fa_idx: int,
    has_full_attn: bool,
    ssm_idx: int,
    has_linear: bool,
) -> None:
    # Hacks to make make_mask happy.
    original = model.make_cache

    def patched() -> list[ArraysCache | KVCache]:
        cache = original()
        if not has_full_attn:
            entry = cache[fa_idx]
            orig_make_mask = entry.make_mask
            entry.make_mask = lambda n, **_kw: orig_make_mask(n)  # type: ignore
        if not has_linear:
            orig_ssm_make_mask = cache[ssm_idx].make_mask

            def _ssm_mask(
                n: int, **kw: bool | int | None
            ) -> mx.array | Literal["causal"] | None:
                return orig_ssm_make_mask(n, **kw) if kw else None

            cache[ssm_idx].make_mask = _ssm_mask  # type: ignore
        return cache

    model.make_cache = patched


def pipeline_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: PipelineShardMetadata,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    """
    Automatically parallelize a model across multiple devices.
    Args:
    model: The model to parallelize (must have a 'layers' or 'h' property)
    model_shard_meta: The metadata for the model shard
    Returns:
    The parallelized model
    """
    inner_model_instance: nn.Module = get_inner_model(model)

    layers = get_layers(inner_model_instance)

    start_layer, end_layer = model_shard_meta.start_layer, model_shard_meta.end_layer
    device_rank, world_size = model_shard_meta.device_rank, model_shard_meta.world_size

    layers = layers[start_layer:end_layer]
    total = len(layers)
    for i, layer in enumerate(layers):
        mx.eval(layer)  # type: ignore
        mx.clear_cache()
        yield ModelLoadingResponse(layers_loaded=i, total=total)

    is_kimi_k3 = type(model).__module__ == "mlx_lm.models.kimi_k3"
    if is_kimi_k3:
        from exo.worker.engines.mlx.kimi_k3_pipeline import (
            wrap_kimi_k3_pipeline_layers,
        )

        block_size = getattr(
            getattr(inner_model_instance, "args", None),
            "attn_res_block_size",
            None,
        )
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(
                "Kimi K3 pipeline support requires a positive attn_res_block_size"
            )
        layers = wrap_kimi_k3_pipeline_layers(
            layers,
            start_layer=start_layer,
            end_layer=end_layer,
            total_layers=model_shard_meta.n_layers,
            block_size=block_size,
            rank=device_rank,
            world_size=world_size,
            group=group,
        )
    else:
        layers[0] = PipelineFirstLayer(layers[0], device_rank, group=group)
        layers[-1] = PipelineLastLayer(
            layers[-1],
            device_rank,
            world_size,
            group=group,
        )

    if isinstance(inner_model_instance, GptOssMoeModel):
        inner_model_instance.layer_types = inner_model_instance.layer_types[
            start_layer:end_layer
        ]
        # We can assume the model has at least one layer thanks to placement.
        # If a layer type doesn't exist, we can set it to 0.
        inner_model_instance.swa_idx = (
            0
            if "sliding_attention" not in inner_model_instance.layer_types
            else inner_model_instance.layer_types.index("sliding_attention")
        )
        inner_model_instance.ga_idx = (
            0
            if "full_attention" not in inner_model_instance.layer_types
            else inner_model_instance.layer_types.index("full_attention")
        )

    if isinstance(inner_model_instance, Step35InnerModel):
        inner_model_instance.num_layers = len(layers)
        sliding_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_sliding", False)
        ]
        full_layers = [
            i
            for i, layer in enumerate(layers)
            if not getattr(layer, "is_sliding", True)
        ]
        inner_model_instance._swa_idx = 0 if not sliding_layers else sliding_layers[0]
        inner_model_instance._full_idx = 0 if not full_layers else full_layers[0]

    if isinstance(inner_model_instance, (Qwen3_5TextModelInner, Qwen3NextInnerModel)):
        full_attn_layers = [
            i for i, layer in enumerate(layers) if not getattr(layer, "is_linear", True)
        ]
        linear_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_linear", False)
        ]
        inner_model_instance.fa_idx = full_attn_layers[0] if full_attn_layers else 0
        inner_model_instance.ssm_idx = linear_layers[0] if linear_layers else 0
        if not full_attn_layers or not linear_layers:
            _patch_hybrid_cache(
                cast(Qwen3_5TextModel | Qwen3NextModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=bool(full_attn_layers),
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=bool(linear_layers),
            )

    if isinstance(inner_model_instance, NemotronHInnerModel):
        # NemotronH uses block_type: "M" (Mamba/SSM), "*" (Attention), "E" (MoE), "-" (MLP)
        # Only "M" and "*" blocks have cache entries.
        # Recompute fa_idx and ssm_idx as cache-array indices for the shard's layers.
        cache_idx = 0
        fa_idx: int | None = None
        ssm_idx: int | None = None
        for layer in layers:
            block_type = getattr(layer, "block_type", None)
            if block_type == "*":
                if fa_idx is None:
                    fa_idx = cache_idx
                cache_idx += 1
            elif block_type == "M":
                if ssm_idx is None:
                    ssm_idx = cache_idx
                cache_idx += 1
        has_attn = fa_idx is not None
        has_mamba = ssm_idx is not None
        inner_model_instance.fa_idx = fa_idx if fa_idx is not None else 0
        inner_model_instance.ssm_idx = ssm_idx if ssm_idx is not None else 0
        if not has_attn or not has_mamba:
            _patch_hybrid_cache(
                cast(NemotronHModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=has_attn,
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=has_mamba,
            )

    _set_layers(model, layers)
    if is_kimi_k3:
        from exo.worker.engines.mlx.kimi_k3_pipeline import (
            configure_kimi_k3_local_cache_indices,
        )

        configure_kimi_k3_local_cache_indices(inner_model_instance, layers)

    assert isinstance(layers, list), (
        "Expected a list of layers after auto-parallel initialisation"
    )

    return patch_pipeline_model(model, group)


def patch_pipeline_model[T](model: T, group: mx.distributed.Group) -> T:
    # Patch __call__ on the model's class
    cls = model.__class__
    original_call = cls.__call__  # type :ignore
    call_signature = signature(original_call)  # type :ignore

    def patched_call(
        self: T,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        logits: mx.array = original_call(self, *args, **kwargs)  # type: ignore

        relay_context = get_active_relay_context(cast(nn.Module, self))
        if relay_context is not None and not relay_context.is_last_rank:
            # Token-relay decode: this rank's final hidden state is not the
            # last pipeline stage's, so its logits are meaningless. Returning
            # a detached zeros array drops the final norm / lm_head from the
            # lazy graph entirely — they are never computed on this rank.
            logits = mx.zeros_like(logits)

        cache = call_signature.bind_partial(self, *args, **kwargs).arguments.get(
            "cache", None
        )

        # Add dependency to last cache entry to ensure distributed ops are evaluated
        if cache is not None and len(cache) > 0:  # type: ignore
            last = cache[-1]  # type: ignore
            dep_cache = last[0] if hasattr(last, "caches") else last  # type: ignore
            if hasattr(dep_cache, "keys") and dep_cache.keys is not None:  # type: ignore
                dep_cache.keys = mx.depends(dep_cache.keys, logits)  # type: ignore

        return logits

    cls.__call__ = patched_call
    return model


def patch_tensor_model[T](model: T) -> T:
    """Patch model's __call__ to ensure distributed ops sync during inference."""
    cls = model.__class__
    original_call = cls.__call__
    call_signature = signature(original_call)

    def patched_call(
        self: T,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        logits: mx.array = original_call(self, *args, **kwargs)  # pyright: ignore[reportAny]
        cache = call_signature.bind_partial(self, *args, **kwargs).arguments.get(
            "cache", None
        )

        # Add dependency to last cache entry to ensure distributed ops are evaluated
        if cache is not None and len(cache) > 0:  # pyright: ignore[reportAny]
            last = cache[-1]  # pyright: ignore[reportAny]
            dep_cache = last[0] if hasattr(last, "caches") else last  # pyright: ignore[reportAny]
            if hasattr(dep_cache, "keys"):  # type: ignore
                dep_cache.keys = mx.depends(dep_cache.keys, logits)  # pyright: ignore[reportAny]

        return logits

    cls.__call__ = patched_call
    return model


def tensor_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    all_to_sharded_linear = partial(
        shard_linear,
        sharding="all-to-sharded",
        group=group,
    )
    sharded_to_all_linear = partial(
        shard_linear,
        sharding="sharded-to-all",
        group=group,
    )

    segments: int = 1

    def _all_to_sharded(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - all to sharded")
            return weight.ndim - 1, segments
        return max(weight.ndim - 2, 0), segments

    all_to_sharded_linear_in_place = partial(
        shard_inplace,
        sharding=_all_to_sharded,  # type: ignore
        group=group,
    )

    n = group.size()

    def _sharded_to_all(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - sharded to all")
            weight /= n
            return None
        return -1, segments

    sharded_to_all_linear_in_place = partial(
        shard_inplace,
        sharding=_sharded_to_all,  # type: ignore
        group=group,
    )

    if isinstance(model, (LlamaModel, Ministral3Model)):
        tensor_parallel_sharding_strategy = LlamaShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, DeepseekV4Model):
        tensor_parallel_sharding_strategy = DeepseekV4ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, MiniMaxModel):
        tensor_parallel_sharding_strategy = MiniMaxShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GLM4MoeLiteModel):
        tensor_parallel_sharding_strategy = GLM4MoeLiteShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Glm4MoeModel):
        tensor_parallel_sharding_strategy = Glm4MoeShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(
        model,
        (
            Qwen3Model,
            Qwen3MoeModel,
            Qwen3NextModel,
            Qwen3_5TextModel,
            Qwen3_5MoeModel,
            Qwen3VLModel,
        ),
    ):
        tensor_parallel_sharding_strategy = QwenShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GptOssModel):
        tensor_parallel_sharding_strategy = GptOssShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Step35Model):
        tensor_parallel_sharding_strategy = Step35ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, NemotronHModel):
        tensor_parallel_sharding_strategy = NemotronHShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Gemma4Model):
        tensor_parallel_sharding_strategy = Gemma4ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    model = yield from tensor_parallel_sharding_strategy.shard_model(model)
    return patch_tensor_model(model)


class TensorParallelShardingStrategy(ABC):
    def __init__(
        self,
        group: mx.distributed.Group,
        all_to_sharded_linear: Callable[..., nn.Linear],
        sharded_to_all_linear: Callable[..., nn.Linear],
        all_to_sharded_linear_in_place: Callable[..., None],
        sharded_to_all_linear_in_place: Callable[..., None],
    ):
        self.all_to_sharded_linear = all_to_sharded_linear
        self.sharded_to_all_linear = sharded_to_all_linear
        self.all_to_sharded_linear_in_place = all_to_sharded_linear_in_place
        self.sharded_to_all_linear_in_place = sharded_to_all_linear_in_place
        self.group = group
        self.N = group.size()

    @abstractmethod
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]: ...


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(LlamaModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            # Force load weights before sharding to avoid FAST_SYNCH deadlock
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            if layer.self_attn.n_kv_heads is not None:
                layer.self_attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
            mx.eval(layer)

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


def _set_layers(model: nn.Module, layers: list[_LayerCallable]) -> None:
    inner_model_instance = get_inner_model(model)
    if hasattr(inner_model_instance, "layers"):
        inner_model_instance.layers = layers

        # Update DeepSeek V3 specific parameters when layers are shrunk
        if isinstance(
            model,
            (
                DeepseekV3Model,
                DeepseekV32Model,
                DeepseekV4Model,
                Glm4MoeModel,
                KimiK25Model,
            ),
        ) and hasattr(inner_model_instance, "num_layers"):
            logger.info(
                f"Setting num_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.start_idx = 0
            inner_model_instance.end_idx = len(layers)
            inner_model_instance.num_layers = len(layers)
        elif isinstance(model, Qwen3MoeModel):
            logger.info(
                f"Setting num_hidden_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.num_hidden_layers = len(layers)
    elif hasattr(inner_model_instance, "h"):
        inner_model_instance.h = layers
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")


class DeepSeekShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(DeepseekV3Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # Shard the self attention
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )

            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            # Logic from upstream mlx
            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            # Shard the MLP
            if isinstance(layer.mlp, (DeepseekV3MLP, DeepseekV32MLP)):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            # Shard the MoE.
            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

            mx.eval(layer)

            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


class ShardedMoE(CustomMlxLayer):
    """Wraps any MoE layer with distributed sum_gradients / all_sum."""

    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class ShardedMoEV4(CustomMlxLayer):
    """Same as ShardedMoE but for DeepseekV4MoE which takes (x, input_ids)."""

    def __init__(self, layer: DeepseekV4MoE):
        super().__init__(cast(_LayerCallable, cast(object, layer)))
        self._v4_inner = layer
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self._v4_inner(x, input_ids)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


def _shard_quantized_rows(
    q: nn.QuantizedLinear,
    head_dim: int,
    slicer: Callable[[mx.array, int], mx.array],
) -> None:
    weight = q["weight"]
    scales = q["scales"]
    assert isinstance(weight, mx.array)
    assert isinstance(scales, mx.array)
    q.weight = slicer(weight, head_dim)
    q.scales = slicer(scales, head_dim)
    biases = q.get("biases")
    if isinstance(biases, mx.array):
        q.biases = slicer(biases, head_dim)


class _AllSumLinear(nn.Module):
    """Wraps an unsharded wo_b that takes a head-sharded partial wo_a output.

    Flow per rank:
      1. all_sum the incoming partial wo_a output (summed across the head
         input shards → full wo_a_out on every rank)
      2. apply the unsharded wo_b → full hidden on every rank

    One collective per layer on the smaller of (n_groups * o_lora_rank) vs
    hidden. wo_b compute is replicated, but at decode B=1 it's only ~30M FLOPs
    per layer and 61 extra all_gathers/token cost more than running wo_b on
    every rank.
    """

    def __init__(self, inner: nn.Module, group: mx.distributed.Group):
        super().__init__()
        self.inner = inner
        self._group = group

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.distributed.all_sum(x, group=self._group)
        return cast(Callable[[mx.array], mx.array], self.inner)(x)


def _shard_v4_attention_heads(
    attn: V4Attention,
    world_size: int,
    rank: int,
) -> None:
    """Interleaved-per-group head sharding for V4Attention.

    V4 uses a grouped low-rank output projection: `_grouped_output_projection`
    reshapes the flat `n_heads * head_dim` dim into `(o_groups, heads_per_group,
    head_dim)`, so group g owns heads `[g * heads_per_group : (g+1) * heads_per_group]`.

    A naive contiguous `shard_linear("all-to-sharded")` on wq_b puts whole
    original groups on each rank — the per-rank "group g" ends up containing
    heads that don't belong to original group g. That breaks the wo_a grouped
    weight mapping. We instead slice heads interleaved-by-group: each rank
    owns `heads_per_group / N` heads *from every original group*, kept in
    group-major order so SDPA → reshape → wo_a preserves the group mapping.

    Affects `wq_b.weight` / `wq_b.bias`, `attn_sink`. wo_a is sharded via a
    normal input-dim block split (the default axis-(-1) behavior of
    shard_inplace), which now correctly aligns with the interleaved head
    layout because the last dim of out after reshape is `heads_per_group/N *
    head_dim` per group.
    """
    n_heads: int = attn.n_heads
    head_dim: int = attn.head_dim
    o_groups: int = attn.n_groups
    assert n_heads % o_groups == 0, "n_heads must be divisible by o_groups"
    heads_per_group = n_heads // o_groups
    assert heads_per_group % world_size == 0, (
        f"heads_per_group ({heads_per_group}) must be divisible by world_size "
        f"({world_size}) for interleaved per-group head sharding"
    )
    hpg_per_rank = heads_per_group // world_size
    start = rank * hpg_per_rank
    end = start + hpg_per_rank

    def _slice_head_major_flat(arr: mx.array, stride: int) -> mx.array:
        """Slice arr on axis 0 where the flat 0-axis is (o_groups *
        heads_per_group * stride), returning a fresh contiguous allocation
        so the full unsharded array can be freed. Without the contiguous
        copy the slice is a view and the original weight stays resident —
        OOM on large V4. Quantized packed weights don't round-trip through
        numpy so we use mx.contiguous directly."""
        rest = arr.shape[1:]
        reshaped = arr.reshape(o_groups, heads_per_group, stride, *rest)
        sliced = reshaped[:, start:end].reshape(o_groups * hpg_per_rank * stride, *rest)
        detached = mx.contiguous(sliced)
        mx.eval(detached)
        return detached

    wq_b: nn.Module = attn.wq_b
    if isinstance(wq_b, nn.QuantizedLinear):
        # Packed weight: (n_heads*head_dim, q_lora_rank/el_per_int).
        # scales/biases: (n_heads*head_dim, q_lora_rank/group_size).
        # Slice axis 0 interleaved-by-group with head_dim stride.
        _shard_quantized_rows(wq_b, head_dim, _slice_head_major_flat)
    else:
        dense = wq_b
        assert isinstance(dense, nn.Linear)
        w = dense.weight
        q_lora_rank = w.shape[-1]
        w_sharded = _slice_head_major_flat(w, head_dim)
        has_bias = "bias" in dense
        new_wq_b = nn.Linear(q_lora_rank, w_sharded.shape[0], bias=has_bias)
        new_wq_b.weight = w_sharded
        if has_bias:
            b = dense.bias
            assert b is not None
            new_wq_b.bias = _slice_head_major_flat(b[:, None], head_dim).reshape(-1)
        attn.wq_b = new_wq_b

    sink = attn.attn_sink
    reshaped = sink.reshape(o_groups, heads_per_group)[:, start:end].reshape(-1)
    detached_sink = mx.contiguous(reshaped)
    mx.eval(detached_sink)
    attn.attn_sink = detached_sink
    attn.n_heads = o_groups * hpg_per_rank


class DeepseekV4ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(DeepseekV4Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # Head-parallel attention with interleaved-per-group sharding.
            _shard_v4_attention_heads(layer.attn, self.N, self.group.rank())
            self.sharded_to_all_linear_in_place(layer.attn.wo_a)
            layer.attn.wo_b = _AllSumLinear(layer.attn.wo_b, self.group)  # type: ignore

            ffn = layer.ffn
            if getattr(ffn, "shared_experts", None) is not None:
                self.all_to_sharded_linear_in_place(ffn.shared_experts.gate_proj)
                self.sharded_to_all_linear_in_place(ffn.shared_experts.down_proj)
                self.all_to_sharded_linear_in_place(ffn.shared_experts.up_proj)
            self.all_to_sharded_linear_in_place(ffn.switch_mlp.gate_proj)
            self.sharded_to_all_linear_in_place(ffn.switch_mlp.down_proj)
            self.all_to_sharded_linear_in_place(ffn.switch_mlp.up_proj)
            wrapped = ShardedMoEV4(ffn)
            wrapped.sharding_group = self.group
            layer.ffn = wrapped  # type: ignore

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


class GLM4MoeLiteShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(GLM4MoeLiteModel, model)
        total = len(model.layers)  # type: ignore
        for i, layer in enumerate(model.layers):  # type: ignore
            layer = cast(Glm4MoeLiteDecoderLayer, layer)
            mx.eval(layer.parameters())
            if layer.self_attn.q_lora_rank is None:  # type: ignore
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )

            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            # Logic from upstream mlx
            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, Glm4MoeLiteMLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group  # type: ignore
            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


class WrappedMiniMaxAttention(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable, group: mx.distributed.Group):
        super().__init__(layer)
        self.group = group

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: "Cache | None" = None,
    ) -> mx.array:
        batch_dim, seq_dim, _ = x.shape

        self._original_layer = cast(MiniMaxAttention, self.original_layer)  # type: ignore

        queries: mx.array = self._original_layer.q_proj(x)
        keys: mx.array = self._original_layer.k_proj(x)
        values: mx.array = self._original_layer.v_proj(x)

        if getattr(self, "use_qk_norm", False):
            q_dim = queries.shape[-1]
            k_dim = keys.shape[-1]
            n = self.group.size()

            qk = mx.concatenate(
                [queries, keys], axis=-1
            )  # (batch_dim, seq_dim, q_dim + k_dim)
            qk = mx.distributed.all_gather(
                qk, group=self.group
            )  # (n*batch_dim, seq_dim, q_dim + k_dim)

            qk = qk.reshape(n, batch_dim, seq_dim, q_dim + k_dim).transpose(1, 2, 0, 3)
            queries = qk[..., :q_dim].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * q_dim)
            keys = qk[..., q_dim:].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * k_dim)

            queries = self._original_layer.q_norm(queries)
            keys = self._original_layer.k_norm(keys)

            # Split back and take this rank's portion
            queries = mx.split(queries, n, axis=-1)[self.group.rank()]
            keys = mx.split(keys, n, axis=-1)[self.group.rank()]

        queries = queries.reshape(
            batch_dim, seq_dim, self._original_layer.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self._original_layer.rope(queries, offset=cache.offset)
            keys = self._original_layer.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self._original_layer.rope(queries)
            keys = self._original_layer.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self._original_layer.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_dim, seq_dim, -1)

        return self._original_layer.o_proj(output)


class MiniMaxShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(MiniMaxModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N

            layer.self_attn = WrappedMiniMaxAttention(layer.self_attn, self.group)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]

            # Shard the MoE.
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.gate_proj
            )
            self.sharded_to_all_linear_in_place(
                layer.block_sparse_moe.switch_mlp.down_proj
            )
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.up_proj
            )
            layer.block_sparse_moe = ShardedMoE(layer.block_sparse_moe)  # type: ignore
            layer.block_sparse_moe.sharding_group = self.group
            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class QwenShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(
            Qwen3Model
            | Qwen3MoeModel
            | Qwen3NextModel
            | Qwen3_5TextModel
            | Qwen3_5MoeModel
            | Qwen3VLModel,
            model,
        )
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            if isinstance(layer, (Qwen3MoeDecoderLayer, Qwen3TransformerBlock)):
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj
                )
                layer.self_attn.n_heads //= self.N
                layer.self_attn.n_kv_heads //= self.N
            else:
                assert isinstance(layer, (Qwen3NextDecoderLayer, Qwen3_5DecoderLayer))
                if hasattr(layer, "linear_attn"):
                    linear_attn = layer.linear_attn

                    if isinstance(linear_attn, Qwen3NextGatedDeltaNet):
                        # Qwen3-Next: combined projections
                        linear_attn.in_proj_qkvz = self.all_to_sharded_linear(
                            linear_attn.in_proj_qkvz
                        )
                        linear_attn.in_proj_ba = self.all_to_sharded_linear(
                            linear_attn.in_proj_ba
                        )
                    else:
                        # Qwen3.5: separate projections
                        # in_proj_qkv has sections [q(key_dim), k(key_dim), v(value_dim)]
                        # that must be split section-aware, not as a contiguous block
                        key_dim = linear_attn.key_dim
                        value_dim = linear_attn.value_dim
                        linear_attn.in_proj_qkv = shard_linear(
                            linear_attn.in_proj_qkv,
                            "all-to-sharded",
                            segments=[key_dim, key_dim + key_dim],
                            group=self.group,
                        )
                        linear_attn.in_proj_z = self.all_to_sharded_linear(
                            linear_attn.in_proj_z
                        )
                        linear_attn.in_proj_b = self.all_to_sharded_linear(
                            linear_attn.in_proj_b
                        )
                        linear_attn.in_proj_a = self.all_to_sharded_linear(
                            linear_attn.in_proj_a
                        )
                    linear_attn.out_proj = self.sharded_to_all_linear(
                        linear_attn.out_proj
                    )

                    # Shard conv1d: depthwise conv with non-contiguous channel slicing.
                    # Channel layout is [q(key_dim), k(key_dim), v(value_dim)].
                    # Each rank takes its head-slice from each of the three sections.
                    rank = self.group.rank()
                    key_dim = linear_attn.key_dim
                    value_dim = linear_attn.value_dim
                    key_dim_shard = key_dim // self.N
                    value_dim_shard = value_dim // self.N

                    q_idx = mx.arange(rank * key_dim_shard, (rank + 1) * key_dim_shard)
                    k_idx = mx.arange(
                        key_dim + rank * key_dim_shard,
                        key_dim + (rank + 1) * key_dim_shard,
                    )
                    v_idx = mx.arange(
                        2 * key_dim + rank * value_dim_shard,
                        2 * key_dim + (rank + 1) * value_dim_shard,
                    )
                    conv_indices = mx.concatenate([q_idx, k_idx, v_idx])
                    linear_attn.conv1d.weight = linear_attn.conv1d.weight[conv_indices]
                    new_conv_dim = key_dim_shard * 2 + value_dim_shard
                    linear_attn.conv1d.groups = new_conv_dim

                    num_v_shard = linear_attn.num_v_heads // self.N
                    v_start = rank * num_v_shard
                    v_end = v_start + num_v_shard
                    linear_attn.A_log = linear_attn.A_log[v_start:v_end]
                    linear_attn.dt_bias = linear_attn.dt_bias[v_start:v_end]

                    linear_attn.num_k_heads //= self.N
                    linear_attn.num_v_heads //= self.N
                    linear_attn.key_dim = (
                        linear_attn.head_k_dim * linear_attn.num_k_heads
                    )
                    linear_attn.value_dim = (
                        linear_attn.head_v_dim * linear_attn.num_v_heads
                    )
                    linear_attn.conv_dim = (
                        linear_attn.key_dim * 2 + linear_attn.value_dim
                    )
                else:
                    layer.self_attn.q_proj = self.all_to_sharded_linear(
                        layer.self_attn.q_proj
                    )
                    layer.self_attn.k_proj = self.all_to_sharded_linear(
                        layer.self_attn.k_proj
                    )
                    layer.self_attn.v_proj = self.all_to_sharded_linear(
                        layer.self_attn.v_proj
                    )
                    layer.self_attn.o_proj = self.sharded_to_all_linear(
                        layer.self_attn.o_proj
                    )
                    layer.self_attn.num_attention_heads //= self.N
                    layer.self_attn.num_key_value_heads //= self.N

            # Shard the MoE.
            if isinstance(
                layer.mlp,
                (
                    Qwen3MoeSparseMoeBlock,
                    Qwen3NextSparseMoeBlock,
                    Qwen3_5SparseMoeBlock,
                ),
            ):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                if isinstance(
                    layer.mlp, (Qwen3NextSparseMoeBlock, Qwen3_5SparseMoeBlock)
                ):
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_expert.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_expert.down_proj
                    )
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_expert.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class Glm4MoeShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Glm4MoeModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            layer.self_attn.n_kv_heads //= self.N

            if isinstance(layer.mlp, MoE):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class GptOssShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(GptOssMoeModel, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads
                // layer.self_attn.num_key_value_heads
            )

            layer.self_attn.sinks = layer.self_attn.sinks[
                layer.self_attn.num_attention_heads
                * self.group.rank() : layer.self_attn.num_attention_heads
                * (self.group.rank() + 1)
            ]

            self.all_to_sharded_linear_in_place(layer.mlp.experts.gate_proj)
            self.sharded_to_all_linear_in_place(layer.mlp.experts.down_proj)
            self.all_to_sharded_linear_in_place(layer.mlp.experts.up_proj)

            layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
            layer.mlp.sharding_group = self.group
            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class Step35ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Step35Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_heads //= self.N
            layer.self_attn.num_kv_heads //= self.N

            if getattr(layer.self_attn, "use_head_wise_attn_gate", False):
                layer.self_attn.g_proj = self.all_to_sharded_linear(
                    layer.self_attn.g_proj
                )

            if isinstance(layer.mlp, Step35MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            else:
                layer.mlp.sharding_group = self.group
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.up_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.share_expert.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class NemotronHShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(NemotronHModel, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            mixer = layer.mixer

            if isinstance(mixer, NemotronHAttention):
                mixer.q_proj = self.all_to_sharded_linear(mixer.q_proj)
                mixer.k_proj = self.all_to_sharded_linear(mixer.k_proj)
                mixer.v_proj = self.all_to_sharded_linear(mixer.v_proj)
                mixer.o_proj = self.sharded_to_all_linear(mixer.o_proj)
                mixer.num_heads //= self.N
                mixer.num_key_value_heads //= self.N

            elif isinstance(mixer, NemotronHMamba2Mixer):
                self._shard_mamba2_mixer(mixer, rank)

            elif isinstance(mixer, NemotronHMoE):
                # Shard routed experts (SwitchMLP uses fc1/fc2)
                self.all_to_sharded_linear_in_place(mixer.switch_mlp.fc1)
                self.sharded_to_all_linear_in_place(mixer.switch_mlp.fc2)
                # Shard shared expert in-place (no all-reduce — ShardedMoE handles that)
                if hasattr(mixer, "shared_experts"):
                    self.all_to_sharded_linear_in_place(mixer.shared_experts.up_proj)
                    self.sharded_to_all_linear_in_place(mixer.shared_experts.down_proj)
                mixer = ShardedMoE(mixer)  # pyright: ignore[reportArgumentType]
                mixer.sharding_group = self.group
                layer.mixer = mixer  # pyright: ignore[reportAttributeAccessIssue]

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model

    def _shard_mamba2_mixer(self, mixer: NemotronHMamba2Mixer, rank: int) -> None:
        """Shard the Mamba2 mixer along the head dimension."""
        world_size = self.N
        num_heads = mixer.num_heads
        head_dim = mixer.head_dim
        n_groups = mixer.n_groups
        ssm_state_size = mixer.ssm_state_size
        intermediate_size = mixer.intermediate_size  # = num_heads * head_dim

        # Per-rank sizes
        heads_per_rank = num_heads // world_size
        groups_per_rank = n_groups // world_size
        is_per_rank = heads_per_rank * head_dim
        bc_per_rank = groups_per_rank * ssm_state_size

        # === in_proj: output layout is [gate:IS | conv_ssm:IS | B:NG*SS | C:NG*SS | dt:NH] ===
        gate_start = 0
        conv_ssm_start = intermediate_size
        b_start = 2 * intermediate_size
        c_start = b_start + n_groups * ssm_state_size
        dt_start = c_start + n_groups * ssm_state_size

        # Build index tensor for this rank's slice of each section
        gate_idx = mx.arange(
            gate_start + rank * is_per_rank, gate_start + (rank + 1) * is_per_rank
        )
        conv_ssm_idx = mx.arange(
            conv_ssm_start + rank * is_per_rank,
            conv_ssm_start + (rank + 1) * is_per_rank,
        )
        b_idx = mx.arange(
            b_start + rank * bc_per_rank, b_start + (rank + 1) * bc_per_rank
        )
        c_idx = mx.arange(
            c_start + rank * bc_per_rank, c_start + (rank + 1) * bc_per_rank
        )
        dt_idx = mx.arange(
            dt_start + rank * heads_per_rank, dt_start + (rank + 1) * heads_per_rank
        )

        indices = mx.concatenate([gate_idx, conv_ssm_idx, b_idx, c_idx, dt_idx])
        mixer.in_proj.weight = mixer.in_proj.weight[indices]

        # === out_proj: input is intermediate_size (sharded) → hidden_size (reduce) ===
        mixer.out_proj = self.sharded_to_all_linear(mixer.out_proj)

        # === conv1d: depthwise conv on conv_dim channels ===
        # conv_dim layout: [ssm_hidden:IS | B:NG*SS | C:NG*SS]
        conv_ssm_idx_local = mx.arange(rank * is_per_rank, (rank + 1) * is_per_rank)
        conv_b_idx = mx.arange(
            intermediate_size + rank * bc_per_rank,
            intermediate_size + (rank + 1) * bc_per_rank,
        )
        conv_c_idx = mx.arange(
            intermediate_size + n_groups * ssm_state_size + rank * bc_per_rank,
            intermediate_size + n_groups * ssm_state_size + (rank + 1) * bc_per_rank,
        )
        conv_indices = mx.concatenate([conv_ssm_idx_local, conv_b_idx, conv_c_idx])
        mixer.conv1d.weight = mixer.conv1d.weight[conv_indices]
        new_conv_dim = is_per_rank + 2 * bc_per_rank
        mixer.conv1d.groups = new_conv_dim
        if mixer.conv1d.bias is not None:
            mixer.conv1d.bias = mixer.conv1d.bias[conv_indices]

        # === Per-head parameters ===
        h_start = rank * heads_per_rank
        h_end = h_start + heads_per_rank
        mixer.dt_bias = mixer.dt_bias[h_start:h_end]
        mixer.A_log = mixer.A_log[h_start:h_end]
        mixer.D = mixer.D[h_start:h_end]

        # === Norm: weight is intermediate_size ===
        mixer.norm.weight = mixer.norm.weight[
            rank * is_per_rank : (rank + 1) * is_per_rank
        ]

        # === Update dimensions ===
        mixer.num_heads = heads_per_rank
        mixer.n_groups = groups_per_rank
        mixer.intermediate_size = is_per_rank
        mixer.conv_dim = new_conv_dim
        mixer.heads_per_group = heads_per_rank // groups_per_rank


class WrappedGemma4Experts(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y: mx.array = self.original_layer(x, top_k_indices, top_k_weights)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class Gemma4ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Gemma4Model, model)
        layers = model.language_model.model.layers
        total = len(layers)
        for i, layer in enumerate(layers):
            mx.eval(layer.parameters())

            attn = layer.self_attn
            attn.q_proj = self.all_to_sharded_linear(attn.q_proj)
            has_kv: bool = cast(bool, attn.has_kv)
            if has_kv:
                attn.k_proj = self.all_to_sharded_linear(attn.k_proj)
                if not attn.use_k_eq_v:
                    attn.v_proj = self.all_to_sharded_linear(attn.v_proj)
            attn.o_proj = self.sharded_to_all_linear(attn.o_proj)
            attn.n_heads //= self.N
            attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            if layer.enable_moe:
                self.all_to_sharded_linear_in_place(layer.experts.switch_glu.gate_proj)
                self.sharded_to_all_linear_in_place(layer.experts.switch_glu.down_proj)
                self.all_to_sharded_linear_in_place(layer.experts.switch_glu.up_proj)
                layer.experts = WrappedGemma4Experts(layer.experts)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]
                layer.experts.sharding_group = self.group

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model
