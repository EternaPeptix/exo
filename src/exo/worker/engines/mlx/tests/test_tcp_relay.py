"""Tests for the CUDA<->CUDA TcpRelay wire protocol (dtype-aware serialization).

These exercise the parts of TcpRelay that are independent of CUDA: the array
serialization round-trip over real loopback TCP sockets. They must pass on the
Macs (no CUDA) because the relay's whole point is to be a CUDA-agnostic
transport once a CUDA<->CUDA pair is detected. GLM-5.2 is bf16, so the bf16
round-trip is the most important case.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest

# TcpRelay reads env vars at construction; set a 2-host loopback layout.
os.environ.setdefault("MLX_HOSTS_JSON", '[{"ip": "127.0.0.1"}, {"ip": "127.0.0.1"}]')
os.environ.setdefault("MLX_TCPRELAY_PORT", "0")  # 0 => OS-assigned ephemeral per rank? No: port = base + rank.
# Use a high base unlikely to collide; each test pins a unique base via monkeypatch.

from exo.worker.engines.mlx.auto_parallel import TcpRelay  # noqa: E402


def _free_port_base() -> int:
    """Reserve two adjacent OS-ephemeral ports (rank 0 and rank 1) and return
    the base. Prevents cross-test collisions that hash-based assignment caused.
    """
    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s1.bind(("127.0.0.1", 0))
    p1 = s1.getsockname()[1]
    # Ensure p1+1 is also free (for rank 1). If not, retry.
    while True:
        try:
            s2.bind(("127.0.0.1", p1 + 1))
            break
        except OSError:
            s1.close(); s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s1.bind(("127.0.0.1", 0)); p1 = s1.getsockname()[1]
    s1.close(); s2.close()
    return p1


def _make_relay(rank: int, port_base: int, monkeypatch) -> TcpRelay:
    """Construct a TcpRelay for `rank` using a unique port base (collision-free)."""
    monkeypatch.setenv("MLX_HOSTS_JSON", '[{"ip": "127.0.0.1"}, {"ip": "127.0.0.1"}]')
    monkeypatch.setenv("MLX_TCPRELAY_PORT", str(port_base))
    monkeypatch.setenv("MLX_RANK", str(rank))
    monkeypatch.setenv("MLX_CUDA_RANKS", "")  # don't gate on cuda detection
    return TcpRelay()


@pytest.mark.parametrize(
    "dtype, values",
    [
        (mx.bfloat16, [1.5, -2.25, 3.125, 0.0, 7.75]),   # GLM-5.2 dtype — the critical case
        (mx.float32, [1.5, -2.25, 3.125, 0.0, 7.75]),
        (mx.float16, [1.5, -2.25, 3.0, 0.0, 7.5]),
        (mx.int32, [1, -2, 3, 0, 7]),
        (mx.int64, [1, -2, 3, 0, 7]),
        (mx.uint8, [0, 1, 2, 3, 255]),
        (mx.bool_, [True, False, True, True, False]),
    ],
)
def test_send_recv_roundtrip_loopback(dtype, values, monkeypatch):
    """send() on rank 0 must round-trip to an identical mlx array on rank 1."""
    # Each test gets a unique port base to avoid collisions.
    port_base = _free_port_base()
    sender = _make_relay(rank=0, port_base=port_base, monkeypatch=monkeypatch)
    receiver = _make_relay(rank=1, port_base=port_base, monkeypatch=monkeypatch)

    original = mx.array(values, dtype=dtype)

    received_box: dict[str, object] = {}
    exc_box: dict[str, BaseException] = {}

    def receiver_thread():
        try:
            # recv from rank 0 (the sender)
            got = receiver.recv_like(original, src=0)
            received_box["arr"] = got
        except BaseException as e:
            exc_box["err"] = e

    t = threading.Thread(target=receiver_thread)
    t.start()
    # give the receiver's server socket time to listen
    time.sleep(0.3)
    sender.send(original, dst=1)
    t.join(timeout=10)

    if "err" in exc_box:
        pytest.fail(f"receiver raised: {exc_box['err']!r}")
    got = received_box.get("arr")
    assert got is not None, "receiver produced no array (timeout?)"

    got = mx.array(got)
    assert got.dtype == original.dtype, f"dtype changed: {original.dtype} -> {got.dtype}"
    assert got.shape == original.shape, f"shape changed: {original.shape} -> {got.shape}"
    assert mx.array_equal(original, got).item(), (
        f"values differ: original={original.tolist()} got={got.tolist()}"
    )
    sender.close()
    receiver.close()


def test_send_recv_multidim_shape(monkeypatch):
    """A 2-D bf16 array (realistic prefill activation shape) round-trips."""
    port_base = _free_port_base()
    sender = _make_relay(rank=0, port_base=port_base, monkeypatch=monkeypatch)
    receiver = _make_relay(rank=1, port_base=port_base, monkeypatch=monkeypatch)
    original = mx.random.uniform(shape=(4, 6144)).astype(mx.bfloat16)  # hidden_size of GLM-5.2

    received = {}
    def rt():
        received["arr"] = receiver.recv_like(original, src=0)
    t = threading.Thread(target=rt); t.start()
    time.sleep(0.3)
    sender.send(original, dst=1)
    t.join(timeout=10)
    got = mx.array(received["arr"])
    assert got.dtype == mx.bfloat16
    assert got.shape == (4, 6144)
    # bf16 round-trip must be exact (we transport as f32, cast back)
    assert mx.array_equal(original, got).item()
    sender.close(); receiver.close()


def test_recv_exact_handles_partial_reads(monkeypatch):
    """_recv_exact must loop until all n bytes arrive, even with tiny recvs."""
    port_base = _free_port_base()
    relay = _make_relay(rank=0, port_base=port_base, monkeypatch=monkeypatch)

    # A fake socket whose recv() returns 1 byte at a time.
    class TrickleSock:
        def __init__(self, data: bytes):
            self._data = data
        def recv(self, n):
            if not self._data:
                raise ConnectionError("closed")
            one = self._data[:1]
            self._data = self._data[1:]
            return one

    payload = bytes(range(50))
    out = relay._recv_exact(TrickleSock(payload), 50)
    assert out == payload
    relay.close()


def test_dtype_map_does_not_reference_missing_numpy_bfloat16():
    """Constructing TcpRelay must not crash on numpy without np.bfloat16.

    numpy >= 2.0 removed/never-had np.bfloat16 in many builds; the relay's old
    DTYPE_TO_CODE dict referenced it directly and crashed at __init__.
    """
    assert not hasattr(np, "bfloat16"), "this test assumes numpy lacks np.bfloat16"
    # If construction completes without AttributeError, the dict is safe.
    os.environ["MLX_HOSTS_JSON"] = '[{"ip":"127.0.0.1"},{"ip":"127.0.0.1"}]'
    os.environ["MLX_TCPRELAY_PORT"] = "43250"
    os.environ["MLX_RANK"] = "0"
    os.environ["MLX_CUDA_RANKS"] = ""
    relay = TcpRelay()  # must not raise
    relay.close()
