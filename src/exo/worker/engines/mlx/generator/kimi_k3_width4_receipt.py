"""Fail-closed terminal receipts for the Kimi K3 width-four candidate."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

WIDTH4_RECEIPT_LOG_ENV = "EXO_MLX_KIMI_K3_WIDTH4_DISPATCH_RECEIPT_LOG"
MLX_LM_WIDTH4_RECEIPT_ENV = "MLX_LM_KIMI_K3_WIDTH4_DISPATCH_RECEIPT"
MLX_Q4_RECEIPT_ENV = "MLX_METAL_K3_AFFINE6_Q4_DISPATCH_RECEIPT"
RECEIPT_MARKER = "K3_WIDTH4_DISPATCH_RECEIPT "
EXPECTED_LIBMLX_SHA256_ENV = "EXO_MLX_KIMI_K3_WIDTH4_LIBMLX_SHA256"


class _MlxModule(Protocol):
    @property
    def __file__(self) -> str | None: ...


class _NativeCounterGetter(Protocol):
    argtypes: list[object]
    restype: object

    def __call__(self) -> int: ...


class _Width4ReceiptModule(Protocol):
    def snapshot_k3_width4_dispatch_receipt(self) -> object: ...


class Width4ReceiptError(RuntimeError):
    """The opt-in receipt could not prove its exact runtime path."""


@lru_cache(maxsize=1)
def width4_receipt_log_enabled() -> bool:
    """Parse the independent, strict and default-off EXO receipt selector."""

    value = os.environ.get(WIDTH4_RECEIPT_LOG_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{WIDTH4_RECEIPT_LOG_ENV} must be exactly '0' or '1'")
    return value == "1"


def _require_child_receipt_selectors() -> None:
    required = (MLX_LM_WIDTH4_RECEIPT_ENV, MLX_Q4_RECEIPT_ENV)
    differing = {
        name: os.environ.get(name) for name in required if os.environ.get(name) != "1"
    }
    if differing:
        raise Width4ReceiptError(
            "EXO width-four receipt logging requires exact child selectors: "
            f"{differing}"
        )


def _runtime_libmlx_path(mx_module: _MlxModule) -> Path:
    core_file = mx_module.__file__
    if not core_file:
        raise Width4ReceiptError("mlx.core has no filesystem identity")
    core_path = Path(core_file).resolve(strict=True)
    candidate = core_path.parent / "lib" / "libmlx.dylib"
    if candidate.is_symlink() or not candidate.is_file():
        raise Width4ReceiptError(
            "receipt libmlx must be a regular, non-symlink sibling of mlx.core"
        )
    return candidate.resolve(strict=True)


def _native_q4_dispatch_count(mx_module: _MlxModule) -> tuple[int, str, str]:
    libmlx_path = _runtime_libmlx_path(mx_module)
    expected_sha256 = os.environ.get(EXPECTED_LIBMLX_SHA256_ENV)
    if (
        expected_sha256 is None
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise Width4ReceiptError(
            f"{EXPECTED_LIBMLX_SHA256_ENV} must be an exact lowercase SHA-256"
        )
    digest = hashlib.sha256()
    with libmlx_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise Width4ReceiptError(
            f"authenticated libmlx SHA-256 differs: {actual_sha256} != {expected_sha256}"
        )
    library = ctypes.CDLL(str(libmlx_path))
    try:
        getter = cast(
            _NativeCounterGetter,
            cast(object, library.mlx_k3_affine6_q4_quad_dispatch_count),
        )
    except AttributeError as error:
        raise Width4ReceiptError(
            "authenticated libmlx lacks the native Q4 receipt symbol"
        ) from error
    getter.argtypes = []
    getter.restype = ctypes.c_uint64
    return int(getter()), str(libmlx_path), actual_sha256


def _integer_field(fields: dict[str, object], name: str) -> int:
    value = fields.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Width4ReceiptError(f"MLX-LM width-four receipt {name} is not an integer")
    return value


def validate_width4_python_receipt(snapshot: dict[str, object]) -> None:
    if snapshot.get("schema_version") != 3:
        raise Width4ReceiptError("MLX-LM width-four receipt schema must be 3")
    generation = _integer_field(snapshot, "generation")
    if generation < 0:
        raise Width4ReceiptError("MLX-LM width-four receipt generation is invalid")
    if snapshot.get("enabled") is not True:
        raise Width4ReceiptError("MLX-LM width-four receipt is not enabled")
    totals = snapshot.get("totals")
    if not isinstance(totals, dict):
        raise Width4ReceiptError("MLX-LM width-four receipt totals are missing")
    typed_totals = cast(dict[str, object], cast(object, totals))
    attempted = _integer_field(typed_totals, "attempted")
    dispatched = _integer_field(typed_totals, "dispatched")
    fallback = _integer_field(typed_totals, "fallback")
    error = _integer_field(typed_totals, "error")
    supported = _integer_field(typed_totals, "supported")
    values = (attempted, dispatched, fallback, error, supported)
    if any(value < 0 for value in values):
        raise Width4ReceiptError("MLX-LM width-four receipt totals are negative")
    if attempted != dispatched + fallback + error:
        raise Width4ReceiptError("MLX-LM width-four receipt is nonterminal")
    if not dispatched <= supported <= attempted:
        raise Width4ReceiptError("MLX-LM width-four supported total is impossible")
    stale = snapshot.get("stale_completions")
    if not isinstance(stale, dict):
        raise Width4ReceiptError(
            "MLX-LM width-four receipt stale-completion totals are missing"
        )
    typed_stale = cast(dict[str, object], cast(object, stale))
    if _integer_field(typed_stale, "total") != 0:
        raise Width4ReceiptError(
            "MLX-LM width-four receipt contains stale pre-generation completions"
        )


def capture_width4_dispatch_receipt(
    *,
    rank: int,
    mx_module: _MlxModule,
    process_id: int,
    host: str,
) -> dict[str, object] | None:
    """Capture cumulative rank-local receipts after terminal output consumption.

    This function never resets counters or synchronizes Metal. C1 must use a
    fresh process and one isolated request so the cumulative receipt is also a
    request receipt.
    """

    if not width4_receipt_log_enabled():
        return None
    if rank not in (0, 1):
        raise Width4ReceiptError(f"width-four TP2 receipt rank must be 0/1: {rank}")
    if process_id <= 0 or not host:
        raise Width4ReceiptError(
            "width-four receipt requires process and host identity"
        )
    _require_child_receipt_selectors()
    receipt_module = cast(
        _Width4ReceiptModule,
        cast(
            object,
            importlib.import_module("mlx_lm.models.kimi_k3_fused_expert"),
        ),
    )
    python_receipt_raw = receipt_module.snapshot_k3_width4_dispatch_receipt()
    if not isinstance(python_receipt_raw, dict):
        raise Width4ReceiptError("MLX-LM width-four receipt must be an object")
    python_receipt = cast(dict[str, object], cast(object, python_receipt_raw))
    validate_width4_python_receipt(python_receipt)
    native_count, libmlx_path, libmlx_sha256 = _native_q4_dispatch_count(mx_module)
    totals = cast(dict[str, object], python_receipt["totals"])
    if _integer_field(totals, "dispatched") <= 0:
        raise Width4ReceiptError("MLX-LM width-four receipt has no dispatch")
    if _integer_field(totals, "fallback") != 0:
        raise Width4ReceiptError("MLX-LM width-four receipt contains fallback")
    if _integer_field(totals, "error") != 0:
        raise Width4ReceiptError("MLX-LM width-four receipt contains errors")
    if native_count <= 0:
        raise Width4ReceiptError("native Q4 receipt has no dispatch")
    return {
        "schema": "k3-width4-dispatch-receipt/v1",
        "scope": "process-cumulative",
        "rank": rank,
        "process_id": process_id,
        "host": host,
        "python_width4": python_receipt,
        "native_q4_dispatch_count": native_count,
        "libmlx_path": libmlx_path,
        "libmlx_sha256": libmlx_sha256,
        "terminal_output_consumed": True,
    }


def format_width4_dispatch_receipt(receipt: dict[str, object]) -> str:
    """Render one stable, grep-safe terminal log record."""

    return RECEIPT_MARKER + json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    )
