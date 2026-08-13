"""Fail-closed request receipts for the Kimi K3 width-four candidate."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

WIDTH4_RECEIPT_LOG_ENV = "EXO_MLX_KIMI_K3_WIDTH4_DISPATCH_RECEIPT_LOG"
WIDTH4_RECEIPT_SESSION_ID_ENV = "EXO_MLX_KIMI_K3_WIDTH4_RECEIPT_SESSION_ID"
EXPECTED_LIBMLX_SHA256_ENV = "EXO_MLX_KIMI_K3_WIDTH4_LIBMLX_SHA256"

MLX_LM_WIDTH4_RECEIPT_ENV = "MLX_LM_KIMI_K3_WIDTH4_DISPATCH_RECEIPT"
MLX_LM_FUSED_EXPERT_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERTS"
MLX_LM_FUSED_DOWN_REDUCE_ENV = "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE"
MLX_LM_WIDTH4_EXACT_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH4_EXACT"
MLX_LM_DERIVE_AFFINE2_BIAS_ENV = "MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS"
MLX_Q4_RECEIPT_ENV = "MLX_METAL_K3_AFFINE6_Q4_DISPATCH_RECEIPT"
MLX_Q4_SELECTOR_ENV = "MLX_METAL_K3_AFFINE6_Q4_QUAD"

RECEIPT_MARKER = "K3_WIDTH4_DISPATCH_RECEIPT "

_PYTHON_SELECTOR_ENVS = (
    MLX_LM_WIDTH4_RECEIPT_ENV,
    MLX_LM_FUSED_EXPERT_ENV,
    MLX_LM_FUSED_DOWN_REDUCE_ENV,
    MLX_LM_WIDTH4_EXACT_ENV,
    MLX_LM_DERIVE_AFFINE2_BIAS_ENV,
)
_REQUIRED_SELECTOR_ENVS = (
    *_PYTHON_SELECTOR_ENVS,
    MLX_Q4_RECEIPT_ENV,
    MLX_Q4_SELECTOR_ENV,
)
_RECEIPT_PATHS = ("switch_glu", "switch_glu_reduce")
_RECEIPT_METRICS = ("attempted", "supported", "dispatched", "fallback", "error")
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z", flags=re.ASCII)


class _MlxModule(Protocol):
    @property
    def __file__(self) -> str | None: ...


class _NativeCounterGetter(Protocol):
    argtypes: list[object]
    restype: object

    def __call__(self) -> int: ...


class _NativeCounterResetter(Protocol):
    argtypes: list[object]
    restype: object

    def __call__(self) -> None: ...


class _Width4ReceiptModule(Protocol):
    def reset_k3_width4_dispatch_receipt(self) -> None: ...

    def snapshot_k3_width4_dispatch_receipt(self) -> object: ...


@dataclass(frozen=True)
class _NativeCounterApi:
    library: object = field(repr=False, compare=False)
    getter: _NativeCounterGetter = field(repr=False, compare=False)
    resetter: _NativeCounterResetter = field(repr=False, compare=False)
    libmlx_path: str
    libmlx_sha256: str


@dataclass(frozen=True)
class Width4CounterReset:
    """An agreed zero baseline created after warmup."""

    generation: int
    native_baseline: int
    libmlx_path: str
    libmlx_sha256: str

    def agreement_contract(self) -> str:
        return json.dumps(
            {
                "generation": self.generation,
                "native_baseline": self.native_baseline,
                "libmlx_sha256": self.libmlx_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class Width4RequestReceiptContext:
    """Authenticated, request-local receipt state established before prefill."""

    session_id: str
    model_id: str
    model_fingerprint: str
    request_fingerprint: tuple[int, ...]
    request_fingerprint_sha256: str
    rank: int
    world_size: int
    verify_width: int
    generation: int
    native_baseline: int
    selectors: tuple[tuple[str, str], ...]
    libmlx_path: str
    libmlx_sha256: str
    native_api: _NativeCounterApi = field(repr=False, compare=False)

    def agreement_contract(self) -> str:
        """Return the exact rank-common contract agreed before prefill."""

        return json.dumps(
            {
                "session_id": self.session_id,
                "model_id": self.model_id,
                "model_fingerprint": self.model_fingerprint,
                "request_fingerprint": self.request_fingerprint,
                "request_fingerprint_sha256": self.request_fingerprint_sha256,
                "world_size": self.world_size,
                "verify_width": self.verify_width,
                "generation": self.generation,
                "native_baseline": self.native_baseline,
                "selectors": self.selectors,
                "libmlx_sha256": self.libmlx_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class Width4ReceiptError(RuntimeError):
    """The opt-in receipt could not prove its exact runtime path."""


_one_shot_lock = Lock()
_request_attempted = False
_capture_attempted = False
_active_context_signature: tuple[object, ...] | None = None


@lru_cache(maxsize=1)
def width4_receipt_log_enabled() -> bool:
    """Parse the independent, strict and default-off EXO receipt selector."""

    value = os.environ.get(WIDTH4_RECEIPT_LOG_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{WIDTH4_RECEIPT_LOG_ENV} must be exactly '0' or '1'")
    return value == "1"


def width4_receipt_session_id() -> str:
    """Return the externally pinned, reducer-safe receipt session identity."""

    value = os.environ.get(WIDTH4_RECEIPT_SESSION_ID_ENV)
    if value is None or _SESSION_ID_PATTERN.fullmatch(value) is None:
        raise Width4ReceiptError(
            f"{WIDTH4_RECEIPT_SESSION_ID_ENV} must be 1-128 ASCII characters "
            "from [A-Za-z0-9._:-]"
        )
    return value


def _require_exact_receipt_selectors() -> dict[str, str]:
    selectors = {name: os.environ.get(name) for name in _REQUIRED_SELECTOR_ENVS}
    differing = {name: value for name, value in selectors.items() if value != "1"}
    if differing:
        raise Width4ReceiptError(
            "EXO width-four receipt logging requires exact candidate selectors: "
            f"{differing}"
        )
    return cast(dict[str, str], selectors)


def validate_width4_request_contract(
    *,
    rank: int,
    world_size: int,
    verify_width: int,
    model_id: str,
) -> tuple[str, dict[str, str]]:
    """Validate the exact TP2/W4 request envelope before model execution."""

    if world_size != 2 or rank not in (0, 1):
        raise Width4ReceiptError(
            "width-four receipt requires exact TP2 ranks 0/1; "
            f"got rank={rank}, world_size={world_size}"
        )
    if verify_width != 4:
        raise Width4ReceiptError(
            f"width-four receipt requires DSpark verify_width=4; got {verify_width}"
        )
    if not model_id or not model_id.isascii():
        raise Width4ReceiptError("width-four receipt requires an ASCII model ID")
    session_id = width4_receipt_session_id()
    return session_id, _require_exact_receipt_selectors()


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


def _native_counter_api(mx_module: _MlxModule) -> _NativeCounterApi:
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
        resetter = cast(
            _NativeCounterResetter,
            cast(object, library.mlx_k3_affine6_q4_quad_reset_dispatch_count),
        )
    except AttributeError as error:
        raise Width4ReceiptError(
            "authenticated libmlx lacks the native Q4 receipt/reset symbols"
        ) from error
    getter.argtypes = []
    getter.restype = ctypes.c_uint64
    resetter.argtypes = []
    resetter.restype = None
    return _NativeCounterApi(
        library=library,
        getter=getter,
        resetter=resetter,
        libmlx_path=str(libmlx_path),
        libmlx_sha256=actual_sha256,
    )


def _receipt_module() -> _Width4ReceiptModule:
    return cast(
        _Width4ReceiptModule,
        cast(
            object,
            importlib.import_module("mlx_lm.models.kimi_k3_fused_expert"),
        ),
    )


def _integer_field(fields: dict[str, object], name: str) -> int:
    value = fields.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Width4ReceiptError(f"MLX-LM width-four receipt {name} is not an integer")
    return value


def _object_field(fields: dict[str, object], name: str) -> dict[str, object]:
    value = fields.get(name)
    if not isinstance(value, dict):
        raise Width4ReceiptError(f"MLX-LM width-four receipt {name} is missing")
    return cast(dict[str, object], cast(object, value))


def _list_field(fields: dict[str, object], name: str) -> list[object]:
    value = fields.get(name)
    if not isinstance(value, list):
        raise Width4ReceiptError(f"MLX-LM width-four receipt {name} is missing")
    return cast(list[object], cast(object, value))


def _validate_metric_counts(
    fields: dict[str, object],
    *,
    label: str,
) -> dict[str, int]:
    if set(fields) != set(_RECEIPT_METRICS):
        raise Width4ReceiptError(
            f"MLX-LM width-four {label} metric fields differ: {sorted(fields)}"
        )
    counts = {name: _integer_field(fields, name) for name in _RECEIPT_METRICS}
    if any(value < 0 for value in counts.values()):
        raise Width4ReceiptError(f"MLX-LM width-four {label} counts are negative")
    if counts["attempted"] != (
        counts["dispatched"] + counts["fallback"] + counts["error"]
    ):
        raise Width4ReceiptError(f"MLX-LM width-four {label} is nonterminal")
    if not counts["dispatched"] <= counts["supported"] <= counts["attempted"]:
        raise Width4ReceiptError(
            f"MLX-LM width-four {label} supported count is impossible"
        )
    return counts


def _validate_selectors(raw: object, *, label: str) -> None:
    if not isinstance(raw, dict):
        raise Width4ReceiptError(f"MLX-LM width-four {label} selectors are missing")
    selectors = cast(dict[str, object], cast(object, raw))
    expected = {name: "1" for name in _PYTHON_SELECTOR_ENVS}
    if selectors != expected:
        raise Width4ReceiptError(
            f"MLX-LM width-four {label} selectors differ: {selectors}"
        )


def validate_width4_python_receipt(
    snapshot: dict[str, object],
    *,
    expected_generation: int | None = None,
    require_dispatch: bool = True,
) -> None:
    """Validate the complete transactional MLX-LM W4 receipt schema."""

    if snapshot.get("schema_version") != 3:
        raise Width4ReceiptError("MLX-LM width-four receipt schema must be 3")
    generation = _integer_field(snapshot, "generation")
    if generation < 0 or (
        expected_generation is not None and generation != expected_generation
    ):
        raise Width4ReceiptError(
            "MLX-LM width-four receipt generation differs from request baseline"
        )
    if snapshot.get("enabled") is not True:
        raise Width4ReceiptError("MLX-LM width-four receipt is not enabled")
    _validate_selectors(snapshot.get("current_selectors"), label="current")

    totals = _validate_metric_counts(
        _object_field(snapshot, "totals"),
        label="total",
    )
    paths = _object_field(snapshot, "paths")
    if set(paths) != set(_RECEIPT_PATHS):
        raise Width4ReceiptError("MLX-LM width-four receipt paths differ")
    path_counts = {
        path: _validate_metric_counts(
            _object_field(paths, path),
            label=f"{path} path",
        )
        for path in _RECEIPT_PATHS
    }
    if any(path_counts["switch_glu"].values()):
        raise Width4ReceiptError("MLX-LM switch_glu path must remain unused")
    if path_counts["switch_glu_reduce"] != totals:
        raise Width4ReceiptError(
            "MLX-LM switch_glu_reduce path does not equal receipt totals"
        )
    if totals["fallback"] != 0 or totals["error"] != 0:
        raise Width4ReceiptError("MLX-LM width-four receipt contains fallback/errors")
    if totals["supported"] != totals["attempted"]:
        raise Width4ReceiptError("MLX-LM width-four receipt contains unsupported work")
    if require_dispatch and totals["dispatched"] <= 0:
        raise Width4ReceiptError("MLX-LM switch_glu_reduce path has no dispatch")
    if not require_dispatch and any(totals.values()):
        raise Width4ReceiptError("MLX-LM width-four reset baseline is not zero")

    if _object_field(snapshot, "fallback_reason_classes"):
        raise Width4ReceiptError("MLX-LM width-four receipt has fallback classes")
    if _object_field(snapshot, "error_reason_classes"):
        raise Width4ReceiptError("MLX-LM width-four receipt has error classes")

    terminal_records = _list_field(snapshot, "terminal_records")
    selector_states = _list_field(snapshot, "selector_states")
    if not require_dispatch:
        if terminal_records or selector_states:
            raise Width4ReceiptError("MLX-LM reset baseline contains terminal records")
    else:
        terminal_dispatches = 0
        for raw_record in terminal_records:
            if not isinstance(raw_record, dict):
                raise Width4ReceiptError("MLX-LM terminal record is not an object")
            record = cast(dict[str, object], cast(object, raw_record))
            count = _integer_field(record, "count")
            _validate_selectors(record.get("selectors"), label="terminal record")
            if (
                record.get("path") != "switch_glu_reduce"
                or record.get("outcome") != "dispatched"
                or record.get("supported") is not True
                or record.get("reason_class") is not None
                or count <= 0
            ):
                raise Width4ReceiptError(
                    "MLX-LM terminal record is not a clean switch_glu_reduce dispatch"
                )
            terminal_dispatches += count
        if terminal_dispatches != totals["dispatched"]:
            raise Width4ReceiptError(
                "MLX-LM terminal record counts differ from dispatched total"
            )
        if len(selector_states) != 1 or not isinstance(selector_states[0], dict):
            raise Width4ReceiptError("MLX-LM selector-state receipt is not singular")
        selector_state = cast(dict[str, object], cast(object, selector_states[0]))
        _validate_selectors(selector_state.get("selectors"), label="selector state")
        if (
            selector_state.get("path") != "switch_glu_reduce"
            or _integer_field(selector_state, "attempted") != totals["attempted"]
        ):
            raise Width4ReceiptError("MLX-LM selector-state receipt differs")

    stale = _object_field(snapshot, "stale_completions")
    if set(stale) != {"total", "outcomes", "paths"}:
        raise Width4ReceiptError("MLX-LM width-four stale-completion fields differ")
    if _integer_field(stale, "total") != 0:
        raise Width4ReceiptError(
            "MLX-LM width-four receipt contains stale pre-generation completions"
        )
    for label in ("outcomes", "paths"):
        values = _object_field(stale, label)
        expected_keys = (
            {"dispatched", "fallback", "error"}
            if label == "outcomes"
            else set(_RECEIPT_PATHS)
        )
        if set(values) != expected_keys:
            raise Width4ReceiptError(
                f"MLX-LM width-four stale-completion {label} fields differ"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value != 0
            for value in values.values()
        ):
            raise Width4ReceiptError(
                f"MLX-LM width-four stale-completion {label} are nonzero"
            )


def _reset_counters(
    mx_module: _MlxModule,
) -> tuple[Width4CounterReset, _NativeCounterApi]:
    receipt_module = _receipt_module()
    native_api = _native_counter_api(mx_module)
    receipt_module.reset_k3_width4_dispatch_receipt()
    native_api.resetter()
    native_baseline = int(native_api.getter())
    if native_baseline != 0:
        raise Width4ReceiptError(
            f"native Q4 reset baseline must be zero; got {native_baseline}"
        )
    snapshot_raw = receipt_module.snapshot_k3_width4_dispatch_receipt()
    if not isinstance(snapshot_raw, dict):
        raise Width4ReceiptError("MLX-LM width-four receipt must be an object")
    snapshot = cast(dict[str, object], cast(object, snapshot_raw))
    generation = _integer_field(snapshot, "generation")
    validate_width4_python_receipt(
        snapshot,
        expected_generation=generation,
        require_dispatch=False,
    )
    return (
        Width4CounterReset(
            generation=generation,
            native_baseline=native_baseline,
            libmlx_path=native_api.libmlx_path,
            libmlx_sha256=native_api.libmlx_sha256,
        ),
        native_api,
    )


def reset_width4_dispatch_counters_after_warmup(
    *,
    rank: int,
    world_size: int,
    verify_width: int,
    model_id: str,
    mx_module: _MlxModule,
) -> Width4CounterReset | None:
    """Remove warmup pollution without consuming the one-shot request."""

    if not width4_receipt_log_enabled():
        return None
    validate_width4_request_contract(
        rank=rank,
        world_size=world_size,
        verify_width=verify_width,
        model_id=model_id,
    )
    reset, _native_api = _reset_counters(mx_module)
    return reset


def _fingerprint_sha256(fingerprint: tuple[int, ...]) -> str:
    if len(fingerprint) != 4 or any(
        type(word) is not int or not 0 <= word <= 0x7FFFFFFF for word in fingerprint
    ):
        raise Width4ReceiptError(
            "width-four receipt requires the rank-agreed four-word setup fingerprint"
        )
    digest = hashlib.sha256(b"exo-kimi-k3-width4-request-fingerprint/v2\0")
    for word in fingerprint:
        digest.update(word.to_bytes(4, "big"))
    return digest.hexdigest()


def _model_fingerprint(model_id: str) -> str:
    return hashlib.sha256(
        b"exo-kimi-k3-model-id/v1\0" + model_id.encode("ascii")
    ).hexdigest()


def _context_signature(context: Width4RequestReceiptContext) -> tuple[object, ...]:
    return (
        context.session_id,
        context.request_fingerprint,
        context.rank,
        context.generation,
        context.libmlx_sha256,
    )


def begin_width4_request_receipt(
    *,
    rank: int,
    world_size: int,
    verify_width: int,
    model_id: str,
    request_fingerprint: tuple[int, ...],
    mx_module: _MlxModule,
) -> Width4RequestReceiptContext | None:
    """Claim the worker once and reset counters immediately before C1 prefill."""

    if not width4_receipt_log_enabled():
        return None

    global _request_attempted
    with _one_shot_lock:
        if _request_attempted:
            raise Width4ReceiptError(
                "width-four receipt worker is one-shot; a request was already attempted"
            )
        # Consume the worker before any failure-prone validation/reset operation.
        _request_attempted = True

    session_id, selectors = validate_width4_request_contract(
        rank=rank,
        world_size=world_size,
        verify_width=verify_width,
        model_id=model_id,
    )
    request_fingerprint_sha256 = _fingerprint_sha256(request_fingerprint)
    reset, native_api = _reset_counters(mx_module)
    context = Width4RequestReceiptContext(
        session_id=session_id,
        model_id=model_id,
        model_fingerprint=_model_fingerprint(model_id),
        request_fingerprint=request_fingerprint,
        request_fingerprint_sha256=request_fingerprint_sha256,
        rank=rank,
        world_size=world_size,
        verify_width=verify_width,
        generation=reset.generation,
        native_baseline=reset.native_baseline,
        selectors=tuple(sorted(selectors.items())),
        libmlx_path=reset.libmlx_path,
        libmlx_sha256=reset.libmlx_sha256,
        native_api=native_api,
    )
    global _active_context_signature
    with _one_shot_lock:
        _active_context_signature = _context_signature(context)
    return context


def capture_width4_dispatch_receipt(
    *,
    context: Width4RequestReceiptContext | None,
    process_id: int,
    host: str,
    response_built: bool,
    generation_callback_complete: bool,
    terminal_barrier_complete: bool,
    confidence_finalization_complete: bool,
    packed_attestation_complete: bool,
) -> dict[str, object] | None:
    """Capture one request-local receipt after every terminal boundary."""

    if context is None:
        return None

    global _capture_attempted
    with _one_shot_lock:
        if _capture_attempted:
            raise Width4ReceiptError(
                "width-four receipt capture was already attempted in this worker"
            )
        _capture_attempted = True
        if _active_context_signature != _context_signature(context):
            raise Width4ReceiptError("width-four receipt context is not active")

    if process_id <= 0 or not host:
        raise Width4ReceiptError(
            "width-four receipt requires process and host identity"
        )
    boundaries = {
        "response_built": response_built,
        "generation_callback_complete": generation_callback_complete,
        "terminal_barrier_complete": terminal_barrier_complete,
        "confidence_finalization_complete": confidence_finalization_complete,
        "packed_attestation_complete": packed_attestation_complete,
        "rank_agreement_pending": True,
    }
    if any(value is not True for value in boundaries.values()):
        raise Width4ReceiptError(
            f"width-four receipt terminal boundaries are incomplete: {boundaries}"
        )
    session_id, selectors = validate_width4_request_contract(
        rank=context.rank,
        world_size=context.world_size,
        verify_width=context.verify_width,
        model_id=context.model_id,
    )
    if session_id != context.session_id or tuple(sorted(selectors.items())) != (
        context.selectors
    ):
        raise Width4ReceiptError("width-four receipt environment changed mid-request")

    receipt_module = _receipt_module()
    python_receipt_raw = receipt_module.snapshot_k3_width4_dispatch_receipt()
    if not isinstance(python_receipt_raw, dict):
        raise Width4ReceiptError("MLX-LM width-four receipt must be an object")
    python_receipt = cast(dict[str, object], cast(object, python_receipt_raw))
    validate_width4_python_receipt(
        python_receipt,
        expected_generation=context.generation,
        require_dispatch=True,
    )

    native_final = int(context.native_api.getter())
    native_delta = native_final - context.native_baseline
    if native_delta <= 0:
        raise Width4ReceiptError("native Q4 request delta has no dispatch")

    paths = cast(dict[str, object], python_receipt["paths"])
    return {
        "schema": "k3-width4-dispatch-receipt/v2",
        "scope": "single-request-reset-after-warmup",
        "session_id": context.session_id,
        "model_id": context.model_id,
        "model_fingerprint": {
            "kind": "model-id-sha256/v1",
            "sha256": context.model_fingerprint,
        },
        "request_fingerprint": list(context.request_fingerprint),
        "request_fingerprint_sha256": context.request_fingerprint_sha256,
        "rank": context.rank,
        "world_size": context.world_size,
        "verify_width": context.verify_width,
        "process_id": process_id,
        "host": host,
        "generation": context.generation,
        "selectors": dict(context.selectors),
        "path_counts": {
            path: cast(dict[str, object], paths[path]) for path in _RECEIPT_PATHS
        },
        "python_width4": python_receipt,
        "native_q4_dispatch": {
            "baseline": context.native_baseline,
            "final": native_final,
            "delta": native_delta,
        },
        "libmlx_path": context.libmlx_path,
        "libmlx_sha256": context.libmlx_sha256,
        "terminal_boundaries": {
            **boundaries,
            # The caller performs rank agreement around this capture and logs
            # only after it returns. The reducer can require this declaration.
            "rank_agreement_pending": True,
        },
    }


def mark_width4_receipt_rank_agreed(receipt: dict[str, object]) -> None:
    """Mark a locally captured receipt after all ranks agreed capture succeeded."""

    terminal_boundaries = receipt.get("terminal_boundaries")
    if not isinstance(terminal_boundaries, dict):
        raise Width4ReceiptError("width-four receipt terminal boundaries are missing")
    boundaries = cast(dict[str, object], cast(object, terminal_boundaries))
    if boundaries.get("rank_agreement_pending") is not True:
        raise Width4ReceiptError("width-four receipt was not pending rank agreement")
    boundaries["rank_agreement_pending"] = False
    boundaries["rank_capture_agreed"] = True


def width4_receipt_agreement_contract(receipt: dict[str, object]) -> str:
    """Return the full rank-common terminal contract for exact agreement."""

    required = (
        "schema",
        "scope",
        "session_id",
        "model_id",
        "model_fingerprint",
        "request_fingerprint",
        "request_fingerprint_sha256",
        "world_size",
        "verify_width",
        "generation",
        "selectors",
        "path_counts",
        "python_width4",
        "native_q4_dispatch",
        "libmlx_sha256",
        "terminal_boundaries",
    )
    missing = [name for name in required if name not in receipt]
    if missing:
        raise Width4ReceiptError(
            f"width-four terminal agreement contract is missing fields: {missing}"
        )
    return json.dumps(
        {name: receipt[name] for name in required},
        sort_keys=True,
        separators=(",", ":"),
    )


def format_width4_dispatch_receipt(receipt: dict[str, object]) -> str:
    """Render one stable, grep-safe terminal log record."""

    return RECEIPT_MARKER + json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    )
