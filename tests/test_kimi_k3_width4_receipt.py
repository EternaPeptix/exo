from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from exo.worker.engines.mlx.generator import kimi_k3_width4_receipt as subject


class _FakeMlxModule:
    __file__: str | None

    def __init__(self, path: str | None) -> None:
        self.__file__ = path


class _CounterState:
    def __init__(self, count: int = 99) -> None:
        self.count = count
        self.reset_calls = 0


class _Getter:
    argtypes: list[object] | None = None
    restype: object | None = None

    def __init__(self, state: _CounterState) -> None:
        self.state = state

    def __call__(self) -> int:
        return self.state.count


class _Resetter:
    argtypes: list[object] | None = None
    restype: object | None = object()

    def __init__(self, state: _CounterState) -> None:
        self.state = state

    def __call__(self) -> None:
        self.state.reset_calls += 1
        self.state.count = 0


class _ReceiptState:
    def __init__(self) -> None:
        self.generation = 0
        self.dispatched = 0
        self.reset_calls = 0

    def reset_k3_width4_dispatch_receipt(self) -> None:
        self.generation += 1
        self.dispatched = 0
        self.reset_calls += 1

    def snapshot_k3_width4_dispatch_receipt(self) -> dict[str, object]:
        return _python_receipt(
            generation=self.generation,
            dispatched=self.dispatched,
        )


def _python_selectors() -> dict[str, str]:
    return {
        name: "1"
        for name in (
            subject.MLX_LM_WIDTH4_RECEIPT_ENV,
            subject.MLX_LM_FUSED_EXPERT_ENV,
            subject.MLX_LM_FUSED_DOWN_REDUCE_ENV,
            subject.MLX_LM_WIDTH4_EXACT_ENV,
            subject.MLX_LM_DERIVE_AFFINE2_BIAS_ENV,
        )
    }


def _python_receipt(
    *,
    generation: int = 1,
    dispatched: int = 3,
) -> dict[str, object]:
    totals = {
        "attempted": dispatched,
        "supported": dispatched,
        "dispatched": dispatched,
        "fallback": 0,
        "error": 0,
    }
    zero = {metric: 0 for metric in subject._RECEIPT_METRICS}
    terminal_records: list[dict[str, object]] = []
    selector_states: list[dict[str, object]] = []
    if dispatched:
        terminal_records.append(
            {
                "path": "switch_glu_reduce",
                "outcome": "dispatched",
                "supported": True,
                "reason_class": None,
                "selectors": _python_selectors(),
                "count": dispatched,
            }
        )
        selector_states.append(
            {
                "path": "switch_glu_reduce",
                "attempted": dispatched,
                "selectors": _python_selectors(),
            }
        )
    return {
        "schema_version": 3,
        "generation": generation,
        "enabled": True,
        "current_selectors": _python_selectors(),
        "totals": totals,
        "paths": {
            "switch_glu": zero,
            "switch_glu_reduce": dict(totals),
        },
        "fallback_reason_classes": {},
        "error_reason_classes": {},
        "selector_states": selector_states,
        "terminal_records": terminal_records,
        "stale_completions": {
            "total": 0,
            "outcomes": {name: 0 for name in ("dispatched", "fallback", "error")},
            "paths": {name: 0 for name in subject._RECEIPT_PATHS},
        },
    }


@pytest.fixture(autouse=True)
def _reset_process_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(subject, "_request_attempted", False)
    monkeypatch.setattr(subject, "_capture_attempted", False)
    monkeypatch.setattr(subject, "_active_request_claim", None)
    monkeypatch.setattr(subject, "_active_context_signature", None)
    subject.width4_receipt_log_enabled.cache_clear()
    yield
    subject.width4_receipt_log_enabled.cache_clear()


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        subject.WIDTH4_RECEIPT_LOG_ENV,
        *subject._REQUIRED_SELECTOR_ENVS,
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv(subject.WIDTH4_RECEIPT_SESSION_ID_ENV, "c1-20260813:trial_01")
    monkeypatch.setenv(
        subject.EXPECTED_LIBMLX_SHA256_ENV,
        hashlib.sha256(b"libmlx").hexdigest(),
    )
    subject.width4_receipt_log_enabled.cache_clear()


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[_FakeMlxModule, _CounterState, _ReceiptState, _Getter, _Resetter]:
    core = tmp_path / "mlx" / "core.cpython-313-darwin.so"
    libmlx = tmp_path / "mlx" / "lib" / "libmlx.dylib"
    libmlx.parent.mkdir(parents=True)
    core.write_bytes(b"core")
    libmlx.write_bytes(b"libmlx")
    counter = _CounterState()
    getter = _Getter(counter)
    resetter = _Resetter(counter)
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _path: SimpleNamespace(
            mlx_k3_affine6_q4_quad_dispatch_count=getter,
            mlx_k3_affine6_q4_quad_reset_dispatch_count=resetter,
        ),
    )
    receipt_state = _ReceiptState()
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_lm.models", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.kimi_k3_fused_expert",
        receipt_state,
    )
    return _FakeMlxModule(str(core)), counter, receipt_state, getter, resetter


def _begin(
    mx_module: _FakeMlxModule,
) -> subject.Width4RequestReceiptContext:
    claim = subject.claim_width4_request_receipt_attempt()
    assert claim is not None
    context = subject.begin_width4_request_receipt(
        rank=1,
        world_size=2,
        verify_width=4,
        model_id="kernelpool/Kimi-K3-2bit-UVMAX",
        request_fingerprint=(1, 2, 3, 4),
        mx_module=mx_module,
        claim=claim,
    )
    assert context is not None
    return context


def _capture(
    context: subject.Width4RequestReceiptContext | None,
    **boundaries: bool,
) -> dict[str, object] | None:
    values = {
        "response_built": True,
        "generation_callback_complete": True,
        "terminal_barrier_complete": True,
        "confidence_finalization_complete": True,
        "packed_attestation_complete": True,
    }
    values.update(boundaries)
    return subject.capture_width4_dispatch_receipt(
        context=context,
        process_id=8123,
        host="512S2.local",
        **values,
    )


def test_default_off_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(subject.WIDTH4_RECEIPT_LOG_ENV, raising=False)
    subject.width4_receipt_log_enabled.cache_clear()
    assert (
        subject.reset_width4_dispatch_counters_after_warmup(
            rank=0,
            world_size=1,
            verify_width=0,
            model_id="",
            mx_module=_FakeMlxModule(None),
        )
        is None
    )
    assert (
        subject.begin_width4_request_receipt(
            rank=0,
            world_size=1,
            verify_width=0,
            model_id="",
            request_fingerprint=(),
            mx_module=_FakeMlxModule(None),
            claim=None,
        )
        is None
    )
    assert _capture(None) is None
    assert subject._request_attempted is False


def test_selector_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(subject.WIDTH4_RECEIPT_LOG_ENV, "true")
    subject.width4_receipt_log_enabled.cache_clear()
    with pytest.raises(ValueError, match="must be exactly"):
        subject.width4_receipt_log_enabled()


@pytest.mark.parametrize(
    "session_id",
    ("", "contains space", "slash/not-allowed", "café", "a" * 129),
)
def test_session_id_is_explicit_ascii_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    session_id: str,
) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv(subject.WIDTH4_RECEIPT_SESSION_ID_ENV, session_id)
    with pytest.raises(subject.Width4ReceiptError, match="1-128 ASCII"):
        subject.width4_receipt_session_id()


@pytest.mark.parametrize(
    ("rank", "world_size", "verify_width"),
    ((2, 2, 4), (0, 3, 4), (0, 2, 3)),
)
def test_exact_tp2_width4_contract_is_required(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    world_size: int,
    verify_width: int,
) -> None:
    _enable(monkeypatch)
    with pytest.raises(subject.Width4ReceiptError):
        subject.validate_width4_request_contract(
            rank=rank,
            world_size=world_size,
            verify_width=verify_width,
            model_id="kernelpool/Kimi-K3-2bit-UVMAX",
        )


def test_every_candidate_selector_is_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    for name in subject._REQUIRED_SELECTOR_ENVS:
        monkeypatch.setenv(name, "0")
        with pytest.raises(subject.Width4ReceiptError, match="exact candidate"):
            subject.validate_width4_request_contract(
                rank=0,
                world_size=2,
                verify_width=4,
                model_id="kernelpool/Kimi-K3-2bit-UVMAX",
            )
        monkeypatch.setenv(name, "1")


def test_failed_first_attempt_still_consumes_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv(subject.MLX_Q4_SELECTOR_ENV, "0")
    with pytest.raises(subject.Width4ReceiptError, match="exact candidate"):
        _begin(_FakeMlxModule(None))
    monkeypatch.setenv(subject.MLX_Q4_SELECTOR_ENV, "1")
    with pytest.raises(subject.Width4ReceiptError, match="one-shot"):
        _begin(_FakeMlxModule(None))


def test_service_admission_precedes_all_request_setup() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "src/exo/worker/runner/llm_inference/batch_generator.py"
    )
    source = source_path.read_text()
    admission = source.index(
        "width4_receipt_claim = claim_width4_request_receipt_attempt()"
    )
    task_digest = source.index("return _task_digest(task)", admission)
    template = source.index("_check_for_debug_prompts(task.task_params)", task_digest)
    parser = source.index("self._build_output_generator(task, queue)", task_digest)
    assert admission < task_digest < template
    assert admission < task_digest < parser


def test_post_warmup_reset_does_not_consume_one_shot_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    mx_module, counter, receipt_state, getter, resetter = _install_runtime(
        monkeypatch, tmp_path
    )
    reset = subject.reset_width4_dispatch_counters_after_warmup(
        rank=1,
        world_size=2,
        verify_width=4,
        model_id="kernelpool/Kimi-K3-2bit-UVMAX",
        mx_module=mx_module,
    )
    assert reset is not None
    assert reset.generation == 1
    assert reset.native_baseline == 0
    assert subject._request_attempted is False
    context = _begin(mx_module)
    assert context.generation == 2
    assert counter.count == 0
    assert receipt_state.reset_calls == 2
    assert resetter.state.reset_calls == 2
    assert getter.argtypes == []
    assert getter.restype is ctypes.c_uint64
    assert resetter.argtypes == []
    assert resetter.restype is None


@pytest.mark.parametrize(
    "mutation",
    ("switch_glu", "fallback", "error", "stale", "selector"),
)
def test_nonexact_python_path_receipts_are_rejected(mutation: str) -> None:
    receipt = _python_receipt()
    totals = receipt["totals"]
    paths = receipt["paths"]
    assert isinstance(totals, dict) and isinstance(paths, dict)
    reduce_path = paths["switch_glu_reduce"]
    assert isinstance(reduce_path, dict)
    if mutation == "switch_glu":
        switch_path = paths["switch_glu"]
        assert isinstance(switch_path, dict)
        switch_path.update(totals)
    elif mutation == "fallback":
        totals.update(attempted=3, supported=2, dispatched=2, fallback=1)
        reduce_path.update(totals)
    elif mutation == "error":
        totals.update(attempted=3, supported=3, dispatched=2, error=1)
        reduce_path.update(totals)
    elif mutation == "stale":
        stale = receipt["stale_completions"]
        assert isinstance(stale, dict)
        stale["total"] = 1
    else:
        selectors = receipt["current_selectors"]
        assert isinstance(selectors, dict)
        selectors[subject.MLX_LM_WIDTH4_EXACT_ENV] = "0"
    with pytest.raises(subject.Width4ReceiptError):
        subject.validate_width4_python_receipt(receipt)


def test_capture_binds_request_and_exact_counter_deltas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    mx_module, counter, receipt_state, _getter, _resetter = _install_runtime(
        monkeypatch, tmp_path
    )
    context = _begin(mx_module)
    receipt_state.dispatched = 7
    counter.count = 17
    receipt = _capture(context)
    assert receipt is not None
    subject.mark_width4_receipt_rank_agreed(receipt)
    assert receipt["schema"] == "k3-width4-dispatch-receipt/v3"
    assert receipt["session_id"] == "c1-20260813:trial_01"
    assert receipt["model_id"] == "kernelpool/Kimi-K3-2bit-UVMAX"
    assert receipt["request_fingerprint"] == [1, 2, 3, 4]
    assert receipt["rank"] == 1
    assert receipt["world_size"] == 2
    assert receipt["generation"] == 1
    assert receipt["native_q4_dispatch"] == {
        "baseline": 0,
        "final": 17,
        "delta": 17,
    }
    assert receipt["selectors"] == {
        name: "1" for name in subject._REQUIRED_SELECTOR_ENVS
    }
    boundaries = receipt["terminal_boundaries"]
    assert isinstance(boundaries, dict)
    assert boundaries["rank_agreement_pending"] is False
    assert boundaries["rank_capture_agreed"] is True
    rendered = subject.format_width4_dispatch_receipt(receipt)
    assert rendered.startswith(subject.RECEIPT_MARKER)
    assert json.loads(rendered.removeprefix(subject.RECEIPT_MARKER)) == receipt


def test_terminal_agreement_contract_excludes_only_rank_local_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    mx_module, counter, receipt_state, _getter, _resetter = _install_runtime(
        monkeypatch, tmp_path
    )
    context = _begin(mx_module)
    receipt_state.dispatched = 2
    counter.count = 2
    receipt = _capture(context)
    assert receipt is not None
    contract = subject.width4_receipt_agreement_contract(receipt)
    peer = dict(receipt)
    peer.update(rank=0, process_id=999, host="512S1.local", libmlx_path="/peer/libmlx")
    assert subject.width4_receipt_agreement_contract(peer) == contract
    changed = dict(peer)
    changed["generation"] = 999
    assert subject.width4_receipt_agreement_contract(changed) != contract


def test_generation_mismatch_and_zero_native_delta_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    mx_module, counter, receipt_state, _getter, _resetter = _install_runtime(
        monkeypatch, tmp_path
    )
    context = _begin(mx_module)
    receipt_state.dispatched = 1
    receipt_state.generation += 1
    counter.count = 1
    with pytest.raises(subject.Width4ReceiptError, match="generation"):
        _capture(context)


def test_incomplete_terminal_boundary_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    mx_module, counter, receipt_state, _getter, _resetter = _install_runtime(
        monkeypatch, tmp_path
    )
    context = _begin(mx_module)
    receipt_state.dispatched = 1
    counter.count = 1
    with pytest.raises(subject.Width4ReceiptError, match="boundaries"):
        _capture(context, packed_attestation_complete=False)


def test_missing_native_reset_symbol_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    core = tmp_path / "mlx" / "core.cpython-313-darwin.so"
    libmlx = tmp_path / "mlx" / "lib" / "libmlx.dylib"
    libmlx.parent.mkdir(parents=True)
    core.write_bytes(b"core")
    libmlx.write_bytes(b"libmlx")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _path: SimpleNamespace(
            mlx_k3_affine6_q4_quad_dispatch_count=_Getter(_CounterState()),
        ),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_lm.models", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.kimi_k3_fused_expert",
        _ReceiptState(),
    )
    with pytest.raises(subject.Width4ReceiptError, match="receipt/reset"):
        _begin(_FakeMlxModule(str(core)))


def test_generate_terminal_receipt_order_is_promotion_safe() -> None:
    source_path = (
        Path(__file__).parents[1] / "src/exo/worker/engines/mlx/generator/generate.py"
    )
    source = source_path.read_text()
    assert '"width-four packed agreement attestation"' in source
    callback = source.index('"generation progress callback"')
    barrier = source.index(
        "if is_done and dspark_runtime is not None and not is_pipeline:"
    )
    confidence = source.index('"confidence capture finalization"', barrier)
    attestation = source.index(
        "dspark_runtime.log_packed_agreement_attestation(", confidence
    )
    rank_agreement = source.index(
        '"width-four packed agreement attestation"', confidence
    )
    required = source.index("required=True", rank_agreement)
    capture = source.index(
        "if is_done and width4_receipt_context is not None:", attestation
    )
    capture_rank_agreement = source.index(
        '"width-four terminal receipt capture contract"', capture
    )
    capture_diagnostic = source.index(
        "preserve_unanimous_error=True", capture_rank_agreement
    )
    log = source.index("logger.info(format_width4_dispatch_receipt(receipt))", capture)
    flush = source.index("logger.complete()", log)
    yield_response = source.index("yield response", capture)
    assert callback < barrier < confidence < rank_agreement < attestation
    assert attestation < required < capture
    assert capture < capture_rank_agreement < capture_diagnostic < log
    assert log < flush < yield_response


def test_generate_resets_after_warmup_and_immediately_before_prefill() -> None:
    source_path = (
        Path(__file__).parents[1] / "src/exo/worker/engines/mlx/generator/generate.py"
    )
    source = source_path.read_text()
    warmup_call = source.index('width4_receipt_scope="warmup"')
    warmup_barrier = source.index("mx_barrier(group)", warmup_call)
    post_warmup_reset = source.index("post_warmup_reset =", warmup_barrier)
    begin = source.index("def begin_receipt_contract()")
    reset_rank_agreement = source.index(
        '"width-four request receipt reset contract"', begin
    )
    reset_diagnostic = source.index(
        "preserve_unanimous_error=True", reset_rank_agreement
    )
    prefill = source.index(
        "prefill_tps, prefill_tokens = dspark_runtime.seed_prompt", begin
    )
    assert warmup_call < warmup_barrier < post_warmup_reset
    assert begin < reset_rank_agreement < reset_diagnostic < prefill
    assert source.count("preserve_unanimous_error=True") == 2
