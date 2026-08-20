from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest

_THIS_FILE = Path(__file__).resolve()


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _install_headless_receipt_imports() -> None:
    """Install request-test doubles without loading Metal or MLX-LM models."""

    os.environ.setdefault("EXO_DASHBOARD_DIR", ".")
    os.environ.setdefault("EXO_RESOURCES_DIR", ".")

    class _Array:
        pass

    class _Module:
        pass

    class _Logger:
        def __getattr__(self, _name: str) -> Any:
            return lambda *_args, **_kwargs: None

    def noop(*_args: object, **_kwargs: object) -> None:
        return None

    def identity(value: object, *_args: object, **_kwargs: object) -> object:
        return value

    mx = _module(
        "mlx.core",
        array=_Array,
        eval=noop,
        async_eval=noop,
        new_stream=lambda *_args: object(),
        default_device=lambda: object(),
        distributed=SimpleNamespace(Group=object, all_gather=noop),
        random=SimpleNamespace(seed=noop),
        float32=object(),
        int32=object(),
    )
    for name in (
        "argmax",
        "argpartition",
        "argsort",
        "clear_cache",
        "concatenate",
        "get_active_memory",
        "get_cache_memory",
        "get_peak_memory",
        "max",
        "reset_peak_memory",
        "stream",
        "synchronize",
        "where",
        "zeros_like",
    ):
        setattr(mx, name, noop)
    nn = _module("mlx.nn", Module=_Module)
    mlx = _module("mlx", core=mx, nn=nn)
    mlx.__path__ = []  # type: ignore[attr-defined]
    sys.modules.update({"mlx": mlx, "mlx.core": mx, "mlx.nn": nn})

    bootstrap = _module("exo.worker.runner.bootstrap", logger=_Logger())
    sys.modules[bootstrap.__name__] = bootstrap

    width4 = _module(
        "exo.worker.engines.mlx.generator.kimi_k3_width4_receipt",
        Width4RequestReceiptClaim=type("Width4RequestReceiptClaim", (), {}),
        Width4RequestReceiptContext=type("Width4RequestReceiptContext", (), {}),
        begin_width4_request_receipt=noop,
        capture_width4_dispatch_receipt=noop,
        format_width4_dispatch_receipt=noop,
        mark_width4_receipt_rank_agreed=noop,
        reset_width4_dispatch_counters_after_warmup=noop,
        validate_width4_request_contract=noop,
        width4_receipt_agreement_contract=noop,
        width4_receipt_log_enabled=lambda: False,
    )
    sys.modules[width4.__name__] = width4

    # This module owns the real causal event and attestation dataclasses used by
    # the receipt. It imports only the doubles above in this headless process.
    __import__("exo.worker.engines.mlx.generator.kimi_k3_dspark")

    auto_parallel = _module(
        "exo.worker.engines.mlx.auto_parallel",
        PipelineFirstLayer=type("PipelineFirstLayer", (), {}),
        PipelineLastLayer=type("PipelineLastLayer", (), {}),
    )
    for name in (
        "clear_prefill_sends",
        "discard_unsent_prefill_sends_after_agreed_cancel",
        "flush_prefill_sends",
        "get_active_relay_context",
        "relay_sampled_tokens",
        "set_pipeline_prefill",
        "set_pipeline_queue_sends",
        "set_pipeline_token_relay",
    ):
        setattr(auto_parallel, name, noop)
    sys.modules[auto_parallel.__name__] = auto_parallel

    cache = _module(
        "exo.worker.engines.mlx.cache",
        CacheSnapshot=type("CacheSnapshot", (), {}),
        KVPrefixCache=type("KVPrefixCache", (), {}),
    )
    for name in (
        "copy_snapshot_entry",
        "encode_prompt",
        "has_non_kv_caches",
        "is_non_trimmable_cache_entry",
        "make_kv_cache",
        "snapshot_ssm_states",
    ):
        setattr(cache, name, noop)
    sys.modules[cache.__name__] = cache

    remote_prefill_module = _module(
        "exo.worker.engines.mlx.generator.remote_prefill",
        remote_prefill=noop,
    )
    sys.modules[remote_prefill_module.__name__] = remote_prefill_module

    mlx_types = _module(
        "exo.worker.engines.mlx.types",
        KVCacheType=list,
        Model=type("Model", (), {}),
    )
    sys.modules[mlx_types.__name__] = mlx_types

    utils = _module("exo.worker.engines.mlx.utils_mlx")
    for name in (
        "apply_chat_template",
        "fix_unmatched_think_end_tokens",
        "mx_barrier",
        "mx_ranks_agree_on_value",
        "rank_agreed_local_stage",
        "system_prompt_token_count",
    ):
        setattr(utils, name, noop)
    sys.modules[utils.__name__] = utils

    vision = _module(
        "exo.worker.engines.mlx.vision",
        MediaRegion=type("MediaRegion", (), {}),
        VisionProcessor=type("VisionProcessor", (), {}),
        VisionResult=type("VisionResult", (), {}),
        get_inner_model=identity,
        prepare_vision=noop,
    )
    sys.modules[vision.__name__] = vision

    mlx_lm = _module("mlx_lm")
    mlx_lm.__path__ = []  # type: ignore[attr-defined]
    mlx_lm_generate = _module(
        "mlx_lm.generate",
        GenerationResponse=type("GenerationResponse", (), {}),
        maybe_quantize_kv_cache=noop,
        stream_generate=noop,
    )
    sample_utils = _module(
        "mlx_lm.sample_utils",
        make_logits_processors=noop,
        make_sampler=noop,
    )
    tokenizer_utils = _module(
        "mlx_lm.tokenizer_utils",
        TokenizerWrapper=type("TokenizerWrapper", (), {}),
    )
    sys.modules.update(
        {
            "mlx_lm": mlx_lm,
            "mlx_lm.generate": mlx_lm_generate,
            "mlx_lm.sample_utils": sample_utils,
            "mlx_lm.tokenizer_utils": tokenizer_utils,
        }
    )


def _expect_error(
    error_type: type[BaseException],
    match: str,
    operation: Callable[[], object],
) -> None:
    with pytest.raises(error_type, match=match):
        operation()


def _reset_phase_state(generate_module: ModuleType) -> None:
    generate_module._COMPOSITION_RECEIPT_ATTEMPTED_PHASES.clear()
    generate_module._COMPOSITION_RECEIPT_PUBLISHED_PHASES.clear()
    generate_module._COMPOSITION_RECEIPT_COMPLETED_PHASES.clear()
    generate_module._COMPOSITION_RECEIPT_PHASE_IDENTITIES.clear()
    generate_module._COMPOSITION_RECEIPT_ACTIVE_PHASE = None


def _refresh_launch_sha(generate_module: ModuleType) -> str:
    prefixes = ("EXO_MLX_", "MLX_LM_", "MLX_METAL_K3_", "MLX_JACCL_")
    nonprefixed_common = {
        "EXO_ADVERTISED_MODEL_IDS",
        "EXO_NO_BATCH",
        "EXO_OFFLINE",
        "MLX_METAL_FAST_SYNCH",
    }
    deployment = {
        "EXO_MLX_KIMI_K3_DSPARK_CHECKPOINT",
        "EXO_MLX_KIMI_K3_DSPARK_CONFIDENCE_JSONL",
        "EXO_MLX_KIMI_K3_DSPARK_CONFIDENCE_SESSION",
        "EXO_MLX_KIMI_K3_DSPARK_YARN_CHECKPOINT",
        "EXO_MLX_RANK_LOCAL_CHECKPOINT",
        "EXO_MLX_RANK_LOCAL_LOADER",
        "MLX_JACCL_COORDINATOR",
        "MLX_JACCL_RING",
        "MLX_LM_ROOT",
    }
    launch_map = {
        name: value
        for name, value in os.environ.items()
        if (name.startswith(prefixes) or name in nonprefixed_common)
        and name not in deployment
        and name != generate_module._LAUNCH_CONTRACT_SHA256_ENV
    }
    canonical = json.dumps(
        launch_map,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    expected = hashlib.sha256(canonical).hexdigest()
    os.environ[generate_module._LAUNCH_CONTRACT_SHA256_ENV] = expected
    return expected


def _configure_selector_environment(
    generate_module: ModuleType,
    *,
    packed: int,
    kda: int,
    deferred: int,
) -> None:
    prefixes = ("EXO_MLX_", "MLX_LM_", "MLX_METAL_K3_", "MLX_JACCL_")
    nonprefixed_common = {
        "EXO_ADVERTISED_MODEL_IDS",
        "EXO_NO_BATCH",
        "EXO_OFFLINE",
        "MLX_METAL_FAST_SYNCH",
    }
    for name in tuple(os.environ):
        if name.startswith(prefixes) or name in nonprefixed_common:
            os.environ.pop(name)
    values = {
        generate_module._PACKED_FRONT_DIAGNOSTIC_ENV: "1",
        generate_module._PACKED_FRONT_EXPECTED_LAYERS_ENV: "92",
        generate_module._KDA_EXPECTED_LAYERS_ENV: "69",
        generate_module._DEFERRED_EXPECTED_ROOTS_ENV: "12",
        generate_module._EXPECTED_LIBMLX_SHA256_ENV: (
            generate_module._SEALED_LIBMLX_SHA256
        ),
        generate_module._MLX_PACKED_FRONT_RECEIPT_ENV: "1",
        generate_module._MLX_AUTHORITATIVE_PACKED_FRONT_ENV: str(packed),
        generate_module._MLX_AUTHORITATIVE_PACKED_FRONT_WIDTH3_ENV: str(packed),
        generate_module._MLX_KDA_PREWORK_ENV: str(kda),
        generate_module._EXO_DEFERRED_WIDTH3_ENV: str(deferred),
        generate_module._MLX_DEFERRED_WIDTH3_ENV: str(deferred),
        generate_module._MLX_DUPLICATING_PACKED_FRONT_ENV: "0",
        generate_module._MLX_PACKED_FRONT_WIDTH8_ENV: "0",
        generate_module._MLX_MULTIBANK_PACKED_FRONT_ENV: "0",
        generate_module._MLX_COMPILED_DECODE_ENV: "0",
        generate_module._MLX_NATIVE_AFFINE8_Q3_TRIPLET_ENV: "1",
        generate_module._MLX_NATIVE_AFFINE8_Q3_RECEIPT_ENV: "1",
        generate_module._EXO_TAIL_OVERLAP_ENV: "1",
        generate_module._WIDTH4_RECEIPT_ENV: "0",
        "EXO_NO_BATCH": "1",
        "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE": "1",
        generate_module._MLX_BATCHED_REPLAYSSM_COMMIT_ENV: "1",
        generate_module._MLX_BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "69",
        generate_module._MLX_FULL_ACCEPT_IDENTITY_COMMIT_ENV: "0",
        "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE": "1",
        generate_module._MLX_PROJECTED_KV_CACHE_MAX_TOKENS_ENV: "32768",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES": "laguna8",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE": "hidden",
    }
    os.environ.update(values)
    _refresh_launch_sha(generate_module)


class _FakeRuntime:
    def __init__(self, *, target_cache_offset: int, deferred: object) -> None:
        self.target_cache_offset = target_cache_offset
        self.deferred_async_width3_attestation = deferred
        self.attached: object | None = None
        self.agreement_failure: BaseException | None = None
        self.side_effect_calls: list[str] = []
        self.side_effect_failure: str | None = None
        self.collective = SimpleNamespace(rank=0)

    def agree_text(self, _name: str, operation: Callable[[], str]) -> str:
        if self.agreement_failure is not None:
            raise self.agreement_failure
        return operation()

    def attach_composition_receipt_sink(self, sink: object | None) -> None:
        self.attached = sink

    def agree_local_side_effect(
        self,
        name: str,
        operation: Callable[[], None],
    ) -> None:
        self.side_effect_calls.append(name)
        if self.side_effect_failure == name:
            raise RuntimeError(f"injected {name} failure")
        operation()


def _raw_receipt(
    generate_module: ModuleType,
    *,
    request: object,
    full_rounds: int,
    tail_rounds: int,
    prefill_width1: int,
    prefill_width3: int,
    prefill_noncontract: int,
) -> dict[str, object]:
    selectors = request.selectors
    counters = {name: 0 for name in generate_module._PACKED_FRONT_COUNTER_FIELDS}
    width1 = (tail_rounds + prefill_width1) * 92
    width3 = (full_rounds + prefill_width3) * 92
    noncontract = prefill_noncontract * 92
    if selectors.packed:
        counters.update(
            {
                "helper_calls": width1 + width3 + noncontract,
                "eligible_width1_calls": width1,
                "eligible_width3_calls": width3,
                "packed_width1_hits": width1,
                "packed_width3_hits": width3,
                "packed_hits": width1 + width3,
                "packed_width1_output_tensors": width1 * 4,
                "packed_width3_output_tensors": width3 * 4,
                "packed_output_tensors": (width1 + width3) * 4,
                "noncontract_calls": noncontract,
            }
        )
        if request.phase == generate_module._PACKED_FRONT_PHASE_ENGINE_STARTUP:
            counters.update(
                {
                    "packed_width3_installs": 92,
                    "lazy_installs": 92,
                    "pack_count_before": 0,
                    "pack_count_after": 92,
                }
            )
        else:
            counters.update({"pack_count_before": 92, "pack_count_after": 92})
    else:
        helper_calls = (
            full_rounds
            + tail_rounds
            + prefill_width1
            + prefill_width3
            + prefill_noncontract
        ) * 92
        counters.update(
            {"helper_calls": helper_calls, "gate_disabled_calls": helper_calls}
        )

    kda_helpers = (full_rounds + prefill_width3 + prefill_noncontract) * 69
    counters["kda_helper_calls"] = kda_helpers
    if selectors.kda:
        kda_successes = full_rounds * 69
        counters.update(
            {
                "kda_noncontract_calls": (prefill_width3 + prefill_noncontract) * 69,
                "kda_admitted_calls": kda_successes,
                "kda_success_calls": kda_successes,
            }
        )
    else:
        counters["kda_gate_disabled_calls"] = kda_helpers

    return {
        "schema": generate_module._PACKED_FRONT_RECEIPT_SCHEMA,
        "request_sequence": request._handle[0],
        "request_token": request._handle[1],
        "expected_sparse_layers": 92,
        "expected_kda_layers": 69,
        "finalized": True,
        "aborted": False,
        "poisoned": False,
        "packed_authoritative_enabled": selectors.packed,
        "packed_width3_enabled": selectors.packed,
        "kda_prework_enabled": selectors.kda,
        "replayssm_speculative_enabled": True,
        "projected_kv_cache_enabled": True,
        "projected_kv_cache_max_tokens": 32768,
        "async_decode_boundaries": "laguna8",
        "async_decode_state": "hidden",
        "async_decode_width3_enabled": selectors.deferred,
        "native_q3_triplet_enabled": True,
        "native_q3_dispatch_receipt_enabled": True,
        **counters,
    }


def _replayssm_telemetry_payload(
    generate_module: ModuleType,
    *,
    revision: int,
    counters: dict[str, int],
) -> dict[str, object]:
    return {
        "schema": generate_module._BATCHED_REPLAYSSM_TELEMETRY_SCHEMA,
        "revision": revision,
        "counters": dict(counters),
        "latest_attestation": None,
    }


def _build_fixture(
    generate_module: ModuleType,
    *,
    arm_code: int = 7,
    phase: int = 2,
    full_rounds: int = 2,
    tail_rounds: int = 0,
    prefill_width1: int = 0,
    prefill_width3: int = 0,
    prefill_noncontract: int = 0,
    accepted: list[int] | None = None,
    identity_enabled: bool = False,
    visible_tokens: int | None = None,
    submit_final: bool = False,
) -> SimpleNamespace:
    from exo.worker.engines.mlx.generator import kimi_k3_dspark

    packed = bool(arm_code & 1)
    kda = bool(arm_code & 2)
    deferred_enabled = bool(arm_code & 4)
    selectors = generate_module._CompositionSelectors(
        packed=packed,
        kda=kda,
        deferred=deferred_enabled,
        identity_commit=identity_enabled,
        arm_code=arm_code,
        canonical=arm_code in {0, 7},
        digest=(1, 2, 3, 4),
        launch_digest=(5, 6, 7, 8),
    )
    native_counts = {"total": 0, "n4480": 0, "n6144": 0, "n10624": 0}
    native_api = generate_module._NativeQ3ReceiptAPI(
        library=object(),
        total=lambda: native_counts["total"],
        n4480=lambda: native_counts["n4480"],
        n6144=lambda: native_counts["n6144"],
        n10624=lambda: native_counts["n10624"],
        reset=lambda: native_counts.update(
            {"total": 0, "n4480": 0, "n6144": 0, "n10624": 0}
        ),
        lib_path=_THIS_FILE,
        lib_fd=-1,
        lib_identity=(),
        lib_digest=(17, 18, 19, 20),
    )
    replayssm_baseline_counters = {
        name: index + 3
        for index, name in enumerate(generate_module._BATCHED_REPLAYSSM_COUNTER_FIELDS)
    }
    replayssm_state: dict[str, object] = {
        "current": _replayssm_telemetry_payload(
            generate_module,
            revision=41,
            counters=replayssm_baseline_counters,
        )
    }
    api = generate_module._PackedFrontReceiptAPI(
        begin=lambda *_args, **_kwargs: (11, 29),
        finish=lambda *_args, **_kwargs: {},
        abort=lambda *_args: None,
        commit_telemetry=lambda: replayssm_state["current"],
        source_digest=(9, 10, 11, 12),
    )
    request = generate_module._PackedFrontReceiptRequest(
        model=object(),
        enabled=True,
        phase=phase,
        selectors=selectors,
        selector_digest=selectors.digest,
        launch_contract_digest=selectors.launch_digest,
        api=api,
        native_api=native_api,
        exo_source_digest=(13, 14, 15, 16),
        setup_digest=(21, 22, 23, 24),
    )
    request._handle = (11, 29)
    request._native_baseline = (0, 0, 0, 0)
    request._replayssm_baseline = (
        generate_module._validated_batched_replayssm_telemetry(
            replayssm_state["current"]
        )
    )
    request._assert_runtime_identity = lambda: None

    empty_attestation = kimi_k3_dspark.DeferredAsyncWidth3Attestation(
        enabled=False,
        validated_rounds=0,
        materialized_rounds=0,
        validated_roots=0,
        submitted_roots=0,
        first_initial_offset=None,
        last_initial_offset=None,
        last_final_offset=None,
    )
    begin_cache = 10
    runtime = _FakeRuntime(
        target_cache_offset=begin_cache,
        deferred=empty_attestation,
    )
    request._runtime = runtime
    request.begin_decode(runtime, anchor_token=100)
    telemetry = generate_module._PromptLookupTelemetry(composition_receipt=request)
    telemetry.prefill_width1_chunks = prefill_width1
    telemetry.prefill_width3_chunks = prefill_width3
    telemetry.prefill_noncontract_chunks = prefill_noncontract

    accepted_values = accepted if accepted is not None else [2] * full_rounds
    if len(accepted_values) != full_rounds:
        raise AssertionError("accepted schedule length differs from full rounds")
    pre_cache = begin_cache
    anchor = 100
    committed: list[int] = []
    root_batches: list[tuple[int, int, int]] = []
    prior_submission: tuple[int, int] | None = None
    for round_index in range(full_rounds):
        if prior_submission is not None:
            request.observe_causal_event(
                round_index,
                "tail_prelaunch_used",
                prior_submission,
            )
        request.observe_causal_event(
            round_index,
            "proposal_context_agreed",
            (pre_cache,),
        )
        proposal = (anchor, anchor + 1, anchor + 2)
        request.observe_causal_event(round_index, "proposal_agreed", proposal)
        if deferred_enabled:
            request.observe_causal_event(
                round_index,
                "deferred_roots_validated",
                (12,),
            )
        request.observe_causal_event(round_index, "target_graph_agreed", (2,))
        if deferred_enabled:
            for ordinal in range(12):
                request.observe_causal_event(
                    round_index,
                    "deferred_root_submitted",
                    (ordinal, 12),
                )
            batch = (pre_cache, pre_cache + 3, 12)
            request.observe_causal_event(
                round_index,
                "deferred_roots_submitted",
                batch,
            )
            root_batches.append(batch)
        accepted_count = accepted_values[round_index]
        request.observe_causal_event(
            round_index,
            "acceptance_agreed",
            (accepted_count,),
        )
        emitted = tuple(range(anchor + 1, anchor + accepted_count + 2))
        request.observe_causal_event(round_index, "committed_tokens", emitted)
        post_cache = pre_cache + len(emitted)
        should_submit = round_index < full_rounds - 1 or (
            submit_final and round_index == full_rounds - 1
        )
        if should_submit:
            prior_submission = (post_cache, emitted[-1])
            request.observe_causal_event(
                round_index,
                "tail_prelaunch_submitted",
                prior_submission,
            )
        else:
            prior_submission = None
        request.observe_causal_event(
            round_index,
            "target_commit",
            (len(emitted),),
        )
        stats = kimi_k3_dspark.DSparkRoundTelemetry(
            round_index=round_index,
            rank=0,
            draft_ms=0.0,
            target_verify_ms=0.0,
            target_commit_ms=0.0,
            draft_commit_ms=0.0,
            collective_ms=0.0,
            proposed=2,
            accepted=accepted_count,
            emitted=len(emitted),
            fallback=False,
            error=None,
            prelaunch_submitted=should_submit,
            prelaunch_used=round_index > 0,
        )
        runtime.target_cache_offset = post_cache
        telemetry.observe_dspark_round(stats, target_cache_tokens=post_cache)
        committed.extend(emitted)
        anchor = emitted[-1]
        pre_cache = post_cache

    for tail_index in range(tail_rounds):
        round_index = full_rounds + tail_index
        if prior_submission is not None:
            request.observe_causal_event(
                round_index,
                "tail_prelaunch_discarded",
                (1, *prior_submission),
            )
            prior_submission = None
        request.observe_causal_event(
            round_index,
            "ordinary_anchor_agreed",
            (anchor,),
        )
        emitted = (anchor + 1,)
        request.observe_causal_event(round_index, "committed_tokens", emitted)
        request.observe_causal_event(
            round_index,
            "ordinary_target_commit",
            (1,),
        )
        post_cache = pre_cache + 1
        stats = kimi_k3_dspark.DSparkRoundTelemetry(
            round_index=round_index,
            rank=0,
            draft_ms=0.0,
            target_verify_ms=0.0,
            target_commit_ms=0.0,
            draft_commit_ms=0.0,
            collective_ms=0.0,
            proposed=0,
            accepted=0,
            emitted=1,
            fallback=False,
            error=None,
        )
        runtime.target_cache_offset = post_cache
        telemetry.observe_dspark_round(stats, target_cache_tokens=post_cache)
        committed.extend(emitted)
        anchor = emitted[-1]
        pre_cache = post_cache

    if prior_submission is not None:
        request.observe_causal_event(
            full_rounds - 1,
            "tail_prelaunch_discarded",
            (2, *prior_submission),
        )

    if deferred_enabled:
        runtime.deferred_async_width3_attestation = (
            kimi_k3_dspark.DeferredAsyncWidth3Attestation(
                enabled=True,
                validated_rounds=full_rounds,
                materialized_rounds=full_rounds,
                validated_roots=full_rounds * 12,
                submitted_roots=full_rounds * 12,
                first_initial_offset=root_batches[0][0],
                last_initial_offset=root_batches[-1][0],
                last_final_offset=root_batches[-1][1],
            )
        )

    visible_count = len(committed) if visible_tokens is None else visible_tokens
    for index, token in enumerate(committed[:visible_count]):
        request.observe_output_token(token, from_draft=bool(index % 2))

    raw = _raw_receipt(
        generate_module,
        request=request,
        full_rounds=full_rounds,
        tail_rounds=tail_rounds,
        prefill_width1=prefill_width1,
        prefill_width3=prefill_width3,
        prefill_noncontract=prefill_noncontract,
    )
    q3_layer_calls = (full_rounds + prefill_width3) * 92
    native_counts.update(
        {
            "total": q3_layer_calls * (3 if packed else 6),
            "n4480": 0,
            "n6144": 0,
            "n10624": (full_rounds + prefill_width3) * 92 if packed else 0,
        }
    )
    full_accept_rounds = sum(int(value == 2) for value in accepted_values)
    identity_rounds = full_accept_rounds if identity_enabled else 0
    batched_rounds = full_rounds - identity_rounds
    replayssm_deltas = {
        "attempted_prepares": full_rounds,
        "batched_prepares": batched_rounds,
        "batched_commits": batched_rounds,
        "identity_prepares": identity_rounds,
        "identity_commits": identity_rounds,
        "fallback_prepares": 0,
        "fallback_commits": 0,
        "batched_errors": 0,
        "identity_errors": 0,
        "layers_batched": batched_rounds * 69,
        "layers_identity_committed": identity_rounds * 69,
    }
    replayssm_final_counters = {
        name: replayssm_baseline_counters[name] + replayssm_deltas[name]
        for name in generate_module._BATCHED_REPLAYSSM_COUNTER_FIELDS
    }
    replayssm_state["current"] = _replayssm_telemetry_payload(
        generate_module,
        revision=41 + full_rounds * 2,
        counters=replayssm_final_counters,
    )
    return SimpleNamespace(
        request=request,
        runtime=runtime,
        telemetry=telemetry,
        raw=raw,
        native_counts=native_counts,
        committed=tuple(committed),
        replayssm_state=replayssm_state,
        replayssm_baseline_counters=replayssm_baseline_counters,
        replayssm_deltas=replayssm_deltas,
    )


def _validate_fixture(generate_module: ModuleType, fixture: SimpleNamespace) -> object:
    from exo.api.types import K3W3CompositionReceipt

    marker = fixture.request._validated_marker(
        fixture.raw,
        fixture.telemetry,
        fixture.runtime,
    )
    receipt = K3W3CompositionReceipt.model_validate(
        {"schema": generate_module._PACKED_FRONT_MARKER_SCHEMA, **marker}
    )
    assert receipt.model_dump(exclude={"receipt_schema"}) == marker
    return receipt


def _scenario_selector_admission(generate_module: ModuleType) -> None:
    for arm_code in range(8):
        _configure_selector_environment(
            generate_module,
            packed=arm_code & 1,
            kda=(arm_code >> 1) & 1,
            deferred=(arm_code >> 2) & 1,
        )
        selectors = generate_module._packed_front_selector_contract(
            allow_noncanonical_source_test=True
        )
        assert selectors.arm_code == arm_code
        assert selectors.canonical is (arm_code in {0, 7})
        if arm_code in {0, 7}:
            assert generate_module._packed_front_selector_contract() == selectors
        else:
            _expect_error(
                ValueError,
                "canonical control",
                generate_module._packed_front_selector_contract,
            )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ[generate_module._MLX_AUTHORITATIVE_PACKED_FRONT_WIDTH3_ENV] = "0"
    _expect_error(
        ValueError,
        "jointly disabled or jointly enabled",
        generate_module._packed_front_selector_contract,
    )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ[generate_module._MLX_DEFERRED_WIDTH3_ENV] = "0"
    _expect_error(
        ValueError,
        "deferred width-three selectors must match",
        generate_module._packed_front_selector_contract,
    )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ["EXO_MLX_RANK_LOCAL_LOADER"] = "/rank0/loader.py"
    os.environ["EXO_MLX_RANK_LOCAL_CHECKPOINT"] = "/models/rank0"
    digest_rank0 = generate_module._packed_front_selector_contract().launch_digest
    os.environ["EXO_MLX_RANK_LOCAL_LOADER"] = "/rank1/loader.py"
    os.environ["EXO_MLX_RANK_LOCAL_CHECKPOINT"] = "/models/rank1"
    digest_rank1 = generate_module._packed_front_selector_contract().launch_digest
    assert digest_rank0 == digest_rank1

    # C1 is rank-common behavior: admit it only inside the authenticated map,
    # and prove an unrefreshed flip is rejected before a request can start.
    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ[generate_module._MLX_FULL_ACCEPT_IDENTITY_COMMIT_ENV] = "1"
    identity_enabled_sha = _refresh_launch_sha(generate_module)
    identity_enabled = generate_module._packed_front_selector_contract()
    assert identity_enabled.arm_code == 7
    assert identity_enabled.identity_commit is True
    os.environ[generate_module._MLX_FULL_ACCEPT_IDENTITY_COMMIT_ENV] = "0"
    _expect_error(
        ValueError,
        "launch-contract SHA",
        generate_module._packed_front_selector_contract,
    )
    identity_disabled_sha = _refresh_launch_sha(generate_module)
    assert identity_disabled_sha != identity_enabled_sha
    identity_disabled = generate_module._packed_front_selector_contract()
    assert identity_disabled.arm_code == 7
    assert identity_disabled.identity_commit is False

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ[generate_module._MLX_FULL_ACCEPT_IDENTITY_COMMIT_ENV] = "true"
    _refresh_launch_sha(generate_module)
    _expect_error(
        ValueError,
        "must be exactly 0 or 1",
        generate_module._packed_front_selector_contract,
    )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ[generate_module._MLX_BATCHED_REPLAYSSM_COMMIT_ENV] = "0"
    _refresh_launch_sha(generate_module)
    _expect_error(
        ValueError,
        "must be exactly 1",
        generate_module._packed_front_selector_contract,
    )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ[generate_module._MLX_BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV] = "68"
    _refresh_launch_sha(generate_module)
    _expect_error(
        ValueError,
        "must be exactly 69",
        generate_module._packed_front_selector_contract,
    )

    # Keep every non-candidate selector from the accepted 22.517 tok/s W3
    # control inside the authenticated launch map.  These selectors are
    # symmetric across protected arms; admitting them is not permission to
    # vary them.  The historical control launcher omitted the auxiliary Q4
    # metallib variable, so cover both its absent and explicit-zero forms.
    production_companions = {
        "MLX_LM_KIMI_K3_BATCHED_REPLAYSSM_COMMIT": "1",
        "MLX_LM_KIMI_K3_BATCHED_REPLAYSSM_EXPECTED_LAYERS": "69",
        "MLX_LM_KIMI_K3_DUALSOURCE_AFFINE2_PREFILL": "0",
        "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH3": "1",
        "MLX_LM_KIMI_K3_MOK_PREFILL_OVERLAP": "1",
        "MLX_LM_KIMI_K3_MOK_ROUTED_SHARED_OVERLAP": "1",
        "MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE": "1",
        "MLX_LM_KIMI_K3_TP2_ABSORBED_VERIFY": "0",
        "MLX_LM_KIMI_K3_TP2_ABSORBED_VERIFY_MAX_Q": "3",
        "MLX_METAL_K3_AFFINE2_EXPERT_TASKS": "1",
        "MLX_METAL_K3_AFFINE2_GATHER_BM8": "1",
        "MLX_METAL_K3_AFFINE6_Q3_TRIPLET": "0",
        "MLX_METAL_K3_AFFINE6_Q3_TRIPLET_V2": "1",
    }
    _configure_selector_environment(generate_module, packed=0, kda=0, deferred=0)
    os.environ.update(production_companions)
    absent_aux_digest = _refresh_launch_sha(generate_module)
    selectors = generate_module._packed_front_selector_contract()
    assert selectors.arm_code == 0
    os.environ["MLX_METAL_K3_AFFINE6_Q4_AUX_METALLIB"] = "0"
    explicit_aux_digest = _refresh_launch_sha(generate_module)
    assert explicit_aux_digest != absent_aux_digest
    assert generate_module._packed_front_selector_contract().arm_code == 0
    os.environ["MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH3"] = "0"
    _expect_error(
        ValueError,
        "launch-contract SHA",
        generate_module._packed_front_selector_contract,
    )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ["EXO_MLX_PREFILL_STEP_SIZE"] = "2048"
    _expect_error(
        ValueError,
        "launch-contract SHA",
        generate_module._packed_front_selector_contract,
    )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ[generate_module._MLX_PROJECTED_KV_CACHE_MAX_TOKENS_ENV] = "16384"
    _refresh_launch_sha(generate_module)
    _expect_error(
        ValueError,
        "must be exactly 32768",
        generate_module._packed_front_selector_contract,
    )

    _configure_selector_environment(generate_module, packed=1, kda=1, deferred=1)
    os.environ["MLX_LM_KIMI_K3_UNCLASSIFIED_EXPERIMENT"] = "1"
    _expect_error(
        ValueError,
        "unclassified experiment keys",
        generate_module._packed_front_selector_contract,
    )


def _scenario_strict_off(generate_module: ModuleType) -> None:
    from exo.worker.engines.mlx.generator import kimi_k3_dspark

    prefixes = ("EXO_MLX_", "MLX_LM_", "MLX_METAL_K3_", "MLX_JACCL_")
    for name in tuple(os.environ):
        if name.startswith(prefixes):
            os.environ.pop(name)
    _reset_phase_state(generate_module)
    request = generate_module._PackedFrontReceiptRequest.from_environment(
        object(),
        None,
        phase="api",
    )
    assert request.enabled is False
    assert not generate_module._COMPOSITION_RECEIPT_ATTEMPTED_PHASES

    os.environ[generate_module._MLX_PACKED_FRONT_RECEIPT_ENV] = "1"
    _expect_error(
        ValueError,
        "requires .*COMPOSITION_RECEIPT=1",
        lambda: generate_module._PackedFrontReceiptRequest.from_environment(
            object(), None, phase="api"
        ),
    )
    os.environ.pop(generate_module._MLX_PACKED_FRONT_RECEIPT_ENV)
    os.environ[generate_module._EXPECTED_LIBMLX_SHA256_ENV] = (
        generate_module._SEALED_LIBMLX_SHA256
    )
    _expect_error(
        ValueError,
        "requires .*COMPOSITION_RECEIPT=1",
        lambda: generate_module._PackedFrontReceiptRequest.from_environment(
            object(), None, phase="api"
        ),
    )

    os.environ.pop(generate_module._EXPECTED_LIBMLX_SHA256_ENV)
    os.environ["MLX_LM_KIMI_K3_FUSED_ROUTER"] = "1"
    request = generate_module._PackedFrontReceiptRequest.from_environment(
        object(), None, phase="api"
    )
    assert request.enabled is False

    repeated_root = SimpleNamespace(
        shape=(
            1,
            kimi_k3_dspark.DSPARK_CONSERVATIVE_VERIFY_WIDTH,
            kimi_k3_dspark.KIMI_K3_TARGET_HIDDEN_SIZE,
        )
    )
    forward = SimpleNamespace(
        deferred_async_decode_states=(repeated_root,)
        * kimi_k3_dspark.KIMI_K3_DEFERRED_ASYNC_WIDTH3_BOUNDARY_COUNT
    )
    roots = kimi_k3_dspark._validated_deferred_async_decode_states(
        forward,
        expected_width=kimi_k3_dspark.DSPARK_CONSERVATIVE_VERIFY_WIDTH,
        enabled=True,
    )
    assert len(roots) == kimi_k3_dspark.KIMI_K3_DEFERRED_ASYNC_WIDTH3_BOUNDARY_COUNT
    _expect_error(
        ValueError,
        "must be 12 distinct roots",
        lambda: kimi_k3_dspark._validated_deferred_async_decode_states(
            forward,
            expected_width=kimi_k3_dspark.DSPARK_CONSERVATIVE_VERIFY_WIDTH,
            enabled=True,
            require_distinct=True,
        ),
    )
    submitted: list[object] = []
    kimi_k3_dspark._submit_deferred_async_decode_states(
        roots,
        submitted.append,
    )
    assert submitted == list(roots)

    runtime_source = inspect.getsource(kimi_k3_dspark.KimiK3DSparkRequestRuntime)
    assert runtime_source.count("on_deferred_root_submitted=(") == 2
    assert "if self._composition_receipt_sink is not None" in runtime_source
    assert (
        "on_deferred_root_submitted=self._record_deferred_root_submitted"
        not in runtime_source
    )
    round_source = inspect.getsource(kimi_k3_dspark.KimiK3DSparkRoundEngine)
    assert "if self.composition_receipt_sink is not None:" in round_source


def _scenario_lifecycle(generate_module: ModuleType) -> None:
    startup = generate_module._PACKED_FRONT_PHASE_ENGINE_STARTUP
    api = generate_module._PACKED_FRONT_PHASE_API
    identity = tuple(range(20))

    _reset_phase_state(generate_module)
    _expect_error(
        RuntimeError,
        "requires a published startup receipt",
        lambda: generate_module._claim_composition_receipt_phase(api),
    )
    generate_module._claim_composition_receipt_phase(startup)
    _expect_error(
        RuntimeError,
        "already active",
        lambda: generate_module._claim_composition_receipt_phase(api),
    )
    generate_module._release_composition_receipt_phase(startup)
    _expect_error(
        RuntimeError,
        "one-shot",
        lambda: generate_module._claim_composition_receipt_phase(startup),
    )
    _expect_error(
        RuntimeError,
        "requires a published startup receipt",
        lambda: generate_module._claim_composition_receipt_phase(api),
    )

    _reset_phase_state(generate_module)
    generate_module._claim_composition_receipt_phase(startup)
    generate_module._record_composition_receipt_publication(startup, identity)
    assert startup in generate_module._COMPOSITION_RECEIPT_PUBLISHED_PHASES
    assert startup not in generate_module._COMPOSITION_RECEIPT_COMPLETED_PHASES
    _expect_error(
        RuntimeError,
        "requires a published startup receipt",
        lambda: generate_module._claim_composition_receipt_phase(api),
    )
    generate_module._complete_composition_startup_after_warmup()
    generate_module._claim_composition_receipt_phase(api)
    generate_module._require_composition_api_identity(identity)
    _expect_error(
        RuntimeError,
        "identity differs",
        lambda: generate_module._require_composition_api_identity(
            identity[:-1] + (99,)
        ),
    )
    generate_module._record_composition_receipt_publication(api, identity)
    assert api in generate_module._COMPOSITION_RECEIPT_COMPLETED_PHASES
    _expect_error(
        RuntimeError,
        "one-shot",
        lambda: generate_module._claim_composition_receipt_phase(api),
    )

    _reset_phase_state(generate_module)
    start = threading.Barrier(3)
    outcomes: list[str] = []
    protected_work: list[int] = []

    def concurrent_claim() -> None:
        start.wait()
        try:
            generate_module._claim_composition_receipt_phase(startup)
        except RuntimeError:
            outcomes.append("rejected")
        else:
            protected_work.append(1)
            outcomes.append("admitted")

    threads = [threading.Thread(target=concurrent_claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["admitted", "rejected"]
    assert protected_work == [1]
    generate_module._release_composition_receipt_phase(startup)

    startup_fixture = _build_fixture(generate_module, phase=startup)
    api_fixture = _build_fixture(generate_module, phase=api)
    original_native = api_fixture.request.native_api
    api_fixture.request.native_api = generate_module._NativeQ3ReceiptAPI(
        library=original_native.library,
        total=original_native.total,
        n4480=original_native.n4480,
        n6144=original_native.n6144,
        n10624=original_native.n10624,
        reset=original_native.reset,
        lib_path=original_native.lib_path,
        lib_fd=original_native.lib_fd,
        lib_identity=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        lib_digest=original_native.lib_digest,
    )
    startup_identity = startup_fixture.request._phase_identity()
    api_identity = api_fixture.request._phase_identity()
    assert startup_identity != api_identity
    _reset_phase_state(generate_module)
    generate_module._claim_composition_receipt_phase(startup)
    generate_module._record_composition_receipt_publication(startup, startup_identity)
    generate_module._complete_composition_startup_after_warmup()
    _expect_error(
        RuntimeError,
        "identity differs",
        lambda: generate_module._require_composition_api_identity(api_identity),
    )

    warmup_source = inspect.getsource(generate_module.warmup_inference)
    validation_index = warmup_source.index('"inference warmup validation"')
    terminal_barrier_index = warmup_source.index("mx_barrier(group)", validation_index)
    completion_index = warmup_source.index(
        "_complete_composition_startup_after_warmup,"
    )
    assert validation_index < terminal_barrier_index < completion_index
    assert warmup_source.count("_complete_composition_startup_after_warmup,") == 1


def _scenario_schema_and_default_wire(generate_module: ModuleType) -> None:
    from exo.api.types import GenerationStats, K3W3CompositionReceipt
    from exo.shared.types.memory import Memory

    stats = GenerationStats(
        prompt_tps=1.0,
        generation_tps=2.0,
        prompt_tokens=3,
        generation_tokens=4,
        peak_memory_usage=Memory.from_bytes(5),
    )
    assert "k3_w3_composition_receipt" not in stats.model_dump()
    assert "k3_w3_composition_receipt" not in stats.model_dump_json()

    fixture = _build_fixture(generate_module, visible_tokens=2)
    receipt = _validate_fixture(generate_module, fixture)
    assert isinstance(receipt, K3W3CompositionReceipt)
    assert receipt.receipt_schema == generate_module._PACKED_FRONT_MARKER_SCHEMA
    assert receipt.receipt_schema_version == 2
    assert set(fixture.raw) == set(generate_module._PACKED_FRONT_MLX_KEYS)
    assert type(fixture.raw["projected_kv_cache_max_tokens"]) is int
    assert fixture.raw["projected_kv_cache_max_tokens"] == 32768
    marker = receipt.model_dump(exclude={"receipt_schema"})
    assert marker["projected_kv_cache_max_tokens"] == 32768
    assert all(type(value) in {bool, int} for value in marker.values())
    assert marker["kda_pending_calls"] == 0
    assert marker["visible_output_tokens"] == 2
    assert marker["emitted_tokens"] == len(fixture.committed)
    assert marker["identity_commit_enabled"] is False
    assert marker["speculative_full_accept_rounds"] == 2
    assert marker["speculative_partial_accept_rounds"] == 0
    assert marker["replayssm_batched_commits_delta"] == 2
    assert marker["replayssm_identity_commits_delta"] == 0
    assert marker["replayssm_telemetry_revision_delta"] == 4
    assert tuple(
        marker[f"visible_output_digest_word_{index}"] for index in range(4)
    ) != tuple(marker[f"committed_output_digest_word_{index}"] for index in range(4))
    assert all(
        forbidden not in name
        for name in marker
        for forbidden in (
            "timing",
            "hostname",
            "prompt_text",
            "raw_token",
            "attestation",
        )
    )

    wrong_type = _build_fixture(generate_module)
    wrong_type.raw["projected_kv_cache_max_tokens"] = "32768"
    _expect_error(
        ValueError,
        "selector snapshot projected_kv_cache_max_tokens drifted",
        lambda: _validate_fixture(generate_module, wrong_type),
    )
    wrong_fields = _build_fixture(generate_module)
    wrong_fields.raw["unexpected"] = 1
    _expect_error(
        ValueError,
        "receipt fields are invalid",
        lambda: _validate_fixture(generate_module, wrong_fields),
    )

    bool_layer_count = _build_fixture(generate_module)
    bool_layer_count.raw["expected_sparse_layers"] = True
    _expect_error(
        ValueError,
        "receipt header is invalid",
        lambda: _validate_fixture(generate_module, bool_layer_count),
    )

    bool_sequence = _build_fixture(generate_module)
    bool_sequence.request._handle = (1, 29)
    bool_sequence.raw["request_sequence"] = True
    _expect_error(
        ValueError,
        "receipt binding is invalid",
        lambda: _validate_fixture(generate_module, bool_sequence),
    )

    bool_token = _build_fixture(generate_module)
    bool_token.request._handle = (11, 1)
    bool_token.raw["request_token"] = True
    _expect_error(
        ValueError,
        "receipt binding is invalid",
        lambda: _validate_fixture(generate_module, bool_token),
    )


def _scenario_candidate_and_control(generate_module: ModuleType) -> None:
    for arm_code in (0, 7):
        fixture = _build_fixture(generate_module, arm_code=arm_code)
        receipt = _validate_fixture(generate_module, fixture)
        marker = receipt.model_dump(exclude={"receipt_schema"})
        assert marker["arm_code"] == arm_code
        assert marker["canonical_arm"] is (arm_code in {0, 7})
        assert marker["packed_enabled"] is bool(arm_code & 1)
        assert marker["kda_enabled"] is bool(arm_code & 2)
        assert marker["deferred_enabled"] is bool(arm_code & 4)

    subset = _build_fixture(generate_module, arm_code=3)
    _expect_error(
        ValueError,
        "requires canonical control or candidate",
        lambda: _validate_fixture(generate_module, subset),
    )

    startup = _build_fixture(
        generate_module,
        arm_code=7,
        phase=generate_module._PACKED_FRONT_PHASE_ENGINE_STARTUP,
    )
    marker = _validate_fixture(generate_module, startup).model_dump(
        exclude={"receipt_schema"}
    )
    assert marker["pack_count_before"] == 0
    assert marker["pack_count_after"] == 92
    assert marker["lazy_installs"] == 92


def _scenario_identity_commit_telemetry(generate_module: ModuleType) -> None:
    selector_off = _build_fixture(
        generate_module,
        full_rounds=3,
        accepted=[2, 1, 0],
        identity_enabled=False,
    )
    off_marker = _validate_fixture(generate_module, selector_off).model_dump(
        exclude={"receipt_schema"}
    )
    assert off_marker["identity_commit_enabled"] is False
    assert off_marker["speculative_full_accept_rounds"] == 1
    assert off_marker["speculative_partial_accept_rounds"] == 2
    assert off_marker["replayssm_attempted_prepares_delta"] == 3
    assert off_marker["replayssm_batched_prepares_delta"] == 3
    assert off_marker["replayssm_batched_commits_delta"] == 3
    assert off_marker["replayssm_identity_prepares_delta"] == 0
    assert off_marker["replayssm_identity_commits_delta"] == 0
    assert off_marker["replayssm_layers_batched_delta"] == 3 * 69
    assert off_marker["replayssm_layers_identity_committed_delta"] == 0

    selector_on = _build_fixture(
        generate_module,
        full_rounds=3,
        accepted=[2, 1, 2],
        identity_enabled=True,
    )
    on_payload = selector_on.replayssm_state["current"]
    assert type(on_payload) is dict
    on_payload["latest_attestation"] = {
        "status": "identity_committed",
        "raw_reference_commit": True,
    }
    on_marker = _validate_fixture(generate_module, selector_on).model_dump(
        exclude={"receipt_schema"}
    )
    assert on_marker["identity_commit_enabled"] is True
    assert on_marker["speculative_full_accept_rounds"] == 2
    assert on_marker["speculative_partial_accept_rounds"] == 1
    assert on_marker["replayssm_batched_prepares_delta"] == 1
    assert on_marker["replayssm_batched_commits_delta"] == 1
    assert on_marker["replayssm_identity_prepares_delta"] == 2
    assert on_marker["replayssm_identity_commits_delta"] == 2
    assert on_marker["replayssm_layers_batched_delta"] == 69
    assert on_marker["replayssm_layers_identity_committed_delta"] == 2 * 69
    assert on_marker["replayssm_fallback_prepares_delta"] == 0
    assert on_marker["replayssm_fallback_commits_delta"] == 0
    assert on_marker["replayssm_batched_errors_delta"] == 0
    assert on_marker["replayssm_identity_errors_delta"] == 0
    assert "latest_attestation" not in on_marker
    assert all(type(value) in {bool, int} for value in on_marker.values())

    wrong_schema = _build_fixture(generate_module)
    wrong_schema.replayssm_state["current"]["schema"] = "wrong"
    _expect_error(
        ValueError,
        "telemetry schema is invalid",
        lambda: _validate_fixture(generate_module, wrong_schema),
    )

    wrong_top_level = _build_fixture(generate_module)
    wrong_top_level.replayssm_state["current"]["extra"] = 1
    _expect_error(
        ValueError,
        "telemetry fields are invalid",
        lambda: _validate_fixture(generate_module, wrong_top_level),
    )

    wrong_counter_keys = _build_fixture(generate_module)
    wrong_counter_payload = wrong_counter_keys.replayssm_state["current"]
    assert type(wrong_counter_payload) is dict
    wrong_counters = wrong_counter_payload["counters"]
    assert type(wrong_counters) is dict
    wrong_counters.pop("identity_errors")
    _expect_error(
        ValueError,
        "counter fields are invalid",
        lambda: _validate_fixture(generate_module, wrong_counter_keys),
    )

    bool_counter = _build_fixture(generate_module)
    bool_payload = bool_counter.replayssm_state["current"]
    assert type(bool_payload) is dict
    bool_counters = bool_payload["counters"]
    assert type(bool_counters) is dict
    bool_counters["identity_errors"] = False
    _expect_error(
        ValueError,
        "counter identity_errors is invalid",
        lambda: _validate_fixture(generate_module, bool_counter),
    )

    decreased_counter = _build_fixture(generate_module)
    decreased_payload = decreased_counter.replayssm_state["current"]
    assert type(decreased_payload) is dict
    decreased_counters = decreased_payload["counters"]
    assert type(decreased_counters) is dict
    decreased_counters["identity_errors"] = (
        decreased_counter.replayssm_baseline_counters["identity_errors"] - 1
    )
    _expect_error(
        ValueError,
        "counter identity_errors decreased",
        lambda: _validate_fixture(generate_module, decreased_counter),
    )

    decreased_revision = _build_fixture(generate_module)
    decreased_revision.replayssm_state["current"]["revision"] = 40
    _expect_error(
        ValueError,
        "telemetry revision decreased",
        lambda: _validate_fixture(generate_module, decreased_revision),
    )

    wrong_identity_count = _build_fixture(
        generate_module,
        identity_enabled=True,
    )
    wrong_identity_payload = wrong_identity_count.replayssm_state["current"]
    assert type(wrong_identity_payload) is dict
    wrong_identity_counters = wrong_identity_payload["counters"]
    assert type(wrong_identity_counters) is dict
    wrong_identity_counters["identity_commits"] -= 1
    _expect_error(
        ValueError,
        "request-local counter algebra is invalid",
        lambda: _validate_fixture(generate_module, wrong_identity_count),
    )

    wrong_revision_delta = _build_fixture(generate_module)
    wrong_revision_delta.replayssm_state["current"]["revision"] += 1
    _expect_error(
        ValueError,
        "revision delta is invalid",
        lambda: _validate_fixture(generate_module, wrong_revision_delta),
    )


def _scenario_prefill_geometry(generate_module: ModuleType) -> None:
    width1 = _build_fixture(generate_module, prefill_width1=1)
    width1_marker = _validate_fixture(generate_module, width1).model_dump(
        exclude={"receipt_schema"}
    )
    assert width1_marker["packed_width1_hits"] == 92
    assert width1_marker["kda_helper_calls"] == 2 * 69

    width3 = _build_fixture(generate_module, prefill_width3=1)
    width3_marker = _validate_fixture(generate_module, width3).model_dump(
        exclude={"receipt_schema"}
    )
    assert width3_marker["packed_width3_hits"] == 3 * 92
    assert width3_marker["kda_noncontract_calls"] == 69
    assert width3_marker["native_q3_n10624"] == 3 * 92

    noncontract = _build_fixture(generate_module, prefill_noncontract=2)
    noncontract_marker = _validate_fixture(generate_module, noncontract).model_dump(
        exclude={"receipt_schema"}
    )
    assert noncontract_marker["noncontract_calls"] == 2 * 92
    assert noncontract_marker["kda_noncontract_calls"] == 2 * 69
    assert noncontract_marker["native_q3_n10624"] == 2 * 92

    control = _build_fixture(
        generate_module,
        arm_code=0,
        prefill_noncontract=1,
    )
    control_marker = _validate_fixture(generate_module, control).model_dump(
        exclude={"receipt_schema"}
    )
    assert control_marker["gate_disabled_calls"] == 3 * 92
    assert control_marker["kda_gate_disabled_calls"] == 3 * 69
    assert control_marker["native_q3_n10624"] == 0


def _scenario_canonical_totals(generate_module: ModuleType) -> None:
    accepted = [2] * 36 + [1] * 9
    fixture = _build_fixture(
        generate_module,
        full_rounds=45,
        tail_rounds=2,
        accepted=accepted,
        visible_tokens=47,
    )
    marker = _validate_fixture(generate_module, fixture).model_dump(
        exclude={"receipt_schema"}
    )
    assert marker["packed_width3_hits"] == 4140
    assert marker["kda_success_calls"] == 3105
    assert marker["deferred_submitted_roots"] == 540
    assert marker["proposed_tokens"] == 90
    assert marker["accepted_tokens"] == 81
    assert marker["emitted_tokens"] == 128
    assert marker["visible_output_tokens"] == 47
    assert marker["native_q3_n10624"] == 4140
    assert marker["native_q3_total"] == 12420
    validator_source = inspect.getsource(
        generate_module._PackedFrontReceiptRequest._validated_marker
    )
    assert "4140" not in validator_source
    assert "3105" not in validator_source
    assert "540" not in validator_source


def _scenario_causal_fail_closed(generate_module: ModuleType) -> None:
    valid = _build_fixture(generate_module)
    _validate_fixture(generate_module, valid)

    over_capacity = _build_fixture(generate_module)
    shift = generate_module._PROJECTED_KV_CACHE_MAX_TOKENS
    over_capacity.request._decode_begin_cache_offset += shift
    over_capacity.request._round_records = [
        (*record[:-1], record[-1] + shift)
        for record in over_capacity.request._round_records
    ]
    shifted_events: list[tuple[int, str, tuple[int, ...]]] = []
    for round_index, event, values in over_capacity.request._causal_events:
        if event in {"proposal_context_agreed", "deferred_roots_submitted"}:
            values = (
                (values[0] + shift,)
                if event == "proposal_context_agreed"
                else (values[0] + shift, values[1] + shift, values[2])
            )
        elif event in {"tail_prelaunch_submitted", "tail_prelaunch_used"}:
            values = (values[0] + shift, *values[1:])
        elif event == "tail_prelaunch_discarded":
            values = (values[0], values[1] + shift, *values[2:])
        shifted_events.append((round_index, event, values))
    over_capacity.request._causal_events = shifted_events
    over_capacity.runtime.target_cache_offset += shift
    attestation = over_capacity.runtime.deferred_async_width3_attestation
    over_capacity.runtime.deferred_async_width3_attestation = attestation.__class__(
        enabled=attestation.enabled,
        validated_rounds=attestation.validated_rounds,
        materialized_rounds=attestation.materialized_rounds,
        validated_roots=attestation.validated_roots,
        submitted_roots=attestation.submitted_roots,
        first_initial_offset=attestation.first_initial_offset + shift,
        last_initial_offset=attestation.last_initial_offset + shift,
        last_final_offset=attestation.last_final_offset + shift,
    )
    _expect_error(
        ValueError,
        "acceptance/emission algebra is invalid|root batch order/offset",
        lambda: _validate_fixture(generate_module, over_capacity),
    )

    wrong_first_cache = _build_fixture(generate_module)
    record = wrong_first_cache.request._round_records[0]
    wrong_first_cache.request._round_records[0] = (*record[:-1], record[-1] + 1)
    _expect_error(
        ValueError,
        "target-cache schedule is noncontiguous",
        lambda: _validate_fixture(generate_module, wrong_first_cache),
    )

    wrong_root = _build_fixture(generate_module)
    for index, (round_index, event, values) in enumerate(
        wrong_root.request._causal_events
    ):
        if event == "deferred_roots_submitted":
            wrong_root.request._causal_events[index] = (
                round_index,
                event,
                (values[0] + 1, values[1] + 1, values[2]),
            )
            break
    _expect_error(
        ValueError,
        "root batch order/offset",
        lambda: _validate_fixture(generate_module, wrong_root),
    )

    wrong_context = _build_fixture(generate_module)
    for index, (round_index, event, values) in enumerate(
        wrong_context.request._causal_events
    ):
        if event == "proposal_context_agreed":
            wrong_context.request._causal_events[index] = (
                round_index,
                event,
                (values[0] + 1,),
            )
            break
    _expect_error(
        ValueError,
        "proposal context differs",
        lambda: _validate_fixture(generate_module, wrong_context),
    )

    wrong_tail_anchor = _build_fixture(generate_module)
    for index, (round_index, event, values) in enumerate(
        wrong_tail_anchor.request._causal_events
    ):
        if event == "tail_prelaunch_submitted":
            wrong_tail_anchor.request._causal_events[index] = (
                round_index,
                event,
                (values[0], values[1] ^ 1, *values[2:]),
            )
            break
    _expect_error(
        ValueError,
        "submission is unbound",
        lambda: _validate_fixture(generate_module, wrong_tail_anchor),
    )

    wrong_emission = _build_fixture(generate_module)
    record = wrong_emission.request._round_records[0]
    wrong_emission.request._round_records[0] = (
        record[0],
        record[1],
        record[2],
        record[2],
        *record[4:],
    )
    _expect_error(
        ValueError,
        "telemetry totals are invalid|acceptance/emission algebra",
        lambda: _validate_fixture(generate_module, wrong_emission),
    )

    reordered = _build_fixture(generate_module)
    events = reordered.request._causal_events
    acceptance_index = next(
        index for index, item in enumerate(events) if item[1] == "acceptance_agreed"
    )
    events[acceptance_index], events[acceptance_index + 1] = (
        events[acceptance_index + 1],
        events[acceptance_index],
    )
    _expect_error(
        ValueError,
        "causal event ordering|committed output anchor is invalid",
        lambda: _validate_fixture(generate_module, reordered),
    )

    inert_tail = _build_fixture(generate_module)
    inert_tail.request._causal_events = [
        event
        for event in inert_tail.request._causal_events
        if not event[1].startswith("tail_prelaunch_")
    ]
    inert_tail.request._round_records = [
        (*record[:6], 0, 0, record[8]) for record in inert_tail.request._round_records
    ]
    inert_tail.request._tail_prelaunch_submitted = 0
    inert_tail.request._tail_prelaunch_used = 0
    inert_tail.request._tail_prelaunch_discarded = 0
    _expect_error(
        ValueError,
        "tail-prelaunch causal chain is incomplete",
        lambda: _validate_fixture(generate_module, inert_tail),
    )

    orphan_used = _build_fixture(generate_module)
    record = orphan_used.request._round_records[0]
    orphan_used.request._round_records[0] = (
        *record[:7],
        1,
        record[8],
    )
    orphan_used.request._tail_prelaunch_used += 1
    _expect_error(
        ValueError,
        "used without a submission",
        lambda: _validate_fixture(generate_module, orphan_used),
    )

    unconsumed = _build_fixture(generate_module, submit_final=True)
    unconsumed.request._causal_events = [
        event
        for event in unconsumed.request._causal_events
        if not (
            event[0] == 1
            and event[1] == "tail_prelaunch_discarded"
            and event[2][0] == 2
        )
    ]
    unconsumed.request._tail_prelaunch_discarded -= 1
    _expect_error(
        ValueError,
        "causal event ordering|causal chain is incomplete",
        lambda: _validate_fixture(generate_module, unconsumed),
    )

    poisoned = _build_fixture(generate_module)
    poisoned.request._poisoned = True
    _expect_error(
        ValueError,
        "poisoned or incomplete",
        lambda: _validate_fixture(generate_module, poisoned),
    )
    fallback = _build_fixture(generate_module)
    fallback.telemetry.fallback_rounds = 1
    _expect_error(
        ValueError,
        "fallback or error rounds",
        lambda: _validate_fixture(generate_module, fallback),
    )
    duplicate = _build_fixture(generate_module)
    duplicate.request._duplicate_events = 1
    _expect_error(
        ValueError,
        "poisoned or incomplete",
        lambda: _validate_fixture(generate_module, duplicate),
    )
    stale = _build_fixture(generate_module)
    stale.request._stale_events = 1
    _expect_error(
        ValueError,
        "poisoned or incomplete",
        lambda: _validate_fixture(generate_module, stale),
    )


def _scenario_native_counter_algebra(generate_module: ModuleType) -> None:
    named_overflow = _build_fixture(generate_module)
    named_overflow.native_counts.update(
        {"total": 1, "n4480": 1, "n6144": 1, "n10624": 1}
    )
    _expect_error(
        ValueError,
        "named counters exceed the aggregate",
        named_overflow.request._native_counts,
    )

    wrong_n10624 = _build_fixture(generate_module)
    wrong_n10624.native_counts["n10624"] -= 1
    _expect_error(
        ValueError,
        "N10624 dispatches do not equal packed W3 hits",
        lambda: _validate_fixture(generate_module, wrong_n10624),
    )

    unrelated_named = _build_fixture(generate_module)
    unrelated_named.native_counts["n4480"] = 1
    unrelated_named.native_counts["total"] += 1
    _expect_error(
        ValueError,
        "aggregate/other algebra",
        lambda: _validate_fixture(generate_module, unrelated_named),
    )

    wrong_total = _build_fixture(generate_module)
    wrong_total.native_counts["total"] += 1
    _expect_error(
        ValueError,
        "aggregate/other algebra",
        lambda: _validate_fixture(generate_module, wrong_total),
    )

    decreased = _build_fixture(generate_module)
    decreased.request._native_baseline = (
        decreased.native_counts["total"] + 1,
        0,
        0,
        0,
    )
    _expect_error(
        ValueError,
        "counters decreased",
        lambda: _validate_fixture(generate_module, decreased),
    )


def _scenario_identity_mutation(generate_module: ModuleType) -> None:
    fixture = _build_fixture(generate_module)
    request = fixture.request
    del request.__dict__["_assert_runtime_identity"]
    state = {
        "launch": request.launch_contract_digest,
        "mlx": request.api.source_digest,
        "exo": request.exo_source_digest,
        "native_error": False,
    }
    generate_module._launch_contract_digest = lambda: state["launch"]
    generate_module._runtime_source_digest = lambda: state["mlx"]
    generate_module._exo_source_digest = lambda: state["exo"]

    def assert_native(_api: object) -> None:
        if state["native_error"]:
            raise ValueError("native image changed")

    generate_module._assert_native_library_identity = assert_native
    request._assert_runtime_identity()

    state["launch"] = (99, 6, 7, 8)
    _expect_error(
        ValueError,
        "launch contract changed",
        request._assert_runtime_identity,
    )
    state["launch"] = request.launch_contract_digest
    state["mlx"] = (99, 10, 11, 12)
    _expect_error(
        ValueError,
        "MLX source changed",
        request._assert_runtime_identity,
    )
    state["mlx"] = request.api.source_digest
    state["exo"] = (99, 14, 15, 16)
    _expect_error(
        ValueError,
        "EXO source changed",
        request._assert_runtime_identity,
    )
    state["exo"] = request.exo_source_digest
    state["native_error"] = True
    _expect_error(ValueError, "native image changed", request._assert_runtime_identity)

    begin_fixture = _build_fixture(generate_module)
    begin_fixture.request._handle = None
    begin_fixture.request._receipt = None
    begin_fixture.request._runtime = None
    begin_fixture.request._replayssm_baseline = None
    begin_fixture.request._replayssm_final = None
    del begin_fixture.request.__dict__["_assert_runtime_identity"]
    reset_calls: list[str] = []
    begin_calls: list[str] = []
    begin_fixture.request.native_api = generate_module._NativeQ3ReceiptAPI(
        library=object(),
        total=lambda: 0,
        n4480=lambda: 0,
        n6144=lambda: 0,
        n10624=lambda: 0,
        reset=lambda: reset_calls.append("reset"),
        lib_path=_THIS_FILE,
        lib_fd=-1,
        lib_identity=(),
        lib_digest=begin_fixture.request.native_api.lib_digest,
    )
    begin_fixture.request.api = generate_module._PackedFrontReceiptAPI(
        begin=lambda *_args, **_kwargs: begin_calls.append("begin") or (11, 29),
        finish=lambda *_args, **_kwargs: {},
        abort=lambda *_args: None,
        commit_telemetry=begin_fixture.request.api.commit_telemetry,
        source_digest=begin_fixture.request.api.source_digest,
    )
    state["launch"] = (99, 6, 7, 8)
    begin_fixture.request.launch_contract_digest = (5, 6, 7, 8)
    _expect_error(
        ValueError,
        "launch contract changed",
        lambda: begin_fixture.request.begin(begin_fixture.runtime, (1, 2, 3)),
    )
    assert reset_calls == []
    assert begin_calls == []

    finish_fixture = _build_fixture(
        generate_module,
        phase=generate_module._PACKED_FRONT_PHASE_ENGINE_STARTUP,
    )
    finish_request = finish_fixture.request
    expected_native = dict(finish_fixture.native_counts)
    finish_request._handle = None
    finish_request._receipt = None
    finish_request._marker = None
    finish_request._runtime = None
    finish_request._replayssm_baseline = None
    finish_request._replayssm_final = None
    del finish_request.__dict__["_assert_runtime_identity"]
    finish_calls: list[str] = []
    finish_request.api = generate_module._PackedFrontReceiptAPI(
        begin=lambda request_token, *_args, **_kwargs: (11, request_token),
        finish=lambda *_args, **_kwargs: (
            finish_calls.append("finish") or dict(finish_fixture.raw)
        ),
        abort=lambda *_args: None,
        commit_telemetry=finish_request.api.commit_telemetry,
        source_digest=finish_request.api.source_digest,
    )
    state.update(
        {
            "launch": finish_request.launch_contract_digest,
            "mlx": finish_request.api.source_digest,
            "exo": finish_request.exo_source_digest,
            "native_error": False,
        }
    )
    _reset_phase_state(generate_module)
    generate_module._claim_composition_receipt_phase(finish_request.phase)
    finish_request.begin(finish_fixture.runtime, (1, 2, 3))
    assert finish_request._handle is not None
    finish_fixture.raw["request_sequence"] = finish_request._handle[0]
    finish_fixture.raw["request_token"] = finish_request._handle[1]
    finish_fixture.native_counts.update(expected_native)
    state["launch"] = (99, 6, 7, 8)
    _expect_error(
        ValueError,
        "launch contract changed",
        lambda: finish_request.finish(
            finish_fixture.runtime,
            finish_fixture.telemetry,
        ),
    )
    assert finish_calls == []
    assert finish_request._published is False
    assert (
        finish_request.phase
        not in generate_module._COMPOSITION_RECEIPT_PUBLISHED_PHASES
    )
    assert (
        finish_request.phase
        not in generate_module._COMPOSITION_RECEIPT_COMPLETED_PHASES
    )
    finish_request.abort()


def _scenario_native_image_identity(generate_module: ModuleType) -> None:
    class _Symbol:
        argtypes: list[object]
        restype: object

        def __call__(self) -> int:
            return 0

    class _Library:
        mlx_k3_affine8_q3_triplet_dispatch_count = _Symbol()
        mlx_k3_affine8_q3_triplet_dispatch_count_n4480 = _Symbol()
        mlx_k3_affine8_q3_triplet_dispatch_count_n6144 = _Symbol()
        mlx_k3_affine8_q3_triplet_dispatch_count_n10624 = _Symbol()
        mlx_k3_affine8_q3_triplet_reset_dispatch_counts = _Symbol()

    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / "mlx"
        lib_dir = package / "lib"
        lib_dir.mkdir(parents=True)
        core = package / "core.so"
        core.write_bytes(b"core")
        library_path = lib_dir / "libmlx.dylib"
        library_path.write_bytes(b"sealed-native-image")
        generate_module.mx.__file__ = str(core)
        os.environ[generate_module._EXPECTED_LIBMLX_SHA256_ENV] = "0" * 64
        generate_module.ctypes.CDLL = lambda *_args, **_kwargs: _Library()
        _expect_error(
            RuntimeError,
            "must equal the sealed diagnostic",
            generate_module._load_native_q3_receipt_api,
        )

        generate_module._SEALED_LIBMLX_SHA256 = hashlib.sha256(
            b"sealed-native-image"
        ).hexdigest()
        os.environ[generate_module._EXPECTED_LIBMLX_SHA256_ENV] = (
            generate_module._SEALED_LIBMLX_SHA256
        )
        api = generate_module._load_native_q3_receipt_api()
        assert api.lib_path == library_path.resolve()
        generate_module._assert_native_library_identity(api)
        original_path = library_path.with_suffix(".authenticated")
        library_path.rename(original_path)
        library_path.write_bytes(b"replacement-native-image")
        _expect_error(
            ValueError,
            "loaded .* identity changed|path no longer names the loaded file",
            lambda: generate_module._assert_native_library_identity(api),
        )
        os.close(api.lib_fd)

        symlink = lib_dir / "symlink.dylib"
        symlink.symlink_to(library_path)
        _expect_error(
            OSError,
            "",
            lambda: generate_module._open_native_snapshot(symlink),
        )


def _scenario_finish_abort_and_publication(generate_module: ModuleType) -> None:
    fixture = _build_fixture(generate_module)
    aborts: list[tuple[int, int]] = []
    fixture.request.api = generate_module._PackedFrontReceiptAPI(
        begin=lambda *_args, **_kwargs: (11, 29),
        finish=lambda *_args, **_kwargs: dict(fixture.raw),
        abort=lambda sequence, token: aborts.append((sequence, token)),
        commit_telemetry=fixture.request.api.commit_telemetry,
        source_digest=fixture.request.api.source_digest,
    )
    receipt = fixture.request.finish(fixture.runtime, fixture.telemetry)
    assert receipt.request_sequence == 11
    assert fixture.request._handle is None
    assert fixture.runtime.attached is None

    _reset_phase_state(generate_module)
    generate_module._COMPOSITION_RECEIPT_ATTEMPTED_PHASES.add(fixture.request.phase)
    generate_module._COMPOSITION_RECEIPT_ACTIVE_PHASE = fixture.request.phase
    generate_module._publish_composition_receipt(
        fixture.runtime,
        fixture.request,
    )
    assert fixture.request._published is True
    assert fixture.request._native_fd_closed is True
    assert (
        fixture.request.phase in generate_module._COMPOSITION_RECEIPT_PUBLISHED_PHASES
    )
    assert fixture.runtime.side_effect_calls == [
        "W3 composition marker prepublication verification",
        "W3 composition marker publication",
        "W3 composition marker lifecycle completion",
    ]

    startup_phase = generate_module._PACKED_FRONT_PHASE_ENGINE_STARTUP
    api_phase = generate_module._PACKED_FRONT_PHASE_API

    publication_failure = _build_fixture(
        generate_module,
        phase=startup_phase,
    )
    publication_failure.request.api = generate_module._PackedFrontReceiptAPI(
        begin=lambda *_args, **_kwargs: (11, 29),
        finish=lambda *_args, **_kwargs: dict(publication_failure.raw),
        abort=lambda *_args: None,
        commit_telemetry=publication_failure.request.api.commit_telemetry,
        source_digest=publication_failure.request.api.source_digest,
    )
    publication_failure.request.finish(
        publication_failure.runtime,
        publication_failure.telemetry,
    )
    _reset_phase_state(generate_module)
    generate_module._claim_composition_receipt_phase(startup_phase)
    publication_failure.runtime.side_effect_failure = (
        "W3 composition marker publication"
    )
    _expect_error(
        RuntimeError,
        "injected .* publication failure",
        lambda: generate_module._publish_composition_receipt(
            publication_failure.runtime,
            publication_failure.request,
        ),
    )
    assert publication_failure.request._published is False
    assert not generate_module._COMPOSITION_RECEIPT_PUBLISHED_PHASES
    assert not generate_module._COMPOSITION_RECEIPT_COMPLETED_PHASES
    publication_failure.request.abort()
    _expect_error(
        RuntimeError,
        "requires a published startup receipt",
        lambda: generate_module._claim_composition_receipt_phase(api_phase),
    )

    mark_failure = _build_fixture(generate_module, phase=startup_phase)
    mark_failure.request.api = generate_module._PackedFrontReceiptAPI(
        begin=lambda *_args, **_kwargs: (11, 29),
        finish=lambda *_args, **_kwargs: dict(mark_failure.raw),
        abort=lambda *_args: None,
        commit_telemetry=mark_failure.request.api.commit_telemetry,
        source_digest=mark_failure.request.api.source_digest,
    )
    mark_failure.request.finish(mark_failure.runtime, mark_failure.telemetry)
    _reset_phase_state(generate_module)
    generate_module._claim_composition_receipt_phase(startup_phase)
    original_record = generate_module._record_composition_receipt_publication

    def reject_mark(_phase: int, _identity: tuple[int, ...]) -> None:
        raise RuntimeError("injected lifecycle mark failure")

    generate_module._record_composition_receipt_publication = reject_mark
    try:
        _expect_error(
            RuntimeError,
            "injected lifecycle mark failure",
            lambda: generate_module._publish_composition_receipt(
                mark_failure.runtime,
                mark_failure.request,
            ),
        )
    finally:
        generate_module._record_composition_receipt_publication = original_record
    assert mark_failure.request._published is False
    assert not generate_module._COMPOSITION_RECEIPT_PUBLISHED_PHASES
    assert not generate_module._COMPOSITION_RECEIPT_COMPLETED_PHASES
    mark_failure.request.abort()
    _expect_error(
        RuntimeError,
        "requires a published startup receipt",
        lambda: generate_module._claim_composition_receipt_phase(api_phase),
    )

    divergent = _build_fixture(generate_module)
    divergent.runtime.agreement_failure = RuntimeError("rank marker mismatch")
    divergent_aborts: list[tuple[int, int]] = []
    divergent.request.api = generate_module._PackedFrontReceiptAPI(
        begin=lambda *_args, **_kwargs: (11, 29),
        finish=lambda *_args, **_kwargs: dict(divergent.raw),
        abort=lambda sequence, token: divergent_aborts.append((sequence, token)),
        commit_telemetry=divergent.request.api.commit_telemetry,
        source_digest=divergent.request.api.source_digest,
    )
    _expect_error(
        RuntimeError,
        "rank marker mismatch",
        lambda: divergent.request.finish(divergent.runtime, divergent.telemetry),
    )
    divergent.request.abort()
    assert divergent_aborts == [(11, 29)]
    assert divergent.request._handle is None
    assert divergent.runtime.attached is None
    assert divergent.request._native_fd_closed is True

    implementation_source = inspect.getsource(generate_module._mlx_generate_impl)
    close_index = implementation_source.index("packed-front terminal decode close")
    barrier_index = implementation_source.index("mx_barrier(group)", close_index)
    finish_index = implementation_source.index(
        "packed_front_receipt.finish", barrier_index
    )
    publication_index = implementation_source.index(
        "_publish_composition_receipt(", finish_index
    )
    assert close_index < barrier_index < finish_index < publication_index
    publication_source = inspect.getsource(generate_module._publish_composition_receipt)
    verification_stage = publication_source.index(
        "W3 composition marker prepublication verification"
    )
    emit_stage = publication_source.index("W3 composition marker publication")
    mark_stage = publication_source.index("W3 composition marker lifecycle completion")
    assert verification_stage < emit_stage < mark_stage
    assert publication_source.count("runtime.agree_local_side_effect(") == 3


def _scenario_rank_local_sha256_gather(_generate_module: ModuleType) -> None:
    from exo.worker.engines.mlx.generator.kimi_k3_dspark import MlxRankAgreement

    rank0 = hashlib.sha256(b"rank0").hexdigest()
    rank1 = hashlib.sha256(b"rank1").hexdigest()
    agreement = MlxRankAgreement(None)

    def two_rank_rows(row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        rank1_raw = bytes.fromhex(rank1)
        rank1_words = tuple(
            int.from_bytes(rank1_raw[offset : offset + 2], "big")
            for offset in range(0, 32, 2)
        )
        return row, (row[0], 1, *rank1_words)

    agreement._all_gather_rows = two_rank_rows  # type: ignore[method-assign]
    agreement._group = SimpleNamespace(  # type: ignore[assignment]
        rank=lambda: 0,
        size=lambda: 2,
    )
    assert agreement.gather_rank_local_sha256("frontier", rank0) == (rank0, rank1)
    _expect_error(
        ValueError,
        "input is invalid",
        lambda: agreement.gather_rank_local_sha256("frontier", "A" * 64),
    )

    def duplicate_rank(row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        return row, row

    agreement._all_gather_rows = duplicate_rank  # type: ignore[method-assign]
    _expect_error(
        RuntimeError,
        "identity diverged",
        lambda: agreement.gather_rank_local_sha256("frontier", rank0),
    )


def _scenario_frontier_receipt_v4(generate_module: ModuleType) -> None:
    from exo.api.types import GenerationStats, K3W3CompositionReceiptV4
    from exo.shared.types.memory import Memory

    fixture = _build_fixture(generate_module, identity_enabled=True)
    fixture.request.api = generate_module._PackedFrontReceiptAPI(
        begin=lambda *_args, **_kwargs: (11, 29),
        finish=lambda *_args, **_kwargs: dict(fixture.raw),
        abort=lambda *_args: None,
        commit_telemetry=fixture.request.api.commit_telemetry,
        source_digest=fixture.request.api.source_digest,
    )
    nonce = hashlib.sha256(b"nonce").hexdigest()
    core_sha = hashlib.sha256(b"core").hexdigest()
    rank0_file = hashlib.sha256(b"rank0-file").hexdigest()
    rank1_file = hashlib.sha256(b"rank1-file").hexdigest()
    context = object()
    fixture.request._frontier_context = context
    original_finalize = generate_module.k3_frontier_telemetry.finalize
    original_poison = generate_module.k3_frontier_telemetry.poison_finalized

    def finalize(_context: object, core: dict[str, object]) -> SimpleNamespace:
        assert _context is context
        assert core["receipt_schema_version"] == 2
        return SimpleNamespace(
            request_index=1,
            request_nonce_sha256=nonce,
            reset_generation=1,
            receipt_v2_core_sha256=core_sha,
            request_file_sha256=rank0_file,
            process_complete_file_sha256=None,
        )

    def gather(name: str, digest: str) -> tuple[str, str]:
        if name == "frontier-request-file":
            assert digest == rank0_file
            return rank0_file, rank1_file
        assert name == "frontier-process-complete"
        assert digest == "0" * 64
        return "0" * 64, "0" * 64

    generate_module.k3_frontier_telemetry.finalize = finalize
    generate_module.k3_frontier_telemetry.poison_finalized = lambda _context: None
    fixture.runtime.collective = SimpleNamespace(
        rank=0,
        gather_rank_local_sha256=gather,
    )
    try:
        receipt = fixture.request.finish(fixture.runtime, fixture.telemetry)
    finally:
        generate_module.k3_frontier_telemetry.finalize = original_finalize
        generate_module.k3_frontier_telemetry.poison_finalized = original_poison
    assert isinstance(receipt, K3W3CompositionReceiptV4)
    assert receipt.receipt_schema == "kimi-k3-w3-composition-receipt/v4"
    assert receipt.receipt_schema_version == 4
    assert receipt.frontier_request_index == 1
    assert receipt.frontier_reset_generation == 1
    nonce_words = tuple(
        getattr(receipt, f"frontier_request_nonce_digest_word_{index}")
        for index in range(4)
    )
    assert b"".join(word.to_bytes(8, "big") for word in nonce_words).hex() == nonce
    rank1_words = tuple(
        getattr(receipt, f"frontier_telemetry_file_digest_rank1_word_{index}")
        for index in range(4)
    )
    assert b"".join(word.to_bytes(8, "big") for word in rank1_words).hex() == rank1_file
    assert fixture.request._marker is not None
    assert '"receipt_schema_version":4' in fixture.request._marker
    serialized = GenerationStats(
        prompt_tps=1.0,
        generation_tps=2.0,
        prompt_tokens=3,
        generation_tokens=4,
        peak_memory_usage=Memory.from_bytes(5),
        k3_w3_composition_receipt=receipt,
    ).model_dump()["k3_w3_composition_receipt"]
    assert serialized["schema"] == "kimi-k3-w3-composition-receipt/v4"
    assert serialized["frontier_request_index"] == 1
    assert "frontier_telemetry_file_digest_rank1_word_3" in serialized
    assert not any(
        key.startswith(("process_rss_", "rss_", "metal_")) for key in serialized
    )


_SCENARIOS: dict[str, Callable[[ModuleType], None]] = {
    "candidate-control": _scenario_candidate_and_control,
    "canonical-totals": _scenario_canonical_totals,
    "causal-fail-closed": _scenario_causal_fail_closed,
    "finish-abort-publication": _scenario_finish_abort_and_publication,
    "frontier-receipt-v4": _scenario_frontier_receipt_v4,
    "identity-mutation": _scenario_identity_mutation,
    "identity-commit-telemetry": _scenario_identity_commit_telemetry,
    "lifecycle": _scenario_lifecycle,
    "native-counter-algebra": _scenario_native_counter_algebra,
    "native-image-identity": _scenario_native_image_identity,
    "prefill-geometry": _scenario_prefill_geometry,
    "rank-local-sha256-gather": _scenario_rank_local_sha256_gather,
    "schema-default-wire": _scenario_schema_and_default_wire,
    "selector-admission": _scenario_selector_admission,
    "strict-off": _scenario_strict_off,
}


def _run_inner_scenario(name: str) -> None:
    _install_headless_receipt_imports()
    import exo.worker.engines.mlx.generator.generate as generate_module

    if name == "import":
        assert (
            generate_module._PACKED_FRONT_RECEIPT_SCHEMA
            == "kimi-k3-w3-composition-receipt/v1"
        )
        assert (
            generate_module._PACKED_FRONT_MARKER_SCHEMA
            == "kimi-k3-w3-composition-receipt/v2"
        )
        assert generate_module._PACKED_FRONT_EXPECTED_LAYERS == 92
        assert generate_module._KDA_EXPECTED_LAYERS == 69
        assert generate_module._DEFERRED_EXPECTED_ROOTS == 12
        return
    scenario = _SCENARIOS.get(name)
    if scenario is None:
        raise AssertionError(f"unknown receipt scenario: {name}")
    scenario(generate_module)


def _run_scenario(name: str) -> None:
    environment = os.environ.copy()
    source_root = _THIS_FILE.parents[5]
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, str(_THIS_FILE), "--scenario", name],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("scenario", ["import", *_SCENARIOS])
def test_k3_w3_composition_receipt_contract(scenario: str) -> None:
    _run_scenario(scenario)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--scenario":
        raise SystemExit("usage: test_k3_w3_composition_receipt.py --scenario NAME")
    _run_inner_scenario(sys.argv[2])
