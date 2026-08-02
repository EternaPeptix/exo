from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, cast
from unittest.mock import patch

import pytest
from anyio import ClosedResourceError, WouldBlock

from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import ChunkGenerated
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.generator import generate as generate_module
from exo.worker.engines.mlx.generator.generate import (
    greedy_vocab_parallel_stream_kwargs,
    warmup_inference,
)
from exo.worker.engines.mlx.generator.kimi_k3_dspark import (
    DSparkDistributedStateError,
    DSparkRoundResult,
    DSparkRoundTelemetry,
    LoadedMlxDSpark,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import rank_agreed_local_stage
from exo.worker.runner.llm_inference import batch_generator
from exo.worker.runner.llm_inference.batch_generator import (
    GeneratorQueue,
    SequentialGenerator,
)


def _kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "temperature": 0.0,
        "logprobs": False,
        "has_logits_processors": False,
        "is_pipeline": False,
        "speculative": False,
    }
    values.update(overrides)
    return values


def test_compact_greedy_is_default_off() -> None:
    with patch.dict("os.environ", {}, clear=True):
        assert greedy_vocab_parallel_stream_kwargs(**_kwargs()) == {}  # type: ignore[arg-type]


def test_compact_greedy_exact_request_shape() -> None:
    with patch.dict(
        "os.environ",
        {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "1"},
        clear=True,
    ):
        assert greedy_vocab_parallel_stream_kwargs(**_kwargs()) == {  # type: ignore[arg-type]
            "greedy_vocab_parallel_no_logprobs": True
        }


@pytest.mark.parametrize(
    "override",
    [
        {"temperature": 0.1},
        {"logprobs": True},
        {"has_logits_processors": True},
        {"is_pipeline": True},
        {"speculative": True},
    ],
)
def test_compact_greedy_falls_back_for_incompatible_requests(
    override: dict[str, object],
) -> None:
    with patch.dict(
        "os.environ",
        {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "1"},
        clear=True,
    ):
        assert (
            greedy_vocab_parallel_stream_kwargs(  # type: ignore[arg-type]
                **_kwargs(**override)
            )
            == {}
        )


def test_compact_greedy_rejects_malformed_opt_in() -> None:
    with (
        patch.dict(
            "os.environ",
            {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "true"},
            clear=True,
        ),
        pytest.raises(
            ValueError,
            match="EXO_MLX_K3_VOCAB_PARALLEL_GREEDY must be 0 or 1",
        ),
    ):
        greedy_vocab_parallel_stream_kwargs(**_kwargs())  # type: ignore[arg-type]


@pytest.mark.parametrize("verify_width", [3, 8])
def test_dspark_warmup_requires_two_real_speculative_rounds(
    monkeypatch: pytest.MonkeyPatch,
    verify_width: int,
) -> None:
    captured: list[object] = []
    expected_tokens = 2 * verify_width

    def fake_generate(**kwargs: object):
        task = kwargs["task"]
        captured.append(task)
        for index in range(expected_tokens):
            stats = (
                SimpleNamespace(
                    speculative_rounds=2,
                    speculative_drafted_tokens=2 * (verify_width - 1),
                    speculative_fallback_rounds=0,
                    speculative_error_rounds=0,
                )
                if index == expected_tokens - 1
                else None
            )
            yield SimpleNamespace(stats=stats)

    monkeypatch.setenv("EXO_MLX_WARMUP_OUTPUT_TOKENS", "4")
    monkeypatch.setattr(generate_module, "apply_chat_template", lambda **_kwargs: "p")
    monkeypatch.setattr(generate_module, "mlx_generate", fake_generate)
    monkeypatch.setattr(generate_module, "mx_barrier", lambda _group: None)

    warmup_inference(
        model=cast(Model, object()),
        tokenizer=cast(object, object()),
        group=None,
        model_id=ModelId("kernelpool/Kimi-K3-2bit-UVMAX"),
        dspark=cast(
            LoadedMlxDSpark,
            cast(object, SimpleNamespace(verify_width=verify_width)),
        ),
    )

    assert len(captured) == 1
    task = captured[0]
    assert task.max_output_tokens == expected_tokens
    assert task.bench is True


def test_dspark_warmup_rejects_fallback_only_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_generate(**_kwargs: object):
        for index in range(6):
            yield SimpleNamespace(
                stats=(
                    SimpleNamespace(
                        speculative_rounds=2,
                        speculative_drafted_tokens=4,
                        speculative_fallback_rounds=1,
                        speculative_error_rounds=1,
                    )
                    if index == 5
                    else None
                )
            )

    monkeypatch.setattr(generate_module, "apply_chat_template", lambda **_kwargs: "p")
    monkeypatch.setattr(generate_module, "mlx_generate", fake_generate)
    monkeypatch.setattr(generate_module, "mx_barrier", lambda _group: None)

    with pytest.raises(RuntimeError, match="two clean speculative rounds"):
        warmup_inference(
            model=cast(Model, object()),
            tokenizer=cast(object, object()),
            group=None,
            model_id=ModelId("kernelpool/Kimi-K3-2bit-UVMAX"),
            dspark=cast(
                LoadedMlxDSpark,
                cast(object, SimpleNamespace(verify_width=3)),
            ),
        )


def test_rank_agreed_local_stage_fail_stops_before_later_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collectives: list[tuple[str, object]] = []

    def any_rank(value: bool, _group: object) -> bool:
        collectives.append(("any", value))
        return value

    def ranks_agree(value: int, _group: object) -> bool:
        collectives.append(("agree", value))
        return len(collectives) != 2

    def fail() -> object:
        raise ValueError("rank-local load failed")

    monkeypatch.setattr(utils_mlx, "mx_any", any_rank)
    monkeypatch.setattr(utils_mlx, "mx_ranks_agree_on_value", ranks_agree)

    with pytest.raises(RuntimeError, match="failed on at least one distributed rank"):
        rank_agreed_local_stage(
            "rank-local startup",
            cast(object, object()),
            fail,
            lambda _result: "unused",
        )

    assert collectives[0] == ("any", True)
    assert collectives[1][0] == "agree"
    assert collectives[1][1] != 0
    assert collectives[2] == ("agree", 0)


def test_rank_agreed_local_stage_accepts_unanimous_disabled_result() -> None:
    assert (
        rank_agreed_local_stage(
            "optional startup",
            None,
            lambda: None,
            lambda result: "disabled" if result is None else "enabled",
        )
        is None
    )


def test_rank_agreed_local_stage_rejects_success_contract_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreements = iter((True, False))
    monkeypatch.setattr(utils_mlx, "mx_any", lambda _value, _group: False)
    monkeypatch.setattr(
        utils_mlx,
        "mx_ranks_agree_on_value",
        lambda _value, _group: next(agreements),
    )

    with pytest.raises(RuntimeError, match="result diverged"):
        rank_agreed_local_stage(
            "replicated draft load",
            cast(object, object()),
            lambda: object(),
            lambda _result: "local contract",
        )


def test_rank_agreed_fail_stop_uses_one_collective_and_preserves_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flags: list[bool] = []
    monkeypatch.setattr(
        utils_mlx,
        "mx_any",
        lambda value, _group: flags.append(value) or False,
    )

    result = utils_mlx.rank_agreed_fail_stop(
        "parser",
        cast(object, object()),
        lambda: "parsed",
    )

    assert result == "parsed"
    assert flags == [False]


def test_rank_agreed_fail_stop_observes_peer_failure_after_local_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(utils_mlx, "mx_any", lambda _value, _group: True)

    with pytest.raises(RuntimeError, match="failed on at least one"):
        utils_mlx.rank_agreed_fail_stop(
            "parser",
            cast(object, object()),
            lambda: calls.append("local") or "parsed",
        )

    assert calls == ["local"]


def test_distributed_wired_limit_contract_allows_uneven_rank_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts: list[tuple[str, str]] = []

    def agreed_stage(
        stage: str,
        _group: object,
        operation: Callable[[], object],
        contract: Callable[[object], str],
    ) -> object:
        result = operation()
        if stage == "distributed MLX wired-limit setup":
            contracts.append((contract(1), contract(999)))
        return result

    def paused_shard_load(*_args: object, **_kwargs: object):
        yield ModelLoadingResponse(layers_loaded=0, total=1)

    fake_shard = SimpleNamespace()
    fake_bound = SimpleNamespace(bound_shard=fake_shard)
    monkeypatch.setattr(utils_mlx, "rank_agreed_local_stage", agreed_stage)
    monkeypatch.setattr(
        utils_mlx,
        "get_weights_size",
        lambda _shard: SimpleNamespace(in_bytes=123),
    )
    monkeypatch.setattr(utils_mlx, "set_wired_limit_for_model", lambda _size: None)
    monkeypatch.setattr(utils_mlx, "shard_and_load", paused_shard_load)

    loader = utils_mlx.load_mlx_items(
        cast(object, fake_bound),
        cast(object, object()),
    )
    assert next(loader).layers_loaded == 0
    assert contracts == [("configured", "configured")]


def test_rank_local_load_has_no_progress_yield_after_terminal_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensorShard:
        n_layers = 93

    model = utils_mlx.nn.Module()
    tokenizer = object()
    prepared = SimpleNamespace(
        checkpoint_template="/models/rank{rank}",
        runtime=SimpleNamespace(
            loader_path="/audited/rank_local_loader.py",
            verify_file_hashes=False,
        ),
        vocab_parallel_head=True,
        load_model=lambda *_args: None,
    )
    loaded = SimpleNamespace(
        model=model,
        checkpoint_path=SimpleNamespace(),
        config={},
    )
    events: list[str] = []

    def agreed_stage(
        stage: str,
        _group: object,
        operation: Callable[[], object],
        contract: Callable[[object], str],
    ) -> object:
        events.append(stage)
        result = operation()
        contract(result)
        return result

    monkeypatch.setenv("EXO_MLX_RANK_LOCAL_CHECKPOINT", "/models/rank{rank}")
    monkeypatch.setattr(utils_mlx, "TensorShardMetadata", FakeTensorShard)
    monkeypatch.setattr(utils_mlx, "mx_ranks_agree_on_value", lambda *_args: True)
    monkeypatch.setattr(
        utils_mlx,
        "preflight_configured_rank_local_model",
        lambda *_args: prepared,
    )
    monkeypatch.setattr(
        utils_mlx,
        "load_preflighted_rank_local_model",
        lambda *_args: loaded,
    )
    monkeypatch.setattr(
        utils_mlx,
        "_patch_rank_local_tensor_model_once",
        lambda value: value,
    )
    monkeypatch.setattr(utils_mlx, "get_tokenizer", lambda *_args: tokenizer)
    monkeypatch.setattr(utils_mlx, "rank_agreed_local_stage", agreed_stage)
    monkeypatch.setattr(
        utils_mlx,
        "mx_barrier",
        lambda _group: events.append("terminal barrier"),
    )

    loader = utils_mlx.shard_and_load(
        cast(object, FakeTensorShard()),
        cast(object, object()),
    )
    with pytest.raises(StopIteration) as stopped:
        next(loader)

    assert stopped.value.value == (model, tokenizer)
    assert events[-2:] == [
        "rank-local model/tokenizer finalization",
        "terminal barrier",
    ]


def test_peer_rank_local_preflight_failure_prevents_external_loader_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensorShard:
        n_layers = 93

    class FakeGroup:
        pass

    external_loads: list[None] = []

    def external_load(*_args: object, **_kwargs: object) -> object:
        external_loads.append(None)
        return object()

    prepared = SimpleNamespace(
        checkpoint_template="/models/rank{rank}",
        runtime=SimpleNamespace(
            loader_path="/audited/rank_local_loader.py",
            verify_file_hashes=False,
        ),
        vocab_parallel_head=True,
        load_model=external_load,
    )
    agreement_calls = 0

    def ranks_agree(_value: int, _group: object) -> bool:
        nonlocal agreement_calls
        agreement_calls += 1
        # Enablement agrees. During the local-preflight stage, rank 0 has no
        # error while mocked rank 1 failed, then the success fingerprints are
        # still gathered in the required fixed order.
        return agreement_calls != 2

    monkeypatch.setenv("EXO_MLX_RANK_LOCAL_CHECKPOINT", "/models/rank{rank}")
    monkeypatch.setattr(utils_mlx, "TensorShardMetadata", FakeTensorShard)
    monkeypatch.setattr(
        utils_mlx,
        "preflight_configured_rank_local_model",
        lambda _shard, _group: prepared,
    )
    monkeypatch.setattr(
        utils_mlx,
        "load_preflighted_rank_local_model",
        external_load,
    )
    monkeypatch.setattr(utils_mlx, "mx_any", lambda _value, _group: True)
    monkeypatch.setattr(utils_mlx, "mx_ranks_agree_on_value", ranks_agree)

    generator = utils_mlx.shard_and_load(
        cast(object, FakeTensorShard()),
        cast(object, FakeGroup()),
    )
    with pytest.raises(RuntimeError, match="preflight failed on at least one"):
        next(generator)

    assert agreement_calls == 3
    assert external_loads == []


@pytest.mark.parametrize("verify_width", [3, 8])
def test_builder_threads_dspark_into_real_sequential_warmup(
    monkeypatch: pytest.MonkeyPatch,
    verify_width: int,
) -> None:
    target_model = cast(Model, object())
    tokenizer = SimpleNamespace(
        has_tool_calling=False,
        tool_call_start=None,
        tool_call_end=None,
        tool_parser=None,
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    dspark = cast(
        LoadedMlxDSpark,
        cast(
            object,
            SimpleNamespace(verify_width=verify_width, target_model=target_model),
        ),
    )
    builder = MlxBuilder(
        model_id=ModelId("kernelpool/Kimi-K3-2bit-UVMAX"),
        event_sender=cast(object, object()),
        cancel_receiver=cast(object, object()),
        inference_model=target_model,
        tokenizer=cast(object, tokenizer),
        group=cast(object, group),
        dspark=dspark,
    )
    captured: list[object] = []

    def fake_warmup(**kwargs: object) -> int:
        captured.append(kwargs["dspark"])
        return 17

    monkeypatch.setenv("EXO_NO_BATCH", "1")
    monkeypatch.setattr(batch_generator, "warmup_inference", fake_warmup)

    engine = builder.build()
    assert isinstance(engine, SequentialGenerator)
    assert engine.dspark is dspark
    engine.warmup()
    assert captured == [dspark]
    assert engine.check_for_cancel_every == 17


def _sequential_for_callback_test(cancel_receiver: object) -> SequentialGenerator:
    return SequentialGenerator(
        model=cast(Model, object()),
        tokenizer=cast(object, object()),
        group=cast(object, object()),
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=ModelId("kernelpool/Kimi-K3-2bit-UVMAX"),
        device_rank=0,
        cancel_receiver=cast(object, cancel_receiver),
        event_sender=cast(object, object()),
    )


def _fake_dspark_task(*, bench: bool = True) -> TextGeneration:
    return cast(
        TextGeneration,
        SimpleNamespace(
            task_id=TaskId("00000000-0000-0000-0000-000000000001"),
            command_id=CommandId("command-1"),
            task_params=SimpleNamespace(bench=bench, tools=None, input=[]),
            model_dump=lambda **_kwargs: {
                "task_params": {"bench": bench},
                "task_id": "00000000-0000-0000-0000-000000000001",
            },
        ),
    )


def test_dspark_task_digest_is_canonical_across_dict_order() -> None:
    first = cast(
        TextGeneration,
        SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "task_id": "request",
                "task_params": {"temperature": 0.0, "top_p": 1.0},
            }
        ),
    )
    second = cast(
        TextGeneration,
        SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "task_params": {"top_p": 1.0, "temperature": 0.0},
                "task_id": "request",
            }
        ),
    )

    assert batch_generator._task_digest(first) == batch_generator._task_digest(second)


def test_dspark_task_digest_ignores_rank_local_lifecycle_state() -> None:
    pending = TextGeneration(
        task_id=TaskId("00000000-0000-0000-0000-000000000001"),
        task_status=TaskStatus.Pending,
        instance_id=InstanceId("instance-1"),
        command_id=CommandId("command-1"),
        task_params=TextGenerationTaskParams(
            model=ModelId("kernelpool/Kimi-K3-2bit-UVMAX"),
            input=[InputMessage(role="user", content="hello")],
            max_output_tokens=16,
        ),
    )
    running = pending.model_copy(update={"task_status": TaskStatus.Running})
    failed = pending.model_copy(
        update={
            "task_status": TaskStatus.Failed,
            "error_type": "rank-local diagnostic",
            "error_message": "must not alter the immutable request binding",
        }
    )
    changed_request = pending.model_copy(
        update={
            "task_params": pending.task_params.model_copy(
                update={"max_output_tokens": 17}
            )
        }
    )

    expected = batch_generator._task_digest(pending)
    assert batch_generator._task_digest(running) == expected
    assert batch_generator._task_digest(failed) == expected
    assert batch_generator._task_digest(changed_request) != expected


def _dspark_sequential(
    *,
    sender: object,
    device_rank: int = 0,
) -> SequentialGenerator:
    generator = SequentialGenerator(
        model=cast(Model, object()),
        tokenizer=cast(object, object()),
        group=cast(object, object()),
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=ModelId("kernelpool/Kimi-K3-2bit-UVMAX"),
        device_rank=device_rank,
        cancel_receiver=cast(object, SimpleNamespace(collect=list)),
        event_sender=cast(object, sender),
        dspark=cast(LoadedMlxDSpark, object()),
    )
    return generator


def test_dspark_task_binding_precedes_all_request_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sender = SimpleNamespace(
        send=lambda _event: events.append("blocking error"),
        send_nowait=lambda _event: events.append("error"),
    )
    generator = _dspark_sequential(sender=sender)
    generator._queue.append(_fake_dspark_task())

    def agreed_stage(
        stage: str,
        _group: object,
        operation: Callable[[], object],
        contract: Callable[[object], str],
    ) -> object:
        events.append(stage)
        result = operation()
        contract(result)
        if stage == "Kimi K3 DSpark task binding":
            raise RuntimeError("task digest diverged across ranks")
        return result

    def unexpected_build(_task: TextGeneration) -> object:
        raise AssertionError("request side effects ran before task binding")

    monkeypatch.setattr(batch_generator, "rank_agreed_local_stage", agreed_stage)
    monkeypatch.setattr(generator, "_build_generator", unexpected_build)

    with pytest.raises(RuntimeError, match="task digest diverged"):
        generator._start_next()

    assert events == ["Kimi K3 DSpark task binding", "error"]
    assert generator._active is None


def test_dspark_peer_prep_failure_prevents_first_generator_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sender = SimpleNamespace(
        send=lambda _event: events.append("blocking error"),
        send_nowait=lambda _event: events.append("error"),
    )
    generator = _dspark_sequential(sender=sender)
    generator._queue.append(_fake_dspark_task())

    def lazy_generation():
        events.append("next generator")
        yield cast(object, SimpleNamespace(finish_reason=None))

    def build(_task: TextGeneration):
        events.append("build generator")
        return lazy_generation()

    def agreed_stage(
        stage: str,
        _group: object,
        operation: Callable[[], object],
        contract: Callable[[object], str],
    ) -> object:
        events.append(stage)
        result = operation()
        contract(result)
        if stage == "Kimi K3 DSpark request preparation":
            raise RuntimeError("peer request preparation failed")
        return result

    monkeypatch.setattr(batch_generator, "rank_agreed_local_stage", agreed_stage)
    monkeypatch.setattr(generator, "_build_generator", build)

    with pytest.raises(RuntimeError, match="peer request preparation failed"):
        generator._start_next()

    assert events == [
        "Kimi K3 DSpark task binding",
        "Kimi K3 DSpark request preparation",
        "build generator",
        "error",
    ]
    assert "next generator" not in events
    assert generator._active is None


def test_dspark_prefill_progress_publication_is_nonblocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: list[Callable[[int, int], None]] = []
    blocking_sends: list[object] = []
    nonblocking_sends: list[object] = []

    def fake_generate(**kwargs: object):
        callbacks.append(
            cast(Callable[[int, int], None], kwargs["on_prefill_progress"])
        )
        return iter(())

    def blocking_send(event: object) -> None:
        blocking_sends.append(event)
        raise AssertionError("DSpark prefill publication must not block")

    def full_send_nowait(event: object) -> None:
        nonblocking_sends.append(event)
        raise WouldBlock

    generator = _dspark_sequential(
        sender=SimpleNamespace(
            send=blocking_send,
            send_nowait=full_send_nowait,
        )
    )
    monkeypatch.setattr(batch_generator, "apply_chat_template", lambda *_args: "p")
    monkeypatch.setattr(batch_generator, "mlx_generate", fake_generate)

    generator._build_generator(_fake_dspark_task())
    with pytest.raises(WouldBlock):
        callbacks[0](1, 2)

    assert blocking_sends == []
    assert len(nonblocking_sends) == 1


def _activate_dspark_response(
    generator: SequentialGenerator,
    *,
    task: TextGeneration,
    chunk: TokenChunk,
    finish_reason: str | None = None,
) -> list[str]:
    generator_events: list[str] = []

    def responses():
        generator_events.append("next response")
        yield cast(object, SimpleNamespace(finish_reason=finish_reason))
        generator_events.append("next response")
        yield cast(object, SimpleNamespace(finish_reason=None))

    generator._active = cast(
        object,
        (
            task,
            responses(),
            GeneratorQueue(),
            iter((chunk, None)),
        ),
    )
    return generator_events


@pytest.mark.parametrize("finish_reason", [None, "stop"])
def test_dspark_parser_stage_publishes_once_without_runner_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str | None,
) -> None:
    sent: list[object] = []
    sender = SimpleNamespace(send=sent.append, send_nowait=sent.append)
    generator = _dspark_sequential(sender=sender)
    task = _fake_dspark_task()
    chunk = TokenChunk(
        model=generator.model_id,
        text="x",
        token_id=7,
        usage=None,
    )
    generator_events = _activate_dspark_response(
        generator,
        task=task,
        chunk=chunk,
        finish_reason=finish_reason,
    )
    flags: list[bool] = []
    monkeypatch.setattr(
        utils_mlx,
        "mx_any",
        lambda value, _group: flags.append(value) or False,
    )

    assert list(generator.step()) == []

    assert generator_events == ["next response"]
    assert flags == [False]
    assert len(sent) == 1
    assert isinstance(sent[0], ChunkGenerated)
    assert sent[0].chunk is chunk


@pytest.mark.parametrize("publish_error", [WouldBlock, ClosedResourceError])
@pytest.mark.parametrize("finish_reason", [None, "stop"])
def test_dspark_publication_failure_fail_stops_before_next_response(
    monkeypatch: pytest.MonkeyPatch,
    publish_error: type[Exception],
    finish_reason: str | None,
) -> None:
    sent: list[object] = []

    def fail_nowait(_event: object) -> None:
        raise publish_error

    def unexpected_blocking_send(_event: object) -> None:
        raise AssertionError("DSpark failure handling must remain nonblocking")

    generator = _dspark_sequential(
        sender=SimpleNamespace(
            send=unexpected_blocking_send,
            send_nowait=fail_nowait,
        )
    )
    task = _fake_dspark_task()
    chunk = TokenChunk(
        model=generator.model_id,
        text="x",
        token_id=7,
        usage=None,
    )
    generator_events = _activate_dspark_response(
        generator,
        task=task,
        chunk=chunk,
        finish_reason=finish_reason,
    )
    flags: list[bool] = []
    monkeypatch.setattr(
        utils_mlx,
        "mx_any",
        lambda value, _group: flags.append(value) or value,
    )

    with pytest.raises(RuntimeError, match="parser/publication failed"):
        generator.step()

    assert flags == [True]
    assert generator_events == ["next response"]
    assert generator._active is None
    assert sent == []


@pytest.mark.parametrize("finish_reason", [None, "stop"])
def test_dspark_peer_parser_failure_prevents_next_response(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str | None,
) -> None:
    sent: list[object] = []
    generator = _dspark_sequential(
        sender=SimpleNamespace(send=sent.append, send_nowait=sent.append)
    )
    task = _fake_dspark_task()
    chunk = TokenChunk(
        model=generator.model_id,
        text="x",
        token_id=7,
        usage=None,
    )
    generator_events = _activate_dspark_response(
        generator,
        task=task,
        chunk=chunk,
        finish_reason=finish_reason,
    )

    def peer_failure(
        _stage: str,
        _group: object,
        operation: Callable[[], object],
    ) -> object:
        operation()
        raise RuntimeError("peer parser failed")

    monkeypatch.setattr(batch_generator, "rank_agreed_fail_stop", peer_failure)

    with pytest.raises(RuntimeError, match="peer parser failed"):
        generator.step()

    assert generator_events == ["next response"]
    assert generator._active is None
    assert len(sent) == 2
    assert sent[0].chunk is chunk
    assert isinstance(sent[1].chunk, batch_generator.ErrorChunk)


def test_cancellation_receiver_failure_is_collectively_fail_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingReceiver:
        def collect(self) -> list[object]:
            raise RuntimeError("receiver failed")

    collective_flags: list[bool] = []

    def any_rank(value: bool, _group: object) -> bool:
        collective_flags.append(value)
        return value

    monkeypatch.setattr(batch_generator, "mx_any", any_rank)
    generator = _sequential_for_callback_test(FailingReceiver())

    with pytest.raises(RuntimeError, match="cancellation collection failed"):
        generator.agree_on_cancellations()

    assert collective_flags == [True]


def test_invalid_task_id_fails_before_task_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collective_flags: list[bool] = []

    def any_rank(value: bool, _group: object) -> bool:
        collective_flags.append(value)
        return value

    def unexpected_gather(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid task IDs must not enter task gather")

    monkeypatch.setattr(batch_generator, "mx_any", any_rank)
    monkeypatch.setattr(batch_generator, "mx_all_gather_tasks", unexpected_gather)
    generator = _sequential_for_callback_test(SimpleNamespace(collect=list))
    generator._maybe_queue = cast(
        list[TextGeneration],
        [SimpleNamespace(task_id="not-a-uuid")],
    )

    with pytest.raises(RuntimeError, match="task-ID preflight failed"):
        generator.agree_on_tasks()

    assert collective_flags == [True]


@dataclass(frozen=True)
class _ScalarToken:
    value: int

    def item(self) -> int:
        return self.value


@dataclass(frozen=True)
class _PromptTokens:
    values: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: int | slice) -> _ScalarToken | _PromptTokens:
        if isinstance(key, slice):
            return _PromptTokens(self.values[key])
        return _ScalarToken(self.values[key])

    def tolist(self) -> list[int]:
        return list(self.values)


@dataclass
class _LocalAgreement:
    stage_outcome: bool | None = True
    token_outcome: int | None = None
    use_local_token: bool = True
    stage_calls: list[bool] = field(default_factory=list)
    token_calls: list[int | None] = field(default_factory=list)

    @property
    def rank(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 2

    def agree_stage_success(self, local_success: bool) -> bool | None:
        self.stage_calls.append(local_success)
        return self.stage_outcome

    def agree_token(self, local_token: int | None) -> int | None:
        self.token_calls.append(local_token)
        return local_token if self.use_local_token else self.token_outcome


def test_dspark_setup_fingerprint_binds_generation_callback_presence() -> None:
    common = {
        "prompt_tokens": cast(object, _PromptTokens((1, 2, 3))),
        "max_tokens": 16,
        "prefill_step_size": 4,
        "capacity_hint": 18,
        "verify_width": 3,
        "seed": 42,
        "is_bench": False,
        "compact_greedy": False,
        "eos_token_ids": (2,),
        "banned_token_ids": (),
        "terminal_token_ids": (2,),
        "stop_sequences": (),
    }

    assert generate_module._dspark_setup_fingerprint(  # type: ignore[arg-type]
        **common,
        generation_progress=False,
    ) != generate_module._dspark_setup_fingerprint(  # type: ignore[arg-type]
        **common,
        generation_progress=True,
    )


@dataclass
class _Detokenizer:
    pieces: dict[int, str]
    last_segment: str = ""

    def reset(self) -> None:
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        self.last_segment = self.pieces.get(token, f"<{token}>")

    def finalize(self) -> None:
        return None


@dataclass
class _ScenarioRoundEngine:
    speculative: list[DSparkRoundResult]
    ordinary: list[DSparkRoundResult] = field(default_factory=list)
    verify_width: int = 3

    def decode_round(self, _anchor_token: int) -> DSparkRoundResult:
        return self.speculative.pop(0)

    def decode_ordinary_tail(self, _anchor_token: int) -> DSparkRoundResult:
        return self.ordinary.pop(0)


def _round(
    tokens: tuple[int, ...],
    *,
    proposed: int,
    accepted: int,
) -> DSparkRoundResult:
    return DSparkRoundResult(
        emitted_tokens=tokens,
        telemetry=DSparkRoundTelemetry(
            round_index=0,
            rank=0,
            draft_ms=1.0,
            target_verify_ms=2.0,
            target_commit_ms=3.0,
            draft_commit_ms=4.0,
            collective_ms=5.0,
            proposed=proposed,
            accepted=accepted,
            emitted=len(tokens),
            fallback=False,
            error=None,
        ),
    )


def _run_mlx_generate_dspark_scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: _ScenarioRoundEngine,
    pieces: dict[int, str],
    eos_ids: tuple[int, ...],
    stop: str | None,
    max_tokens: int,
    seed_contracts: list[dict[str, object]] | None = None,
    setup_agreement: _LocalAgreement | None = None,
    environment: dict[str, str] | None = None,
    barrier_calls: list[object] | None = None,
    runtime_create_error: Exception | None = None,
    generation_callback: Callable[[], None] | None = None,
    lifecycle_events: list[str] | None = None,
) -> list[object]:
    model = object()
    tokenizer = SimpleNamespace(detokenizer=_Detokenizer(pieces))

    def seed_prompt(*_args: object, **kwargs: object) -> tuple[float, int]:
        if seed_contracts is not None:
            seed_contracts.append(
                {
                    "max_tokens": kwargs["max_tokens"],
                    "stop_sequences": kwargs["stop_sequences"],
                }
            )
        return 100.0, 2

    def agree_local_value(name: str, operation: Callable[[], object]) -> object:
        if lifecycle_events is not None:
            lifecycle_events.append(f"agree:{name}")
        return operation()

    def agree_local_side_effect(name: str, operation: Callable[[], None]) -> None:
        if lifecycle_events is not None:
            lifecycle_events.append(f"agree:{name}")
        operation()

    runtime = SimpleNamespace(
        seed_prompt=seed_prompt,
        make_round_engine=lambda: engine,
        agree_local_value=agree_local_value,
        agree_local_side_effect=agree_local_side_effect,
        agree_text=lambda _name, operation: operation(),
        agree_response_control=lambda operation: operation(),
    )

    def create_runtime(*_args: object, **_kwargs: object) -> object:
        if runtime_create_error is not None:
            raise runtime_create_error
        return runtime

    dspark = SimpleNamespace(
        verify_width=engine.verify_width,
        target_model=model,
    )
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    task = SimpleNamespace(
        model=ModelId("kernelpool/Kimi-K3-2bit-UVMAX"),
        seed=42,
        bench=False,
        use_prefix_cache=False,
        repetition_penalty=None,
        repetition_context_size=None,
        presence_penalty=None,
        frequency_penalty=None,
        temperature=0.0,
        top_p=1.0,
        min_p=None,
        top_k=0,
        stop=stop,
        max_output_tokens=max_tokens,
        logprobs=False,
        top_logprobs=None,
        prefill_endpoint=None,
        images=[],
        image_hashes={},
    )

    monkeypatch.setattr(
        generate_module, "_has_pipeline_communication_layer", lambda _m: False
    )
    monkeypatch.setattr(generate_module, "prompt_lookup_config", lambda **_kwargs: None)
    monkeypatch.setattr(
        generate_module, "_validate_dspark_request", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        generate_module, "encode_prompt", lambda *_args: _PromptTokens((1, 2, 10))
    )
    monkeypatch.setattr(
        generate_module, "fix_unmatched_think_end_tokens", lambda tokens, _tok: tokens
    )
    monkeypatch.setattr(generate_module, "system_prompt_token_count", lambda *_args: 0)
    monkeypatch.setattr(generate_module, "make_kv_cache", lambda **_kwargs: [])
    monkeypatch.setattr(generate_module, "make_logits_processors", lambda **_kwargs: [])
    monkeypatch.setattr(generate_module, "make_sampler", lambda **_kwargs: object())

    def observe_barrier(called_group: object) -> None:
        if barrier_calls is not None:
            barrier_calls.append(called_group)
        if lifecycle_events is not None:
            lifecycle_events.append("barrier")

    monkeypatch.setattr(generate_module, "mx_barrier", observe_barrier)
    monkeypatch.setattr(
        generate_module,
        "MlxRankAgreement",
        lambda _group: setup_agreement or _LocalAgreement(),
    )
    monkeypatch.setattr(generate_module, "eos_ids_from_tokenizer", lambda _tok: eos_ids)
    monkeypatch.setattr(
        generate_module.KimiK3DSparkRequestRuntime,
        "create",
        create_runtime,
    )
    monkeypatch.setattr(
        generate_module.mx, "reset_peak_memory", lambda: None, raising=False
    )
    monkeypatch.setattr(
        generate_module.mx,
        "random",
        SimpleNamespace(seed=lambda _seed: None),
        raising=False,
    )
    monkeypatch.setattr(
        generate_module.mx, "array", lambda *_args, **_kwargs: (), raising=False
    )
    monkeypatch.setattr(generate_module.mx, "float32", object(), raising=False)
    monkeypatch.setattr(generate_module.mx, "get_peak_memory", lambda: 0, raising=False)

    with patch.dict("os.environ", environment or {}, clear=True):
        return list(
            generate_module.mlx_generate(
                model=cast(Model, model),
                tokenizer=cast(object, tokenizer),
                task=cast(object, task),
                prompt="prompt",
                kv_prefix_cache=None,
                group=cast(object, group),
                dspark=cast(LoadedMlxDSpark, cast(object, dspark)),
                on_generation_token=generation_callback,
            )
        )


@pytest.mark.parametrize(
    "environment",
    [
        {"EXO_MLX_PREFILL_STEP_SIZE": "malformed"},
        {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "true"},
    ],
)
def test_mlx_generate_rank_agrees_asymmetric_dspark_setup_failure_before_barrier(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    agreement = _LocalAgreement(stage_outcome=None)
    seed_contracts: list[dict[str, object]] = []
    barrier_calls: list[object] = []

    with pytest.raises(
        DSparkDistributedStateError,
        match="setup outcomes or request controls disagreed",
    ):
        _run_mlx_generate_dspark_scenario(
            monkeypatch,
            engine=_ScenarioRoundEngine([]),
            pieces={},
            eos_ids=(),
            stop=None,
            max_tokens=3,
            seed_contracts=seed_contracts,
            setup_agreement=agreement,
            environment=environment,
            barrier_calls=barrier_calls,
        )

    assert agreement.stage_calls == [False]
    assert len(agreement.token_calls) == 5
    assert seed_contracts == []
    assert barrier_calls == []


def test_mlx_generate_rank_agrees_asymmetric_dspark_cache_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreement = _LocalAgreement(stage_outcome=None)
    barrier_calls: list[object] = []

    with pytest.raises(
        DSparkDistributedStateError,
        match="setup outcomes or request controls disagreed",
    ):
        _run_mlx_generate_dspark_scenario(
            monkeypatch,
            engine=_ScenarioRoundEngine([]),
            pieces={},
            eos_ids=(),
            stop=None,
            max_tokens=3,
            setup_agreement=agreement,
            barrier_calls=barrier_calls,
            runtime_create_error=ValueError("rank-local target cache is invalid"),
        )

    assert agreement.stage_calls == [False]
    assert len(agreement.token_calls) == 5
    assert barrier_calls == []


def test_mlx_generate_dspark_text_stop_keeps_round_level_usage_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _run_mlx_generate_dspark_scenario(
        monkeypatch,
        engine=_ScenarioRoundEngine([_round((11, 12, 13), proposed=2, accepted=2)]),
        pieces={11: "a", 12: "STOP", 13: "z"},
        eos_ids=(),
        stop="STOP",
        max_tokens=3,
    )

    assert len(responses) == 2
    final = responses[-1]
    assert final.finish_reason == "stop"
    usage = final.usage
    stats = final.stats
    assert usage.completion_tokens == 2
    assert usage.completion_tokens_details.accepted_prediction_tokens == 2
    assert usage.completion_tokens_details.rejected_prediction_tokens == 0
    assert stats.generation_tokens == 2
    assert stats.speculative_drafted_tokens == 2
    assert stats.speculative_accepted_tokens == 2
    assert stats.speculative_committed_tokens == 3


def test_mlx_generate_binds_normalized_stop_and_max_tokens_before_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_contracts: list[dict[str, object]] = []
    _run_mlx_generate_dspark_scenario(
        monkeypatch,
        engine=_ScenarioRoundEngine(
            [_round((11,), proposed=0, accepted=0)],
        ),
        pieces={11: "STOP"},
        eos_ids=(),
        stop="STOP",
        max_tokens=7,
        seed_contracts=seed_contracts,
    )

    assert seed_contracts == [
        {
            "max_tokens": 7,
            "stop_sequences": ("STOP",),
        }
    ]


def test_mlx_generate_dspark_first_token_stop_keeps_usage_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _run_mlx_generate_dspark_scenario(
        monkeypatch,
        engine=_ScenarioRoundEngine([_round((11, 12, 13), proposed=2, accepted=2)]),
        pieces={11: "STOP", 12: "b", 13: "c"},
        eos_ids=(),
        stop="STOP",
        max_tokens=3,
    )

    assert len(responses) == 1
    final = responses[-1]
    usage = final.usage
    stats = final.stats
    assert usage.completion_tokens == 1
    assert usage.completion_tokens_details.accepted_prediction_tokens == 1
    assert usage.completion_tokens_details.rejected_prediction_tokens == 0
    assert stats.speculative_accepted_tokens == 2
    assert stats.speculative_committed_tokens == 3


def test_mlx_generate_dspark_eos_reports_committed_prefix_and_public_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _run_mlx_generate_dspark_scenario(
        monkeypatch,
        engine=_ScenarioRoundEngine([_round((11, 12), proposed=2, accepted=1)]),
        pieces={11: "a", 12: ""},
        eos_ids=(12,),
        stop=None,
        max_tokens=3,
    )

    final = responses[-1]
    assert len(responses) == 2
    assert final.finish_reason == "stop"
    usage = final.usage
    stats = final.stats
    assert usage.completion_tokens == 2
    assert usage.completion_tokens_details.accepted_prediction_tokens == 1
    assert usage.completion_tokens_details.rejected_prediction_tokens == 1
    assert stats.speculative_committed_tokens == 2


def test_mlx_generate_dspark_max_length_reports_final_stats_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _run_mlx_generate_dspark_scenario(
        monkeypatch,
        engine=_ScenarioRoundEngine([_round((11, 12, 13), proposed=2, accepted=2)]),
        pieces={11: "a", 12: "b", 13: "c"},
        eos_ids=(),
        stop=None,
        max_tokens=3,
    )

    final = responses[-1]
    assert len(responses) == 3
    assert final.finish_reason == "length"
    usage = final.usage
    stats = final.stats
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 6
    assert stats.generation_tokens == 3
    assert stats.speculative_rounds == 1
    assert stats.speculative_drafted_tokens == 2
    assert stats.speculative_accepted_tokens == 2
    assert stats.speculative_committed_tokens == 3


def test_mlx_generate_dspark_completes_terminal_barrier_before_yield_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []

    _run_mlx_generate_dspark_scenario(
        monkeypatch,
        engine=_ScenarioRoundEngine(
            [],
            ordinary=[_round((11,), proposed=0, accepted=0)],
        ),
        pieces={11: "done"},
        eos_ids=(),
        stop=None,
        max_tokens=1,
        generation_callback=lambda: lifecycle.append("callback"),
        lifecycle_events=lifecycle,
    )

    assert lifecycle[-4:] == [
        "agree:public response construction",
        "agree:generation progress callback",
        "callback",
        "barrier",
    ]
