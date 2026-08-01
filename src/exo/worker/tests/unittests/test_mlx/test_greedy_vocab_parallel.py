from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.generator import generate as generate_module
from exo.worker.engines.mlx.generator.generate import (
    greedy_vocab_parallel_stream_kwargs,
    warmup_inference,
)
from exo.worker.engines.mlx.generator.kimi_k3_dspark import (
    DSparkRoundResult,
    DSparkRoundTelemetry,
    LoadedMlxDSpark,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.llm_inference import batch_generator
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator


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
        assert greedy_vocab_parallel_stream_kwargs(  # type: ignore[arg-type]
            **_kwargs(**override)
        ) == {}


def test_compact_greedy_rejects_malformed_opt_in() -> None:
    with patch.dict(
        "os.environ",
        {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "true"},
        clear=True,
    ), pytest.raises(
        ValueError,
        match="EXO_MLX_K3_VOCAB_PARALLEL_GREEDY must be 0 or 1",
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
) -> list[object]:
    model = object()
    tokenizer = SimpleNamespace(detokenizer=_Detokenizer(pieces))
    runtime = SimpleNamespace(
        seed_prompt=lambda *_args, **_kwargs: (100.0, 2),
        make_round_engine=lambda: engine,
    )
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

    monkeypatch.setattr(generate_module, "_has_pipeline_communication_layer", lambda _m: False)
    monkeypatch.setattr(generate_module, "prompt_lookup_config", lambda **_kwargs: None)
    monkeypatch.setattr(generate_module, "_validate_dspark_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generate_module, "encode_prompt", lambda *_args: _PromptTokens((1, 2, 10)))
    monkeypatch.setattr(generate_module, "fix_unmatched_think_end_tokens", lambda tokens, _tok: tokens)
    monkeypatch.setattr(generate_module, "system_prompt_token_count", lambda *_args: 0)
    monkeypatch.setattr(generate_module, "make_kv_cache", lambda **_kwargs: [])
    monkeypatch.setattr(generate_module, "make_logits_processors", lambda **_kwargs: [])
    monkeypatch.setattr(generate_module, "make_sampler", lambda **_kwargs: object())
    monkeypatch.setattr(generate_module, "mx_barrier", lambda _group: None)
    monkeypatch.setattr(generate_module, "eos_ids_from_tokenizer", lambda _tok: eos_ids)
    monkeypatch.setattr(
        generate_module.KimiK3DSparkRequestRuntime,
        "create",
        lambda *_args, **_kwargs: runtime,
    )
    monkeypatch.setattr(generate_module.mx, "reset_peak_memory", lambda: None, raising=False)
    monkeypatch.setattr(
        generate_module.mx,
        "random",
        SimpleNamespace(seed=lambda _seed: None),
        raising=False,
    )
    monkeypatch.setattr(generate_module.mx, "array", lambda *_args, **_kwargs: (), raising=False)
    monkeypatch.setattr(generate_module.mx, "float32", object(), raising=False)
    monkeypatch.setattr(generate_module.mx, "get_peak_memory", lambda: 0, raising=False)

    with patch.dict("os.environ", {}, clear=True):
        return list(
            generate_module.mlx_generate(
                model=cast(Model, model),
                tokenizer=cast(object, tokenizer),
                task=cast(object, task),
                prompt="prompt",
                kv_prefix_cache=None,
                group=cast(object, group),
                dspark=cast(LoadedMlxDSpark, cast(object, dspark)),
            )
        )


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
