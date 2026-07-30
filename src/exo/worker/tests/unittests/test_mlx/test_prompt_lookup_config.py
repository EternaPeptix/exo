from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import pytest

import exo.worker.engines.mlx.generator.generate as generate_module
from exo.worker.engines.mlx.generator.generate import (
    _PromptLookupTelemetry,
    prompt_lookup_config,
    prompt_lookup_stream_kwargs,
)

_PROMPT_LOOKUP_ENV = (
    "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS",
    "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE",
    "EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY",
)


@dataclass
class _RoundStats:
    round_index: int
    source: str
    drafted_tokens: int
    accepted_tokens: int
    committed_tokens: int
    target_cache_tokens: int
    cancelled: bool


@pytest.fixture(autouse=True)
def clear_prompt_lookup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROMPT_LOOKUP_ENV:
        monkeypatch.delenv(name, raising=False)


def test_prompt_lookup_is_inert_by_default() -> None:
    config = prompt_lookup_config(is_pipeline=False, is_batch=False)

    assert config is None
    assert prompt_lookup_stream_kwargs(config, None, mx.array([1, 2])) is None


def test_prompt_lookup_passes_explicit_full_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS", "7")
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE", "6")
    history = mx.array([11, 12, 11, 12, 13])

    config = prompt_lookup_config(is_pipeline=False, is_batch=False)
    kwargs = prompt_lookup_stream_kwargs(config, None, history)

    assert kwargs is not None
    assert kwargs["prompt_lookup_num_tokens"] == 7
    assert kwargs["prompt_lookup_max_ngram_size"] == 6
    assert kwargs["prompt_lookup_history"] is history
    assert kwargs["speculative_round_callback"] is None


def test_round_telemetry_is_opt_in_and_reports_every_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS", "3")
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY", "1")
    logged: list[str] = []

    def capture_log(message: str) -> None:
        logged.append(message)

    monkeypatch.setattr(
        generate_module.logger,
        "info",
        capture_log,
    )

    config = prompt_lookup_config(is_pipeline=False, is_batch=False)
    kwargs = prompt_lookup_stream_kwargs(config, None, mx.array([1, 2]))
    assert kwargs is not None
    callback = kwargs["speculative_round_callback"]
    assert callback is not None
    callback(
        _RoundStats(
            round_index=4,
            source="prompt_lookup",
            drafted_tokens=3,
            accepted_tokens=2,
            committed_tokens=3,
            target_cache_tokens=3,
            cancelled=False,
        )
    )

    assert logged == [
        "MLX prompt-lookup round: rank=0, round=4, source=prompt_lookup, "
        "drafted=3, accepted=2, committed=3, target_cache=3, cancelled=False"
    ]


def test_round_callback_accumulates_without_verbose_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS", "1")
    logged: list[str] = []
    monkeypatch.setattr(generate_module.logger, "info", logged.append)
    telemetry = _PromptLookupTelemetry()

    config = prompt_lookup_config(is_pipeline=False, is_batch=False)
    kwargs = prompt_lookup_stream_kwargs(
        config,
        None,
        mx.array([1, 2]),
        telemetry,
    )
    assert kwargs is not None
    callback = kwargs["speculative_round_callback"]
    assert callback is not None
    callback(
        _RoundStats(
            round_index=0,
            source="prompt_lookup",
            drafted_tokens=1,
            accepted_tokens=1,
            committed_tokens=2,
            target_cache_tokens=2,
            cancelled=False,
        )
    )
    callback(
        _RoundStats(
            round_index=1,
            source="prompt_lookup",
            drafted_tokens=1,
            accepted_tokens=0,
            committed_tokens=1,
            target_cache_tokens=1,
            cancelled=False,
        )
    )

    assert logged == []
    assert telemetry.rounds == 2
    assert telemetry.drafted_tokens == 2
    assert telemetry.accepted_tokens == 1
    assert telemetry.committed_tokens == 3


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS",
            "0",
            "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS must be an integer between 1 and 7",
        ),
        (
            "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS",
            "8",
            "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS must be an integer between 1 and 7",
        ),
        (
            "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS",
            " 2",
            "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS must be an integer between 1 and 7",
        ),
        (
            "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE",
            "1",
            "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE must be an integer between 2 and 64",
        ),
        (
            "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE",
            "65",
            "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE must be an integer between 2 and 64",
        ),
        (
            "EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY",
            "true",
            "EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY must be 0 or 1",
        ),
    ],
)
def test_prompt_lookup_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS", "2")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        prompt_lookup_config(is_pipeline=False, is_batch=False)


@pytest.mark.parametrize(
    "companion",
    [
        "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE",
        "EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY",
    ],
)
def test_prompt_lookup_rejects_companion_without_enablement(
    monkeypatch: pytest.MonkeyPatch,
    companion: str,
) -> None:
    monkeypatch.setenv(companion, "4" if "NGRAM" in companion else "0")

    with pytest.raises(
        ValueError,
        match=f"{companion} requires EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS",
    ):
        prompt_lookup_config(is_pipeline=False, is_batch=False)


@pytest.mark.parametrize(
    ("is_pipeline", "is_batch", "message"),
    [
        (True, False, "does not support pipeline parallelism"),
        (False, True, "does not support batch generation"),
    ],
)
def test_prompt_lookup_rejects_unsupported_generation_modes(
    monkeypatch: pytest.MonkeyPatch,
    is_pipeline: bool,
    is_batch: bool,
    message: str,
) -> None:
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS", "2")

    with pytest.raises(ValueError, match=message):
        prompt_lookup_config(is_pipeline=is_pipeline, is_batch=is_batch)


def test_prompt_lookup_requires_useful_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS", "2")
    config = prompt_lookup_config(is_pipeline=False, is_batch=False)

    with pytest.raises(ValueError, match="requires at least two tokens"):
        prompt_lookup_stream_kwargs(config, None, mx.array([1]))
