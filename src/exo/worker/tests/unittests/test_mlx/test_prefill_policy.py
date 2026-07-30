from __future__ import annotations

import pytest

from exo.worker.engines.mlx.generator.generate import (
    _memory_from_mlx_decimal_gb,
    _prefill_memory_log_interval,
    _prefill_step_size,
)


def test_prefill_step_default_preserves_nominal_pipeline_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_MLX_PREFILL_STEP_SIZE", raising=False)
    monkeypatch.delenv("EXO_MLX_PIPELINE_LONG_CONTEXT_MIN_TOKENS", raising=False)
    monkeypatch.delenv("EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE", raising=False)
    monkeypatch.delenv(
        "EXO_MLX_PIPELINE_MAX_ATTENTION_CELLS_PER_CHUNK",
        raising=False,
    )

    assert _prefill_step_size(65_536, is_pipeline=False) == 4096
    assert (
        _prefill_step_size(
            65_536,
            is_pipeline=True,
            pipeline_divisor=2,
        )
        == 4096
    )


def test_attention_cell_budget_scales_effective_pipeline_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PREFILL_STEP_SIZE", "8192")
    monkeypatch.setenv("EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE", "8192")
    monkeypatch.setenv(
        "EXO_MLX_PIPELINE_MAX_ATTENTION_CELLS_PER_CHUNK",
        str(65_536 * 4096),
    )

    assert (
        _prefill_step_size(
            65_536,
            is_pipeline=True,
            pipeline_divisor=2,
        )
        // 2
        == 4096
    )
    assert (
        _prefill_step_size(
            131_072,
            is_pipeline=True,
            pipeline_divisor=2,
        )
        // 2
        == 2048
    )
    assert (
        _prefill_step_size(
            262_144,
            is_pipeline=True,
            pipeline_divisor=2,
        )
        // 2
        == 1024
    )


def test_attention_cell_budget_scales_tensor_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PREFILL_STEP_SIZE", "4096")
    monkeypatch.setenv("EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE", "4096")
    monkeypatch.setenv(
        "EXO_MLX_PIPELINE_MAX_ATTENTION_CELLS_PER_CHUNK",
        str(65_536 * 4096),
    )

    assert _prefill_step_size(65_536, is_pipeline=False) == 4096
    assert _prefill_step_size(131_072, is_pipeline=False) == 2048
    assert _prefill_step_size(262_144, is_pipeline=False) == 1024
    assert _prefill_step_size(524_288, is_pipeline=False) == 512


def test_attention_cell_budget_snaps_to_power_of_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PREFILL_STEP_SIZE", "4096")
    monkeypatch.setenv("EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE", "4096")
    monkeypatch.setenv(
        "EXO_MLX_PIPELINE_MAX_ATTENTION_CELLS_PER_CHUNK",
        str(65_536 * 4096),
    )

    assert _prefill_step_size(122_000, is_pipeline=False) == 2048


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "EXO_MLX_PREFILL_STEP_SIZE",
            "3",
            "EXO_MLX_PREFILL_STEP_SIZE must be at least 4",
        ),
        (
            "EXO_MLX_PIPELINE_LONG_CONTEXT_MIN_TOKENS",
            "0",
            "EXO_MLX_PIPELINE_LONG_CONTEXT_MIN_TOKENS must be positive",
        ),
        (
            "EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE",
            "3",
            "EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE must be at least 4",
        ),
        (
            "EXO_MLX_PIPELINE_MAX_ATTENTION_CELLS_PER_CHUNK",
            "-1",
            "EXO_MLX_MAX_ATTENTION_CELLS_PER_CHUNK cannot be negative",
        ),
    ],
)
def test_prefill_policy_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        _prefill_step_size(
            65_536,
            is_pipeline=True,
            pipeline_divisor=2,
        )


def test_prefill_memory_log_interval_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_MLX_PREFILL_MEMORY_LOG_INTERVAL", raising=False)
    assert _prefill_memory_log_interval() == 0

    monkeypatch.setenv("EXO_MLX_PREFILL_MEMORY_LOG_INTERVAL", "8")
    assert _prefill_memory_log_interval() == 8

    monkeypatch.setenv("EXO_MLX_PREFILL_MEMORY_LOG_INTERVAL", "-1")
    with pytest.raises(
        ValueError,
        match="EXO_MLX_PREFILL_MEMORY_LOG_INTERVAL cannot be negative",
    ):
        _prefill_memory_log_interval()


def test_mlx_decimal_gigabytes_convert_to_bytes_without_gib_inflation() -> None:
    memory = _memory_from_mlx_decimal_gb(453.126599178)

    assert memory.in_bytes == 453_126_599_178
