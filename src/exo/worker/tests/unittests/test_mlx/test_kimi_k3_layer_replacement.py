from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx import auto_parallel


class _FakeKimiK3Layer(nn.Module):
    def __init__(self, *, is_linear: bool) -> None:
        super().__init__()
        self.is_linear = is_linear

    def __call__(
        self,
        x: mx.array,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        return x


class _FakeKimiK3TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers: list[_FakeKimiK3Layer] = []
        self.start_idx = 17
        self.end_idx = 61
        self.num_layers = 44
        self.ssm_idx: int | None = 23
        self.attn_idx: int | None = 29
        self.cache_index_refreshes = 0

    def _set_cache_indices(
        self,
        layers: Sequence[_FakeKimiK3Layer] | None = None,
    ) -> None:
        assert layers is None
        active_layers = self.layers
        self.cache_index_refreshes += 1
        self.ssm_idx = next(
            (index for index, layer in enumerate(active_layers) if layer.is_linear),
            None,
        )
        self.attn_idx = next(
            (index for index, layer in enumerate(active_layers) if not layer.is_linear),
            None,
        )


class _FakeKimiK3Model(nn.Module):
    __module__ = "mlx_lm.models.kimi_k3"

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeKimiK3TextModel()


@pytest.mark.parametrize(
    ("linear_pattern", "expected_ssm_idx", "expected_attn_idx"),
    (
        ((True, True, False), 0, 2),
        ((False, True, False), 1, 0),
        ((True, True), 0, None),
        ((False, False), None, 0),
    ),
)
def test_kimi_k3_layer_replacement_refreshes_local_cache_indices(
    linear_pattern: tuple[bool, ...],
    expected_ssm_idx: int | None,
    expected_attn_idx: int | None,
) -> None:
    model = _FakeKimiK3Model()
    layers = [_FakeKimiK3Layer(is_linear=is_linear) for is_linear in linear_pattern]

    auto_parallel._set_layers(  # pyright: ignore[reportPrivateUsage]
        model,
        layers,  # pyright: ignore[reportArgumentType]
    )

    assert model.model.layers == layers
    assert model.model.start_idx == 0
    assert model.model.end_idx == len(layers)
    assert model.model.num_layers == len(layers)
    assert model.model.cache_index_refreshes == 1
    assert model.model.ssm_idx == expected_ssm_idx
    assert model.model.attn_idx == expected_attn_idx
