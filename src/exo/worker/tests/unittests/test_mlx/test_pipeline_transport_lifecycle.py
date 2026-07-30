from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator import generate


class _FakeCacheEntry:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.state = mx.array([], dtype=mx.int32)
        self.trim_calls: list[int] = []

    def append(self, tokens: list[int]) -> None:
        self.tokens.extend(tokens)
        self.state = mx.array(self.tokens, dtype=mx.int32)

    def trim(self, count: int) -> None:
        self.trim_calls.append(count)
        self.tokens = self.tokens[:-count]
        self.state = mx.array(self.tokens, dtype=mx.int32)


class _KimiK3FakeModel:
    """Small callable with the same module identity as the upstream K3 model."""

    __module__ = "mlx_lm.models.kimi_k3"

    def __init__(self) -> None:
        self.layers: list[object] = []
        self.calls: list[list[int]] = []

    def __call__(self, tokens: mx.array, *, cache: list[_FakeCacheEntry]) -> None:
        token_list = [int(token) for token in tokens.reshape(-1).tolist()]
        self.calls.append(token_list)
        cache[0].append(token_list)


class _GenericFakeModel(_KimiK3FakeModel):
    __module__ = __name__


class _RankZeroSingletonGroup:
    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def size() -> int:
        return 1


def _run_fake_pipeline_prefill(
    monkeypatch: pytest.MonkeyPatch,
    model: _KimiK3FakeModel,
) -> _FakeCacheEntry:
    monkeypatch.setattr(
        generate,
        "maybe_quantize_kv_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(generate, "clear_prefill_sends", lambda: None)
    monkeypatch.setattr(generate, "flush_prefill_sends", lambda: None)
    cache_entry = _FakeCacheEntry()
    generate.pipeline_parallel_prefill(
        model=cast(Any, model),
        prompt=mx.array([10, 11, 12, 13], dtype=mx.int32),
        prompt_cache=cast(Any, [cache_entry]),
        prefill_step_size=16,
        kv_group_size=None,
        kv_bits=None,
        prompt_progress_callback=lambda *_args: None,
        distributed_prompt_progress_callback=None,
        group=cast(Any, _RankZeroSingletonGroup()),
    )
    return cache_entry


def test_kimi_k3_pipeline_prefill_stops_at_decode_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _KimiK3FakeModel()

    cache_entry = _run_fake_pipeline_prefill(monkeypatch, model)

    # The prefill argument is full_prompt[:-1]. Its last token is intentionally
    # left for stream_generate(full_prompt[-2:]), so only prompt[:-1] is cached.
    assert model.calls == [[10, 11, 12]]
    assert cache_entry.tokens == [10, 11, 12]
    assert cache_entry.trim_calls == []


def test_generic_pipeline_prefill_keeps_legacy_disposable_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _GenericFakeModel()

    cache_entry = _run_fake_pipeline_prefill(monkeypatch, model)

    assert model.calls == [[10, 11, 12], [13], [13]]
    assert cache_entry.tokens == [10, 11, 12, 13, 13]


def test_kimi_k3_prefill_skips_snapshots_and_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _KimiK3FakeModel()
    cache_entry = _FakeCacheEntry()
    monkeypatch.setattr(generate, "_has_pipeline_communication_layer", lambda _: True)
    monkeypatch.setattr(generate, "has_non_kv_caches", lambda _: True)
    monkeypatch.setattr(
        generate,
        "snapshot_ssm_states",
        lambda _: pytest.fail("rollback-free K3 prefill must not snapshot"),
    )
    monkeypatch.setattr(
        generate,
        "maybe_quantize_kv_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(generate, "mx_barrier", lambda _: None)
    monkeypatch.setattr(generate, "clear_prefill_sends", lambda: None)
    monkeypatch.setattr(generate, "flush_prefill_sends", lambda: None)
    monkeypatch.setattr(generate, "set_pipeline_prefill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        generate,
        "set_pipeline_queue_sends",
        lambda *_args, **_kwargs: None,
    )

    _, num_tokens, snapshots = generate.prefill(
        model=cast(Any, model),
        tokenizer=cast(Any, object()),
        sampler=lambda logits: logits,
        prompt_tokens=mx.array([20, 21, 22], dtype=mx.int32),
        cache=cast(Any, [cache_entry]),
        group=cast(Any, _RankZeroSingletonGroup()),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert num_tokens == 3
    assert snapshots == []
    assert model.calls == [[20, 21]]
    assert cache_entry.tokens == [20, 21]
    assert cache_entry.trim_calls == []


def test_kimi_k3_tensor_prefill_stops_at_decode_boundary_without_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TwoCacheKimiModel(_KimiK3FakeModel):
        __module__ = "mlx_lm.models.kimi_k3"

        def __call__(
            self,
            tokens: mx.array,
            *,
            cache: list[_FakeCacheEntry],
        ) -> None:
            token_list = [int(token) for token in tokens.reshape(-1).tolist()]
            self.calls.append(token_list)
            for entry in cache:
                entry.append(token_list)

    model = _TwoCacheKimiModel()
    cache_entries = [_FakeCacheEntry(), _FakeCacheEntry()]
    progress: list[tuple[int, int]] = []
    monkeypatch.setattr(generate, "_has_pipeline_communication_layer", lambda _: False)
    monkeypatch.setattr(generate, "has_non_kv_caches", lambda _: True)
    monkeypatch.setattr(
        generate,
        "snapshot_ssm_states",
        lambda _: pytest.fail("exact K3 tensor prefill must not snapshot"),
    )
    monkeypatch.setattr(
        generate,
        "stream_generate",
        lambda *_args, **_kwargs: pytest.fail(
            "exact K3 tensor prefill must not sample"
        ),
    )
    monkeypatch.setattr(
        generate,
        "maybe_quantize_kv_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(generate, "mx_barrier", lambda _: None)
    monkeypatch.setattr(generate, "set_pipeline_prefill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        generate,
        "set_pipeline_queue_sends",
        lambda *_args, **_kwargs: None,
    )

    _, num_tokens, snapshots = generate.prefill(
        model=cast(Any, model),
        tokenizer=cast(Any, object()),
        sampler=lambda logits: logits,
        prompt_tokens=mx.array([20, 21, 22, 23], dtype=mx.int32),
        cache=cast(Any, cache_entries),
        group=cast(Any, _RankZeroSingletonGroup()),
        on_prefill_progress=lambda done, total: progress.append((done, total)),
        distributed_prompt_progress_callback=None,
    )

    assert num_tokens == 4
    assert snapshots == []
    assert model.calls == [[20, 21, 22]]
    assert [entry.tokens for entry in cache_entries] == [
        [20, 21, 22],
        [20, 21, 22],
    ]
    assert all(entry.trim_calls == [] for entry in cache_entries)
    assert progress == [(0, 4), (3, 4), (4, 4)]


@pytest.mark.parametrize(
    ("num_tokens", "expected_chunk_sizes"),
    [
        (1, []),
        (4097, [4096]),
        (8193, [4096, 4096]),
    ],
)
def test_kimi_k3_exact_prefill_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    expected_chunk_sizes: list[int],
) -> None:
    model = _KimiK3FakeModel()
    cache_entry = _FakeCacheEntry()
    progress: list[tuple[int, int]] = []
    monkeypatch.setattr(
        generate,
        "maybe_quantize_kv_cache",
        lambda *_args, **_kwargs: None,
    )

    generate.kimi_k3_exact_prefill(
        model=cast(Any, model),
        prompt=mx.arange(num_tokens, dtype=mx.int32),
        prompt_cache=cast(Any, [cache_entry]),
        prefill_step_size=4096,
        kv_group_size=None,
        kv_bits=None,
        prompt_progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert [len(call) for call in model.calls] == expected_chunk_sizes
    assert cache_entry.tokens == list(range(max(0, num_tokens - 1)))
    expected_progress = [(0, num_tokens)]
    processed = 0
    for chunk_size in expected_chunk_sizes:
        processed += chunk_size
        expected_progress.append((processed, num_tokens))
    expected_progress.append((num_tokens, num_tokens))
    assert progress == expected_progress


def test_token_relay_scope_restores_state_when_generator_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    transitions: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        generate,
        "set_pipeline_token_relay",
        lambda actual, enabled: transitions.append((actual, enabled)),
    )

    def suspended_decode() -> Generator[str]:
        with generate._pipeline_token_relay_scope(
            cast(Any, model),
            enabled=True,
        ):
            yield "token"

    stream = suspended_decode()
    assert next(stream) == "token"
    assert transitions == [(model, True)]

    stream.close()
    assert transitions == [(model, True), (model, False)]


def test_token_relay_scope_restores_state_on_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    transitions: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        generate,
        "set_pipeline_token_relay",
        lambda actual, enabled: transitions.append((actual, enabled)),
    )

    with (
        pytest.raises(RuntimeError, match="relay initialization failed"),
        generate._pipeline_token_relay_scope(
            cast(Any, model),
            enabled=True,
        ),
    ):
        raise RuntimeError("relay initialization failed")

    assert transitions == [(model, True), (model, False)]


def test_agreed_prefill_cancellation_discards_before_generic_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        generate,
        "discard_unsent_prefill_sends_after_agreed_cancel",
        lambda: events.append("discard"),
    )
    monkeypatch.setattr(
        generate,
        "clear_prefill_sends",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        generate,
        "maybe_quantize_kv_cache",
        lambda *_args, **_kwargs: None,
    )

    class _RankZeroGroup:
        @staticmethod
        def rank() -> int:
            return 0

        @staticmethod
        def size() -> int:
            return 2

    def cancel() -> None:
        raise generate.PrefillCancelled()

    model = _KimiK3FakeModel()
    with pytest.raises(generate.PrefillCancelled):
        generate.pipeline_parallel_prefill(
            model=cast(Any, model),
            prompt=mx.array([1, 2], dtype=mx.int32),
            prompt_cache=cast(Any, [_FakeCacheEntry()]),
            prefill_step_size=4,
            kv_group_size=None,
            kv_bits=None,
            prompt_progress_callback=lambda *_args: None,
            distributed_prompt_progress_callback=cancel,
            group=cast(Any, _RankZeroGroup()),
        )

    assert events == ["clear", "discard", "clear"]
    assert model.calls == [[1]]
