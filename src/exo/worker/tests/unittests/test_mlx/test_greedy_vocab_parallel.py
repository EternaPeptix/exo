from unittest.mock import patch

import pytest

from exo.worker.engines.mlx.generator.generate import (
    greedy_vocab_parallel_stream_kwargs,
)


def _kwargs(**overrides):
    values = {
        "temperature": 0.0,
        "logprobs": False,
        "has_logits_processors": False,
        "is_pipeline": False,
        "speculative": False,
    }
    values.update(overrides)
    return values


def test_compact_greedy_is_default_off():
    with patch.dict("os.environ", {}, clear=True):
        assert greedy_vocab_parallel_stream_kwargs(**_kwargs()) == {}


def test_compact_greedy_exact_request_shape():
    with patch.dict(
        "os.environ",
        {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "1"},
        clear=True,
    ):
        assert greedy_vocab_parallel_stream_kwargs(**_kwargs()) == {
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
def test_compact_greedy_falls_back_for_incompatible_requests(override):
    with patch.dict(
        "os.environ",
        {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "1"},
        clear=True,
    ):
        assert greedy_vocab_parallel_stream_kwargs(**_kwargs(**override)) == {}


def test_compact_greedy_rejects_malformed_opt_in():
    with patch.dict(
        "os.environ",
        {"EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "true"},
        clear=True,
    ), pytest.raises(
        ValueError,
        match="EXO_MLX_K3_VOCAB_PARALLEL_GREEDY must be 0 or 1",
    ):
        greedy_vocab_parallel_stream_kwargs(**_kwargs())
