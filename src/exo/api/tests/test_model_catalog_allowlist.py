# pyright: reportPrivateUsage=false
import pytest

from exo.api.main import (
    _filter_advertised_model_cards,
    _parse_advertised_model_ids,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory


def _card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory(),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def test_parse_advertised_model_ids_defaults_to_unfiltered() -> None:
    assert _parse_advertised_model_ids(None) is None


def test_parse_advertised_model_ids_accepts_exact_order_independent_set() -> None:
    assert _parse_advertised_model_ids("org/model-a,org/model-b") == frozenset(
        (ModelId("org/model-a"), ModelId("org/model-b"))
    )


@pytest.mark.parametrize("raw", ("", ",org/model", "org/model,", "org/model, other/model"))
def test_parse_advertised_model_ids_rejects_empty_or_padded_values(raw: str) -> None:
    with pytest.raises(ValueError, match="non-empty model IDs"):
        _parse_advertised_model_ids(raw)


def test_parse_advertised_model_ids_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate model IDs"):
        _parse_advertised_model_ids("org/model,org/model")


def test_filter_advertised_model_cards_preserves_cache_when_unset() -> None:
    cards = [_card("org/model-a"), _card("org/model-b")]
    assert _filter_advertised_model_cards(cards, None) == cards


def test_filter_advertised_model_cards_returns_only_allowlisted_card() -> None:
    cards = [_card("org/model-a"), _card("org/model-b")]
    assert _filter_advertised_model_cards(
        cards, frozenset((ModelId("org/model-b"),))
    ) == [cards[1]]


def test_filter_advertised_model_cards_fails_closed_on_missing_card() -> None:
    with pytest.raises(RuntimeError, match="org/missing"):
        _filter_advertised_model_cards(
            [_card("org/model")], frozenset((ModelId("org/missing"),))
        )
