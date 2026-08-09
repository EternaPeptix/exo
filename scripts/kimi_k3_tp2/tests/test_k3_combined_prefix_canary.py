from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import k3_combined_prefix_canary as canary  # noqa: E402


class _Tokenizer:
    def __init__(self, rendered: dict[str, list[int]]) -> None:
        self.rendered = rendered

    def apply_chat_template(self, messages: object, **_kwargs: object) -> list[int]:
        assert isinstance(messages, list)
        return self.rendered[str(messages[0]["content"])]


def _fixture_tokens() -> tuple[_Tokenizer, dict[str, list[dict[str, str]]]]:
    base = list(range(canary.BASE_PROMPT_TOKENS))
    tokenizer = _Tokenizer(
        {
            "spec": list(range(canary.SPECULATIVE_PROMPT_TOKENS)),
            "base": base,
            "append": [*base, 99],
        }
    )
    fixture = {
        "speculative_4300": [{"role": "user", "content": "spec"}],
        "base_32768": [{"role": "user", "content": "base"}],
        "append_32769": [{"role": "user", "content": "append"}],
    }
    return tokenizer, fixture


def test_fixture_requires_exact_lengths_and_one_token_append() -> None:
    tokenizer, fixture = _fixture_tokens()

    prompts = canary.validate_fixture(tokenizer, fixture)

    assert len(prompts["speculative_4300"]["tokens"]) == 4_300
    assert len(prompts["base_32768"]["tokens"]) == 32_768
    assert len(prompts["append_32769"]["tokens"]) == 32_769


def test_fixture_rejects_non_prefix_append() -> None:
    tokenizer, fixture = _fixture_tokens()
    tokenizer.rendered["append"][0] = -1

    with pytest.raises(RuntimeError, match="strict one-token append"):
        canary.validate_fixture(tokenizer, fixture)


def _row(
    label: str,
    length: int,
    enabled: bool,
    hit: str,
    cached: int,
    speculative: bool,
    digest: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "prompt_tokens": length,
        "use_prefix_cache": enabled,
        "cached_tokens": cached,
        "prompt_token_sha256": digest,
        "completion_sha256": f"completion-{digest}",
        "usage": {
            "prompt_tokens": length,
            "completion_tokens": canary.DECODE_TOKENS,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
        "generation_stats": {
            "prefix_cache_hit": hit,
            "generation_tokens": canary.DECODE_TOKENS,
            "generation_tps": 21.0,
            "speculative_drafted_tokens": 20 if speculative else 0,
            "speculative_accepted_tokens": 12 if speculative else 0,
            "speculative_committed_tokens": canary.DECODE_TOKENS,
            "speculative_error_rounds": 0,
            "speculative_fallback_rounds": 0,
        },
    }


def _passing_rows() -> list[dict[str, Any]]:
    return [
        _row("spec_control_4300", 4_300, False, "none", 0, True, "spec"),
        _row("spec_requested_4300", 4_300, True, "none", 0, True, "spec"),
        _row("cold_base_32768", 32_768, True, "none", 0, False, "base"),
        _row("exact_base_32768", 32_768, True, "exact", 32_767, False, "base"),
        _row("cold_append_32769", 32_769, False, "none", 0, False, "append"),
        _row(
            "cached_append_32769",
            32_769,
            True,
            "partial",
            32_767,
            False,
            "append",
        ),
        _row(
            "exact_append_32769",
            32_769,
            True,
            "exact",
            32_768,
            False,
            "append",
        ),
    ]


def test_analysis_accepts_decode_preservation_and_exact_append_parity() -> None:
    report = canary.analyze(_passing_rows())

    assert report["status"] == "pass"
    assert report["exact_32768_parity"] is True
    assert report["append_32769_parity"] is True


def test_analysis_fails_closed_on_speculative_cache_or_tps_regression() -> None:
    rows = _passing_rows()
    rows[1]["cached_tokens"] = 4_299
    with pytest.raises(RuntimeError, match="contract drifted"):
        canary.analyze(rows)

    rows = _passing_rows()
    rows[1]["generation_stats"]["generation_tps"] = 19.9
    with pytest.raises(RuntimeError, match="throughput was not preserved"):
        canary.analyze(rows)


def test_analysis_fails_closed_on_completion_mismatch() -> None:
    rows = _passing_rows()
    rows[-1]["completion_sha256"] = "different"

    with pytest.raises(RuntimeError, match="parity failed"):
        canary.analyze(rows)
