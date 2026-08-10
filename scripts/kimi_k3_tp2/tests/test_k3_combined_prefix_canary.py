from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import k3_combined_prefix_canary as canary  # noqa: E402


class _Tokenizer:
    def apply_chat_template(self, messages: object, **_kwargs: object) -> str:
        assert isinstance(messages, list)
        assert isinstance(messages[0], dict)
        return str(messages[0]["content"])

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: object, **_kwargs: object) -> str:
        assert isinstance(tokens, list)
        return "".join(chr(token) for token in tokens)


class _TemplateMutatingTokenizer(_Tokenizer):
    def apply_chat_template(self, messages: object, **_kwargs: object) -> str:
        rendered = super().apply_chat_template(messages)
        return "T" + rendered[1:]


def _fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Tokenizer, dict[str, object]]:
    tokenizer = _Tokenizer()
    sealed = "s" * canary.SEALED_PROMPT_TOKENS
    sealed_hash = canary.sha256_bytes(sealed.encode())
    monkeypatch.setattr(canary, "SEALED_PROMPT_SHA256", sealed_hash)
    fixture = {
        "schema": canary.FIXTURE_SCHEMA,
        "reviewed_corpus_sha256": canary.REVIEWED_CORPUS_SHA256,
        "sealed_prompt_sha256": sealed_hash,
        "prompts": {
            "decode_4294": [{"role": "user", "content": sealed}],
            "base_32768": [
                {"role": "user", "content": "b" * canary.BASE_PROMPT_TOKENS}
            ],
            "append_32769": [
                {"role": "user", "content": "b" * canary.BASE_PROMPT_TOKENS},
                {"role": "assistant", "content": "\n"},
            ],
            "retention_seed_29494": [
                {
                    "role": "user",
                    "content": "r" * canary.RETENTION_SEED_PROMPT_TOKENS,
                }
            ],
            "retention_append_32768": [
                {
                    "role": "user",
                    "content": "r" * canary.RETENTION_SEED_PROMPT_TOKENS,
                },
                {
                    "role": "assistant",
                    "content": "a"
                    * (
                        canary.RETENTION_APPEND_PROMPT_TOKENS
                        - canary.RETENTION_SEED_PROMPT_TOKENS
                    ),
                },
            ],
        },
    }
    return tokenizer, fixture


def test_fixture_validates_exact_lengths_and_one_token_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, fixture = _fixture(monkeypatch)

    prompts = canary.validate_fixture(tokenizer, fixture)

    assert len(prompts["decode_4294"]["tokens"]) == 4_294
    assert len(prompts["base_32768"]["tokens"]) == 32_768
    assert len(prompts["append_32769"]["tokens"]) == 32_769
    assert len(prompts["retention_seed_29494"]["tokens"]) == 29_494
    assert len(prompts["retention_append_32768"]["tokens"]) == 32_768


def test_fixture_attests_raw_sealed_prompt_not_rendered_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tokenizer, fixture = _fixture(monkeypatch)
    tokenizer = _TemplateMutatingTokenizer()
    prompts = canary.validate_fixture(tokenizer, fixture)
    raw_prompt = fixture["prompts"]["decode_4294"][0]["content"]
    rendered_prompt = canary.render_text(tokenizer, fixture["prompts"]["decode_4294"])

    assert raw_prompt != rendered_prompt
    assert prompts["decode_4294"]["prompt_utf8_sha256"] == canary.sha256_bytes(
        raw_prompt.encode()
    )
    assert prompts["decode_4294"]["prompt_utf8_sha256"] != canary.sha256_bytes(
        rendered_prompt.encode()
    )


def test_fixture_builder_uses_reviewed_prompt_and_constructs_strict_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer()
    sealed = "s" * canary.SEALED_PROMPT_TOKENS
    sealed_hash = canary.sha256_bytes(sealed.encode())
    monkeypatch.setattr(canary, "SEALED_PROMPT_SHA256", sealed_hash)
    monkeypatch.setattr(
        canary,
        "_reviewed_prompt",
        lambda _path: (sealed, canary.REVIEWED_CORPUS_SHA256),
    )

    fixture = canary.build_fixture(tokenizer, Path("reviewed.jsonl"))
    prompts = canary.validate_fixture(tokenizer, fixture)

    assert len(prompts["base_32768"]["tokens"]) == 32_768
    assert prompts["append_32769"]["tokens"][:-1] == prompts["base_32768"]["tokens"]
    assert (
        prompts["retention_append_32768"]["tokens"][:29_494]
        == prompts["retention_seed_29494"]["tokens"]
    )


def test_fixture_rejects_non_prefix_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, fixture = _fixture(monkeypatch)
    prompts = fixture["prompts"]
    assert isinstance(prompts, dict)
    prompts["append_32769"] = [
        {"role": "user", "content": "x" * canary.APPEND_PROMPT_TOKENS}
    ]

    with pytest.raises(RuntimeError, match="strict one-token append"):
        canary.validate_fixture(tokenizer, fixture)


def _decode_row(
    label: str,
    tps: float = 21.1,
    *,
    use_prefix_cache: bool = False,
) -> dict[str, Any]:
    return {
        "label": label,
        "prompt_tokens": canary.SEALED_PROMPT_TOKENS,
        "prompt_utf8_sha256": canary.SEALED_PROMPT_SHA256,
        "completion_sha256": canary.SEALED_COMPLETION_SHA256,
        "use_prefix_cache": use_prefix_cache,
        "cached_tokens": 0,
        "ttft_seconds": 1.0,
        "usage": {
            "prompt_tokens": canary.SEALED_PROMPT_TOKENS,
            "completion_tokens": canary.SEALED_OUTPUT_TOKENS,
        },
        "generation_stats": {
            **canary.SEALED_SCHEDULE,
            "prefix_cache_hit": "none",
            "effective_generation_tps": tps,
            "peak_memory_usage": {"in_bytes": 410_000_000_000},
        },
    }


def _decode_rows(tps: float = 21.1) -> list[dict[str, Any]]:
    return [
        _decode_row("decode_warmup", 1.0),
        *[_decode_row(f"decode_rep_{index}", tps) for index in range(1, 6)],
        _decode_row(
            canary.DECODE_PREFIX_SMOKE_LABEL,
            1.0,
            use_prefix_cache=True,
        ),
        _decode_row(
            canary.DECODE_POST_PREFIX_SMOKE_LABEL,
            1.0,
            use_prefix_cache=True,
        ),
    ]


def test_decode_analysis_requires_sealed_schedule_hash_and_five_sample_median() -> None:
    report = canary.analyze_decode(_decode_rows())

    assert report["effective_generation_tps"]["median"] == pytest.approx(21.1)
    assert report["effective_generation_tps"]["samples"] == pytest.approx([21.1] * 5)
    assert report["warmups"] == 1
    assert report["repetitions"] == 5
    assert report["measured_use_prefix_cache"] is False
    assert report["prefix_requested_smoke"]["sealed_hash_and_schedule_match"] is True
    assert report["post_prefix_smoke"]["sealed_hash_and_schedule_match"] is True
    assert report["peak_memory_usage_bytes"]["maximum"] == 410_000_000_000


def test_decode_analysis_fails_below_21_or_on_schedule_drift() -> None:
    with pytest.raises(RuntimeError, match="below 21"):
        canary.analyze_decode(_decode_rows(20.99))

    rows = _decode_rows()
    rows[-1]["generation_stats"]["speculative_rounds"] = 46
    with pytest.raises(RuntimeError, match="schedule or hash drifted"):
        canary.analyze_decode(rows)

    rows = _decode_rows()
    rows[-1]["use_prefix_cache"] = False
    with pytest.raises(RuntimeError, match="schedule or hash drifted"):
        canary.analyze_decode(rows)

    rows = _decode_rows()
    rows[-1]["generation_stats"]["peak_memory_usage"]["in_bytes"] = (
        canary.MAX_PEAK_MEMORY_BYTES + 1
    )
    with pytest.raises(RuntimeError, match="peak memory"):
        canary.analyze_decode(rows)


def _prefix_row(
    label: str,
    length: int,
    enabled: bool,
    hit: str,
    cached: int,
    ttft: float,
    digest: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "prompt_tokens": length,
        "use_prefix_cache": enabled,
        "cached_tokens": cached,
        "prompt_token_sha256": digest,
        "completion_sha256": f"completion-{digest}",
        "ttft_seconds": ttft,
        "usage": {
            "prompt_tokens": length,
            "completion_tokens": canary.PREFIX_OUTPUT_TOKENS,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
        "generation_stats": {
            "prefix_cache_hit": hit,
            "generation_tokens": canary.PREFIX_OUTPUT_TOKENS,
            "speculative_drafted_tokens": 0,
            "speculative_accepted_tokens": 0,
            "speculative_committed_tokens": canary.PREFIX_OUTPUT_TOKENS,
            "speculative_error_rounds": 0,
            "speculative_fallback_rounds": 0,
            "peak_memory_usage": {"in_bytes": 420_000_000_000},
        },
    }


def _prefix_rows() -> list[dict[str, Any]]:
    rows = [
        _prefix_row("retention_seed_29494", 29_494, True, "none", 0, 155.0, "seed"),
        _prefix_row(
            "cold_retention_append_32768",
            32_768,
            False,
            "none",
            0,
            170.0,
            "retention-append",
        ),
        _prefix_row(
            "cached_retention_append_32768",
            32_768,
            True,
            "partial",
            canary.RETENTION_CACHED_TOKENS,
            20.0,
            "retention-append",
        ),
        _prefix_row(
            "exact_retention_append_32768",
            32_768,
            True,
            "exact",
            32_767,
            0.5,
            "retention-append",
        ),
        _prefix_row("cold_base_32768", 32_768, True, "none", 0, 12.0, "base"),
        _prefix_row("exact_base_32768", 32_768, True, "exact", 32_767, 1.0, "base"),
        _prefix_row("cold_append_32769", 32_769, False, "none", 0, 8.0, "append"),
        _prefix_row(
            "cached_append_32769",
            32_769,
            True,
            "partial",
            32_767,
            3.0,
            "append",
        ),
        _prefix_row(
            "exact_append_32769",
            32_769,
            True,
            "exact",
            32_768,
            0.5,
            "append",
        ),
    ]
    rows[0]["generation_stats"].update(
        {
            "speculative_rounds": 6,
            "speculative_drafted_tokens": 10,
            "speculative_accepted_tokens": 8,
        }
    )
    return rows


def test_prefix_analysis_reports_ttft_speedups_and_parity() -> None:
    report = canary.analyze_prefix(_prefix_rows())

    assert report["ttft_speedup"]["retention_append_32768"] == pytest.approx(8.5)
    assert report["retention"]["fraction"] >= 0.90
    assert report["retention"]["effective_full_prompt_tps"]["cached"] == pytest.approx(
        32_768 / 20.0
    )
    assert report["ttft_speedup"]["exact_32768"] == pytest.approx(12.0)
    assert report["ttft_speedup"]["partial_append_32769"] == pytest.approx(8 / 3)
    assert report["exact_32768_parity"] is True
    assert report["append_32769_parity"] is True


def test_prefix_analysis_fails_closed_on_ttft_or_completion_regression() -> None:
    rows = _prefix_rows()
    rows[5]["ttft_seconds"] = 1.21
    with pytest.raises(RuntimeError, match="below 10x"):
        canary.analyze_prefix(rows)

    rows = _prefix_rows()
    rows[7]["completion_sha256"] = "different"
    with pytest.raises(RuntimeError, match="parity failed"):
        canary.analyze_prefix(rows)

    rows = _prefix_rows()
    rows[2]["ttft_seconds"] = 40.0
    with pytest.raises(RuntimeError, match="below 5x"):
        canary.analyze_prefix(rows)

    rows = _prefix_rows()
    rows[2]["generation_stats"]["peak_memory_usage"]["in_bytes"] = (
        canary.MAX_PEAK_MEMORY_BYTES + 1
    )
    with pytest.raises(RuntimeError, match="peak memory"):
        canary.analyze_prefix(rows)

    rows = _prefix_rows()
    rows[0]["generation_stats"]["speculative_drafted_tokens"] = 0
    with pytest.raises(RuntimeError, match="clean speculative decode"):
        canary.analyze_prefix(rows)


class _Response:
    status = 200

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        stats = {
            "prefix_cache_hit": "none",
            "generation_tokens": 1,
        }
        usage = {"prompt_tokens_details": {"cached_tokens": 0}}
        yield f": generation_stats {json.dumps(stats)}\n".encode()
        yield b'data: {"choices":[{"delta":{"content":"token"}}]}\n'
        yield f"data: {json.dumps({'usage': usage, 'choices': []})}\n".encode()
        yield b"data: [DONE]\n"


def test_stream_case_records_first_token_ttft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canary.urllib.request, "urlopen", lambda *_a, **_k: _Response())
    prompt = {
        "messages": [{"role": "user", "content": "prompt"}],
        "tokens": [1, 2],
        "token_sha256": "tokens",
        "prompt_utf8_sha256": "prompt",
    }

    row = canary.stream_case(
        canary.LOCAL_ENDPOINT,
        label="ttft",
        prompt=prompt,
        use_prefix_cache=False,
        max_tokens=1,
        timeout=1.0,
    )

    assert 0 < row["ttft_seconds"] <= row["elapsed_seconds"]


def test_write_new_preserves_existing_failure_evidence(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    canary.write_new(output, {"status": "fail"})

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        canary.write_new(output, {"status": "pass"})
