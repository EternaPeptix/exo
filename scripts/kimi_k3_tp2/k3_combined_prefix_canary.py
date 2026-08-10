#!/usr/bin/env python3
"""Fail-closed live canary for the 21 tok/s plus ordinary-prefix EXO stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

MODEL_ID = "kernelpool/Kimi-K3-2bit-UVMAX"
SCHEMA = "k3-21tps-prefix-retention-canary-v1"
FIXTURE_SCHEMA = "k3-21tps-prefix-retention-fixture-v1"
MLX_LM_COMMIT = "7216b0d2c71b09f21e9e55c642c24467cb4fc15e"
LOCAL_ENDPOINT = "http://127.0.0.1:52415"
ORDINARY_AFTER_CONTEXT = 32_768
SEALED_CASE_ID = "natural-long-4300"
SEALED_PROMPT_TOKENS = 4_294
SEALED_OUTPUT_TOKENS = 128
SEALED_PROMPT_SHA256 = (
    "0f4f9b99b977244b748d10c8e85b9632ace701536b3123bfabb44fbc5e590b25"
)
SEALED_COMPLETION_SHA256 = (
    "b2a0f82808f10d4aef14291f6ab7f8d09daa7ef2d6d93ae75a9b438360b05a39"
)
REVIEWED_CORPUS_SHA256 = (
    "ccf61ae61f20d40827135f9266dcc1c8474e1d2dbfca0353ac10bbec335f2da8"
)
SEALED_SCHEDULE = {
    "generation_tokens": 128,
    "speculative_rounds": 47,
    "speculative_drafted_tokens": 90,
    "speculative_accepted_tokens": 81,
    "speculative_committed_tokens": 128,
    "speculative_fallback_rounds": 0,
    "speculative_error_rounds": 0,
}
DECODE_WARMUPS = 1
DECODE_REPETITIONS = 5
DECODE_PREFIX_SMOKE_LABEL = "decode_prefix_requested_smoke"
DECODE_POST_PREFIX_SMOKE_LABEL = "decode_post_prefix_smoke"
MIN_DECODE_MEDIAN_TPS = 21.0
BASE_PROMPT_TOKENS = 32_768
APPEND_PROMPT_TOKENS = 32_769
RETENTION_SEED_PROMPT_TOKENS = 29_494
RETENTION_CACHED_TOKENS = RETENTION_SEED_PROMPT_TOKENS - 1
RETENTION_APPEND_PROMPT_TOKENS = 32_768
PREFIX_OUTPUT_TOKENS = 16
MIN_EXACT_TTFT_SPEEDUP = 10.0
MIN_PARTIAL_APPEND_TTFT_SPEEDUP = 2.0
MIN_RETENTION_FRACTION = 0.90
MIN_RETENTION_TTFT_SPEEDUP = 5.0
MIN_RETENTION_EFFECTIVE_FULL_PROMPT_TPS = 1_000.0
MAX_PEAK_MEMORY_BYTES = 430_000_000_000
HEX40 = re.compile(r"[0-9a-f]{40}")


class Tokenizer(Protocol):
    def apply_chat_template(self, messages: object, **kwargs: object) -> object: ...

    def encode(self, text: str, **kwargs: object) -> object: ...

    def decode(self, tokens: Sequence[int], **kwargs: object) -> str: ...


@dataclass(frozen=True)
class Case:
    label: str
    prompt: str
    use_prefix_cache: bool


PREFIX_CASES = (
    Case("retention_seed_29494", "retention_seed_29494", True),
    Case("cold_retention_append_32768", "retention_append_32768", False),
    Case("cached_retention_append_32768", "retention_append_32768", True),
    Case("exact_retention_append_32768", "retention_append_32768", True),
    Case("cold_base_32768", "base_32768", True),
    Case("exact_base_32768", "base_32768", True),
    Case("cold_append_32769", "append_32769", False),
    Case("cached_append_32769", "append_32769", True),
    Case("exact_append_32769", "append_32769", True),
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_sha256(tokens: Sequence[int]) -> str:
    return sha256_bytes(canonical_bytes([int(token) for token in tokens]))


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def attest_runtime(
    pid_file: Path,
    exo_root: Path,
    expected_exo_commit: str,
    mlx_lm_root: Path,
) -> dict[str, object]:
    if HEX40.fullmatch(expected_exo_commit) is None:
        raise RuntimeError("expected EXO commit is not one full commit ID")
    if pid_file.is_symlink() or not pid_file.is_file():
        raise RuntimeError("PID file is absent or not regular")
    raw_pid = pid_file.read_text().strip()
    if not raw_pid.isdecimal() or int(raw_pid) <= 0:
        raise RuntimeError("PID file does not contain one positive integer")
    pid = int(raw_pid)
    process = subprocess.check_output(
        ["/bin/ps", "eww", "-p", str(pid), "-o", "command="], text=True
    ).strip()
    required = (
        "EXO_MLX_KIMI_K3_DSPARK_PREFIX_CACHE=1",
        "EXO_MLX_KIMI_K3_DSPARK_PREFIX_CACHE_SEED_MIN_CONTEXT=29494",
        "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_AFTER_CONTEXT=32768",
        str(exo_root / "src"),
        str(mlx_lm_root),
    )
    missing = [value for value in required if value not in process]
    if missing:
        raise RuntimeError(f"prefix-enabled process contract is incomplete: {missing}")
    exo_commit = _git_head(exo_root)
    mlx_lm_commit = _git_head(mlx_lm_root)
    if exo_commit != expected_exo_commit:
        raise RuntimeError("running EXO checkout does not match the requested commit")
    if mlx_lm_commit != MLX_LM_COMMIT:
        raise RuntimeError("running MLX-LM checkout is not the verified 7216 runtime")
    return {
        "pid": pid,
        "process_sha256": sha256_bytes(process.encode()),
        "exo_commit": exo_commit,
        "mlx_lm_commit": mlx_lm_commit,
    }


def render_text(tokenizer: Tokenizer, messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("fixture messages must be one non-empty list")
    templated = messages
    partial_assistant = ""
    final = messages[-1]
    if isinstance(final, dict) and final.get("role") == "assistant":
        content = final.get("content")
        if not isinstance(content, str) or not content:
            raise RuntimeError("partial assistant suffix must be non-empty text")
        partial_assistant = content
        templated = messages[:-1]
    rendered = tokenizer.apply_chat_template(
        templated,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        thinking=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("chat template did not return text")
    return rendered + partial_assistant


def render_tokens(tokenizer: Tokenizer, messages: object) -> list[int]:
    tokens = tokenizer.encode(
        render_text(tokenizer, messages), add_special_tokens=False
    )
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise RuntimeError("tokenizer returned invalid prompt tokens")
    return cast(list[int], tokens)


def _reviewed_prompt(corpus: Path) -> tuple[str, str]:
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    with corpus.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise RuntimeError(f"reviewed corpus has blank row {line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"reviewed corpus row {line_number} is not an object"
                )
            rows.append(cast(dict[str, object], value))
            if value.get("case_id") == SEALED_CASE_ID:
                selected.append(cast(dict[str, object], value))
    corpus_sha256 = sha256_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
    if corpus_sha256 != REVIEWED_CORPUS_SHA256:
        raise RuntimeError("reviewed corpus digest drifted")
    if len(selected) != 1:
        raise RuntimeError(f"reviewed corpus must contain one {SEALED_CASE_ID}")
    row = selected[0]
    prompt = row.get("prompt")
    if (
        not isinstance(prompt, str)
        or sha256_bytes(prompt.encode()) != SEALED_PROMPT_SHA256
        or row.get("prompt_sha256") != SEALED_PROMPT_SHA256
        or row.get("requested_decode_tokens") != SEALED_OUTPUT_TOKENS
    ):
        raise RuntimeError("sealed reviewed-corpus case drifted")
    return prompt, corpus_sha256


def _exact_base_messages(
    tokenizer: Tokenizer, source_text: str, target: int
) -> list[dict[str, str]]:
    repeated = (source_text + "\n\n") * max(10, target // 256)
    source_ids = tokenizer.encode(repeated, add_special_tokens=False)
    if not isinstance(source_ids, list) or any(
        type(token) is not int for token in source_ids
    ):
        raise RuntimeError("tokenizer could not encode the reviewed source")
    typed_source_ids = cast(list[int], source_ids)
    empty_overhead = len(render_tokens(tokenizer, [{"role": "user", "content": ""}]))
    center = target - empty_overhead
    for radius in (128, 1024, 4096):
        for length in range(max(1, center - radius), center + radius + 1):
            content = tokenizer.decode(
                typed_source_ids[:length],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            messages = [{"role": "user", "content": content}]
            if len(render_tokens(tokenizer, messages)) == target:
                return messages
    raise RuntimeError(f"could not construct an exact {target}-token base prompt")


def _strict_one_token_append(
    tokenizer: Tokenizer, base: list[dict[str, str]]
) -> list[dict[str, str]]:
    base_tokens = render_tokens(tokenizer, base)
    candidates = ("\n", "\n\n", " a", " the", ".", " x", "0", "!")
    for suffix in candidates:
        appended = [*base, {"role": "assistant", "content": suffix}]
        appended_tokens = render_tokens(tokenizer, appended)
        if len(appended_tokens) == len(base_tokens) + 1 and (
            appended_tokens[:-1] == base_tokens
        ):
            return appended
    raise RuntimeError("could not construct a strict one-token append prompt")


def _strict_append_to_target(
    tokenizer: Tokenizer,
    base: list[dict[str, str]],
    source_text: str,
    target: int,
) -> list[dict[str, str]]:
    """Extend a rendered prompt without changing any existing token IDs."""

    base_text = render_text(tokenizer, base)
    base_tokens = render_tokens(tokenizer, base)
    if target <= len(base_tokens):
        raise RuntimeError("strict append target must exceed the base length")
    source_ids = tokenizer.encode(source_text, add_special_tokens=False)
    if not isinstance(source_ids, list) or not source_ids:
        raise RuntimeError("strict append source did not produce tokens")
    repetitions = max(
        2,
        math.ceil(2 * (target - len(base_tokens)) / len(source_ids)) + 2,
    )
    repeated = (source_text + "\n\n") * repetitions
    for boundary in ("\n", "\n\n", " ", " x", ".", "!"):
        combined = tokenizer.encode(
            base_text + boundary + repeated,
            add_special_tokens=False,
        )
        if (
            not isinstance(combined, list)
            or any(type(token) is not int for token in combined)
            or len(combined) < target
            or combined[: len(base_tokens)] != base_tokens
        ):
            continue
        suffix = tokenizer.decode(
            cast(list[int], combined)[len(base_tokens) : target],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        appended = [*base, {"role": "assistant", "content": suffix}]
        appended_tokens = render_tokens(tokenizer, appended)
        if (
            len(appended_tokens) == target
            and appended_tokens[: len(base_tokens)] == base_tokens
        ):
            return appended
    raise RuntimeError(f"could not construct a strict {target}-token append prompt")


def build_fixture(tokenizer: Tokenizer, corpus: Path) -> dict[str, object]:
    sealed_prompt, corpus_sha256 = _reviewed_prompt(corpus)
    decode_messages = [{"role": "user", "content": sealed_prompt}]
    if len(render_tokens(tokenizer, decode_messages)) != SEALED_PROMPT_TOKENS:
        raise RuntimeError("sealed prompt no longer tokenizes to 4294 tokens")
    base = _exact_base_messages(tokenizer, sealed_prompt, BASE_PROMPT_TOKENS)
    appended = _strict_one_token_append(tokenizer, base)
    retention_seed = _exact_base_messages(
        tokenizer,
        sealed_prompt[::-1],
        RETENTION_SEED_PROMPT_TOKENS,
    )
    retention_append = _strict_append_to_target(
        tokenizer,
        retention_seed,
        sealed_prompt,
        RETENTION_APPEND_PROMPT_TOKENS,
    )
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "reviewed_corpus_sha256": corpus_sha256,
        "sealed_prompt_sha256": SEALED_PROMPT_SHA256,
        "prompts": {
            "decode_4294": decode_messages,
            "base_32768": base,
            "append_32769": appended,
            "retention_seed_29494": retention_seed,
            "retention_append_32768": retention_append,
        },
    }
    validate_fixture(tokenizer, fixture)
    return fixture


def validate_fixture(tokenizer: Tokenizer, raw: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "reviewed_corpus_sha256",
        "sealed_prompt_sha256",
        "prompts",
    }:
        raise RuntimeError("fixture envelope keys are invalid")
    if (
        raw.get("schema") != FIXTURE_SCHEMA
        or raw.get("reviewed_corpus_sha256") != REVIEWED_CORPUS_SHA256
        or raw.get("sealed_prompt_sha256") != SEALED_PROMPT_SHA256
    ):
        raise RuntimeError("fixture is not bound to the reviewed sealed corpus")
    raw_prompts = raw.get("prompts")
    required = {
        "decode_4294",
        "base_32768",
        "append_32769",
        "retention_seed_29494",
        "retention_append_32768",
    }
    if not isinstance(raw_prompts, dict) or set(raw_prompts) != required:
        raise RuntimeError("fixture must contain exactly the five canary prompts")
    prompts: dict[str, dict[str, object]] = {}
    expected = {
        "decode_4294": SEALED_PROMPT_TOKENS,
        "base_32768": BASE_PROMPT_TOKENS,
        "append_32769": APPEND_PROMPT_TOKENS,
        "retention_seed_29494": RETENTION_SEED_PROMPT_TOKENS,
        "retention_append_32768": RETENTION_APPEND_PROMPT_TOKENS,
    }
    for name, length in expected.items():
        messages = raw_prompts[name]
        tokens = render_tokens(tokenizer, messages)
        if len(tokens) != length:
            raise RuntimeError(f"{name} rendered to {len(tokens)}, expected {length}")
        prompts[name] = {
            "messages": messages,
            "tokens": tokens,
            "token_sha256": token_sha256(tokens),
            "prompt_utf8_sha256": sha256_bytes(
                render_text(tokenizer, messages).encode()
            ),
        }
    decode_messages = raw_prompts["decode_4294"]
    if (
        not isinstance(decode_messages, list)
        or len(decode_messages) != 1
        or not isinstance(decode_messages[0], dict)
        or decode_messages[0].get("role") != "user"
        or not isinstance(decode_messages[0].get("content"), str)
    ):
        raise RuntimeError("sealed decode fixture must be one user message")
    sealed_content = cast(str, decode_messages[0]["content"])
    sealed_content_sha256 = sha256_bytes(sealed_content.encode())
    if sealed_content_sha256 != SEALED_PROMPT_SHA256:
        raise RuntimeError("sealed decode prompt hash drifted")
    prompts["decode_4294"]["prompt_utf8_sha256"] = sealed_content_sha256
    base = prompts["base_32768"]["tokens"]
    appended = prompts["append_32769"]["tokens"]
    if not isinstance(base, list) or not isinstance(appended, list):
        raise AssertionError("validated token lists changed type")
    if appended[:-1] != base:
        raise RuntimeError("32769 prompt is not a strict one-token append")
    retention_seed = prompts["retention_seed_29494"]["tokens"]
    retention_append = prompts["retention_append_32768"]["tokens"]
    if not isinstance(retention_seed, list) or not isinstance(retention_append, list):
        raise AssertionError("validated retention token lists changed type")
    if retention_append[: len(retention_seed)] != retention_seed:
        raise RuntimeError("32768 retention prompt does not strictly extend its seed")
    if len(retention_append) != RETENTION_APPEND_PROMPT_TOKENS:
        raise RuntimeError("retention append length drifted")
    if RETENTION_CACHED_TOKENS / len(retention_append) < MIN_RETENTION_FRACTION:
        raise RuntimeError("retention fixture cannot retain at least 90% of the append")
    if token_sha256(retention_append) == token_sha256(base):
        raise RuntimeError("retention and legacy 32768 prompts must be distinct")
    return prompts


def stream_case(
    endpoint: str,
    *,
    label: str,
    prompt: dict[str, object],
    use_prefix_cache: bool,
    max_tokens: int,
    timeout: float,
) -> dict[str, object]:
    messages = prompt["messages"]
    tokens = prompt["tokens"]
    if not isinstance(tokens, list):
        raise AssertionError("validated prompt tokens changed type")
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "seed": 0,
        "enable_thinking": False,
        "use_prefix_cache": use_prefix_cache,
    }
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/bench/chat/completions",
        data=canonical_bytes(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None
    saw_done = False
    first_token_seconds: float | None = None
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{label} returned HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode().strip()
            if line.startswith(": generation_stats "):
                value = json.loads(line.removeprefix(": generation_stats "))
                if isinstance(value, dict):
                    stats = value
                continue
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            if not isinstance(event, dict) or "error" in event:
                raise RuntimeError(f"{label} returned an invalid stream event")
            choices = event.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                if isinstance(delta, dict):
                    for field in ("reasoning_content", "content"):
                        part = delta.get(field)
                        if isinstance(part, str) and part:
                            if first_token_seconds is None:
                                first_token_seconds = time.perf_counter() - started
                            text_parts.append(part)
            candidate_usage = event.get("usage")
            if isinstance(candidate_usage, dict):
                usage = candidate_usage
    if not saw_done or first_token_seconds is None or usage is None or stats is None:
        raise RuntimeError(f"{label} stream omitted terminal or TTFT evidence")
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached) is not int:
        raise RuntimeError(f"{label} omitted cached_tokens")
    return {
        "label": label,
        "use_prefix_cache": use_prefix_cache,
        "prompt_tokens": len(tokens),
        "prompt_token_sha256": prompt["token_sha256"],
        "prompt_utf8_sha256": prompt["prompt_utf8_sha256"],
        "cached_tokens": cached,
        "completion_sha256": sha256_bytes("".join(text_parts).encode()),
        "ttft_seconds": first_token_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "usage": usage,
        "generation_stats": stats,
    }


def _finite_positive(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"{label} is not finite and positive")
    return result


def _peak_memory_bytes(row: dict[str, object], label: str) -> int:
    stats = row.get("generation_stats")
    peak = stats.get("peak_memory_usage") if isinstance(stats, dict) else None
    value = peak.get("in_bytes") if isinstance(peak, dict) else None
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{label} omitted exact peak_memory_usage.in_bytes")
    if value > MAX_PEAK_MEMORY_BYTES:
        raise RuntimeError(
            f"{label} peak memory {value} exceeds {MAX_PEAK_MEMORY_BYTES} bytes"
        )
    return value


def analyze_decode(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    measured_labels = {
        "decode_warmup",
        *[f"decode_rep_{i}" for i in range(1, DECODE_REPETITIONS + 1)],
    }
    expected_labels = {
        *measured_labels,
        DECODE_PREFIX_SMOKE_LABEL,
        DECODE_POST_PREFIX_SMOKE_LABEL,
    }
    by_label = {str(row.get("label")): row for row in rows}
    if len(by_label) != len(rows) or set(by_label) != expected_labels:
        raise RuntimeError("decode-preservation rows are incomplete or duplicated")
    measured_tps: list[float] = []
    peak_memory: dict[str, int] = {}
    for label, row in by_label.items():
        stats = row.get("generation_stats")
        usage = row.get("usage")
        if not isinstance(stats, dict) or not isinstance(usage, dict):
            raise RuntimeError(f"{label} omitted stats or usage")
        mismatches = {
            key: (stats.get(key), expected)
            for key, expected in SEALED_SCHEDULE.items()
            if stats.get(key) != expected
        }
        expected_prefix_request = label in {
            DECODE_PREFIX_SMOKE_LABEL,
            DECODE_POST_PREFIX_SMOKE_LABEL,
        }
        if (
            row.get("prompt_tokens") != SEALED_PROMPT_TOKENS
            or row.get("prompt_utf8_sha256") != SEALED_PROMPT_SHA256
            or row.get("completion_sha256") != SEALED_COMPLETION_SHA256
            or row.get("use_prefix_cache") is not expected_prefix_request
            or row.get("cached_tokens") != 0
            or stats.get("prefix_cache_hit") != "none"
            or usage.get("prompt_tokens") != SEALED_PROMPT_TOKENS
            or usage.get("completion_tokens") != SEALED_OUTPUT_TOKENS
            or mismatches
        ):
            raise RuntimeError(f"{label} sealed schedule or hash drifted: {mismatches}")
        _finite_positive(row.get("ttft_seconds"), f"{label} TTFT")
        peak_memory[label] = _peak_memory_bytes(row, label)
        tps = _finite_positive(stats.get("effective_generation_tps"), f"{label} TPS")
        if label.startswith("decode_rep_"):
            measured_tps.append(tps)
    median_tps = statistics.median(measured_tps)
    if median_tps < MIN_DECODE_MEDIAN_TPS:
        raise RuntimeError("sealed 4,294/128 median decode fell below 21 tok/s")
    return {
        "status": "pass",
        "warmups": DECODE_WARMUPS,
        "repetitions": DECODE_REPETITIONS,
        "measured_use_prefix_cache": False,
        "prefix_requested_smoke": {
            "label": DECODE_PREFIX_SMOKE_LABEL,
            "use_prefix_cache": True,
            "prefix_cache_hit": "none",
            "cached_tokens": 0,
            "sealed_hash_and_schedule_match": True,
        },
        "post_prefix_smoke": {
            "label": DECODE_POST_PREFIX_SMOKE_LABEL,
            "use_prefix_cache": True,
            "prefix_cache_hit": "none",
            "cached_tokens": 0,
            "sealed_hash_and_schedule_match": True,
        },
        "effective_generation_tps": {
            "samples": measured_tps,
            "minimum": min(measured_tps),
            "median": median_tps,
            "maximum": max(measured_tps),
        },
        "completion_sha256": SEALED_COMPLETION_SHA256,
        "schedule": SEALED_SCHEDULE,
        "peak_memory_usage_bytes": {
            "by_label": peak_memory,
            "maximum": max(peak_memory.values()),
            "ceiling": MAX_PEAK_MEMORY_BYTES,
        },
    }


def analyze_prefix(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_label = {str(row.get("label")): row for row in rows}
    if len(by_label) != len(rows) or set(by_label) != {
        case.label for case in PREFIX_CASES
    }:
        raise RuntimeError("prefix rows are incomplete or duplicated")
    expected = {
        "retention_seed_29494": (29_494, True, "none", 0),
        "cold_retention_append_32768": (32_768, False, "none", 0),
        "cached_retention_append_32768": (
            32_768,
            True,
            "partial",
            RETENTION_CACHED_TOKENS,
        ),
        "exact_retention_append_32768": (32_768, True, "exact", 32_767),
        "cold_base_32768": (32_768, True, "none", 0),
        "exact_base_32768": (32_768, True, "exact", 32_767),
        "cold_append_32769": (32_769, False, "none", 0),
        "cached_append_32769": (32_769, True, "partial", 32_767),
        "exact_append_32769": (32_769, True, "exact", 32_768),
    }
    ttft: dict[str, float] = {}
    peak_memory: dict[str, int] = {}
    for label, (length, enabled, hit, cached) in expected.items():
        row = by_label[label]
        stats = row.get("generation_stats")
        usage = row.get("usage")
        if not isinstance(stats, dict) or not isinstance(usage, dict):
            raise RuntimeError(f"{label} omitted stats or usage")
        details = usage.get("prompt_tokens_details")
        if (
            row.get("prompt_tokens") != length
            or row.get("use_prefix_cache") is not enabled
            or row.get("cached_tokens") != cached
            or not isinstance(details, dict)
            or details.get("cached_tokens") != cached
            or usage.get("prompt_tokens") != length
            or usage.get("completion_tokens") != PREFIX_OUTPUT_TOKENS
            or stats.get("prefix_cache_hit") != hit
            or stats.get("generation_tokens") != PREFIX_OUTPUT_TOKENS
            or stats.get("speculative_committed_tokens") != PREFIX_OUTPUT_TOKENS
            or stats.get("speculative_error_rounds") != 0
            or stats.get("speculative_fallback_rounds") != 0
        ):
            raise RuntimeError(f"{label} ordinary-prefix contract drifted")
        if label == "retention_seed_29494":
            drafted = stats.get("speculative_drafted_tokens")
            accepted = stats.get("speculative_accepted_tokens")
            rounds = stats.get("speculative_rounds")
            if (
                type(drafted) is not int
                or drafted <= 0
                or type(accepted) is not int
                or not 0 <= accepted <= drafted
                or type(rounds) is not int
                or rounds <= 0
            ):
                raise RuntimeError(
                    "retention seed did not use clean speculative decode"
                )
        elif (
            stats.get("speculative_drafted_tokens") != 0
            or stats.get("speculative_accepted_tokens") != 0
        ):
            raise RuntimeError(f"{label} unexpectedly used speculative decode")
        ttft[label] = _finite_positive(row.get("ttft_seconds"), f"{label} TTFT")
        peak_memory[label] = _peak_memory_bytes(row, label)
    parity_groups = (
        (
            "cold_retention_append_32768",
            "cached_retention_append_32768",
            "exact_retention_append_32768",
        ),
        ("cold_base_32768", "exact_base_32768"),
        ("cold_append_32769", "cached_append_32769", "exact_append_32769"),
    )
    for labels in parity_groups:
        for field in ("prompt_token_sha256", "completion_sha256"):
            if len({by_label[label].get(field) for label in labels}) != 1:
                raise RuntimeError(f"{field} parity failed for {labels}")
    speedups = {
        "retention_append_32768": (
            ttft["cold_retention_append_32768"] / ttft["cached_retention_append_32768"]
        ),
        "exact_32768": ttft["cold_base_32768"] / ttft["exact_base_32768"],
        "partial_append_32769": (
            ttft["cold_append_32769"] / ttft["cached_append_32769"]
        ),
        "exact_append_32769": (ttft["cold_append_32769"] / ttft["exact_append_32769"]),
    }
    if speedups["exact_32768"] < MIN_EXACT_TTFT_SPEEDUP:
        raise RuntimeError("32,768 exact-hit TTFT speedup is below 10x")
    if speedups["partial_append_32769"] < MIN_PARTIAL_APPEND_TTFT_SPEEDUP:
        raise RuntimeError("32,769 partial-append TTFT speedup is below 2x")
    if speedups["retention_append_32768"] < MIN_RETENTION_TTFT_SPEEDUP:
        raise RuntimeError("90%-retained 32,768 append TTFT speedup is below 5x")
    retained_fraction = RETENTION_CACHED_TOKENS / RETENTION_APPEND_PROMPT_TOKENS
    if retained_fraction < MIN_RETENTION_FRACTION:
        raise RuntimeError("cached retention fraction fell below 90%")
    effective_full_prompt_tps = {
        "cold": (RETENTION_APPEND_PROMPT_TOKENS / ttft["cold_retention_append_32768"]),
        "cached": (
            RETENTION_APPEND_PROMPT_TOKENS / ttft["cached_retention_append_32768"]
        ),
    }
    if effective_full_prompt_tps["cached"] < MIN_RETENTION_EFFECTIVE_FULL_PROMPT_TPS:
        raise RuntimeError(
            "90%-retained effective full-prompt throughput is below 1000 tok/s"
        )
    return {
        "status": "pass",
        "ttft_seconds": ttft,
        "ttft_speedup": speedups,
        "exact_32768_parity": True,
        "append_32769_parity": True,
        "retention_append_32768_parity": True,
        "retention": {
            "seed_prompt_tokens": RETENTION_SEED_PROMPT_TOKENS,
            "cached_tokens": RETENTION_CACHED_TOKENS,
            "append_prompt_tokens": RETENTION_APPEND_PROMPT_TOKENS,
            "fraction": retained_fraction,
            "minimum_fraction": MIN_RETENTION_FRACTION,
            "effective_full_prompt_tps": effective_full_prompt_tps,
            "minimum_cached_effective_full_prompt_tps": (
                MIN_RETENTION_EFFECTIVE_FULL_PROMPT_TPS
            ),
            "minimum_ttft_speedup": MIN_RETENTION_TTFT_SPEEDUP,
        },
        "peak_memory_usage_bytes": {
            "by_label": peak_memory,
            "maximum": max(peak_memory.values()),
            "ceiling": MAX_PEAK_MEMORY_BYTES,
        },
    }


def write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--build-fixture-output", type=Path)
    parser.add_argument("--reviewed-corpus", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--mlx-lm-root", type=Path)
    parser.add_argument("--expected-exo-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint", default=LOCAL_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    from transformers import AutoTokenizer

    tokenizer = cast(
        Tokenizer,
        AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=True
        ),
    )
    if args.build_fixture_output is not None:
        if args.reviewed_corpus is None or any(
            value is not None
            for value in (
                args.fixture,
                args.pid_file,
                args.mlx_lm_root,
                args.expected_exo_commit,
                args.output,
            )
        ):
            raise RuntimeError(
                "fixture build requires only --tokenizer, --reviewed-corpus, "
                "and --build-fixture-output"
            )
        write_new(
            args.build_fixture_output,
            build_fixture(tokenizer, args.reviewed_corpus),
        )
        return 0
    if any(
        value is None
        for value in (
            args.fixture,
            args.pid_file,
            args.mlx_lm_root,
            args.expected_exo_commit,
            args.output,
        )
    ):
        raise RuntimeError("live run is missing a required argument")
    if args.endpoint.rstrip("/") != LOCAL_ENDPOINT:
        raise RuntimeError("live canary is restricted to the local EXO endpoint")
    fixture = cast(Path, args.fixture)
    pid_file = cast(Path, args.pid_file)
    mlx_lm_root = cast(Path, args.mlx_lm_root)
    expected_exo_commit = cast(str, args.expected_exo_commit)
    output = cast(Path, args.output)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite {output}")
    artifact: dict[str, object] = {
        "schema": SCHEMA,
        "status": "running",
        "decode_rows": [],
        "prefix_rows": [],
    }
    try:
        prompts = validate_fixture(tokenizer, json.loads(fixture.read_text()))
        exo_root = Path(__file__).resolve().parents[2]
        before = attest_runtime(pid_file, exo_root, expected_exo_commit, mlx_lm_root)
        artifact["runtime_before"] = before
        decode_rows = cast(list[dict[str, object]], artifact["decode_rows"])
        for index in range(DECODE_WARMUPS + DECODE_REPETITIONS):
            label = "decode_warmup" if index == 0 else f"decode_rep_{index}"
            decode_rows.append(
                stream_case(
                    args.endpoint,
                    label=label,
                    prompt=prompts["decode_4294"],
                    use_prefix_cache=False,
                    max_tokens=SEALED_OUTPUT_TOKENS,
                    timeout=args.timeout,
                )
            )
        decode_rows.append(
            stream_case(
                args.endpoint,
                label=DECODE_PREFIX_SMOKE_LABEL,
                prompt=prompts["decode_4294"],
                use_prefix_cache=True,
                max_tokens=SEALED_OUTPUT_TOKENS,
                timeout=args.timeout,
            )
        )
        prefix_rows = cast(list[dict[str, object]], artifact["prefix_rows"])
        for case in PREFIX_CASES:
            prefix_rows.append(
                stream_case(
                    args.endpoint,
                    label=case.label,
                    prompt=prompts[case.prompt],
                    use_prefix_cache=case.use_prefix_cache,
                    max_tokens=PREFIX_OUTPUT_TOKENS,
                    timeout=args.timeout,
                )
            )
        # Exercise the short speculative path only after a full 32K cache is
        # retained.  It must bypass the cache and preserve the sealed schedule.
        decode_rows.append(
            stream_case(
                args.endpoint,
                label=DECODE_POST_PREFIX_SMOKE_LABEL,
                prompt=prompts["decode_4294"],
                use_prefix_cache=True,
                max_tokens=SEALED_OUTPUT_TOKENS,
                timeout=args.timeout,
            )
        )
        after = attest_runtime(pid_file, exo_root, expected_exo_commit, mlx_lm_root)
        artifact["runtime_after"] = after
        if before != after:
            raise RuntimeError("runtime identity changed during the canary")
        artifact["decode_analysis"] = analyze_decode(decode_rows)
        artifact["prefix_analysis"] = analyze_prefix(prefix_rows)
        artifact["status"] = "pass"
    except Exception as error:
        artifact["status"] = "fail"
        artifact["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_new(output, artifact)
        raise
    write_new(output, artifact)
    print(json.dumps({"status": "pass", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
