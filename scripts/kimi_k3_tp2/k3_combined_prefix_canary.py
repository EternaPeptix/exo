#!/usr/bin/env python3
"""Fail-closed live canary for the 21 tok/s plus ordinary-prefix EXO stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

MODEL_ID = "kernelpool/Kimi-K3-2bit-UVMAX"
SCHEMA = "k3-21tps-ordinary-prefix-canary-v1"
MLX_LM_COMMIT = "7216b0d2c71b09f21e9e55c642c24467cb4fc15e"
LOCAL_ENDPOINT = "http://127.0.0.1:52415"
ORDINARY_AFTER_CONTEXT = 32_768
SPECULATIVE_PROMPT_TOKENS = 4_300
BASE_PROMPT_TOKENS = 32_768
APPEND_PROMPT_TOKENS = 32_769
DECODE_TOKENS = 16
MIN_SPECULATIVE_TPS = 20.0
MIN_SPECULATIVE_RATIO = 0.95
HEX40 = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Case:
    label: str
    prompt: str
    use_prefix_cache: bool


CASES = (
    Case("spec_control_4300", "speculative_4300", False),
    Case("spec_requested_4300", "speculative_4300", True),
    Case("cold_base_32768", "base_32768", True),
    Case("exact_base_32768", "base_32768", True),
    Case("cold_append_32769", "append_32769", False),
    Case("cached_append_32769", "append_32769", True),
    Case("exact_append_32769", "append_32769", True),
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


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


def render_tokens(tokenizer: Any, messages: object) -> list[int]:
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("fixture messages must be one non-empty list")
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        thinking=False,
    )
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise RuntimeError("tokenizer returned invalid prompt tokens")
    return tokens


def validate_fixture(tokenizer: Any, raw: object) -> dict[str, dict[str, object]]:
    required = {"speculative_4300", "base_32768", "append_32769"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise RuntimeError("fixture must contain exactly the three canary prompts")
    prompts: dict[str, dict[str, object]] = {}
    expected = {
        "speculative_4300": SPECULATIVE_PROMPT_TOKENS,
        "base_32768": BASE_PROMPT_TOKENS,
        "append_32769": APPEND_PROMPT_TOKENS,
    }
    for name, length in expected.items():
        messages = raw[name]
        tokens = render_tokens(tokenizer, messages)
        if len(tokens) != length:
            raise RuntimeError(f"{name} rendered to {len(tokens)}, expected {length}")
        prompts[name] = {
            "messages": messages,
            "tokens": tokens,
            "token_sha256": token_sha256(tokens),
        }
    base = prompts["base_32768"]["tokens"]
    appended = prompts["append_32769"]["tokens"]
    if not isinstance(base, list) or not isinstance(appended, list):
        raise AssertionError("validated token lists changed type")
    if appended[: len(base)] != base:
        raise RuntimeError("32769 prompt is not a strict one-token append")
    return prompts


def stream_case(
    endpoint: str,
    case: Case,
    prompt: dict[str, object],
    *,
    timeout: float,
) -> dict[str, object]:
    messages = prompt["messages"]
    tokens = prompt["tokens"]
    if not isinstance(tokens, list):
        raise AssertionError("validated prompt tokens changed type")
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": DECODE_TOKENS,
        "stream": True,
        "temperature": 0.0,
        "seed": 0,
        "enable_thinking": False,
        "use_prefix_cache": case.use_prefix_cache,
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
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{case.label} returned HTTP {response.status}")
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
                raise RuntimeError(f"{case.label} returned an invalid stream event")
            choices = event.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                if isinstance(delta, dict):
                    for field in ("reasoning_content", "content"):
                        part = delta.get(field)
                        if isinstance(part, str):
                            text_parts.append(part)
            candidate_usage = event.get("usage")
            if isinstance(candidate_usage, dict):
                usage = candidate_usage
    if not saw_done or usage is None or stats is None:
        raise RuntimeError(f"{case.label} stream omitted terminal evidence")
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached) is not int:
        raise RuntimeError(f"{case.label} omitted cached_tokens")
    return {
        "label": case.label,
        "use_prefix_cache": case.use_prefix_cache,
        "prompt_tokens": len(tokens),
        "prompt_token_sha256": prompt["token_sha256"],
        "cached_tokens": cached,
        "completion_sha256": sha256_bytes("".join(text_parts).encode()),
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


def analyze(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_label = {str(row.get("label")): row for row in rows}
    if len(by_label) != len(rows) or set(by_label) != {case.label for case in CASES}:
        raise RuntimeError("canary result labels are incomplete or duplicated")
    expected = {
        "spec_control_4300": (4_300, False, "none", 0, True),
        "spec_requested_4300": (4_300, True, "none", 0, True),
        "cold_base_32768": (32_768, True, "none", 0, False),
        "exact_base_32768": (32_768, True, "exact", 32_767, False),
        "cold_append_32769": (32_769, False, "none", 0, False),
        "cached_append_32769": (32_769, True, "partial", 32_767, False),
        "exact_append_32769": (32_769, True, "exact", 32_768, False),
    }
    tps: dict[str, float] = {}
    for label, (length, enabled, hit, cached, speculative) in expected.items():
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
            or usage.get("completion_tokens") != DECODE_TOKENS
            or stats.get("prefix_cache_hit") != hit
            or stats.get("generation_tokens") != DECODE_TOKENS
            or stats.get("speculative_error_rounds") != 0
            or stats.get("speculative_fallback_rounds") != 0
        ):
            raise RuntimeError(f"{label} contract drifted")
        drafted = stats.get("speculative_drafted_tokens")
        accepted = stats.get("speculative_accepted_tokens")
        committed = stats.get("speculative_committed_tokens")
        if speculative:
            if (
                not isinstance(drafted, int)
                or drafted <= 0
                or not isinstance(accepted, int)
                or accepted <= 0
                or committed != DECODE_TOKENS
            ):
                raise RuntimeError(f"{label} did not preserve speculative decode")
            tps[label] = _finite_positive(stats.get("generation_tps"), f"{label} TPS")
        elif drafted != 0 or accepted != 0 or committed != DECODE_TOKENS:
            raise RuntimeError(f"{label} did not preserve ordinary decode")
    parity_groups = (
        ("spec_control_4300", "spec_requested_4300"),
        ("cold_base_32768", "exact_base_32768"),
        ("cold_append_32769", "cached_append_32769", "exact_append_32769"),
    )
    for labels in parity_groups:
        for field in ("prompt_token_sha256", "completion_sha256"):
            if len({by_label[label].get(field) for label in labels}) != 1:
                raise RuntimeError(f"{field} parity failed for {labels}")
    ratio = tps["spec_requested_4300"] / tps["spec_control_4300"]
    if min(tps.values()) < MIN_SPECULATIVE_TPS or ratio < MIN_SPECULATIVE_RATIO:
        raise RuntimeError("4.3K speculative decode throughput was not preserved")
    return {
        "status": "pass",
        "speculative_tps": tps,
        "speculative_ratio": ratio,
        "exact_32768_parity": True,
        "append_32769_parity": True,
    }


def write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--mlx-lm-root", type=Path, required=True)
    parser.add_argument("--expected-exo-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=LOCAL_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.endpoint.rstrip("/") != LOCAL_ENDPOINT:
        raise RuntimeError("live canary is restricted to the local EXO endpoint")
    exo_root = Path(__file__).resolve().parents[2]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    prompts = validate_fixture(tokenizer, json.loads(args.fixture.read_text()))
    before = attest_runtime(
        args.pid_file, exo_root, args.expected_exo_commit, args.mlx_lm_root
    )
    rows = [
        stream_case(
            args.endpoint,
            case,
            prompts[case.prompt],
            timeout=args.timeout,
        )
        for case in CASES
    ]
    after = attest_runtime(
        args.pid_file, exo_root, args.expected_exo_commit, args.mlx_lm_root
    )
    if before != after:
        raise RuntimeError("runtime identity changed during the canary")
    report = {
        "schema": SCHEMA,
        "runtime": before,
        "analysis": analyze(rows),
        "results": rows,
    }
    write_new(args.output, report)
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
