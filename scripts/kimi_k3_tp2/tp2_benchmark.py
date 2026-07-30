#!/usr/bin/env python3
"""Fail-closed two-host JACCL TP2 benchmark for rank-local Kimi K3.

This intentionally does not use ``mlx_lm.utils.sharded_load``: that loader
materializes the complete checkpoint on every rank before sharding.  The
rank-local loader instead shards the empty model structure first and loads
only the checkpoint selected for the current distributed rank.

The timed path uses ``mlx_lm.generate.generate_step`` with greedy sampling,
fresh prompt-cache state, fixed output length, and synchronous token
boundaries.  That makes the phase definitions explicit:

* prefill: synchronized start through the first generated token;
* decode: tokens 2..N, excluding the first token already charged to prefill;
* end-to-end: synchronized start through the final generated token.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

from rank_local_loader import (
    CHECKPOINT_MLX_LM_KIMI_K3_SHA256,
    CONTRACT_DIGEST,
    CONTRACT_VERSION,
    DTYPE_FIX_CONTRACT,
    EFFECTIVE_SOURCE_REVISION,
    MLX_LM_COMMIT,
    MLX_LM_KIMI_K3_SHA256,
    RUNTIME_MLX_LM_COMMIT,
    SCHEMA,
    SOURCE_CONFIG_SHA256,
    SOURCE_INDEX_SHA256,
    SOURCE_REPO,
    SOURCE_REVISION,
    load_rank_local,
)

ARTIFACT_SCHEMA = "k3-rank-local-tp2-benchmark/v2"
WORLD_SIZE = 2
TRANSPORT_CONTRACT_SCHEMA = "k3-jaccl-transport/v1"
TRANSPORT_CONTRACT_ENV = "K3_TP_TRANSPORT_CONTRACT"
EXPECTED_RAIL_COUNT = 4
DEFAULT_PROMPT = (
    "You are inspecting a large software repository. Explain how to trace a "
    "request from an HTTP handler through validation, scheduling, execution, "
    "and persistence, and identify the evidence needed before changing it."
)


class BenchmarkError(RuntimeError):
    """A fail-closed benchmark precondition or consistency failure."""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_transport_contract(path: str | Path | None = None) -> dict[str, Any]:
    """Load an explicit operator-supplied JACCL topology contract."""

    configured = path if path is not None else os.environ.get(TRANSPORT_CONTRACT_ENV)
    if not configured:
        raise BenchmarkError(
            f"{TRANSPORT_CONTRACT_ENV} must name an explicit transport contract"
        )
    contract_path = Path(configured).expanduser()
    if contract_path.is_symlink():
        raise BenchmarkError(
            f"transport contract must not be a symlink: {contract_path}"
        )
    try:
        contract_path = contract_path.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkError(
            f"transport contract does not resolve: {configured}"
        ) from exc
    if not contract_path.is_file():
        raise BenchmarkError(
            f"transport contract must be a regular file: {contract_path}"
        )
    contract = _json_object(contract_path)
    if contract.get("schema") != TRANSPORT_CONTRACT_SCHEMA:
        raise BenchmarkError(
            f"transport contract schema must be {TRANSPORT_CONTRACT_SCHEMA!r}"
        )
    coordinator_host = contract.get("coordinator_host")
    if (
        not isinstance(coordinator_host, str)
        or not coordinator_host
        or ":" in coordinator_host
    ):
        raise BenchmarkError(
            "transport contract coordinator_host must be a nonempty IPv4/host name"
        )
    matrix = contract.get("device_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != WORLD_SIZE
        or any(not isinstance(row, list) or len(row) != WORLD_SIZE for row in matrix)
        or any(matrix[index][index] is not None for index in range(WORLD_SIZE))
    ):
        raise BenchmarkError(
            f"transport contract device_matrix must be {WORLD_SIZE}x{WORLD_SIZE} "
            "with null diagonal entries"
        )
    for source in range(WORLD_SIZE):
        for destination in range(WORLD_SIZE):
            if source == destination:
                continue
            devices = matrix[source][destination]
            if (
                not isinstance(devices, list)
                or len(devices) != EXPECTED_RAIL_COUNT
                or any(not isinstance(device, str) or not device for device in devices)
                or len(set(devices)) != EXPECTED_RAIL_COUNT
            ):
                raise BenchmarkError(
                    "transport contract requires exactly four unique named RDMA "
                    f"rails for matrix entry [{source}][{destination}]"
                )
    normalized = {
        "schema": TRANSPORT_CONTRACT_SCHEMA,
        "coordinator_host": coordinator_host,
        "device_matrix": matrix,
    }
    return {
        **normalized,
        "path": str(contract_path),
        "sha256": _sha256_json(normalized),
    }


def inspect_jaccl_ring_transport(
    *,
    rank: int,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate the exact two-Mac, four-rail JACCL ring environment.

    ``mlx.launch --backend jaccl-ring`` materializes the hostfile's RDMA matrix
    into a per-rank JSON file and exports its path through ``MLX_IBV_DEVICES``.
    The runtime values must match a separate explicit contract so the source
    tree contains no deployment inventory.
    """

    contract = load_transport_contract(contract_path)
    if os.environ.get("MLX_JACCL_RING") != "1":
        raise BenchmarkError(
            "MLX_JACCL_RING=1 is required; launch with --backend jaccl-ring"
        )
    if os.environ.get("MLX_RANK") != str(rank):
        raise BenchmarkError(f"MLX_RANK does not match distributed rank {rank}")

    coordinator = os.environ.get("MLX_JACCL_COORDINATOR", "")
    host, separator, port_text = coordinator.rpartition(":")
    if (
        separator != ":"
        or host != contract["coordinator_host"]
        or not port_text.isdigit()
        or not 1 <= int(port_text) <= 65535
    ):
        raise BenchmarkError(
            "MLX_JACCL_COORDINATOR does not match the explicit transport "
            f"contract host {contract['coordinator_host']!r}"
        )

    matrix_text = os.environ.get("MLX_IBV_DEVICES")
    if not matrix_text:
        raise BenchmarkError("MLX_IBV_DEVICES is not set")
    matrix_path = Path(matrix_text).expanduser()
    if matrix_path.is_symlink():
        raise BenchmarkError(f"MLX_IBV_DEVICES must not be a symlink: {matrix_path}")
    try:
        matrix_path = matrix_path.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkError(
            f"MLX_IBV_DEVICES does not resolve to a file: {matrix_text}"
        ) from exc
    if not matrix_path.is_file():
        raise BenchmarkError(
            f"MLX_IBV_DEVICES must be a regular non-symlink file: {matrix_path}"
        )
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"cannot read MLX_IBV_DEVICES matrix {matrix_path}: {exc}"
        ) from exc
    if matrix != contract["device_matrix"]:
        raise BenchmarkError(
            "MLX_IBV_DEVICES does not match the explicit four-rail transport contract"
        )

    return {
        "mode": "jaccl-ring",
        "ring": True,
        "coordinator": coordinator,
        "coordinator_sha256": hashlib.sha256(coordinator.encode("utf-8")).hexdigest(),
        "device_matrix": matrix,
        "device_matrix_sha256": _sha256_json(matrix),
        "device_matrix_path": str(matrix_path),
        "transport_contract_path": contract["path"],
        "transport_contract_sha256": contract["sha256"],
    }


def token_digest(tokens: Iterable[int]) -> str:
    """Hash token IDs in an endian-independent fixed-width representation."""

    digest = hashlib.sha256()
    for token in tokens:
        value = int(token)
        if not 0 <= value < 2**32:
            raise BenchmarkError(f"token ID is outside uint32: {value}")
        digest.update(value.to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def _token_list(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise BenchmarkError(
            "tokenizer.apply_chat_template did not return a token sequence"
        )
    return [int(token) for token in value]


def _chat_tokens(tokenizer: Any, content: str) -> list[int]:
    messages = [{"role": "user", "content": content}]
    try:
        value = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    except TypeError:
        # Some tokenizer wrappers omit the explicit ``tokenize`` parameter but
        # still return IDs, which is the API used by upstream sharded_generate.
        value = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    return _token_list(value)


def _common_affixes(empty: Sequence[int], populated: Sequence[int]) -> tuple[int, int]:
    prefix = 0
    prefix_limit = min(len(empty), len(populated))
    while prefix < prefix_limit and empty[prefix] == populated[prefix]:
        prefix += 1

    suffix = 0
    suffix_limit = min(len(empty) - prefix, len(populated) - prefix)
    while (
        suffix < suffix_limit
        and empty[len(empty) - 1 - suffix] == populated[len(populated) - 1 - suffix]
    ):
        suffix += 1
    return prefix, suffix


def build_prompt_tokens(
    tokenizer: Any,
    prompt: str,
    token_target: int | None,
) -> tuple[list[int], str]:
    """Apply the chat template, optionally producing an exact synthetic length.

    For an exact target, the common empty/full template prefix and generation
    suffix are preserved.  Only the user-content token span is repeated or
    truncated.  This avoids chopping off Kimi's assistant-generation marker.
    """

    populated = _chat_tokens(tokenizer, prompt)
    if not populated:
        raise BenchmarkError("chat template produced an empty prompt")
    if token_target is None:
        return populated, "chat_template"
    if token_target <= 0:
        raise BenchmarkError("--prompt-token-target must be positive")

    empty = _chat_tokens(tokenizer, "")
    prefix_count, suffix_count = _common_affixes(empty, populated)
    suffix_start = len(populated) - suffix_count if suffix_count else len(populated)
    prefix = populated[:prefix_count]
    suffix = populated[suffix_start:]
    payload = populated[prefix_count:suffix_start]
    overhead = len(prefix) + len(suffix)
    if token_target < overhead:
        raise BenchmarkError(
            f"prompt target {token_target} is smaller than the chat-template "
            f"overhead {overhead}"
        )
    payload_target = token_target - overhead
    if payload_target and not payload:
        raise BenchmarkError(
            "cannot synthesize prompt: user payload token span is empty"
        )
    if payload_target:
        repeats = (payload_target + len(payload) - 1) // len(payload)
        payload = (payload * repeats)[:payload_target]
    else:
        payload = []
    result = prefix + payload + suffix
    if len(result) != token_target:
        raise AssertionError("exact prompt construction returned the wrong length")
    return result, "chat_template_exact_repeated_payload"


def select_rank_checkpoint(
    configured: Sequence[str | Path],
    *,
    rank: int,
    world_size: int,
) -> Path:
    if world_size != WORLD_SIZE:
        raise BenchmarkError(f"Kimi K3 launcher requires TP2, got TP{world_size}")
    if len(configured) != WORLD_SIZE:
        raise BenchmarkError(
            f"expected exactly {WORLD_SIZE} --rank-checkpoint values, "
            f"got {len(configured)}"
        )
    if not 0 <= rank < world_size:
        raise BenchmarkError(f"invalid distributed rank {rank} for TP{world_size}")
    selected = Path(configured[rank]).expanduser()
    try:
        selected = selected.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkError(
            f"rank {rank} checkpoint does not exist: {selected}"
        ) from exc
    if not selected.is_dir():
        raise BenchmarkError(f"rank {rank} checkpoint is not a directory: {selected}")
    return selected


def inspect_pinned_manifest(model_dir: Path, *, rank: int) -> dict[str, Any]:
    """Authenticate the immutable source/converter/TP identity before loading."""

    manifest_path = model_dir / "tp_manifest.json"
    manifest = _json_object(manifest_path)
    source = manifest.get("source")
    runtime = manifest.get("runtime")
    tp = manifest.get("tp")
    files = manifest.get("files")
    tensors = manifest.get("tensors")
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise BenchmarkError("rank-local manifest is incomplete or has wrong schema")
    if not isinstance(source, dict) or not isinstance(runtime, dict):
        raise BenchmarkError("rank-local manifest source/runtime records are malformed")
    if not isinstance(tp, dict):
        raise BenchmarkError("rank-local manifest TP record is malformed")
    expected_source = {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "config_sha256": SOURCE_CONFIG_SHA256,
        "index_sha256": SOURCE_INDEX_SHA256,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise BenchmarkError(
                f"manifest source {key} is not pinned: "
                f"expected {expected!r}, got {source.get(key)!r}"
            )
    expected_converter = {
        "mlx_lm_commit": MLX_LM_COMMIT,
        "mlx_lm_kimi_k3_sha256": CHECKPOINT_MLX_LM_KIMI_K3_SHA256,
    }
    for key, expected in expected_converter.items():
        if runtime.get(key) != expected:
            raise BenchmarkError(
                f"manifest checkpoint converter {key} is not pinned: "
                f"expected {expected!r}, got {runtime.get(key)!r}"
            )
    expected_tp = {
        "rank": rank,
        "world_size": WORLD_SIZE,
        "contract": CONTRACT_VERSION,
        "contract_digest": CONTRACT_DIGEST,
    }
    for key, expected in expected_tp.items():
        if tp.get(key) != expected:
            raise BenchmarkError(
                f"manifest TP {key} mismatch: "
                f"expected {expected!r}, got {tp.get(key)!r}"
            )
    if not isinstance(files, dict) or not files:
        raise BenchmarkError("rank-local manifest has no files")
    if not isinstance(tensors, dict) or not tensors:
        raise BenchmarkError("rank-local manifest has no tensors")
    config_path = model_dir / "config.json"
    if _sha256_file(config_path) != SOURCE_CONFIG_SHA256:
        raise BenchmarkError("rank-local config.json is not the pinned K3 config")
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise BenchmarkError("rank-local model.safetensors.index.json is missing")
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "file_count": len(files),
        "tensor_count": len(tensors),
        "rank_data_bytes": int(manifest.get("rank_data_bytes", 0)),
    }


def inspect_runtime_k3_source() -> dict[str, str]:
    """Prove that Python resolved the pinned execution-time K3 source file."""

    from mlx_lm.models import kimi_k3

    source = Path(inspect.getfile(kimi_k3)).resolve()
    actual = _sha256_file(source)
    if actual != MLX_LM_KIMI_K3_SHA256:
        raise BenchmarkError(
            "imported mlx_lm.models.kimi_k3 does not match the runtime pin: "
            f"expected {MLX_LM_KIMI_K3_SHA256}, got {actual} at {source}"
        )
    return {"path": str(source), "sha256": actual}


def init_distributed(mx: Any, backend: str = "jaccl") -> Any:
    """Request strict backend initialization, falling back only for old APIs."""

    try:
        return mx.distributed.init(backend=backend, strict=True)
    except TypeError as exc:
        message = str(exc).lower()
        if "strict" not in message and "keyword" not in message:
            raise
        return mx.distributed.init(backend=backend)


def _flatten_scalars(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_scalars(item))
        return flattened
    return [value]


def barrier(mx: Any, group: Any) -> None:
    reached = mx.distributed.all_sum(
        mx.array(1, dtype=mx.int32),
        group=group,
    )
    mx.eval(reached)
    if int(reached.item()) != int(group.size()):
        raise BenchmarkError("distributed completion barrier returned wrong rank count")


def gather_floats(mx: Any, group: Any, values: Sequence[float]) -> list[list[float]]:
    local = mx.array(list(values), dtype=mx.float32)
    gathered = mx.distributed.all_gather(local, group=group)
    mx.eval(gathered)
    flat = [float(item) for item in _flatten_scalars(gathered)]
    width = len(values)
    expected = int(group.size()) * width
    if len(flat) != expected:
        raise BenchmarkError(
            f"all_gather returned {len(flat)} timing values, expected {expected}"
        )
    return [flat[start : start + width] for start in range(0, expected, width)]


def gather_digests(mx: Any, group: Any, digest: str) -> list[str]:
    try:
        raw = bytes.fromhex(digest)
    except ValueError as exc:
        raise BenchmarkError(f"invalid SHA-256 digest {digest!r}") from exc
    if len(raw) != 32:
        raise BenchmarkError(f"expected SHA-256 digest, got {len(raw)} bytes")
    local = mx.array(list(raw), dtype=mx.int32)
    gathered = mx.distributed.all_gather(local, group=group)
    mx.eval(gathered)
    flat = [int(item) for item in _flatten_scalars(gathered)]
    expected = int(group.size()) * 32
    if len(flat) != expected:
        raise BenchmarkError(
            f"all_gather returned {len(flat)} digest bytes, expected {expected}"
        )
    result = []
    for start in range(0, expected, 32):
        chunk = flat[start : start + 32]
        if any(not 0 <= item <= 255 for item in chunk):
            raise BenchmarkError("all_gather returned an invalid digest byte")
        result.append(bytes(chunk).hex())
    return result


def require_shared_digest(
    mx: Any,
    group: Any,
    digest: str,
    label: str,
) -> list[str]:
    gathered = gather_digests(mx, group, digest)
    if any(value != digest for value in gathered):
        raise BenchmarkError(f"{label} differs across TP ranks: {gathered}")
    return gathered


def phase_metrics(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    start: float,
    first_token: float,
    end: float,
) -> dict[str, float | int | None]:
    if not start <= first_token <= end:
        raise BenchmarkError("invalid phase clock ordering")
    if prompt_tokens <= 0 or generated_tokens <= 0:
        raise BenchmarkError("phase metrics require positive token counts")
    prefill_seconds = first_token - start
    decode_seconds = end - first_token
    e2e_seconds = end - start
    decode_tokens = max(generated_tokens - 1, 0)
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "e2e_seconds": e2e_seconds,
        "prefill_tps": (
            prompt_tokens / prefill_seconds if prefill_seconds > 0 else None
        ),
        "decode_tps": (
            decode_tokens / decode_seconds
            if decode_tokens > 0 and decode_seconds > 0
            else None
        ),
    }


def critical_path_metrics(
    per_rank: Sequence[Sequence[float]],
    *,
    prompt_tokens: int,
    generated_tokens: int,
) -> dict[str, float | int | None]:
    if not per_rank or any(len(row) != 4 for row in per_rank):
        raise BenchmarkError("expected prefill/decode/e2e/peak values per rank")
    prefill_seconds = max(row[0] for row in per_rank)
    decode_seconds = max(row[1] for row in per_rank)
    e2e_seconds = max(row[2] for row in per_rank)
    decode_tokens = max(generated_tokens - 1, 0)
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "e2e_seconds": e2e_seconds,
        "prefill_tps": (
            prompt_tokens / prefill_seconds if prefill_seconds > 0 else None
        ),
        "decode_tps": (
            decode_tokens / decode_seconds
            if decode_tokens > 0 and decode_seconds > 0
            else None
        ),
        "peak_memory_gb": max(row[3] for row in per_rank),
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete artifact without overwriting an existing result."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{os.getpid()}"
    if temporary.exists():
        raise BenchmarkError(f"stale artifact staging path exists: {temporary}")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise BenchmarkError(f"refusing to overwrite artifact: {path}") from exc
    except OSError as exc:
        raise BenchmarkError(f"cannot publish artifact {path}: {exc}") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _event(rank: int, event: str, **values: Any) -> None:
    payload = {"event": event, "rank": rank, **values}
    print(
        "K3_TP2_EVENT " + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def _median(values: Sequence[float | None]) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    return statistics.median(concrete) if concrete else None


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    import mlx.core as mx
    from mlx_lm.generate import generate_step, generation_stream, wired_limit

    init_started = time.perf_counter()
    group = init_distributed(mx, backend="jaccl")
    rank = int(group.rank())
    world_size = int(group.size())
    if world_size != WORLD_SIZE:
        raise BenchmarkError(f"expected JACCL TP2, got TP{world_size}")
    init_seconds = time.perf_counter() - init_started
    transport = inspect_jaccl_ring_transport(rank=rank)
    transport_matrix_digests = gather_digests(
        mx,
        group,
        transport["device_matrix_sha256"],
    )
    transport_coordinator_digests = gather_digests(
        mx,
        group,
        transport["coordinator_sha256"],
    )
    transport_contract_digests = gather_digests(
        mx,
        group,
        transport["transport_contract_sha256"],
    )
    if any(
        value != transport["device_matrix_sha256"] for value in transport_matrix_digests
    ):
        raise BenchmarkError(
            f"JACCL device matrix differs across ranks: {transport_matrix_digests}"
        )
    if any(
        value != transport["coordinator_sha256"]
        for value in transport_coordinator_digests
    ):
        raise BenchmarkError(
            f"JACCL coordinator differs across ranks: {transport_coordinator_digests}"
        )
    if any(
        value != transport["transport_contract_sha256"]
        for value in transport_contract_digests
    ):
        raise BenchmarkError(
            "JACCL transport contract differs across ranks: "
            f"{transport_contract_digests}"
        )

    model_dir = select_rank_checkpoint(
        args.rank_checkpoint,
        rank=rank,
        world_size=world_size,
    )
    manifest = inspect_pinned_manifest(model_dir, rank=rank)
    manifest_digests = gather_digests(mx, group, manifest["manifest_sha256"])
    runtime_source = inspect_runtime_k3_source()
    runtime_source_digests = gather_digests(mx, group, runtime_source["sha256"])
    if any(value != MLX_LM_KIMI_K3_SHA256 for value in runtime_source_digests):
        raise BenchmarkError(
            f"K3 source digest differs across ranks: {runtime_source_digests}"
        )
    _event(
        rank,
        "runtime_preflight",
        mlx_lm_commit=RUNTIME_MLX_LM_COMMIT,
        kimi_k3_path=runtime_source["path"],
        kimi_k3_sha256=runtime_source["sha256"],
    )

    _event(rank, "load_start", checkpoint=str(model_dir))
    barrier(mx, group)
    load_started = time.perf_counter()
    model, tokenizer, loaded_config = load_rank_local(
        model_dir,
        tensor_group=group,
        verify_file_hashes=args.verify_file_hashes,
        return_config=True,
    )
    compatibility_transform = loaded_config.get("_rank_local_compatibility_transform")
    if (
        not isinstance(compatibility_transform, dict)
        or compatibility_transform.get("contract") != DTYPE_FIX_CONTRACT
        or compatibility_transform.get("effective_source_revision")
        != EFFECTIVE_SOURCE_REVISION
    ):
        raise BenchmarkError("rank-local K3 dtype correction was not attested")
    compatibility_transform_sha256 = _sha256_json(compatibility_transform)
    compatibility_transform_sha256_by_rank = require_shared_digest(
        mx,
        group,
        compatibility_transform_sha256,
        "rank-local compatibility transform",
    )
    load_seconds = time.perf_counter() - load_started
    load_peak_gb = float(mx.get_peak_memory()) / 1e9
    barrier(mx, group)
    load_rows = gather_floats(mx, group, [load_seconds, load_peak_gb])
    _event(rank, "load_complete", seconds=round(load_seconds, 6))

    prompt_ids, prompt_mode = build_prompt_tokens(
        tokenizer,
        args.prompt,
        args.prompt_token_target,
    )
    prompt_sha256 = token_digest(prompt_ids)
    prompt_sha256_by_rank = require_shared_digest(
        mx,
        group,
        prompt_sha256,
        "prompt token IDs",
    )
    prompt = mx.array(prompt_ids, dtype=mx.uint32)
    mx.eval(prompt)
    if args.max_kv_size is not None and args.max_kv_size <= 0:
        raise BenchmarkError("--max-kv-size must be positive")

    mx.random.seed(args.seed)
    device_info = mx.device_info()
    wired_target_bytes = (
        int(device_info["max_recommended_working_set_size"])
        if args.wired_limit and mx.metal.is_available()
        else None
    )
    run_records: list[dict[str, Any]] = []
    expected_output_sha: str | None = None
    for run_index in range(args.runs):
        # This resets only the memory watermark.  generate_step receives
        # prompt_cache=None below and therefore allocates a fresh KV state.
        mx.reset_peak_memory()
        barrier(mx, group)
        _event(rank, "run_start", run=run_index + 1)
        started = time.perf_counter()
        first_token_at: float | None = None
        output_ids: list[int] = []
        generation_context = (
            wired_limit(model, [generation_stream])
            if args.wired_limit
            else contextlib.nullcontext()
        )
        with generation_context:
            generator = generate_step(
                prompt,
                model,
                max_tokens=args.output_tokens,
                async_lookahead=args.async_lookahead,
                max_kv_size=args.max_kv_size,
                prompt_cache=None,
                prefill_step_size=args.prefill_step_size,
            )
            for token, _logprobs in generator:
                observed_at = time.perf_counter()
                if first_token_at is None:
                    first_token_at = observed_at
                output_ids.append(int(token))
            mx.synchronize()
        ended = time.perf_counter()
        if first_token_at is None:
            raise BenchmarkError("generation produced no tokens")
        if len(output_ids) != args.output_tokens:
            raise BenchmarkError(
                f"expected {args.output_tokens} generated tokens, got {len(output_ids)}"
            )
        output_sha = token_digest(output_ids)
        require_shared_digest(mx, group, output_sha, "generated token IDs")
        if expected_output_sha is None:
            expected_output_sha = output_sha
        elif output_sha != expected_output_sha:
            raise BenchmarkError(
                "greedy generated token IDs changed between fresh-cache runs"
            )

        local_metrics = phase_metrics(
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(output_ids),
            start=started,
            first_token=first_token_at,
            end=ended,
        )
        peak_memory_gb = float(mx.get_peak_memory()) / 1e9
        timing_rows = gather_floats(
            mx,
            group,
            [
                float(local_metrics["prefill_seconds"]),
                float(local_metrics["decode_seconds"]),
                float(local_metrics["e2e_seconds"]),
                peak_memory_gb,
            ],
        )
        critical = critical_path_metrics(
            timing_rows,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(output_ids),
        )
        record = {
            "run": run_index + 1,
            "critical_path": critical,
            "per_rank": [
                {
                    "rank": timing_rank,
                    "prefill_seconds": row[0],
                    "decode_seconds": row[1],
                    "e2e_seconds": row[2],
                    "peak_memory_gb": row[3],
                }
                for timing_rank, row in enumerate(timing_rows)
            ],
            "output_token_sha256": output_sha,
            "output_token_ids": output_ids,
        }
        run_records.append(record)
        _event(
            rank,
            "run_complete",
            run=run_index + 1,
            prefill_tps=critical["prefill_tps"],
            decode_tps=critical["decode_tps"],
        )

    if rank != 0:
        return None

    try:
        output_text = tokenizer.decode(
            run_records[0]["output_token_ids"],
            skip_special_tokens=False,
        )
    except TypeError:
        output_text = tokenizer.decode(run_records[0]["output_token_ids"])
    critical_rows = [record["critical_path"] for record in run_records]
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "status": "PASS",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "effective_revision": EFFECTIVE_SOURCE_REVISION,
            "config_sha256": SOURCE_CONFIG_SHA256,
            "index_sha256": SOURCE_INDEX_SHA256,
            "compatibility_transform": compatibility_transform,
            "compatibility_transform_sha256_by_rank": (
                compatibility_transform_sha256_by_rank
            ),
        },
        "runtime_contract": {
            "checkpoint_converter_mlx_lm_commit": MLX_LM_COMMIT,
            "checkpoint_mlx_lm_kimi_k3_sha256": (CHECKPOINT_MLX_LM_KIMI_K3_SHA256),
            "execution_runtime_mlx_lm_commit": RUNTIME_MLX_LM_COMMIT,
            "execution_runtime_kimi_k3_sha256": MLX_LM_KIMI_K3_SHA256,
            "imported_kimi_k3_path_rank0": runtime_source["path"],
            "imported_kimi_k3_sha256_by_rank": runtime_source_digests,
            "tp_contract": CONTRACT_VERSION,
            "tp_contract_digest": CONTRACT_DIGEST,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "distributed": {
            "backend": "jaccl-ring",
            "initialization_backend": "jaccl",
            "world_size": world_size,
            "strict_init_requested": True,
            "init_seconds_rank0": init_seconds,
            "checkpoint_argument_by_rank": [str(path) for path in args.rank_checkpoint],
            "checkpoint_manifest_sha256_by_rank": manifest_digests,
            "load_seconds_by_rank": [row[0] for row in load_rows],
            "load_peak_memory_gb_by_rank": [row[1] for row in load_rows],
            "transport": {
                "mode": transport["mode"],
                "ring": transport["ring"],
                "coordinator_rank0": transport["coordinator"],
                "coordinator_sha256_by_rank": transport_coordinator_digests,
                "device_matrix": transport["device_matrix"],
                "device_matrix_sha256_by_rank": transport_matrix_digests,
                "transport_contract_sha256_by_rank": transport_contract_digests,
                "environment_verified_by_rank": [True] * world_size,
            },
        },
        "benchmark": {
            "runs": args.runs,
            "prompt_mode": prompt_mode,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": prompt_sha256,
            "prompt_token_sha256_by_rank": prompt_sha256_by_rank,
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "sampling": "greedy_argmax",
            "eos_early_stop": False,
            "prefix_cache": "disabled_fresh_per_run",
            "async_lookahead": args.async_lookahead,
            "wired_memory": {
                "enabled": args.wired_limit,
                "implementation": (
                    "mlx_lm.generate.wired_limit"
                    if args.wired_limit
                    else "disabled_control"
                ),
                "target_bytes": wired_target_bytes,
                "target_gb_decimal": (
                    wired_target_bytes / 1e9 if wired_target_bytes is not None else None
                ),
            },
            "prefill_step_size": args.prefill_step_size,
            "max_kv_size": args.max_kv_size,
            "timing_semantics": {
                "prefill": "barrier-to-first-generated-token",
                "decode": "tokens-2-through-N; first token excluded",
                "e2e": "barrier-to-final-generated-token",
                "aggregation": "maximum elapsed phase across ranks",
            },
            "median_prefill_tps": _median(
                [row["prefill_tps"] for row in critical_rows]
            ),
            "median_decode_tps": _median([row["decode_tps"] for row in critical_rows]),
            "median_e2e_seconds": _median(
                [row["e2e_seconds"] for row in critical_rows]
            ),
            "output_text": output_text,
            "runs_detail": run_records,
        },
    }
    atomic_write_json(args.artifact, artifact)
    compact = {
        "artifact": str(args.artifact.expanduser().resolve()),
        "decode_tps_median": artifact["benchmark"]["median_decode_tps"],
        "e2e_seconds_median": artifact["benchmark"]["median_e2e_seconds"],
        "output_sha256": expected_output_sha,
        "prefill_tps_median": artifact["benchmark"]["median_prefill_tps"],
        "prompt_tokens": len(prompt_ids),
        "status": "PASS",
    }
    print(
        "K3_TP2_RESULT " + json.dumps(compact, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct rank-local Kimi K3 JACCL TP2 smoke/benchmark"
    )
    parser.add_argument(
        "--rank-checkpoint",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "checkpoint root for the corresponding rank; pass exactly twice "
            "in rank order (paths may be identical across host-local namespaces)"
        ),
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-token-target", type=int)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-kv-size", type=int)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--wired-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use mlx-lm's model-sized Metal wired-memory context during "
            "generation (default: enabled)"
        ),
    )
    parser.add_argument(
        "--async-lookahead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use mlx-lm's default asynchronous token lookahead (default: enabled)",
    )
    parser.add_argument(
        "--verify-file-hashes",
        action="store_true",
        help="rehash every rank file before loading (a full ~400 GB disk pass)",
    )
    args = parser.parse_args(argv)
    if len(args.rank_checkpoint) != WORLD_SIZE:
        parser.error(f"--rank-checkpoint must be supplied exactly {WORLD_SIZE} times")
    if args.output_tokens <= 0:
        parser.error("--output-tokens must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.prefill_step_size <= 0:
        parser.error("--prefill-step-size must be positive")
    if args.prompt_token_target is not None and args.prompt_token_target <= 0:
        parser.error("--prompt-token-target must be positive")
    if args.max_kv_size is not None and args.max_kv_size <= 0:
        parser.error("--max-kv-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    rank_text = os.environ.get("MLX_RANK")
    try:
        args = parse_args(argv)
        run(args)
        return 0
    except SystemExit as exc:
        # Preserve argparse's normal help/error exit codes and output.
        return int(exc.code) if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        error: BaseException = BenchmarkError("interrupted")
    except BaseException as exc:
        error = exc
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "rank": int(rank_text) if rank_text and rank_text.isdigit() else None,
        "status": "ERROR",
    }
    print(
        "K3_TP2_ERROR " + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )
    if os.environ.get("K3_TP2_TRACEBACK") == "1":
        traceback.print_exception(error, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
