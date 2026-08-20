"""Fail-closed, request-local Kimi K3 frontier telemetry.

The production path is unchanged unless ``ENABLE_ENV`` is exactly ``1``.
An enabled TP2 process accepts exactly sixteen strictly ordered requests.  Each
request is bound to an executor-supplied index and nonce SHA-256, owns one reset
generation, and publishes one rank-local, sanitized O_EXCL receipt.  The last
request also publishes an aggregate completion receipt covering all sixteen
request files from the same PID.

No prompt, completion, request body, environment map, hostname, or filesystem
path is retained.  RSS is sampled by a bounded request-scoped thread and is
therefore named an *observed* peak.  ``resource.ru_maxrss`` is separately named
as the process-lifetime high-water mark.  MLX active-memory peak is the
authoritative allocator high-water counter after the existing request reset;
Metal residency (active + cache) is a bounded lifecycle observation because
MLX exposes no residency high-water counter.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, cast

ENABLE_ENV = "EXO_MLX_KIMI_K3_FRONTIER_TELEMETRY_V4"
DIRECTORY_ENV = "EXO_MLX_KIMI_K3_FRONTIER_TELEMETRY_DIR"
SESSION_SHA256_ENV = "EXO_MLX_KIMI_K3_FRONTIER_TELEMETRY_SESSION_SHA256"
SCHEMA = "kimi-k3-frontier-request-telemetry/v4"
PROCESS_SCHEMA = "kimi-k3-frontier-process-telemetry/v4"
MARKER_SCHEMA = "kimi-k3-w3-composition-receipt/v4"
CORE_SCHEMA = "kimi-k3-w3-composition-receipt/v2"
EXPECTED_REQUESTS = 16
RSS_SAMPLE_INTERVAL_SECONDS = 0.010
RSS_SAMPLE_LIMIT = 120_000
_HEX = frozenset("0123456789abcdef")
_MAX_COUNTER = 0x7FFFFFFFFFFFFFFF

REPLAYSSM_FIELDS = (
    "replayssm_telemetry_revision_before",
    "replayssm_telemetry_revision_after",
    "replayssm_telemetry_revision_delta",
    "replayssm_attempted_prepares_delta",
    "replayssm_batched_prepares_delta",
    "replayssm_batched_commits_delta",
    "replayssm_identity_prepares_delta",
    "replayssm_identity_commits_delta",
    "replayssm_fallback_prepares_delta",
    "replayssm_fallback_commits_delta",
    "replayssm_batched_errors_delta",
    "replayssm_identity_errors_delta",
    "replayssm_layers_batched_delta",
    "replayssm_layers_identity_committed_delta",
)
REQUEST_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "session_sha256",
        "request_index",
        "request_nonce_sha256",
        "rank",
        "world_size",
        "pid",
        "reset_generation",
        "reset_fence_sha256",
        "receipt_v2_core_sha256",
        "receipt_request_sequence",
        "receipt_request_token",
        *REPLAYSSM_FIELDS,
        "process_rss_baseline_bytes",
        "process_rss_final_bytes",
        "process_rss_final_delta_bytes",
        "process_rss_observed_peak_bytes",
        "process_rss_observed_peak_delta_bytes",
        "process_rss_lifetime_highwater_baseline_bytes",
        "process_rss_lifetime_highwater_after_bytes",
        "process_rss_lifetime_highwater_delta_bytes",
        "rss_sample_interval_ns",
        "rss_sample_count",
        "metal_active_baseline_bytes",
        "metal_active_final_bytes",
        "metal_active_final_delta_bytes",
        "metal_active_peak_bytes",
        "metal_active_peak_delta_bytes",
        "metal_cache_baseline_bytes",
        "metal_cache_final_bytes",
        "metal_cache_final_delta_bytes",
        "metal_residency_baseline_bytes",
        "metal_residency_final_bytes",
        "metal_residency_final_delta_bytes",
        "metal_residency_observed_peak_bytes",
        "metal_residency_observed_peak_delta_bytes",
        "metal_observation_count",
        "rss_peak_semantics",
        "rss_lifetime_highwater_semantics",
        "metal_active_peak_semantics",
        "metal_residency_peak_semantics",
        "evidence_sha256",
    }
)
PROCESS_REQUEST_ROW_KEYS = frozenset(
    {
        "request_index",
        "request_nonce_sha256",
        "request_file_sha256",
        "receipt_request_sequence",
        "receipt_request_token",
    }
)
PROCESS_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "session_sha256",
        "rank",
        "world_size",
        "pid",
        "request_count",
        "requests",
        "request_chain_sha256",
        "evidence_sha256",
    }
)


class _MxMemory(Protocol):
    def get_active_memory(self) -> int: ...

    def get_cache_memory(self) -> int: ...

    def get_peak_memory(self) -> int: ...


class _ProcessMemory(Protocol):
    def memory_info(self) -> object: ...


@dataclass(frozen=True)
class _MemorySample:
    rss_bytes: int
    rss_lifetime_highwater_bytes: int
    metal_active_bytes: int
    metal_cache_bytes: int
    metal_residency_bytes: int
    metal_active_peak_bytes: int


@dataclass
class FrontierTelemetryContext:
    request_index: int
    request_nonce_sha256: str
    session_sha256: str
    rank: int
    world_size: int
    pid: int
    reset_generation: int
    directory: Path
    directory_device: int
    directory_inode: int
    mx_module: _MxMemory
    process: _ProcessMemory
    reset_captured: bool = False
    finalized: bool = False
    baseline: _MemorySample | None = None
    rss_observed_peak_bytes: int = 0
    rss_lifetime_highwater_after_bytes: int = 0
    metal_active_peak_bytes: int = 0
    metal_residency_observed_peak_bytes: int = 0
    metal_observation_count: int = 0
    rss_sample_count: int = 0
    rss_sampler_error: str | None = None
    reset_fence_sha256: str | None = None
    _sample_lock: threading.Lock = field(default_factory=threading.Lock)
    _sampler_stop: threading.Event = field(default_factory=threading.Event)
    _sampler: threading.Thread | None = None


@dataclass(frozen=True)
class FrontierTelemetryEvidence:
    request_index: int
    request_nonce_sha256: str
    reset_generation: int
    reset_fence_sha256: str
    receipt_v2_core_sha256: str
    request_file_sha256: str
    process_complete_file_sha256: str | None


_state_lock = threading.Lock()
_active: FrontierTelemetryContext | None = None
_poisoned = False
_attempted_indices: set[int] = set()
_published: dict[int, dict[str, object]] = {}
_used_nonces: set[str] = set()
_used_receipt_tokens: set[int] = set()
_next_reset_generation = 0


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(cast(str, value)) == 64
        and all(character in _HEX for character in cast(str, value))
    )


def enabled() -> bool:
    raw = os.environ.get(ENABLE_ENV, "0")
    if raw not in {"0", "1"}:
        raise ValueError(f"{ENABLE_ENV} must be exactly 0 or 1")
    return raw == "1"


def reject_orphan_request_fields(
    request_index: object,
    request_nonce_sha256: object,
) -> None:
    """Reject a half-enabled request while keeping the disabled path inert."""

    if request_index is not None or request_nonce_sha256 is not None:
        raise ValueError(
            "Kimi K3 frontier request headers require request telemetry v4"
        )


def _directory_contract() -> tuple[Path, int, int]:
    raw = os.environ.get(DIRECTORY_ENV)
    if raw is None or raw == "":
        raise ValueError(f"{DIRECTORY_ENV} is required when telemetry v4 is enabled")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{DIRECTORY_ENV} must be an absolute directory")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{DIRECTORY_ENV} must be a non-symlink directory")
    if info.st_uid != os.getuid():
        raise ValueError("frontier telemetry directory must be process-owned")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("frontier telemetry directory mode must be exactly 0700")
    return path, info.st_dev, info.st_ino


def _require_exact_prior_inventory(
    path: Path,
    *,
    device: int,
    inode: int,
    index: int,
    rank: int,
) -> None:
    """Prove a fresh directory or the exact files from this live sequence."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_fd = os.open(path, flags)
    try:
        opened_directory = os.fstat(directory_fd)
        if (opened_directory.st_dev, opened_directory.st_ino) != (device, inode):
            raise RuntimeError("frontier telemetry directory changed during open")
        expected = _expected_prior_inventory(index=index, rank=rank)
        _require_exact_inventory_fd(directory_fd, expected)
    finally:
        os.close(directory_fd)


def _expected_prior_inventory(*, index: int, rank: int) -> dict[str, str]:
    expected = {
        f"request-{prior:02d}-rank{rank}.json": cast(
            str,
            _published[prior]["request_file_sha256"],
        )
        for prior in range(1, index)
        if prior in _published
    }
    if len(expected) != index - 1:
        raise RuntimeError("frontier telemetry live sequence ledger is incomplete")
    return expected


def _require_exact_inventory_fd(
    directory_fd: int,
    expected: Mapping[str, str],
    *,
    optional_names: frozenset[str] = frozenset(),
) -> None:
    observed_names = set(os.listdir(directory_fd))
    if (
        not set(expected) <= observed_names
        or not (observed_names - set(expected)) <= optional_names
    ):
        raise RuntimeError(
            "frontier telemetry directory is not the exact live sequence"
        )
    for name, expected_sha in expected.items():
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
        ):
            raise RuntimeError("frontier telemetry prior evidence inode is invalid")
        read_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        file_fd = os.open(name, read_flags, dir_fd=directory_fd)
        try:
            digest = hashlib.sha256()
            while True:
                chunk = os.read(file_fd, 65_536)
                if not chunk:
                    break
                digest.update(chunk)
            opened = os.fstat(file_fd)
            if (opened.st_dev, opened.st_ino) != (
                info.st_dev,
                info.st_ino,
            ) or digest.hexdigest() != expected_sha:
                raise RuntimeError("frontier telemetry prior evidence digest drifted")
        finally:
            os.close(file_fd)


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= cast(int, value) <= _MAX_COUNTER:
        raise ValueError(f"{name} must be a bounded nonnegative integer")
    return cast(int, value)


def _rss_lifetime_highwater_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if type(value) not in {int, float} or value < 0:
        raise ValueError("process RSS lifetime high-water counter is invalid")
    # Darwin reports bytes. Linux and the BSD-derived Python contract outside
    # Darwin report KiB; the live target is Darwin but tests remain portable.
    multiplier = 1 if sys.platform == "darwin" else 1024
    result = int(value) * multiplier
    return _nonnegative_int(result, "process RSS lifetime high-water")


def _rss_bytes(process: _ProcessMemory) -> int:
    info = process.memory_info()
    return _nonnegative_int(getattr(info, "rss", None), "process RSS")


def _metal_sample(mx_module: _MxMemory) -> tuple[int, int, int, int]:
    active = _nonnegative_int(mx_module.get_active_memory(), "Metal active memory")
    cache = _nonnegative_int(mx_module.get_cache_memory(), "Metal cache memory")
    peak = _nonnegative_int(mx_module.get_peak_memory(), "Metal active peak memory")
    residency = active + cache
    if residency > _MAX_COUNTER or peak < active:
        raise ValueError("Metal memory counters violate allocator algebra")
    return active, cache, residency, peak


def _sample_memory(context: FrontierTelemetryContext) -> _MemorySample:
    rss = _rss_bytes(context.process)
    lifetime = _rss_lifetime_highwater_bytes()
    active, cache, residency, peak = _metal_sample(context.mx_module)
    if lifetime < rss:
        # ru_maxrss and current RSS may be sampled at slightly different
        # instants. A smaller lifetime value cannot be an authoritative bound.
        raise ValueError("process RSS lifetime high-water is below current RSS")
    return _MemorySample(
        rss_bytes=rss,
        rss_lifetime_highwater_bytes=lifetime,
        metal_active_bytes=active,
        metal_cache_bytes=cache,
        metal_residency_bytes=residency,
        metal_active_peak_bytes=peak,
    )


def _sample_rss_loop(context: FrontierTelemetryContext) -> None:
    while not context._sampler_stop.wait(RSS_SAMPLE_INTERVAL_SECONDS):
        try:
            rss = _rss_bytes(context.process)
            lifetime = _rss_lifetime_highwater_bytes()
            with context._sample_lock:
                if context.rss_sample_count >= RSS_SAMPLE_LIMIT:
                    context.rss_sampler_error = "RSS sampler exceeded its fixed bound"
                    return
                context.rss_sample_count += 1
                context.rss_observed_peak_bytes = max(
                    context.rss_observed_peak_bytes,
                    rss,
                )
                context.rss_lifetime_highwater_after_bytes = max(
                    context.rss_lifetime_highwater_after_bytes,
                    lifetime,
                )
        except Exception as error:
            with context._sample_lock:
                context.rss_sampler_error = f"{type(error).__name__}: {error}"
            return


def _poison(context: FrontierTelemetryContext | None = None) -> None:
    global _active, _poisoned
    with _state_lock:
        _poisoned = True
        if context is not None and _active is context:
            _active = None


def claim(
    *,
    request_index: object,
    request_nonce_sha256: object,
    rank: int,
    world_size: int,
    mx_module: _MxMemory,
) -> FrontierTelemetryContext | None:
    """Claim the next request before its allocator reset.

    Any duplicate, replay, gap, overlap, or post-failure attempt poisons the
    process-local telemetry sequence.  Failed attempts are never retried.
    """

    global _active, _next_reset_generation, _poisoned
    if not enabled():
        reject_orphan_request_fields(request_index, request_nonce_sha256)
        for companion in (DIRECTORY_ENV, SESSION_SHA256_ENV):
            if companion in os.environ:
                raise ValueError(f"{companion} requires {ENABLE_ENV}=1")
        return None
    if (
        type(request_index) is not int
        or not 1 <= cast(int, request_index) <= EXPECTED_REQUESTS
    ):
        raise ValueError("frontier request index must be an integer in [1, 16]")
    index = cast(int, request_index)
    if not _valid_sha256(request_nonce_sha256):
        raise ValueError("frontier request nonce SHA-256 must be lowercase hex")
    nonce = cast(str, request_nonce_sha256)
    if type(rank) is not int or type(world_size) is not int:
        raise TypeError("frontier rank identity must be integer-valued")
    if world_size != 2 or rank not in {0, 1}:
        raise ValueError("frontier request telemetry v4 requires exact TP2 identity")
    session = os.environ.get(SESSION_SHA256_ENV)
    if not _valid_sha256(session):
        raise ValueError(f"{SESSION_SHA256_ENV} must be lowercase SHA-256")
    directory, device, inode = _directory_contract()
    try:
        import psutil

        process = cast(_ProcessMemory, psutil.Process(os.getpid()))
    except Exception as error:
        raise RuntimeError("process RSS API is unavailable") from error
    with _state_lock:
        expected = len(_attempted_indices) + 1
        if (
            _poisoned
            or _active is not None
            or index != expected
            or index in _attempted_indices
            or nonce in _used_nonces
            or len(_attempted_indices) >= EXPECTED_REQUESTS
        ):
            _poisoned = True
            raise RuntimeError(
                "frontier telemetry request overlap, replay, or sequence gap"
            )
        _require_exact_prior_inventory(
            directory,
            device=device,
            inode=inode,
            index=index,
            rank=rank,
        )
        _next_reset_generation += 1
        if _next_reset_generation != index:
            _poisoned = True
            raise RuntimeError("frontier telemetry reset generation diverged")
        context = FrontierTelemetryContext(
            request_index=index,
            request_nonce_sha256=nonce,
            session_sha256=cast(str, session),
            rank=rank,
            world_size=world_size,
            pid=os.getpid(),
            reset_generation=_next_reset_generation,
            directory=directory,
            directory_device=device,
            directory_inode=inode,
            mx_module=mx_module,
            process=process,
        )
        _attempted_indices.add(index)
        _used_nonces.add(nonce)
        _active = context
        return context


def capture_reset_baseline(context: FrontierTelemetryContext | None) -> None:
    """Capture the post-reset fence and arm the bounded RSS sampler."""

    if context is None:
        return
    try:
        with _state_lock:
            if (
                _poisoned
                or _active is not context
                or context.reset_captured
                or context.finalized
            ):
                raise RuntimeError("frontier telemetry reset fence state is invalid")
        sample = _sample_memory(context)
        baseline = {
            "session_sha256": context.session_sha256,
            "request_index": context.request_index,
            "request_nonce_sha256": context.request_nonce_sha256,
            "rank": context.rank,
            "world_size": context.world_size,
            "pid": context.pid,
            "reset_generation": context.reset_generation,
            "process_rss_baseline_bytes": sample.rss_bytes,
            "process_rss_lifetime_highwater_baseline_bytes": (
                sample.rss_lifetime_highwater_bytes
            ),
            "metal_active_baseline_bytes": sample.metal_active_bytes,
            "metal_cache_baseline_bytes": sample.metal_cache_bytes,
            "metal_residency_baseline_bytes": sample.metal_residency_bytes,
            "metal_active_peak_baseline_bytes": sample.metal_active_peak_bytes,
        }
        context.baseline = sample
        context.rss_observed_peak_bytes = sample.rss_bytes
        context.rss_lifetime_highwater_after_bytes = sample.rss_lifetime_highwater_bytes
        context.metal_active_peak_bytes = sample.metal_active_peak_bytes
        context.metal_residency_observed_peak_bytes = sample.metal_residency_bytes
        context.metal_observation_count = 1
        context.rss_sample_count = 1
        context.reset_fence_sha256 = _sha256(baseline)
        context.reset_captured = True
        sampler = threading.Thread(
            target=_sample_rss_loop,
            args=(context,),
            name=f"k3-frontier-rss-{context.request_index}",
            daemon=True,
        )
        context._sampler = sampler
        sampler.start()
    except BaseException:
        context._sampler_stop.set()
        _poison(context)
        raise


def observe_metal(context: FrontierTelemetryContext | None) -> None:
    """Bounded lifecycle observation for residency; never called per token."""

    if context is None:
        return
    try:
        with _state_lock:
            if _poisoned or _active is not context or not context.reset_captured:
                raise RuntimeError("frontier Metal observation state is invalid")
        active, _cache, residency, peak = _metal_sample(context.mx_module)
        with context._sample_lock:
            context.metal_active_peak_bytes = max(
                context.metal_active_peak_bytes,
                active,
                peak,
            )
            context.metal_residency_observed_peak_bytes = max(
                context.metal_residency_observed_peak_bytes,
                residency,
            )
            context.metal_observation_count += 1
    except BaseException:
        _poison(context)
        raise


def _stop_sampler(context: FrontierTelemetryContext) -> None:
    context._sampler_stop.set()
    sampler = context._sampler
    if sampler is not None:
        sampler.join(timeout=1.0)
        if sampler.is_alive():
            raise RuntimeError("frontier RSS sampler did not stop within one second")


def _validate_core_receipt(core: Mapping[str, object]) -> dict[str, int | bool]:
    if core.get("receipt_schema_version") != 2:
        raise ValueError("frontier telemetry requires the exact C1 receipt-v2 core")
    required = {
        "request_sequence",
        "request_token",
        "receipt_schema_version",
        *REPLAYSSM_FIELDS,
    }
    if not required <= set(core):
        raise ValueError("C1 receipt-v2 core is missing telemetry counters")
    numeric: dict[str, int | bool] = {}
    for name, value in core.items():
        if type(value) not in {bool, int}:
            raise TypeError("C1 receipt-v2 core must contain numeric scalars")
        if type(value) is int and not 0 <= cast(int, value) <= _MAX_COUNTER:
            raise ValueError("C1 receipt-v2 core counter is out of range")
        numeric[name] = cast(int | bool, value)
    before = cast(int, numeric["replayssm_telemetry_revision_before"])
    after = cast(int, numeric["replayssm_telemetry_revision_after"])
    delta = cast(int, numeric["replayssm_telemetry_revision_delta"])
    if after < before or after - before != delta:
        raise ValueError("C1 receipt-v2 revision algebra is invalid")
    return numeric


def _verify_directory(context: FrontierTelemetryContext) -> int:
    info = context.directory.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_dev != context.directory_device
        or info.st_ino != context.directory_inode
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RuntimeError("frontier telemetry directory identity drifted")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_fd = os.open(context.directory, flags)
    opened = os.fstat(directory_fd)
    if (opened.st_dev, opened.st_ino) != (
        context.directory_device,
        context.directory_inode,
    ):
        os.close(directory_fd)
        raise RuntimeError("frontier telemetry directory changed during open")
    return directory_fd


def _write_o_excl(
    context: FrontierTelemetryContext,
    name: str,
    payload: object,
    *,
    expected_inventory: Mapping[str, str],
) -> str:
    if "/" in name or name in {"", ".", ".."}:
        raise ValueError("frontier telemetry evidence name is invalid")
    data = _canonical_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    directory_fd = _verify_directory(context)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_fd: int | None = None
    check_fd: int | None = None
    try:
        # The intended name may have appeared after claim.  It is the only
        # optional entry here so O_EXCL itself remains the collision authority;
        # every unrelated late entry still fails the exact-inventory fence.
        _require_exact_inventory_fd(
            directory_fd,
            expected_inventory,
            optional_names=frozenset({name}),
        )
        file_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        offset = 0
        while offset < len(data):
            written = os.write(file_fd, data[offset:])
            if written <= 0:
                raise OSError("frontier telemetry evidence write made no progress")
            offset += written
        os.fsync(file_fd)
        info = os.fstat(file_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_size != len(data)
        ):
            raise RuntimeError("frontier telemetry evidence inode is invalid")
        os.close(file_fd)
        file_fd = None
        check_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            check_flags |= os.O_NOFOLLOW
        check_fd = os.open(name, check_flags, dir_fd=directory_fd)
        observed = bytearray()
        while True:
            chunk = os.read(check_fd, 65_536)
            if not chunk:
                break
            observed.extend(chunk)
        check = os.fstat(check_fd)
        if (
            bytes(observed) != data
            or hashlib.sha256(observed).hexdigest() != digest
            or check.st_nlink != 1
            or stat.S_IMODE(check.st_mode) != 0o600
        ):
            raise RuntimeError("frontier telemetry evidence verification failed")
        _require_exact_inventory_fd(
            directory_fd,
            {**expected_inventory, name: digest},
        )
        os.fsync(directory_fd)
    finally:
        if check_fd is not None:
            os.close(check_fd)
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)
    return digest


def finalize(
    context: FrontierTelemetryContext | None,
    core_receipt: Mapping[str, object],
) -> FrontierTelemetryEvidence | None:
    """Close one request, publish immutable rank-local evidence, and return hashes."""

    global _active
    if context is None:
        return None
    try:
        with _state_lock:
            if (
                _poisoned
                or _active is not context
                or not context.reset_captured
                or context.finalized
                or context.baseline is None
                or context.reset_fence_sha256 is None
            ):
                raise RuntimeError("frontier telemetry finalization state is invalid")
        _stop_sampler(context)
        final = _sample_memory(context)
        core = _validate_core_receipt(core_receipt)
        receipt_sequence = cast(int, core["request_sequence"])
        receipt_token = cast(int, core["request_token"])
        with _state_lock:
            prior_sequences = [
                cast(int, row["receipt_request_sequence"])
                for row in _published.values()
            ]
            if (
                prior_sequences and receipt_sequence <= max(prior_sequences)
            ) or receipt_token in _used_receipt_tokens:
                raise RuntimeError(
                    "frontier receipt sequence/token replayed or decreased"
                )
        with context._sample_lock:
            if context.rss_sampler_error is not None:
                raise RuntimeError(
                    f"frontier RSS sampler failed: {context.rss_sampler_error}"
                )
            context.rss_sample_count += 1
            context.rss_observed_peak_bytes = max(
                context.rss_observed_peak_bytes,
                final.rss_bytes,
            )
            context.rss_lifetime_highwater_after_bytes = max(
                context.rss_lifetime_highwater_after_bytes,
                final.rss_lifetime_highwater_bytes,
            )
            context.metal_active_peak_bytes = max(
                context.metal_active_peak_bytes,
                final.metal_active_bytes,
                final.metal_active_peak_bytes,
            )
            context.metal_residency_observed_peak_bytes = max(
                context.metal_residency_observed_peak_bytes,
                final.metal_residency_bytes,
            )
            context.metal_observation_count += 1
            baseline = context.baseline
            rss_peak = context.rss_observed_peak_bytes
            rss_lifetime_after = context.rss_lifetime_highwater_after_bytes
            metal_active_peak = context.metal_active_peak_bytes
            metal_residency_peak = context.metal_residency_observed_peak_bytes
            rss_samples = context.rss_sample_count
            metal_samples = context.metal_observation_count
        if (
            rss_peak < baseline.rss_bytes
            or rss_lifetime_after < baseline.rss_lifetime_highwater_bytes
            or metal_active_peak < baseline.metal_active_bytes
            or metal_residency_peak < baseline.metal_residency_bytes
        ):
            raise ValueError("frontier request memory peak decreased below baseline")
        core_payload = {"schema": CORE_SCHEMA, **core}
        core_sha = _sha256(core_payload)
        evidence_without_hash: dict[str, object] = {
            "schema": SCHEMA,
            "schema_version": 4,
            "session_sha256": context.session_sha256,
            "request_index": context.request_index,
            "request_nonce_sha256": context.request_nonce_sha256,
            "rank": context.rank,
            "world_size": context.world_size,
            "pid": context.pid,
            "reset_generation": context.reset_generation,
            "reset_fence_sha256": context.reset_fence_sha256,
            "receipt_v2_core_sha256": core_sha,
            "receipt_request_sequence": core["request_sequence"],
            "receipt_request_token": core["request_token"],
            **{name: core[name] for name in REPLAYSSM_FIELDS},
            "process_rss_baseline_bytes": baseline.rss_bytes,
            "process_rss_final_bytes": final.rss_bytes,
            "process_rss_final_delta_bytes": final.rss_bytes - baseline.rss_bytes,
            "process_rss_observed_peak_bytes": rss_peak,
            "process_rss_observed_peak_delta_bytes": rss_peak - baseline.rss_bytes,
            "process_rss_lifetime_highwater_baseline_bytes": (
                baseline.rss_lifetime_highwater_bytes
            ),
            "process_rss_lifetime_highwater_after_bytes": rss_lifetime_after,
            "process_rss_lifetime_highwater_delta_bytes": (
                rss_lifetime_after - baseline.rss_lifetime_highwater_bytes
            ),
            "rss_sample_interval_ns": int(RSS_SAMPLE_INTERVAL_SECONDS * 1e9),
            "rss_sample_count": rss_samples,
            "metal_active_baseline_bytes": baseline.metal_active_bytes,
            "metal_active_final_bytes": final.metal_active_bytes,
            "metal_active_final_delta_bytes": (
                final.metal_active_bytes - baseline.metal_active_bytes
            ),
            "metal_active_peak_bytes": metal_active_peak,
            "metal_active_peak_delta_bytes": (
                metal_active_peak - baseline.metal_active_bytes
            ),
            "metal_cache_baseline_bytes": baseline.metal_cache_bytes,
            "metal_cache_final_bytes": final.metal_cache_bytes,
            "metal_cache_final_delta_bytes": (
                final.metal_cache_bytes - baseline.metal_cache_bytes
            ),
            "metal_residency_baseline_bytes": baseline.metal_residency_bytes,
            "metal_residency_final_bytes": final.metal_residency_bytes,
            "metal_residency_final_delta_bytes": (
                final.metal_residency_bytes - baseline.metal_residency_bytes
            ),
            "metal_residency_observed_peak_bytes": metal_residency_peak,
            "metal_residency_observed_peak_delta_bytes": (
                metal_residency_peak - baseline.metal_residency_bytes
            ),
            "metal_observation_count": metal_samples,
            "rss_peak_semantics": "bounded-10ms-request-sampler",
            "rss_lifetime_highwater_semantics": "process-lifetime-authoritative",
            "metal_active_peak_semantics": "mlx-reset-scoped-authoritative",
            "metal_residency_peak_semantics": "bounded-lifecycle-observed",
        }
        evidence = {
            **evidence_without_hash,
            "evidence_sha256": _sha256(evidence_without_hash),
        }
        if set(evidence) != REQUEST_EVIDENCE_KEYS:
            raise RuntimeError("frontier request evidence schema drifted")
        with _state_lock:
            expected_inventory = _expected_prior_inventory(
                index=context.request_index,
                rank=context.rank,
            )
        request_file_sha = _write_o_excl(
            context,
            f"request-{context.request_index:02d}-rank{context.rank}.json",
            evidence,
            expected_inventory=expected_inventory,
        )
        published_row = {
            "request_index": context.request_index,
            "request_nonce_sha256": context.request_nonce_sha256,
            "request_file_sha256": request_file_sha,
            "receipt_request_sequence": core["request_sequence"],
            "receipt_request_token": core["request_token"],
        }
        process_complete_sha: str | None = None
        with _state_lock:
            if _active is not context or context.request_index in _published:
                raise RuntimeError("frontier telemetry publication ownership drifted")
            _published[context.request_index] = published_row
            _used_receipt_tokens.add(receipt_token)
            context.finalized = True
            _active = None
            complete_rows = (
                [dict(_published[index]) for index in range(1, 17)]
                if len(_published) == EXPECTED_REQUESTS
                else None
            )
            complete_inventory = (
                _expected_prior_inventory(
                    index=EXPECTED_REQUESTS + 1,
                    rank=context.rank,
                )
                if complete_rows is not None
                else None
            )
        if complete_rows is not None:
            assert complete_inventory is not None
            process_without_hash = {
                "schema": PROCESS_SCHEMA,
                "schema_version": 4,
                "session_sha256": context.session_sha256,
                "rank": context.rank,
                "world_size": context.world_size,
                "pid": context.pid,
                "request_count": EXPECTED_REQUESTS,
                "requests": complete_rows,
                "request_chain_sha256": _sha256(complete_rows),
            }
            process_payload = {
                **process_without_hash,
                "evidence_sha256": _sha256(process_without_hash),
            }
            if set(process_payload) != PROCESS_EVIDENCE_KEYS or any(
                set(row) != PROCESS_REQUEST_ROW_KEYS for row in complete_rows
            ):
                raise RuntimeError("frontier process evidence schema drifted")
            process_complete_sha = _write_o_excl(
                context,
                f"process-complete-rank{context.rank}.json",
                process_payload,
                expected_inventory=complete_inventory,
            )
        return FrontierTelemetryEvidence(
            request_index=context.request_index,
            request_nonce_sha256=context.request_nonce_sha256,
            reset_generation=context.reset_generation,
            reset_fence_sha256=context.reset_fence_sha256,
            receipt_v2_core_sha256=core_sha,
            request_file_sha256=request_file_sha,
            process_complete_file_sha256=process_complete_sha,
        )
    except BaseException:
        context._sampler_stop.set()
        _poison(context)
        raise


def abort(context: FrontierTelemetryContext | None) -> None:
    """Poison an attempted request unless immutable evidence was finalized."""

    if context is None or context.finalized:
        return
    context._sampler_stop.set()
    sampler = context._sampler
    if sampler is not None:
        sampler.join(timeout=1.0)
    _poison(context)


def poison_finalized(context: FrontierTelemetryContext | None) -> None:
    """Fail the process if a post-file collective or marker join fails."""

    if context is not None:
        _poison()


def _reset_state_for_tests() -> None:
    """Hermetic test hook; never called by production code."""

    global _active, _poisoned, _next_reset_generation
    with _state_lock:
        if _active is not None:
            _active._sampler_stop.set()
        _active = None
        _poisoned = False
        _attempted_indices.clear()
        _published.clear()
        _used_nonces.clear()
        _used_receipt_tokens.clear()
        _next_reset_generation = 0
