import contextlib
import ctypes
import functools
import hashlib
import importlib
import json
import math
import os
import platform
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Generator, Literal, Protocol, TypedDict, cast, get_args

import mlx.core as mx
from mlx_lm.generate import (
    GenerationResponse as MlxGenerationResponse,
)
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    K3W3CompositionReceipt,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    discard_unsent_prefill_sends_after_agreed_cancel,
    flush_prefill_sends,
    get_active_relay_context,
    relay_sampled_tokens,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
    set_pipeline_token_relay,
)
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    copy_snapshot_entry,
    encode_prompt,
    has_non_kv_caches,
    is_non_trimmable_cache_entry,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_CACHE_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.generator.kimi_k3_dspark import (
    DSparkConfidenceCaptureConfig,
    DSparkDecodedToken,
    DSparkDistributedStateError,
    DSparkRoundTelemetry,
    KimiK3DSparkProposerSelection,
    KimiK3DSparkRequestRuntime,
    LoadedKimiK3DSpark,
    MlxRankAgreement,
    dspark_confidence_capture_request,
    dspark_context_capacity_hint,
    dspark_decode_tokens,
    select_loaded_mlx_dspark,
    validate_dspark_greedy_sampling,
)
from exo.worker.engines.mlx.generator.kimi_k3_width4_receipt import (
    Width4RequestReceiptClaim,
    Width4RequestReceiptContext,
    begin_width4_request_receipt,
    capture_width4_dispatch_receipt,
    format_width4_dispatch_receipt,
    mark_width4_receipt_rank_agreed,
    reset_width4_dispatch_counters_after_warmup,
    validate_width4_request_contract,
    width4_receipt_agreement_contract,
    width4_receipt_log_enabled,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_barrier,
    mx_ranks_agree_on_value,
    rank_agreed_local_stage,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    get_inner_model,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

REMOTE_PREFILL_MIN_TOKENS = 1000

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5


@dataclass(frozen=True)
class _PromptLookupConfig:
    num_tokens: int
    max_ngram_size: int
    round_telemetry: bool


class _DSparkDetokenizer(Protocol):
    last_segment: str

    def reset(self) -> None: ...

    def add_token(self, token: int) -> None: ...

    def finalize(self) -> None: ...


@dataclass(frozen=True)
class _DSparkRequestSetup:
    """Local DSpark request state agreed before any target TP collective."""

    is_pipeline: bool
    prompt_lookup_configuration: _PromptLookupConfig | None
    all_prompt_tokens: mx.array
    caches: KVCacheType
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]]
    sampler: Callable[[mx.array], mx.array]
    stop_sequences: tuple[str, ...]
    max_stop_len: int
    max_tokens: int
    is_bench: bool
    eos_token_ids: tuple[int, ...]
    anchor_token: int
    detokenizer: _DSparkDetokenizer
    empty_logprobs: mx.array
    prefill_step_size: int
    force_ordinary: bool
    packed_agreements: bool
    runtime: KimiK3DSparkRequestRuntime
    proposer_selection: KimiK3DSparkProposerSelection | None
    fingerprint: tuple[int, ...]
    verify_width: int
    receipt_session_id: str | None


@dataclass
class _PromptLookupTelemetry:
    rounds: int = 0
    full_width_rounds: int = 0
    target_width1_rounds: int = 0
    prefill_width1_chunks: int = 0
    prefill_width3_chunks: int = 0
    prefill_noncontract_chunks: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    committed_tokens: int = 0
    visible_accepted_tokens: int = 0
    fallback_rounds: int = 0
    error_rounds: int = 0
    composition_receipt: "_PackedFrontReceiptRequest | None" = None

    def observe(self, stats: "_SpeculativeRoundStatsLike") -> None:
        self.rounds += 1
        self.drafted_tokens += int(stats.drafted_tokens)
        self.accepted_tokens += int(stats.accepted_tokens)
        self.committed_tokens += int(stats.committed_tokens)

    def observe_dspark_round(
        self,
        stats: DSparkRoundTelemetry,
        *,
        target_cache_tokens: int | None = None,
    ) -> None:
        """Account committed speculative work at the target transaction boundary.

        A textual stop can end response iteration partway through an already
        committed multi-token round.  These counters intentionally describe
        internal speculative decisions, while public completion_tokens remains
        the number of visible response tokens.
        """

        self.rounds += 1
        self.full_width_rounds += int(stats.proposed == 2)
        self.target_width1_rounds += int(stats.proposed == 0)
        self.drafted_tokens += stats.proposed
        self.accepted_tokens += stats.accepted
        self.committed_tokens += stats.emitted
        self.fallback_rounds += int(stats.fallback)
        self.error_rounds += int(stats.error is not None)
        if self.composition_receipt is not None:
            self.composition_receipt.observe_round(
                stats,
                target_cache_tokens=target_cache_tokens,
            )

    def observe_visible_token(self, *, from_draft: bool) -> None:
        """Track accepted predictions that remain in the public completion."""

        self.visible_accepted_tokens += int(from_draft)


def _exact_prefill_chunk_geometry(
    token_count: int,
    step_size: int,
) -> tuple[int, int, int]:
    """Count exact width-one, width-three, and other DSpark prefill chunks."""

    if type(token_count) is not int or token_count < 0:
        raise ValueError("packed-front prefill token count is invalid")
    if type(step_size) is not int or step_size < 4:
        raise ValueError("packed-front prefill step size is invalid")
    full_chunks, tail = divmod(token_count, step_size)
    width1_chunks = int(tail == 1)
    width3_chunks = int(tail == 3)
    noncontract_chunks = full_chunks + int(tail not in {0, 1, 3})
    return width1_chunks, width3_chunks, noncontract_chunks


class _SpeculativeRoundStatsLike(Protocol):
    round_index: int
    source: str
    drafted_tokens: int
    accepted_tokens: int
    committed_tokens: int
    target_cache_tokens: int
    cancelled: bool


class _PromptLookupStreamKwargs(TypedDict):
    prompt_lookup_num_tokens: int
    prompt_lookup_max_ngram_size: int
    prompt_lookup_history: mx.array
    speculative_round_callback: Callable[[_SpeculativeRoundStatsLike], None] | None


class _GreedyVocabParallelStreamKwargs(TypedDict, total=False):
    greedy_vocab_parallel_no_logprobs: bool


class _PromptLookupStreamGenerate(Protocol):
    def __call__(
        self,
        *,
        prompt_lookup_num_tokens: int,
        prompt_lookup_max_ngram_size: int,
        prompt_lookup_history: mx.array,
        speculative_round_callback: (
            Callable[[_SpeculativeRoundStatsLike], None] | None
        ),
        **kwargs: object,
    ) -> Generator[MlxGenerationResponse, None, None]: ...


def _strict_env_int(name: str, value: str, *, minimum: int, maximum: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return parsed


def _strict_env_flag(name: str, value: str) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


_PACKED_FRONT_DIAGNOSTIC_ENV = "EXO_MLX_KIMI_K3_W3_COMPOSITION_RECEIPT"
_PACKED_FRONT_EXPECTED_LAYERS_ENV = (
    "EXO_MLX_KIMI_K3_W3_COMPOSITION_EXPECTED_SPARSE_LAYERS"
)
_KDA_EXPECTED_LAYERS_ENV = "EXO_MLX_KIMI_K3_W3_COMPOSITION_EXPECTED_KDA_LAYERS"
_DEFERRED_EXPECTED_ROOTS_ENV = "EXO_MLX_KIMI_K3_W3_COMPOSITION_EXPECTED_DEFERRED_ROOTS"
_EXPECTED_LIBMLX_SHA256_ENV = "EXO_MLX_KIMI_K3_W3_COMPOSITION_LIBMLX_SHA256"
_SEALED_LIBMLX_SHA256 = (
    "91f742bfa20f3559c2fb85e6b3b5aad7a5b6264e1158d2cfc1a089a35d1b18cd"
)
_LAUNCH_CONTRACT_SHA256_ENV = "EXO_MLX_KIMI_K3_W3_COMPOSITION_LAUNCH_CONTRACT_SHA256"
_MLX_PACKED_FRONT_RECEIPT_ENV = "MLX_LM_KIMI_K3_W3_COMPOSITION_RECEIPT"
_MLX_AUTHORITATIVE_PACKED_FRONT_ENV = "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT"
_MLX_AUTHORITATIVE_PACKED_FRONT_WIDTH3_ENV = (
    "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3"
)
_MLX_DUPLICATING_PACKED_FRONT_ENV = "MLX_LM_KIMI_K3_PACKED_MOE_FRONT"
_MLX_PACKED_FRONT_WIDTH8_ENV = "MLX_LM_KIMI_K3_PACKED_MOE_FRONT_WIDTH8"
_MLX_MULTIBANK_PACKED_FRONT_ENV = "MLX_LM_KIMI_K3_MULTIBANK_MOE_FRONT"
_MLX_COMPILED_DECODE_ENV = "MLX_LM_KIMI_K3_COMPILED_DECODE"
_MLX_KDA_PREWORK_ENV = "MLX_LM_KIMI_K3_W3_PREWORK_HISTORY"
_MLX_PROJECTED_KV_CACHE_MAX_TOKENS_ENV = "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE_MAX_TOKENS"
_EXO_DEFERRED_WIDTH3_ENV = "EXO_MLX_KIMI_K3_DEFERRED_ASYNC_WIDTH3"
_MLX_DEFERRED_WIDTH3_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3"
_EXO_TAIL_OVERLAP_ENV = "EXO_MLX_KIMI_K3_DSPARK_TAIL_OVERLAP"
_MLX_NATIVE_AFFINE8_Q3_TRIPLET_ENV = "MLX_METAL_K3_AFFINE8_Q3_TRIPLET"
_MLX_NATIVE_AFFINE8_Q3_RECEIPT_ENV = "MLX_METAL_K3_AFFINE8_Q3_DISPATCH_RECEIPT"
_WIDTH4_RECEIPT_ENV = "EXO_MLX_KIMI_K3_WIDTH4_DISPATCH_RECEIPT_LOG"
_PACKED_FRONT_RECEIPT_SCHEMA = "kimi-k3-w3-composition-receipt/v1"
_PACKED_FRONT_EXPECTED_LAYERS = 92
_KDA_EXPECTED_LAYERS = 69
_DEFERRED_EXPECTED_ROOTS = 12
_PROJECTED_KV_CACHE_MAX_TOKENS = 32768
_PACKED_FRONT_COUNTER_LIMIT = 1_000_000_000
_PACKED_FRONT_SEQUENCE_LIMIT = 0x7FFFFFFFFFFFFFFF
_PACKED_FRONT_PHASE_ENGINE_STARTUP = 1
_PACKED_FRONT_PHASE_API = 2
_PACKED_FRONT_PHASE_CODES = {
    "engine_startup": _PACKED_FRONT_PHASE_ENGINE_STARTUP,
    "api": _PACKED_FRONT_PHASE_API,
}
_COMPOSITION_RECEIPT_LOCK = Lock()
_COMPOSITION_RECEIPT_ATTEMPTED_PHASES: set[int] = set()
_COMPOSITION_RECEIPT_PUBLISHED_PHASES: set[int] = set()
_COMPOSITION_RECEIPT_COMPLETED_PHASES: set[int] = set()
_COMPOSITION_RECEIPT_PHASE_IDENTITIES: dict[int, tuple[int, ...]] = {}
_COMPOSITION_RECEIPT_ACTIVE_PHASE: int | None = None
_PACKED_FRONT_MLX_KEYS = frozenset(
    {
        "schema",
        "request_sequence",
        "request_token",
        "expected_sparse_layers",
        "expected_kda_layers",
        "finalized",
        "aborted",
        "poisoned",
        "packed_authoritative_enabled",
        "packed_width3_enabled",
        "kda_prework_enabled",
        "replayssm_speculative_enabled",
        "projected_kv_cache_enabled",
        "projected_kv_cache_max_tokens",
        "async_decode_boundaries",
        "async_decode_state",
        "async_decode_width3_enabled",
        "native_q3_triplet_enabled",
        "native_q3_dispatch_receipt_enabled",
        "helper_calls",
        "eligible_width1_calls",
        "eligible_width3_calls",
        "packed_width1_hits",
        "packed_width3_hits",
        "packed_hits",
        "packed_width1_output_tensors",
        "packed_width3_output_tensors",
        "packed_output_tensors",
        "packed_width1_installs",
        "packed_width3_installs",
        "lazy_installs",
        "gate_disabled_calls",
        "noncontract_calls",
        "width1_unsupported_calls",
        "width3_unsupported_calls",
        "unsupported_calls",
        "width1_dispatch_fallback_calls",
        "width3_dispatch_fallback_calls",
        "packed_dispatch_fallback_calls",
        "invalidations",
        "stale_resets",
        "pack_count_before",
        "pack_count_after",
        "kda_helper_calls",
        "kda_gate_disabled_calls",
        "kda_noncontract_calls",
        "kda_admitted_calls",
        "kda_success_calls",
        "kda_fallback_calls",
        "kda_pending_calls",
    }
)
_PACKED_FRONT_COUNTER_FIELDS = (
    "helper_calls",
    "eligible_width1_calls",
    "eligible_width3_calls",
    "packed_width1_hits",
    "packed_width3_hits",
    "packed_hits",
    "packed_width1_output_tensors",
    "packed_width3_output_tensors",
    "packed_output_tensors",
    "packed_width1_installs",
    "packed_width3_installs",
    "lazy_installs",
    "gate_disabled_calls",
    "noncontract_calls",
    "width1_unsupported_calls",
    "width3_unsupported_calls",
    "unsupported_calls",
    "width1_dispatch_fallback_calls",
    "width3_dispatch_fallback_calls",
    "packed_dispatch_fallback_calls",
    "invalidations",
    "stale_resets",
    "pack_count_before",
    "pack_count_after",
    "kda_helper_calls",
    "kda_gate_disabled_calls",
    "kda_noncontract_calls",
    "kda_admitted_calls",
    "kda_success_calls",
    "kda_fallback_calls",
    "kda_pending_calls",
)


@dataclass(frozen=True)
class _PackedFrontReceiptAPI:
    begin: Callable[..., object]
    finish: Callable[..., object]
    abort: Callable[[int, int], None]
    source_digest: tuple[int, int, int, int]


class _NativeQ3Getter(Protocol):
    argtypes: list[object]
    restype: object

    def __call__(self) -> int: ...


class _NativeQ3Resetter(Protocol):
    argtypes: list[object]
    restype: object

    def __call__(self) -> None: ...


@dataclass(frozen=True)
class _NativeQ3ReceiptAPI:
    library: object = field(repr=False, compare=False)
    total: _NativeQ3Getter = field(repr=False, compare=False)
    n4480: _NativeQ3Getter = field(repr=False, compare=False)
    n6144: _NativeQ3Getter = field(repr=False, compare=False)
    n10624: _NativeQ3Getter = field(repr=False, compare=False)
    reset: _NativeQ3Resetter = field(repr=False, compare=False)
    lib_path: Path = field(repr=False, compare=False)
    lib_fd: int = field(repr=False, compare=False)
    lib_identity: tuple[int, ...]
    lib_digest: tuple[int, int, int, int]


def _digest_words(digest: bytes) -> tuple[int, int, int, int]:
    if len(digest) != 32:
        raise ValueError("W3 composition digest must be SHA-256")
    return cast(
        tuple[int, int, int, int],
        tuple(
            int.from_bytes(digest[offset : offset + 8], "big")
            & _PACKED_FRONT_SEQUENCE_LIMIT
            for offset in range(0, 32, 8)
        ),
    )


def _regular_file_sha256(path: Path, *, identity: str) -> bytes:
    """Hash one authenticated regular file without following a replacement link."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise RuntimeError(f"{identity} requires O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if type(cloexec) is int:
        flags |= cloexec
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{identity} must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"{identity} changed while being hashed")
        named = os.lstat(path)
        if (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError(f"{identity} path changed while being hashed")
        return digest.digest()
    finally:
        os.close(fd)


def _native_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError("W3 composition libmlx is not a regular file")
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_native_fd(fd: int) -> tuple[tuple[int, ...], bytes]:
    """Hash one pinned dylib descriptor and reject concurrent mutation."""

    before = _native_stat_identity(os.fstat(fd))
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
    after = _native_stat_identity(os.fstat(fd))
    if after != before:
        raise RuntimeError("W3 composition libmlx changed while being hashed")
    return after, digest.digest()


def _open_native_snapshot(path: Path) -> tuple[int, tuple[int, ...], bytes]:
    """Open the named dylib without following its final path component."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise RuntimeError("W3 composition libmlx requires O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if type(cloexec) is int:
        flags |= cloexec
    fd = os.open(path, flags)
    try:
        identity, digest = _hash_native_fd(fd)
        named_identity = _native_stat_identity(os.lstat(path))
        if named_identity != identity:
            raise RuntimeError("W3 composition libmlx path changed while opening")
        return fd, identity, digest
    except BaseException:
        os.close(fd)
        raise


def _assert_native_library_identity(api: _NativeQ3ReceiptAPI) -> None:
    """Recheck both the pinned inode and its canonical name at receipt finish."""

    pinned_identity, pinned_digest = _hash_native_fd(api.lib_fd)
    if (
        pinned_identity != api.lib_identity
        or _digest_words(pinned_digest) != api.lib_digest
    ):
        raise ValueError("loaded W3 composition libmlx identity changed")
    named_fd, named_identity, named_digest = _open_native_snapshot(api.lib_path)
    try:
        if (
            named_identity != api.lib_identity
            or _digest_words(named_digest) != api.lib_digest
        ):
            raise ValueError(
                "W3 composition libmlx path no longer names the loaded file"
            )
    finally:
        os.close(named_fd)


def _runtime_source_digest() -> tuple[int, int, int, int]:
    digest = hashlib.sha256(b"exo-kimi-k3-w3-composition-source/v1\0")
    for module_name in (
        "mlx_lm.models.kimi_k3",
        "mlx_lm.models.kimi_k3_packed_moe_front",
        "mlx_lm.models.kimi_k3_w3_prework",
    ):
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if type(raw_path) is not str:
            raise RuntimeError(f"{module_name} has no source identity")
        path = Path(raw_path)
        digest.update(module_name.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            _regular_file_sha256(path, identity=f"{module_name} source identity")
        )
    return _digest_words(digest.digest())


def _exo_source_digest() -> tuple[int, int, int, int]:
    """Bind the EXO validator, causal observer, schema, and loader bridge."""

    generate_path = Path(__file__)
    repository = generate_path.resolve(strict=True).parents[6]
    api_module = importlib.import_module("exo.api.types.api")
    api_init_module = importlib.import_module("exo.api.types")
    rank_local_module = importlib.import_module(
        "exo.worker.engines.mlx.rank_local_checkpoint"
    )

    def module_path(module: object, name: str) -> Path:
        raw_path = getattr(module, "__file__", None)
        if type(raw_path) is not str:
            raise RuntimeError(f"{name} has no filesystem identity")
        return Path(raw_path)

    configured_loader = os.environ.get("EXO_MLX_RANK_LOCAL_LOADER")
    loader_path = (
        Path(configured_loader)
        if configured_loader is not None
        else repository / "scripts" / "kimi_k3_tp2" / "rank_local_loader.py"
    )

    module_paths = {
        "exo.generate": generate_path,
        "exo.kimi_k3_dspark": generate_path.with_name("kimi_k3_dspark.py"),
        "exo.builder": generate_path.parent.parent / "builder.py",
        "exo.api.types.api": module_path(api_module, "exo.api.types.api"),
        "exo.api.types.init": module_path(api_init_module, "exo.api.types"),
        "exo.rank_local_checkpoint": module_path(
            rank_local_module,
            "exo.worker.engines.mlx.rank_local_checkpoint",
        ),
        "scripts.rank_local_loader": loader_path,
    }
    digest = hashlib.sha256(b"exo-kimi-k3-w3-exo-source/v1\0")
    for name, path in module_paths.items():
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_regular_file_sha256(path, identity=f"{name} source identity"))
    return _digest_words(digest.digest())


def _load_native_q3_receipt_api() -> _NativeQ3ReceiptAPI:
    core_file = getattr(mx, "__file__", None)
    if type(core_file) is not str:
        raise RuntimeError("mlx.core has no filesystem identity")
    core_path = Path(core_file).resolve(strict=True)
    libmlx_path = core_path.parent / "lib" / "libmlx.dylib"
    expected_sha256 = os.environ.get(_EXPECTED_LIBMLX_SHA256_ENV)
    if expected_sha256 != _SEALED_LIBMLX_SHA256:
        raise RuntimeError(
            f"{_EXPECTED_LIBMLX_SHA256_ENV} must equal the sealed diagnostic "
            "libmlx SHA-256"
        )
    lib_fd, lib_identity, file_digest = _open_native_snapshot(libmlx_path)
    try:
        canonical_libmlx_path = libmlx_path.resolve(strict=True)
        if canonical_libmlx_path != libmlx_path:
            raise RuntimeError("W3 composition libmlx path is not canonical")
        actual_sha256 = file_digest.hex()
        if actual_sha256 != expected_sha256:
            raise RuntimeError("authenticated W3 composition libmlx SHA-256 differs")
        rtld_noload = getattr(os, "RTLD_NOLOAD", None)
        rtld_now = getattr(os, "RTLD_NOW", None)
        rtld_local = getattr(os, "RTLD_LOCAL", None)
        if any(type(value) is not int for value in (rtld_noload, rtld_now, rtld_local)):
            raise RuntimeError("W3 composition native receipt requires RTLD_NOLOAD")
        # mlx.core has already loaded this image. RTLD_NOLOAD ensures a path
        # replacement can never instantiate a different counter-bearing dylib.
        library = ctypes.CDLL(
            str(canonical_libmlx_path),
            mode=cast(int, rtld_noload) | cast(int, rtld_now) | cast(int, rtld_local),
        )
        pinned_after, digest_after = _hash_native_fd(lib_fd)
        named_fd, named_after, named_digest_after = _open_native_snapshot(
            canonical_libmlx_path
        )
        try:
            if (
                pinned_after != lib_identity
                or named_after != lib_identity
                or digest_after != file_digest
                or named_digest_after != file_digest
            ):
                raise RuntimeError(
                    "W3 composition libmlx changed across native symbol binding"
                )
        finally:
            os.close(named_fd)
    except BaseException:
        os.close(lib_fd)
        raise
    try:
        symbols = {
            "total": "mlx_k3_affine8_q3_triplet_dispatch_count",
            "n4480": "mlx_k3_affine8_q3_triplet_dispatch_count_n4480",
            "n6144": "mlx_k3_affine8_q3_triplet_dispatch_count_n6144",
            "n10624": "mlx_k3_affine8_q3_triplet_dispatch_count_n10624",
            "reset": "mlx_k3_affine8_q3_triplet_reset_dispatch_counts",
        }
        total = cast(_NativeQ3Getter, getattr(library, symbols["total"]))
        n4480 = cast(_NativeQ3Getter, getattr(library, symbols["n4480"]))
        n6144 = cast(_NativeQ3Getter, getattr(library, symbols["n6144"]))
        n10624 = cast(_NativeQ3Getter, getattr(library, symbols["n10624"]))
        reset = cast(_NativeQ3Resetter, getattr(library, symbols["reset"]))
        for getter in (total, n4480, n6144, n10624):
            getter.argtypes = []
            getter.restype = ctypes.c_uint64
        reset.argtypes = []
        reset.restype = None
    except BaseException as error:
        os.close(lib_fd)
        if isinstance(error, AttributeError):
            raise RuntimeError(
                "authenticated libmlx lacks dedicated affine8 Q3 receipt symbols"
            ) from error
        raise
    return _NativeQ3ReceiptAPI(
        library=library,
        total=total,
        n4480=n4480,
        n6144=n6144,
        n10624=n10624,
        reset=reset,
        lib_path=canonical_libmlx_path,
        lib_fd=lib_fd,
        lib_identity=lib_identity,
        lib_digest=_digest_words(file_digest),
    )


def _load_packed_front_receipt_api() -> _PackedFrontReceiptAPI:
    module = importlib.import_module("mlx_lm.models.kimi_k3_packed_moe_front")
    if (
        getattr(module, "K3_W3_COMPOSITION_RECEIPT_SCHEMA", None)
        != _PACKED_FRONT_RECEIPT_SCHEMA
    ):
        raise RuntimeError("MLX-LM W3 composition receipt schema is unavailable")
    begin = getattr(module, "begin_k3_w3_composition_receipt", None)
    finish = getattr(module, "finish_k3_w3_composition_receipt", None)
    abort = getattr(module, "abort_k3_w3_composition_receipt", None)
    if not callable(begin) or not callable(finish) or not callable(abort):
        raise RuntimeError("MLX-LM W3 composition receipt APIs are unavailable")
    return _PackedFrontReceiptAPI(
        begin=begin,
        finish=finish,
        abort=cast(Callable[[int, int], None], abort),
        source_digest=_runtime_source_digest(),
    )


def _require_exact_selector(name: str, expected: str) -> None:
    if os.environ.get(name) != expected:
        raise ValueError(f"{name} must be exactly {expected} in diagnostic mode")


def _reject_orphan_composition_receipt_environment() -> None:
    for name in (
        _PACKED_FRONT_EXPECTED_LAYERS_ENV,
        _KDA_EXPECTED_LAYERS_ENV,
        _DEFERRED_EXPECTED_ROOTS_ENV,
        _EXPECTED_LIBMLX_SHA256_ENV,
        _LAUNCH_CONTRACT_SHA256_ENV,
    ):
        if name in os.environ:
            raise ValueError(f"{name} requires {_PACKED_FRONT_DIAGNOSTIC_ENV}=1")
    for name in (
        _MLX_PACKED_FRONT_RECEIPT_ENV,
        _MLX_NATIVE_AFFINE8_Q3_RECEIPT_ENV,
    ):
        if os.environ.get(name, "0") != "0":
            raise ValueError(f"{name} requires {_PACKED_FRONT_DIAGNOSTIC_ENV}=1")


@dataclass(frozen=True)
class _CompositionSelectors:
    packed: bool
    kda: bool
    deferred: bool
    arm_code: int
    canonical: bool
    digest: tuple[int, int, int, int]
    launch_digest: tuple[int, int, int, int]


def _launch_contract_digest() -> tuple[int, int, int, int]:
    """Authenticate the complete experiment environment without publishing it."""

    common_names = {
        "EXO_ADVERTISED_MODEL_IDS",
        "EXO_NO_BATCH",
        "EXO_OFFLINE",
        "MLX_METAL_FAST_SYNCH",
        "EXO_MLX_JACCL_FORCE_MESH",
        "EXO_MLX_K3_REQUANT_ATTENTION_QKVG_MXFP4",
        "EXO_MLX_K3_REQUANT_ROUTED_LATENT_MXFP4",
        "EXO_MLX_K3_VOCAB_PARALLEL_GREEDY",
        "EXO_MLX_K3_VOCAB_PARALLEL_HEAD",
        "EXO_MLX_KIMI_K3_DEFERRED_ASYNC_WIDTH3",
        "EXO_MLX_KIMI_K3_DSPARK_AUX_ONLY_PREFILL",
        "EXO_MLX_KIMI_K3_DSPARK_DUAL_PROPOSER",
        "EXO_MLX_KIMI_K3_DSPARK_FORCE_ORDINARY",
        "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_AFTER_CONTEXT",
        "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_W3_GATE",
        "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_W3_GATE_POLICY",
        "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_W3_GATE_POLICY_SHA256",
        "EXO_MLX_KIMI_K3_DSPARK_PACKED_AGREEMENTS",
        "EXO_MLX_KIMI_K3_DSPARK_PREFIX_CACHE",
        "EXO_MLX_KIMI_K3_DSPARK_RANK_ZERO_PROPOSAL_RECOVERY",
        "EXO_MLX_KIMI_K3_DSPARK_ROUND_TELEMETRY",
        "EXO_MLX_KIMI_K3_DSPARK_SPECULATIVE",
        "EXO_MLX_KIMI_K3_DSPARK_TAIL_OVERLAP",
        "EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH",
        "EXO_MLX_KIMI_K3_W3_COMPOSITION_EXPECTED_DEFERRED_ROOTS",
        "EXO_MLX_KIMI_K3_W3_COMPOSITION_EXPECTED_KDA_LAYERS",
        "EXO_MLX_KIMI_K3_W3_COMPOSITION_EXPECTED_SPARSE_LAYERS",
        "EXO_MLX_KIMI_K3_W3_COMPOSITION_LIBMLX_SHA256",
        "EXO_MLX_KIMI_K3_W3_COMPOSITION_RECEIPT",
        "EXO_MLX_KIMI_K3_WIDTH4_DISPATCH_RECEIPT_LOG",
        "EXO_MLX_KIMI_K3_WIDTH4_LIBMLX_SHA256",
        "EXO_MLX_KIMI_K3_WIDTH4_RECEIPT_SESSION_ID",
        "EXO_MLX_MAX_ATTENTION_CELLS_PER_CHUNK",
        "EXO_MLX_PIPELINE_LONG_CONTEXT_MIN_TOKENS",
        "EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE",
        "EXO_MLX_PIPELINE_MAX_ATTENTION_CELLS_PER_CHUNK",
        "EXO_MLX_PREFILL_MEMORY_LOG_INTERVAL",
        "EXO_MLX_PREFILL_STEP_SIZE",
        "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE",
        "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS",
        "EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY",
        "EXO_MLX_RANK_LOCAL_VERIFY_HASHES",
        "EXO_MLX_WARMUP_OUTPUT_TOKENS",
        "MLX_JACCL_TP2_HYBRID",
        "MLX_LM_CACHE_SHA256",
        "MLX_LM_COMMIT",
        "MLX_LM_DSPARK_0731_COMMIT",
        "MLX_LM_DSPARK_COMMIT",
        "MLX_LM_FACTORIZED_WIRE_COMMIT",
        "MLX_LM_GATED_DELTA_SHA256",
        "MLX_LM_GENERATE_SHA256",
        "MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256",
        "MLX_LM_KIMI_K3_DSPARK_0731_SHA256",
        "MLX_LM_KIMI_K3_DSPARK_SHA256",
        "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256",
        "MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256",
        "MLX_LM_KIMI_K3_FUSED_ROUTER_SHA256",
        "MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256",
        "MLX_LM_KIMI_K3_PACKED_MOE_FRONT_SHA256",
        "MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE_SHA256",
        "MLX_LM_KIMI_K3_SHA256",
        "MLX_LM_KIMI_K3_TUNED_GATHER_QMV_SHA256",
        "MLX_LM_KIMI_K3_W3_PREWORK_SHA256",
        "MLX_LM_KIMI_K3_WIDTH4_FUSED_EXPERT_SHA256",
        "MLX_LM_PR",
        "MLX_LM_SWITCH_LAYERS_SHA256",
        "MLX_LM_EXPERIMENTAL_KDA_ROW_DECODE",
        "MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3",
        "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT",
        "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3",
        "MLX_LM_KIMI_K3_COMPILED_DECODE",
        "MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS",
        "MLX_LM_KIMI_K3_DSPARK_PROPOSER",
        "MLX_LM_KIMI_K3_DSPARK_SEGMENTED_SDPA",
        "MLX_LM_KIMI_K3_ELIDE_AFFINE2_BIAS",
        "MLX_LM_KIMI_K3_EXACT_SPECULATIVE_KDA",
        "MLX_LM_KIMI_K3_EXACT_WIDE_SHORT_CONV",
        "MLX_LM_KIMI_K3_EXPERT_TOP_K",
        "MLX_LM_KIMI_K3_FACTORIZED_SDPA_PREFILL",
        "MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS",
        "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE",
        "MLX_LM_KIMI_K3_FUSED_EXPERTS",
        "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH2",
        "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH4_EXACT",
        "MLX_LM_KIMI_K3_FUSED_POST_KDA_RMS_SIGMOID_GATE",
        "MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD",
        "MLX_LM_KIMI_K3_FUSED_ROUTER",
        "MLX_LM_KIMI_K3_MULTIBANK_MOE_FRONT",
        "MLX_LM_KIMI_K3_PACKED_KDA_SKINNY",
        "MLX_LM_KIMI_K3_PACKED_KDA_WIDE",
        "MLX_LM_KIMI_K3_PACKED_MOE_FRONT",
        "MLX_LM_KIMI_K3_PACKED_MOE_FRONT_WIDTH8",
        "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE",
        "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE_MAX_TOKENS",
        "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE",
        "MLX_LM_KIMI_K3_W3_COMPOSITION_RECEIPT",
        "MLX_LM_KIMI_K3_W3_PREWORK_HISTORY",
        "MLX_LM_KIMI_K3_WIDTH4_DISPATCH_RECEIPT",
        "MLX_METAL_K3_AFFINE6_Q4_DISPATCH_RECEIPT",
        "MLX_METAL_K3_AFFINE6_Q4_QUAD",
        "MLX_METAL_K3_AFFINE8_Q3_DISPATCH_RECEIPT",
        "MLX_METAL_K3_AFFINE8_Q3_TRIPLET",
        "MLX_METAL_K3_AFFINE8_ROWPAIR",
        "MLX_METAL_K3_PACKED_FRONT_ROWPAIR",
    }
    experiment_prefixes = (
        "EXO_MLX_",
        "MLX_LM_",
        "MLX_METAL_K3_",
        "MLX_JACCL_",
    )
    deployment_names = {
        "EXO_MLX_KIMI_K3_DSPARK_CHECKPOINT",
        "EXO_MLX_KIMI_K3_DSPARK_CONFIDENCE_JSONL",
        "EXO_MLX_KIMI_K3_DSPARK_CONFIDENCE_SESSION",
        "EXO_MLX_KIMI_K3_DSPARK_YARN_CHECKPOINT",
        "EXO_MLX_RANK_LOCAL_CHECKPOINT",
        "EXO_MLX_RANK_LOCAL_LOADER",
        "MLX_JACCL_COORDINATOR",
        "MLX_JACCL_RING",
        "MLX_LM_ROOT",
    }
    experiment_names = {
        name for name in os.environ if name.startswith(experiment_prefixes)
    }
    unexpected = (
        experiment_names
        - common_names
        - deployment_names
        - {_LAUNCH_CONTRACT_SHA256_ENV}
    )
    if unexpected:
        raise ValueError(
            "W3 composition launch contract contains unclassified experiment keys"
        )
    launch_map = {
        name: value
        for name, value in os.environ.items()
        if name != _LAUNCH_CONTRACT_SHA256_ENV and name in common_names
    }
    if not launch_map:
        raise ValueError("W3 composition launch contract is empty")
    if any(
        not name.isascii()
        or not value.isascii()
        or len(name) > 256
        or len(value) > 16_384
        for name, value in launch_map.items()
    ):
        raise ValueError("W3 composition launch contract is not bounded ASCII")
    canonical = json.dumps(
        launch_map,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    actual = hashlib.sha256(canonical).hexdigest()
    expected = os.environ.get(_LAUNCH_CONTRACT_SHA256_ENV)
    if (
        expected is None
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or expected != actual
    ):
        raise ValueError(
            "W3 composition launch-contract SHA does not match the complete "
            "EXO/MLX experiment environment"
        )
    return _digest_words(bytes.fromhex(actual))


def _packed_front_selector_contract(
    *,
    allow_noncanonical_source_test: bool = False,
) -> _CompositionSelectors:
    _require_exact_selector(
        _PACKED_FRONT_EXPECTED_LAYERS_ENV,
        str(_PACKED_FRONT_EXPECTED_LAYERS),
    )
    _require_exact_selector(_KDA_EXPECTED_LAYERS_ENV, str(_KDA_EXPECTED_LAYERS))
    _require_exact_selector(
        _DEFERRED_EXPECTED_ROOTS_ENV,
        str(_DEFERRED_EXPECTED_ROOTS),
    )
    _require_exact_selector(_MLX_PACKED_FRONT_RECEIPT_ENV, "1")
    authoritative = os.environ.get(_MLX_AUTHORITATIVE_PACKED_FRONT_ENV)
    width3 = os.environ.get(_MLX_AUTHORITATIVE_PACKED_FRONT_WIDTH3_ENV)
    if authoritative not in {"0", "1"}:
        raise ValueError(
            f"{_MLX_AUTHORITATIVE_PACKED_FRONT_ENV} must be exactly 0 or 1 "
            "in diagnostic mode"
        )
    if width3 not in {"0", "1"}:
        raise ValueError(
            f"{_MLX_AUTHORITATIVE_PACKED_FRONT_WIDTH3_ENV} must be exactly 0 or 1 "
            "in diagnostic mode"
        )
    if authoritative != width3:
        raise ValueError(
            "authoritative packed-front and width-three selectors must be "
            "jointly disabled or jointly enabled in diagnostic mode"
        )
    kda = os.environ.get(_MLX_KDA_PREWORK_ENV)
    deferred_exo = os.environ.get(_EXO_DEFERRED_WIDTH3_ENV)
    deferred_mlx = os.environ.get(_MLX_DEFERRED_WIDTH3_ENV)
    for name, value in (
        (_MLX_KDA_PREWORK_ENV, kda),
        (_EXO_DEFERRED_WIDTH3_ENV, deferred_exo),
        (_MLX_DEFERRED_WIDTH3_ENV, deferred_mlx),
    ):
        if value not in {"0", "1"}:
            raise ValueError(f"{name} must be exactly 0 or 1 in diagnostic mode")
    if deferred_exo != deferred_mlx:
        raise ValueError("EXO and MLX-LM deferred width-three selectors must match")
    packed_enabled = authoritative == "1"
    kda_enabled = kda == "1"
    deferred_enabled = deferred_exo == "1"
    arm_code = (
        int(packed_enabled) | (int(kda_enabled) << 1) | (int(deferred_enabled) << 2)
    )
    canonical = arm_code in {0, 7}
    if not allow_noncanonical_source_test and not canonical:
        raise ValueError(
            "diagnostic runtime requires canonical control 0/0/0 or full "
            "packed/KDA/deferred candidate 1/1/1"
        )
    _require_exact_selector(_MLX_DUPLICATING_PACKED_FRONT_ENV, "0")
    _require_exact_selector(_MLX_PACKED_FRONT_WIDTH8_ENV, "0")
    _require_exact_selector(_MLX_MULTIBANK_PACKED_FRONT_ENV, "0")
    _require_exact_selector(_MLX_COMPILED_DECODE_ENV, "0")
    _require_exact_selector(_MLX_NATIVE_AFFINE8_Q3_TRIPLET_ENV, "1")
    _require_exact_selector(_MLX_NATIVE_AFFINE8_Q3_RECEIPT_ENV, "1")
    _require_exact_selector(_EXO_TAIL_OVERLAP_ENV, "1")
    _require_exact_selector(_WIDTH4_RECEIPT_ENV, "0")
    base_selectors = {
        "EXO_NO_BATCH": "1",
        "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE": "1",
        "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE": "1",
        _MLX_PROJECTED_KV_CACHE_MAX_TOKENS_ENV: str(_PROJECTED_KV_CACHE_MAX_TOKENS),
        "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES": "laguna8",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE": "hidden",
    }
    for name, expected in base_selectors.items():
        _require_exact_selector(name, expected)
    selector_map = {
        _PACKED_FRONT_DIAGNOSTIC_ENV: 1,
        _PACKED_FRONT_EXPECTED_LAYERS_ENV: _PACKED_FRONT_EXPECTED_LAYERS,
        _MLX_PACKED_FRONT_RECEIPT_ENV: 1,
        _MLX_AUTHORITATIVE_PACKED_FRONT_ENV: int(authoritative),
        _MLX_AUTHORITATIVE_PACKED_FRONT_WIDTH3_ENV: int(width3),
        _MLX_KDA_PREWORK_ENV: int(cast(str, kda)),
        _EXO_DEFERRED_WIDTH3_ENV: int(cast(str, deferred_exo)),
        _MLX_DEFERRED_WIDTH3_ENV: int(cast(str, deferred_mlx)),
        _MLX_DUPLICATING_PACKED_FRONT_ENV: 0,
        _MLX_PACKED_FRONT_WIDTH8_ENV: 0,
        _MLX_MULTIBANK_PACKED_FRONT_ENV: 0,
        _MLX_COMPILED_DECODE_ENV: 0,
        _MLX_NATIVE_AFFINE8_Q3_TRIPLET_ENV: 1,
        _MLX_NATIVE_AFFINE8_Q3_RECEIPT_ENV: 1,
        _EXO_TAIL_OVERLAP_ENV: 1,
        _WIDTH4_RECEIPT_ENV: 0,
        **base_selectors,
    }
    canonical_selector = json.dumps(
        selector_map,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(canonical_selector).digest()
    words = tuple(
        int.from_bytes(digest[offset : offset + 8], "big")
        & _PACKED_FRONT_SEQUENCE_LIMIT
        for offset in range(0, 32, 8)
    )
    return _CompositionSelectors(
        packed=packed_enabled,
        kda=kda_enabled,
        deferred=deferred_enabled,
        arm_code=arm_code,
        canonical=canonical,
        digest=cast(tuple[int, int, int, int], words),
        launch_digest=_launch_contract_digest(),
    )


def _canonical_numeric_json(payload: dict[str, object]) -> str:
    if any(type(value) not in {bool, int} for value in payload.values()):
        raise TypeError("packed-front marker values must be numeric scalars")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _parse_canonical_numeric_json(text: str) -> dict[str, object]:
    value = cast(object, json.loads(text))
    if not isinstance(value, dict):
        raise TypeError("packed-front agreement must be scalar numeric JSON")
    parsed: dict[str, object] = {}
    for name, item in cast(dict[object, object], value).items():
        if type(name) is not str or type(item) not in {bool, int}:
            raise TypeError("packed-front agreement must be scalar numeric JSON")
        parsed[name] = item
    canonical = _canonical_numeric_json(parsed)
    if canonical != text:
        raise ValueError("packed-front agreement JSON is not canonical")
    return parsed


def _packed_front_request_token(
    setup_fingerprint: tuple[int, ...],
    *,
    phase: int,
    selector_digest: tuple[int, int, int, int],
    launch_contract_digest: tuple[int, int, int, int],
    source_digest: tuple[int, int, int, int],
    exo_source_digest: tuple[int, int, int, int],
    native_lib_digest: tuple[int, int, int, int],
) -> int:
    if not setup_fingerprint or any(
        type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF
        for value in setup_fingerprint
    ):
        raise ValueError("packed-front request setup fingerprint is invalid")
    digest = hashlib.sha256(b"exo-kimi-k3-w3-composition-request/v1\0")
    digest.update(phase.to_bytes(1, "big"))
    for value in (
        *selector_digest,
        *launch_contract_digest,
        *source_digest,
        *exo_source_digest,
        *native_lib_digest,
        *setup_fingerprint,
    ):
        digest.update(value.to_bytes(8, "big", signed=False))
    return int.from_bytes(digest.digest()[:8], "big") & _PACKED_FRONT_SEQUENCE_LIMIT


class _NumericReceiptDigest:
    """Domain-separated numeric digest that never retains raw token values."""

    def __init__(self, domain: bytes):
        self._digest = hashlib.sha256(domain)

    def update(self, *values: int) -> None:
        for value in values:
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError("W3 composition digest value is invalid")
            self._digest.update(value.to_bytes(8, "big"))

    def words(self) -> tuple[int, int, int, int]:
        return _digest_words(self._digest.copy().digest())


def _anchor_digest(round_index: int, token: int) -> tuple[int, int, int, int]:
    if (
        type(round_index) is not int
        or round_index < 0
        or type(token) is not int
        or not 0 <= token <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("W3 composition anchor fingerprint input is invalid")
    digest = hashlib.sha256(b"exo-kimi-k3-w3-anchor/v1\0")
    digest.update(round_index.to_bytes(8, "big"))
    digest.update(token.to_bytes(8, "big"))
    return _digest_words(digest.digest())


def _claim_composition_receipt_phase(phase: int) -> None:
    global _COMPOSITION_RECEIPT_ACTIVE_PHASE
    with _COMPOSITION_RECEIPT_LOCK:
        if _COMPOSITION_RECEIPT_ACTIVE_PHASE is not None:
            raise RuntimeError("a W3 composition receipt phase is already active")
        if phase in _COMPOSITION_RECEIPT_ATTEMPTED_PHASES:
            raise RuntimeError(
                "W3 composition receipt phases are process-local one-shot"
            )
        if (
            phase == _PACKED_FRONT_PHASE_API
            and _PACKED_FRONT_PHASE_ENGINE_STARTUP
            not in _COMPOSITION_RECEIPT_COMPLETED_PHASES
        ):
            raise RuntimeError(
                "W3 composition API receipt requires a published startup receipt"
            )
        _COMPOSITION_RECEIPT_ATTEMPTED_PHASES.add(phase)
        _COMPOSITION_RECEIPT_ACTIVE_PHASE = phase


def _release_composition_receipt_phase(phase: int) -> None:
    global _COMPOSITION_RECEIPT_ACTIVE_PHASE
    with _COMPOSITION_RECEIPT_LOCK:
        if phase == _COMPOSITION_RECEIPT_ACTIVE_PHASE:
            _COMPOSITION_RECEIPT_ACTIVE_PHASE = None


def _record_composition_receipt_publication(
    phase: int,
    identity: tuple[int, ...],
) -> None:
    global _COMPOSITION_RECEIPT_ACTIVE_PHASE
    with _COMPOSITION_RECEIPT_LOCK:
        if phase != _COMPOSITION_RECEIPT_ACTIVE_PHASE:
            raise RuntimeError("W3 composition receipt phase is not active")
        if phase in _COMPOSITION_RECEIPT_PUBLISHED_PHASES:
            raise RuntimeError("W3 composition receipt phase was already published")
        _COMPOSITION_RECEIPT_PUBLISHED_PHASES.add(phase)
        _COMPOSITION_RECEIPT_PHASE_IDENTITIES[phase] = identity
        if phase == _PACKED_FRONT_PHASE_API:
            _COMPOSITION_RECEIPT_COMPLETED_PHASES.add(phase)
        _COMPOSITION_RECEIPT_ACTIVE_PHASE = None


def _complete_composition_startup_after_warmup() -> None:
    with _COMPOSITION_RECEIPT_LOCK:
        phase = _PACKED_FRONT_PHASE_ENGINE_STARTUP
        if (
            phase not in _COMPOSITION_RECEIPT_ATTEMPTED_PHASES
            or phase not in _COMPOSITION_RECEIPT_PUBLISHED_PHASES
            or phase not in _COMPOSITION_RECEIPT_PHASE_IDENTITIES
            or phase in _COMPOSITION_RECEIPT_COMPLETED_PHASES
            or _COMPOSITION_RECEIPT_ACTIVE_PHASE is not None
        ):
            raise RuntimeError(
                "W3 composition startup cannot complete before published warmup close"
            )
        _COMPOSITION_RECEIPT_COMPLETED_PHASES.add(phase)


def _require_composition_api_identity(identity: tuple[int, ...]) -> None:
    with _COMPOSITION_RECEIPT_LOCK:
        startup = _PACKED_FRONT_PHASE_ENGINE_STARTUP
        if (
            startup not in _COMPOSITION_RECEIPT_COMPLETED_PHASES
            or _COMPOSITION_RECEIPT_PHASE_IDENTITIES.get(startup) != identity
        ):
            raise RuntimeError(
                "W3 composition API identity differs from completed startup"
            )


@dataclass
class _PackedFrontReceiptRequest:
    model: Model
    enabled: bool
    phase: int
    selectors: _CompositionSelectors | None = None
    selector_digest: tuple[int, int, int, int] = (0, 0, 0, 0)
    launch_contract_digest: tuple[int, int, int, int] = (0, 0, 0, 0)
    api: _PackedFrontReceiptAPI | None = None
    native_api: _NativeQ3ReceiptAPI | None = None
    exo_source_digest: tuple[int, int, int, int] = (0, 0, 0, 0)
    setup_digest: tuple[int, int, int, int] = (0, 0, 0, 0)
    _native_baseline: tuple[int, int, int, int] | None = None
    _runtime: KimiK3DSparkRequestRuntime | None = None
    _handle: tuple[int, int] | None = None
    _receipt: K3W3CompositionReceipt | None = None
    _marker: str | None = None
    _published: bool = False
    _poisoned: bool = False
    _stale_events: int = 0
    _duplicate_events: int = 0
    _round_records: list[tuple[int, ...]] = field(default_factory=list)
    _causal_events: list[tuple[int, str, tuple[int, ...]]] = field(default_factory=list)
    _seen_events: set[tuple[int, str, int]] = field(default_factory=set)
    _visible_output_tokens: int = 0
    _decode_begin_cache_offset: int | None = None
    _expected_anchor_digest: tuple[int, int, int, int] | None = None
    _tail_prelaunch_submitted: int = 0
    _tail_prelaunch_used: int = 0
    _tail_prelaunch_discarded: int = 0
    _native_fd_closed: bool = False
    _schedule_digest: _NumericReceiptDigest = field(
        default_factory=lambda: _NumericReceiptDigest(b"exo-kimi-k3-w3-schedule/v1\0")
    )
    _proposal_digest: _NumericReceiptDigest = field(
        default_factory=lambda: _NumericReceiptDigest(b"exo-kimi-k3-w3-proposal/v1\0")
    )
    _acceptance_digest: _NumericReceiptDigest = field(
        default_factory=lambda: _NumericReceiptDigest(b"exo-kimi-k3-w3-acceptance/v1\0")
    )
    _committed_output_digest: _NumericReceiptDigest = field(
        default_factory=lambda: _NumericReceiptDigest(
            b"exo-kimi-k3-w3-committed-output/v1\0"
        )
    )
    _visible_output_digest: _NumericReceiptDigest = field(
        default_factory=lambda: _NumericReceiptDigest(
            b"exo-kimi-k3-w3-visible-output/v1\0"
        )
    )
    _cache_digest: _NumericReceiptDigest = field(
        default_factory=lambda: _NumericReceiptDigest(b"exo-kimi-k3-w3-cache/v1\0")
    )
    _causal_digest: _NumericReceiptDigest = field(
        default_factory=lambda: _NumericReceiptDigest(b"exo-kimi-k3-w3-causal/v1\0")
    )

    @property
    def candidate(self) -> bool:
        """Compatibility label for the packed-only donor validator."""

        return self.selectors is not None and self.selectors.arm_code == 7

    @classmethod
    def from_environment(
        cls,
        model: Model,
        dspark: LoadedKimiK3DSpark | None,
        *,
        phase: Literal["engine_startup", "api"],
    ) -> "_PackedFrontReceiptRequest":
        if phase not in _PACKED_FRONT_PHASE_CODES:
            raise ValueError("packed-front receipt phase is invalid")
        enabled = _strict_env_flag(
            _PACKED_FRONT_DIAGNOSTIC_ENV,
            os.environ.get(_PACKED_FRONT_DIAGNOSTIC_ENV, "0"),
        )
        phase_code = _PACKED_FRONT_PHASE_CODES[phase]
        if not enabled:
            _reject_orphan_composition_receipt_environment()
            return cls(model=model, enabled=False, phase=phase_code)
        if dspark is None or int(dspark.verify_width) != 3:
            raise ValueError(
                "packed-front diagnostic mode requires Kimi K3 DSpark width three"
            )
        _claim_composition_receipt_phase(phase_code)
        native_api: _NativeQ3ReceiptAPI | None = None
        try:
            selectors = _packed_front_selector_contract()
            exo_source_digest = _exo_source_digest()
            api = _load_packed_front_receipt_api()
            native_api = _load_native_q3_receipt_api()
            identity = (
                *selectors.digest,
                *selectors.launch_digest,
                *api.source_digest,
                *exo_source_digest,
                *native_api.lib_digest,
                *native_api.lib_identity,
            )
            if phase_code == _PACKED_FRONT_PHASE_API:
                _require_composition_api_identity(identity)
        except BaseException:
            # The phase remains attempted, so neither malformed startup nor a
            # failed loader can be retried around the process-global counter
            # interval. Releasing only the active slot makes the failure state
            # explicit and lets the API-before-success gate produce its own
            # deterministic rejection.
            if native_api is not None:
                with contextlib.suppress(OSError):
                    os.close(native_api.lib_fd)
            _release_composition_receipt_phase(phase_code)
            raise
        return cls(
            model=model,
            enabled=True,
            phase=phase_code,
            selectors=selectors,
            selector_digest=selectors.digest,
            launch_contract_digest=selectors.launch_digest,
            api=api,
            native_api=native_api,
            exo_source_digest=exo_source_digest,
        )

    def _phase_identity(self) -> tuple[int, ...]:
        if self.api is None or self.native_api is None:
            raise RuntimeError("W3 composition phase identity is unavailable")
        return (
            *self.selector_digest,
            *self.launch_contract_digest,
            *self.api.source_digest,
            *self.exo_source_digest,
            *self.native_api.lib_digest,
            *self.native_api.lib_identity,
        )

    def _close_native_fd(self) -> None:
        if self.native_api is None or self._native_fd_closed:
            return
        self._native_fd_closed = True
        with contextlib.suppress(OSError):
            os.close(self.native_api.lib_fd)

    def _native_counts(self) -> tuple[int, int, int, int]:
        if self.native_api is None:
            raise RuntimeError("native affine8 Q3 receipt API is unavailable")
        raw_values = (
            self.native_api.total(),
            self.native_api.n4480(),
            self.native_api.n6144(),
            self.native_api.n10624(),
        )
        if any(type(value) is not int for value in raw_values):
            raise TypeError("native affine8 Q3 receipt counter is not an integer")
        values = cast(tuple[int, int, int, int], raw_values)
        if any(not 0 <= value <= _PACKED_FRONT_COUNTER_LIMIT for value in values):
            raise ValueError("native affine8 Q3 receipt counter is invalid")
        if values[0] < sum(values[1:]):
            raise ValueError("native affine8 Q3 named counters exceed the aggregate")
        return values

    def _assert_runtime_identity(self) -> None:
        if self.api is None or self.native_api is None:
            raise RuntimeError("W3 composition runtime identity is unavailable")
        if _launch_contract_digest() != self.launch_contract_digest:
            raise ValueError("W3 composition launch contract changed during request")
        if _runtime_source_digest() != self.api.source_digest:
            raise ValueError("W3 composition MLX source changed during request")
        if _exo_source_digest() != self.exo_source_digest:
            raise ValueError("W3 composition EXO source changed during request")
        _assert_native_library_identity(self.native_api)

    def observe_causal_event(
        self,
        round_index: int,
        event: str,
        values: tuple[int, ...] = (),
    ) -> None:
        """Hash an in-memory causal event; this observer never raises."""

        event_codes = {
            "proposal_agreed": 1,
            "deferred_roots_validated": 2,
            "target_graph_agreed": 3,
            "deferred_root_submitted": 4,
            "deferred_roots_submitted": 5,
            "acceptance_agreed": 6,
            "tail_prelaunch_submitted": 7,
            "target_commit": 8,
            "tail_prelaunch_used": 9,
            "tail_prelaunch_discarded": 10,
            "ordinary_target_commit": 11,
            "committed_tokens": 12,
            "ordinary_anchor_agreed": 13,
            "proposal_context_agreed": 14,
        }
        try:
            if (
                type(round_index) is not int
                or round_index < 0
                or event not in event_codes
                or any(
                    type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF
                    for value in values
                )
            ):
                raise ValueError("invalid W3 causal event")
            retained_values = values
            if event == "proposal_agreed":
                if len(values) != 3:
                    raise ValueError("width-three proposal block must contain 3 tokens")
                if (
                    self._expected_anchor_digest is None
                    or _anchor_digest(round_index, values[0])
                    != self._expected_anchor_digest
                ):
                    raise ValueError("width-three proposal anchor is not contiguous")
                self._proposal_digest.update(
                    round_index,
                    len(values),
                    *values,
                )
                retained_values = (len(values),)
            elif event == "ordinary_anchor_agreed":
                if (
                    len(values) != 1
                    or self._expected_anchor_digest is None
                    or _anchor_digest(round_index, values[0])
                    != self._expected_anchor_digest
                ):
                    raise ValueError("ordinary target anchor is not contiguous")
                self._proposal_digest.update(
                    round_index,
                    1,
                    values[0],
                )
                retained_values = ()
            elif event == "committed_tokens":
                if not values:
                    raise ValueError("committed token block must not be empty")
                self._committed_output_digest.update(
                    round_index,
                    len(values),
                    *values,
                )
                self._expected_anchor_digest = _anchor_digest(
                    round_index + 1,
                    values[-1],
                )
                retained_values = (
                    len(values),
                    *self._expected_anchor_digest,
                )
            elif event in {"tail_prelaunch_submitted", "tail_prelaunch_used"}:
                if len(values) != 2:
                    raise ValueError("tail-prelaunch identity is invalid")
                anchor_round = round_index + int(event == "tail_prelaunch_submitted")
                retained_values = (
                    values[0],
                    *_anchor_digest(anchor_round, values[1]),
                )
            elif event == "tail_prelaunch_discarded":
                if len(values) != 3 or values[0] not in {1, 2}:
                    raise ValueError("tail-prelaunch discard reason is invalid")
                self._tail_prelaunch_discarded += 1
                anchor_round = round_index + int(values[0] == 2)
                retained_values = (
                    values[0],
                    values[1],
                    *_anchor_digest(anchor_round, values[2]),
                )
            elif event == "proposal_context_agreed":
                if len(values) != 1:
                    raise ValueError("proposal context offset is invalid")
            elif event == "acceptance_agreed":
                if len(values) != 1 or values[0] > 2:
                    raise ValueError("width-three acceptance is invalid")
                self._acceptance_digest.update(round_index, values[0])
            discriminator = (
                retained_values[0]
                if event in {"deferred_root_submitted", "tail_prelaunch_discarded"}
                and retained_values
                else 0
            )
            key = (round_index, event, discriminator)
            if key in self._seen_events:
                self._duplicate_events += 1
                return
            self._seen_events.add(key)
            if self._causal_events and round_index < self._causal_events[-1][0]:
                self._stale_events += 1
            self._causal_events.append((round_index, event, retained_values))
            self._causal_digest.update(
                round_index,
                event_codes[event],
                len(retained_values),
                *retained_values,
            )
        except Exception:
            self._poisoned = True

    def observe_round(
        self,
        stats: DSparkRoundTelemetry,
        *,
        target_cache_tokens: int | None,
    ) -> None:
        """Bind the committed schedule and cache boundary without timings."""

        try:
            if (
                stats.round_index != len(self._round_records)
                or type(target_cache_tokens) is not int
                or target_cache_tokens < 0
                or target_cache_tokens > _PROJECTED_KV_CACHE_MAX_TOKENS
            ):
                raise ValueError("W3 receipt round/cache sequence is invalid")
            record = (
                stats.round_index,
                stats.proposed,
                stats.accepted,
                stats.emitted,
                int(stats.fallback),
                int(stats.error is not None),
                int(stats.prelaunch_submitted),
                int(stats.prelaunch_used),
                target_cache_tokens,
            )
            if any(type(value) is not int or value < 0 for value in record):
                raise ValueError("W3 receipt round values are invalid")
            self._round_records.append(record)
            self._schedule_digest.update(*record[:-1])
            self._cache_digest.update(stats.round_index, target_cache_tokens)
            self._tail_prelaunch_submitted += int(stats.prelaunch_submitted)
            self._tail_prelaunch_used += int(stats.prelaunch_used)
        except Exception:
            self._poisoned = True

    def observe_output_token(self, token: int, *, from_draft: bool) -> None:
        """Hash an agreed visible token immediately and retain no token value."""

        try:
            if (
                type(token) is not int
                or not 0 <= token <= 0xFFFFFFFFFFFFFFFF
                or type(from_draft) is not bool
            ):
                raise ValueError("W3 receipt output token is invalid")
            self._visible_output_digest.update(
                self._visible_output_tokens,
                token,
                int(from_draft),
            )
            self._visible_output_tokens += 1
        except Exception:
            self._poisoned = True

    def begin(
        self,
        runtime: KimiK3DSparkRequestRuntime,
        setup_fingerprint: tuple[int, ...],
    ) -> None:
        if not self.enabled:
            return
        if (
            self.api is None
            or self.native_api is None
            or self.selectors is None
            or self._handle is not None
            or self._receipt is not None
        ):
            raise RuntimeError("packed-front receipt begin state is invalid")
        setup_hasher = hashlib.sha256(b"exo-kimi-k3-w3-setup/v1\0")
        for word in setup_fingerprint:
            if type(word) is not int or not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError("W3 composition setup fingerprint is invalid")
            setup_hasher.update(word.to_bytes(8, "big"))
        self.setup_digest = _digest_words(setup_hasher.digest())
        request_token = _packed_front_request_token(
            setup_fingerprint,
            phase=self.phase,
            selector_digest=self.selector_digest,
            launch_contract_digest=self.launch_contract_digest,
            source_digest=self.api.source_digest,
            exo_source_digest=self.exo_source_digest,
            native_lib_digest=self.native_api.lib_digest,
        )
        local_handles: list[tuple[int, int]] = []

        def begin_local() -> str:
            assert self.api is not None
            assert self.native_api is not None
            self._assert_runtime_identity()
            self.native_api.reset()
            baseline = self._native_counts()
            if baseline != (0, 0, 0, 0):
                raise ValueError("native affine8 Q3 receipt reset is not zero")
            self._native_baseline = baseline
            handle = self.api.begin(
                request_token,
                self.model,
                expected_sparse_layers=_PACKED_FRONT_EXPECTED_LAYERS,
                expected_kda_layers=_KDA_EXPECTED_LAYERS,
            )
            if not isinstance(handle, tuple):
                raise ValueError("MLX-LM packed-front receipt handle is invalid")
            handle_values = cast(tuple[object, ...], handle)
            if len(handle_values) != 2:
                raise ValueError("MLX-LM packed-front receipt handle is invalid")
            request_sequence_value, request_token_value = handle_values
            if (
                type(request_sequence_value) is not int
                or type(request_token_value) is not int
                or not 1 <= request_sequence_value <= _PACKED_FRONT_SEQUENCE_LIMIT
                or request_token_value != request_token
            ):
                raise ValueError("MLX-LM packed-front receipt handle is invalid")
            validated_handle = (request_sequence_value, request_token_value)
            self._handle = validated_handle
            local_handles.append(validated_handle)
            return _canonical_numeric_json(
                {
                    "request_sequence": validated_handle[0],
                    "request_token": validated_handle[1],
                }
            )

        agreed = runtime.agree_text("packed-front receipt begin", begin_local)
        handle_payload = _parse_canonical_numeric_json(agreed)
        if set(handle_payload) != {"request_sequence", "request_token"}:
            raise ValueError("packed-front receipt begin agreement schema is invalid")
        request_sequence = handle_payload["request_sequence"]
        request_token_value = handle_payload["request_token"]
        if type(request_sequence) is not int or type(request_token_value) is not int:
            raise TypeError("packed-front receipt begin agreement values are invalid")
        agreed_handle = (request_sequence, request_token_value)
        if local_handles != [agreed_handle]:
            raise RuntimeError("packed-front receipt begin agreement diverged")
        runtime.attach_composition_receipt_sink(self)
        self._runtime = runtime

    def begin_decode(
        self,
        runtime: KimiK3DSparkRequestRuntime,
        *,
        anchor_token: int,
    ) -> None:
        """Bind the post-prefill cache boundary and first rank-agreed anchor."""

        if not self.enabled:
            return
        if (
            runtime is not self._runtime
            or self._handle is None
            or self._decode_begin_cache_offset is not None
            or type(anchor_token) is not int
            or anchor_token < 0
        ):
            raise RuntimeError("W3 composition decode-start state is invalid")

        def local_boundary() -> str:
            return _canonical_numeric_json(
                {
                    "anchor_token": anchor_token,
                    "target_cache_offset": runtime.target_cache_offset,
                }
            )

        agreed = runtime.agree_text("W3 composition decode start", local_boundary)
        payload = _parse_canonical_numeric_json(agreed)
        if set(payload) != {"anchor_token", "target_cache_offset"}:
            raise ValueError("W3 composition decode-start schema is invalid")
        agreed_anchor = payload["anchor_token"]
        begin_cache_offset = payload["target_cache_offset"]
        if (
            type(agreed_anchor) is not int
            or agreed_anchor != anchor_token
            or type(begin_cache_offset) is not int
            or begin_cache_offset < 0
            or begin_cache_offset > _PROJECTED_KV_CACHE_MAX_TOKENS
        ):
            raise ValueError("W3 composition decode-start agreement diverged")
        self._expected_anchor_digest = _anchor_digest(0, agreed_anchor)
        self._decode_begin_cache_offset = begin_cache_offset
        self._cache_digest.update(
            _PACKED_FRONT_SEQUENCE_LIMIT,
            begin_cache_offset,
        )

    def _validated_marker(
        self,
        raw: dict[str, object],
        telemetry: "_PromptLookupTelemetry",
        runtime: KimiK3DSparkRequestRuntime,
    ) -> dict[str, object]:
        """Validate exact MLX/native/schedule algebra and emit numeric scalars."""

        if set(raw) != set(_PACKED_FRONT_MLX_KEYS):
            raise ValueError("MLX-LM W3 composition receipt fields are invalid")
        if (
            raw["schema"] != _PACKED_FRONT_RECEIPT_SCHEMA
            or raw["finalized"] is not True
            or raw["aborted"] is not False
            or raw["poisoned"] is not False
            or type(raw["expected_sparse_layers"]) is not int
            or raw["expected_sparse_layers"] != _PACKED_FRONT_EXPECTED_LAYERS
            or type(raw["expected_kda_layers"]) is not int
            or raw["expected_kda_layers"] != _KDA_EXPECTED_LAYERS
        ):
            raise ValueError("MLX-LM W3 composition receipt header is invalid")
        if (
            self._handle is None
            or self.selectors is None
            or self.api is None
            or self.native_api is None
        ):
            raise RuntimeError("W3 composition receipt binding is unavailable")
        if not self.selectors.canonical or self.selectors.arm_code not in {0, 7}:
            raise ValueError(
                "W3 composition receipt requires canonical control or candidate"
            )
        raw_sequence = raw["request_sequence"]
        raw_token = raw["request_token"]
        if (
            type(raw_sequence) is not int
            or not 1 <= raw_sequence <= _PACKED_FRONT_SEQUENCE_LIMIT
            or raw_sequence != self._handle[0]
            or type(raw_token) is not int
            or not 0 <= raw_token <= _PACKED_FRONT_SEQUENCE_LIMIT
            or raw_token != self._handle[1]
        ):
            raise ValueError("MLX-LM W3 composition receipt binding is invalid")
        self._assert_runtime_identity()

        expected_selector_snapshot: dict[str, object] = {
            "packed_authoritative_enabled": self.selectors.packed,
            "packed_width3_enabled": self.selectors.packed,
            "kda_prework_enabled": self.selectors.kda,
            "replayssm_speculative_enabled": True,
            "projected_kv_cache_enabled": True,
            "projected_kv_cache_max_tokens": _PROJECTED_KV_CACHE_MAX_TOKENS,
            "async_decode_boundaries": "laguna8",
            "async_decode_state": "hidden",
            "async_decode_width3_enabled": self.selectors.deferred,
            "native_q3_triplet_enabled": True,
            "native_q3_dispatch_receipt_enabled": True,
        }
        for name, expected in expected_selector_snapshot.items():
            if type(raw[name]) is not type(expected) or raw[name] != expected:
                raise ValueError(f"MLX-LM W3 selector snapshot {name} drifted")

        counters: dict[str, int] = {}
        for name in _PACKED_FRONT_COUNTER_FIELDS:
            value = raw[name]
            if type(value) is not int or not 0 <= value <= _PACKED_FRONT_COUNTER_LIMIT:
                raise ValueError(f"MLX-LM W3 receipt counter {name} is invalid")
            counters[name] = value
        if counters["helper_calls"] != sum(
            counters[name]
            for name in (
                "gate_disabled_calls",
                "noncontract_calls",
                "packed_hits",
                "unsupported_calls",
                "packed_dispatch_fallback_calls",
            )
        ):
            raise ValueError("MLX-LM packed helper partition is invalid")
        if counters["eligible_width1_calls"] != sum(
            counters[name]
            for name in (
                "packed_width1_hits",
                "width1_unsupported_calls",
                "width1_dispatch_fallback_calls",
            )
        ) or counters["eligible_width3_calls"] != sum(
            counters[name]
            for name in (
                "packed_width3_hits",
                "width3_unsupported_calls",
                "width3_dispatch_fallback_calls",
            )
        ):
            raise ValueError("MLX-LM packed eligible-width partition is invalid")
        aggregate_pairs = (
            ("packed_hits", "packed_width1_hits", "packed_width3_hits"),
            (
                "packed_output_tensors",
                "packed_width1_output_tensors",
                "packed_width3_output_tensors",
            ),
            ("lazy_installs", "packed_width1_installs", "packed_width3_installs"),
            (
                "unsupported_calls",
                "width1_unsupported_calls",
                "width3_unsupported_calls",
            ),
            (
                "packed_dispatch_fallback_calls",
                "width1_dispatch_fallback_calls",
                "width3_dispatch_fallback_calls",
            ),
        )
        if any(
            counters[total] != counters[first] + counters[second]
            for total, first, second in aggregate_pairs
        ):
            raise ValueError("MLX-LM packed aggregate partition is invalid")
        if (
            counters["kda_admitted_calls"]
            != counters["kda_success_calls"]
            + counters["kda_fallback_calls"]
            + counters["kda_pending_calls"]
            or counters["kda_helper_calls"]
            != counters["kda_gate_disabled_calls"]
            + counters["kda_noncontract_calls"]
            + counters["kda_admitted_calls"]
            or counters["kda_pending_calls"] != 0
        ):
            raise ValueError("MLX-LM KDA helper partition is invalid")

        if telemetry.fallback_rounds != 0 or telemetry.error_rounds != 0:
            raise ValueError("W3 diagnostic request had fallback or error rounds")
        full_rounds = telemetry.full_width_rounds
        tail_rounds = telemetry.target_width1_rounds
        if (
            full_rounds + tail_rounds != telemetry.rounds
            or len(self._round_records) != telemetry.rounds
            or any(
                record[0] != index for index, record in enumerate(self._round_records)
            )
        ):
            raise ValueError("W3 diagnostic round schedule is incomplete")
        prefill_width1 = telemetry.prefill_width1_chunks
        prefill_width3 = telemetry.prefill_width3_chunks
        prefill_noncontract = telemetry.prefill_noncontract_chunks
        if (
            min(prefill_width1, prefill_width3, prefill_noncontract) < 0
            or prefill_width1 not in {0, 1}
            or prefill_width3 not in {0, 1}
            or prefill_width1 + prefill_width3 > 1
        ):
            raise ValueError("W3 diagnostic prefill geometry is invalid")

        expected_width1 = (tail_rounds + prefill_width1) * _PACKED_FRONT_EXPECTED_LAYERS
        expected_width3 = (full_rounds + prefill_width3) * _PACKED_FRONT_EXPECTED_LAYERS
        if self.selectors.packed:
            zero_packed_failures = (
                "gate_disabled_calls",
                "unsupported_calls",
                "packed_dispatch_fallback_calls",
                "invalidations",
                "stale_resets",
            )
            if any(counters[name] != 0 for name in zero_packed_failures):
                raise ValueError("candidate packed receipt contains fallback drift")
            if (
                counters["packed_width1_hits"] != expected_width1
                or counters["eligible_width1_calls"] != expected_width1
                or counters["packed_width3_hits"] != expected_width3
                or counters["eligible_width3_calls"] != expected_width3
                or counters["packed_hits"] != expected_width1 + expected_width3
                or counters["packed_width1_output_tensors"] != expected_width1 * 4
                or counters["packed_width3_output_tensors"] != expected_width3 * 4
                or counters["packed_output_tensors"]
                != (expected_width1 + expected_width3) * 4
                or counters["noncontract_calls"]
                != prefill_noncontract * _PACKED_FRONT_EXPECTED_LAYERS
            ):
                raise ValueError("candidate packed W3 hit algebra is invalid")
            if self.phase == _PACKED_FRONT_PHASE_ENGINE_STARTUP:
                installs = (
                    counters["packed_width1_installs"],
                    counters["packed_width3_installs"],
                )
                if (
                    full_rounds == 0
                    or counters["pack_count_before"] != 0
                    or counters["pack_count_after"] != _PACKED_FRONT_EXPECTED_LAYERS
                    or installs
                    not in {
                        (_PACKED_FRONT_EXPECTED_LAYERS, 0),
                        (0, _PACKED_FRONT_EXPECTED_LAYERS),
                    }
                    or (installs[0] > 0 and expected_width1 == 0)
                    or (installs[1] > 0 and expected_width3 == 0)
                    or counters["lazy_installs"] != _PACKED_FRONT_EXPECTED_LAYERS
                ):
                    raise ValueError("startup did not prove the exact 0-to-92 install")
            elif (
                counters["pack_count_before"] != _PACKED_FRONT_EXPECTED_LAYERS
                or counters["pack_count_after"] != _PACKED_FRONT_EXPECTED_LAYERS
                or counters["lazy_installs"] != 0
            ):
                raise ValueError("API request did not preserve all 92 packed parents")
        else:
            expected_helpers = (
                telemetry.rounds + prefill_width1 + prefill_width3 + prefill_noncontract
            ) * _PACKED_FRONT_EXPECTED_LAYERS
            if (
                counters["helper_calls"] != expected_helpers
                or counters["gate_disabled_calls"] != expected_helpers
            ):
                raise ValueError("control packed gate accounting is invalid")
            packed_noncontrol_fields = (
                "eligible_width1_calls",
                "eligible_width3_calls",
                "packed_width1_hits",
                "packed_width3_hits",
                "packed_hits",
                "packed_width1_output_tensors",
                "packed_width3_output_tensors",
                "packed_output_tensors",
                "packed_width1_installs",
                "packed_width3_installs",
                "lazy_installs",
                "noncontract_calls",
                "width1_unsupported_calls",
                "width3_unsupported_calls",
                "unsupported_calls",
                "width1_dispatch_fallback_calls",
                "width3_dispatch_fallback_calls",
                "packed_dispatch_fallback_calls",
                "invalidations",
                "stale_resets",
                "pack_count_before",
                "pack_count_after",
            )
            if any(counters[name] != 0 for name in packed_noncontrol_fields):
                raise ValueError("control packed receipt contains candidate work")

        expected_kda_helpers = (
            full_rounds + prefill_width3 + prefill_noncontract
        ) * _KDA_EXPECTED_LAYERS
        if counters["kda_helper_calls"] != expected_kda_helpers:
            raise ValueError("KDA helper-call geometry is invalid")
        if self.selectors.kda:
            expected_kda_success = full_rounds * _KDA_EXPECTED_LAYERS
            expected_kda_noncontract = (
                prefill_width3 + prefill_noncontract
            ) * _KDA_EXPECTED_LAYERS
            if (
                counters["kda_gate_disabled_calls"] != 0
                or counters["kda_noncontract_calls"] != expected_kda_noncontract
                or counters["kda_admitted_calls"] != expected_kda_success
                or counters["kda_success_calls"] != expected_kda_success
                or counters["kda_fallback_calls"] != 0
            ):
                raise ValueError("candidate KDA W3 algebra is invalid")
        elif counters["kda_gate_disabled_calls"] != expected_kda_helpers or any(
            counters[name] != 0
            for name in (
                "kda_noncontract_calls",
                "kda_admitted_calls",
                "kda_success_calls",
                "kda_fallback_calls",
            )
        ):
            raise ValueError("control KDA gate accounting is invalid")

        deferred = runtime.deferred_async_width3_attestation
        if self.selectors.deferred:
            if (
                deferred.enabled is not True
                or deferred.validated_rounds != full_rounds
                or deferred.materialized_rounds != full_rounds
                or deferred.validated_roots != full_rounds * _DEFERRED_EXPECTED_ROOTS
                or deferred.submitted_roots != full_rounds * _DEFERRED_EXPECTED_ROOTS
                or full_rounds <= 0
                or deferred.first_initial_offset is None
                or deferred.last_initial_offset is None
                or deferred.last_final_offset != deferred.last_initial_offset + 3
            ):
                raise ValueError("deferred W3 root/materialization algebra is invalid")
        elif (
            deferred.enabled
            or deferred.validated_rounds != 0
            or deferred.materialized_rounds != 0
            or deferred.validated_roots != 0
            or deferred.submitted_roots != 0
            or deferred.first_initial_offset is not None
            or deferred.last_initial_offset is not None
            or deferred.last_final_offset is not None
        ):
            raise ValueError("disabled deferred W3 receipt contains work")

        events_by_round: dict[int, list[tuple[str, tuple[int, ...]]]] = {
            index: [] for index in range(telemetry.rounds)
        }
        for round_index, event, values in self._causal_events:
            if round_index not in events_by_round:
                raise ValueError("W3 causal event refers to an unknown round")
            events_by_round[round_index].append((event, values))
        if (
            self._decode_begin_cache_offset is None
            or self._expected_anchor_digest is None
            or telemetry.rounds <= 0
            or sum(record[1] for record in self._round_records)
            != telemetry.drafted_tokens
            or sum(record[2] for record in self._round_records)
            != telemetry.accepted_tokens
            or sum(record[3] for record in self._round_records)
            != telemetry.committed_tokens
        ):
            raise ValueError("W3 decode baseline or telemetry totals are invalid")

        previous_cache = self._decode_begin_cache_offset
        prior_prelaunch_attestation: tuple[int, ...] | None = None
        root_batches: list[tuple[int, int, int]] = []
        for record in self._round_records:
            round_index, proposed, accepted, emitted = record[:4]
            prelaunch_submitted = bool(record[6])
            prelaunch_used = bool(record[7])
            actual = events_by_round[round_index]
            if (
                record[4] != 0
                or record[5] != 0
                or record[6] not in {0, 1}
                or record[7] not in {0, 1}
                or record[8] > _PROJECTED_KV_CACHE_MAX_TOKENS
                or (proposed == 2 and not (0 <= accepted <= 2))
                or (proposed == 2 and emitted != accepted + 1)
                or (proposed == 0 and (accepted != 0 or emitted != 1))
                or proposed not in {0, 2}
            ):
                raise ValueError("W3 round acceptance/emission algebra is invalid")
            consumed = accepted + 1 if proposed == 2 else 1
            if record[8] != previous_cache + consumed:
                raise ValueError("W3 target-cache schedule is noncontiguous")
            expected_events: list[tuple[str, tuple[int, ...]]] = []
            if prior_prelaunch_attestation is not None:
                expected_events.append(
                    (
                        "tail_prelaunch_used",
                        prior_prelaunch_attestation,
                    )
                    if prelaunch_used
                    else (
                        "tail_prelaunch_discarded",
                        (1, *prior_prelaunch_attestation),
                    )
                )
            elif prelaunch_used:
                raise ValueError("W3 tail prelaunch was used without a submission")
            if proposed == 2:
                proposal_context = (
                    actual[len(expected_events)]
                    if len(actual) > len(expected_events)
                    else None
                )
                if (
                    proposal_context is None
                    or proposal_context[0] != "proposal_context_agreed"
                    or len(proposal_context[1]) != 1
                    or proposal_context[1][0] != previous_cache
                    or (
                        prelaunch_used
                        and prior_prelaunch_attestation is not None
                        and proposal_context[1][0] != prior_prelaunch_attestation[0]
                    )
                ):
                    raise ValueError("W3 proposal context differs from tail handoff")
                expected_events.append(proposal_context)
                expected_events.append(("proposal_agreed", (3,)))
                if self.selectors.deferred:
                    expected_events.append(
                        ("deferred_roots_validated", (_DEFERRED_EXPECTED_ROOTS,))
                    )
                expected_events.append(("target_graph_agreed", (2,)))
                if self.selectors.deferred:
                    expected_events.extend(
                        (
                            "deferred_root_submitted",
                            (ordinal, _DEFERRED_EXPECTED_ROOTS),
                        )
                        for ordinal in range(_DEFERRED_EXPECTED_ROOTS)
                    )
                    root_batch = (
                        actual[len(expected_events)]
                        if len(actual) > len(expected_events)
                        else None
                    )
                    if (
                        root_batch is None
                        or root_batch[0] != "deferred_roots_submitted"
                        or len(root_batch[1]) != 3
                        or root_batch[1][0] != previous_cache
                        or root_batch[1][1] != root_batch[1][0] + 3
                        or root_batch[1][2] != _DEFERRED_EXPECTED_ROOTS
                        or root_batch[1][1] > _PROJECTED_KV_CACHE_MAX_TOKENS
                    ):
                        raise ValueError("deferred root batch order/offset is invalid")
                    expected_events.append(root_batch)
                    root_batches.append(cast(tuple[int, int, int], root_batch[1]))
                expected_events.append(("acceptance_agreed", (accepted,)))
                committed_event = (
                    actual[len(expected_events)]
                    if len(actual) > len(expected_events)
                    else None
                )
                if (
                    committed_event is None
                    or committed_event[0] != "committed_tokens"
                    or len(committed_event[1]) != 5
                    or committed_event[1][0] != emitted
                ):
                    raise ValueError("W3 committed output anchor is invalid")
                expected_events.append(committed_event)
                committed_anchor_digest = committed_event[1][1:]
                current_prelaunch_attestation: tuple[int, ...] | None = None
                if prelaunch_submitted:
                    submission = (
                        actual[len(expected_events)]
                        if len(actual) > len(expected_events)
                        else None
                    )
                    if (
                        submission is None
                        or submission[0] != "tail_prelaunch_submitted"
                        or len(submission[1]) != 5
                        or submission[1][0] != record[8]
                        or submission[1][1:] != committed_anchor_digest
                    ):
                        raise ValueError("W3 tail-prelaunch submission is unbound")
                    current_prelaunch_attestation = submission[1]
                    expected_events.append(submission)
                expected_events.append(("target_commit", (accepted + 1,)))
            else:
                current_prelaunch_attestation = None
                expected_events.append(("ordinary_anchor_agreed", ()))
                committed_event = (
                    actual[len(expected_events)]
                    if len(actual) > len(expected_events)
                    else None
                )
                if (
                    committed_event is None
                    or committed_event[0] != "committed_tokens"
                    or len(committed_event[1]) != 5
                    or committed_event[1][0] != 1
                ):
                    raise ValueError("ordinary committed output anchor is invalid")
                expected_events.append(committed_event)
                expected_events.append(("ordinary_target_commit", (1,)))
            if (
                round_index == telemetry.rounds - 1
                and current_prelaunch_attestation is not None
            ):
                expected_events.append(
                    (
                        "tail_prelaunch_discarded",
                        (2, *current_prelaunch_attestation),
                    )
                )
            if actual != expected_events:
                raise ValueError("W3 causal event ordering differs from the schedule")
            previous_cache = record[8]
            prior_prelaunch_attestation = current_prelaunch_attestation

        submitted_count = sum(record[6] for record in self._round_records)
        used_count = sum(record[7] for record in self._round_records)
        if (
            submitted_count != self._tail_prelaunch_submitted
            or used_count != self._tail_prelaunch_used
            or submitted_count
            != self._tail_prelaunch_used + self._tail_prelaunch_discarded
            or (
                telemetry.rounds > 1
                and full_rounds > 0
                and (submitted_count == 0 or used_count == 0)
            )
        ):
            raise ValueError("W3 tail-prelaunch causal chain is incomplete")
        if self.selectors.deferred and (
            not root_batches
            or deferred.first_initial_offset != root_batches[0][0]
            or deferred.last_initial_offset != root_batches[-1][0]
            or deferred.last_final_offset != root_batches[-1][1]
        ):
            raise ValueError("deferred W3 aggregate offsets differ from causal roots")

        if (
            self._poisoned
            or self._stale_events != 0
            or self._duplicate_events != 0
            or self._visible_output_tokens <= 0
            or self._visible_output_tokens > telemetry.committed_tokens
            or runtime.target_cache_offset != previous_cache
        ):
            raise ValueError("W3 request-local receipt state is poisoned or incomplete")

        if self.native_api is None or self._native_baseline is None:
            raise RuntimeError("native affine8 Q3 receipt baseline is unavailable")
        native_final = self._native_counts()
        native_delta = tuple(
            final - baseline
            for final, baseline in zip(
                native_final,
                self._native_baseline,
                strict=True,
            )
        )
        if any(value < 0 for value in native_delta):
            raise ValueError("native affine8 Q3 counters decreased")
        if native_delta[3] != counters["packed_width3_hits"]:
            raise ValueError(
                "native affine8 Q3 N10624 dispatches do not equal packed W3 hits"
            )
        q3_sparse_layer_calls = (
            full_rounds + prefill_width3
        ) * _PACKED_FRONT_EXPECTED_LAYERS
        expected_native_total = q3_sparse_layer_calls * (
            3 if self.selectors.packed else 6
        )
        expected_native_other = q3_sparse_layer_calls * (
            2 if self.selectors.packed else 6
        )
        native_other = native_delta[0] - sum(native_delta[1:])
        if (
            native_delta[0] != expected_native_total
            or native_delta[1] != 0
            or native_delta[2] != 0
            or native_other != expected_native_other
        ):
            raise ValueError("native affine8 Q3 aggregate/other algebra is invalid")

        def digest_fields(
            prefix: str,
            words: tuple[int, int, int, int],
        ) -> dict[str, int]:
            return {f"{prefix}_word_{index}": word for index, word in enumerate(words)}

        deferred_offsets = (
            deferred.first_initial_offset or 0,
            deferred.last_initial_offset or 0,
            deferred.last_final_offset or 0,
        )
        marker: dict[str, object] = {
            "receipt_schema_version": 1,
            "request_phase": self.phase,
            "request_sequence": self._handle[0],
            "request_token": self._handle[1],
            "finalized": True,
            "diagnostic_only": True,
            "rank_agreed": True,
            "arm_code": self.selectors.arm_code,
            "canonical_arm": self.selectors.canonical,
            "packed_enabled": self.selectors.packed,
            "kda_enabled": self.selectors.kda,
            "deferred_enabled": self.selectors.deferred,
            "native_triplet_enabled": True,
            "tail_overlap_enabled": True,
            "expected_sparse_layers": _PACKED_FRONT_EXPECTED_LAYERS,
            "expected_kda_layers": _KDA_EXPECTED_LAYERS,
            "expected_deferred_roots_per_round": _DEFERRED_EXPECTED_ROOTS,
            "projected_kv_cache_max_tokens": _PROJECTED_KV_CACHE_MAX_TOKENS,
            **digest_fields("selector_digest", self.selector_digest),
            **digest_fields(
                "launch_contract_digest",
                self.launch_contract_digest,
            ),
            **digest_fields("setup_digest", self.setup_digest),
            **digest_fields("source_digest", self.api.source_digest),
            **digest_fields("exo_source_digest", self.exo_source_digest),
            **digest_fields("native_lib_digest", self.native_api.lib_digest),
            **{name: counters[name] for name in _PACKED_FRONT_COUNTER_FIELDS},
            "prefill_width1_chunks": prefill_width1,
            "prefill_width3_chunks": prefill_width3,
            "prefill_noncontract_chunks": prefill_noncontract,
            "target_width1_rounds": tail_rounds,
            "speculative_full_width_rounds": full_rounds,
            "proposed_tokens": telemetry.drafted_tokens,
            "accepted_tokens": telemetry.accepted_tokens,
            "emitted_tokens": telemetry.committed_tokens,
            "visible_output_tokens": self._visible_output_tokens,
            "decode_begin_cache_offset": self._decode_begin_cache_offset,
            "fallback_rounds": 0,
            "error_rounds": 0,
            "receipt_poisoned": False,
            "stale_events": 0,
            "duplicate_events": 0,
            "deferred_validated_rounds": deferred.validated_rounds,
            "deferred_materialized_rounds": deferred.materialized_rounds,
            "deferred_validated_roots": deferred.validated_roots,
            "deferred_submitted_roots": deferred.submitted_roots,
            "deferred_first_initial_offset": deferred_offsets[0],
            "deferred_last_initial_offset": deferred_offsets[1],
            "deferred_last_final_offset": deferred_offsets[2],
            "tail_prelaunch_submitted": self._tail_prelaunch_submitted,
            "tail_prelaunch_used": self._tail_prelaunch_used,
            "tail_prelaunch_discarded": self._tail_prelaunch_discarded,
            "native_q3_total": native_delta[0],
            "native_q3_n4480": native_delta[1],
            "native_q3_n6144": native_delta[2],
            "native_q3_n10624": native_delta[3],
            "native_q3_other": native_other,
            **digest_fields("schedule_digest", self._schedule_digest.words()),
            **digest_fields("proposal_digest", self._proposal_digest.words()),
            **digest_fields("acceptance_digest", self._acceptance_digest.words()),
            **digest_fields(
                "committed_output_digest",
                self._committed_output_digest.words(),
            ),
            **digest_fields(
                "visible_output_digest",
                self._visible_output_digest.words(),
            ),
            **digest_fields("cache_digest", self._cache_digest.words()),
            **digest_fields("causal_digest", self._causal_digest.words()),
        }
        if any(type(value) not in {bool, int} for value in marker.values()):
            raise TypeError("W3 composition marker must contain numeric scalars")
        return marker

    def finish(
        self,
        runtime: KimiK3DSparkRequestRuntime,
        telemetry: "_PromptLookupTelemetry",
    ) -> K3W3CompositionReceipt:
        if not self.enabled or self.api is None or self._handle is None:
            raise RuntimeError("packed-front receipt finish state is invalid")

        def finish_local() -> str:
            assert self.api is not None
            assert self._handle is not None
            self._assert_runtime_identity()
            raw = self.api.finish(*self._handle, self.model)
            if not isinstance(raw, dict):
                raise TypeError("MLX-LM packed-front receipt must be a mapping")
            typed_raw: dict[str, object] = {}
            for name, value in cast(dict[object, object], raw).items():
                if type(name) is not str:
                    raise TypeError("MLX-LM packed-front receipt keys must be strings")
                typed_raw[name] = value
            marker = self._validated_marker(typed_raw, telemetry, runtime)
            return _canonical_numeric_json(marker)

        agreed_marker = runtime.agree_text("packed-front receipt", finish_local)
        marker_payload = _parse_canonical_numeric_json(agreed_marker)
        receipt = K3W3CompositionReceipt.model_validate(
            {
                "schema": _PACKED_FRONT_RECEIPT_SCHEMA,
                **marker_payload,
            }
        )
        if receipt.model_dump(exclude={"receipt_schema"}) != marker_payload:
            raise ValueError("packed-front receipt marker changed during validation")
        self._handle = None
        runtime.attach_composition_receipt_sink(None)
        self._runtime = None
        self._receipt = receipt
        self._marker = agreed_marker
        return receipt

    def verify_marker_publication(self) -> None:
        if self._receipt is None or self._marker is None or self._published:
            raise RuntimeError("packed-front receipt marker state is invalid")
        self._assert_runtime_identity()

    def emit_marker(self, *, rank_zero: bool) -> None:
        self.verify_marker_publication()
        if rank_zero:
            logger.info(f"K3_W3_COMPOSITION_RECEIPT {self._marker}")
        logger.complete()

    def mark_marker_published(self) -> None:
        if self._receipt is None or self._marker is None or self._published:
            raise RuntimeError("packed-front receipt marker state is invalid")
        _record_composition_receipt_publication(
            self.phase,
            self._phase_identity(),
        )
        self._published = True
        self._close_native_fd()

    def abort(self) -> None:
        handle = self._handle
        self._handle = None
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            with contextlib.suppress(Exception):
                runtime.attach_composition_receipt_sink(None)
        _release_composition_receipt_phase(self.phase)
        self._close_native_fd()
        if handle is None or self.api is None:
            return
        # The pinned MLX-LM abort clears its context in a finally block.
        # Generator teardown has no later target collective to protect.
        with contextlib.suppress(Exception):
            self.api.abort(*handle)


def _publish_composition_receipt(
    runtime: KimiK3DSparkRequestRuntime,
    receipt: _PackedFrontReceiptRequest,
) -> None:
    """Rank-agree every external marker and lifecycle side effect in order."""

    runtime.agree_local_side_effect(
        "W3 composition marker prepublication verification",
        receipt.verify_marker_publication,
    )
    runtime.agree_local_side_effect(
        "W3 composition marker publication",
        lambda: receipt.emit_marker(rank_zero=runtime.collective.rank == 0),
    )
    runtime.agree_local_side_effect(
        "W3 composition marker lifecycle completion",
        receipt.mark_marker_published,
    )


_DSPARK_ORDINARY_AFTER_CONTEXT_ENV = "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_AFTER_CONTEXT"
_DSPARK_FORCE_ORDINARY_ENV = "EXO_MLX_KIMI_K3_DSPARK_FORCE_ORDINARY"
_DSPARK_MAX_ORDINARY_AFTER_CONTEXT = 1_048_576


def _dspark_ordinary_after_context() -> int:
    """Parse the optional request-level ordinary-decode cutoff.

    Unset is the default-off state.  Once explicitly configured, zero and all
    non-positive/out-of-range values are rejected before any request state or
    TP graph is built.
    """

    raw = os.environ.get(_DSPARK_ORDINARY_AFTER_CONTEXT_ENV)
    if raw is None:
        return 0
    return _strict_env_int(
        _DSPARK_ORDINARY_AFTER_CONTEXT_ENV,
        raw,
        minimum=1,
        maximum=_DSPARK_MAX_ORDINARY_AFTER_CONTEXT,
    )


def _dspark_force_ordinary(prompt_tokens: int) -> bool:
    """Select target-only decode at/after the opt-in prompt cutoff."""

    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("Kimi K3 DSpark prompt token count must be non-negative")
    forced = _strict_env_flag(
        _DSPARK_FORCE_ORDINARY_ENV,
        os.environ.get(_DSPARK_FORCE_ORDINARY_ENV, "0"),
    )
    cutoff = _dspark_ordinary_after_context()
    return forced or (cutoff != 0 and prompt_tokens >= cutoff)


def greedy_vocab_parallel_stream_kwargs(
    *,
    temperature: float,
    logprobs: bool,
    has_logits_processors: bool,
    is_pipeline: bool,
    speculative: bool,
) -> _GreedyVocabParallelStreamKwargs:
    """Enable compact TP argmax only for the exact request shape it supports."""

    enabled = _strict_env_flag(
        "EXO_MLX_K3_VOCAB_PARALLEL_GREEDY",
        os.environ.get("EXO_MLX_K3_VOCAB_PARALLEL_GREEDY", "0"),
    )
    if (
        not enabled
        or temperature != 0.0
        or logprobs
        or has_logits_processors
        or is_pipeline
        or speculative
    ):
        return {}
    return {"greedy_vocab_parallel_no_logprobs": True}


def prompt_lookup_config(
    *,
    is_pipeline: bool,
    is_batch: bool,
) -> _PromptLookupConfig | None:
    """Parse the opt-in MLX-LM prompt-lookup configuration.

    Companion settings without the enabling token count are rejected so a
    misspelled or incomplete deployment does not silently fall back to normal
    decode. The upper draft bound matches Kimi K3's transactional target-cache
    implementation (draft + verification token <= 8).
    """
    num_tokens_raw = os.environ.get("EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS")
    max_ngram_raw = os.environ.get("EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE")
    telemetry_raw = os.environ.get("EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY")

    if num_tokens_raw is None:
        configured_companions = [
            name
            for name, value in (
                ("EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE", max_ngram_raw),
                ("EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY", telemetry_raw),
            )
            if value is not None
        ]
        if configured_companions:
            raise ValueError(
                f"{', '.join(configured_companions)} requires "
                "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS"
            )
        return None

    num_tokens = _strict_env_int(
        "EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS",
        num_tokens_raw,
        minimum=1,
        maximum=7,
    )
    max_ngram_size = _strict_env_int(
        "EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE",
        max_ngram_raw if max_ngram_raw is not None else "4",
        minimum=2,
        maximum=64,
    )
    round_telemetry = _strict_env_flag(
        "EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY",
        telemetry_raw if telemetry_raw is not None else "0",
    )

    if is_pipeline:
        raise ValueError(
            "MLX prompt-lookup decoding does not support pipeline parallelism"
        )
    if is_batch:
        raise ValueError("MLX prompt-lookup decoding does not support batch generation")

    return _PromptLookupConfig(
        num_tokens=num_tokens,
        max_ngram_size=max_ngram_size,
        round_telemetry=round_telemetry,
    )


def _prompt_lookup_round_callback(
    group: mx.distributed.Group | None,
    *,
    telemetry: _PromptLookupTelemetry | None = None,
    log_rounds: bool = True,
) -> Callable[[_SpeculativeRoundStatsLike], None]:
    rank = group.rank() if group is not None else 0

    def report(stats: _SpeculativeRoundStatsLike) -> None:
        if telemetry is not None:
            telemetry.observe(stats)
        if log_rounds:
            logger.info(
                "MLX prompt-lookup round: "
                f"rank={rank}, "
                f"round={stats.round_index}, "
                f"source={stats.source}, "
                f"drafted={stats.drafted_tokens}, "
                f"accepted={stats.accepted_tokens}, "
                f"committed={stats.committed_tokens}, "
                f"target_cache={stats.target_cache_tokens}, "
                f"cancelled={stats.cancelled}"
            )

    return report


def prompt_lookup_stream_kwargs(
    config: _PromptLookupConfig | None,
    group: mx.distributed.Group | None,
    history: mx.array,
    telemetry: _PromptLookupTelemetry | None = None,
) -> _PromptLookupStreamKwargs | None:
    if config is None:
        return None
    if len(history) < 2:
        raise ValueError("MLX prompt-lookup decoding requires at least two tokens")

    return {
        "prompt_lookup_num_tokens": config.num_tokens,
        "prompt_lookup_max_ngram_size": config.max_ngram_size,
        # EXO pre-fills its cache directly, then gives stream_generate only the
        # final two-token decode boundary. Seed lookup from the complete logical
        # history without asking MLX-LM to process those tokens a second time.
        "prompt_lookup_history": history,
        "speculative_round_callback": (
            _prompt_lookup_round_callback(
                group,
                telemetry=telemetry,
                log_rounds=config.round_telemetry,
            )
            if telemetry is not None or config.round_telemetry
            else None
        ),
    }


def _prefill_step_size(
    num_tokens: int,
    *,
    is_pipeline: bool,
    pipeline_divisor: int = 1,
) -> int:
    """Choose the nominal MLX-LM prefill step.

    Pipeline prefill divides this value by ``pipeline_divisor`` before building
    real chunks; tensor prefill uses it directly. Keeping the public override
    nominal preserves MLX-LM's existing ``prefill_step_size`` semantics while
    the attention-cell budget bounds the effective per-rank
    ``query_chunk * context_length`` workspace.
    """
    base_step = int(os.environ.get("EXO_MLX_PREFILL_STEP_SIZE", "4096"))
    if base_step < 4:
        raise ValueError("EXO_MLX_PREFILL_STEP_SIZE must be at least 4")
    long_context_min = int(
        os.environ.get("EXO_MLX_PIPELINE_LONG_CONTEXT_MIN_TOKENS", "4096")
    )
    long_context_step = int(
        os.environ.get(
            "EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE",
            str(base_step),
        )
    )
    if long_context_min < 1:
        raise ValueError("EXO_MLX_PIPELINE_LONG_CONTEXT_MIN_TOKENS must be positive")
    if long_context_step < 4:
        raise ValueError("EXO_MLX_PIPELINE_LONG_CONTEXT_STEP_SIZE must be at least 4")
    if num_tokens < long_context_min:
        return base_step

    max_attention_cells = int(
        os.environ.get(
            "EXO_MLX_MAX_ATTENTION_CELLS_PER_CHUNK",
            os.environ.get(
                "EXO_MLX_PIPELINE_MAX_ATTENTION_CELLS_PER_CHUNK",
                "0",
            ),
        )
    )
    if max_attention_cells < 0:
        raise ValueError("EXO_MLX_MAX_ATTENTION_CELLS_PER_CHUNK cannot be negative")
    if max_attention_cells == 0:
        return long_context_step

    pipeline_divisor = max(1, pipeline_divisor if is_pipeline else 1)
    max_query_chunk = max(1, max_attention_cells // num_tokens)
    # Power-of-two chunks give stable Metal kernel shapes and make benchmark
    # comparisons reproducible as the context crosses a budget boundary.
    query_chunk = 1 << (max_query_chunk.bit_length() - 1)
    bounded_step = max(4, query_chunk * pipeline_divisor)
    return min(long_context_step, bounded_step)


def _prefill_memory_log_interval() -> int:
    """Return the optional chunk interval for synchronized MLX memory traces."""
    interval = int(os.environ.get("EXO_MLX_PREFILL_MEMORY_LOG_INTERVAL", "0"))
    if interval < 0:
        raise ValueError("EXO_MLX_PREFILL_MEMORY_LOG_INTERVAL cannot be negative")
    return interval


def _memory_from_mlx_decimal_gb(value: float) -> Memory:
    """Convert MLX-LM's decimal-gigabyte metric back to exact bytes."""
    return Memory.from_bytes(round(value * 1e9))


@contextlib.contextmanager
def patch_embed_tokens(
    model: Model,
    embeddings: mx.array,
    start_offset: int = 0,
    token_count: int = 0,
    image_token_id: int | None = None,
) -> Generator[None]:
    inner = get_inner_model(model)  # type: ignore
    original_embed = inner.embed_tokens  # type: ignore
    end_offset = start_offset + token_count
    offset = [start_offset]

    def _inject(input_ids: mx.array) -> mx.array:
        chunk_start = offset[0]
        chunk_len = input_ids.shape[-1]
        chunk_end = chunk_start + chunk_len
        offset[0] = chunk_end

        # The injection window is [start_offset, end_offset).
        if chunk_end <= start_offset or chunk_start >= end_offset:
            return original_embed(input_ids)  # type: ignore

        # Mixed chunk: splice the pre-computed embeddings for the overlap
        # into `original_embed(input_ids)` for any text-only fringes.
        overlap_start = max(chunk_start, start_offset)
        overlap_end = min(chunk_end, end_offset)
        dst_start = overlap_start - chunk_start
        dst_end = overlap_end - chunk_start
        text_embeds: mx.array = original_embed(input_ids)  # type: ignore
        return mx.concatenate(
            [
                text_embeds[:, :dst_start, :],
                embeddings[:, overlap_start:overlap_end, :],
                text_embeds[:, dst_end:, :],
            ],
            axis=1,
        )

    for attr in dir(original_embed):  # type: ignore
        if not attr.startswith("_") and not hasattr(_inject, attr):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(_inject, attr, getattr(original_embed, attr))  # type: ignore

    inner.embed_tokens = _inject

    # Gemma 4 (e2b/e4b) has a second, independent embedding table that produces
    # per-layer conditioning signals via self.embed_tokens_per_layer(input_ids).
    # The injected vision embeddings live in the main residual stream only, so
    # if image_token_id positions are passed through as-is the per-layer table
    # produces garbage signals at those positions (the `<image>` token was never
    # trained to have meaningful per-layer inputs).
    original_per_layer = getattr(inner, "embed_tokens_per_layer", None)  # type: ignore
    if original_per_layer is not None and image_token_id is not None:

        def _clean_per_layer(input_ids: mx.array) -> mx.array:
            clean_ids = mx.where(
                input_ids == image_token_id, mx.zeros_like(input_ids), input_ids
            )
            return original_per_layer(clean_ids)  # type: ignore

        inner.embed_tokens_per_layer = _clean_per_layer

    try:
        yield
    finally:
        inner.embed_tokens = original_embed
        if original_per_layer is not None and image_token_id is not None:
            inner.embed_tokens_per_layer = original_per_layer


class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


def _has_pipeline_communication_layer(model: Model):
    for layer in model.layers:
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            return True
    return False


def _is_kimi_k3_model(model: Model) -> bool:
    """Identify the upstream Kimi K3 model without widening other MLX paths."""
    return type(model).__module__ == "mlx_lm.models.kimi_k3"


@contextlib.contextmanager
def _pipeline_token_relay_scope(
    model: Model,
    enabled: bool,
) -> Generator[None]:
    """Restore token-relay state on every exit, including generator close."""
    try:
        if enabled:
            set_pipeline_token_relay(model, True)
        yield
    finally:
        if enabled:
            set_pipeline_token_relay(model, False)


def pipeline_parallel_prefill(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
    distributed_prompt_progress_callback: Callable[[], None] | None,
    group: mx.distributed.Group,
) -> None:
    """Prefill the KV cache for pipeline parallel with overlapping stages.

    Each rank processes the full prompt through its real cache, offset by leading
    and trailing dummy iterations.

    Total iterations per rank = N_real_chunks + world_size - 1:
      - rank r leading dummies  (skip_pipeline_io, throwaway cache)
      - N_real_chunks real      (pipeline IO active, real cache)
      - (world_size-1-r) trailing dummies (skip_pipeline_io, throwaway cache)

    e.g.
    Timeline (2 ranks, 3 chunks of 10240 tokens @ step=4096):
        iter 0: R0 real[0:4096]     R1 dummy
        iter 1: R0 real[4096:8192]  R1 real[0:4096]
        iter 2: R0 real[8192:10240] R1 real[4096:8192]
        iter 3: R0 dummy            R1 real[8192:10240]

    This function is designed to match mlx_lm's stream_generate exactly in terms of
    side effects (given the same prefill step size)
    """
    prefill_step_size = prefill_step_size // min(4, group.size())

    quantize_cache_fn: Callable[..., None] = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    _prompt_cache: KVCacheType = prompt_cache
    rank = group.rank()
    world_size = group.size()

    # Build list of real prompt chunk sizes
    total = len(prompt)
    real_chunk_sizes: list[int] = []
    remaining = total - 1
    while remaining:
        n = min(prefill_step_size, remaining)
        real_chunk_sizes.append(n)
        remaining -= n
    n_real = len(real_chunk_sizes)

    # Each rank does: [rank leading dummies] [N real chunks] [world_size-1-rank trailing dummies]
    n_leading = rank
    n_trailing = world_size - 1 - rank
    n_total = n_leading + n_real + n_trailing

    memory_log_interval = _prefill_memory_log_interval()
    t_start = time.perf_counter()
    processed = 0
    logger.info(
        f"[R{rank}] Pipeline prefill: {n_real} real + {n_leading} leading + {n_trailing} trailing = {n_total} iterations"
    )
    clear_prefill_sends()

    # Initial callback matching generate_step
    prompt_progress_callback(0, total)

    try:
        with mx.stream(generation_stream):
            for _ in range(n_leading):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                model(
                    prompt[processed : processed + chunk_size][None],
                    cache=_prompt_cache,
                )
                quantize_cache_fn(_prompt_cache)
                processed += chunk_size

                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

                flush_prefill_sends()

                prompt_progress_callback(processed, total)
                if memory_log_interval and (
                    (i + 1) % memory_log_interval == 0 or i + 1 == n_real
                ):
                    mx.synchronize()
                    logger.info(
                        f"[R{rank}] Prefill memory after chunk {i + 1}/{n_real}: "
                        f"processed={processed}, chunk_size={chunk_size}, "
                        f"active_bytes={mx.get_active_memory()}, "
                        f"cache_bytes={mx.get_cache_memory()}, "
                        f"peak_bytes={mx.get_peak_memory()}"
                    )

            for _ in range(n_trailing):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

        if not _is_kimi_k3_model(model):
            # Legacy pipeline prefill generates two disposable cache entries,
            # which prefill() rolls back before decode. K3's non-trimmable
            # ArraysCache makes that rollback require detached host snapshots.
            # Its real loop already leaves the cache at this function's
            # prompt[:-1] (the full prompt[:-2]), exactly where
            # stream_generate(prompt=full_prompt[-2:]) expects it.
            for _ in range(2):
                with mx.stream(generation_stream):
                    model(prompt[-1:][None], cache=_prompt_cache)
                    quantize_cache_fn(_prompt_cache)
                flush_prefill_sends()

        assert _prompt_cache is not None
        with mx.stream(generation_stream):
            mx.eval([c.state for c in _prompt_cache])  # type: ignore
    except PrefillCancelled:
        # Cancellation is agreed by every rank before the receiver posts the
        # next prefill receive. The queued frame is therefore still wholly
        # local and can be discarded without invalidating transport sequence
        # numbers. Unknown failures continue through the poisoning cleanup.
        discard_unsent_prefill_sends_after_agreed_cancel()
        raise
    finally:
        clear_prefill_sends()

    # Final callback matching generate_step
    prompt_progress_callback(total, total)

    logger.info(
        f"[R{rank}] Prefill: {n_real} real + {n_leading}+{n_trailing} dummy iterations, "
        f"Processed {processed} tokens in {(time.perf_counter() - t_start) * 1000:.1f}ms"
    )


def kimi_k3_exact_prefill(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
) -> None:
    """Fill a non-pipeline K3 cache exactly to the decode boundary.

    ``prefill()`` receives the remaining full prompt without its final token.
    Decode subsequently starts from that argument's final two tokens, so K3
    must cache exactly ``prompt[:-1]``.  Calling ``stream_generate`` would
    process the disposable final token and require hundreds of MiB of detached
    KDA state per rollback checkpoint.  This direct loop reaches the same
    boundary without sampling, trimming, restoring, or retaining snapshots.
    """
    quantize_cache_fn: Callable[..., None] = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )
    total = len(prompt)
    processed = 0
    memory_log_interval = _prefill_memory_log_interval()
    prompt_progress_callback(0, total)

    with mx.stream(generation_stream):
        while total - processed > 1:
            remaining = (total - processed) - 1
            chunk_size = min(prefill_step_size, remaining)
            model(
                prompt[processed : processed + chunk_size][None],
                cache=prompt_cache,
            )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])  # type: ignore
            processed += chunk_size
            prompt_progress_callback(processed, total)
            mx.clear_cache()

            chunk_index = math.ceil(processed / prefill_step_size)
            if memory_log_interval and (
                chunk_index % memory_log_interval == 0 or total - processed <= 1
            ):
                mx.synchronize()
                logger.info(
                    f"Kimi K3 exact-prefill memory after chunk {chunk_index}: "
                    f"processed={processed}/{total - 1}, "
                    f"chunk_size={chunk_size}, "
                    f"active_bytes={mx.get_active_memory()}, "
                    f"cache_bytes={mx.get_cache_memory()}, "
                    f"peak_bytes={mx.get_peak_memory()}"
                )

    prompt_progress_callback(total, total)


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache, then
    trims off the extra generated token. Kimi K3 prefill stops at the decode
    boundary directly and therefore needs neither snapshots nor trim.

    Returns:
        (tokens_per_sec, num_tokens, snapshots)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    is_pipeline = _has_pipeline_communication_layer(model)
    rollback_free_k3 = _is_kimi_k3_model(model)
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm and not rollback_free_k3:
            snapshots.append(snapshot_ssm_states(cache))

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    def combined_progress_callback(processed: int, total: int) -> None:
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    set_pipeline_prefill(model, is_prefill=True)
    try:
        mx_barrier(group)
        logger.info("Starting prefill")

        pipeline_divisor = (
            min(4, group.size()) if is_pipeline and group is not None else 1
        )
        prefill_step_size = _prefill_step_size(
            num_tokens,
            is_pipeline=is_pipeline,
            pipeline_divisor=pipeline_divisor,
        )
        effective_chunk_size = (
            prefill_step_size // pipeline_divisor if is_pipeline else prefill_step_size
        )
        logger.info(
            f"Prefill step size: nominal={prefill_step_size}, "
            f"effective_chunk_size={effective_chunk_size}, tokens={num_tokens}"
        )

        if is_pipeline:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            pipeline_parallel_prefill(
                model=model,
                prompt=prompt_tokens,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                prompt_progress_callback=progress_callback,
                distributed_prompt_progress_callback=distributed_prompt_progress_callback,
                group=group,
            )
        elif rollback_free_k3:
            kimi_k3_exact_prefill(
                model=model,
                prompt=prompt_tokens,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                prompt_progress_callback=combined_progress_callback,
            )
        else:
            # Use max_tokens=1 because max_tokens=0 does not work.
            # We just throw away the generated token - we only care about filling the cache
            for _ in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=1,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                prompt_progress_callback=combined_progress_callback,
            ):
                break  # Stop after first iteration - cache is now filled
    finally:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)

    if not rollback_free_k3:
        # stream_generate added 1 extra generated token to the cache, so we should trim it.
        # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
        pre_gen = snapshots[-2] if has_ssm else None
        for i, c in enumerate(cache):
            non_trimmable = is_non_trimmable_cache_entry(c)
            if has_ssm and non_trimmable:
                assert pre_gen is not None
                restored = copy_snapshot_entry(pre_gen.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            else:
                assert not non_trimmable
                c.trim(2)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Exclude the last snapshot
    return tokens_per_sec, num_tokens, snapshots[:-1] if snapshots else []


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
    dspark: LoadedKimiK3DSpark | None = None,
) -> int:
    logger.info(f"warming up inference for instance: {model_id}")

    def prepare_warmup() -> tuple[TextGenerationTaskParams, str, int, int, bool]:
        content = InputMessageContent(
            "Prompt to warm up the inference engine. Repeat this."
        )
        default_warmup_tokens = (
            4 if model_id == ModelId("kernelpool/Kimi-K3-2bit-UVMAX") else 50
        )
        try:
            warmup_tokens = int(
                os.environ.get(
                    "EXO_MLX_WARMUP_OUTPUT_TOKENS",
                    str(default_warmup_tokens),
                )
            )
        except ValueError as error:
            raise ValueError(
                "EXO_MLX_WARMUP_OUTPUT_TOKENS must be an integer"
            ) from error
        if not 1 <= warmup_tokens <= 256:
            raise ValueError("EXO_MLX_WARMUP_OUTPUT_TOKENS must be between 1 and 256")
        verify_width = 0
        force_ordinary = False
        if dspark is not None:
            verify_width = dspark.verify_width
            warmup_tokens = max(warmup_tokens, 2 * verify_width)
            force_ordinary = _strict_env_flag(
                "EXO_MLX_KIMI_K3_DSPARK_FORCE_ORDINARY",
                os.environ.get("EXO_MLX_KIMI_K3_DSPARK_FORCE_ORDINARY", "0"),
            )

        task = TextGenerationTaskParams(
            model=model_id,
            input=[InputMessage(role="user", content=content)],
            max_output_tokens=warmup_tokens,
            temperature=0.0,
            bench=dspark is not None,
        )
        prompt = apply_chat_template(
            tokenizer=tokenizer,
            task_params=task,
        )
        return task, prompt, warmup_tokens, verify_width, force_ordinary

    def warmup_contract(
        setup: tuple[TextGenerationTaskParams, str, int, int, bool],
    ) -> str:
        _task, prompt, warmup_tokens, verify_width, force_ordinary = setup
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return (
            f"{model_id}\0{int(dspark is not None)}\0{verify_width}\0"
            f"{warmup_tokens}\0{int(force_ordinary)}\0{prompt_digest}"
        )

    (
        warmup_task_params,
        warmup_prompt,
        _warmup_tokens,
        verify_width,
        force_ordinary,
    ) = rank_agreed_local_stage(
        "inference warmup setup",
        group,
        prepare_warmup,
        warmup_contract,
    )

    tokens_generated = 0
    final_stats: GenerationStats | None = None
    composition_warmup_requested = (
        os.environ.get(_PACKED_FRONT_DIAGNOSTIC_ENV, "0") == "1"
    )

    mx_barrier(group)

    logger.info("Generating warmup tokens")

    t = time.monotonic()

    for response in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=warmup_task_params,
        prompt=warmup_prompt,
        kv_prefix_cache=None,
        group=group,
        dspark=dspark,
        width4_receipt_scope="warmup",
        _packed_front_receipt_phase="engine_startup",
    ):
        tokens_generated += 1
        if response.stats is not None:
            final_stats = response.stats

    def validate_warmup() -> tuple[int, tuple[int, ...]]:
        if dspark is not None:
            if tokens_generated < 2 * verify_width or final_stats is None:
                raise RuntimeError(
                    "Kimi K3 DSpark warmup did not complete its two-round output budget"
                )
            if force_ordinary:
                if (
                    final_stats.speculative_rounds != tokens_generated
                    or final_stats.speculative_drafted_tokens != 0
                    or final_stats.speculative_accepted_tokens != 0
                    or final_stats.speculative_committed_tokens != tokens_generated
                    or final_stats.speculative_fallback_rounds != 0
                    or final_stats.speculative_error_rounds != 0
                ):
                    raise RuntimeError(
                        "Kimi K3 DSpark aligned-control warmup did not remain "
                        "strictly target-only"
                    )
                counters = (
                    tokens_generated,
                    verify_width,
                    final_stats.speculative_rounds,
                    final_stats.speculative_drafted_tokens,
                    final_stats.speculative_accepted_tokens,
                    final_stats.speculative_committed_tokens,
                    final_stats.speculative_fallback_rounds,
                    final_stats.speculative_error_rounds,
                )
            else:
                required_drafted = 2 * (verify_width - 1)
                if (
                    final_stats.speculative_rounds < 2
                    or final_stats.speculative_drafted_tokens < required_drafted
                    or final_stats.speculative_fallback_rounds != 0
                    or final_stats.speculative_error_rounds != 0
                ):
                    raise RuntimeError(
                        "Kimi K3 DSpark warmup did not complete two clean "
                        "speculative rounds"
                    )
                counters = (
                    tokens_generated,
                    verify_width,
                    final_stats.speculative_rounds,
                    final_stats.speculative_drafted_tokens,
                    final_stats.speculative_fallback_rounds,
                    final_stats.speculative_error_rounds,
                )
        else:
            counters = (tokens_generated, 0)
        if composition_warmup_requested:
            receipt = (
                None if final_stats is None else final_stats.k3_w3_composition_receipt
            )
            if (
                receipt is None
                or receipt.request_phase != _PACKED_FRONT_PHASE_ENGINE_STARTUP
            ):
                raise RuntimeError(
                    "W3 composition warmup has no finalized startup receipt"
                )
        elapsed = max(time.monotonic() - t, 0.001)
        cadence = min(math.ceil(tokens_generated / elapsed), 100)
        return cadence, counters

    check_for_cancel_every, _warmup_counters = rank_agreed_local_stage(
        "inference warmup validation",
        group,
        validate_warmup,
        lambda validated: repr(validated[1]),
    )

    mx_barrier(group)

    if composition_warmup_requested:
        rank_agreed_local_stage(
            "W3 composition startup lifecycle completion",
            group,
            _complete_composition_startup_after_warmup,
            lambda _completed: "completed",
        )

    if dspark is not None and width4_receipt_log_enabled():
        post_warmup_reset = rank_agreed_local_stage(
            "width-four receipt post-warmup reset",
            group,
            lambda: reset_width4_dispatch_counters_after_warmup(
                rank=group.rank() if group is not None else 0,
                world_size=group.size() if group is not None else 1,
                verify_width=verify_width,
                model_id=str(model_id),
                mx_module=mx,
            ),
            lambda reset: (
                reset.agreement_contract() if reset is not None else "disabled"
            ),
        )
        if post_warmup_reset is None:
            raise RuntimeError("enabled width-four receipt reset returned disabled")

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    if group is not None:
        check_for_cancel_every = int(
            mx.max(
                mx.distributed.all_gather(
                    mx.array([check_for_cancel_every]),
                    group=group,
                )
            ).item()
        )

    logger.info(
        f"runner checking for cancellation every {check_for_cancel_every} tokens"
    )

    return check_for_cancel_every


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def _validate_dspark_request(
    dspark: LoadedKimiK3DSpark,
    *,
    model: Model,
    task: TextGenerationTaskParams,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None,
    vision_processor: VisionProcessor | None,
    is_pipeline: bool,
    prompt_lookup_configuration: _PromptLookupConfig | None,
) -> None:
    """Fail closed outside the first sequential, fresh-cache, greedy TP2 canary."""

    if is_pipeline:
        raise ValueError("Kimi K3 DSpark does not support pipeline parallelism")
    if group is None or group.size() != 2:
        raise ValueError("Kimi K3 DSpark requires exactly two tensor ranks")
    if dspark.target_model is not model:
        raise ValueError("Kimi K3 DSpark is bound to a different target model")
    if type(model).__module__ != "mlx_lm.models.kimi_k3":
        raise ValueError("Kimi K3 DSpark requires the exact MLX-LM Kimi K3 target")
    if kv_prefix_cache is not None or task.use_prefix_cache:
        raise ValueError("Kimi K3 DSpark requires a fresh per-request target cache")
    if task.prefill_endpoint is not None:
        raise ValueError("Kimi K3 DSpark does not support remote prefill")
    if vision_processor is not None or task.images or task.image_hashes:
        raise ValueError("Kimi K3 DSpark does not support vision requests")
    if KV_BITS is not None or KV_CACHE_BITS is not None:
        raise ValueError("Kimi K3 DSpark does not support quantized KV caches")
    if prompt_lookup_configuration is not None:
        raise ValueError("Kimi K3 DSpark cannot be combined with prompt lookup")
    validate_dspark_greedy_sampling(
        temperature=task.temperature,
        top_p=task.top_p,
        top_k=task.top_k,
        min_p=task.min_p,
        logprobs=task.logprobs,
        top_logprobs=task.top_logprobs,
        repetition_penalty=task.repetition_penalty,
        repetition_context_size=task.repetition_context_size,
        presence_penalty=task.presence_penalty,
        frequency_penalty=task.frequency_penalty,
    )


def _dspark_setup_fingerprint(
    *,
    prompt_tokens: mx.array,
    max_tokens: int,
    prefill_step_size: int,
    capacity_hint: int,
    verify_width: int,
    seed: int,
    is_bench: bool,
    compact_greedy: bool,
    generation_progress: bool,
    force_ordinary: bool = False,
    ordinary_after_context: int = 0,
    packed_agreements: bool = False,
    deferred_async_width3: bool = False,
    authoritative_packed_width3: bool = False,
    w3_prework_history: bool = False,
    native_packed_q3: bool = False,
    eos_token_ids: tuple[int, ...],
    banned_token_ids: tuple[int, ...],
    terminal_token_ids: tuple[int, ...],
    stop_sequences: tuple[str, ...],
    confidence_capture: DSparkConfidenceCaptureConfig | None = None,
    target_route_top_k: int | None = None,
    proposer_selection: KimiK3DSparkProposerSelection | None = None,
    receipt_session_id: str | None = None,
    receipt_model_id: str | None = None,
) -> tuple[int, ...]:
    """Bind every local choice that can alter DSpark graph or loop ordering."""

    tolist = getattr(prompt_tokens, "tolist", None)
    if not callable(tolist):
        raise TypeError("Kimi K3 DSpark prompt tokens must be an MLX array")
    raw_prompt_tokens: object = tolist()
    if not isinstance(raw_prompt_tokens, list):
        raise ValueError("Kimi K3 DSpark prompt tokens must be one-dimensional")
    raw_prompt_values = cast(list[object], raw_prompt_tokens)
    if any(type(token) is not int for token in raw_prompt_values):
        raise ValueError("Kimi K3 DSpark prompt must contain integer tokens")
    logical_prompt_tokens = cast(tuple[int, ...], tuple(raw_prompt_values))

    digest = hashlib.sha256(b"exo-kimi-k3-dspark-setup/v1\0")

    def add_integer(value: int, *, name: str) -> None:
        if type(value) is not int or not 0 <= value <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError(f"Kimi K3 DSpark {name} must fit non-negative int64")
        digest.update(value.to_bytes(8, "big"))

    def add_tokens(tokens: tuple[int, ...], *, name: str) -> None:
        add_integer(len(tokens), name=f"{name} length")
        for token in tokens:
            if type(token) is not int or not 0 <= token <= 0x7FFFFFFF:
                raise ValueError(
                    f"Kimi K3 DSpark {name} must contain non-negative int32 tokens"
                )
            digest.update(token.to_bytes(4, "big"))

    def add_text(value: str, *, name: str) -> None:
        encoded = value.encode("utf-8")
        add_integer(len(encoded), name=f"{name} byte length")
        digest.update(encoded)

    add_tokens(logical_prompt_tokens, name="prompt")
    add_integer(max_tokens, name="max tokens")
    add_integer(prefill_step_size, name="prefill step size")
    add_integer(capacity_hint, name="capacity hint")
    add_integer(verify_width, name="verify width")
    add_integer(seed, name="seed")
    add_integer(int(is_bench), name="benchmark flag")
    add_integer(int(compact_greedy), name="compact greedy flag")
    add_integer(int(generation_progress), name="generation progress flag")
    add_integer(int(force_ordinary), name="force ordinary flag")
    add_integer(ordinary_after_context, name="ordinary-after-context threshold")
    if packed_agreements:
        digest.update(b"exo-kimi-k3-dspark-packed-agreements/v1\0")
    digest.update(b"exo-kimi-k3-w3-composition/v1\0")
    add_integer(int(deferred_async_width3), name="deferred W3 async flag")
    add_integer(
        int(authoritative_packed_width3),
        name="authoritative packed W3 flag",
    )
    add_integer(int(w3_prework_history), name="W3 KDA prework flag")
    add_integer(int(native_packed_q3), name="native packed Q3 flag")
    add_tokens(eos_token_ids, name="EOS tokens")
    add_tokens(banned_token_ids, name="banned tokens")
    add_tokens(terminal_token_ids, name="terminal tokens")
    add_integer(int(confidence_capture is not None), name="confidence capture flag")
    if confidence_capture is not None:
        if target_route_top_k != 8:
            raise ValueError(
                "Kimi K3 DSpark confidence capture requires target route top-k 8"
            )
        add_integer(target_route_top_k, name="target route top-k")
        add_text(
            str(confidence_capture.jsonl_path),
            name="confidence capture path",
        )
        add_text(confidence_capture.session_id, name="confidence capture session")
    add_integer(len(stop_sequences), name="stop sequence count")
    for stop in stop_sequences:
        encoded = stop.encode("utf-8")
        add_integer(len(encoded), name="stop sequence byte length")
        digest.update(encoded)
    if proposer_selection is not None:
        digest.update(b"exo-kimi-k3-dspark-dual-selection/v1\0")
        add_text(proposer_selection.role, name="dual proposer role")
        add_integer(
            proposer_selection.initial_prompt_tokens,
            name="dual proposer initial prompt tokens",
        )
        add_integer(
            proposer_selection.threshold_tokens,
            name="dual proposer threshold tokens",
        )
        add_text(proposer_selection.revision, name="dual proposer revision")
        add_text(
            proposer_selection.config_sha256,
            name="dual proposer config SHA-256",
        )
        add_text(
            proposer_selection.model_sha256,
            name="dual proposer model SHA-256",
        )
        add_text(
            proposer_selection.identity_sha256,
            name="dual proposer identity SHA-256",
        )
    if receipt_session_id is not None:
        if receipt_model_id is None:
            raise ValueError("width-four receipt model ID is missing")
        if target_route_top_k != 8:
            raise ValueError(
                "width-four receipt requires attested target route top-k 8"
            )
        digest.update(b"exo-kimi-k3-width4-receipt-binding/v3\0")
        add_text(receipt_session_id, name="width-four receipt session")
        add_text(receipt_model_id, name="width-four receipt model ID")
        add_integer(target_route_top_k, name="width-four target route top-k")

    raw = digest.digest()
    return tuple(
        int.from_bytes(raw[offset : offset + 4], "big") & 0x7FFFFFFF
        for offset in range(0, 16, 4)
    )


def _dspark_setup_error_fingerprint(error: str | None) -> int:
    if error is None:
        return 0
    return (
        int.from_bytes(hashlib.sha256(error.encode()).digest()[:4], "big") & 0x7FFFFFFF
    ) or 1


def _rank_agreed_dspark_setup(
    agreement: MlxRankAgreement,
    operation: Callable[[], _DSparkRequestSetup],
) -> _DSparkRequestSetup:
    """Run local setup, then fixed-order agree success and its full contract."""

    result: _DSparkRequestSetup | None = None
    local_error: str | None = None
    try:
        result = operation()
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"

    outcome = agreement.agree_stage_success(local_error is None)
    error_fingerprint = agreement.agree_token(
        _dspark_setup_error_fingerprint(local_error)
    )
    local_fingerprint = result.fingerprint if result is not None else (0, 0, 0, 0)
    fingerprint_agreement = tuple(
        agreement.agree_token(word) for word in local_fingerprint
    )

    if (
        outcome is not True
        or error_fingerprint != 0
        or fingerprint_agreement != local_fingerprint
    ):
        detail = (
            f"failed on every rank: {local_error}"
            if outcome is False and local_error is not None
            else "outcomes or request controls disagreed across ranks"
        )
        raise DSparkDistributedStateError(
            f"Kimi K3 DSpark setup {detail}; no target TP graph was built"
        ) from None
    agreed_result = cast(_DSparkRequestSetup, result)
    if agreed_result.packed_agreements:
        agreement.activate_packed_agreements()
    return agreed_result


def _prepare_dspark_request_setup(
    *,
    dspark: LoadedKimiK3DSpark,
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group,
    vision_processor: VisionProcessor | None,
    agreement: MlxRankAgreement,
    generation_progress: bool,
    width4_receipt_scope: Literal["request", "warmup"],
) -> _DSparkRequestSetup:
    """Build all failure-prone local request state without entering TP graphs."""

    mx.reset_peak_memory()
    is_pipeline = _has_pipeline_communication_layer(model)
    prompt_lookup_configuration = prompt_lookup_config(
        is_pipeline=is_pipeline,
        is_batch=False,
    )
    _validate_dspark_request(
        dspark,
        model=model,
        task=task,
        group=group,
        kv_prefix_cache=kv_prefix_cache,
        vision_processor=vision_processor,
        is_pipeline=is_pipeline,
        prompt_lookup_configuration=prompt_lookup_configuration,
    )

    seed = task.seed or 42
    mx.random.seed(seed)
    all_prompt_tokens = fix_unmatched_think_end_tokens(
        encode_prompt(tokenizer, prompt),
        tokenizer,
    )
    if len(all_prompt_tokens) < 2:
        raise ValueError("Kimi K3 DSpark requires at least two prompt tokens")
    request_dspark, proposer_selection = select_loaded_mlx_dspark(
        dspark,
        initial_prompt_tokens=len(all_prompt_tokens),
    )
    receipt_session_id: str | None = None
    receipt_model_id: str | None = None
    if width4_receipt_scope == "request" and width4_receipt_log_enabled():
        receipt_model_id = str(task.model)
        receipt_session_id, _receipt_selectors = validate_width4_request_contract(
            rank=group.rank(),
            world_size=group.size(),
            verify_width=request_dspark.verify_width,
            model_id=receipt_model_id,
        )
    anchor_token = int(all_prompt_tokens[-1].item())
    if not 0 <= anchor_token <= 0x7FFFFFFF:
        raise ValueError("Kimi K3 DSpark anchor token must fit non-negative int32")

    detokenizer = cast(_DSparkDetokenizer, cast(object, tokenizer.detokenizer))
    if not all(
        callable(getattr(detokenizer, name, None))
        for name in ("reset", "add_token", "finalize")
    ):
        raise TypeError("Kimi K3 DSpark tokenizer has no streaming detokenizer")
    detokenizer.reset()
    empty_logprobs = mx.array([], dtype=mx.float32)

    is_bench = task.bench
    caches = make_kv_cache(model=model)
    logits_processors = make_logits_processors(
        repetition_penalty=task.repetition_penalty,
        repetition_context_size=(
            task.repetition_context_size
            if task.repetition_context_size is not None
            else 20
        ),
        presence_penalty=task.presence_penalty,
        frequency_penalty=task.frequency_penalty,
    )
    if is_bench:
        logits_processors = [
            ban_token_ids(eos_ids_from_tokenizer(tokenizer)),
            *logits_processors,
        ]
    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        min_p=task.min_p if task.min_p is not None else 0.05,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    stop_sequences = tuple(
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else ()
    )
    max_stop_len = max((len(stop) for stop in stop_sequences), default=0)
    max_tokens = task.max_output_tokens or MAX_TOKENS
    eos_token_ids = tuple(eos_ids_from_tokenizer(tokenizer))
    banned_token_ids = eos_token_ids if is_bench else ()
    terminal_token_ids = () if is_bench else eos_token_ids
    compact_greedy = bool(
        greedy_vocab_parallel_stream_kwargs(
            temperature=0.0,
            logprobs=False,
            has_logits_processors=False,
            is_pipeline=False,
            speculative=False,
        )
    )
    ordinary_after_context = _dspark_ordinary_after_context()
    force_ordinary = _dspark_force_ordinary(len(all_prompt_tokens))
    prefill_step_size = _prefill_step_size(
        len(all_prompt_tokens) - 1,
        is_pipeline=False,
    )
    capacity_hint = dspark_context_capacity_hint(
        prompt_tokens=len(all_prompt_tokens),
        max_tokens=max_tokens,
        verify_width=request_dspark.verify_width,
    )
    confidence_request = None
    if request_dspark.config.confidence_capture is not None:
        raw_prompt_tokens = all_prompt_tokens.tolist()
        if not isinstance(raw_prompt_tokens, list):
            raise ValueError("Kimi K3 DSpark prompt tokens must be one-dimensional")
        confidence_request = dspark_confidence_capture_request(
            cast(list[int], raw_prompt_tokens)
        )
    runtime = KimiK3DSparkRequestRuntime.create(
        request_dspark,
        model,
        caches,
        agreement,
        capacity_hint=capacity_hint,
        banned_token_ids=banned_token_ids,
        terminal_token_ids=terminal_token_ids,
        compact_greedy=compact_greedy,
        confidence_request=confidence_request,
    )
    fingerprint = _dspark_setup_fingerprint(
        prompt_tokens=all_prompt_tokens,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        capacity_hint=capacity_hint,
        verify_width=request_dspark.verify_width,
        seed=seed,
        is_bench=is_bench,
        compact_greedy=compact_greedy,
        generation_progress=generation_progress,
        force_ordinary=force_ordinary,
        ordinary_after_context=ordinary_after_context,
        packed_agreements=request_dspark.config.packed_agreements,
        deferred_async_width3=getattr(
            request_dspark.config,
            "deferred_async_width3",
            False,
        ),
        authoritative_packed_width3=getattr(
            request_dspark.config,
            "authoritative_packed_width3",
            False,
        ),
        w3_prework_history=getattr(
            request_dspark.config,
            "w3_prework_history",
            False,
        ),
        native_packed_q3=getattr(
            request_dspark.config,
            "native_packed_q3",
            False,
        ),
        eos_token_ids=eos_token_ids,
        banned_token_ids=banned_token_ids,
        terminal_token_ids=terminal_token_ids,
        stop_sequences=stop_sequences,
        confidence_capture=request_dspark.config.confidence_capture,
        target_route_top_k=request_dspark.target_route_top_k,
        proposer_selection=proposer_selection,
        receipt_session_id=receipt_session_id,
        receipt_model_id=receipt_model_id,
    )
    return _DSparkRequestSetup(
        is_pipeline=is_pipeline,
        prompt_lookup_configuration=prompt_lookup_configuration,
        all_prompt_tokens=all_prompt_tokens,
        caches=caches,
        logits_processors=logits_processors,
        sampler=sampler,
        stop_sequences=stop_sequences,
        max_stop_len=max_stop_len,
        max_tokens=max_tokens,
        is_bench=is_bench,
        eos_token_ids=eos_token_ids,
        anchor_token=anchor_token,
        detokenizer=detokenizer,
        empty_logprobs=empty_logprobs,
        prefill_step_size=prefill_step_size,
        force_ordinary=force_ordinary,
        packed_agreements=request_dspark.config.packed_agreements,
        runtime=runtime,
        proposer_selection=proposer_selection,
        fingerprint=fingerprint,
        verify_width=request_dspark.verify_width,
        receipt_session_id=receipt_session_id,
    )


def _dspark_mlx_responses(
    runtime: KimiK3DSparkRequestRuntime,
    detokenizer: _DSparkDetokenizer,
    empty_logprobs: mx.array,
    *,
    anchor_token: int,
    max_tokens: int,
    eos_token_ids: tuple[int, ...],
    force_ordinary: bool,
    telemetry: _PromptLookupTelemetry,
) -> Generator[MlxGenerationResponse, None, None]:
    engine = runtime.agree_local_value(
        "round engine construction",
        runtime.make_round_engine,
    )
    started = runtime.agree_local_value(
        "decode timing initialization",
        time.perf_counter,
    )
    for generation_tokens, decoded in enumerate(
        dspark_decode_tokens(
            engine,
            anchor_token=anchor_token,
            max_tokens=max_tokens,
            eos_token_ids=eos_token_ids,
            force_ordinary=force_ordinary,
            round_observer=lambda stats: telemetry.observe_dspark_round(
                stats,
                target_cache_tokens=runtime.target_cache_offset,
            ),
        ),
        start=1,
    ):
        if telemetry.composition_receipt is not None:
            telemetry.composition_receipt.observe_output_token(
                decoded.token,
                from_draft=decoded.from_draft,
            )

        def render_token(decoded: object = decoded) -> str:
            decoded = cast(DSparkDecodedToken, decoded)
            if decoded.finish_reason == "stop":
                detokenizer.finalize()
            else:
                detokenizer.add_token(decoded.token)
                if decoded.finish_reason == "length":
                    detokenizer.finalize()
            return detokenizer.last_segment

        text = runtime.agree_text("detokenizer output", render_token)

        def build_response(
            decoded: object = decoded,
            text: str = text,
            generation_tokens: int = generation_tokens,
        ) -> MlxGenerationResponse:
            decoded = cast(DSparkDecodedToken, decoded)
            elapsed = time.perf_counter() - started
            return MlxGenerationResponse(
                text=text,
                token=decoded.token,
                logprobs=empty_logprobs,
                from_draft=decoded.from_draft,
                prompt_tokens=1,
                prompt_tps=0.0,
                generation_tokens=generation_tokens,
                generation_tps=(generation_tokens / elapsed if elapsed > 0 else 0.0),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=decoded.finish_reason,
            )

        yield runtime.agree_local_value(
            "response construction",
            build_response,
        )


def extract_top_logprobs(
    logprobs: mx.array,
    tokenizer: TokenizerWrapper,
    top_logprobs: int,
    selected_token: int,
    precomputed_indices: list[int] | None = None,
    precomputed_values: list[float] | None = None,
    precomputed_selected: float | None = None,
) -> tuple[float, list[TopLogprobItem]]:
    if (
        precomputed_indices is not None
        and precomputed_values is not None
        and precomputed_selected is not None
    ):
        top_indices_list: list[int] = precomputed_indices[:top_logprobs]
        top_values_list: list[float] = precomputed_values[:top_logprobs]
        selected_logprob = precomputed_selected
    else:
        selected_logprob_arr = logprobs[selected_token]
        top_logprobs = min(top_logprobs, logprobs.shape[0] - 1)
        top_indices = mx.argpartition(-logprobs, top_logprobs)[:top_logprobs]
        top_values = logprobs[top_indices]
        sort_order = mx.argsort(-top_values)
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]
        mx.eval(selected_logprob_arr, top_indices, top_values)
        selected_logprob = float(selected_logprob_arr.item())
        top_indices_list = top_indices.tolist()  # type: ignore
        top_values_list = top_values.tolist()  # type: ignore

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for token_id, token_logprob in zip(top_indices_list, top_values_list, strict=True):
        if math.isnan(token_logprob):
            continue

        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=list(token_str.encode("utf-8")),
            )
        )

    return selected_logprob, top_logprob_items


def _mlx_generate_impl(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
    vision_processor: VisionProcessor | None = None,
    dspark: LoadedKimiK3DSpark | None = None,
    width4_receipt_scope: Literal["request", "warmup"] = "request",
    width4_receipt_claim: Width4RequestReceiptClaim | None = None,
    packed_front_receipt: _PackedFrontReceiptRequest | None = None,
) -> Generator[GenerationResponse]:
    if width4_receipt_scope not in {"request", "warmup"}:
        raise ValueError(f"invalid width-four receipt scope: {width4_receipt_scope}")
    if width4_receipt_scope == "warmup" and width4_receipt_claim is not None:
        raise ValueError("width-four warmup cannot carry a request-admission claim")
    receipt_enabled = width4_receipt_scope == "request" and width4_receipt_log_enabled()
    if receipt_enabled and width4_receipt_claim is None:
        raise ValueError(
            "enabled width-four receipt requires a service-admission claim"
        )
    if not receipt_enabled and width4_receipt_claim is not None:
        raise ValueError("width-four request-admission claim reached disabled receipt")
    if receipt_enabled and dspark is None:
        raise ValueError("enabled width-four receipt requires Kimi K3 DSpark")
    if packed_front_receipt is None:
        raise ValueError("packed-front receipt request context is missing")
    if packed_front_receipt.enabled and (
        task.use_prefix_cache or kv_prefix_cache is not None
    ):
        raise ValueError(
            "packed-front diagnostic mode requires prefix caching to be disabled"
        )
    dspark_setup: _DSparkRequestSetup | None = None
    if dspark is not None:
        # A prior uncertain target submission can leave this process's shared
        # Metal/JACCL stream unsafe. Reject locally before any setup agreement.
        dspark.assert_healthy()
        if group is None:
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark setup requires a tensor-parallel group"
            )
        agreement = MlxRankAgreement(
            group,
            rank_zero_proposal_recovery=dspark.config.rank_zero_proposal_recovery,
        )
        dspark_setup = _rank_agreed_dspark_setup(
            agreement,
            lambda: _prepare_dspark_request_setup(
                dspark=dspark,
                model=model,
                tokenizer=tokenizer,
                task=task,
                prompt=prompt,
                kv_prefix_cache=kv_prefix_cache,
                group=group,
                vision_processor=vision_processor,
                agreement=agreement,
                generation_progress=on_generation_token is not None,
                width4_receipt_scope=width4_receipt_scope,
            ),
        )
        selection = dspark_setup.proposer_selection
        if selection is not None:
            logger.info(
                "Kimi K3 dual DSpark request selection: "
                f"role={selection.role}, "
                f"initial_prompt_tokens={selection.initial_prompt_tokens}, "
                f"threshold_tokens={selection.threshold_tokens}, "
                f"revision={selection.revision}, "
                f"config_sha256={selection.config_sha256}, "
                f"model_sha256={selection.model_sha256}, "
                f"identity_sha256={selection.identity_sha256}"
            )
        is_pipeline = dspark_setup.is_pipeline
        if packed_front_receipt.enabled and is_pipeline:
            raise ValueError(
                "W3 composition diagnostic requires non-pipeline DSpark TP"
            )
        prompt_lookup_configuration = dspark_setup.prompt_lookup_configuration
        all_prompt_tokens = dspark_setup.all_prompt_tokens
        min_prefix_hit_length = 1000
        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []
    else:
        # Ensure that generation stats only contains this request's peak memory.
        mx.reset_peak_memory()
        is_pipeline = _has_pipeline_communication_layer(model)
        prompt_lookup_configuration = prompt_lookup_config(
            is_pipeline=is_pipeline,
            is_batch=False,
        )
        # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
        seed = task.seed or 42
        mx.random.seed(seed)

        # Encode prompt once at the top and fix unmatched think tags
        all_prompt_tokens = encode_prompt(tokenizer, prompt)
        all_prompt_tokens = fix_unmatched_think_end_tokens(
            all_prompt_tokens,
            tokenizer,
        )
        min_prefix_hit_length = max(1000, system_prompt_token_count(task, tokenizer))

        vision = None
        if vision_processor is not None:
            try:
                vision = prepare_vision(
                    images=task.images,
                    chat_template_messages=task.chat_template_messages,
                    vision_processor=vision_processor,
                    tokenizer=tokenizer,
                    model=model,
                    model_id=task.model,
                    task_params=task,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )
        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
        media_regions = vision.media_regions if vision else []

    prompt_lookup_telemetry = _PromptLookupTelemetry()
    if dspark_setup is not None:
        (
            prompt_lookup_telemetry.prefill_width1_chunks,
            prompt_lookup_telemetry.prefill_width3_chunks,
            prompt_lookup_telemetry.prefill_noncontract_chunks,
        ) = _exact_prefill_chunk_geometry(
            token_count=len(dspark_setup.all_prompt_tokens) - 1,
            step_size=dspark_setup.prefill_step_size,
        )
        packed_front_receipt.begin(
            dspark_setup.runtime,
            dspark_setup.fingerprint,
        )
        if packed_front_receipt.enabled:
            prompt_lookup_telemetry.composition_receipt = packed_front_receipt

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench and not task.use_prefix_cache:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    is_exact_hit = False
    if dspark_setup is not None:
        kv_prefix_cache = None
        caches = dspark_setup.caches
        prompt_tokens = all_prompt_tokens
    elif kv_prefix_cache is None:
        caches = make_kv_cache(model=model)
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index, is_exact_hit = (
            kv_prefix_cache.get_kv_cache(
                model, all_prompt_tokens, media_regions=media_regions
            )
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if not mx_ranks_agree_on_value(prefix_hit_length, group):
            # Divergent restore positions would make ranks prefill different
            # token counts and deadlock the pipeline.
            logger.warning(
                "KV prefix cache hit lengths diverge across pipeline ranks; "
                "discarding the hit to keep prefill in lockstep"
            )
            caches = make_kv_cache(model=model)
            prompt_tokens = all_prompt_tokens
            prefix_hit_length = 0
            matched_index = None
            is_exact_hit = False
        elif prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )

    if dspark_setup is not None:
        logits_processors = dspark_setup.logits_processors
        sampler = dspark_setup.sampler
    else:
        logits_processors = make_logits_processors(
            repetition_penalty=task.repetition_penalty,
            repetition_context_size=task.repetition_context_size
            if task.repetition_context_size is not None
            else 20,
            presence_penalty=task.presence_penalty,
            frequency_penalty=task.frequency_penalty,
        )
        if is_bench:
            # Only sample length eos tokens
            eos_ids = eos_ids_from_tokenizer(tokenizer)
            logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        sampler = make_sampler(
            temp=task.temperature if task.temperature is not None else 0.7,
            top_p=task.top_p if task.top_p is not None else 1.0,
            min_p=task.min_p if task.min_p is not None else 0.05,
            top_k=task.top_k if task.top_k is not None else 0,
        )

    # Normalize stop sequences to a list
    if dspark_setup is not None:
        stop_sequences = dspark_setup.stop_sequences
        max_stop_len = dspark_setup.max_stop_len
        max_tokens = dspark_setup.max_tokens
    else:
        stop_sequences = tuple(
            ([task.stop] if isinstance(task.stop, str) else task.stop)
            if task.stop is not None
            else ()
        )
        max_stop_len = max((len(s) for s in stop_sequences), default=0)
        max_tokens = task.max_output_tokens or MAX_TOKENS

    maybe_vision_ctx = (
        patch_embed_tokens(
            model,
            vision.embeddings,
            prefix_hit_length,
            len(prompt_tokens) - 1,
            image_token_id=vision.image_token_id,
        )
        if vision is not None
        else contextlib.nullcontext()
    )
    use_remote = (
        len(prompt_tokens) > REMOTE_PREFILL_MIN_TOKENS
        and task.prefill_endpoint is not None
    )
    remote_prefilled = False
    prefill_tps = 0.0
    prefill_tokens = 0
    ssm_snapshots_list: list[CacheSnapshot] = []
    dspark_runtime: KimiK3DSparkRequestRuntime | None = None
    width4_receipt_context: Width4RequestReceiptContext | None = None
    if dspark_setup is not None:
        dspark_runtime = dspark_setup.runtime
        if dspark_setup.receipt_session_id is not None:
            local_receipt_context: Width4RequestReceiptContext | None = None

            def begin_receipt_contract() -> str:
                nonlocal local_receipt_context
                context = begin_width4_request_receipt(
                    rank=group.rank() if group is not None else 0,
                    world_size=group.size() if group is not None else 1,
                    verify_width=dspark_setup.verify_width,
                    model_id=str(task.model),
                    request_fingerprint=dspark_setup.fingerprint,
                    mx_module=mx,
                    claim=width4_receipt_claim,
                )
                if context is None:
                    raise RuntimeError(
                        "rank-agreed width-four receipt became disabled before prefill"
                    )
                if context.session_id != dspark_setup.receipt_session_id:
                    raise RuntimeError(
                        "width-four receipt session changed after setup agreement"
                    )
                local_receipt_context = context
                return context.agreement_contract()

            dspark_runtime.agree_text(
                "width-four request receipt reset contract",
                begin_receipt_contract,
                preserve_unanimous_error=True,
            )
            if local_receipt_context is None:
                raise RuntimeError(
                    "width-four request receipt context was not retained"
                )
            width4_receipt_context = local_receipt_context
    with maybe_vision_ctx:
        if dspark_setup is not None:
            assert dspark_runtime is not None
            prefill_tps, prefill_tokens = dspark_runtime.seed_prompt(
                prompt_tokens[:-1],
                prefill_step_size=dspark_setup.prefill_step_size,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                progress_callback=on_prefill_progress or (lambda _done, _total: None),
                distributed_progress_callback=distributed_prompt_progress_callback,
            )
        elif use_remote and task.prefill_endpoint is not None:
            try:
                prefill_tps, prefill_tokens, ssm_snapshots_list = remote_prefill(
                    prompt_tokens[:-1],
                    caches,
                    on_prefill_progress,
                    endpoint=task.prefill_endpoint,
                    request_id=str(uuid.uuid4()),
                    model_id=str(task.model),
                    start_pos=prefix_hit_length,
                )
                remote_prefilled = True
            except Exception:
                logger.opt(exception=True).warning(
                    "Remote prefill failed, falling back to local prefill"
                )
        if dspark is None and not remote_prefilled:
            prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
                model,
                tokenizer,
                sampler,
                prompt_tokens[:-1],
                caches,
                group,
                on_prefill_progress,
                distributed_prompt_progress_callback,
            )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    if kv_prefix_cache is not None and matched_index is not None and is_exact_hit:
        prefill_tps = kv_prefix_cache.prefill_tps[matched_index]

    skip_kimi_k3_exact_rewrite = (
        kv_prefix_cache is not None
        and matched_index is not None
        and is_exact_hit
        and _is_kimi_k3_model(model)
    )
    if kv_prefix_cache is not None and not skip_kimi_k3_exact_rewrite:
        hit_ratio = (
            prefix_hit_length / len(all_prompt_tokens)
            if len(all_prompt_tokens) > 0
            else 0.0
        )
        if matched_index is not None and (
            prefix_hit_length >= min_prefix_hit_length
            and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
        ):
            kv_prefix_cache.update_kv_cache(
                matched_index,
                all_prompt_tokens,
                caches,
                cache_snapshots,
                restore_pos=prefix_hit_length,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )
        else:
            kv_prefix_cache.add_kv_cache(
                all_prompt_tokens,
                caches,
                cache_snapshots,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )

    # stream_generate starts from the last two tokens. DSpark instead owns the
    # exact prompt[:-1] target/draft boundary and starts from prompt[-1].
    last_token = prompt_tokens[-2:] if dspark_runtime is None else None
    relay_enabled = is_pipeline and not task.logprobs
    decode_sampler = sampler
    with _pipeline_token_relay_scope(model, relay_enabled):
        if relay_enabled:
            relay_context = get_active_relay_context(model)
            if relay_context is None:
                raise RuntimeError("pipeline token relay failed to initialize")
            base_sampler = sampler

            def relay_sampler(logprobs: mx.array, /) -> mx.array:
                return relay_sampled_tokens(base_sampler(logprobs), relay_context)

            decode_sampler = relay_sampler

        accumulated_text = ""
        generated_text_parts: list[str] = []
        generation_start_time = time.perf_counter()
        logger.info("Starting decode")
        # Pipeline prefill and decode share one ordered P2P stream. A collective
        # here can race a sender whose final prefill frame was already received but
        # has not reported local completion. The first decode P2P operation safely
        # drains that tail without changing JACCL operation classes.
        if not is_pipeline:
            mx_barrier(group)
        if dspark_runtime is not None and packed_front_receipt.enabled:
            assert dspark_setup is not None
            packed_front_receipt.begin_decode(
                dspark_runtime,
                anchor_token=dspark_setup.anchor_token,
            )

        prompt_lookup_kwargs = prompt_lookup_stream_kwargs(
            prompt_lookup_configuration,
            group,
            all_prompt_tokens,
            prompt_lookup_telemetry,
        )
        greedy_vocab_parallel_kwargs: _GreedyVocabParallelStreamKwargs = (
            {}
            if dspark_runtime is not None
            else greedy_vocab_parallel_stream_kwargs(
                temperature=(task.temperature if task.temperature is not None else 0.7),
                logprobs=task.logprobs,
                has_logits_processors=bool(logits_processors),
                is_pipeline=is_pipeline,
                speculative=prompt_lookup_kwargs is not None,
            )
        )
        # MLX-LM normally launches token N+1 before yielding token N. If token N
        # is EOS (or EXO matches a stop sequence), abandoning that lookahead
        # leaves pipeline rank zero in an unmatched send while the final rank
        # enters the completion barrier.
        if dspark_runtime is not None:
            assert dspark_setup is not None
            decode_outputs = _dspark_mlx_responses(
                dspark_runtime,
                dspark_setup.detokenizer,
                dspark_setup.empty_logprobs,
                anchor_token=dspark_setup.anchor_token,
                max_tokens=max_tokens,
                eos_token_ids=(() if is_bench else dspark_setup.eos_token_ids),
                force_ordinary=dspark_setup.force_ordinary,
                telemetry=prompt_lookup_telemetry,
            )
        elif prompt_lookup_kwargs is None:
            assert last_token is not None
            decode_outputs = stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=last_token,
                max_tokens=max_tokens,
                async_lookahead=not is_pipeline,
                sampler=decode_sampler,
                logits_processors=logits_processors,
                prompt_cache=caches,
                prefill_step_size=1,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                **greedy_vocab_parallel_kwargs,
            )
        else:
            assert last_token is not None
            prompt_lookup_generate = cast(
                _PromptLookupStreamGenerate,
                stream_generate,
            )
            decode_outputs = prompt_lookup_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=last_token,
                max_tokens=max_tokens,
                async_lookahead=not is_pipeline,
                sampler=decode_sampler,
                logits_processors=logits_processors,
                prompt_cache=caches,
                prefill_step_size=1,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                **prompt_lookup_kwargs,
            )

        for completion_tokens, out in enumerate(decode_outputs, start=1):

            def resolve_response_control(
                out: MlxGenerationResponse = out,
                previous_text: str = accumulated_text,
            ) -> tuple[
                int,
                str | None,
                bool,
                str,
                str,
            ]:
                candidate_text = previous_text + out.text
                text = out.text
                finish_reason = out.finish_reason
                stop_matched = False

                for stop_seq in stop_sequences:
                    if stop_seq in candidate_text:
                        # Trim text to just before the stop sequence.
                        stop_index = candidate_text.find(stop_seq)
                        text_before_stop = candidate_text[:stop_index]
                        chunk_start = len(candidate_text) - len(out.text)
                        text = text_before_stop[chunk_start:]
                        finish_reason = "stop"
                        stop_matched = True
                        break

                future_text = candidate_text[-max_stop_len:] if max_stop_len > 0 else ""
                return (
                    int(out.token),
                    finish_reason,
                    stop_matched,
                    future_text,
                    text,
                )

            if dspark_runtime is not None:
                (
                    _agreed_token,
                    raw_finish_reason,
                    stop_matched,
                    accumulated_text,
                    text,
                ) = dspark_runtime.agree_response_control(resolve_response_control)
            else:
                (
                    _agreed_token,
                    raw_finish_reason,
                    stop_matched,
                    accumulated_text,
                    text,
                ) = resolve_response_control()
            finish_reason = cast(FinishReason | None, raw_finish_reason)

            is_done = finish_reason is not None
            terminal_packed_front_receipt: K3W3CompositionReceipt | None = None
            composition_terminal_barrier_complete = False
            if is_done and packed_front_receipt.enabled:
                if dspark_runtime is None:
                    raise RuntimeError(
                        "packed-front diagnostic request lost its DSpark runtime"
                    )
                dspark_runtime.agree_local_side_effect(
                    "packed-front terminal decode close",
                    decode_outputs.close,
                )
                if not is_pipeline:
                    # Close the global native-counter interval only after the
                    # terminal distributed boundary. The counters record branch
                    # entry, but a positive receipt must still prove that no
                    # later terminal error escaped the interval.
                    mx_barrier(group)
                    composition_terminal_barrier_complete = True
                terminal_packed_front_receipt = packed_front_receipt.finish(
                    dspark_runtime,
                    prompt_lookup_telemetry,
                )

            def build_public_response(
                out: MlxGenerationResponse = out,
                is_done: bool = is_done,
                completion_tokens: int = completion_tokens,
                stop_matched: bool = stop_matched,
                text: str = text,
                agreed_token: int = _agreed_token,
                finish_reason: FinishReason | None = finish_reason,
                terminal_packed_front_receipt: K3W3CompositionReceipt
                | None = terminal_packed_front_receipt,
            ) -> GenerationResponse:
                prompt_lookup_telemetry.observe_visible_token(
                    from_draft=bool(out.from_draft)
                )
                generated_text_parts.append(out.text)
                stats: GenerationStats | None = None
                response_usage: Usage | None = None
                if is_done:
                    # Resolve the terminal inner generator and its committed
                    # speculative telemetry before serializing public stats.
                    if packed_front_receipt.enabled:
                        if terminal_packed_front_receipt is None:
                            raise RuntimeError(
                                "packed-front receipt was not finalized before stats"
                            )
                    else:
                        decode_outputs.close()
                    decode_elapsed_seconds = time.perf_counter() - generation_start_time
                    effective_generation_tps = (
                        completion_tokens / decode_elapsed_seconds
                        if decode_elapsed_seconds > 0
                        else 0.0
                    )
                    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
                    if prefix_hit_length > 0:
                        prefix_cache_hit = "exact" if is_exact_hit else "partial"
                    stats = GenerationStats(
                        prompt_tps=float(prefill_tps or out.prompt_tps),
                        generation_tps=float(out.generation_tps),
                        prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                        generation_tokens=int(out.generation_tokens),
                        peak_memory_usage=_memory_from_mlx_decimal_gb(out.peak_memory),
                        prefix_cache_hit=prefix_cache_hit,
                        decode_elapsed_seconds=decode_elapsed_seconds,
                        effective_generation_tps=effective_generation_tps,
                        speculative_rounds=prompt_lookup_telemetry.rounds,
                        speculative_drafted_tokens=(
                            prompt_lookup_telemetry.drafted_tokens
                        ),
                        speculative_accepted_tokens=(
                            prompt_lookup_telemetry.accepted_tokens
                        ),
                        speculative_committed_tokens=(
                            prompt_lookup_telemetry.committed_tokens
                        ),
                        speculative_fallback_rounds=(
                            prompt_lookup_telemetry.fallback_rounds
                        ),
                        speculative_error_rounds=(prompt_lookup_telemetry.error_rounds),
                        speculative_full_width_rounds=(
                            prompt_lookup_telemetry.full_width_rounds
                        ),
                        target_width1_rounds=(
                            prompt_lookup_telemetry.target_width1_rounds
                        ),
                        prefill_width1_chunks=(
                            prompt_lookup_telemetry.prefill_width1_chunks
                        ),
                        prefill_width3_chunks=(
                            prompt_lookup_telemetry.prefill_width3_chunks
                        ),
                        prefill_noncontract_chunks=(
                            prompt_lookup_telemetry.prefill_noncontract_chunks
                        ),
                        k3_w3_composition_receipt=terminal_packed_front_receipt,
                    )
                    if not stop_matched and out.finish_reason not in get_args(
                        FinishReason
                    ):
                        logger.warning(
                            "Model generated unexpected finish_reason: "
                            f"{out.finish_reason}"
                        )

                    total_prompt_tokens = len(all_prompt_tokens)
                    response_usage = Usage(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_prompt_tokens + completion_tokens,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=prefix_hit_length
                        ),
                        completion_tokens_details=CompletionTokensDetails(
                            reasoning_tokens=0,
                            accepted_prediction_tokens=(
                                prompt_lookup_telemetry.visible_accepted_tokens
                            ),
                            rejected_prediction_tokens=max(
                                0,
                                prompt_lookup_telemetry.drafted_tokens
                                - prompt_lookup_telemetry.accepted_tokens,
                            ),
                        ),
                    )

                logprob: float | None = None
                top_logprobs: list[TopLogprobItem] | None = None
                if task.logprobs:
                    with mx.stream(generation_stream):
                        logprob, top_logprobs = extract_top_logprobs(
                            logprobs=out.logprobs,
                            tokenizer=tokenizer,
                            top_logprobs=(task.top_logprobs or DEFAULT_TOP_LOGPROBS),
                            selected_token=out.token,
                        )

                if is_done:
                    generation_elapsed = (
                        stats.decode_elapsed_seconds
                        if stats is not None
                        and stats.decode_elapsed_seconds is not None
                        else time.perf_counter() - generation_start_time
                    )
                    generated_tokens = len(generated_text_parts)
                    generation_tps = (
                        generated_tokens / generation_elapsed
                        if generation_elapsed > 0
                        else 0.0
                    )
                    logger.debug(
                        f"Generation complete: prefill {prompt_tokens} tokens @ "
                        f"{prefill_tps:.1f} tok/s, generated {generated_tokens} "
                        f"tokens @ {generation_tps:.1f} tok/s"
                    )

                return GenerationResponse(
                    text=text,
                    token=agreed_token,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                    finish_reason=finish_reason,
                    stats=stats,
                    usage=response_usage,
                )

            response = (
                dspark_runtime.agree_local_value(
                    "public response construction",
                    build_public_response,
                )
                if dspark_runtime is not None
                else build_public_response()
            )

            if on_generation_token is not None:
                if dspark_runtime is not None:
                    dspark_runtime.agree_local_side_effect(
                        "generation progress callback",
                        on_generation_token,
                    )
                else:
                    on_generation_token()

            if (
                is_done
                and dspark_runtime is not None
                and not is_pipeline
                and not composition_terminal_barrier_complete
            ):
                # Complete the distributed terminal boundary before yielding;
                # a downstream parser may not resume this generator.
                mx_barrier(group)

            if (
                is_done
                and dspark_runtime is not None
                and dspark_runtime.confidence_capture_enabled
            ):
                # A capture is complete only after every terminal side effect and
                # rank boundary succeeds.  The public response has not been yielded
                # yet, so a downstream parser cannot abandon this generator first.
                dspark_runtime.agree_local_side_effect(
                    "confidence capture finalization",
                    lambda: dspark_runtime.finalize_confidence_capture(complete=True),
                )

            if is_done and dspark_runtime is not None:
                if width4_receipt_context is not None:
                    dspark_runtime.agree_local_side_effect(
                        "width-four packed agreement attestation",
                        lambda: dspark_runtime.log_packed_agreement_attestation(
                            required=True
                        ),
                    )
                else:
                    dspark_runtime.log_packed_agreement_attestation()
                dspark_runtime.log_deferred_async_width3_attestation()

            if is_done and width4_receipt_context is not None:
                if dspark_runtime is None:
                    raise RuntimeError(
                        "width-four receipt reached terminal output without DSpark"
                    )

                local_receipt: dict[str, object] | None = None

                def capture_receipt_contract() -> str:
                    nonlocal local_receipt
                    receipt = capture_width4_dispatch_receipt(
                        context=width4_receipt_context,
                        process_id=os.getpid(),
                        host=platform.node(),
                        response_built=True,
                        generation_callback_complete=True,
                        terminal_barrier_complete=True,
                        confidence_finalization_complete=True,
                        packed_attestation_complete=True,
                    )
                    if receipt is None:
                        raise RuntimeError(
                            "enabled width-four terminal receipt became disabled"
                        )
                    local_receipt = receipt
                    return width4_receipt_agreement_contract(receipt)

                dspark_runtime.agree_text(
                    "width-four terminal receipt capture contract",
                    capture_receipt_contract,
                    preserve_unanimous_error=True,
                )
                if local_receipt is None:
                    raise RuntimeError("width-four terminal receipt was not retained")
                receipt = local_receipt
                mark_width4_receipt_rank_agreed(receipt)
                logger.info(format_width4_dispatch_receipt(receipt))
                # Receipt workers are one-shot. Drain every enqueue=True sink
                # before the terminal response can escape, so a successful C1
                # cannot race collection of its own evidence record.
                logger.complete()

            if is_done and packed_front_receipt.enabled:
                assert dspark_runtime is not None
                _publish_composition_receipt(
                    dspark_runtime,
                    packed_front_receipt,
                )

            yield response

            if is_done:
                if dspark_runtime is None and not is_pipeline:
                    mx_barrier(group)
                break


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
    vision_processor: VisionProcessor | None = None,
    dspark: LoadedKimiK3DSpark | None = None,
    width4_receipt_scope: Literal["request", "warmup"] = "request",
    width4_receipt_claim: Width4RequestReceiptClaim | None = None,
    *,
    _packed_front_receipt_phase: Literal["engine_startup", "api"] = "api",
) -> Generator[GenerationResponse]:
    """Generate with an optional fail-closed request-scoped packed-front receipt."""

    packed_front_receipt = _PackedFrontReceiptRequest.from_environment(
        model,
        dspark,
        phase=_packed_front_receipt_phase,
    )
    try:
        yield from _mlx_generate_impl(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=group,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=(distributed_prompt_progress_callback),
            on_generation_token=on_generation_token,
            vision_processor=vision_processor,
            dspark=dspark,
            width4_receipt_scope=width4_receipt_scope,
            width4_receipt_claim=width4_receipt_claim,
            packed_front_receipt=packed_front_receipt,
        )
    finally:
        packed_front_receipt.abort()
