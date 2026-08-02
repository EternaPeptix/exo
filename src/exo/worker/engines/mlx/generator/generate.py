import contextlib
import functools
import hashlib
import math
import os
import time
import uuid
from dataclasses import dataclass
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
    DSparkDecodedToken,
    DSparkDistributedStateError,
    DSparkRoundTelemetry,
    KimiK3DSparkRequestRuntime,
    LoadedMlxDSpark,
    MlxRankAgreement,
    dspark_context_capacity_hint,
    dspark_decode_tokens,
    validate_dspark_greedy_sampling,
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
    runtime: KimiK3DSparkRequestRuntime
    fingerprint: tuple[int, ...]


@dataclass
class _PromptLookupTelemetry:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    committed_tokens: int = 0
    visible_accepted_tokens: int = 0
    fallback_rounds: int = 0
    error_rounds: int = 0

    def observe(self, stats: "_SpeculativeRoundStatsLike") -> None:
        self.rounds += 1
        self.drafted_tokens += int(stats.drafted_tokens)
        self.accepted_tokens += int(stats.accepted_tokens)
        self.committed_tokens += int(stats.committed_tokens)

    def observe_dspark_round(self, stats: DSparkRoundTelemetry) -> None:
        """Account committed speculative work at the target transaction boundary.

        A textual stop can end response iteration partway through an already
        committed multi-token round.  These counters intentionally describe
        internal speculative decisions, while public completion_tokens remains
        the number of visible response tokens.
        """

        self.rounds += 1
        self.drafted_tokens += stats.proposed
        self.accepted_tokens += stats.accepted
        self.committed_tokens += stats.emitted
        self.fallback_rounds += int(stats.fallback)
        self.error_rounds += int(stats.error is not None)

    def observe_visible_token(self, *, from_draft: bool) -> None:
        """Track accepted predictions that remain in the public completion."""

        self.visible_accepted_tokens += int(from_draft)


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
    dspark: LoadedMlxDSpark | None = None,
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
    ) = (
        rank_agreed_local_stage(
            "inference warmup setup",
            group,
            prepare_warmup,
            warmup_contract,
        )
    )

    tokens_generated = 0
    final_stats: GenerationStats | None = None

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
    dspark: LoadedMlxDSpark,
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
    eos_token_ids: tuple[int, ...],
    banned_token_ids: tuple[int, ...],
    terminal_token_ids: tuple[int, ...],
    stop_sequences: tuple[str, ...],
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
    add_tokens(eos_token_ids, name="EOS tokens")
    add_tokens(banned_token_ids, name="banned tokens")
    add_tokens(terminal_token_ids, name="terminal tokens")
    add_integer(len(stop_sequences), name="stop sequence count")
    for stop in stop_sequences:
        encoded = stop.encode("utf-8")
        add_integer(len(encoded), name="stop sequence byte length")
        digest.update(encoded)

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
    return cast(_DSparkRequestSetup, result)


def _prepare_dspark_request_setup(
    *,
    dspark: LoadedMlxDSpark,
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group,
    vision_processor: VisionProcessor | None,
    agreement: MlxRankAgreement,
    generation_progress: bool,
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
    force_ordinary = _strict_env_flag(
        "EXO_MLX_KIMI_K3_DSPARK_FORCE_ORDINARY",
        os.environ.get("EXO_MLX_KIMI_K3_DSPARK_FORCE_ORDINARY", "0"),
    )
    prefill_step_size = _prefill_step_size(
        len(all_prompt_tokens) - 1,
        is_pipeline=False,
    )
    capacity_hint = dspark_context_capacity_hint(
        prompt_tokens=len(all_prompt_tokens),
        max_tokens=max_tokens,
        verify_width=dspark.verify_width,
    )
    runtime = KimiK3DSparkRequestRuntime.create(
        dspark,
        model,
        caches,
        agreement,
        capacity_hint=capacity_hint,
        banned_token_ids=banned_token_ids,
        terminal_token_ids=terminal_token_ids,
        compact_greedy=compact_greedy,
    )
    fingerprint = _dspark_setup_fingerprint(
        prompt_tokens=all_prompt_tokens,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        capacity_hint=capacity_hint,
        verify_width=dspark.verify_width,
        seed=seed,
        is_bench=is_bench,
        compact_greedy=compact_greedy,
        generation_progress=generation_progress,
        force_ordinary=force_ordinary,
        eos_token_ids=eos_token_ids,
        banned_token_ids=banned_token_ids,
        terminal_token_ids=terminal_token_ids,
        stop_sequences=stop_sequences,
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
        runtime=runtime,
        fingerprint=fingerprint,
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
            round_observer=telemetry.observe_dspark_round,
        ),
        start=1,
    ):

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
    dspark: LoadedMlxDSpark | None = None,
) -> Generator[GenerationResponse]:
    dspark_setup: _DSparkRequestSetup | None = None
    if dspark is not None:
        if group is None:
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark setup requires a tensor-parallel group"
            )
        agreement = MlxRankAgreement(group)
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
            ),
        )
        is_pipeline = dspark_setup.is_pipeline
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
    with maybe_vision_ctx:
        if dspark_setup is not None:
            dspark_runtime = dspark_setup.runtime
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
        prompt_lookup_telemetry = _PromptLookupTelemetry()
        logger.info("Starting decode")
        # Pipeline prefill and decode share one ordered P2P stream. A collective
        # here can race a sender whose final prefill frame was already received but
        # has not reported local completion. The first decode P2P operation safely
        # drains that tail without changing JACCL operation classes.
        if not is_pipeline:
            mx_barrier(group)

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

            def build_public_response(
                out: MlxGenerationResponse = out,
                is_done: bool = is_done,
                completion_tokens: int = completion_tokens,
                stop_matched: bool = stop_matched,
                text: str = text,
                agreed_token: int = _agreed_token,
                finish_reason: FinishReason | None = finish_reason,
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

            if is_done and dspark_runtime is not None and not is_pipeline:
                # Complete the distributed terminal boundary before yielding;
                # a downstream parser may not resume this generator.
                mx_barrier(group)

            yield response

            if is_done:
                if dspark_runtime is None and not is_pipeline:
                    mx_barrier(group)
                break
