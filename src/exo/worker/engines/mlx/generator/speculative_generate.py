"""MTP / n-gram speculative decoding generator for exo.

Drop-in replacement for mlx_lm's ``stream_generate`` in exo's decode loop
(``generator/generate.py``). Drafts candidate tokens with either the model's
built-in MTP (NextN) head or an n-gram (prompt-lookup) speculator, verifies them
in one forward through the parent model, accepts the longest matching prefix,
and rewinds the KV cache on rejection.

Invoked ONLY when ``EXO_MTP_SPECULATIVE`` or ``EXO_NGRAM_SPECULATIVE`` is set;
otherwise exo's decode loop calls the stock ``stream_generate``. Yields
``mlx_lm.generate.GenerationResponse`` so the existing decode-loop body works
unchanged.

Algorithm (mirrors mlx_lm's ``speculative_generate_step``, adapted):
  * exo pre-prefills ``prompt_cache`` (the model caches), so we SKIP the
    speculative prefill — ``prompt`` is just the last 2 seed tokens.
  * The MTP draft head is a submodule of the parent model (``model.mtp_head``);
    its KV cache is the LAST entry of ``prompt_cache`` (appended by the patched
    ``make_cache``). The drafter runs LOCAL to the verifying node — no
    distributed collectives (the head is replicated on every node).
  * The verifying forward goes through the parent model's sharded layers exactly
    as ``stream_generate`` does; collectives happen inside the model. We call
    ``model.model(...)`` then ``model.lm_head(...)`` so we can capture the
    pre-lm_head hidden to seed the next MTP draft (no extra forward).
  * int8 MLA KV is preserved by calling ``maybe_quantize_kv_cache`` after each
    parent forward, matching MLX's ``_step``.
  * ``num_draft_tokens`` defaults to 1 (``MTP_NUM_DRAFT_TOKENS``) — the GLM-5.2
    head has ``num_nextn_predict_layers=1``.
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import GenerationResponse, maybe_quantize_kv_cache, stream_generate
from mlx_lm.generate import generation_stream  # thread-local stream mlx_lm uses
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.constants import (
    EXO_MTP_SPECULATIVE,
    EXO_NGRAM_SPECULATIVE,
    MTP_NUM_DRAFT_TOKENS,
)
from exo.worker.engines.mlx.mtp import MTPDraftModel, NGramSpeculator

# Reuse mlx_lm's thread-local generation_stream. Creating our own stream here
# DEADLOCKS exo's sharded model: the distributed collectives (all_sum in o_proj,
# sum_gradients in MoE) are scheduled on this stream by the model layers, but
# the distributed backend expects them on the SAME thread-local stream the group
# was initialized with. mlx_lm's generate_step/stream_generate all use this one.


def _to_int(token) -> int:
    """Robustly extract a Python int from a sampler result.

    Samplers may return a 0-dim array, a 1-element 1-dim array, or a plain int.
    Handles all three without subscript errors.
    """
    if hasattr(token, "tolist"):
        token = token.tolist()
    if isinstance(token, list):
        return int(token[0])
    return int(token)


def _emit(
    detokenizer,
    token: int,
    logprobs,
    from_draft: bool,
    prompt_size: int,
    prompt_tps: float,
    ntoks: int,
    tic: float,
    first_token: bool,
    eos_ids: set[int],
    max_tokens: int,
):
    """Helper: yield a GenerationResponse, updating detokenizer + bookkeeping."""
    detokenizer.add_token(token)
    finish = "stop" if token in eos_ids else None
    resp = GenerationResponse(
        text=detokenizer.last_segment,
        token=token,
        logprobs=logprobs,
        from_draft=from_draft,
        prompt_tokens=prompt_size,
        prompt_tps=prompt_tps,
        generation_tokens=ntoks,
        generation_tps=ntoks / max(time.perf_counter() - tic, 1e-6),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason=finish,
    )
    return resp


def speculative_generate(
    model: nn.Module,
    tokenizer,
    prompt,
    max_tokens: int,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[list[Callable]] = None,
    prompt_cache: Optional[list] = None,
    prefill_step_size: int = 1,
    kv_group_size: int = 64,
    kv_bits: Optional[int] = None,
    *,
    num_draft_tokens: Optional[int] = None,
    prompt_tokens: Optional[list[int]] = None,
    **_unused,
):
    """Speculative generator yielding ``GenerationResponse``.

    Args mirror ``stream_generate`` (drop-in for exo's decode loop).
    ``prompt_tokens`` is the full prompt id list (seeds the n-gram history).
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        prompt = mx.array(prompt)

    n_draft = num_draft_tokens if num_draft_tokens is not None else MTP_NUM_DRAFT_TOKENS
    n_draft = max(1, n_draft)
    prompt_tokens = prompt_tokens or []

    mtp_enabled = EXO_MTP_SPECULATIVE and getattr(model, "mtp_head", None) is not None
    ngram_enabled = EXO_NGRAM_SPECULATIVE

    # Decide drafter. MTP wins if both are set and the head is present.
    if mtp_enabled:
        kind = "mtp"
    elif ngram_enabled:
        kind = "ngram"
    else:
        yield from stream_generate(
            model, tokenizer, prompt, max_tokens,
            sampler=sampler, logits_processors=logits_processors,
            prompt_cache=prompt_cache, prefill_step_size=prefill_step_size,
            kv_group_size=kv_group_size, kv_bits=kv_bits,
        )
        return

    # The model cache is prompt_cache as-is. The MTP draft cache is built and
    # owned by MTPDraftModel (it is NOT appended to the model cache list — doing
    # so breaks auto_parallel's patched_call, which walks cache[-1] and calls
    # mx.depends on its .keys; an untouched MTP cache has .keys=None).
    model_cache = prompt_cache

    # Speculative decode requires a trimmable model cache (rewind on reject).
    if model_cache is not None and not can_trim_prompt_cache(
        [c for c in model_cache if c is not None]
    ):
        yield from stream_generate(
            model, tokenizer, prompt, max_tokens,
            sampler=sampler, logits_processors=logits_processors,
            prompt_cache=model_cache, prefill_step_size=prefill_step_size,
            kv_group_size=kv_group_size, kv_bits=kv_bits,
        )
        return

    # Build the drafter. For MTP, the draft cache is owned by MTPDraftModel.
    if kind == "mtp":
        drafter = MTPDraftModel(model)
        draft_cache = drafter.draft_cache
    else:
        drafter = NGramSpeculator(ngram_size=3, max_draft=n_draft)
        drafter.add_tokens(prompt_tokens)
        draft_cache = None  # n-gram keeps no KV state

    sampler_fn = sampler or (lambda x: mx.argmax(x, axis=-1))
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    detokenizer = tokenizer.detokenizer
    eos_ids = set(tokenizer.eos_token_ids or [])

    # ------------------------------------------------------------------
    # RANK SYNCHRONIZATION. In tensor-parallel runs every rank executes this
    # loop and must issue an IDENTICAL sequence of collectives. Draft tokens
    # (local MTP forward) and sampled tokens (per-rank RNG) are NOT guaranteed
    # to be bitwise identical across ranks; a single disagreement changes
    # n_accept -> ntoks -> iteration count, desynchronizing the ranks'
    # collective streams (observed as a Jaccl LOC_LEN_ERR when one rank's
    # small barrier all_sum met the other rank's L=2 hidden all_sum).
    # Fix: broadcast rank 0's tokens with a tiny fixed-size all_sum so every
    # rank-dependent decision (accept/reject, EOS, max_tokens) is identical.
    # ------------------------------------------------------------------
    try:
        _group = mx.distributed.init()
    except Exception:
        _group = None
    if _group is not None and _group.size() <= 1:
        _group = None

    def _sync_tokens(vals: list[int]) -> list[int]:
        """Broadcast rank 0's integer tokens to all ranks (fixed-size all_sum)."""
        if _group is None or not vals:
            return vals
        x = mx.array(vals, dtype=mx.int32)
        if _group.rank() != 0:
            x = mx.zeros_like(x)
        with mx.stream(generation_stream):
            x = mx.distributed.all_sum(x, group=_group)
        mx.eval(x)
        out = x.tolist()
        return [int(v) for v in (out if isinstance(out, list) else [out])]

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler_fn(logprobs)
        return y, logprobs

    def _verify(inp: mx.array, n_predict: int):
        """Parent forward: returns (logits[0,-n_predict:,:], last_hidden_seeded)."""
        with mx.stream(generation_stream):
            hidden = model.model(inp[None], cache=model_cache)
            logits = model.lm_head(hidden)
            quantize_cache_fn(model_cache)
        # Force-evaluate the distributed collectives (all_gather/all_sum in the
        # sharded model) AND the KV cache state before anything else runs. Without
        # this, the lazy collective from the verify stays pending while the MTP
        # draft enqueues local compute — the next collective (or the draft's
        # cache ops) then deadlocks waiting for the unevaluated all_gather.
        # mlx_lm's generate_step does the same: mx.eval([c.state for c in cache]).
        mx.eval(logits)
        if model_cache is not None:
            mx.eval([c.state for c in model_cache if c is not None and hasattr(c, "state")])
        if kind == "mtp":
            drafter.seed_from_parent(hidden)  # captures [:, -1:, :]
        return logits[0, -n_predict:, :]  # [n_predict, vocab]

    # ------------------------------------------------------------------
    # BOOTSTRAP: run the parent on the seed tokens to produce token #1 and
    # capture the hidden that seeds the first MTP draft. (For n-gram we still
    # need this first token to have something to draft from.)
    # ------------------------------------------------------------------
    y = prompt.astype(mx.uint32)
    prev = mx.array(prompt_tokens, dtype=mx.uint32) if prompt_tokens else None

    tic = time.perf_counter()
    prompt_tps = 0.0
    ntoks = 0
    accepted_total = 0
    drafted_total = 0
    first_token = True
    last_logprobs = None

    # Bootstrap forward.
    logits = _verify(y, n_predict=1)
    mx.eval(logits)
    t0, lp0 = _process_and_sample(prev if prev is not None else y, logits[0, :])
    token0 = _sync_tokens([_to_int(t0)])[0]
    last_logprobs = lp0
    ntoks += 1
    if first_token:
        prompt_tps = prompt.size / max(time.perf_counter() - tic, 1e-6)
        tic = time.perf_counter()
        first_token = False
    detokenizer.add_token(token0)
    yield GenerationResponse(
        text=detokenizer.last_segment, token=token0, logprobs=lp0, from_draft=False,
        prompt_tokens=prompt.size, prompt_tps=prompt_tps, generation_tokens=ntoks,
        generation_tps=ntoks / max(time.perf_counter() - tic, 1e-6),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="stop" if token0 in eos_ids else None,
    )
    if token0 in eos_ids or ntoks >= max_tokens:
        detokenizer.finalize()
        return
    y = mx.array([token0], dtype=mx.uint32)
    prev = mx.concatenate([prev, y]) if prev is not None else y

    # ------------------------------------------------------------------
    # MAIN LOOP: draft -> verify -> accept/reject -> rewind.
    # ------------------------------------------------------------------
    try:
        while ntoks < max_tokens:
            num_draft = min(n_draft, max_tokens - ntoks)

            # 1. DRAFT -------------------------------------------------------
            if kind == "mtp":
                draft_tokens = drafter.draft(y, draft_cache, num_draft)
            else:  # ngram
                recent = prev.tolist() if prev is not None else []
                draft_tokens = drafter.propose_drafts(recent)
                draft_tokens = draft_tokens[-num_draft:] if draft_tokens else []
            # Sync drafts across ranks (rank 0 wins). Pad to num_draft so the
            # collective is fixed-size; -1 marks unused slots.
            if _group is not None:
                _padded = [int(t) for t in draft_tokens]
                _padded += [-1] * (num_draft - len(_padded))
                draft_tokens = [t for t in _sync_tokens(_padded) if t >= 0]
            drafted_total += len(draft_tokens)

            # 2. VERIFY: feed [y, *draft_tokens] through the parent. ----------
            if len(draft_tokens) == 0:
                verify_input = y
                n_predict = 1
            else:
                verify_input = mx.concatenate([y, mx.array(draft_tokens, dtype=mx.uint32)])
                n_predict = len(draft_tokens) + 1

            logits = _verify(verify_input, n_predict)
            mx.eval(logits)

            # 3. SAMPLE verified tokens position-by-position (greedy match).
            out_y, out_lp = [], []
            _t = verify_input
            if n_predict > 1 and prev is not None:
                _t = verify_input[: -(n_predict - 1)]
            for i in range(n_predict):
                if prev is not None:
                    prev_i = mx.concatenate([prev, _t]) if _t.size else prev
                else:
                    prev_i = _t
                ti, lpi = _process_and_sample(prev_i, logits[i, :])
                out_y.append(ti)
                out_lp.append(lpi)
            verified = [_to_int(t) for t in out_y]
            # Sync verified tokens across ranks (rank 0 wins) so accept/reject,
            # EOS and max_tokens decisions are identical on every rank.
            verified = _sync_tokens(verified)

            # 4. ACCEPT/REJECT ----------------------------------------------
            n_accept = 0
            for d, v in zip(draft_tokens, verified, strict=False):
                if d == v:
                    n_accept += 1
                else:
                    break
            accepted_total += n_accept

            # 5. EMIT accepted draft tokens, then the bonus verified token.
            stop = False
            for i in range(n_accept):
                tok = verified[i]
                ntoks += 1
                detokenizer.add_token(tok)
                yield GenerationResponse(
                    text=detokenizer.last_segment, token=tok, logprobs=out_lp[i],
                    from_draft=True, prompt_tokens=prompt.size, prompt_tps=prompt_tps,
                    generation_tokens=ntoks,
                    generation_tps=ntoks / max(time.perf_counter() - tic, 1e-6),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason="stop" if tok in eos_ids else None,
                )
                if tok in eos_ids:
                    stop = True
                    break
                if ntoks >= max_tokens:
                    stop = True
                    break
                prev = mx.concatenate([prev, mx.array([tok], dtype=mx.uint32)])
            if stop:
                break

            # Bonus token (the first non-accepted verified token). For zero
            # drafts this is just the next decoded token.
            bonus_idx = n_accept
            if bonus_idx >= len(verified):
                break
            tok = verified[bonus_idx]
            last_logprobs = out_lp[bonus_idx]
            ntoks += 1
            detokenizer.add_token(tok)
            yield GenerationResponse(
                text=detokenizer.last_segment, token=tok, logprobs=last_logprobs,
                from_draft=False, prompt_tokens=prompt.size, prompt_tps=prompt_tps,
                generation_tokens=ntoks,
                generation_tps=ntoks / max(time.perf_counter() - tic, 1e-6),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop" if tok in eos_ids else None,
            )
            if tok in eos_ids or ntoks >= max_tokens:
                break
            y = mx.array([tok], dtype=mx.uint32)
            prev = mx.concatenate([prev, y])

            # 6. REWIND caches ---------------------------------------------
            # The verify forward wrote n_predict positions; we keep
            # (n_accept + 1): the accepted drafts + the bonus. Trim the rest.
            excess_model = n_predict - (n_accept + 1)
            if excess_model > 0 and model_cache is not None:
                trim_prompt_cache(model_cache, excess_model)
            if kind == "mtp" and draft_cache is not None and len(draft_tokens) > 0:
                excess_draft = len(draft_tokens) - n_accept
                if excess_draft > 0:
                    # draft_cache is a CacheList(KVCache, KVCache); trim_prompt_cache
                    # iterates it and trims each inner KV cache.
                    try:
                        trim_prompt_cache([draft_cache], excess_draft)
                    except Exception:
                        # Fallback: trim the inner caches directly.
                        for c in draft_cache:
                            if hasattr(c, "trim"):
                                c.trim(excess_draft)
    finally:
        if drafted_total > 0:
            rate = accepted_total / drafted_total
            print(
                f"[speculative_generate] {kind}: accepted {accepted_total}/"
                f"{drafted_total} draft tokens ({rate:.1%})",
                flush=True,
            )

    detokenizer.finalize()
    if not first_token and ntoks > 0:
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=verified[-1] if verified else token0,
            logprobs=last_logprobs if last_logprobs is not None else lp0,
            from_draft=False,
            prompt_tokens=prompt.size, prompt_tps=prompt_tps,
            generation_tokens=ntoks,
            generation_tps=ntoks / max(time.perf_counter() - tic, 1e-6),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if verified and verified[-1] in eos_ids else "length",
        )
