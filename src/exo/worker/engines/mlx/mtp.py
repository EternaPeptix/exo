"""MTP (Multi-Token Prediction / NextN) draft head for GLM-5.2 speculative decoding.

This module implements the DeepSeek-V3-style NextN predictor as a draft model for
speculative decoding. GLM-5.2 ships with ``num_nextn_predict_layers=1`` MTP layer
(layer index 78, i.e. one past the 78 decoder layers) that, given the main model's
last hidden state and a candidate next token, predicts the token *after* it.

The head is a single transformer block (MLA attention + 256-expert MoE) wrapped
in the NextN projection: it takes ``concat(enorm(h_prev), hnorm(emb(token)))``
and projects it back to hidden_size via ``eh_proj`` before the block. It reuses
the main model's ``embed_tokens`` and ``lm_head`` (both replicated, never sharded)
and has its own private KV cache appended at the end of the model's cache list.

Design notes
------------
* The head is **replicated on every node** (not sharded). It is 1/78 of the model
  and the draft forward is local to the verifying node — no collectives. Both
  nodes draft independently and arrive at the same tokens deterministically, so
  the verifying forward sees consistent inputs.
* ``MTPDraftModel`` wraps a parent ``Model`` + its ``mtp_head`` and exposes the
  ``draft`` interface that ``speculative_generate`` expects.
* ``NGramSpeculator`` is the no-weights fallback used when MTP is unavailable
  (grafting failed or head missing) but speculative decode was requested.
"""
from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


def _make_decoder_layer(config: Any, layer_idx: int):
    """Build the transformer block matching the parent model family.

    GLM-5.2 (``glm_moe_dsa``) uses ``GlmMoeDsaDecoderLayer`` whose attention takes
    and returns ``prev_topk_indices``; DeepSeek-V3.2 uses ``DeepseekV32DecoderLayer``
    with plain attention. We prefer the DSA layer if importable (GLM-5.2 path) and
    fall back to V3.2. Returns ``(layer, returns_topk)``.
    """
    try:
        from mlx_lm.models.glm_moe_dsa import GlmMoeDsaDecoderLayer

        return GlmMoeDsaDecoderLayer(config, layer_idx), True
    except Exception:
        from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer

        return DeepseekV32DecoderLayer(config, layer_idx), False


class _SharedHead(nn.Module):
    """Namespace for the ``shared_head.norm`` weight key (matches checkpoint)."""

    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps)


class MTPHead(nn.Module):
    """DeepSeek-V3 / GLM-5.2 NextN (MTP) draft head.

    Parameter tree (matches keys produced by sanitize() renaming
    ``model.layers.78.*`` -> ``mtp_head.*``)::

        mtp_head.enorm.weight                 # RMSNorm on prev hidden
        mtp_head.hnorm.weight                 # RMSNorm on candidate embedding
        mtp_head.eh_proj.weight               # concat([enorm(h), hnorm(emb)]) -> hidden
        mtp_head.input_layernorm.weight       #  }  these belong to the inner
        mtp_head.post_attention_layernorm.weight  #  decoder block (GlmMoeDsaDecoderLayer)
        mtp_head.self_attn.*                  # full MLA attention
        mtp_head.mlp.*                        # 256-expert MoE (after stacking)
        mtp_head.shared_head.norm.weight      # final norm before lm_head

    The inner decoder block is reused verbatim (so attention/MoE shapes match the
    served model exactly); we only add the NextN projection (enorm/hnorm/eh_proj)
    and the shared_head.norm. Because the inner block is a named submodule, its
    weights load via the standard path after we stack experts + reshape kv_b_proj
    in ``load_weights``.
    """

    def __init__(self, config: Any, shard_path: Optional[str] = None):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        eps = config.rms_norm_eps

        # NextN projection (the part unique to MTP).
        self.enorm = nn.RMSNorm(hidden, eps=eps)
        self.hnorm = nn.RMSNorm(hidden, eps=eps)
        # Input is concat([enorm(h_prev), hnorm(emb)]) -> 2*hidden, output hidden.
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)

        # The transformer block: reused from the parent family so attention/MoE
        # match exactly. Built at layer_idx = num_hidden_layers (the MTP layer).
        # We LIFT the block's named children directly onto self (rather than
        # nesting under a ``_block`` submodule) so the parameter-tree keys match
        # the checkpoint's ``mtp_head.{input_layernorm,self_attn,mlp,...}.*``.
        layer_idx = config.num_hidden_layers

        # GlmMoeDsaAttention.__init__ reads config.indexer_types[layer_idx] to set
        # skip_topk. The config builds indexer_types for layers 0..num_hidden_layers-1
        # (length num_hidden_layers), so the MTP layer is out of range. Extend it
        # so the MTP head's attention is treated as a "full" indexer layer (it
        # carries its own indexer weights in the checkpoint). Mutating config is
        # acceptable here because MTPHead is only built once, at Model.__init__.
        idx_types = getattr(config, "indexer_types", None)
        if idx_types is not None and len(idx_types) <= layer_idx:
            # Pad with "full" entries up to and including the MTP layer.
            idx_types = list(idx_types) + ["full"] * (layer_idx + 1 - len(idx_types))
            config.indexer_types = idx_types

        _block, self._returns_topk = _make_decoder_layer(config, layer_idx)
        for _name in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp"):
            setattr(self, _name, getattr(_block, _name))

        # The MTP head is REPLICATED on every node (not sharded), so its MoE must
        # NOT participate in any distributed collective. DeepseekV32MoE defaults
        # sharding_group=None, but we assert/force it here to guarantee the draft
        # forward runs locally without blocking on all_sum/sum_gradients (which
        # would deadlock since only one rank enters the draft path).
        if hasattr(self.mlp, "sharding_group"):
            self.mlp.sharding_group = None

        # Final norm applied before the shared lm_head.
        self.shared_head = _SharedHead(hidden, eps)

        # Self-load weights from the standalone MTP shard if provided. The shard
        # is kept OUTSIDE the main model index so exo's download-integrity check
        # doesn't "repair" it back to the upstream (MTP-stripped) index. Keys in
        # the shard are model.layers.78.*; load_weights renames them internally.
        if shard_path is not None:
            import os as _os
            if _os.path.exists(shard_path):
                import mlx.core as _mx
                _raw = dict(_mx.load(shard_path))
                self.load_weights(_raw, strict=False)
            # else: leave params uninitialized; MTPDraftModel will detect and
            # the speculative path falls back to non-speculative gracefully.

    def load_weights(self, weights: dict[str, mx.array], strict: bool = True) -> None:
        """Load raw weights, stacking experts and reshaping kv_b_proj into
        embed_q/unembed_out exactly as the upstream body sanitize step does,
        then hand the cleaned dict to nn.Module.load_weights.

        Accepts keys with either prefix:
          - ``mtp_head.*``  (if sanitize renamed them), or
          - ``model.layers.78.*`` (the raw standalone shard; we strip the
            ``model.layers.78.`` prefix to get the bare submodule path).
        """
        mine: dict[str, mx.array] = {}
        for k, v in weights.items():
            if k.startswith("mtp_head."):
                mine[k[len("mtp_head."):]] = v
            elif k.startswith("model.layers.78."):
                mine[k[len("model.layers.78."):]] = v

        n_experts = self.config.n_routed_experts
        stacked: dict[str, mx.array] = {}
        consumed: set[str] = set()
        for proj in ("gate_proj", "down_proj", "up_proj"):
            for kind in ("weight", "scales", "biases"):
                key0 = f"mlp.experts.0.{proj}.{kind}"
                if key0 in mine:
                    to_join = []
                    for e in range(n_experts):
                        ek = f"mlp.experts.{e}.{proj}.{kind}"
                        to_join.append(mine[ek])
                        consumed.add(ek)
                    stacked[f"mlp.switch_mlp.{proj}.{kind}"] = mx.stack(to_join)
        for k in consumed:
            mine.pop(k, None)
        mine.update(stacked)

        # Reshape self_attn.kv_b_proj -> embed_q + unembed_out (MultiLinear layout).
        kv_key = "self_attn.kv_b_proj.weight"
        if kv_key in mine:
            v = mine.pop(kv_key)
            head_dim = self.config.qk_nope_head_dim + self.config.v_head_dim
            num_heads = self.config.num_attention_heads
            if v.ndim == 2:
                v = v.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(v[:, : self.config.qk_nope_head_dim, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, self.config.qk_nope_head_dim :, :])
            mine["self_attn.embed_q.weight"] = wk
            mine["self_attn.unembed_out.weight"] = wv

        super().load_weights(list(mine.items()), strict=strict)

    def __call__(
        self,
        h_prev: mx.array,
        next_token_emb: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """One MTP forward. Returns hidden state [B, L, hidden] for lm_head.

        Canonical DeepSeek-V3 NextN order (verified against the upstream
        modeling code + vLLM/HF references):
            norm_e = enorm(embedding)         # enorm normalizes the EMBEDDING
            norm_h = hnorm(h_prev)            # hnorm normalizes the HIDDEN
            eh = eh_proj(concat([norm_e, norm_h], dim=-1))  # embedding first
        """
        from mlx_lm.models.base import create_attention_mask

        norm_e = self.enorm(next_token_emb)
        norm_h = self.hnorm(h_prev)
        eh = self.eh_proj(mx.concatenate([norm_e, norm_h], axis=-1))

        h = eh
        # At decode time L=1, causal mask is trivial; build it only if a cache
        # with offset>0 is present (matches the parent model's pattern).
        mask = create_attention_mask(h, cache[0] if cache else None, return_array=True)

        # Run the lifted decoder-layer attention + MoE. GlmMoeDsaAttention returns
        # (output, topk_indices) and accepts prev_topk_indices; the MTP head is a
        # standalone single layer so prev_topk_indices=None on the first step.
        if self._returns_topk:
            r, _topk = self.self_attn(self.input_layernorm(h), mask, cache, None)
        else:
            r = self.self_attn(self.input_layernorm(h), mask, cache)
        h = h + r
        r = self.mlp(self.post_attention_layernorm(h))
        h = h + r
        return self.shared_head.norm(h)

    def make_draft_cache(self):
        """Build the MTP head's own KV cache.

        This is kept SEPARATE from the model's main cache list — the MTP draft
        cache is managed entirely by speculative_generate / MTPDraftModel and
        never appears in the cache list that auto_parallel's patched_call walks
        (which would break on the untouched MTP cache's None .keys during warmup).
        The shape matches a full-indexer decoder layer: CacheList(KVCache, KVCache).
        """
        from mlx_lm.models.cache import CacheList, KVCache

        return CacheList(KVCache(), KVCache())


class MTPDraftModel:
    """Wraps a parent GLM-5.2 ``Model`` + its ``mtp_head`` to produce draft tokens.

    The drafter shares the parent's ``embed_tokens`` and ``lm_head`` (both
    replicated, never sharded) and runs the ``mtp_head`` locally — NO distributed
    collectives. ``seed_from_parent`` captures the last verified hidden state to
    start a draft chain; each ``draft`` call extends the chain by ``n_draft``
    tokens.
    """

    def __init__(self, parent_model: nn.Module):
        self.parent = parent_model
        self.head: MTPHead = parent_model.mtp_head
        self._last_h: Optional[mx.array] = None
        # Build the draft cache ONCE and reuse across draft calls. It accumulates
        # KV state as the draft chain extends, and is rewound on rejection by
        # speculative_generate via trim_prompt_cache.
        self.draft_cache = self.head.make_draft_cache()

    def seed_from_parent(self, parent_hidden: mx.array) -> None:
        """Capture the parent's last hidden state ([B, L, hidden] -> [B,1,hidden])."""
        self._last_h = parent_hidden[:, -1:, :]

    def draft(
        self,
        token_ids: mx.array,
        draft_cache: Any,
        n_draft: int,
    ) -> list[int]:
        """Produce up to ``n_draft`` candidate next tokens via the MTP head."""
        if self._last_h is None:
            return []
        embed = self.parent.model.embed_tokens
        lm_head = self.parent.lm_head

        drafts: list[int] = []
        h_prev = self._last_h
        # Normalize token_ids to 2-dim [B, L] and take the last position [B, 1].
        # Callers may pass a 1-dim array (e.g. mx.array([16])).
        y = token_ids
        if y.ndim == 1:
            y = y[None, :]
        cur_token = y[:, -1:]  # [B, 1]

        for _ in range(n_draft):
            emb = embed(cur_token)  # [B, 1, hidden]
            h_out = self.head(h_prev, emb, cache=draft_cache)  # [B, 1, hidden]
            logits = lm_head(h_out)  # [B, 1, vocab]
            next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            drafts.append(next_tok)
            h_prev = h_out
            cur_token = mx.array([[next_tok]], dtype=mx.uint32)

        return drafts


class NGramSpeculator:
    """Prompt-lookup / n-gram speculative drafter (zero-weight fallback).

    Looks up the longest suffix of the recent token sequence in the prompt +
    generated history and proposes the tokens that followed it previously. Works
    with any model, but only helps on text that repeats the prompt (summarization,
    RAG, code completion) — low acceptance on novel prose.

    Used as the fallback when MTP is requested but the head is unavailable, and as
    the primary path when EXO_NGRAM_SPECULATIVE=1.
    """

    def __init__(self, ngram_size: int = 3, max_draft: int = 4):
        self.ngram_size = ngram_size
        self.max_draft = max_draft
        self._history: list[int] = []

    def add_tokens(self, tokens) -> None:
        self._history.extend(int(t) for t in tokens)

    def propose_drafts(self, recent) -> list[int]:
        """Return tokens that historically followed the n-gram ending at ``recent``.

        Returns the continuation after the FIRST occurrence of the needle's
        n-gram (earliest match gives the longest continuation and matches the
        original test contract). Empty if the n-gram never appeared.
        """
        if len(recent) < self.ngram_size:
            return []
        needle = tuple(int(t) for t in recent[-self.ngram_size:])
        hist = self._history
        n = len(hist)
        ng = self.ngram_size
        # Search forwards for the first occurrence.
        for i in range(n - ng):
            if tuple(hist[i : i + ng]) == needle:
                start = i + ng
                end = min(start + self.max_draft, n)
                drafts = hist[start:end]
                if drafts:
                    return drafts
        return []

    def reset(self) -> None:
        self._history.clear()
