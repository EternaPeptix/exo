import os

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?


def _parse_optional_int(env_var: str) -> int | None:
    raw = os.environ.get(env_var)
    if raw is None or raw == "":
        return None
    return int(raw)


KV_GROUP_SIZE: int = int(os.environ.get("EXO_KV_GROUP_SIZE", "64"))
# int8 MLA latent KV (see mlx-lm MLACacheList.to_quantized). Indexer cache stays fp16.
KV_BITS: int | None = _parse_optional_int("EXO_KV_BITS")
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
# Used only for non-MLA models without make_cache(); MLA models use KV_BITS instead.
KV_CACHE_BITS: int | None = _parse_optional_int("EXO_KV_CACHE_BITS")

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True

# Expert-parallel sharding: when enabled, attention is replicated (not head-split)
# while MoE experts are weight-sharded.
EXO_EXPERT_PARALLEL: bool = os.environ.get("EXO_EXPERT_PARALLEL", "").lower() in ("1", "true", "yes")

# MTP (Multi-Token Prediction) speculative decoding: uses model's NextN prediction head.
EXO_MTP_SPECULATIVE: bool = os.environ.get("EXO_MTP_SPECULATIVE", "").lower() in ("1", "true", "yes")
MTP_NUM_DRAFT_TOKENS: int = int(os.environ.get("EXO_MTP_NUM_DRAFT", "1"))

# N-gram speculative decoding: uses tokenizer n-gram statistics.
EXO_NGRAM_SPECULATIVE: bool = os.environ.get("EXO_NGRAM_SPECULATIVE", "").lower() in ("1", "true", "yes")
TRUST_REMOTE_CODE: bool = True
