#!/usr/bin/env python3
"""Patch constants.py to add EXO_EXPERT_PARALLEL and EXO_MTP_SPECULATIVE env vars."""
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# Add env vars after the existing ones
old = '''DEFAULT_TOP_LOGPROBS: int = 5'''

new = '''DEFAULT_TOP_LOGPROBS: int = 5

# Expert-parallel sharding: when enabled, attention is replicated (not head-split)
# while MoE experts are weight-sharded. This avoids MLA latent KV replication
# across tensor-parallel ranks, preventing token corruption with int8 KV cache.
# Set EXO_EXPERT_PARALLEL=1 to enable. Only affects MLA MoE models (DeepSeek V3/V3.2, GLM-5.2).
EXO_EXPERT_PARALLEL: bool = os.environ.get("EXO_EXPERT_PARALLEL", "").lower() in ("1", "true", "yes")

# MTP (Multi-Token Prediction) speculative decoding: when enabled, the model's
# built-in NextN prediction head is used to draft candidate tokens that the main
# model verifies. This can give 1.5-2x decode throughput with zero quality loss.
# Set EXO_MTP_SPECULATIVE=1 to enable. Only affects models with num_nextn_predict_layers > 0.
EXO_MTP_SPECULATIVE: bool = os.environ.get("EXO_MTP_SPECULATIVE", "").lower() in ("1", "true", "yes")

# Number of draft tokens to speculate with MTP. The model's MTP head supports
# num_nextn_predict_layers draft tokens. Default to 1 (safest, highest acceptance rate).
MTP_NUM_DRAFT_TOKENS: int = int(os.environ.get("EXO_MTP_NUM_DRAFT", "1"))'''

assert old in content, "Could not find DEFAULT_TOP_LOGPROBS"
content = content.replace(old, new, 1)

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
