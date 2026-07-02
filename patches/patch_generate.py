#!/usr/bin/env python3
"""Patch generate.py to use speculative_generate when speculative decoding is enabled."""
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# 1. Add import for speculative_generate
old_import = 'from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill'
new_import = '''from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.generator.speculative_generate import speculative_generate'''

if old_import in content:
    content = content.replace(old_import, new_import, 1)
    print("  Added speculative_generate import")
else:
    print("  WARNING: Could not find remote_prefill import to add speculative_generate")

# 2. Replace the stream_generate call in the decode loop with speculative_generate
old_decode = '''    for completion_tokens, out in enumerate(
        stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=last_token,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=caches,
            prefill_step_size=1,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
        ),
        start=1,
    ):'''

new_decode = '''    for completion_tokens, out in enumerate(
        speculative_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=last_token,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=caches,
            prefill_step_size=1,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
        ),
        start=1,
    ):'''

if old_decode in content:
    content = content.replace(old_decode, new_decode, 1)
    print("  Replaced stream_generate with speculative_generate in decode loop")
else:
    print("  WARNING: Could not find stream_generate call in decode loop to replace")

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
