#!/usr/bin/env python3
"""Patch deepseek_v32.py to conditionally keep MTP weights and add MTP head support."""
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# 1. Add import os at the top if not present
if 'import os' not in content[:200]:
    old_first = content.split('\n')[0]
    content = old_first + '\nimport os' + content[len(old_first):]
    print("  Added 'import os'")

# 2. Modify sanitize to conditionally keep MTP weights
old_sanitize = '''    def sanitize(self, weights):
        # Remove multi-token prediction layers
        mpt_layer = self.args.num_hidden_layers
        new_weights = {}
        for k, v in weights.items():
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers" and int(parts[2]) >= mpt_layer:
                continue
            new_weights[k] = v
        weights = new_weights'''

new_sanitize = '''    def sanitize(self, weights):
        # Conditionally keep multi-token prediction (MTP) layers.
        # When EXO_MTP_SPECULATIVE=1, the MTP head weights are preserved
        # so they can be loaded for speculative decoding.
        keep_mtp = os.environ.get("EXO_MTP_SPECULATIVE", "").lower() in ("1", "true", "yes")
        mpt_layer = self.args.num_hidden_layers
        new_weights = {}
        mtp_weights = {}
        for k, v in weights.items():
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers" and int(parts[2]) >= mpt_layer:
                if keep_mtp:
                    mtp_weights[k] = v
                continue
            new_weights[k] = v
        if keep_mtp and mtp_weights:
            import mlx.nn as nn
            num_mtp = getattr(self.args, "num_nextn_predict_layers", 0) or 1
            self.mtp_layers = []
            for i in range(num_mtp):
                layer_idx = mpt_layer + i
                # Create a decoder layer for MTP
                from .deepseek_v32 import DeepseekV32DecoderLayer
                mtp_layer = DeepseekV32DecoderLayer(self.args, layer_idx)
                self.mtp_layers.append(mtp_layer)
            # Store mtp weights for later loading
            self._mtp_weights = mtp_weights
        weights = new_weights'''

if old_sanitize in content:
    content = content.replace(old_sanitize, new_sanitize, 1)
    print("  Patched sanitize to conditionally keep MTP weights")
else:
    print("  WARNING: Could not find sanitize method")

# 3. Add mtp_head property to Model class
# Add after the lm_head definition in __init__
old_init = '''        self.model = DeepseekV32Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)'''

new_init = '''        self.model = DeepseekV32Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_layers: list = []  # Populated in sanitize when EXO_MTP_SPECULATIVE=1'''

if old_init in content:
    content = content.replace(old_init, new_init, 1)
    print("  Added mtp_layers attribute to Model")
else:
    print("  WARNING: Could not find Model __init__ to add mtp_layers")

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
