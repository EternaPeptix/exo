#!/usr/bin/env python3
"""Patch model_cards.py to add supports_expert_parallel property and field."""
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# 1. Add supports_expert_parallel property after the supports_tensor property block
# Find the end of the supports_tensor property (the closing ] followed by blank line + @model_validator)
marker = '            ["Gemma4ForConditionalGeneration"],\n        ]\n\n    @model_validator'
if marker not in content:
    # Try alternate spacing
    marker = '"Gemma4ForConditionalGeneration"],\n        ]\n\n    @model_validator'

if marker in content:
    new_prop = '''            ["Gemma4ForConditionalGeneration"],
        ]

    @property
    def supports_expert_parallel(self) -> bool:
        """Models that benefit from expert-parallel sharding.

        Expert-parallel keeps attention replicated (avoiding MLA latent KV
        replication across ranks) while splitting MoE experts across nodes.
        Suitable for MLA-based MoE models where tensor-parallel attention
        replicates the compressed latent KV cache.
        """
        return self.architectures in [
            ["GlmMoeDsaForCausalLM"],
            ["DeepseekV32ForCausalLM"],
            ["DeepseekV3ForCausalLM"],
        ]

    @model_validator'''
    content = content.replace(marker, new_prop, 1)
    print("  Added supports_expert_parallel property to ConfigData")
else:
    print("  ERROR: Could not find supports_tensor property end marker")
    sys.exit(1)

# 2. Add supports_expert_parallel field to ModelCard class
old_field = '    supports_tensor: bool\n    num_key_value_heads: PositiveInt | None = None'
new_field = '    supports_tensor: bool\n    supports_expert_parallel: bool = False\n    num_key_value_heads: PositiveInt | None = None'

if old_field in content:
    content = content.replace(old_field, new_field, 1)
    print("  Added supports_expert_parallel field to ModelCard")
else:
    print("  WARNING: Could not find ModelCard field to patch")

# 3. Add to from_config / load method
old_create = '            supports_tensor=config_data.supports_tensor,\n            num_key_value_heads=config_data.num_key_value_heads,'
new_create = '            supports_tensor=config_data.supports_tensor,\n            supports_expert_parallel=config_data.supports_expert_parallel,\n            num_key_value_heads=config_data.num_key_value_heads,'

if old_create in content:
    content = content.replace(old_create, new_create, 1)
    print("  Added supports_expert_parallel to from_config")
else:
    print("  WARNING: Could not find from_config to patch")

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
