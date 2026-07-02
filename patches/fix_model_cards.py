#!/usr/bin/env python3
"""Fix model_cards.py with robust regex matching."""
import re
import sys

path = sys.argv[1]
content = open(path).read()

# 1. Add supports_expert_parallel property after supports_tensor in ConfigData
pattern = r'(    @property\n    def supports_tensor\(self\) -> bool:\n        return self\.architectures in \[\n.*?\n        \])'
match = re.search(pattern, content, re.DOTALL)
if match:
    insert_after = match.group(0)
    new_prop = insert_after + '''

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
        ]'''
    content = content[:match.start()] + new_prop + content[match.end():]
    print("  Added supports_expert_parallel property to ConfigData")
else:
    print("  ERROR: Could not find supports_tensor property")
    sys.exit(1)

# 2. Add supports_expert_parallel field to ModelCard class
old_fields = "    supports_tensor: bool\n    num_key_value_heads: PositiveInt | None = None"
new_fields = "    supports_tensor: bool\n    supports_expert_parallel: bool = False\n    num_key_value_heads: PositiveInt | None = None"
if old_fields in content:
    content = content.replace(old_fields, new_fields, 1)
    print("  Added supports_expert_parallel field to ModelCard")
else:
    print("  WARNING: Could not find ModelCard fields")

# 3. Add to from_config / load method
old_create = "            supports_tensor=config_data.supports_tensor,\n            num_key_value_heads=config_data.num_key_value_heads,"
new_create = "            supports_tensor=config_data.supports_tensor,\n            supports_expert_parallel=config_data.supports_expert_parallel,\n            num_key_value_heads=config_data.num_key_value_heads,"
if old_create in content:
    content = content.replace(old_create, new_create, 1)
    print("  Added to from_config")
else:
    print("  WARNING: Could not find from_config creation")

with open(path, "w") as f:
    f.write(content)
print("Patched model_cards.py successfully")
