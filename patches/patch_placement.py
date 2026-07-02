#!/usr/bin/env python3
"""Patch placement.py to handle ExpertParallel sharding."""
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# Add ExpertParallel validation after Tensor validation
old = '''    if command.sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):'''

new = '''    if command.sharding == Sharding.ExpertParallel:
        if not command.model_card.supports_expert_parallel:
            raise ValueError(
                f"Requested ExpertParallel sharding but this model does not support it: {command.model_card.model_id}"
            )
        # Expert-parallel replicates attention, so we only need hidden_size
        # divisibility is not required (attention is not head-split).
        # We just need enough memory on each node for the full attention + half experts.
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if len(cycle) >= 2  # Expert-parallel requires at least 2 nodes
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "No cycles with sufficient memory for expert-parallel sharding "
                "(requires at least 2 nodes)"
            )

    if command.sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):'''

assert old in content, "Could not find Pipeline/DeepSeek validation"
content = content.replace(old, new, 1)

# Also handle single-node case: ExpertParallel should fall back to Pipeline
old_single = '''    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        command = command.model_copy(
            update={
                "instance_meta": InstanceMeta.MlxRing,
                "sharding": Sharding.Pipeline,
            }
        )'''

new_single = '''    # Single-node: force Pipeline/Ring (Tensor, ExpertParallel and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        command = command.model_copy(
            update={
                "instance_meta": InstanceMeta.MlxRing,
                "sharding": Sharding.Pipeline,
            }
        )'''

if old_single in content:
    content = content.replace(old_single, new_single, 1)

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
