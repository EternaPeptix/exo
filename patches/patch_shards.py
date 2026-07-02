#!/usr/bin/env python3
"""Patch shards.py to add ExpertParallel sharding enum and metadata."""
import re
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# 1. Add ExpertParallel to Sharding enum
old_enum = '''class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"'''
new_enum = '''class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"
    ExpertParallel = "ExpertParallel"'''
assert old_enum in content, "Could not find Sharding enum"
content = content.replace(old_enum, new_enum)

# 2. Add ExpertParallelShardMetadata after TensorShardMetadata
old_tensor = '''@final
class TensorShardMetadata(BaseShardMetadata):
    pass


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata
)'''
new_tensor = '''@final
class TensorShardMetadata(BaseShardMetadata):
    pass


@final
class ExpertParallelShardMetadata(TensorShardMetadata):
    """Expert-parallel shard metadata.

    Same structure as tensor-parallel (each node has all layers), but
    attention is replicated while MoE experts are split across nodes.
    This avoids replicating MLA latent KV across tensor-parallel ranks.
    """


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata | ExpertParallelShardMetadata
)'''
assert old_tensor in content, "Could not find TensorShardMetadata / ShardMetadata TypeAlias"
content = content.replace(old_tensor, new_tensor)

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
