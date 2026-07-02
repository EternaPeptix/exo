#!/usr/bin/env python3
"""Patch auto_parallel.py to add expert-parallel sharding strategy for MLA MoE models."""
import re
import sys

filepath = sys.argv[1]
content = open(filepath).read()

# 1. Add 'import os' at the top if not present
if 'import os' not in content[:300]:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            lines.insert(i, 'import os')
            break
    content = '\n'.join(lines)
    print("  Added 'import os'")

# 2. Add GlmMoeDsaModel import if not present
if 'GlmMoeDsaModel' not in content:
    old_import = 'from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel'
    if old_import in content:
        content = content.replace(old_import, old_import + '\nfrom mlx_lm.models.glm_moe_dsa import Model as GlmMoeDsaModel', 1)
        print("  Added GlmMoeDsaModel import")

# 3. Insert the expert-parallel strategy class right before 'class ShardedMoE'
sharded_moe_class = 'class ShardedMoE(CustomMlxLayer):'

new_strategy = '''class DeepSeekExpertParallelShardingStrategy(DeepSeekShardingStrategy):
    """Expert-parallel sharding for MLA-based MoE models.

    Key difference from DeepSeekShardingStrategy: attention is NOT sharded
    (kept replicated across all ranks). This avoids the MLA latent KV cache
    being replicated across tensor-parallel ranks, which:
    1. Eliminates the token corruption risk from int8-quantized latent KV
       diverging across ranks during dequantization.
    2. Saves the all-reduce on attention output (both ranks compute the
       same result independently).
    3. Trades 2x attention weight memory + 2x attention compute for
       correctness and simpler KV cache management.

    MoE experts are still weight-sharded (tensor-parallel) with all-reduce,
    same as DeepSeekShardingStrategy. Since ~98% of model parameters are in
    expert weights, the memory overhead of replicated attention is small.
    """

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(DeepseekV3Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # ATTENTION: Keep replicated - do NOT shard any attention weights.
            # This is the key difference from DeepSeekShardingStrategy.
            # MLA latent KV stays local to each rank (no replication across
            # tensor-parallel head splits, no all-reduce divergence).

            # MLP / MoE: Shard same as tensor-parallel
            if isinstance(layer.mlp, (DeepseekV3MLP, DeepseekV32MLP)):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
            else:
                # MoE layer
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

            mx.eval(layer)
            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


'''

if sharded_moe_class in content:
    content = content.replace(sharded_moe_class, new_strategy + sharded_moe_class, 1)
    print("  Added DeepSeekExpertParallelShardingStrategy class before ShardedMoE")
else:
    print("  ERROR: Could not find ShardedMoE class")
    sys.exit(1)

# 4. Add _use_expert_parallel flag in tensor_auto_parallel
old_tensor_auto = '    if isinstance(model, (LlamaModel, Ministral3Model)):'
new_tensor_auto = '    _use_expert_parallel = os.environ.get("EXO_EXPERT_PARALLEL", "").lower() in ("1", "true", "yes")\n\n    if isinstance(model, (LlamaModel, Ministral3Model)):'
if old_tensor_auto in content:
    content = content.replace(old_tensor_auto, new_tensor_auto, 1)
    print("  Added _use_expert_parallel flag")

# 5. Replace DeepSeek strategy selection to add expert-parallel option
pattern = r'    elif isinstance\(model, \(DeepseekV3Model, DeepseekV32Model, KimiK25Model\)\):\n        tensor_parallel_sharding_strategy = DeepSeekShardingStrategy\(\n            group,\n            all_to_sharded_linear,\n            sharded_to_all_linear,\n            all_to_sharded_linear_in_place,\n            sharded_to_all_linear_in_place,\n        \)'

replacement = '''    elif isinstance(model, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        if _use_expert_parallel:
            logger.info(
                "Expert-parallel: attention replicated, MoE weight-sharded "
                "(avoids MLA latent KV replication)"
            )
            tensor_parallel_sharding_strategy = DeepSeekExpertParallelShardingStrategy(
                group,
                all_to_sharded_linear,
                sharded_to_all_linear,
                all_to_sharded_linear_in_place,
                sharded_to_all_linear_in_place,
            )
        else:
            tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
                group,
                all_to_sharded_linear,
                sharded_to_all_linear,
                all_to_sharded_linear_in_place,
                sharded_to_all_linear_in_place,
            )'''

new_content = re.sub(pattern, replacement, content, count=1)
if new_content != content:
    content = new_content
    print("  Added expert-parallel strategy selection for DeepSeek models")
else:
    print("  WARNING: Could not patch DeepSeek strategy selection (regex no match)")

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
