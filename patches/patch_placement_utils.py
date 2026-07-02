#!/usr/bin/env python3
"""Patch placement_utils.py to add expert-parallel shard assignments."""
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# Add ExpertParallelShardMetadata import
old_import = 'from exo.shared.types.worker.shards import'
if old_import in content:
    # Check if already imported
    if 'ExpertParallelShardMetadata' not in content:
        content = content.replace(
            old_import,
            old_import + '\nfrom exo.shared.types.worker.shards import ExpertParallelShardMetadata',
            1
        )
        print("  Added ExpertParallelShardMetadata import")

# Add get_shard_assignments_for_expert_parallel function before get_shard_assignments
old_func = '''def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )'''

new_func = '''def get_shard_assignments_for_expert_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    """Create shard assignments for expert-parallel execution.

    Same structure as tensor-parallel (each node has all layers), but uses
    ExpertParallelShardMetadata to signal the worker to use the expert-parallel
    sharding strategy (attention replicated, MoE split).
    """
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = ExpertParallelShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )
        case Sharding.ExpertParallel:
            return get_shard_assignments_for_expert_parallel(
                model_card=model_card,
                cycle=cycle,
            )'''

assert old_func in content, "Could not find get_shard_assignments function"
content = content.replace(old_func, new_func, 1)

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
