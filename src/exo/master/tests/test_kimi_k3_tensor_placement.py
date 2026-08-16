from anyio import Path

from exo.master.placement import place_instance
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_rdma_connection,
    create_socket_connection,
)
from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.profiling import NodeRdmaCtlStatus
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta, MlxJacclInstance
from exo.shared.types.worker.shards import Sharding, TensorShardMetadata


async def test_builtin_kimi_k3_card_admits_two_node_tensor_placement() -> None:
    card = await ModelCard.load_from_path(
        Path(
            RESOURCES_DIR
            / "inference_model_cards"
            / "kernelpool--Kimi-K3-2bit-UVMAX.toml"
        )
    )
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(1))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )
    node_memory = {
        node_a: create_node_memory(500_000_000_000),
        node_b: create_node_memory(500_000_000_000),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    command = PlaceInstance(
        command_id=CommandId("place-kimi-k3-tp2"),
        model_card=card,
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        {node_a: [Backend.MlxMetal], node_b: [Backend.MlxMetal]},
        node_rdma_ctl={
            node_a: NodeRdmaCtlStatus(enabled=True),
            node_b: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert card.supports_tensor is True
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    shards = tuple(instance.shard_assignments.runner_to_shard.values())
    assert len(shards) == 2
    assert all(isinstance(shard, TensorShardMetadata) for shard in shards)
    assert {shard.device_rank for shard in shards} == {0, 1}
    assert {shard.world_size for shard in shards} == {2}
