from __future__ import annotations

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    download_covers_shard,
)
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)

MODEL_ID = ModelId("test-org/test-model")


def _card(n_layers: int = 4) -> ModelCard:
    return ModelCard(
        model_id=MODEL_ID,
        storage_size=Memory.from_bytes(4096),
        n_layers=n_layers,
        hidden_size=8,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _completed(shard: ShardMetadata) -> DownloadCompleted:
    return DownloadCompleted(
        node_id=NodeId("node"),
        shard_metadata=shard,
        total=Memory.from_bytes(4096),
    )


def _pipeline(
    start_layer: int,
    end_layer: int,
    *,
    device_rank: int = 0,
    world_size: int = 1,
    n_layers: int = 4,
) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_card(n_layers),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
    )


def _tensor(
    *,
    rank: int = 0,
    world_size: int = 2,
    start_layer: int = 0,
    end_layer: int = 4,
    n_layers: int = 4,
) -> TensorShardMetadata:
    return TensorShardMetadata(
        model_card=_card(n_layers),
        device_rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
    )


def _cfg(
    *,
    device_rank: int = 0,
    world_size: int = 4,
    start_layer: int = 0,
    end_layer: int = 2,
    n_layers: int = 4,
    cfg_rank: int = 0,
    cfg_world_size: int = 2,
    pipeline_rank: int = 0,
    pipeline_world_size: int = 2,
) -> CfgShardMetadata:
    card = _card(n_layers).model_copy(
        update={"tasks": [ModelTask.TextToImage], "uses_cfg": True}
    )
    return CfgShardMetadata(
        model_card=card,
        device_rank=device_rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        cfg_rank=cfg_rank,
        cfg_world_size=cfg_world_size,
        pipeline_rank=pipeline_rank,
        pipeline_world_size=pipeline_world_size,
    )


def test_pipeline_range_covers_contained_subshard_across_placements() -> None:
    completed = _completed(_pipeline(0, 3, device_rank=0, world_size=1))

    assert download_covers_shard(
        completed,
        _pipeline(1, 2, device_rank=3, world_size=8),
    )
    assert not download_covers_shard(completed, _pipeline(2, 4))
    assert not download_covers_shard(completed, _pipeline(0, 3, n_layers=8))


def test_tensor_rank_zero_does_not_cover_rank_one() -> None:
    completed = _completed(_tensor(rank=0, world_size=2))

    assert download_covers_shard(completed, _tensor(rank=0, world_size=2))
    assert not download_covers_shard(completed, _tensor(rank=1, world_size=2))


def test_tensor_tp2_does_not_cover_tp4() -> None:
    completed = _completed(_tensor(rank=0, world_size=2))

    assert not download_covers_shard(completed, _tensor(rank=0, world_size=4))


def test_tensor_layer_metadata_must_match_exactly() -> None:
    completed = _completed(_tensor(start_layer=0, end_layer=4))

    assert not download_covers_shard(
        completed,
        _tensor(start_layer=0, end_layer=3),
    )


def test_pipeline_completion_does_not_cover_tensor_or_cfg() -> None:
    completed = _completed(_pipeline(0, 4))

    assert not download_covers_shard(completed, _tensor())
    assert not download_covers_shard(completed, _cfg())
    assert not download_covers_shard(_completed(_tensor()), _pipeline(0, 4))
    assert not download_covers_shard(_completed(_cfg()), _pipeline(0, 4))


def test_cfg_full_repository_covers_changed_cfg_placement() -> None:
    completed = _completed(_cfg())

    assert download_covers_shard(completed, _cfg())
    assert download_covers_shard(
        completed,
        _cfg(
            device_rank=3,
            world_size=6,
            start_layer=2,
            end_layer=4,
            cfg_rank=1,
            pipeline_rank=2,
            pipeline_world_size=3,
        ),
    )
    assert not download_covers_shard(completed, _cfg(n_layers=8, end_layer=4))
