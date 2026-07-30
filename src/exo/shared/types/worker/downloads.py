from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class DownloadProgressData(FrozenModel):
    total: Memory
    downloaded: Memory
    downloaded_this_session: Memory

    completed_files: int
    total_files: int

    speed: float
    eta_ms: int

    files: dict[str, "DownloadProgressData"]


class BaseDownloadProgress(TaggedModel):
    node_id: NodeId
    shard_metadata: ShardMetadata
    model_directory: str = ""


class DownloadPending(BaseDownloadProgress):
    downloaded: Memory = Memory()
    total: Memory = Memory()


class DownloadCompleted(BaseDownloadProgress):
    total: Memory
    read_only: bool = False


class DownloadFailed(BaseDownloadProgress):
    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    download_progress: DownloadProgressData


DownloadProgress = (
    DownloadPending | DownloadCompleted | DownloadFailed | DownloadOngoing
)


def download_covers_shard(dp: DownloadProgress, required: ShardMetadata) -> bool:
    """True if a completed download holds every file ``required`` needs.

    Pipeline weights are layer-addressable, so a completed pipeline range can
    cover another pipeline shard contained within it regardless of placement.
    Tensor completions require exact rank/world/layer metadata because one
    rank-local checkpoint cannot serve another rank. CFG downloads always
    contain the full repository, so CFG-to-CFG coverage only depends on model
    and layer-count identity. Different sharding modes never cover each other.
    """
    if not isinstance(dp, DownloadCompleted):
        return False
    have = dp.shard_metadata
    if have.model_card.model_id != required.model_card.model_id:
        return False
    match have, required:
        case PipelineShardMetadata(), PipelineShardMetadata():
            return (
                have.n_layers == required.n_layers
                and have.start_layer <= required.start_layer
                and have.end_layer >= required.end_layer
            )
        case TensorShardMetadata(), TensorShardMetadata():
            return (
                have.device_rank == required.device_rank
                and have.world_size == required.world_size
                and have.start_layer == required.start_layer
                and have.end_layer == required.end_layer
                and have.n_layers == required.n_layers
            )
        case CfgShardMetadata(), CfgShardMetadata():
            return have.n_layers == required.n_layers
        case _:
            return False


class ModelSafetensorsIndexMetadata(BaseModel):
    total_size: PositiveInt | None = None


class ModelSafetensorsIndex(BaseModel):
    metadata: ModelSafetensorsIndexMetadata | None
    weight_map: dict[str, str]


class FileListEntry(BaseModel):
    type: Literal["file", "directory"]
    path: str
    size: int | None = None


class RepoFileDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    file_path: str
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    speed: float
    eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    start_time: float

    model_config = ConfigDict(frozen=True)


class RepoDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    shard: ShardMetadata
    completed_files: int
    total_files: int
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    overall_speed: float
    overall_eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    file_progress: dict[str, RepoFileDownloadProgress] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
