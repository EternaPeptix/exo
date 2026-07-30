from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
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


def _is_full_coverage(shard: ShardMetadata) -> bool:
    """True if a download with this shard metadata fetched every weight file."""
    match shard:
        case PipelineShardMetadata():
            return shard.start_layer == 0 and shard.end_layer == shard.n_layers
        case _:
            # Tensor/CFG downloads always fetch the full repository.
            return True


def download_covers_shard(dp: DownloadProgress, required: ShardMetadata) -> bool:
    """True if a completed download holds every file ``required`` needs.

    Downloads are keyed by model id, but with shard-scoped downloads a node
    may only hold the layer range of a previous placement. A full download
    covers everything; a partial pipeline download only covers pipeline
    shards inside its layer range.
    """
    if not isinstance(dp, DownloadCompleted):
        return False
    have = dp.shard_metadata
    if have.model_card.model_id != required.model_card.model_id:
        return False
    if _is_full_coverage(have):
        return True
    if not isinstance(required, PipelineShardMetadata):
        return False
    return (
        have.n_layers == required.n_layers
        and have.start_layer <= required.start_layer
        and have.end_layer >= required.end_layer
    )


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
