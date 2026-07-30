from pydantic import Field

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class TextGeneration(BaseCommand):
    task_params: TextGenerationTaskParams


class ImageGeneration(BaseCommand):
    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    task_params: ImageEditsTaskParams


class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class TaskCancelled(BaseCommand):
    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    since_idx: int


class SeedSource(FrozenModel):
    """A peer node that can serve model weight files over HTTP.

    ``base_urls`` are ordered by preference (fastest link first), e.g.
    ``["http://169.254.1.2:52416", "http://192.168.0.3:52416"]``.
    """

    node_id: NodeId
    base_urls: list[str] = []


class StartDownload(BaseCommand):
    target_node_id: NodeId
    shard_metadata: ShardMetadata
    # Peers that already hold (part of) this model and can seed files to us.
    # When empty, the downloader falls back to HuggingFace for every file.
    seed_sources: list[SeedSource] = []


class DeleteDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class AddCustomModelCard(BaseCommand):
    model_card: ModelCard


class DeleteCustomModelCard(BaseCommand):
    model_id: ModelId


class SetInstanceLink(BaseCommand):
    link_id: InstanceLinkId
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class DeleteInstanceLink(BaseCommand):
    link_id: InstanceLinkId


DownloadCommand = StartDownload | DeleteDownload | CancelDownload


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | TaskCancelled
    | TaskFinished
    | SendInputChunk
    | AddCustomModelCard
    | DeleteCustomModelCard
    | SetInstanceLink
    | DeleteInstanceLink
)


class ForwarderCommand(FrozenModel):
    origin: SystemId
    command: Command


class ForwarderDownloadCommand(FrozenModel):
    origin: SystemId
    command: DownloadCommand
