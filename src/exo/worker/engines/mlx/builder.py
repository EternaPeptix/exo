import contextlib
import json
import os
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

from .cache import KVPrefixCache
from .generator.kimi_k3_dspark import (
    DSparkConfigurationError,
    KimiK3DSparkConfig,
    LoadedMlxDSpark,
    kimi_k3_dspark_config,
    load_replicated_mlx_dspark,
    preflight_mlx_dspark_segmented_sdpa,
)
from .types import Model
from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
    rank_agreed_local_stage,
)
from .vision import VisionProcessor


@dataclass
class MlxBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None
    dspark_config: KimiK3DSparkConfig | None = None
    dspark: LoadedMlxDSpark | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        shard = bound_instance.bound_shard

        def configure_dspark() -> KimiK3DSparkConfig | None:
            config = kimi_k3_dspark_config(
                is_pipeline=isinstance(shard, PipelineShardMetadata),
                is_batch=os.environ.get("EXO_NO_BATCH") != "1",
            )
            if config is None:
                return None
            preflight_mlx_dspark_segmented_sdpa()
            if not isinstance(shard, TensorShardMetadata) or shard.world_size != 2:
                raise DSparkConfigurationError(
                    "Kimi K3 DSpark first canary requires tensor parallel world size 2"
                )
            if self.group is None or self.group.size() != 2:
                raise DSparkConfigurationError(
                    "Kimi K3 DSpark requires an initialized two-rank MLX group"
                )
            return config

        def dspark_config_contract(config: KimiK3DSparkConfig | None) -> str:
            if config is None:
                return "disabled"
            return json.dumps(
                {
                    "checkpoint": str(config.checkpoint_path),
                    "verify_width": config.verify_width,
                    "round_telemetry": config.round_telemetry,
                    "model_id": config.model_id,
                    "revision": config.revision,
                    "config_sha256": config.config_sha256,
                    "model_bytes": config.model_bytes,
                    "model_sha256": config.model_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )

        self.dspark_config = rank_agreed_local_stage(
            "Kimi K3 DSpark configuration preflight",
            self.group,
            configure_dspark,
            dspark_config_contract,
        )
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
        ) = yield from load_mlx_items(bound_instance, self.group)
        if self.dspark_config is not None:
            dspark_config = self.dspark_config

            def load_dspark() -> LoadedMlxDSpark:
                if self.vision_processor is not None:
                    raise DSparkConfigurationError(
                        "Kimi K3 DSpark does not support vision models"
                    )
                return load_replicated_mlx_dspark(
                    dspark_config,
                    self.inference_model,
                )

            self.dspark = rank_agreed_local_stage(
                "Kimi K3 replicated draft load",
                self.group,
                load_dspark,
                lambda loaded: json.dumps(
                    {
                        "vision_processor_present": self.vision_processor is not None,
                        "placement": loaded.placement,
                        "verify_width": loaded.verify_width,
                        "drafter_class": (
                            f"{type(loaded.drafter).__module__}."
                            f"{type(loaded.drafter).__qualname__}"
                        ),
                        "proposer_class": (
                            f"{type(loaded.proposer).__module__}."
                            f"{type(loaded.proposer).__qualname__}"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group
        with contextlib.suppress(NameError, AttributeError):
            del self.dspark

    def build(
        self,
    ) -> Engine:
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = None if self.dspark is not None else KVPrefixCache(self.group)

        device_rank = 0 if self.group is None else self.group.rank()
        if self.dspark is not None and os.environ.get("EXO_NO_BATCH") != "1":
            raise DSparkConfigurationError(
                "Kimi K3 DSpark requires EXO_NO_BATCH=1 for its entire runner lifetime"
            )
        if self.dspark is not None or os.environ.get("EXO_NO_BATCH"):
            logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
                dspark=self.dspark,
            )
        else:
            logger.info("using BatchGenerator")
            return BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
