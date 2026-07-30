from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from exo.download import download_utils
from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import (
    build_model_path_for_shard,
    resolve_existing_model_for_shard,
)
from exo.download.shard_downloader import NoopShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import ForwarderDownloadCommand
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadFailed
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    TensorShardMetadata,
)
from exo.utils.channels import channel
from exo.worker.engines.mlx import rank_local_checkpoint
from exo.worker.engines.mlx.rank_local_checkpoint import (
    RANK_LOCAL_CHECKPOINT_ENV,
    RANK_LOCAL_LOADER_ENV,
    RANK_LOCAL_VERIFY_HASHES_ENV,
    RankLocalConfigurationError,
    load_configured_rank_local_model,
)

MODEL_ID = ModelId("kernelpool/Kimi-K3-2bit-UVMAX")
TEST_CONFIG = '{"model_type":"kimi_k3"}'
TEST_CONFIG_SHA256 = hashlib.sha256(TEST_CONFIG.encode()).hexdigest()


@pytest.fixture(autouse=True)
def _pin_synthetic_config_hash(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        download_utils,
        "SUPPORTED_SOURCE_CONFIG_SHA256",
        TEST_CONFIG_SHA256,
    )
    loader = tmp_path / "rank_local_loader.py"
    loader.write_text("# synthetic audited loader\n", encoding="utf-8")
    monkeypatch.setattr(
        rank_local_checkpoint,
        "SUPPORTED_LOADER_SHA256",
        hashlib.sha256(loader.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv(RANK_LOCAL_LOADER_ENV, str(loader))
    monkeypatch.delenv(RANK_LOCAL_VERIFY_HASHES_ENV, raising=False)


def _json_object(path: Path) -> dict[str, object]:
    value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _card() -> ModelCard:
    return ModelCard(
        model_id=MODEL_ID,
        storage_size=Memory.from_gb(800),
        n_layers=93,
        hidden_size=7168,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _tensor_shard(rank: int = 1, world_size: int = 2) -> TensorShardMetadata:
    return TensorShardMetadata(
        model_card=_card(),
        device_rank=rank,
        world_size=world_size,
        start_layer=0,
        end_layer=93,
        n_layers=93,
    )


def _pipeline_shard() -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_card(),
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=46,
        n_layers=93,
    )


class _UnusedGroup:
    def rank(self) -> int:
        raise AssertionError("non-tensor shard must not inspect the MLX group")

    def size(self) -> int:
        raise AssertionError("non-tensor shard must not inspect the MLX group")


def _rank_checkpoint(root: Path, *, manifest_rank: int = 1) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.json"
    config.write_text(TEST_CONFIG, encoding="utf-8")
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()

    weight = root / "model-00001-of-00185.safetensors"
    weight.write_bytes(b"rank-local-weight")
    weight_sha256 = hashlib.sha256(weight.read_bytes()).hexdigest()
    tensor_name = "language_model.model.embed_tokens.weight"
    tensor_bytes = weight.stat().st_size
    index = {
        "metadata": {"total_size": tensor_bytes},
        "weight_map": {tensor_name: weight.name},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    manifest = {
        "schema": "k3-rank-local-tp/v1",
        "complete": True,
        "source": {
            "repo": str(MODEL_ID),
            "revision": download_utils.SUPPORTED_SOURCE_REVISION,
            "config_sha256": config_sha256,
            "index_sha256": download_utils.SUPPORTED_SOURCE_INDEX_SHA256,
        },
        "runtime": {
            "mlx_lm_commit": download_utils.SUPPORTED_MLX_LM_COMMIT,
            "mlx_lm_kimi_k3_sha256": (download_utils.SUPPORTED_MLX_LM_KIMI_K3_SHA256),
        },
        "tp": {
            "rank": manifest_rank,
            "world_size": 2,
            "contract": download_utils.SUPPORTED_TP_CONTRACT,
            "contract_digest": download_utils.SUPPORTED_TP_CONTRACT_DIGEST,
        },
        "rank_data_bytes": tensor_bytes,
        "files": {
            weight.name: {
                "name": weight.name,
                "bytes": weight.stat().st_size,
                "sha256": weight_sha256,
            }
        },
        "tensors": {
            tensor_name: {
                "source_file": weight.name,
                "bytes": tensor_bytes,
                "rank_shape": [tensor_bytes],
            }
        },
    }
    (root / "tp_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, weight


def _set_rank_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    checkpoint = tmp_path / "rank1"
    monkeypatch.setenv(
        RANK_LOCAL_CHECKPOINT_ENV,
        str(tmp_path / "rank{rank}"),
    )
    return checkpoint


def test_tensor_rank_checkpoint_is_locally_ready_without_model_dir_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    monkeypatch.setattr(download_utils, "EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr(download_utils, "EXO_MODELS_DIRS", (tmp_path / "models",))

    resolved = resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())

    assert resolved == checkpoint
    assert not (tmp_path / "models" / MODEL_ID.normalize()).exists()


def test_explicit_tensor_rank_checkpoint_wins_over_complete_full_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    models = tmp_path / "models"
    full = models / MODEL_ID.normalize()
    full.mkdir(parents=True)
    full_weight = full / "full.safetensors"
    full_weight.write_bytes(b"legacy-full-weight")
    (full / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": full_weight.stat().st_size},
                "weight_map": {"model.weight": full_weight.name},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(download_utils, "EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr(download_utils, "EXO_MODELS_DIRS", (models,))

    assert build_model_path_for_shard(MODEL_ID, _tensor_shard()) == checkpoint


def test_pipeline_marker_behavior_ignores_tensor_rank_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        RANK_LOCAL_CHECKPOINT_ENV,
        str(tmp_path / "missing-rank{rank}"),
    )
    models = tmp_path / "models"
    partial = models / MODEL_ID.normalize()
    partial.mkdir(parents=True)
    weight = partial / "pipeline.safetensors"
    weight.write_bytes(b"pipeline")
    (partial / ".exo_shard.json").write_text(
        json.dumps(
            {
                "model_id": str(MODEL_ID),
                "start_layer": 0,
                "end_layer": 46,
                "n_layers": 93,
                "files": {weight.name: weight.stat().st_size},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(download_utils, "EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr(download_utils, "EXO_MODELS_DIRS", (models,))

    assert resolve_existing_model_for_shard(MODEL_ID, _pipeline_shard()) == partial


def test_loader_returns_none_for_non_tensor_shard_with_opt_in_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        RANK_LOCAL_CHECKPOINT_ENV,
        str(tmp_path / "rank{rank}"),
    )
    monkeypatch.delenv(RANK_LOCAL_LOADER_ENV)

    assert load_configured_rank_local_model(_pipeline_shard(), _UnusedGroup()) is None


def test_rank_world_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint, manifest_rank=0)

    with pytest.raises(RankLocalConfigurationError, match="TP contract"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_missing_or_truncated_listed_file_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _, weight = _rank_checkpoint(checkpoint)
    weight.write_bytes(b"short")

    with pytest.raises(RankLocalConfigurationError, match="missing or truncated"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_unsafe_manifest_filename_fails_before_file_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    manifest_path = checkpoint / "tp_manifest.json"
    manifest = _json_object(manifest_path)
    files = _object(manifest["files"])
    record = next(iter(files.values()))
    manifest["files"] = {"../outside.safetensors": record}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RankLocalConfigurationError, match="unsafe"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_index_and_manifest_file_sets_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    index = _json_object(index_path)
    weight_map = _object(index["weight_map"])
    weight_map["language_model.model.embed_tokens.weight"] = "other.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(RankLocalConfigurationError, match="different files"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def _rank_checkpoint_reset(checkpoint: Path) -> None:
    for child in checkpoint.iterdir():
        child.unlink()
    _rank_checkpoint(checkpoint)


def test_pinned_source_runtime_and_tp_contracts_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    manifest_path = checkpoint / "tp_manifest.json"

    manifest = _json_object(manifest_path)
    _object(manifest["source"])["revision"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RankLocalConfigurationError, match="source contract"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())

    _rank_checkpoint_reset(checkpoint)
    manifest = _json_object(manifest_path)
    _object(manifest["runtime"])["mlx_lm_commit"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RankLocalConfigurationError, match="runtime contract"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())

    _rank_checkpoint_reset(checkpoint)
    manifest = _json_object(manifest_path)
    _object(manifest["tp"])["contract_digest"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RankLocalConfigurationError, match="TP contract"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_rank_data_byte_totals_are_cross_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    index = _json_object(index_path)
    _object(index["metadata"])["total_size"] = 1
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(RankLocalConfigurationError, match="byte totals"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_weight_payload_is_size_checked_not_hashed_during_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    hashed_paths: list[Path] = []

    def record_small_hash(path: Path) -> str:
        hashed_paths.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(download_utils, "_rank_local_sha256", record_small_hash)

    assert resolve_existing_model_for_shard(MODEL_ID, _tensor_shard()) == checkpoint
    assert hashed_paths == [checkpoint / "config.json"]


def test_missing_rank_local_loader_fails_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    monkeypatch.setenv(RANK_LOCAL_LOADER_ENV, str(tmp_path / "missing_loader.py"))

    with pytest.raises(RankLocalConfigurationError, match="loader does not exist"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_rank_local_loader_must_be_a_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    loader_directory = tmp_path / "loader_directory"
    loader_directory.mkdir()
    monkeypatch.setenv(RANK_LOCAL_LOADER_ENV, str(loader_directory))

    with pytest.raises(RankLocalConfigurationError, match="not a regular file"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_rank_local_loader_hash_must_match_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    unreviewed_loader = tmp_path / "unreviewed_loader.py"
    unreviewed_loader.write_text("# changed loader\n", encoding="utf-8")
    monkeypatch.setenv(RANK_LOCAL_LOADER_ENV, str(unreviewed_loader))

    with pytest.raises(RankLocalConfigurationError, match="SHA-256"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_invalid_verify_hashes_setting_fails_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    monkeypatch.setenv(RANK_LOCAL_VERIFY_HASHES_ENV, "true")

    with pytest.raises(RankLocalConfigurationError, match="exactly 0 or 1"):
        resolve_existing_model_for_shard(MODEL_ID, _tensor_shard())


def test_verify_hashes_setting_accepts_one_during_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _set_rank_template(monkeypatch, tmp_path)
    _rank_checkpoint(checkpoint)
    monkeypatch.setenv(RANK_LOCAL_VERIFY_HASHES_ENV, "1")

    assert resolve_existing_model_for_shard(MODEL_ID, _tensor_shard()) == checkpoint


@pytest.mark.asyncio
async def test_invalid_opt_in_becomes_download_failed_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_rank_template(monkeypatch, tmp_path)
    command_sender, command_receiver = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()
    coordinator = DownloadCoordinator(
        node_id=NodeId("test-node"),
        shard_downloader=NoopShardDownloader(),
        download_command_receiver=command_receiver,
        event_sender=event_sender,
    )

    await coordinator._start_download(  # pyright: ignore[reportPrivateUsage]
        _tensor_shard()
    )

    event = await event_receiver.receive()
    assert isinstance(event, NodeDownloadProgress)
    assert isinstance(event.download_progress, DownloadFailed)
    assert "does not exist" in event.download_progress.error_message
    assert isinstance(coordinator.download_status[MODEL_ID], DownloadFailed)
    assert coordinator.active_downloads == {}
    await command_sender.aclose()
    await command_receiver.aclose()
    await event_sender.aclose()
    await event_receiver.aclose()
