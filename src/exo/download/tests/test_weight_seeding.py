"""Tests for peer weight seeding: shard markers, shard-aware resolution,
coverage helpers, allow-pattern selection, and the seed HTTP server."""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import pytest

from exo.download.download_utils import (
    get_local_weight_map,
    marker_covers_shard,
    resolve_allow_patterns,
    resolve_existing_model_for_shard,
)
from exo.download.peer_source import (
    PeerDownloadError,
    _parse_peer_file_list,
    download_file_from_peer,
    fetch_peer_file_list,
)
from exo.download.seed_server import (
    MAX_PEER_FILE_BYTES,
    PeerProtocolError,
    SeedServer,
    _scan_files,
    validate_normalized_model_id,
    validate_relative_file_path,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    download_covers_shard,
)
from exo.shared.types.worker.shards import PipelineShardMetadata

MODEL_ID = ModelId("test-org/test-model")
NORMALIZED = MODEL_ID.normalize()


def _card(n_layers: int = 4) -> ModelCard:
    return ModelCard(
        model_id=MODEL_ID,
        storage_size=Memory.from_bytes(4096),
        n_layers=n_layers,
        hidden_size=8,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _shard(start: int, end: int, n: int = 4) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_card(n),
        device_rank=0,
        world_size=1,
        start_layer=start,
        end_layer=end,
        n_layers=n,
    )


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1024}, "weight_map": weight_map})
    )


def _make_minimal_safetensors() -> bytes:
    """A valid (empty) safetensors file: 8-byte LE header length + JSON header."""
    header = json.dumps({"__metadata__": {}}).encode()
    return struct.pack("<Q", len(header)) + header


class TestLocalWeightMap:
    def test_parses_local_index(self, tmp_path: Path) -> None:
        d = tmp_path / "models" / NORMALIZED
        _write_index(d, {"layers.0.weight": "shard0.safetensors"})
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ())
            mp.setattr(
                "exo.download.download_utils.EXO_MODELS_DIRS", (tmp_path / "models",)
            )
            assert get_local_weight_map(MODEL_ID) == {
                "layers.0.weight": "shard0.safetensors"
            }

    def test_empty_when_no_index(self, tmp_path: Path) -> None:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ())
            mp.setattr("exo.download.download_utils.EXO_MODELS_DIRS", (tmp_path,))
            assert get_local_weight_map(MODEL_ID) == {}


class TestResolveAllowPatterns:
    def test_pipeline_shard_returns_layer_scoped_patterns(self, tmp_path) -> None:
        d = tmp_path / "models" / NORMALIZED
        _write_index(
            d,
            {
                "model.layers.0.weight": "shard0.safetensors",
                "model.layers.1.weight": "shard0.safetensors",
                "model.layers.2.weight": "shard1.safetensors",
                "model.layers.3.weight": "shard1.safetensors",
            },
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ())
            mp.setattr(
                "exo.download.download_utils.EXO_MODELS_DIRS", (tmp_path / "models",)
            )
            patterns = asyncio.run(resolve_allow_patterns(_shard(0, 2)))
        assert "shard0.safetensors" in patterns
        assert "shard1.safetensors" not in patterns


class TestShardMarkerCoverage:
    def test_full_coverage_marker_covers_any_shard(self) -> None:
        marker = {"n_layers": 4, "start_layer": 0, "end_layer": 4, "files": {}}
        assert marker_covers_shard(marker, _shard(0, 2))
        assert marker_covers_shard(marker, _shard(2, 4))

    def test_partial_marker_covers_subset_shard_only(self) -> None:
        marker = {"n_layers": 4, "start_layer": 0, "end_layer": 2, "files": {}}
        assert marker_covers_shard(marker, _shard(0, 1))
        assert marker_covers_shard(marker, _shard(0, 2))
        assert not marker_covers_shard(marker, _shard(1, 3))
        assert not marker_covers_shard(marker, _shard(2, 4))

    def test_wrong_n_layers_never_covers(self) -> None:
        marker = {"n_layers": 8, "start_layer": 0, "end_layer": 8, "files": {}}
        assert not marker_covers_shard(marker, _shard(0, 2))


class TestDownloadCoversShard:
    def test_full_download_covers_all(self) -> None:
        dp = DownloadCompleted(
            node_id="n1",
            shard_metadata=_shard(0, 4),
            total=Memory.from_bytes(4096),
        )
        assert download_covers_shard(dp, _shard(0, 2))
        assert download_covers_shard(dp, _shard(2, 4))

    def test_partial_pipeline_download_covers_subset(self) -> None:
        dp = DownloadCompleted(
            node_id="n1",
            shard_metadata=_shard(0, 2),
            total=Memory.from_bytes(2048),
        )
        assert download_covers_shard(dp, _shard(0, 1))
        assert not download_covers_shard(dp, _shard(2, 4))

    def test_ongoing_does_not_cover(self) -> None:
        from exo.shared.types.worker.downloads import DownloadProgressData

        dp = DownloadOngoing(
            node_id="n1",
            shard_metadata=_shard(0, 4),
            download_progress=DownloadProgressData(
                total=Memory.from_bytes(4096),
                downloaded=Memory.from_bytes(0),
                downloaded_this_session=Memory.from_bytes(0),
                completed_files=0,
                total_files=1,
                speed=0,
                eta_ms=0,
                files={},
            ),
        )
        assert not download_covers_shard(dp, _shard(0, 2))


class TestResolveExistingModelForShard:
    def test_returns_none_when_no_marker(self, tmp_path: Path) -> None:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ())
            mp.setattr("exo.download.download_utils.EXO_MODELS_DIRS", (tmp_path,))
            assert resolve_existing_model_for_shard(MODEL_ID, _shard(0, 2)) is None

    def test_returns_dir_when_marker_covers_and_files_present(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "models" / NORMALIZED
        model_dir.mkdir(parents=True)
        (model_dir / "shard0.safetensors").write_bytes(b"abcd")
        (model_dir / ".exo_shard.json").write_text(
            json.dumps(
                {
                    "model_id": str(MODEL_ID),
                    "start_layer": 0,
                    "end_layer": 2,
                    "n_layers": 4,
                    "files": {"shard0.safetensors": 4},
                }
            )
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ())
            mp.setattr(
                "exo.download.download_utils.EXO_MODELS_DIRS", (tmp_path / "models",)
            )
            assert resolve_existing_model_for_shard(MODEL_ID, _shard(0, 2)) == model_dir

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models" / NORMALIZED
        model_dir.mkdir(parents=True)
        (model_dir / ".exo_shard.json").write_text(
            json.dumps(
                {
                    "start_layer": 0,
                    "end_layer": 2,
                    "n_layers": 4,
                    "files": {"shard0.safetensors": 4},
                }
            )
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ())
            mp.setattr(
                "exo.download.download_utils.EXO_MODELS_DIRS", (tmp_path / "models",)
            )
            assert resolve_existing_model_for_shard(MODEL_ID, _shard(0, 2)) is None


# ---------------------------------------------------------------------------
# Seed server
# ---------------------------------------------------------------------------


@pytest.fixture
def seed_dir(tmp_path: Path, monkeypatch) -> Path:
    """Point exo model dirs at a temp dir and populate a fake model."""
    models_root = tmp_path / "models"
    model_dir = models_root / NORMALIZED
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    weights = _make_minimal_safetensors()
    (model_dir / "weights.safetensors").write_bytes(weights)
    (model_dir / ".exo_shard.json").write_text("{}")  # hidden, must be skipped
    monkeypatch.setattr("exo.download.seed_server.EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr("exo.download.seed_server.EXO_MODELS_DIRS", (models_root,))
    return models_root


class TestSeedServerScan:
    def test_scan_skips_hidden_and_partials(self, seed_dir) -> None:
        files = _scan_files(NORMALIZED)
        assert "config.json" in files
        assert "weights.safetensors" in files
        assert ".exo_shard.json" not in files
        weights = _make_minimal_safetensors()
        assert files["weights.safetensors"][0] == len(weights)

    def test_scan_rejects_model_traversal(self, seed_dir) -> None:
        with pytest.raises(PeerProtocolError, match="model identifier"):
            _scan_files("..")

    def test_scan_rejects_file_and_directory_symlinks(self, seed_dir: Path) -> None:
        model_dir = seed_dir / NORMALIZED
        outside_file = seed_dir.parent / "outside-secret.txt"
        outside_file.write_text("secret")
        (model_dir / "leak.txt").symlink_to(outside_file)
        outside_dir = seed_dir.parent / "outside-directory"
        outside_dir.mkdir()
        (outside_dir / "nested-secret.txt").write_text("secret")
        (model_dir / "linked-directory").symlink_to(
            outside_dir, target_is_directory=True
        )

        files = _scan_files(NORMALIZED)

        assert "leak.txt" not in files
        assert "linked-directory/nested-secret.txt" not in files


class TestPeerProtocolValidation:
    @pytest.mark.parametrize(
        "value",
        ("", ".", "..", "/absolute", "../escape", "a/../b", "a//b", "a\\b"),
    )
    def test_rejects_unsafe_relative_paths(self, value: str) -> None:
        with pytest.raises(PeerProtocolError):
            validate_relative_file_path(value)

    @pytest.mark.parametrize("value", ("", ".", "..", "../model", "org/model"))
    def test_rejects_unsafe_normalized_model_ids(self, value: str) -> None:
        with pytest.raises(PeerProtocolError):
            validate_normalized_model_id(value)

    @pytest.mark.parametrize("size", (-1, True, MAX_PEER_FILE_BYTES + 1))
    def test_peer_file_list_rejects_invalid_sizes(self, size: object) -> None:
        with pytest.raises(PeerDownloadError, match="size"):
            _parse_peer_file_list({"files": [{"path": "weights.bin", "size": size}]})

    def test_peer_file_list_rejects_traversal_and_duplicates(self) -> None:
        with pytest.raises(PeerDownloadError, match="relative"):
            _parse_peer_file_list({"files": [{"path": "../../escape", "size": 1}]})
        with pytest.raises(PeerDownloadError, match="duplicate"):
            _parse_peer_file_list(
                {
                    "files": [
                        {"path": "weights.bin", "size": 1},
                        {"path": "weights.bin", "size": 1},
                    ]
                }
            )


@pytest.mark.asyncio
class TestSeedServerHTTP:
    async def test_defaults_to_loopback_binding(
        self, seed_dir, unused_tcp_port
    ) -> None:
        server = SeedServer(port=unused_tcp_port)
        assert server.host == "127.0.0.1"

    async def test_serves_file_list_and_file(self, seed_dir, unused_tcp_port) -> None:
        server = SeedServer(port=unused_tcp_port)
        await server.start()
        try:
            base = f"http://127.0.0.1:{unused_tcp_port}"
            weights = _make_minimal_safetensors()
            files = await fetch_peer_file_list(base, NORMALIZED)
            assert files is not None
            assert files["weights.safetensors"] == len(weights)

            target = seed_dir.parent / "download_target"
            target.mkdir()
            path = await download_file_from_peer(
                base,
                NORMALIZED,
                "weights.safetensors",
                target,
                expected_size=len(weights),
            )
            assert path.read_bytes() == weights
        finally:
            await server.stop()

    async def test_quotes_model_file_url_components(
        self, seed_dir: Path, unused_tcp_port: int
    ) -> None:
        relative = "nested/name with # and ?.bin"
        source = seed_dir / NORMALIZED / relative
        source.parent.mkdir()
        source.write_bytes(b"quoted path")
        server = SeedServer(port=unused_tcp_port)
        await server.start()
        try:
            base = f"http://127.0.0.1:{unused_tcp_port}"
            target = seed_dir.parent / "quoted-download"
            target.mkdir()
            result = await download_file_from_peer(
                base,
                NORMALIZED,
                relative,
                target,
                expected_size=len(b"quoted path"),
            )
            assert result.read_bytes() == b"quoted path"
        finally:
            await server.stop()

    async def test_rejects_traversal_before_network_or_disk_io(
        self, seed_dir: Path, unused_tcp_port: int
    ) -> None:
        target = seed_dir.parent / "traversal-target"
        target.mkdir()
        with pytest.raises(PeerDownloadError, match="relative"):
            await download_file_from_peer(
                f"http://127.0.0.1:{unused_tcp_port}",
                NORMALIZED,
                "../../escape.bin",
                target,
                expected_size=1,
            )
        assert not (seed_dir.parent / "escape.bin").exists()

    async def test_rejects_parent_symlink_escape(
        self, seed_dir: Path, unused_tcp_port: int
    ) -> None:
        target = seed_dir.parent / "symlink-target"
        outside = seed_dir.parent / "symlink-outside"
        target.mkdir()
        outside.mkdir()
        (target / "nested").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PeerDownloadError, match="escapes"):
            await download_file_from_peer(
                f"http://127.0.0.1:{unused_tcp_port}",
                NORMALIZED,
                "nested/payload.bin",
                target,
                expected_size=1,
            )
        assert not (outside / "payload.bin").exists()

    async def test_rejects_response_larger_than_expected_size(
        self, seed_dir: Path, unused_tcp_port: int
    ) -> None:
        server = SeedServer(port=unused_tcp_port)
        await server.start()
        try:
            target = seed_dir.parent / "bounded-download"
            target.mkdir()
            with pytest.raises(PeerDownloadError, match="exceeds"):
                await download_file_from_peer(
                    f"http://127.0.0.1:{unused_tcp_port}",
                    NORMALIZED,
                    "config.json",
                    target,
                    expected_size=1,
                )
            assert not (target / "config.json").exists()
        finally:
            await server.stop()

    async def test_downloads_zero_length_file(
        self, seed_dir: Path, unused_tcp_port: int
    ) -> None:
        (seed_dir / NORMALIZED / "empty.bin").write_bytes(b"")
        server = SeedServer(port=unused_tcp_port)
        await server.start()
        try:
            target = seed_dir.parent / "empty-download"
            target.mkdir()
            path = await download_file_from_peer(
                f"http://127.0.0.1:{unused_tcp_port}",
                NORMALIZED,
                "empty.bin",
                target,
                expected_size=0,
            )
            assert path.read_bytes() == b""
        finally:
            await server.stop()

    async def test_missing_file_raises(self, seed_dir, unused_tcp_port) -> None:
        server = SeedServer(port=unused_tcp_port)
        await server.start()
        try:
            base = f"http://127.0.0.1:{unused_tcp_port}"
            target = seed_dir.parent / "download_target"
            target.mkdir()
            with pytest.raises(PeerDownloadError):
                await download_file_from_peer(
                    base,
                    NORMALIZED,
                    "does_not_exist.safetensors",
                    target,
                    expected_size=10,
                )
        finally:
            await server.stop()
