from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rank_local_loader as loader  # noqa: E402


class FakeGroup:
    def __init__(self, rank: int = 0, size: int = 2):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class FakeArray:
    def __init__(self, dtype: str, shape: tuple[int, ...] = (2, 3)):
        self.dtype = dtype
        self.shape = shape
        size = 1
        for dimension in shape:
            size *= dimension
        self.size = size

    def astype(self, dtype: str):
        return FakeArray(dtype, self.shape)


class FakeLoadModel:
    def __init__(self, events: list[str], *, fail_load: bool = False):
        self.events = events
        self.fail_load = fail_load

    def load_weights(self, items, *, strict: bool):
        self.events.append("load_weights")
        assert strict is True
        assert items == [("weight", "raw-array")]
        if self.fail_load:
            raise RuntimeError("strict load failed")

    def eval(self):
        self.events.append("model.eval")

    def parameters(self):
        self.events.append("model.parameters")
        return "parameters"


class FakeMX:
    def __init__(self, events: list[str]):
        self.events = events

    def clear_cache(self):
        self.events.append("mx.clear_cache")

    def eval(self, parameters):
        assert parameters == "parameters"
        self.events.append("mx.eval")


@pytest.mark.parametrize(
    ("raw", "expected"),
    ((None, False), ("0", False), ("1", True)),
)
def test_strict_env_flag(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: bool,
):
    name = "EXO_MLX_K3_VOCAB_PARALLEL_HEAD"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert loader._strict_env_flag(name) is expected


@pytest.mark.parametrize("raw", ("", "true", "2", " 1"))
def test_strict_env_flag_rejects_ambiguous_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
):
    name = "EXO_MLX_K3_VOCAB_PARALLEL_HEAD"
    monkeypatch.setenv(name, raw)
    with pytest.raises(loader.RankLocalLoadError, match="must be 0 or 1"):
        loader._strict_env_flag(name)


def test_strict_load_releases_raw_weights_before_model_evaluation():
    events: list[str] = []
    weights = {"weight": "raw-array"}

    loader._strict_load_release_and_evaluate(
        FakeLoadModel(events),
        weights,
        FakeMX(events),
    )

    assert weights == {}
    assert events == [
        "load_weights",
        "mx.clear_cache",
        "model.eval",
        "model.parameters",
        "mx.eval",
    ]


def test_strict_load_failure_retains_raw_weights_and_skips_evaluation():
    events: list[str] = []
    weights = {"weight": "raw-array"}

    with pytest.raises(RuntimeError, match="strict load failed"):
        loader._strict_load_release_and_evaluate(
            FakeLoadModel(events, fail_load=True),
            weights,
            FakeMX(events),
        )

    assert weights == {"weight": "raw-array"}
    assert events == ["load_weights"]


def _dtype_fix_config() -> dict:
    # K3 uses three KDA layers followed by one full-attention layer.
    kda_layers = [layer for layer in range(1, 93) if layer % 4 != 0]
    assert len(kda_layers) == loader.DTYPE_FIX_KDA_LAYER_COUNT
    return {
        "text_config": {
            "linear_attn_config": {
                "kda_layers": kda_layers,
            }
        }
    }


def _checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    root = tmp_path / "rank0"
    root.mkdir(parents=True)
    config = root / "config.json"
    config.write_text('{"model_type":"kimi_k3"}')
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    monkeypatch.setattr(loader, "SOURCE_CONFIG_SHA256", config_sha)
    license_path = root / "LICENSE"
    license_path.write_text("synthetic Kimi K3 license")
    metadata_files = {
        "LICENSE": {
            "bytes": license_path.stat().st_size,
            "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        },
        "config.json": {
            "bytes": config.stat().st_size,
            "sha256": config_sha,
        },
    }
    monkeypatch.setattr(loader, "PINNED_METADATA_FILES", metadata_files)
    monkeypatch.setattr(
        loader,
        "ALLOWED_METADATA_FILENAMES",
        frozenset(metadata_files),
    )

    weight_name = "language_model.model.embed_tokens.weight"
    filename = "model-00001-of-00185.safetensors"
    weight = root / filename
    weight.write_bytes(b"synthetic rank weight")
    weight_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    tensor = {
        "source_file": filename,
        "dtype": "BF16",
        "source_shape": [4, 4],
        "rank_shape": [4, 4],
        "rule": {"kind": "replicated"},
        "resolved_axis": None,
        "intervals": [],
        "bytes": 32,
    }
    manifest = {
        "schema": loader.SCHEMA,
        "complete": True,
        "source": {
            "repo": loader.SOURCE_REPO,
            "revision": loader.SOURCE_REVISION,
            "config_sha256": config_sha,
            "index_sha256": loader.SOURCE_INDEX_SHA256,
        },
        "runtime": {
            "mlx_lm_commit": loader.MLX_LM_COMMIT,
            "mlx_lm_kimi_k3_sha256": (loader.CHECKPOINT_MLX_LM_KIMI_K3_SHA256),
        },
        "tp": {
            "rank": 0,
            "world_size": 2,
            "contract": loader.CONTRACT_VERSION,
            "contract_digest": loader.CONTRACT_DIGEST,
        },
        "rank_data_bytes": 32,
        "metadata_files": metadata_files,
        "metadata_contract_sha256": loader._canonical_metadata_contract(metadata_files),
        "files": {
            filename: {
                "name": filename,
                "bytes": weight.stat().st_size,
                "sha256": weight_sha,
                "tensor_count": 1,
                "tensors": {weight_name: tensor},
            }
        },
        "tensors": {weight_name: tensor},
    }
    index = {
        "metadata": {"total_size": 32},
        "weight_map": {weight_name: filename},
    }
    (root / "tp_manifest.json").write_text(json.dumps(manifest))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return root, manifest


def test_converter_manifest_is_accepted_by_newer_execution_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert loader.MLX_LM_COMMIT != loader.RUNTIME_MLX_LM_COMMIT
    assert loader.CHECKPOINT_MLX_LM_KIMI_K3_SHA256 != loader.MLX_LM_KIMI_K3_SHA256
    root, _manifest = _checkpoint(tmp_path, monkeypatch)
    result = loader._verify_manifest(
        root,
        FakeGroup(),
        verify_file_hashes=True,
        test_only_allow_unpinned=False,
    )
    assert result["rank_data_bytes"] == 32


def test_execution_runtime_pin_matches_width_four_candidate():
    assert loader.RUNTIME_MLX_LM_COMMIT == ("fd38e8483d852d5bb698d3794581a4d568097a12")
    assert loader.MLX_LM_KIMI_K3_SHA256 == (
        "7851e5548b9e3d0ed9231e09a2078b4e09f29462c77e1ea2d727b4b6eba645d8"
    )
    assert loader.MLX_LM_KIMI_K3_DSPARK_SHA256 == (
        "be221a4dde09ec97011a961a4f2d7de1f5f0327af706395967166f710f968021"
    )
    assert loader.MLX_LM_KIMI_K3_FUSED_ROUTER_SHA256 == (
        "14a7cb54eb3585c992504c109db8bed45eb6ec521ae37e680eb601c419ed8ed1"
    )
    assert loader.MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256 == (
        "0b8eec17606b8a4fbd8c6656acbe085dd293436a33004c9282d8e549b8f2c66d"
    )
    assert loader.MLX_LM_KIMI_K3_WIDTH4_FUSED_EXPERT_SHA256 == (
        "5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832"
    )
    assert loader.MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE_SHA256 == (
        "beaadba191fb762d70c2fe2d9d41f53f06984b857db02b8b7fefcd93ee15a925"
    )
    assert loader.MLX_LM_CACHE_SHA256 == (
        "a83a454942864d6430b0d8f28c716593be4d9286c2a683083b6708df97e04e78"
    )
    assert loader.MLX_LM_GENERATE_SHA256 == (
        "096f24553953a90f8e331cb7c7415a75636db4eec8a5bd159b6ad3e93cc4e369"
    )
    assert loader.MLX_LM_COMMIT == "7d505c285b801108a52c23353c7fb6af07204717"
    assert loader.CHECKPOINT_MLX_LM_KIMI_K3_SHA256 == (
        "3dd2e9db585190bca118d5812bcb5b103d1e7c6ec12187b20351992fed7e63cc"
    )


def test_manifest_always_hashes_tokenizer_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, manifest = _checkpoint(tmp_path, monkeypatch)
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text('{"version":"1"}')
    metadata = manifest["metadata_files"]
    metadata["tokenizer.json"] = {
        "bytes": tokenizer.stat().st_size,
        "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(loader, "PINNED_METADATA_FILES", dict(metadata))
    monkeypatch.setattr(
        loader,
        "ALLOWED_METADATA_FILENAMES",
        frozenset(metadata),
    )
    manifest["metadata_contract_sha256"] = loader._canonical_metadata_contract(metadata)
    (root / "tp_manifest.json").write_text(json.dumps(manifest))

    tokenizer.write_text('{"version":"2"}')
    with pytest.raises(loader.RankLocalLoadError, match="metadata checksum mismatch"):
        loader._verify_manifest(
            root,
            FakeGroup(),
            verify_file_hashes=False,
            test_only_allow_unpinned=False,
        )


def test_manifest_rejects_extra_unmanifested_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, _manifest = _checkpoint(tmp_path, monkeypatch)
    (root / "tokenization_kimi.py").write_text("raise RuntimeError('unreviewed')")

    with pytest.raises(loader.RankLocalLoadError, match="file inventory differs"):
        loader._verify_manifest(
            root,
            FakeGroup(),
            verify_file_hashes=False,
            test_only_allow_unpinned=False,
        )


def test_manifest_rejects_missing_metadata_and_contract_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, manifest = _checkpoint(tmp_path, monkeypatch)
    (root / "LICENSE").unlink()
    with pytest.raises(
        loader.RankLocalLoadError, match="missing or truncated metadata"
    ):
        loader._verify_manifest(
            root,
            FakeGroup(),
            verify_file_hashes=False,
            test_only_allow_unpinned=False,
        )

    root, manifest = _checkpoint(tmp_path / "second", monkeypatch)
    manifest["metadata_contract_sha256"] = "0" * 64
    (root / "tp_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(loader.RankLocalLoadError, match="contract digest mismatch"):
        loader._verify_manifest(
            root,
            FakeGroup(),
            verify_file_hashes=False,
            test_only_allow_unpinned=False,
        )


class _Gathered:
    def __init__(self, values: list[int]):
        self._values = values

    def tolist(self) -> list[int]:
        return self._values


class _FakeDistributed:
    def __init__(self, peer_row: list[int]):
        self.peer_row = peer_row

    def all_gather(self, local: list[int], *, group: FakeGroup) -> _Gathered:
        assert group.size() == 2
        return _Gathered([*local, *self.peer_row])


class _FakeMx:
    int32 = "int32"

    def __init__(self, peer_row: list[int]):
        self.distributed = _FakeDistributed(peer_row)

    @staticmethod
    def array(values: tuple[int, ...], *, dtype: str) -> list[int]:
        assert dtype == "int32"
        return list(values)

    @staticmethod
    def eval(_value: object) -> None:
        return None


def test_metadata_digest_must_agree_across_ranks():
    local = "11" * 32
    peer = list(bytes.fromhex("22" * 32))
    with pytest.raises(loader.RankLocalLoadError, match="differs across"):
        loader._agree_metadata_contract(_FakeMx(peer), FakeGroup(), local)


def test_dspark_runtime_source_must_agree_across_ranks():
    local = "33" * 32
    peer = list(bytes.fromhex("44" * 32))
    with pytest.raises(loader.RankLocalLoadError, match="DSpark.*differs across"):
        loader._agree_dspark_runtime_source(_FakeMx(peer), FakeGroup(), local)


def test_runtime_source_verification_uses_execution_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__path__ = []  # type: ignore[attr-defined]
    models = types.ModuleType("mlx_lm.models")
    models.__path__ = []  # type: ignore[attr-defined]
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)

    pins = {
        "cache": "MLX_LM_CACHE_SHA256",
        "gated_delta": "MLX_LM_GATED_DELTA_SHA256",
        "kimi_k3": "MLX_LM_KIMI_K3_SHA256",
        "kimi_k3_derived_bias": "MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256",
        "kimi_k3_dspark": "MLX_LM_KIMI_K3_DSPARK_SHA256",
        "kimi_k3_fused_down_reduce": "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256",
        "kimi_k3_fused_expert": "MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256",
        "kimi_k3_width4_fused_expert": ("MLX_LM_KIMI_K3_WIDTH4_FUSED_EXPERT_SHA256"),
        "kimi_k3_fused_router": "MLX_LM_KIMI_K3_FUSED_ROUTER_SHA256",
        "kimi_k3_fused_switch_glu": "MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256",
        "kimi_k3_packed_moe_front": "MLX_LM_KIMI_K3_PACKED_MOE_FRONT_SHA256",
        "kimi_k3_prefill_route_combine": (
            "MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE_SHA256"
        ),
        "kimi_k3_tuned_gather_qmv": "MLX_LM_KIMI_K3_TUNED_GATHER_QMV_SHA256",
        "switch_layers": "MLX_LM_SWITCH_LAYERS_SHA256",
    }
    expected: dict[str, str] = {}
    for module_name, constant_name in pins.items():
        source = tmp_path / f"{module_name}.py"
        source.write_bytes(f"pinned {module_name} execution runtime".encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        module = types.ModuleType(f"mlx_lm.models.{module_name}")
        module.__file__ = str(source)
        setattr(models, module_name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(loader, constant_name, digest)
        expected[constant_name] = digest

    generate_source = tmp_path / "generate.py"
    generate_source.write_bytes(b"pinned mlx_lm generate execution runtime")
    generate_digest = hashlib.sha256(generate_source.read_bytes()).hexdigest()
    generate = types.ModuleType("mlx_lm.generate")
    generate.__file__ = str(generate_source)
    mlx_lm.generate = generate
    monkeypatch.setitem(sys.modules, generate.__name__, generate)
    monkeypatch.setattr(loader, "MLX_LM_GENERATE_SHA256", generate_digest)

    assert loader._verify_runtime_source() == expected["MLX_LM_KIMI_K3_DSPARK_SHA256"]
    for module_name, constant_name in pins.items():
        monkeypatch.setattr(loader, constant_name, "0" * 64)
        with pytest.raises(loader.RankLocalLoadError, match=f"{module_name}.py"):
            loader._verify_runtime_source()
        monkeypatch.setattr(loader, constant_name, expected[constant_name])

    projected_kv_runtime_pins = {
        "kimi_k3": (
            "MLX_LM_KIMI_K3_SHA256",
            "c4e4604bdfe520c69fa2a27458ab861099ba4838342ccf72a6e44413bbec9fd5",
        ),
        "kimi_k3_fused_router": (
            "MLX_LM_KIMI_K3_FUSED_ROUTER_SHA256",
            "17bfde08b4d72deb74f9f0e0337822495e1b305166ee966eaab65896f1ed1fc4",
        ),
        "kimi_k3_prefill_route_combine": (
            "MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE_SHA256",
            "0f2d2440d89d327adeba369620aa206d1f641ff06954db6d364ae1785806e80a",
        ),
    }
    for module_name, (constant_name, old_digest) in projected_kv_runtime_pins.items():
        monkeypatch.setattr(loader, constant_name, old_digest)
        with pytest.raises(loader.RankLocalLoadError, match=f"{module_name}.py"):
            loader._verify_runtime_source()
        monkeypatch.setattr(loader, constant_name, expected[constant_name])

    monkeypatch.setattr(loader, "MLX_LM_GENERATE_SHA256", "0" * 64)
    with pytest.raises(loader.RankLocalLoadError, match="generate.py"):
        loader._verify_runtime_source()

    dspark_digest = expected["MLX_LM_KIMI_K3_DSPARK_SHA256"]
    monkeypatch.setattr(loader, "MLX_LM_KIMI_K3_DSPARK_SHA256", "0" * 64)
    monkeypatch.setattr(loader, "MLX_LM_KIMI_K3_DSPARK_0731_SHA256", dspark_digest)
    monkeypatch.setattr(loader, "MLX_LM_GENERATE_SHA256", generate_digest)
    assert loader._verify_runtime_source() == dspark_digest

    monkeypatch.setattr(loader, "MLX_LM_KIMI_K3_DSPARK_0731_SHA256", "1" * 64)
    with pytest.raises(loader.RankLocalLoadError, match="either audited"):
        loader._verify_runtime_source()


@pytest.mark.parametrize("failure_rank", (0, 1))
def test_execution_runtime_preinner_failure_never_enters_checkpoint_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_rank: int,
):
    mlx_package = types.ModuleType("mlx")
    mlx_package.__path__ = []  # type: ignore[attr-defined]
    mlx_core = types.ModuleType("mlx.core")
    mlx_lm_package = types.ModuleType("mlx_lm")
    mlx_lm_package.__path__ = []  # type: ignore[attr-defined]
    mlx_lm_utils = types.ModuleType("mlx_lm.utils")
    monkeypatch.setitem(sys.modules, "mlx", mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_package)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", mlx_lm_utils)

    checkpoint_validations: list[None] = []

    def prepare(_model_dir: object, _utils: object) -> tuple:
        if failure_rank == 0:
            raise OSError("rank-local path preparation failed")
        return (tmp_path, object(), object(), object(), False, "22" * 32)

    def agree(
        _mx: object,
        _group: object,
        label: str,
        local_error: BaseException | None,
    ) -> None:
        assert label == "execution runtime validation"
        if failure_rank == 0:
            assert isinstance(local_error, OSError)
            raise loader.RankLocalLoadError("local pre-inner failure")
        assert local_error is None
        raise loader.RankLocalLoadError("peer pre-inner failure")

    monkeypatch.setattr(loader, "_prepare_execution_runtime", prepare)
    monkeypatch.setattr(loader, "_agree_local_validation", agree)
    monkeypatch.setattr(
        loader,
        "_verify_manifest",
        lambda *_args, **_kwargs: checkpoint_validations.append(None),
    )

    with pytest.raises(loader.RankLocalLoadError, match="pre-inner failure"):
        loader.load_rank_local_model(tmp_path, tensor_group=FakeGroup())

    assert checkpoint_validations == []


def test_manifest_rejects_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, manifest = _checkpoint(tmp_path, monkeypatch)
    _filename, record = next(iter(manifest["files"].items()))
    manifest["files"] = {"../outside.safetensors": record}
    (root / "tp_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(loader.RankLocalLoadError, match="unsafe checkpoint"):
        loader._verify_manifest(
            root,
            FakeGroup(),
            verify_file_hashes=False,
            test_only_allow_unpinned=False,
        )


def test_manifest_rejects_stale_index_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, _manifest = _checkpoint(tmp_path, monkeypatch)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["language_model.model.embed_tokens.weight"] = (
        "model-00002-of-00185.safetensors"
    )
    index_path.write_text(json.dumps(index))
    with pytest.raises(
        loader.RankLocalLoadError,
        match="tensor inventory mismatch|file sets differ",
    ):
        loader._verify_manifest(
            root,
            FakeGroup(),
            verify_file_hashes=False,
            test_only_allow_unpinned=False,
        )


def test_manifest_rejects_wrong_rank_byte_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, manifest = _checkpoint(tmp_path, monkeypatch)
    manifest["rank_data_bytes"] = 31
    (root / "tp_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(loader.RankLocalLoadError, match="byte totals"):
        loader._verify_manifest(
            root,
            FakeGroup(),
            verify_file_hashes=False,
            test_only_allow_unpinned=False,
        )


def test_upstream_dtype_fix_casts_only_the_138_pinned_tensors():
    config = _dtype_fix_config()
    expected = loader._dtype_fix_tensor_names(config)
    weights = {
        name: FakeArray(
            "f32",
            (128,) if name.endswith(".o_norm.weight") else (18432, 4, 1),
        )
        for name in expected
    }
    weights["intentional.A_log"] = FakeArray("f32")

    audit = loader._apply_upstream_dtype_fix(
        weights,
        config,
        float32_dtype="f32",
        bfloat16_dtype="bf16",
    )

    assert len(expected) == 138
    assert all(weights[name].dtype == "bf16" for name in expected)
    assert weights["intentional.A_log"].dtype == "f32"
    assert audit["tensor_count"] == 138
    assert audit["category_counts"] == {
        "o_norm.weight": 69,
        "qkv_conv.conv.weight": 69,
    }
    assert audit["effective_source_revision"] == loader.EFFECTIVE_SOURCE_REVISION
    assert audit["rank_elements_cast"] == loader.DTYPE_FIX_TP2_RANK_ELEMENTS
    assert len(audit["tensor_names_sha256"]) == 64


def test_upstream_dtype_fix_fails_closed_on_missing_or_non_fp32_tensor():
    config = _dtype_fix_config()
    expected = loader._dtype_fix_tensor_names(config)
    weights = {
        name: FakeArray(
            "f32",
            (128,) if name.endswith(".o_norm.weight") else (18432, 4, 1),
        )
        for name in expected
    }
    weights.pop(expected[0])
    with pytest.raises(loader.RankLocalLoadError, match="missing"):
        loader._apply_upstream_dtype_fix(
            weights,
            config,
            float32_dtype="f32",
            bfloat16_dtype="bf16",
        )

    weights = {
        name: FakeArray(
            "f32",
            (128,) if name.endswith(".o_norm.weight") else (18432, 4, 1),
        )
        for name in expected
    }
    weights[expected[0]] = FakeArray("bf16")
    with pytest.raises(loader.RankLocalLoadError, match="float32"):
        loader._apply_upstream_dtype_fix(
            weights,
            config,
            float32_dtype="f32",
            bfloat16_dtype="bf16",
        )
