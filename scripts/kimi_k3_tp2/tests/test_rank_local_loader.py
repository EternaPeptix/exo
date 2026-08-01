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
    root.mkdir()
    config = root / "config.json"
    config.write_text('{"model_type":"kimi_k3"}')
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    monkeypatch.setattr(loader, "SOURCE_CONFIG_SHA256", config_sha)

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


def test_runtime_source_verification_uses_execution_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mlx_lm = types.ModuleType("mlx_lm")
    models = types.ModuleType("mlx_lm.models")
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)

    pins = (
        ("kimi_k3", "MLX_LM_KIMI_K3_SHA256"),
        ("kimi_k3_derived_bias", "MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256"),
        ("kimi_k3_fused_expert", "MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256"),
        (
            "kimi_k3_fused_switch_glu",
            "MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256",
        ),
        (
            "kimi_k3_fused_down_reduce",
            "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256",
        ),
        (
            "kimi_k3_tuned_gather_qmv",
            "MLX_LM_KIMI_K3_TUNED_GATHER_QMV_SHA256",
        ),
    )
    for module_name, pin_name in pins:
        runtime_source = tmp_path / f"{module_name}.py"
        runtime_source.write_bytes(f"pinned {module_name}".encode())
        digest = hashlib.sha256(runtime_source.read_bytes()).hexdigest()
        monkeypatch.setattr(loader, pin_name, digest)
        module = types.ModuleType(f"mlx_lm.models.{module_name}")
        module.__file__ = str(runtime_source)
        setattr(models, module_name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    loader._verify_runtime_source()
    monkeypatch.setattr(loader, "MLX_LM_KIMI_K3_DERIVED_BIAS_SHA256", "0" * 64)
    with pytest.raises(loader.RankLocalLoadError, match="execution runtime pin"):
        loader._verify_runtime_source()


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
