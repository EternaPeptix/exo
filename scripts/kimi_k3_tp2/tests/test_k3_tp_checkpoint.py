from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import k3_tp_checkpoint as checkpoint  # noqa: E402
from k3_tp_checkpoint import (  # noqa: E402
    ConversionError,
    KimiK3ShardingContract,
    SafeTensorFile,
    audit_shard,
    convert_shard,
    main,
    write_rank_shard,
    write_synthetic_safetensors,
)


def synthetic_config() -> dict:
    return {
        "_synthetic_test": True,
        "model_type": "kimi_k3",
        "text_config": {
            "num_hidden_layers": 4,
            "num_experts": 4,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "q_lora_rank": 4,
            "mla_use_output_gate": True,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "use_full_rank_gate": True,
            },
        },
    }


def write_pinned_test_metadata(
    metadata_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    weight_map: dict[str, str],
) -> None:
    metadata_dir.mkdir()
    config_path = metadata_dir / "config.json"
    index_path = metadata_dir / "model.safetensors.index.json"
    config_path.write_text(json.dumps(synthetic_config(), sort_keys=True))
    index_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": 816_773_159_296,
                    "total_parameters": 2_779_483_539_072,
                },
                "weight_map": weight_map,
            },
            sort_keys=True,
        )
    )
    monkeypatch.setattr(
        checkpoint, "SOURCE_CONFIG_SHA256", checkpoint._sha256_file(config_path)
    )
    monkeypatch.setattr(
        checkpoint, "SOURCE_INDEX_SHA256", checkpoint._sha256_file(index_path)
    )


def seq(shape, dtype=np.uint32):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


def expected_segmented(array: np.ndarray, axis: int, rank: int) -> np.ndarray:
    segments = np.split(array, 3, axis=axis)
    return np.concatenate(
        [np.split(segment, 2, axis=axis)[rank] for segment in segments],
        axis=axis,
    )


def expected_half(array: np.ndarray, axis: int, rank: int) -> np.ndarray:
    return np.split(array, 2, axis=axis)[rank]


def valid_v1_resume_journal() -> tuple[dict, dict[str, set[str]], object]:
    filename = "model-00001-of-00185.safetensors"
    tensor_name = "language_model.model.layers.0.self_attn.A_log"
    contract = KimiK3ShardingContract(synthetic_config(), world_size=2)
    desc = checkpoint.TensorDesc(tensor_name, "F32", (6,), 0, 24)
    ranks = {}
    for rank in range(2):
        plan = contract.plan(desc, rank)
        assert plan is not None
        ranks[str(rank)] = {
            "name": filename,
            "bytes": 128,
            "sha256": f"{rank + 1}" * 64,
            "tensor_count": 1,
            "tensors": {
                tensor_name: checkpoint._tensor_manifest(plan, filename),
            },
        }
    journal = {
        "schema": checkpoint.LEGACY_SCHEMA,
        "source_repo": checkpoint.SOURCE_REPO,
        "source_revision": checkpoint.SOURCE_REVISION,
        "config_sha256": "a" * 64,
        "index_sha256": "b" * 64,
        "contract_digest": contract.contract_digest(),
        "world_size": 2,
        "files": {
            filename: {
                "source": {
                    "bytes": 256,
                    "sha256": "c" * 64,
                    "url": None,
                    "resumed": False,
                },
                "ranks": ranks,
                "excluded": False,
                "committed": True,
            }
        },
        "complete": False,
    }
    return journal, {filename: {tensor_name}}, contract


def validate_resume_journal(journal: dict) -> dict:
    _original, keys_by_file, contract = valid_v1_resume_journal()
    return checkpoint._validate_resume_journal(
        journal,
        config_sha256="a" * 64,
        index_sha256="b" * 64,
        world_size=2,
        contract=contract,
        keys_by_file=keys_by_file,
    )


def test_v1_resume_journal_is_strictly_validated_and_migrated_to_v2():
    journal, _keys_by_file, _contract = valid_v1_resume_journal()

    validated = validate_resume_journal(journal)

    assert validated is journal
    assert validated["schema"] == checkpoint.SCHEMA


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("world_size", 3, "world_size"),
        ("contract_digest", "d" * 64, "contract_digest"),
        ("schema", "k3-rank-local-tp/v99", "unsupported schema"),
    ],
)
def test_resume_journal_rejects_incompatible_root_contract(
    field: str,
    value: object,
    message: str,
):
    journal, _keys_by_file, _contract = valid_v1_resume_journal()
    journal[field] = value

    with pytest.raises(ConversionError, match=message):
        validate_resume_journal(journal)


def test_resume_journal_rejects_path_like_rank_record_name():
    journal, _keys_by_file, _contract = valid_v1_resume_journal()
    entry = next(iter(journal["files"].values()))
    entry["ranks"]["0"]["name"] = "nested/model-00001-of-00185.safetensors"

    with pytest.raises(ConversionError, match="safe checkpoint basename"):
        validate_resume_journal(journal)


@pytest.mark.parametrize("files", [[], "not-an-object", None])
def test_resume_journal_rejects_malformed_files_mapping(files: object):
    journal, _keys_by_file, _contract = valid_v1_resume_journal()
    journal["files"] = files

    with pytest.raises(ConversionError, match="files must be an object"):
        validate_resume_journal(journal)


def write_source_free_resume_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[Path], Path, Path, str, dict]:
    tensor_name = "language_model.model.layers.0.self_attn.A_log"
    filename = "model-00001-of-00185.safetensors"
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(
        metadata_dir,
        monkeypatch,
        {tensor_name: filename},
    )
    (metadata_dir / "LICENSE").write_text("Kimi K3 license text\n")
    source_path = tmp_path / "template" / filename
    write_synthetic_safetensors(
        source_path,
        {tensor_name: ("F32", seq((6,), np.float32))},
    )
    contract = KimiK3ShardingContract(synthetic_config(), world_size=2)
    rank_dirs = [tmp_path / "rank0", tmp_path / "rank1"]
    ranks: dict[str, dict] = {}
    with SafeTensorFile(source_path) as source:
        for rank, rank_dir in enumerate(rank_dirs):
            desc = source.tensors[tensor_name]
            plan = contract.plan(desc, rank)
            assert plan is not None
            record = write_rank_shard(
                source,
                rank_dir / filename,
                [plan],
                rank=rank,
                world_size=2,
                max_buffer_bytes=64,
            )
            record["tensors"] = {
                tensor_name: checkpoint._tensor_manifest(plan, filename)
            }
            ranks[str(rank)] = record

    journal = {
        "schema": checkpoint.LEGACY_SCHEMA,
        "source_repo": checkpoint.SOURCE_REPO,
        "source_revision": checkpoint.SOURCE_REVISION,
        "config_sha256": checkpoint._sha256_file(metadata_dir / "config.json"),
        "index_sha256": checkpoint._sha256_file(
            metadata_dir / "model.safetensors.index.json"
        ),
        "contract_digest": contract.contract_digest(),
        "world_size": 2,
        "files": {
            filename: {
                "source": {
                    "bytes": source_path.stat().st_size,
                    "sha256": checkpoint._sha256_file(source_path),
                    "url": None,
                    "resumed": False,
                },
                "ranks": ranks,
                "excluded": False,
                "committed": True,
            }
        },
        "complete": False,
    }
    journal_path = metadata_dir / "tp-conversion-journal.json"
    journal_path.write_text(json.dumps(journal, sort_keys=True))
    return (
        metadata_dir,
        rank_dirs,
        tmp_path / "unavailable-source",
        tmp_path / "cache",
        filename,
        journal,
    )


def test_v1_source_free_resume_revalidates_outputs_and_writes_v2_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metadata_dir, rank_dirs, source_dir, cache_dir, _filename, _journal = (
        write_source_free_resume_fixture(tmp_path, monkeypatch)
    )

    completed = checkpoint.convert_checkpoint(
        metadata_dir=metadata_dir,
        rank_dirs=rank_dirs,
        source_dir=source_dir,
        cache_dir=cache_dir,
    )

    assert completed["schema"] == checkpoint.SCHEMA
    assert completed["complete"] is True
    for rank_dir in rank_dirs:
        manifest = json.loads((rank_dir / "tp_manifest.json").read_text())
        assert manifest["schema"] == checkpoint.SCHEMA


def test_source_free_resume_rejects_symlinked_rank_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metadata_dir, rank_dirs, source_dir, cache_dir, filename, _journal = (
        write_source_free_resume_fixture(tmp_path, monkeypatch)
    )
    rank_path = rank_dirs[0] / filename
    symlink_target = tmp_path / "symlink-target.safetensors"
    rank_path.replace(symlink_target)
    rank_path.symlink_to(symlink_target)

    with pytest.raises(ConversionError, match="missing source shard"):
        checkpoint.convert_checkpoint(
            metadata_dir=metadata_dir,
            rank_dirs=rank_dirs,
            source_dir=source_dir,
            cache_dir=cache_dir,
        )


def test_source_free_resume_rejects_file_with_mismatched_safetensors_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metadata_dir, rank_dirs, source_dir, cache_dir, filename, journal = (
        write_source_free_resume_fixture(tmp_path, monkeypatch)
    )
    rank_path = rank_dirs[0] / filename
    tensor_name = "language_model.model.layers.0.self_attn.A_log"
    write_synthetic_safetensors(
        rank_path,
        {tensor_name: ("F32", seq((2,), np.float32))},
        metadata={
            "format": "mlx",
            "schema": checkpoint.SCHEMA,
            "source_repo": checkpoint.SOURCE_REPO,
            "source_revision": checkpoint.SOURCE_REVISION,
            "source_file": filename,
            "tp_rank": "0",
            "tp_world_size": "2",
            "sharding_contract": checkpoint.CONTRACT_VERSION,
        },
    )
    rank_record = journal["files"][filename]["ranks"]["0"]
    rank_record["bytes"] = rank_path.stat().st_size
    rank_record["sha256"] = checkpoint._sha256_file(rank_path)
    (metadata_dir / "tp-conversion-journal.json").write_text(
        json.dumps(journal, sort_keys=True)
    )

    with pytest.raises(ConversionError, match="missing source shard"):
        checkpoint.convert_checkpoint(
            metadata_dir=metadata_dir,
            rank_dirs=rank_dirs,
            source_dir=source_dir,
            cache_dir=cache_dir,
        )


def test_rule_classification_exact_model_shard_contract():
    contract = KimiK3ShardingContract(synthetic_config(), world_size=2)
    cases = {
        "language_model.model.embed_tokens.weight": "replicated",
        "vision_tower.encoder.weight": "excluded",
        "language_model.model.layers.0.self_attn.qkv_proj.weight": ("all-to-sharded"),
        "language_model.model.layers.0.self_attn.qkv_conv.conv.weight": "axis",
        "language_model.model.layers.0.self_attn.o_proj.scales": ("sharded-to-all"),
        "language_model.model.layers.0.mlp.gate_proj.weight": "all-to-sharded",
        "language_model.model.layers.0.mlp.down_proj.biases": "sharded-to-all",
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": (
            "all-to-sharded"
        ),
        "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight": (
            "sharded-to-all"
        ),
        # Router and latent projections are intentionally replicated by PR #1626.
        "language_model.model.layers.1.mlp.gate.weight": "replicated",
        "language_model.model.layers.1.mlp.routed_expert_down_proj.weight": (
            "replicated"
        ),
        "language_model.model.layers.3.self_attn.q_b_proj.weight": ("all-to-sharded"),
        "language_model.model.layers.3.self_attn.embed_q.weight": "axis",
        "language_model.model.layers.3.self_attn.unembed_out.scales": "axis",
        "language_model.model.layers.3.self_attn.o_proj.weight": ("sharded-to-all"),
    }
    for name, expected in cases.items():
        assert contract.classify(name, 3).kind == expected, name


def test_quantized_suffix_semantics_match_mlx_helpers():
    contract = KimiK3ShardingContract(synthetic_config(), world_size=2)
    # MLX checks path.endswith("bias"), so affine quantization's plural
    # ``biases`` follows the weight/scales sharding axis.
    true_bias = contract.plan(
        _desc(
            "language_model.model.layers.0.self_attn.o_proj.bias",
            "F16",
            (8,),
        ),
        rank=0,
    )
    quant_biases = contract.plan(
        _desc(
            "language_model.model.layers.0.self_attn.o_proj.biases",
            "F16",
            (8, 6),
        ),
        rank=0,
    )
    assert true_bias.axis is None
    assert true_bias.output_shape == (8,)
    assert quant_biases.axis == 1
    assert quant_biases.output_shape == (8, 3)


def _desc(name: str, dtype: str, shape: tuple[int, ...]):
    from k3_tp_checkpoint import TensorDesc

    itemsize = {
        "U32": 4,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
    }[dtype]
    return TensorDesc(name, dtype, shape, 0, int(np.prod(shape)) * itemsize)


def test_streaming_writer_matches_numpy_reference_for_both_ranks(tmp_path: Path):
    names = {
        "qkv_w": "language_model.model.layers.0.self_attn.qkv_proj.weight",
        "qkv_s": "language_model.model.layers.0.self_attn.qkv_proj.scales",
        "conv": "language_model.model.layers.0.self_attn.qkv_conv.conv.weight",
        "alog": "language_model.model.layers.0.self_attn.A_log",
        "oproj": "language_model.model.layers.0.self_attn.o_proj.weight",
        "fa": "language_model.model.layers.0.self_attn.f_a_proj.weight",
        "switch_gate": (
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"
        ),
        "switch_down": (
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight"
        ),
        "router": "language_model.model.layers.1.mlp.gate.weight",
        "embed_q": "language_model.model.layers.3.self_attn.embed_q.weight",
        "unembed_s": ("language_model.model.layers.3.self_attn.unembed_out.scales"),
        "bf16_norm": "language_model.model.layers.3.input_layernorm.weight",
        "vision": "vision_tower.encoder.weight",
    }
    tensors = OrderedDict(
        [
            (names["qkv_w"], ("U32", seq((18, 4)))),
            (names["qkv_s"], ("F16", seq((18, 2), np.float16))),
            (names["conv"], ("F16", seq((18, 4), np.float16))),
            (names["alog"], ("F32", seq((6,), np.float32))),
            (names["oproj"], ("U32", seq((8, 6)))),
            (names["fa"], ("U32", seq((4, 4)))),
            (names["switch_gate"], ("U32", seq((4, 6, 4)))),
            (names["switch_down"], ("U32", seq((4, 8, 6)))),
            (names["router"], ("U32", seq((4, 4)))),
            (names["embed_q"], ("U32", seq((6, 4, 2)))),
            (names["unembed_s"], ("F16", seq((6, 4, 2), np.float16))),
            # Explicit BF16 bits exercise bit-preserving uint16 storage.
            (names["bf16_norm"], ("BF16", seq((8,), np.uint16))),
            (names["vision"], ("F16", seq((2, 2), np.float16))),
        ]
    )
    source_path = tmp_path / "source.safetensors"
    write_synthetic_safetensors(source_path, tensors)
    contract = KimiK3ShardingContract(synthetic_config(), world_size=2)

    references = {name: array for name, (_dtype, array) in tensors.items()}
    for rank in (0, 1):
        out_path = tmp_path / f"rank{rank}.safetensors"
        with SafeTensorFile(source_path) as source:
            plans = [
                plan
                for desc in source.tensors.values()
                if (plan := contract.plan(desc, rank)) is not None
            ]
            record = write_rank_shard(
                source,
                out_path,
                plans,
                rank=rank,
                world_size=2,
                # Force many chunks; validates bounded strided copying.
                max_buffer_bytes=64,
            )
        assert record["tensor_count"] == len(tensors) - 1
        with SafeTensorFile(out_path) as out:
            assert names["vision"] not in out.tensors
            actual = {
                name: np.array(out.array(desc), copy=True)
                for name, desc in out.tensors.items()
            }

        assert np.array_equal(
            actual[names["qkv_w"]],
            expected_segmented(references[names["qkv_w"]], 0, rank),
        )
        assert np.array_equal(
            actual[names["qkv_s"]],
            expected_segmented(references[names["qkv_s"]], 0, rank),
        )
        assert np.array_equal(
            actual[names["conv"]],
            expected_segmented(references[names["conv"]], 0, rank),
        )
        assert np.array_equal(
            actual[names["alog"]],
            expected_half(references[names["alog"]], 0, rank),
        )
        assert np.array_equal(
            actual[names["oproj"]],
            expected_half(references[names["oproj"]], 1, rank),
        )
        assert np.array_equal(actual[names["fa"]], references[names["fa"]])
        assert np.array_equal(
            actual[names["switch_gate"]],
            expected_half(references[names["switch_gate"]], 1, rank),
        )
        assert np.array_equal(
            actual[names["switch_down"]],
            expected_half(references[names["switch_down"]], 2, rank),
        )
        assert np.array_equal(actual[names["router"]], references[names["router"]])
        assert np.array_equal(
            actual[names["embed_q"]],
            expected_half(references[names["embed_q"]], 0, rank),
        )
        assert np.array_equal(
            actual[names["unembed_s"]],
            expected_half(references[names["unembed_s"]], 0, rank),
        )
        assert np.array_equal(
            actual[names["bf16_norm"]], references[names["bf16_norm"]]
        )


def test_non_divisible_segment_fails_closed():
    contract = KimiK3ShardingContract(synthetic_config(), world_size=2)
    desc = _desc(
        "language_model.model.layers.0.self_attn.qkv_proj.weight",
        "U32",
        (15, 4),
    )
    with pytest.raises(ConversionError, match="segment size"):
        contract.plan(desc, rank=0)


def test_safetensors_rejects_header_shape_byte_mismatch(tmp_path: Path):
    path = tmp_path / "bad.safetensors"
    header = {
        "x": {
            "dtype": "U32",
            "shape": [4],
            # Four U32 elements require 16 bytes.
            "data_offsets": [0, 8],
        }
    }
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 8)
    with (
        pytest.raises(ConversionError, match="shape implies"),
        SafeTensorFile(path),
    ):
        pass


def test_audit_shard_is_read_only_and_resolves_both_tp2_plans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    qkv = "language_model.model.layers.0.self_attn.qkv_proj.weight"
    norm = "language_model.model.layers.0.input_layernorm.weight"
    vision = "vision_tower.encoder.weight"
    filename = "model-00001-of-00185.safetensors"
    shard = tmp_path / filename
    write_synthetic_safetensors(
        shard,
        OrderedDict(
            [
                (qkv, ("U32", seq((18, 4)))),
                (norm, ("BF16", seq((8,), np.uint16))),
                (vision, ("F16", seq((2, 2), np.float16))),
            ]
        ),
    )
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(
        metadata_dir,
        monkeypatch,
        {qkv: filename, norm: filename, vision: filename},
    )
    before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    report = audit_shard(metadata_dir, shard)

    after = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert report["read_only"] is True
    assert report["tp_world_size"] == 2
    assert report["shard"]["tensor_count"] == 3
    assert report["excluded_tensor_count"] == 1
    assert [rank["data_bytes"] for rank in report["ranks"]] == [160, 160]
    assert all(rank["predicted_file_bytes"] > 160 for rank in report["ranks"])
    for rank in report["ranks"]:
        tensors = {tensor["name"]: tensor for tensor in rank["tensors"]}
        assert tensors[qkv]["source_shape"] == [18, 4]
        assert tensors[qkv]["rank_shape"] == [9, 4]
        assert tensors[qkv]["rule"]["kind"] == "all-to-sharded"
        assert tensors[qkv]["resolved_axis"] == 0
        assert tensors[norm]["rank_shape"] == [8]
        assert tensors[norm]["rule"]["kind"] == "replicated"


def test_audit_shard_cli_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    name = "language_model.model.layers.0.self_attn.A_log"
    filename = "model-00002-of-00185.safetensors"
    shard = tmp_path / filename
    write_synthetic_safetensors(shard, {name: ("F32", seq((6,), np.float32))})
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(metadata_dir, monkeypatch, {name: filename})

    status = main(
        [
            "audit-shard",
            "--metadata-dir",
            str(metadata_dir),
            "--shard",
            str(shard),
        ]
    )

    assert status == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "k3-rank-local-tp-shard-audit/v1"
    assert [rank["rank"] for rank in report["ranks"]] == [0, 1]
    assert [rank["data_bytes"] for rank in report["ranks"]] == [12, 12]


def test_audit_shard_rejects_index_header_key_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    actual = "language_model.model.layers.0.self_attn.A_log"
    missing = "language_model.model.layers.0.self_attn.dt_bias"
    filename = "model-00003-of-00185.safetensors"
    shard = tmp_path / filename
    write_synthetic_safetensors(shard, {actual: ("F32", seq((6,), np.float32))})
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(
        metadata_dir,
        monkeypatch,
        {actual: filename, missing: filename},
    )

    with pytest.raises(ConversionError, match="index/header mismatch"):
        audit_shard(metadata_dir, shard)


def test_audit_shard_rejects_unsupported_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    name = "language_model.model.layers.0.self_attn.A_log"
    filename = "model-00004-of-00185.safetensors"
    shard = tmp_path / filename
    header = {
        name: {
            "dtype": "NOT_A_DTYPE",
            "shape": [1],
            "data_offsets": [0, 1],
        }
    }
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    shard.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0")
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(metadata_dir, monkeypatch, {name: filename})

    with pytest.raises(ConversionError, match="unsupported dtype"):
        audit_shard(metadata_dir, shard)


def test_audit_shard_rejects_non_divisible_real_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    name = "language_model.model.layers.0.self_attn.qkv_proj.weight"
    filename = "model-00005-of-00185.safetensors"
    shard = tmp_path / filename
    write_synthetic_safetensors(shard, {name: ("U32", seq((15, 4)))})
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(metadata_dir, monkeypatch, {name: filename})

    with pytest.raises(ConversionError, match="segment size"):
        audit_shard(metadata_dir, shard)


def test_convert_shard_writes_valid_pair_and_preserves_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    qkv = "language_model.model.layers.0.self_attn.qkv_proj.weight"
    norm = "language_model.model.layers.0.input_layernorm.weight"
    filename = "model-00006-of-00185.safetensors"
    shard = tmp_path / "source" / filename
    write_synthetic_safetensors(
        shard,
        OrderedDict(
            [
                (qkv, ("U32", seq((18, 4)))),
                (norm, ("BF16", seq((8,), np.uint16))),
            ]
        ),
    )
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(
        metadata_dir,
        monkeypatch,
        {qkv: filename, norm: filename},
    )
    source_before = (shard.read_bytes(), shard.stat().st_mtime_ns)
    rank_dirs = [tmp_path / "rank0", tmp_path / "rank1"]

    report = convert_shard(
        metadata_dir=metadata_dir,
        shard_path=shard,
        rank_dirs=rank_dirs,
        max_buffer_bytes=64,
    )

    assert (shard.read_bytes(), shard.stat().st_mtime_ns) == source_before
    assert report["source"]["modified"] is False
    assert report["source"]["sha256"] == checkpoint._sha256_file(shard)
    assert report["max_buffer_bytes"] == 64
    assert report["transaction"] == {
        "existing_outputs_overwritten": False,
        "both_outputs_validated_before_publish": True,
        "source_deleted": False,
    }
    assert [output["data_bytes"] for output in report["outputs"]] == [160, 160]
    for rank, output in enumerate(report["outputs"]):
        output_path = rank_dirs[rank] / filename
        assert output_path.is_file()
        assert output["path"] == str(output_path.resolve())
        assert output["sha256"] == checkpoint._sha256_file(output_path)
        with SafeTensorFile(output_path) as converted:
            assert converted.metadata["tp_rank"] == str(rank)
            assert converted.metadata["source_file"] == filename
            assert converted.tensors[qkv].shape == (9, 4)
            assert converted.tensors[norm].shape == (8,)
        assert not list(rank_dirs[rank].glob(".*.staged"))
        assert not list(rank_dirs[rank].glob(".*.partial"))


def test_convert_shard_cli_emits_compact_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    name = "language_model.model.layers.0.self_attn.A_log"
    filename = "model-00007-of-00185.safetensors"
    shard = tmp_path / "source" / filename
    write_synthetic_safetensors(shard, {name: ("F32", seq((6,), np.float32))})
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(metadata_dir, monkeypatch, {name: filename})
    rank_dirs = [tmp_path / "rank0", tmp_path / "rank1"]

    status = main(
        [
            "convert-shard",
            "--metadata-dir",
            str(metadata_dir),
            "--shard",
            str(shard),
            "--rank-dir",
            str(rank_dirs[0]),
            "--rank-dir",
            str(rank_dirs[1]),
            "--max-buffer-mib",
            "1",
        ]
    )

    assert status == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "k3-rank-local-tp-shard-conversion/v1"
    assert set(report) == {
        "max_buffer_bytes",
        "outputs",
        "schema",
        "source",
        "source_repo",
        "source_revision",
        "tp_world_size",
        "transaction",
        "transaction_id",
    }
    assert [output["rank"] for output in report["outputs"]] == [0, 1]
    assert [output["data_bytes"] for output in report["outputs"]] == [12, 12]


def test_convert_shard_refuses_duplicate_dirs_and_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    name = "language_model.model.layers.0.self_attn.A_log"
    filename = "model-00008-of-00185.safetensors"
    shard = tmp_path / "source" / filename
    write_synthetic_safetensors(shard, {name: ("F32", seq((6,), np.float32))})
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(metadata_dir, monkeypatch, {name: filename})
    rank0 = tmp_path / "rank0"
    rank1 = tmp_path / "rank1"

    with pytest.raises(ConversionError, match="must be distinct"):
        convert_shard(
            metadata_dir=metadata_dir,
            shard_path=shard,
            rank_dirs=[rank0, rank0],
        )

    rank0.mkdir()
    existing = rank0 / filename
    existing.write_bytes(b"do-not-overwrite")
    with pytest.raises(ConversionError, match="refusing to overwrite"):
        convert_shard(
            metadata_dir=metadata_dir,
            shard_path=shard,
            rank_dirs=[rank0, rank1],
        )
    assert existing.read_bytes() == b"do-not-overwrite"
    assert not rank1.exists()


def test_copy_metadata_uses_allowlist_and_preserves_license(tmp_path: Path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}")
    (metadata / "tokenizer.json").write_text("{}")
    (metadata / "LICENSE").write_text("Kimi K3 license text\n")
    (metadata / ".exo_shard.json").write_text('{"start_layer": 0}')
    (metadata / "model.safetensors.index.json").write_text("{}")
    (metadata / "unexpected.txt").write_text("not part of the allowlist")
    rank_dirs = [tmp_path / "rank0", tmp_path / "rank1"]

    records = checkpoint._copy_metadata_files(metadata, rank_dirs)

    assert records == {
        name: {
            "bytes": (metadata / name).stat().st_size,
            "sha256": hashlib.sha256((metadata / name).read_bytes()).hexdigest(),
        }
        for name in ("LICENSE", "config.json", "tokenizer.json")
    }

    for rank_dir in rank_dirs:
        assert (rank_dir / "config.json").is_file()
        assert (rank_dir / "tokenizer.json").is_file()
        assert (rank_dir / "LICENSE").read_text() == "Kimi K3 license text\n"
        assert not (rank_dir / ".exo_shard.json").exists()
        assert not (rank_dir / "model.safetensors.index.json").exists()
        assert not (rank_dir / "unexpected.txt").exists()
        assert not list(rank_dir.glob(".*.metadata.partial"))


def test_copy_metadata_requires_kimi_license(tmp_path: Path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}")

    with pytest.raises(ConversionError, match="missing Kimi K3 license"):
        checkpoint._copy_metadata_files(
            metadata,
            [tmp_path / "rank0", tmp_path / "rank1"],
        )

    assert not (tmp_path / "rank0").exists()
    assert not (tmp_path / "rank1").exists()


def test_copy_metadata_rejects_source_symlinks(tmp_path: Path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}")
    license_target = tmp_path / "upstream-license"
    license_target.write_text("Kimi K3 license text\n")
    (metadata / "LICENSE").symlink_to(license_target)

    with pytest.raises(ConversionError, match="symlinks are not permitted"):
        checkpoint._copy_metadata_files(
            metadata,
            [tmp_path / "rank0", tmp_path / "rank1"],
        )


def test_copy_metadata_rejects_destination_symlinks(tmp_path: Path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}")
    (metadata / "LICENSE").write_text("Kimi K3 license text\n")
    outside = tmp_path / "outside-config"
    outside.write_text("must remain unchanged")
    rank0 = tmp_path / "rank0"
    rank0.mkdir()
    (rank0 / "config.json").symlink_to(outside)

    with pytest.raises(ConversionError, match="metadata symlink destination"):
        checkpoint._copy_metadata_files(metadata, [rank0, tmp_path / "rank1"])

    assert outside.read_text() == "must remain unchanged"
    assert not (rank0 / "LICENSE").exists()


def test_copy_metadata_reuses_identical_files_without_replacing_them(
    tmp_path: Path,
):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}")
    (metadata / "LICENSE").write_text("Kimi K3 license text\n")
    rank0 = tmp_path / "rank0"
    rank0.mkdir()
    existing = rank0 / "config.json"
    existing.write_text("{}")
    inode_before = existing.stat().st_ino

    checkpoint._copy_metadata_files(metadata, [rank0, tmp_path / "rank1"])

    assert existing.stat().st_ino == inode_before
    assert (rank0 / "LICENSE").read_text() == "Kimi K3 license text\n"


def test_copy_metadata_refuses_different_existing_file_before_publish(
    tmp_path: Path,
):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}")
    (metadata / "LICENSE").write_text("Kimi K3 license text\n")
    rank0 = tmp_path / "rank0"
    rank0.mkdir()
    existing = rank0 / "config.json"
    existing.write_text('{"do": "not overwrite"}')
    rank1 = tmp_path / "rank1"

    with pytest.raises(ConversionError, match="refusing to overwrite different"):
        checkpoint._copy_metadata_files(metadata, [rank0, rank1])

    assert existing.read_text() == '{"do": "not overwrite"}'
    assert not (rank0 / "LICENSE").exists()
    assert not rank1.exists()


def test_copy_metadata_rolls_back_if_atomic_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}")
    (metadata / "LICENSE").write_text("Kimi K3 license text\n")
    rank_dirs = [tmp_path / "rank0", tmp_path / "rank1"]
    real_link = checkpoint.os.link
    calls = 0

    def fail_second_link(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected metadata publish failure")
        return real_link(src, dst)

    monkeypatch.setattr(checkpoint.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected metadata publish failure"):
        checkpoint._copy_metadata_files(metadata, rank_dirs)

    for rank_dir in rank_dirs:
        assert not list(rank_dir.iterdir())


def test_convert_shard_rolls_back_first_staged_rank_if_second_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    name = "language_model.model.layers.0.self_attn.A_log"
    filename = "model-00009-of-00185.safetensors"
    shard = tmp_path / "source" / filename
    write_synthetic_safetensors(shard, {name: ("F32", seq((6,), np.float32))})
    metadata_dir = tmp_path / "metadata"
    write_pinned_test_metadata(metadata_dir, monkeypatch, {name: filename})
    rank_dirs = [tmp_path / "rank0", tmp_path / "rank1"]
    source_before = (shard.read_bytes(), shard.stat().st_mtime_ns)
    real_writer = checkpoint.write_rank_shard

    def fail_rank_one(*args, **kwargs):
        if kwargs["rank"] == 1:
            raise RuntimeError("injected rank-1 failure")
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "write_rank_shard", fail_rank_one)
    with pytest.raises(RuntimeError, match="injected rank-1 failure"):
        convert_shard(
            metadata_dir=metadata_dir,
            shard_path=shard,
            rank_dirs=rank_dirs,
            max_buffer_bytes=64,
        )

    assert (shard.read_bytes(), shard.stat().st_mtime_ns) == source_before
    assert all(not list(directory.iterdir()) for directory in rank_dirs)
