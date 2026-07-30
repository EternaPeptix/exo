from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tp2_benchmark as benchmark  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize=True,
    ):
        assert add_generation_prompt is True
        assert tokenize is True
        content = messages[0]["content"]
        payload = [200 + (ord(char) % 31) for char in content]
        return [11, 12] + payload + [91, 92, 93]


def test_exact_prompt_target_preserves_template_affixes():
    tokenizer = FakeTokenizer()
    tokens, mode = benchmark.build_prompt_tokens(tokenizer, "abc", 17)
    assert len(tokens) == 17
    assert tokens[:2] == [11, 12]
    assert tokens[-3:] == [91, 92, 93]
    assert mode == "chat_template_exact_repeated_payload"
    assert tokens == benchmark.build_prompt_tokens(tokenizer, "abc", 17)[0]


def test_prompt_without_target_matches_upstream_chat_template():
    tokens, mode = benchmark.build_prompt_tokens(FakeTokenizer(), "abc", None)
    assert tokens == [11, 12, 204, 205, 206, 91, 92, 93]
    assert mode == "chat_template"


def test_too_small_target_fails_closed():
    with pytest.raises(benchmark.BenchmarkError, match="template overhead"):
        benchmark.build_prompt_tokens(FakeTokenizer(), "abc", 4)


def test_rank_checkpoint_selection_allows_same_host_local_path(tmp_path: Path):
    selected = benchmark.select_rank_checkpoint(
        [tmp_path, tmp_path],
        rank=1,
        world_size=2,
    )
    assert selected == tmp_path.resolve()
    with pytest.raises(benchmark.BenchmarkError, match="requires TP2"):
        benchmark.select_rank_checkpoint([tmp_path, tmp_path], rank=0, world_size=4)


def test_manifest_identity_is_pinned_and_rank_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "rank1"
    model.mkdir()
    config = model / "config.json"
    config.write_bytes(b"pinned config")
    monkeypatch.setattr(
        benchmark,
        "SOURCE_CONFIG_SHA256",
        hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    (model / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    manifest = {
        "schema": benchmark.SCHEMA,
        "complete": True,
        "source": {
            "repo": benchmark.SOURCE_REPO,
            "revision": benchmark.SOURCE_REVISION,
            "config_sha256": benchmark.SOURCE_CONFIG_SHA256,
            "index_sha256": benchmark.SOURCE_INDEX_SHA256,
        },
        "runtime": {
            "mlx_lm_commit": benchmark.MLX_LM_COMMIT,
            "mlx_lm_kimi_k3_sha256": (benchmark.CHECKPOINT_MLX_LM_KIMI_K3_SHA256),
        },
        "tp": {
            "rank": 1,
            "world_size": 2,
            "contract": benchmark.CONTRACT_VERSION,
            "contract_digest": benchmark.CONTRACT_DIGEST,
        },
        "rank_data_bytes": 123,
        "files": {"model.safetensors": {"name": "model.safetensors"}},
        "tensors": {"x": {"rank_shape": [1]}},
    }
    (model / "tp_manifest.json").write_text(json.dumps(manifest))
    result = benchmark.inspect_pinned_manifest(model, rank=1)
    assert result["rank_data_bytes"] == 123
    with pytest.raises(benchmark.BenchmarkError, match="TP rank mismatch"):
        benchmark.inspect_pinned_manifest(model, rank=0)


class FakeDistributed:
    def __init__(self, reject_strict: bool = False, other_error: bool = False):
        self.calls = []
        self.reject_strict = reject_strict
        self.other_error = other_error

    def init(self, **kwargs):
        self.calls.append(kwargs)
        if self.other_error:
            raise TypeError("backend must be a string")
        if self.reject_strict and "strict" in kwargs:
            raise TypeError("got an unexpected keyword argument 'strict'")
        return "group"


class FakeMx:
    def __init__(self, distributed):
        self.distributed = distributed


def transport_contract(tmp_path: Path) -> tuple[Path, list[list[object]]]:
    matrix: list[list[object]] = [
        [None, ["rdma_a", "rdma_b", "rdma_c", "rdma_d"]],
        [["rdma_w", "rdma_x", "rdma_y", "rdma_z"], None],
    ]
    path = tmp_path / "transport-contract.json"
    path.write_text(
        json.dumps(
            {
                "schema": benchmark.TRANSPORT_CONTRACT_SCHEMA,
                "coordinator_host": "192.0.2.10",
                "device_matrix": matrix,
            }
        )
    )
    return path, matrix


def test_distributed_init_requests_strict_and_only_compat_falls_back():
    current = FakeDistributed()
    assert benchmark.init_distributed(FakeMx(current)) == "group"
    assert current.calls == [{"backend": "jaccl", "strict": True}]

    legacy = FakeDistributed(reject_strict=True)
    assert benchmark.init_distributed(FakeMx(legacy)) == "group"
    assert legacy.calls == [
        {"backend": "jaccl", "strict": True},
        {"backend": "jaccl"},
    ]

    broken = FakeDistributed(other_error=True)
    with pytest.raises(TypeError, match="backend"):
        benchmark.init_distributed(FakeMx(broken))


def test_transport_environment_requires_exact_four_rail_ring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract_path, expected_matrix = transport_contract(tmp_path)
    matrix_path = tmp_path / "ibv-devices.json"
    matrix_path.write_text(json.dumps(expected_matrix))
    environment = {
        benchmark.TRANSPORT_CONTRACT_ENV: str(contract_path),
        "MLX_JACCL_RING": "1",
        "MLX_RANK": "1",
        "MLX_JACCL_COORDINATOR": "192.0.2.10:29337",
        "MLX_IBV_DEVICES": str(matrix_path),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    result = benchmark.inspect_jaccl_ring_transport(rank=1)
    assert result["mode"] == "jaccl-ring"
    assert result["ring"] is True
    assert result["device_matrix"] == expected_matrix
    assert result["transport_contract_path"] == str(contract_path.resolve())

    monkeypatch.setenv("MLX_JACCL_RING", "0")
    with pytest.raises(benchmark.BenchmarkError, match="MLX_JACCL_RING=1"):
        benchmark.inspect_jaccl_ring_transport(rank=1)

    monkeypatch.setenv("MLX_JACCL_RING", "1")
    matrix_path.write_text(
        json.dumps(
            [
                [None, ["rdma_a"]],
                [["rdma_w"], None],
            ]
        )
    )
    with pytest.raises(benchmark.BenchmarkError, match="four-rail transport contract"):
        benchmark.inspect_jaccl_ring_transport(rank=1)


def test_transport_environment_rejects_wrong_coordinator_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract_path, expected_matrix = transport_contract(tmp_path)
    matrix_path = tmp_path / "ibv-devices.json"
    matrix_path.write_text(json.dumps(expected_matrix))
    monkeypatch.setenv(benchmark.TRANSPORT_CONTRACT_ENV, str(contract_path))
    monkeypatch.setenv("MLX_JACCL_RING", "1")
    monkeypatch.setenv("MLX_RANK", "0")
    monkeypatch.setenv("MLX_IBV_DEVICES", str(matrix_path))
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "192.0.2.11:29337")
    with pytest.raises(benchmark.BenchmarkError, match="transport contract"):
        benchmark.inspect_jaccl_ring_transport(rank=0)

    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "192.0.2.10:29337")
    with pytest.raises(benchmark.BenchmarkError, match="MLX_RANK"):
        benchmark.inspect_jaccl_ring_transport(rank=1)


def test_transport_contract_is_explicit_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(benchmark.TRANSPORT_CONTRACT_ENV, raising=False)
    with pytest.raises(
        benchmark.BenchmarkError, match=benchmark.TRANSPORT_CONTRACT_ENV
    ):
        benchmark.load_transport_contract()

    path, matrix = transport_contract(tmp_path)
    contract = benchmark.load_transport_contract(path)
    assert contract["coordinator_host"] == "192.0.2.10"
    assert contract["device_matrix"] == matrix

    data = json.loads(path.read_text())
    data["device_matrix"][0][1] = ["rdma_a"] * 4
    path.write_text(json.dumps(data))
    with pytest.raises(benchmark.BenchmarkError, match="four unique"):
        benchmark.load_transport_contract(path)


def test_phase_and_critical_path_metrics_exclude_first_token_from_decode():
    local = benchmark.phase_metrics(
        prompt_tokens=2000,
        generated_tokens=5,
        start=10.0,
        first_token=12.0,
        end=14.0,
    )
    assert local["prefill_tps"] == 1000.0
    assert local["decode_tokens"] == 4
    assert local["decode_tps"] == 2.0

    critical = benchmark.critical_path_metrics(
        [[2.0, 2.0, 4.0, 410.0], [2.2, 2.5, 4.6, 412.0]],
        prompt_tokens=2000,
        generated_tokens=5,
    )
    assert critical["prefill_tps"] == pytest.approx(2000 / 2.2)
    assert critical["decode_tps"] == pytest.approx(4 / 2.5)
    assert critical["e2e_seconds"] == pytest.approx(4.6)
    assert critical["peak_memory_gb"] == 412.0


def test_token_digest_is_width_delimited_and_stable():
    assert benchmark.token_digest([1, 23]) != benchmark.token_digest([12, 3])
    assert benchmark.token_digest([1, 23]) == benchmark.token_digest([1, 23])
    with pytest.raises(benchmark.BenchmarkError, match="outside uint32"):
        benchmark.token_digest([-1])


def test_shared_digest_proof_is_returned_for_artifact_preservation(
    monkeypatch: pytest.MonkeyPatch,
):
    digest = "ab" * 32
    monkeypatch.setattr(
        benchmark,
        "gather_digests",
        lambda _mx, _group, _digest: [digest, digest],
    )
    assert benchmark.require_shared_digest(None, None, digest, "prompt") == [
        digest,
        digest,
    ]

    monkeypatch.setattr(
        benchmark,
        "gather_digests",
        lambda _mx, _group, _digest: [digest, "cd" * 32],
    )
    with pytest.raises(benchmark.BenchmarkError, match="differs across TP ranks"):
        benchmark.require_shared_digest(None, None, digest, "prompt")


def test_atomic_artifact_publish_is_compact_and_no_clobber(tmp_path: Path):
    artifact = tmp_path / "result.json"
    benchmark.atomic_write_json(artifact, {"z": 1, "a": "ok"})
    assert artifact.read_text() == '{"a":"ok","z":1}\n'
    with pytest.raises(benchmark.BenchmarkError, match="refusing to overwrite"):
        benchmark.atomic_write_json(artifact, {"different": True})


def test_parse_args_requires_two_checkpoints(tmp_path: Path):
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--rank-checkpoint",
                str(tmp_path),
                "--artifact",
                str(tmp_path / "x.json"),
            ]
        )
    parsed = benchmark.parse_args(
        [
            "--rank-checkpoint",
            str(tmp_path),
            "--rank-checkpoint",
            str(tmp_path),
            "--artifact",
            str(tmp_path / "x.json"),
            "--prompt-token-target",
            "2048",
            "--output-tokens",
            "8",
        ]
    )
    assert parsed.prompt_token_target == 2048
    assert parsed.output_tokens == 8
    assert parsed.wired_limit is True
    assert parsed.async_lookahead is True

    control = benchmark.parse_args(
        [
            "--rank-checkpoint",
            str(tmp_path),
            "--rank-checkpoint",
            str(tmp_path),
            "--artifact",
            str(tmp_path / "control.json"),
            "--no-wired-limit",
            "--no-async-lookahead",
        ]
    )
    assert control.wired_limit is False
    assert control.async_lookahead is False


def test_help_keeps_argparse_success_exit(capsys: pytest.CaptureFixture[str]):
    assert benchmark.main(["--help"]) == 0
    assert "Direct rank-local Kimi K3" in capsys.readouterr().out


def test_launcher_uses_mlx_launch_runtime_and_explicit_memory_modes():
    launcher = (ROOT / "launch_tp2_jaccl.sh").read_text(encoding="utf-8")
    assert "python_bin=" not in launcher
    assert '"${python_bin}"' not in launcher
    assert '"${benchmark_args[@]}"' in launcher
    assert "K3_TP_TRANSPORT_CONTRACT" in launcher
    assert "K3_TP_WIRED_LIMIT" in launcher
    assert "K3_TP_ASYNC_LOOKAHEAD" in launcher
