from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
VERIFY_ROOT = HERE if (HERE / "k3_target_verify.py").is_file() else HERE.parent
TP_TOOLS = (
    VERIFY_ROOT
    if (VERIFY_ROOT / "tp2_benchmark.py").is_file()
    else VERIFY_ROOT.parent / "k3_tp_checkpoint_agent"
)
sys.path.insert(0, str(TP_TOOLS))
sys.path.insert(0, str(VERIFY_ROOT))

import k3_target_verify as subject  # noqa: E402


class FakeArray:
    def __init__(self, data, *, shape=None, dtype="float32"):
        self.data = tuple(data)
        self.shape = tuple(shape or (len(self.data),))
        self.dtype = dtype
        self.nbytes = len(self.data) * 4

    def __add__(self, other):
        assert self.shape == other.shape
        return FakeArray(
            [left + right for left, right in zip(self.data, other.data, strict=True)],
            shape=self.shape,
            dtype=self.dtype,
        )


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeMX:
    @staticmethod
    def array(value):
        return FakeArray(value.data, shape=value.shape, dtype=value.dtype)

    @staticmethod
    def zeros_like(value):
        return FakeArray(
            [0] * len(value.data),
            shape=value.shape,
            dtype=value.dtype,
        )

    @staticmethod
    def eval(*_values):
        return None

    @staticmethod
    def array_equal(left, right):
        return FakeScalar(
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.data == right.data
        )


class ArraysCache:
    def __init__(self, size):
        self.cache = [None] * size
        self.left_padding = None
        self.lengths = None


class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0


def populated_cache():
    result = []
    for layer in range(subject.EXPECTED_LAYER_COUNT):
        if layer < subject.EXPECTED_ARRAY_CACHE_COUNT:
            cache = ArraysCache(2)
            cache.cache = [
                FakeArray([layer, layer + 1], shape=(1, 2), dtype="float16"),
                FakeArray([layer + 2], shape=(1, 1), dtype="float32"),
            ]
        else:
            cache = KVCache()
            # Capacity is deliberately larger than offset. The clone must
            # preserve the full backing shape, not only the logical prefix.
            cache.keys = FakeArray(
                [layer] * 8,
                shape=(1, 1, 256, 1),
                dtype="bfloat16",
            )
            cache.values = FakeArray(
                [layer + 1] * 8,
                shape=(1, 1, 256, 1),
                dtype="bfloat16",
            )
            cache.offset = 64
        result.append(cache)
    return result


def timing(seconds):
    return {
        "critical_path_seconds": seconds,
        "verified_tokens_per_second": 1 / seconds,
        "milliseconds_per_verified_token": seconds * 1000,
        "critical_peak_memory_gb": 410.0,
        "critical_peak_delta_gb": 0.2,
        "per_rank": [],
    }


def equivalence(passed=True):
    return {
        "pass_all_ranks": passed,
        "finite": passed,
        "top1_equal": passed,
        "known_continuation_equal": passed,
        "cosine_similarity": 1.0 if passed else 0.0,
        "max_abs_error": 0.0 if passed else 2.0,
    }


def test_clone_preserves_mixed_layout_capacity_offset_and_values():
    source = populated_cache()
    cloned = subject.clone_k3_cache(source, FakeMX)

    layout = subject.cache_layout(cloned)
    assert layout["class_counts"] == {"ArraysCache": 69, "KVCache": 24}
    assert layout["kv_offset"] == 64
    assert layout["detail"][-1]["keys"]["shape"] == [1, 1, 256, 1]
    subject.assert_cache_value_equivalent(source, cloned, FakeMX)

    for original, copied in zip(
        subject._arrays_in_cache(source),
        subject._arrays_in_cache(cloned),
        strict=True,
    ):
        assert original is not copied


def test_exact_cache_check_rejects_changed_state():
    source = populated_cache()
    cloned = subject.clone_k3_cache(source, FakeMX)
    cloned[0].cache[0] = FakeArray([999, 1], shape=(1, 2), dtype="float16")

    with pytest.raises(subject.VerificationError, match="cache arrays differ"):
        subject.assert_cache_value_equivalent(source, cloned, FakeMX)


def test_cache_layout_rejects_wrong_class_count():
    source = populated_cache()
    source[-1] = ArraysCache(2)
    source[-1].cache = [
        FakeArray([1], shape=(1, 1)),
        FakeArray([2], shape=(1, 1)),
    ]

    with pytest.raises(subject.VerificationError, match="wrong K3 cache-class"):
        subject.cache_layout(source)


def test_equivalence_gate_is_fail_closed():
    limits = subject.EquivalenceLimits(min_cosine=0.999, max_abs_error=1.0)
    record = {
        "finite": True,
        "top1_equal": True,
        "known_continuation_equal": True,
        "cosine_similarity": 0.9995,
        "max_abs_error": 0.5,
    }
    assert subject.equivalence_pass(record, limits)

    for key, failing_value in (
        ("finite", False),
        ("top1_equal", False),
        ("known_continuation_equal", False),
        ("cosine_similarity", 0.998),
        ("max_abs_error", 1.01),
    ):
        candidate = dict(record)
        candidate[key] = failing_value
        assert not subject.equivalence_pass(candidate, limits)


def test_width_summary_and_artifact_status_require_exact_contract():
    records = []
    for width in subject.VERIFY_WIDTHS:
        record = subject.summarize_width(
            width=width,
            target_runs=[timing(0.05), timing(0.06), timing(0.055)],
            sequential_runs=[timing(0.08), timing(0.09), timing(0.085)],
            equivalence=equivalence(),
        )
        assert record["target_forward"]["median_critical_path_seconds"] == 0.055
        assert record["target_speedup_vs_sequential"] > 1.5
        records.append(record)
    assert subject.artifact_status(records) == "PASS"

    records[-1]["equivalence"] = equivalence(False)
    assert subject.artifact_status(records) == "FAIL"
    with pytest.raises(subject.VerificationError, match="exactly"):
        subject.artifact_status(records[:-1])


def test_runtime_contract_splits_converter_and_execution_provenance():
    record = subject.runtime_contract_record(
        runtime_source={"path": "/opt/mlx_lm/models/kimi_k3.py", "sha256": "ab" * 32},
        runtime_digests=["ab" * 32, "ab" * 32],
        attestation={"lossy_requantization_enabled": False},
    )
    assert record["checkpoint_converter_mlx_lm_commit"] == subject.base.MLX_LM_COMMIT
    assert record["checkpoint_mlx_lm_kimi_k3_sha256"] == (
        subject.base.CHECKPOINT_MLX_LM_KIMI_K3_SHA256
    )
    assert record["execution_runtime_mlx_lm_commit"] == (
        subject.base.RUNTIME_MLX_LM_COMMIT
    )
    assert record["execution_runtime_kimi_k3_sha256"] == (
        subject.base.MLX_LM_KIMI_K3_SHA256
    )
    assert (
        record["checkpoint_converter_mlx_lm_commit"]
        != (record["execution_runtime_mlx_lm_commit"])
    )
    assert (
        record["checkpoint_mlx_lm_kimi_k3_sha256"]
        != (record["execution_runtime_kimi_k3_sha256"])
    )


def test_launcher_pins_current_exact_runtime_contract():
    launcher = (VERIFY_ROOT / "launch_k3_target_verify.sh").read_text()
    assert "--backend jaccl-ring" in launcher
    assert "K3_TP_TRANSPORT_CONTRACT" in launcher
    assert "EXO_MLX_K3_VOCAB_PARALLEL_HEAD=1" in launcher
    assert "EXO_MLX_K3_REQUANT_ROUTED_LATENT_MXFP4=0" in launcher
    assert "EXO_MLX_K3_REQUANT_ATTENTION_QKVG_MXFP4=0" in launcher
