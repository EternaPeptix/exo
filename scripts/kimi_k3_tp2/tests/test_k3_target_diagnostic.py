from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS_ROOT = HERE.parent
sys.path.insert(0, str(TOOLS_ROOT))

import k3_target_diagnostic as diagnostic  # noqa: E402
import k3_target_verify as verification  # noqa: E402


def test_diagnostic_is_separate_and_cannot_promote_v3():
    assert verification.ARTIFACT_SCHEMA == "k3-tp2-target-verification/v3"
    assert diagnostic.DIAGNOSTIC_SCHEMA == (
        "k3-tp2-target-divergence-diagnostic/v1"
    )
    assert diagnostic.COMPLETE_DIVERGENCE not in {"PASS", "FAIL"}
    assert diagnostic.COMPLETE_NO_DIVERGENCE not in {"PASS", "FAIL"}
    assert diagnostic.DIAGNOSTIC_WIDTHS == (2, 3, 4, 7, 8)


def test_cosine_from_sums_handles_zero_and_rejects_invalid_values():
    assert diagnostic.cosine_from_sums(0.0, 0.0, 0.0, exact=True) == 1.0
    assert diagnostic.cosine_from_sums(0.0, 0.0, 1.0, exact=False) == 0.0
    assert diagnostic.cosine_from_sums(2.0, 4.0, 1.0, exact=False) == 1.0
    with pytest.raises(diagnostic.DiagnosticError, match="negative"):
        diagnostic.cosine_from_sums(0.0, -1.0, 1.0, exact=False)


def test_capture_validation_is_fail_closed():
    complete = diagnostic.ForwardCapture(
        layer_outputs=[
            (0, object()),
            (1, object()),
            (0, object()),
            (1, object()),
        ],
        final_hidden_states=[object(), object()],
    )
    diagnostic.validate_capture(complete, layer_count=2, forward_calls=2)

    wrong_order = diagnostic.ForwardCapture(
        layer_outputs=[
            (0, object()),
            (1, object()),
            (1, object()),
            (0, object()),
        ],
        final_hidden_states=[object(), object()],
    )
    with pytest.raises(diagnostic.DiagnosticError, match="order"):
        diagnostic.validate_capture(wrong_order, layer_count=2, forward_calls=2)

    missing_hidden = diagnostic.ForwardCapture(
        layer_outputs=[(0, object()), (1, object())],
        final_hidden_states=[],
    )
    with pytest.raises(diagnostic.DiagnosticError, match="hidden"):
        diagnostic.validate_capture(missing_hidden, layer_count=2, forward_calls=1)


def test_capture_context_observes_every_layer_and_restores_types():
    class FakeAttention:
        pass

    class FakeLayer:
        is_linear = True

        def __init__(self):
            self.self_attn = FakeAttention()

        def __call__(self, value):
            return value + 1, None

    layers = [
        FakeLayer() for _ in range(diagnostic.target.EXPECTED_LAYER_COUNT)
    ]

    class FakeTextModel:
        def __call__(self, value):
            for layer in layers:
                value, _ = layer(value)
            return value

    text_model = FakeTextModel()

    class FakeLanguageModel:
        model = text_model

    class FakeModel:
        language_model = FakeLanguageModel()

        @property
        def layers(self):
            return layers

    original_layer_call = FakeLayer.__call__
    original_text_call = FakeTextModel.__call__
    with diagnostic.capture_k3_forward(FakeModel()) as capture:
        assert text_model(0) == diagnostic.target.EXPECTED_LAYER_COUNT
    assert FakeLayer.__call__ is original_layer_call
    assert FakeTextModel.__call__ is original_text_call
    diagnostic.validate_capture(
        capture,
        layer_count=diagnostic.target.EXPECTED_LAYER_COUNT,
        forward_calls=1,
    )


def test_first_divergent_layer_supports_exact_and_v3_limits():
    records = [
        {
            "layer": 0,
            "exact_all_ranks": True,
            "within_v3_numeric_limits_all_ranks": True,
        },
        {
            "layer": 1,
            "exact_all_ranks": False,
            "within_v3_numeric_limits_all_ranks": True,
        },
        {
            "layer": 2,
            "exact_all_ranks": False,
            "within_v3_numeric_limits_all_ranks": False,
        },
    ]
    assert diagnostic.first_divergent_layer(records, criterion="exact") == 1
    assert diagnostic.first_divergent_layer(records, criterion="v3_limits") == 2
    with pytest.raises(diagnostic.DiagnosticError, match="unknown"):
        diagnostic.first_divergent_layer(records, criterion="relaxed")


class ArraysCache:
    def __init__(self):
        self.cache = [object(), object()]
        self.left_padding = None
        self.lengths = None


class KVCache:
    def __init__(self):
        self.keys = object()
        self.values = object()


def test_cache_component_identity_is_stable_and_explicit():
    components = diagnostic.cache_components([ArraysCache(), KVCache()])
    identities = [
        (name, layer, cache_class)
        for name, layer, cache_class, _ in components
    ]
    assert identities == [
        ("layer.000.state.0", 0, "ArraysCache"),
        ("layer.000.state.1", 0, "ArraysCache"),
        ("layer.001.keys", 1, "KVCache"),
        ("layer.001.values", 1, "KVCache"),
    ]


def test_diagnostic_status_never_returns_pass():
    status = diagnostic.diagnostic_status(
        logits={
            "within_v3_numeric_limits_all_positions_all_ranks": False,
            "top1_equal_all_positions_all_ranks": True,
            "known_continuation_equal_all_positions_all_ranks": True,
        },
        final_hidden={"exact_all_positions_all_ranks": True},
        layers={"first_exact_divergent_layer": None},
        final_cache={"exact_all_components_all_ranks": True},
        rollout_logits={
            "within_v3_numeric_limits_all_positions_all_ranks": True,
            "top1_equal_all_positions_all_ranks": True,
            "known_continuation_equal_all_positions_all_ranks": True,
        },
        rollout_cache={"exact_all_components_all_ranks": True},
    )
    assert status == diagnostic.COMPLETE_DIVERGENCE
    assert status != "PASS"

    status = diagnostic.diagnostic_status(
        logits={
            "within_v3_numeric_limits_all_positions_all_ranks": True,
            "top1_equal_all_positions_all_ranks": True,
            "known_continuation_equal_all_positions_all_ranks": True,
        },
        final_hidden={"exact_all_positions_all_ranks": True},
        layers={"first_exact_divergent_layer": None},
        final_cache={"exact_all_components_all_ranks": True},
        rollout_logits={
            "within_v3_numeric_limits_all_positions_all_ranks": True,
            "top1_equal_all_positions_all_ranks": True,
            "known_continuation_equal_all_positions_all_ranks": True,
        },
        rollout_cache={"exact_all_components_all_ranks": True},
    )
    assert status == diagnostic.COMPLETE_NO_DIVERGENCE
    assert status != "PASS"

    status = diagnostic.diagnostic_status(
        logits={
            "within_v3_numeric_limits_all_positions_all_ranks": True,
            "top1_equal_all_positions_all_ranks": False,
            "known_continuation_equal_all_positions_all_ranks": False,
        },
        final_hidden={"exact_all_positions_all_ranks": True},
        layers={"first_exact_divergent_layer": None},
        final_cache={"exact_all_components_all_ranks": True},
        rollout_logits={
            "within_v3_numeric_limits_all_positions_all_ranks": True,
            "top1_equal_all_positions_all_ranks": True,
            "known_continuation_equal_all_positions_all_ranks": True,
        },
        rollout_cache={"exact_all_components_all_ranks": True},
    )
    assert status == diagnostic.COMPLETE_DIVERGENCE


def test_localization_prefers_head_then_layer_then_final_norm():
    common = {
        "final_cache": {
            "exact_all_components_all_ranks": False,
            "first_exact_divergent_layer": 0,
        },
        "rollout_logits": {
            "within_v3_numeric_limits_all_positions_all_ranks": False,
        },
    }
    result = diagnostic.localization_summary(
        logits={"within_v3_numeric_limits_all_positions_all_ranks": False},
        final_hidden={
            "within_v3_numeric_limits_all_positions_all_ranks": True
        },
        layers={
            "first_exact_divergent_layer": 0,
            "first_v3_limit_divergent_layer": None,
        },
        **common,
    )
    assert result["inferred_first_logit_limit_divergence_scope"] == (
        "vocabulary_head_or_logit_projection"
    )

    result = diagnostic.localization_summary(
        logits={"within_v3_numeric_limits_all_positions_all_ranks": False},
        final_hidden={
            "within_v3_numeric_limits_all_positions_all_ranks": False
        },
        layers={
            "first_exact_divergent_layer": 0,
            "first_v3_limit_divergent_layer": 4,
        },
        **common,
    )
    assert result["inferred_first_logit_limit_divergence_scope"] == (
        "decoder_layer_4"
    )

    result = diagnostic.localization_summary(
        logits={"within_v3_numeric_limits_all_positions_all_ranks": False},
        final_hidden={
            "within_v3_numeric_limits_all_positions_all_ranks": False
        },
        layers={
            "first_exact_divergent_layer": None,
            "first_v3_limit_divergent_layer": None,
        },
        **common,
    )
    assert result["inferred_first_logit_limit_divergence_scope"] == (
        "final_normalization"
    )


def test_cli_defaults_to_small_prompt_and_first_t_greater_than_one_width():
    args = diagnostic.parse_args(
        [
            "--rank-checkpoint",
            "/rank0",
            "--rank-checkpoint",
            "/rank1",
            "--artifact",
            "/result.json",
        ]
    )
    assert args.prompt_token_target == 128
    assert args.width == 2

    with pytest.raises(SystemExit):
        diagnostic.parse_args(
            [
                "--rank-checkpoint",
                "/rank0",
                "--rank-checkpoint",
                "/rank1",
                "--artifact",
                "/result.json",
                "--width",
                "1",
            ]
        )


def test_launcher_pins_diagnostic_runtime_without_touching_v3():
    launcher = (TOOLS_ROOT / "launch_k3_target_diagnostic.sh").read_text()
    assert "k3_target_diagnostic.py" in launcher
    assert "--backend jaccl-ring" in launcher
    assert "K3_TP_TRANSPORT_CONTRACT" in launcher
    assert "K3_MLX_CORE_OVERRIDE" in launcher
    assert "MLX_LM_KIMI_K3_FUSED_EXPERTS=0" in launcher
    assert 'K3_TARGET_DIAGNOSTIC_PROMPT_TOKENS:-128' in launcher
    assert 'K3_TARGET_DIAGNOSTIC_WIDTH:-2' in launcher
