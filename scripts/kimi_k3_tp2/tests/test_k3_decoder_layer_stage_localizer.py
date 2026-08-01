from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS_ROOT = HERE.parent
sys.path.insert(0, str(TOOLS_ROOT))

import k3_decoder_layer_stage_localizer as localizer  # noqa: E402


def test_contract_is_fixed_non_promotional_and_small():
    assert localizer.SCHEMA == ("k3-tp2-decoder-layer-width2-stage-localizer/v1")
    assert localizer.PROMPT_TOKEN_TARGET == 128
    assert localizer.WIDTH == 2
    assert localizer.LAYER_INDEX == 0
    assert localizer.COMPLETE_DIVERGENCE not in {"PASS", "FAIL"}
    assert localizer.COMPLETE_NO_DIVERGENCE not in {"PASS", "FAIL"}


def test_runtime_flags_require_accepted_control_and_reject_candidate(monkeypatch):
    for name, value in localizer.kda.REQUIRED_RUNTIME_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(localizer.kda.CANDIDATE_EXACT_ENV, "0")
    monkeypatch.setenv(
        localizer.kda.CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV,
        "0",
    )
    actual = localizer.require_runtime_flags()
    assert actual[localizer.kda.CANDIDATE_EXACT_ENV] == "0"
    assert actual[localizer.kda.CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV] == "0"

    monkeypatch.setenv(
        localizer.kda.CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV,
        "1",
    )
    actual = localizer.require_runtime_flags()
    assert actual[localizer.kda.CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV] == "1"

    monkeypatch.setenv(
        localizer.kda.CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV,
        "true",
    )
    with pytest.raises(localizer.kda.LocalizerError, match="must be 0 or 1"):
        localizer.require_runtime_flags()

    monkeypatch.setenv(localizer.kda.CANDIDATE_EXACT_WIDE_SHORT_CONV_ENV, "0")
    monkeypatch.setenv(localizer.kda.CANDIDATE_EXACT_ENV, "1")
    with pytest.raises(localizer.LocalizerError, match="accepted v6 control"):
        localizer.require_runtime_flags()


def test_path_stage_contract_fails_closed():
    with pytest.raises(localizer.LocalizerError, match="cannot be empty"):
        localizer.PathStage("", (1,))
    with pytest.raises(localizer.LocalizerError, match="inconsistent"):
        localizer.PathStage("bad", (1, 2), ("only_one",), (1, 1))
    with pytest.raises(localizer.LocalizerError, match="repeats"):
        localizer.PathStage("bad", (1, 2), ("x", "x"), (1, 1))


class FakeMX:
    def concatenate(self, values, axis):
        return ("concatenate", axis, tuple(values))

    def eval(self, *values):
        assert values

    def synchronize(self):
        return None


def test_sequential_stage_combination_concatenates_tokens_and_keeps_final_state():
    paths = [
        [
            localizer.PathStage(
                "mixed",
                (f"token-{index}", f"state-{index}"),
                ("tokens", "state"),
                (1, None),
            )
        ]
        for index in range(localizer.WIDTH)
    ]
    combined = localizer.combine_sequential_stages(FakeMX(), paths)
    assert combined[0].values[0] == (
        "concatenate",
        1,
        ("token-0", "token-1"),
    )
    assert combined[0].values[1] == "state-1"

    paths[1] = [localizer.PathStage("other", ("x",), ("tokens",), (1,))]
    with pytest.raises(localizer.LocalizerError, match="descriptors differ"):
        localizer.combine_sequential_stages(FakeMX(), paths)


def test_pairing_rejects_reordered_or_reshaped_stage_contract():
    left = [localizer.PathStage("a", (1,), ("value",), (1,))]
    right = [localizer.PathStage("b", (1,), ("value",), (1,))]
    with pytest.raises(localizer.LocalizerError, match="descriptors differ"):
        localizer.pair_stages(left, right)


@pytest.mark.parametrize(
    ("first", "direct_exact", "speculative_exact", "finish_exact", "scope"),
    [
        (None, True, True, True, "none"),
        (
            "prepare.attention_input_rmsnorm",
            True,
            True,
            True,
            "layer0_input_rmsnorm_width_shape",
        ),
        ("prepare.blocks", True, True, True, "layer0_residual_block_inv_rms"),
        (
            "attention.cache",
            False,
            True,
            True,
            "kda_direct_nonhistory_path",
        ),
        ("attention.cache", False, False, True, "kda_attention_core"),
        (
            "attention.output",
            True,
            True,
            True,
            "upstream_prepare_propagation_into_kda",
        ),
        (
            "moe.router",
            True,
            True,
            False,
            "finish_attention_or_moe_width_shape",
        ),
    ],
)
def test_localization_is_conservative(
    first,
    direct_exact,
    speculative_exact,
    finish_exact,
    scope,
):
    result = localizer.infer_localization(
        natural={"first_exact_divergent_stage": first},
        controlled_direct_attention={
            "exact_all_stages_all_ranks": direct_exact,
            "first_exact_divergent_stage": (
                None if direct_exact else "gated_delta_final_state"
            ),
        },
        controlled_speculative_attention={
            "exact_all_stages_all_ranks": speculative_exact
        },
        controlled_finish={"exact_all_stages_all_ranks": finish_exact},
    )
    assert result["inferred_scope"] == scope


def test_cli_requires_exactly_two_checkpoint_roots(tmp_path):
    artifact = tmp_path / "result.json"
    args = localizer.parse_args(
        [
            "--rank-checkpoint",
            "/rank0",
            "--rank-checkpoint",
            "/rank1",
            "--artifact",
            str(artifact),
        ]
    )
    assert args.rank_checkpoint == ["/rank0", "/rank1"]
    assert args.artifact == artifact
    with pytest.raises(SystemExit):
        localizer.parse_args(
            [
                "--rank-checkpoint",
                "/rank0",
                "--artifact",
                str(artifact),
            ]
        )


def test_launcher_pins_mesh_and_every_accepted_v6_feature():
    launcher = (
        TOOLS_ROOT / "launch_k3_decoder_layer_stage_localizer_current.sh"
    ).read_text()
    target_launcher = (TOOLS_ROOT / "launch_k3_target_verify_current.sh").read_text()
    assert "k3_decoder_layer_stage_localizer.py" in launcher
    assert "--backend jaccl" in launcher
    assert "MLX_JACCL_RING must be unset" in launcher
    assert "MLX_LM_KIMI_K3_EXACT_SPECULATIVE_KDA=0" in launcher
    assert "K3_DECODER_LAYER_LOCALIZER_EXACT_WIDE_SHORT_CONV" in launcher
    assert "MLX_LM_KIMI_K3_EXACT_WIDE_SHORT_CONV=" in launcher
    for feature_state in (
        "EXO_MLX_K3_VOCAB_PARALLEL_HEAD=1",
        "MLX_LM_KIMI_K3_FUSED_EXPERTS=1",
        "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE=1",
        "MLX_LM_KIMI_K3_FUSED_ROUTER=1",
        "MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS=1",
        "MLX_LM_KIMI_K3_PACKED_KDA_SKINNY=1",
        "MLX_LM_KIMI_K3_PACKED_KDA_WIDE=1",
        "MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD=1",
        "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1",
        "MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL=1",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES=laguna8",
        "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE=hidden",
    ):
        assert feature_state in launcher
        assert feature_state in target_launcher


def test_source_requires_full_model_endpoint_reproduction_before_artifact():
    source = (TOOLS_ROOT / "k3_decoder_layer_stage_localizer.py").read_text()
    assert "diagnostic._forward_with_capture" in source
    assert "staged decoder-layer paths do not exactly reproduce" in source
    assert "def _actual_direct_attention_endpoint" in source
    assert "return_state_history=False" in source
    assert "controlled_shared_prepared_input_direct_attention" in source
    assert "controlled_shared_prepared_input_speculative_attention" in source
    assert '"full_model_endpoint_reproduction_required": True' in source
    assert "target.assert_cache_value_equivalent" in source
    assert "base.atomic_write_json" in source
    assert source.index("_full_model_endpoint_reproduction(") < source.rindex(
        "base.atomic_write_json"
    )
