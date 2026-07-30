from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS_ROOT = HERE.parent
sys.path.insert(0, str(TOOLS_ROOT))

import k3_kda_stage_localizer as localizer  # noqa: E402


def test_contract_is_fixed_to_one_load_prompt_128_width_two():
    assert localizer.SCHEMA == "k3-tp2-kda-width2-stage-localizer/v1"
    assert localizer.PROMPT_TOKEN_TARGET == 128
    assert localizer.WIDTH == 2
    assert localizer.COMPLETE_DIVERGENCE not in {"PASS", "FAIL"}
    assert localizer.COMPLETE_NO_DIVERGENCE not in {"PASS", "FAIL"}


def test_runtime_flags_fail_closed_and_allow_candidate_toggle(monkeypatch):
    for name, value in localizer.REQUIRED_RUNTIME_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(localizer.CANDIDATE_EXACT_ENV, "1")
    actual = localizer.require_runtime_flags()
    assert actual[localizer.CANDIDATE_EXACT_ENV] == "1"

    monkeypatch.setenv("MLX_LM_KIMI_K3_PACKED_KDA_WIDE", "0")
    with pytest.raises(localizer.LocalizerError, match="not pinned"):
        localizer.require_runtime_flags()


def test_candidate_toggle_rejects_ambiguous_values(monkeypatch):
    for name, value in localizer.REQUIRED_RUNTIME_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(localizer.CANDIDATE_EXACT_ENV, "true")
    with pytest.raises(localizer.LocalizerError, match="must be 0 or 1"):
        localizer.require_runtime_flags()


def test_stage_and_pairing_contracts_fail_closed():
    with pytest.raises(localizer.LocalizerError, match="inconsistent"):
        localizer.Stage("bad", (object(),), ())

    target = localizer.PathCapture(
        [localizer.Stage("q", (1,), (1,))],
        output=1,
        final_conv_state=2,
        final_ssm_state=3,
    )
    reference = localizer.PathCapture(
        [localizer.Stage("k", (1,), (1,))],
        output=1,
        final_conv_state=2,
        final_ssm_state=3,
    )
    with pytest.raises(localizer.LocalizerError, match="order"):
        localizer.pair_paths(target, reference)


def test_cli_requires_exactly_two_rank_checkpoints(tmp_path):
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
