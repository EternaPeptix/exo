from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator import kimi_k3_dspark as dspark_module
from exo.worker.engines.mlx.generator.kimi_k3_dspark import (
    DSPARK_CHECKPOINT_ENV,
    DSPARK_CONSERVATIVE_VERIFY_WIDTH,
    DSPARK_ENABLE_ENV,
    DSPARK_MODEL_NATIVE_VERIFY_WIDTH,
    DSPARK_TELEMETRY_ENV,
    DSPARK_VERIFY_WIDTH_ENV,
    MLX_DSPARK_PROPOSER_ENV,
    MLX_DSPARK_SEGMENTED_SDPA_ENV,
    MLX_REPLAYSSM_ENV,
    DSparkCancellationError,
    DSparkConfigurationError,
    DSparkDistributedStateError,
    DSparkFeatureUnavailableError,
    DSparkRoundResult,
    DSparkRoundTelemetry,
    KimiK3DSparkConfig,
    KimiK3DSparkRequestRuntime,
    KimiK3DSparkRoundEngine,
    LoadedMlxDSpark,
    MlxDSparkFeatures,
    MlxDSparkRequestDraft,
    OrdinaryDecodePlan,
    ReplaySSMTargetAdapter,
    TargetPosterior,
    TargetVerificationPlan,
    accepted_draft_prefix,
    detect_mlx_dspark_features,
    dspark_context_capacity_hint,
    dspark_decode_tokens,
    has_replayssm_target_hooks,
    kimi_k3_dspark_config,
    load_replicated_mlx_dspark,
    preflight_mlx_dspark_segmented_sdpa,
    validate_dspark_greedy_sampling,
    validate_local_dspark_checkpoint,
)


def _enabled_environment(
    checkpoint: Path, *, width: str | None = None
) -> dict[str, str]:
    environment = {
        DSPARK_ENABLE_ENV: "1",
        DSPARK_CHECKPOINT_ENV: str(checkpoint),
        MLX_DSPARK_PROPOSER_ENV: "1",
        MLX_REPLAYSSM_ENV: "1",
    }
    if width is not None:
        environment[DSPARK_VERIFY_WIDTH_ENV] = width
    return environment


def test_dspark_is_inert_by_default_and_rejects_orphan_companions(
    tmp_path: Path,
) -> None:
    assert (
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ={},
        )
        is None
    )

    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_CHECKPOINT_ENV} requires {DSPARK_ENABLE_ENV}=1",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ={DSPARK_CHECKPOINT_ENV: str(tmp_path)},
        )


@pytest.mark.parametrize(
    "missing_opt_in",
    [MLX_DSPARK_PROPOSER_ENV, MLX_REPLAYSSM_ENV],
)
def test_enabled_dspark_requires_both_mlx_lm_opt_ins(
    tmp_path: Path,
    missing_opt_in: str,
) -> None:
    environment = _enabled_environment(tmp_path)
    del environment[missing_opt_in]

    with pytest.raises(
        DSparkConfigurationError,
        match=f"{missing_opt_in}=1 is required",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )


def test_model_native_width_eight_is_the_enabled_default(tmp_path: Path) -> None:
    warnings: list[str] = []
    validated: list[Path] = []
    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=_enabled_environment(tmp_path),
        warning=warnings.append,
        checkpoint_validator=validated.append,
    )

    assert config is not None
    assert config.verify_width == DSPARK_MODEL_NATIVE_VERIFY_WIDTH
    assert config.gamma == 7
    assert config.placement == "replicated"
    assert config.target_layer_ids == (7, 23, 51, 67, 83)
    assert config.target_hidden_state_indices == (7, 23, 51, 67, 83)
    assert validated == [tmp_path]
    assert warnings == []


def test_width_three_requires_explicit_override_and_warns(tmp_path: Path) -> None:
    warnings: list[str] = []
    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=_enabled_environment(
            tmp_path,
            width=str(DSPARK_CONSERVATIVE_VERIFY_WIDTH),
        ),
        warning=warnings.append,
        checkpoint_validator=lambda _path: None,
    )

    assert config is not None
    assert config.verify_width == 3
    assert config.gamma == 2
    assert len(warnings) == 1
    assert "overrides the model-native width 8" in warnings[0]


@pytest.mark.parametrize("width", ["2", "4", "7", "9", " 8", "eight"])
def test_dspark_rejects_unversioned_or_malformed_widths(
    tmp_path: Path,
    width: str,
) -> None:
    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_VERIFY_WIDTH_ENV} must be 3 or 8",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=_enabled_environment(tmp_path, width=width),
            checkpoint_validator=lambda _path: None,
        )


@pytest.mark.parametrize(
    ("is_pipeline", "is_batch", "message"),
    [
        (True, False, "does not support pipeline parallelism"),
        (False, True, "does not support batch generation"),
    ],
)
def test_dspark_rejects_unsupported_generation_modes(
    tmp_path: Path,
    is_pipeline: bool,
    is_batch: bool,
    message: str,
) -> None:
    with pytest.raises(DSparkConfigurationError, match=message):
        kimi_k3_dspark_config(
            is_pipeline=is_pipeline,
            is_batch=is_batch,
            environ=_enabled_environment(tmp_path),
            checkpoint_validator=lambda _path: None,
        )


def test_local_checkpoint_validation_is_hash_pinned(tmp_path: Path) -> None:
    config_contents = b'{"block_size":7}\n'
    (tmp_path / "config.json").write_bytes(config_contents)
    (tmp_path / "model.safetensors").write_bytes(b"x")
    expected = hashlib.sha256(config_contents).hexdigest()

    validate_local_dspark_checkpoint(
        tmp_path,
        expected_config_sha256=expected,
        expected_model_bytes=1,
    )
    with pytest.raises(DSparkConfigurationError, match="pinned config hash"):
        validate_local_dspark_checkpoint(
            tmp_path,
            expected_config_sha256="0" * 64,
            expected_model_bytes=1,
        )


@dataclass
class _FakeDraftRound:
    proposal_tokens: Sequence[int]
    events: list[str]
    commits: list[tuple[int, int, tuple[int, ...]]] = field(default_factory=list)
    cancelled: bool = False
    fail_commit: bool = False
    fail_cancel: bool = False

    def commit(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> None:
        self.events.append("draft_commit")
        if self.fail_commit:
            raise RuntimeError("injected draft commit failure")
        self.commits.append(
            (
                accepted_draft_tokens,
                next_anchor_token,
                target_posterior.tokens,
            )
        )

    def cancel(self) -> None:
        self.events.append("draft_cancel")
        if self.fail_cancel:
            raise RuntimeError("injected draft cancel failure")
        self.cancelled = True


@dataclass
class _FakeDraft:
    proposal_tokens: Sequence[int]
    verify_width: int
    events: list[str]
    placement: str = "replicated"
    rounds: list[_FakeDraftRound] = field(default_factory=list)
    fail_commit: bool = False
    fail_cancel: bool = False
    fail_preflight: bool = False
    fail_build: bool = False
    fail_materialize: bool = False

    def preflight_round(self, anchor_token: int, num_proposals: int) -> int:
        self.events.append("draft_preflight")
        assert num_proposals == self.verify_width - 1
        if self.fail_preflight:
            raise RuntimeError("injected draft preflight failure")
        return 5

    def prepare_round(
        self,
        anchor_token: int,
        num_proposals: int,
    ) -> _FakePreparedDraft:
        self.events.append("draft_build")
        assert num_proposals == self.verify_width - 1
        if self.fail_build:
            raise RuntimeError("injected draft graph-build failure")
        return _FakePreparedDraft(
            owner=self,
            fail_materialize=self.fail_materialize,
        )


@dataclass
class _FakePreparedDraft:
    owner: _FakeDraft
    fail_materialize: bool = False
    cancelled: bool = False

    def materialize(self) -> _FakeDraftRound:
        self.owner.events.append("draft_materialize")
        if self.fail_materialize:
            raise RuntimeError("injected draft materialization failure")
        round_state = _FakeDraftRound(
            self.owner.proposal_tokens,
            self.owner.events,
            fail_commit=self.owner.fail_commit,
            fail_cancel=self.owner.fail_cancel,
        )
        self.owner.rounds.append(round_state)
        return round_state

    def cancel(self) -> None:
        self.owner.events.append("draft_graph_cancel")
        self.cancelled = True


@dataclass
class _FakeTargetRound:
    posterior: TargetPosterior
    events: list[str]
    commits: list[int] = field(default_factory=list)
    cancelled: bool = False
    fail_commit: bool = False

    def commit(self, consumed_input_tokens: int) -> None:
        self.events.append("target_commit")
        if self.fail_commit:
            raise RuntimeError("injected target commit failure")
        self.commits.append(consumed_input_tokens)

    def cancel(self) -> None:
        self.events.append("target_cancel")
        self.cancelled = True


@dataclass
class _FakeTarget:
    posterior_tokens: Sequence[int]
    events: list[str]
    ordinary_token: int = 999
    rounds: list[_FakeTargetRound] = field(default_factory=list)
    ordinary_anchors: list[int] = field(default_factory=list)
    fail_commit: bool = False
    fail_prepare: bool = False
    fail_build: bool = False
    fail_verify: bool = False
    fail_ordinary_preflight: bool = False
    fail_ordinary_build: bool = False
    fail_ordinary_materialize: bool = False
    ordinary_mode: Literal["full", "compact"] = "full"
    verification_mode_code: int = 0

    def prepare_verification(
        self,
        proposal_block: tuple[int, ...],
    ) -> _FakePreparedTarget:
        self.events.append("target_prepare")
        if self.fail_prepare:
            raise RuntimeError("injected target prepare failure")
        return _FakePreparedTarget(
            owner=self,
            proposal_block=proposal_block,
            initial_offset=5,
            mode_code=self.verification_mode_code,
        )

    def preflight_ordinary(self, anchor_token: int) -> OrdinaryDecodePlan:
        self.events.append("ordinary_preflight")
        if self.fail_ordinary_preflight:
            raise RuntimeError("injected ordinary preflight failure")
        return OrdinaryDecodePlan(
            5,
            self.ordinary_mode,
        )

    def prepare_ordinary(
        self,
        anchor_token: int,
        plan: OrdinaryDecodePlan,
    ) -> _FakePreparedOrdinary:
        self.events.append("ordinary_build")
        if self.fail_ordinary_build:
            raise RuntimeError("injected ordinary build failure")
        return _FakePreparedOrdinary(self, anchor_token, plan)


@dataclass
class _FakePreparedTarget:
    owner: _FakeTarget
    proposal_block: tuple[int, ...]
    initial_offset: int
    mode_code: int
    cancelled: bool = False

    def build(self) -> _FakeBuiltTarget:
        self.owner.events.append("target_build")
        if self.owner.fail_build:
            raise RuntimeError("injected target graph-build failure")
        return _FakeBuiltTarget(self.owner, self.proposal_block)

    def cancel(self) -> None:
        self.owner.events.append("target_cancel")
        self.cancelled = True


@dataclass
class _FakeBuiltTarget:
    owner: _FakeTarget
    proposal_block: tuple[int, ...]
    cancelled: bool = False

    def materialize(self) -> _FakeTargetRound:
        self.owner.events.append("target_verify")
        if self.owner.fail_verify:
            raise RuntimeError("injected target verification failure")
        round_state = _FakeTargetRound(
            TargetPosterior(tuple(self.owner.posterior_tokens), ("hidden-taps",)),
            self.owner.events,
            fail_commit=self.owner.fail_commit,
        )
        self.owner.rounds.append(round_state)
        return round_state

    def cancel(self) -> None:
        self.owner.events.append("target_cancel")
        self.cancelled = True


@dataclass
class _FakePreparedOrdinary:
    owner: _FakeTarget
    anchor_token: int
    plan: OrdinaryDecodePlan

    def materialize(self) -> int:
        self.owner.events.append("ordinary_decode")
        if self.owner.fail_ordinary_materialize:
            raise RuntimeError("injected ordinary materialization failure")
        self.owner.ordinary_anchors.append(self.anchor_token)
        return self.owner.ordinary_token


@dataclass
class _FakeAgreement:
    events: list[str]
    reject_proposal: bool = False
    reject_acceptance: bool = False
    rank: int = 0
    size: int = 2
    stage_outcomes: list[bool | None] = field(default_factory=list)
    agreed_tokens: list[int | None] = field(default_factory=list)

    def agree_proposal_block(
        self,
        local_block: tuple[int, ...] | None,
        expected_width: int,
    ) -> tuple[int, ...] | None:
        self.events.append("agree_proposal")
        if self.reject_proposal or local_block is None:
            return None
        assert len(local_block) == expected_width
        return local_block

    def agree_acceptance(
        self,
        local_boundary: int | None,
        local_next_token: int | None,
        maximum_boundary: int,
    ) -> tuple[int, int] | None:
        self.events.append("agree_acceptance")
        if self.reject_acceptance or local_boundary is None or local_next_token is None:
            return None
        assert local_boundary <= maximum_boundary
        return local_boundary, local_next_token

    def agree_stage_success(self, local_success: bool) -> bool | None:
        self.events.append(f"agree_stage_{int(local_success)}")
        if self.stage_outcomes:
            return self.stage_outcomes.pop(0)
        return local_success

    def agree_token(self, local_token: int | None) -> int | None:
        if self.agreed_tokens:
            return self.agreed_tokens.pop(0)
        return local_token


def _config(tmp_path: Path, width: int) -> KimiK3DSparkConfig:
    assert width in (3, 8)
    return KimiK3DSparkConfig(
        checkpoint_path=tmp_path,
        verify_width=width,
        round_telemetry=False,
    )


def _engine(
    tmp_path: Path,
    *,
    proposals: Sequence[int],
    posterior: Sequence[int],
    agreement: _FakeAgreement | None = None,
    terminal_token_ids: tuple[int, ...] = (),
) -> tuple[
    KimiK3DSparkRoundEngine,
    _FakeDraft,
    _FakeTarget,
    _FakeAgreement,
    list[str],
]:
    events = agreement.events if agreement is not None else []
    width = len(proposals) + 1
    draft = _FakeDraft(proposals, width, events)
    target = _FakeTarget(posterior, events)
    collective = agreement or _FakeAgreement(events)
    engine = KimiK3DSparkRoundEngine(
        config=_config(tmp_path, width),
        draft=draft,
        target=target,
        collective=collective,
        terminal_token_ids=terminal_token_ids,
    )
    return engine, draft, target, collective, events


def test_width_eight_accepts_seven_proposals_and_bonus_target_token(
    tmp_path: Path,
) -> None:
    proposals = tuple(range(11, 18))
    posterior = (*proposals, 18)
    engine, draft, target, _collective, events = _engine(
        tmp_path,
        proposals=proposals,
        posterior=posterior,
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == tuple(range(11, 19))
    assert result.telemetry.proposed == 7
    assert result.telemetry.accepted == 7
    assert result.telemetry.emitted == 8
    assert result.telemetry.fallback is False
    assert target.rounds[0].commits == [8]
    assert draft.rounds[0].commits == [(7, 18, posterior)]
    assert events == [
        "draft_preflight",
        "agree_stage_1",
        "draft_build",
        "agree_stage_1",
        "draft_materialize",
        "agree_proposal",
        "target_prepare",
        "agree_stage_1",
        "target_build",
        "agree_stage_1",
        "target_verify",
        "agree_acceptance",
        "target_commit",
        "agree_stage_1",
        "draft_commit",
        "agree_stage_1",
    ]


def test_width_three_override_commits_anchor_plus_accepted_prefix(
    tmp_path: Path,
) -> None:
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(21, 22),
        posterior=(21, 99, 100),
    )

    result = engine.decode_round(20)

    assert result.emitted_tokens == (21, 99)
    assert result.telemetry.proposed == 2
    assert result.telemetry.accepted == 1
    assert result.telemetry.emitted == 2
    assert target.rounds[0].commits == [2]
    assert draft.rounds[0].commits == [(1, 99, (21, 99, 100))]


def test_terminal_proposal_is_reclassified_as_bonus_before_commit(
    tmp_path: Path,
) -> None:
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        terminal_token_ids=(12,),
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (11, 12)
    assert result.telemetry.accepted == 1
    assert target.rounds[0].commits == [2]
    assert draft.rounds[0].commits == [(1, 12, (11, 12, 13))]


def test_acceptance_matches_cumprod_prefix_semantics() -> None:
    assert (
        accepted_draft_prefix(
            (10, 11, 12, 13, 14, 15, 16, 17),
            (11, 12, 99, 14, 15, 16, 17, 18),
        )
        == 2
    )


def test_proposal_disagreement_cancels_draft_and_falls_back_permanently(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, reject_proposal=True)
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=tuple(range(11, 18)),
        posterior=tuple(range(11, 19)),
        agreement=agreement,
    )

    first = engine.decode_round(10)
    second = engine.decode_round(999)

    assert first.emitted_tokens == (999,)
    assert first.telemetry.fallback is True
    assert first.telemetry.error == "DSpark proposal tokens disagreed across ranks"
    assert draft.rounds[0].cancelled is True
    assert target.rounds == []
    assert second.emitted_tokens == (999,)
    assert second.telemetry.fallback is True
    assert len(draft.rounds) == 1
    assert target.ordinary_anchors == [10, 999]


def test_peer_draft_preflight_failure_never_builds_or_materializes_tp_graph(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, stage_outcomes=[None, None])
    engine, _draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    with pytest.raises(DSparkDistributedStateError, match="readiness"):
        engine.decode_round(10)

    assert "draft_build" not in events
    assert "draft_materialize" not in events
    assert "ordinary_build" not in events
    assert "ordinary_decode" not in events
    assert target.rounds == []


def test_decode_anchor_disagreement_prevents_every_lazy_tp_graph(tmp_path: Path) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, agreed_tokens=[None])
    engine, _draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    with pytest.raises(DSparkDistributedStateError, match="anchor disagreed"):
        engine.decode_round(10)

    assert events == []
    assert target.rounds == []
    assert target.ordinary_anchors == []


def test_peer_draft_graph_build_failure_cancels_before_lazy_tp_materialization(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, stage_outcomes=[True, None, True])
    engine, _draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (999,)
    assert "draft_graph_cancel" in events
    assert "draft_materialize" not in events
    assert "target_prepare" not in events
    assert target.ordinary_anchors == [10]


def test_peer_target_prepare_failure_cancels_open_transaction_before_forward(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(
        events,
        stage_outcomes=[True, True, None, True],
    )
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (999,)
    assert "target_cancel" in events
    assert draft.rounds[0].cancelled is True
    assert "target_build" not in events
    assert "target_verify" not in events


def test_target_verifier_mode_disagreement_cancels_before_graph_build(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(
        events,
        agreed_tokens=[10, 0, 5, 0, 0, 5, None],
    )
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (999,)
    assert "target_cancel" in events
    assert draft.rounds[0].cancelled is True
    assert "target_build" not in events
    assert "target_verify" not in events


def test_peer_target_graph_build_failure_cancels_before_target_materialization(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(
        events,
        stage_outcomes=[True, True, True, None, True],
    )
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (999,)
    assert "target_cancel" in events
    assert draft.rounds[0].cancelled is True
    assert "target_verify" not in events
    assert target.rounds == []


def test_ordinary_mode_disagreement_never_builds_target_tp_graph(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, agreed_tokens=[10, 0, None])
    engine, _draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    with pytest.raises(DSparkDistributedStateError, match="readiness"):
        engine.decode_ordinary_tail(10)

    assert "ordinary_build" not in events
    assert "ordinary_decode" not in events
    assert target.ordinary_anchors == []


def test_peer_ordinary_graph_build_failure_never_materializes_target_tp_graph(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, stage_outcomes=[True, None])
    engine, _draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    with pytest.raises(DSparkDistributedStateError, match="graph build"):
        engine.decode_ordinary_tail(10)

    assert "ordinary_build" in events
    assert "ordinary_decode" not in events
    assert target.ordinary_anchors == []


def test_cancel_failure_is_rank_agreed_and_never_enters_fallback(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, reject_proposal=True)
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )
    draft.fail_cancel = True

    with pytest.raises(DSparkDistributedStateError, match="cancellation failed"):
        engine.decode_round(10)

    assert target.ordinary_anchors == []
    assert "agree_stage_0" in events


def test_ordinary_fallback_token_disagreement_is_fail_stop(tmp_path: Path) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(
        events,
        reject_proposal=True,
        # Decode anchor; draft gates; cancellation; ordinary anchor, plan/build,
        # materialization; then inject final token disagreement.
        agreed_tokens=[10, 0, 5, 0, 0, 10, 0, 10, 0, 0, None],
    )
    engine, _draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    with pytest.raises(DSparkDistributedStateError, match="token disagreed"):
        engine.decode_round(10)

    assert target.ordinary_anchors == [10]


def test_uncertain_target_verification_cancel_is_fail_stop(tmp_path: Path) -> None:
    events: list[str] = []
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=_FakeAgreement(events),
    )

    def uncertain(_proposal_block: tuple[int, ...]) -> _FakePreparedTarget:
        raise DSparkCancellationError("target cancellation is uncertain")

    target.prepare_verification = uncertain  # type: ignore[method-assign]

    with pytest.raises(DSparkDistributedStateError, match="cancellation failed"):
        engine.decode_round(10)

    assert draft.rounds[0].cancelled is True
    assert target.ordinary_anchors == []


def test_acceptance_disagreement_rolls_back_both_caches_before_fallback(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(events, reject_acceptance=True)
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=tuple(range(11, 18)),
        posterior=tuple(range(11, 19)),
        agreement=agreement,
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (999,)
    assert result.telemetry.fallback is True
    assert draft.rounds[0].cancelled is True
    assert target.rounds[0].cancelled is True
    assert events == [
        "draft_preflight",
        "agree_stage_1",
        "draft_build",
        "agree_stage_1",
        "draft_materialize",
        "agree_proposal",
        "target_prepare",
        "agree_stage_1",
        "target_build",
        "agree_stage_1",
        "target_verify",
        "agree_acceptance",
        "target_cancel",
        "draft_cancel",
        "agree_stage_1",
        "ordinary_preflight",
        "agree_stage_1",
        "ordinary_build",
        "agree_stage_1",
        "ordinary_decode",
        "agree_stage_1",
    ]


def test_target_commit_outcome_disagreement_is_fail_stop(tmp_path: Path) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(
        events,
        stage_outcomes=[True, True, True, True, None],
    )
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    with pytest.raises(DSparkDistributedStateError, match="disagreed across ranks"):
        engine.decode_round(10)

    assert target.rounds[0].commits == [3]
    assert draft.rounds[0].cancelled is True
    assert target.ordinary_anchors == []


def test_target_commit_failure_never_falls_back_after_commit_phase(
    tmp_path: Path,
) -> None:
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
    )
    target.fail_commit = True

    with pytest.raises(DSparkDistributedStateError, match="failed on every rank"):
        engine.decode_round(10)

    assert target.rounds[0].cancelled is True
    assert draft.rounds[0].cancelled is True
    assert target.ordinary_anchors == []


def test_draft_commit_disagreement_disables_draft_on_every_rank(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    agreement = _FakeAgreement(
        events,
        stage_outcomes=[True, True, True, True, True, None],
    )
    engine, draft, target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
    )

    first = engine.decode_round(10)
    second = engine.decode_round(13)

    assert first.emitted_tokens == (11, 12, 13)
    assert first.telemetry.error == (
        "draft commit outcome disagreed across ranks; DSpark disabled on every rank"
    )
    assert second.emitted_tokens == (999,)
    assert len(draft.rounds) == 1
    assert target.ordinary_anchors == [13]


@dataclass
class _FakeTransaction:
    active: bool = True


@dataclass
class _FakeHookTarget:
    events: list[tuple[str, int]] = field(default_factory=list)

    def forward_with_aux_hidden_states(self) -> None:
        return None

    def begin_speculative_cache(self, cache: object, width: int) -> _FakeTransaction:
        self.events.append(("begin", width))
        return _FakeTransaction()

    def resolve_speculative_cache(
        self,
        transaction: object,
        consumed: int,
    ) -> None:
        assert isinstance(transaction, _FakeTransaction)
        self.events.append(("resolve", consumed))
        transaction.active = False

    def cancel_speculative_cache(self, transaction: object) -> None:
        assert isinstance(transaction, _FakeTransaction)
        self.events.append(("cancel", 0))
        transaction.active = False


@dataclass(frozen=True)
class _FakePosteriorGraph:
    operation: object

    def materialize(self) -> TargetPosterior:
        callback = cast("Callable[[], TargetPosterior]", self.operation)
        return callback()


@dataclass(frozen=True)
class _FakeOrdinaryGraph:
    anchor: int

    def materialize(self) -> int:
        return self.anchor + 1


def test_replayssm_adapter_feature_detects_and_commits_accepted_prefix() -> None:
    target = _FakeHookTarget()
    adapter = ReplaySSMTargetAdapter(
        target,
        target_cache=object(),
        verification_plan=lambda _block: TargetVerificationPlan("full"),
        build_verify=lambda block, _plan: _FakePosteriorGraph(
            lambda: TargetPosterior(tuple(range(20, 20 + len(block))))
        ),
        preflight_ordinary=lambda _anchor: OrdinaryDecodePlan(0, "full"),
        prepare_ordinary=lambda anchor, _plan: _FakeOrdinaryGraph(anchor),
    )

    assert has_replayssm_target_hooks(target)
    prepared = adapter.prepare_verification((10, 11, 12))
    transaction = prepared.build().materialize()
    transaction.commit(2)

    assert transaction.posterior.tokens == (20, 21, 22)
    assert target.events == [("begin", 3), ("resolve", 2)]
    plan = adapter.preflight_ordinary(30)
    assert adapter.prepare_ordinary(30, plan).materialize() == 31


def test_replayssm_adapter_cancels_when_target_verification_raises() -> None:
    target = _FakeHookTarget()

    def fail(
        _block: tuple[int, ...],
        _plan: TargetVerificationPlan,
    ) -> _FakePosteriorGraph:
        raise RuntimeError("verification failed")

    adapter = ReplaySSMTargetAdapter(
        target,
        target_cache=object(),
        verification_plan=lambda _block: TargetVerificationPlan("full"),
        build_verify=fail,
        preflight_ordinary=lambda _anchor: OrdinaryDecodePlan(0, "full"),
        prepare_ordinary=lambda anchor, _plan: _FakeOrdinaryGraph(anchor),
    )

    with pytest.raises(RuntimeError, match="verification failed"):
        adapter.prepare_verification((10, 11, 12)).build()
    assert target.events == [("begin", 3), ("cancel", 0)]


def test_replayssm_adapter_surfaces_uncertain_cancel() -> None:
    class Target(_FakeHookTarget):
        def cancel_speculative_cache(self, transaction: object) -> None:
            del transaction
            raise RuntimeError("cancel failed")

    target = Target()

    def fail(
        _block: tuple[int, ...],
        _plan: TargetVerificationPlan,
    ) -> _FakePosteriorGraph:
        return _FakePosteriorGraph(
            lambda: (_ for _ in ()).throw(RuntimeError("verify failed"))
        )

    adapter = ReplaySSMTargetAdapter(
        target,
        target_cache=object(),
        verification_plan=lambda _block: TargetVerificationPlan("full"),
        build_verify=fail,
        preflight_ordinary=lambda _anchor: OrdinaryDecodePlan(0, "full"),
        prepare_ordinary=lambda anchor, _plan: _FakeOrdinaryGraph(anchor),
    )

    with pytest.raises(DSparkCancellationError, match="could not be cancelled"):
        adapter.prepare_verification((10, 11, 12)).build().materialize()


def test_replayssm_adapter_rejects_incomplete_target_api() -> None:
    with pytest.raises(DSparkFeatureUnavailableError, match="missing hidden-tap"):
        ReplaySSMTargetAdapter(
            object(),
            target_cache=object(),
            verification_plan=lambda _block: TargetVerificationPlan("full"),
            build_verify=lambda _block, _plan: _FakePosteriorGraph(
                lambda: TargetPosterior((1, 2, 3))
            ),
            preflight_ordinary=lambda _anchor: OrdinaryDecodePlan(0, "full"),
            prepare_ordinary=lambda anchor, _plan: _FakeOrdinaryGraph(anchor),
        )


@dataclass(frozen=True)
class _FakeTokenArray:
    rows: list[list[int]]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def tolist(self) -> object:
        return self.rows


@dataclass(frozen=True)
class _FakeHidden:
    shape: tuple[int, int, int]
    name: str

    def __getitem__(self, key: tuple[object, ...]) -> _FakeHidden:
        assert len(key) == 3
        token_slice = key[1]
        assert isinstance(token_slice, slice)
        stop = cast(int, token_slice.stop)
        return _FakeHidden(
            (self.shape[0], stop, self.shape[2]),
            self.name,
        )


@dataclass
class _FakeContextCache:
    length: int = 0
    keys: object | None = None
    values: object | None = None


@dataclass
class _FakeMlxProposer:
    verify_width: int
    calls: list[tuple[object, ...]]
    proposal_rows: list[list[int]]

    def make_context_cache(self, *, capacity_hint: int = 0) -> object:
        self.calls.append(("make_context_cache", capacity_hint))
        return [_FakeContextCache(), _FakeContextCache()]

    def propose(self, anchor_token: int, context_cache: object) -> object:
        self.calls.append(("propose", anchor_token, context_cache))
        return SimpleNamespace(
            tokens=_FakeTokenArray(self.proposal_rows),
            verify_width=self.verify_width,
        )

    def append_target_context(
        self,
        aux_hidden_states: Sequence[object],
        context_offset: int,
        context_cache: object,
    ) -> None:
        assert isinstance(context_cache, list)
        cache_entries = cast(list[object], context_cache)
        hidden = tuple(aux_hidden_states)
        assert hidden
        assert all(isinstance(value, _FakeHidden) for value in hidden)
        shapes = tuple(
            value.shape for value in hidden if isinstance(value, _FakeHidden)
        )
        self.calls.append(("append_target_context", context_offset, shapes))
        consumed = shapes[0][1]
        for entry in cache_entries:
            assert isinstance(entry, _FakeContextCache)
            entry.length += consumed
            entry.keys = object()
            entry.values = object()


def _mock_mlx_features(
    calls: list[tuple[object, ...]],
    proposal_rows: list[list[int]],
) -> MlxDSparkFeatures:
    def load(
        checkpoint_path: Path,
        target_model: object,
        *,
        verify_weights_sha256: bool,
    ) -> object:
        calls.append(("load", checkpoint_path, target_model, verify_weights_sha256))
        return "draft"

    def make_proposer(
        drafter: object,
        *,
        verify_width: int,
        screening_override: bool,
    ) -> object:
        calls.append(("proposer", drafter, verify_width, screening_override))
        return _FakeMlxProposer(verify_width, calls, proposal_rows)

    return MlxDSparkFeatures(
        load_kimi_k3_dspark=load,
        proposer_type=make_proposer,
    )


def test_feature_detection_matches_actual_module_level_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load(_path: object, _target: object, *, verify_weights_sha256: bool) -> None:
        del verify_weights_sha256

    class Proposer:
        pass

    module = SimpleNamespace(
        load_kimi_k3_dspark=load,
        KimiK3DSparkProposer=Proposer,
    )

    def import_module(_name: str) -> object:
        return module

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.kimi_k3_dspark.importlib.import_module",
        import_module,
    )

    features = detect_mlx_dspark_features()

    assert features is not None
    assert features.load_kimi_k3_dspark is load
    assert features.proposer_type is Proposer


@pytest.mark.parametrize(
    ("width", "screening_override", "proposal_rows"),
    [
        (8, False, [[11, 12, 13, 14, 15, 16, 17]]),
        (3, True, [[11, 12]]),
    ],
)
def test_replicated_loader_uses_exact_mlx_signatures_and_proposer_context(
    tmp_path: Path,
    width: int,
    screening_override: bool,
    proposal_rows: list[list[int]],
) -> None:
    calls: list[tuple[object, ...]] = []
    target_model = object()

    loaded = load_replicated_mlx_dspark(
        _config(tmp_path, width),
        target_model,
        features=_mock_mlx_features(calls, proposal_rows),
        evaluate=lambda *_values: None,
    )

    assert loaded.placement == "replicated"
    assert loaded.verify_width == width
    assert calls == [
        ("load", tmp_path, target_model, True),
        ("proposer", "draft", width, screening_override),
    ]
    first = loaded.new_request(capacity_hint=100)
    second = loaded.new_request(capacity_hint=200)
    assert first.context_cache is not second.context_cache
    assert calls[-2:] == [
        ("make_context_cache", 100),
        ("make_context_cache", 200),
    ]


def test_replicated_draft_flattens_proposals_and_appends_only_committed_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MLX_DSPARK_PROPOSER_ENV, "1")
    calls: list[tuple[object, ...]] = []
    loaded = load_replicated_mlx_dspark(
        _config(tmp_path, 3),
        object(),
        features=_mock_mlx_features(calls, [[11, 12]]),
        evaluate=lambda *_values: None,
    )
    request = loaded.new_request(capacity_hint=100)
    assert isinstance(request.context_cache, list)
    context_cache = cast(list[_FakeContextCache], request.context_cache)
    for entry in context_cache:
        entry.length = 5
        entry.keys = object()
        entry.values = object()

    request.preflight_round(10, 2)
    round_state = request.prepare_round(10, 2).materialize()
    assert tuple(round_state.proposal_tokens) == (11, 12)
    round_state.commit(
        1,
        99,
        TargetPosterior(
            (11, 99, 100),
            (
                _FakeHidden((1, 3, 7168), "layer-7"),
                _FakeHidden((1, 3, 7168), "layer-23"),
                _FakeHidden((1, 3, 7168), "layer-51"),
                _FakeHidden((1, 3, 7168), "layer-67"),
                _FakeHidden((1, 3, 7168), "layer-83"),
            ),
        ),
    )

    assert [entry.length for entry in context_cache] == [7, 7]
    assert calls[-2:] == [
        ("propose", 10, context_cache),
        (
            "append_target_context",
            5,
            (
                (1, 2, 7168),
                (1, 2, 7168),
                (1, 2, 7168),
                (1, 2, 7168),
                (1, 2, 7168),
            ),
        ),
    ]
    with pytest.raises(RuntimeError, match="no longer active"):
        round_state.commit(1, 99, TargetPosterior((11, 99, 100)))


@pytest.mark.parametrize(
    ("hidden", "message"),
    [
        (
            tuple(_FakeHidden((1, 3, 7168), f"tap-{index}") for index in range(4)),
            "exactly 5",
        ),
        (
            tuple(_FakeHidden((1, 3, 7168), f"tap-{index}") for index in range(4))
            + (_FakeHidden((1, 3, 1024), "bad-width"),),
            "7168",
        ),
    ],
)
def test_request_draft_rejects_non_exact_target_tap_contract(
    hidden: tuple[_FakeHidden, ...],
    message: str,
) -> None:
    request = MlxDSparkRequestDraft(
        proposer=_FakeMlxProposer(3, [], [[11, 12]]),
        context_cache=[_FakeContextCache(), _FakeContextCache()],
        verify_width=3,
        evaluate=lambda *_values: None,
    )

    with pytest.raises(ValueError, match=message):
        request.seed_target_context(hidden, expected_width=3)


@pytest.mark.parametrize(
    "proposal_rows",
    [
        [[11]],
        [[11, 12], [11, 12]],
        [[11, -1]],
    ],
)
def test_replicated_draft_rejects_malformed_mlx_proposals(
    tmp_path: Path,
    proposal_rows: list[list[int]],
) -> None:
    loaded = load_replicated_mlx_dspark(
        _config(tmp_path, 3),
        object(),
        features=_mock_mlx_features([], proposal_rows),
        evaluate=lambda *_values: None,
    )
    request = loaded.new_request(capacity_hint=100)

    with pytest.raises((TypeError, ValueError)):
        request.prepare_round(10, 2).materialize()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (DSPARK_ENABLE_ENV, "true"),
        (DSPARK_TELEMETRY_ENV, "true"),
        (DSPARK_TELEMETRY_ENV, "2"),
    ],
)
def test_dspark_flags_accept_only_zero_or_one(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    environment = _enabled_environment(tmp_path)
    environment[name] = value

    with pytest.raises(DSparkConfigurationError, match=f"{name} must be 0 or 1"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )


def test_segmented_dspark_preflight_is_inert_when_disabled() -> None:
    imports: list[str] = []

    preflight_mlx_dspark_segmented_sdpa(
        environ={MLX_DSPARK_SEGMENTED_SDPA_ENV: "0"},
        module_loader=lambda name: imports.append(name),
    )

    assert imports == []


def test_segmented_dspark_preflight_requires_bounded_metal_capability() -> None:
    def reject() -> None:
        raise RuntimeError("MLX exposes composite_v1 only")

    module = SimpleNamespace(require_kimi_k3_dspark_segmented_sdpa=reject)

    with pytest.raises(DSparkFeatureUnavailableError, match="bounded_memory_metal_v1"):
        preflight_mlx_dspark_segmented_sdpa(
            environ={MLX_DSPARK_SEGMENTED_SDPA_ENV: "1"},
            module_loader=lambda _name: module,
        )


def test_segmented_dspark_preflight_rejects_ambiguous_flag() -> None:
    with pytest.raises(DSparkConfigurationError, match="must be 0 or 1"):
        preflight_mlx_dspark_segmented_sdpa(
            environ={MLX_DSPARK_SEGMENTED_SDPA_ENV: "true"}
        )


def test_greedy_sampling_accepts_only_canonical_neutral_defaults() -> None:
    validate_dspark_greedy_sampling(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.05,
        logprobs=False,
        top_logprobs=None,
        repetition_penalty=1.0,
        repetition_context_size=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    validate_dspark_greedy_sampling(
        temperature=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        logprobs=False,
        top_logprobs=None,
        repetition_penalty=None,
        repetition_context_size=None,
        presence_penalty=None,
        frequency_penalty=None,
    )


@pytest.mark.parametrize(
    ("top_p", "top_k", "min_p"),
    [
        (0.9, None, None),
        (None, 40, None),
        (None, None, 0.1),
    ],
)
def test_greedy_sampling_rejects_ignored_nondefault_filters(
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
) -> None:
    with pytest.raises(ValueError, match="nondefault"):
        validate_dspark_greedy_sampling(
            temperature=0.0,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            logprobs=False,
            top_logprobs=None,
            repetition_penalty=None,
            repetition_context_size=None,
            presence_penalty=None,
            frequency_penalty=None,
        )


def test_context_capacity_uses_exact_prompt_prefix_and_output_limit() -> None:
    assert (
        dspark_context_capacity_hint(
            prompt_tokens=1_048_500,
            max_tokens=77,
            verify_width=8,
        )
        == 1_048_576
    )
    with pytest.raises(ValueError, match="context limit"):
        dspark_context_capacity_hint(
            prompt_tokens=1_048_500,
            max_tokens=78,
            verify_width=8,
        )


@dataclass
class _FakeRoundEngine:
    rounds: list[tuple[int, ...]]
    verify_width: int = 3
    ordinary_tokens: list[int] = field(default_factory=list)
    anchors: list[int] = field(default_factory=list)
    ordinary_anchors: list[int] = field(default_factory=list)

    def decode_round(self, anchor_token: int) -> DSparkRoundResult:
        self.anchors.append(anchor_token)
        tokens = self.rounds.pop(0)
        return DSparkRoundResult(
            emitted_tokens=tokens,
            telemetry=DSparkRoundTelemetry(
                round_index=len(self.anchors) - 1,
                rank=0,
                draft_ms=1.0,
                target_verify_ms=2.0,
                target_commit_ms=3.0,
                draft_commit_ms=4.0,
                collective_ms=5.0,
                proposed=2,
                accepted=1,
                emitted=len(tokens),
                fallback=False,
                error=None,
            ),
        )

    def decode_ordinary_tail(self, anchor_token: int) -> DSparkRoundResult:
        self.ordinary_anchors.append(anchor_token)
        token = self.ordinary_tokens.pop(0)
        return DSparkRoundResult(
            emitted_tokens=(token,),
            telemetry=DSparkRoundTelemetry(
                round_index=len(self.anchors) + len(self.ordinary_anchors) - 1,
                rank=0,
                draft_ms=0.0,
                target_verify_ms=0.0,
                target_commit_ms=0.0,
                draft_commit_ms=0.0,
                collective_ms=1.0,
                proposed=0,
                accepted=0,
                emitted=1,
                fallback=False,
                error=None,
            ),
        )


def test_decode_token_stream_stops_at_earliest_eos_inside_round() -> None:
    engine = _FakeRoundEngine([(11, 12, 13), (14,)])
    rounds: list[DSparkRoundTelemetry] = []
    accepted: list[bool] = []

    decoded = list(
        dspark_decode_tokens(
            engine,
            anchor_token=10,
            max_tokens=10,
            eos_token_ids=(12,),
            round_observer=rounds.append,
            token_observer=accepted.append,
        )
    )

    assert [(item.token, item.from_draft, item.finish_reason) for item in decoded] == [
        (11, True, None),
        (12, False, "stop"),
    ]
    assert engine.anchors == [10]
    assert len(rounds) == 1
    assert accepted == [True, False]


@pytest.mark.parametrize(
    ("max_tokens", "ordinary_tokens", "expected_tokens", "draft_flags"),
    [
        (1, [91], [91], [False]),
        (2, [91, 92], [91, 92], [False, False]),
        (3, [], [11, 12, 13], [True, False, False]),
        (4, [14], [11, 12, 13, 14], [True, False, False, False]),
    ],
)
def test_decode_token_stream_uses_target_only_for_short_length_tail(
    max_tokens: int,
    ordinary_tokens: list[int],
    expected_tokens: list[int],
    draft_flags: list[bool],
) -> None:
    engine = _FakeRoundEngine(
        [(11, 12, 13)],
        verify_width=3,
        ordinary_tokens=ordinary_tokens,
    )
    accepted: list[bool] = []

    decoded = list(
        dspark_decode_tokens(
            engine,
            anchor_token=10,
            max_tokens=max_tokens,
            eos_token_ids=(),
            token_observer=accepted.append,
        )
    )

    assert [item.token for item in decoded] == expected_tokens
    assert [item.from_draft for item in decoded] == draft_flags
    assert [item.finish_reason for item in decoded] == [
        *([None] * (max_tokens - 1)),
        "length",
    ]
    assert accepted == draft_flags
    expected_speculative_calls = 1 if max_tokens >= engine.verify_width else 0
    assert len(engine.anchors) == expected_speculative_calls
    assert len(engine.ordinary_anchors) == max_tokens - (
        engine.verify_width if expected_speculative_calls else 0
    )


@dataclass
class _FakeTokenLogits:
    shape: tuple[int, int]
    masked: list[tuple[object, float]] = field(default_factory=list)

    def __setitem__(self, key: object, value: float) -> None:
        self.masked.append((key, value))


@dataclass
class _FakeTargetLogits:
    token_logits: _FakeTokenLogits

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, *self.token_logits.shape)

    def __getitem__(self, key: int) -> _FakeTokenLogits:
        assert key == 0
        return self.token_logits


@dataclass(frozen=True)
class _FakeGreedyTokens:
    values: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.values),)

    def astype(self, _dtype: object) -> _FakeGreedyTokens:
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


def test_greedy_target_posterior_masks_benchmark_eos_and_materializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_logits = _FakeTokenLogits((3, 5))
    logits = _FakeTargetLogits(token_logits)
    evaluated: list[object] = []

    def argmax(values: object, *, axis: int) -> _FakeGreedyTokens:
        assert values is token_logits
        assert axis == -1
        return _FakeGreedyTokens((2, 3, 4))

    monkeypatch.setattr(dspark_module.mx, "argmax", argmax, raising=False)
    monkeypatch.setattr(dspark_module.mx, "int32", object(), raising=False)

    tokens = dspark_module.greedy_dspark_posterior_tokens(
        logits,
        expected_width=3,
        banned_token_ids=(1,),
        evaluate=lambda *values: evaluated.extend(values),
    )

    assert tokens == (2, 3, 4)
    assert token_logits.masked == [((Ellipsis, 1), float("-inf"))]
    assert len(evaluated) == 1


@dataclass(frozen=True)
class _FakePromptArray:
    values: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.values)

    def tolist(self) -> list[int]:
        return list(self.values)

    def __getitem__(self, key: slice | None) -> _FakePromptArray | _FakePromptBatch:
        if key is None:
            return _FakePromptBatch(self.values)
        return _FakePromptArray(self.values[key])


@dataclass(frozen=True)
class _FakePromptBatch:
    values: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))


@dataclass(frozen=True)
class _FakePendingForward:
    forward: object

    def materialize(self) -> object:
        return self.forward


@dataclass(frozen=True)
class _TrackingPendingForward:
    forward: object
    materializations: list[None]

    def materialize(self) -> object:
        self.materializations.append(None)
        return self.forward


@dataclass
class _FakeTargetCache:
    offset: int = 0
    state: object = field(default_factory=object)


@dataclass
class _FakeKDATargetCache:
    cache: list[object | None] = field(default_factory=lambda: [None, None])
    speculative_width: int = 0
    speculative_ready: bool = False

    @property
    def state(self) -> object:
        return self.cache


@dataclass
class _FakeRuntimeTarget(_FakeHookTarget):
    layers: tuple[object, ...] = field(
        default_factory=lambda: (object(), object())
    )


def test_prompt_prefix_is_chunk_seeded_into_fresh_target_and_draft_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    target_model = _FakeRuntimeTarget()
    target_cache = [_FakeTargetCache(), _FakeKDATargetCache()]
    proposer = _FakeMlxProposer(3, calls, [[11, 12]])
    context_cache = [_FakeContextCache(), _FakeContextCache()]
    draft = MlxDSparkRequestDraft(
        proposer=proposer,
        context_cache=context_cache,
        verify_width=3,
        evaluate=lambda *_values: None,
    )
    loaded = LoadedMlxDSpark(
        config=_config(tmp_path, 3),
        target_model=target_model,
        drafter=object(),
        proposer=proposer,
        evaluate=lambda *_values: None,
    )
    agreement = _FakeAgreement([])
    runtime = KimiK3DSparkRequestRuntime(
        loaded=loaded,
        target_model=target_model,
        target_cache=target_cache,
        draft=draft,
        collective=agreement,
        evaluate=lambda *_values: None,
    )
    seeded_chunks: list[tuple[int, ...]] = []

    def forward(
        chunk: _FakePromptBatch,
        *,
        initial_offset: int,
        speculative_width: int | None = None,
    ) -> _FakePendingForward:
        assert initial_offset == target_cache[0].offset
        assert speculative_width is None
        seeded_chunks.append(chunk.values)
        target_cache[0].offset += len(chunk.values)
        target_cache[1].cache = [object(), object()]
        return _FakePendingForward(
            SimpleNamespace(
                aux_hidden_states=tuple(
                    _FakeHidden((1, len(chunk.values), 7168), f"tap-{index}")
                    for index in range(5)
                )
            )
        )

    monkeypatch.setattr(runtime, "_build_forward_with_taps", forward)
    monkeypatch.setattr(dspark_module.mx, "clear_cache", lambda: None, raising=False)
    progress: list[tuple[int, int]] = []
    distributed_progress: list[None] = []
    full_prompt = _FakePromptArray((1, 2, 3, 4, 5, 6))

    prompt_tps, prompt_tokens = runtime.seed_prompt(
        cast(mx.array, cast(object, full_prompt[:-1])),
        prefill_step_size=2,
        progress_callback=lambda done, total: progress.append((done, total)),
        distributed_progress_callback=lambda: distributed_progress.append(None),
    )

    assert prompt_tps > 0
    assert prompt_tokens == 5
    assert seeded_chunks == [(1, 2), (3, 4), (5,)]
    assert target_cache[0].offset == 5
    assert [entry.length for entry in context_cache] == [5, 5]
    assert progress == [(0, 5), (2, 5), (4, 5), (5, 5)]
    assert len(distributed_progress) == 3


def test_prompt_content_disagreement_prevents_target_graph_build(
    tmp_path: Path,
) -> None:
    target_model = _FakeRuntimeTarget()
    target_cache = [_FakeTargetCache(), _FakeKDATargetCache()]
    proposer = _FakeMlxProposer(3, [], [[11, 12]])
    draft = MlxDSparkRequestDraft(
        proposer=proposer,
        context_cache=[_FakeContextCache(), _FakeContextCache()],
        verify_width=3,
        evaluate=lambda *_values: None,
    )
    runtime = KimiK3DSparkRequestRuntime(
        loaded=LoadedMlxDSpark(
            config=_config(tmp_path, 3),
            target_model=target_model,
            drafter=object(),
            proposer=proposer,
            evaluate=lambda *_values: None,
        ),
        target_model=target_model,
        target_cache=target_cache,
        draft=draft,
        # Operation fingerprint and length agree; a prompt digest word does not.
        collective=_FakeAgreement([], agreed_tokens=[0, 2, None]),
        evaluate=lambda *_values: None,
    )

    with pytest.raises(DSparkDistributedStateError, match="prompt/tokenizer"):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=None,
        )

    assert target_cache[0].offset == 0
    assert target_cache[1].cache == [None, None]


def test_peer_prompt_graph_build_failure_never_materializes_lazy_target_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_model = _FakeRuntimeTarget()
    target_cache = [_FakeTargetCache(), _FakeKDATargetCache()]
    proposer = _FakeMlxProposer(3, [], [[11, 12]])
    draft = MlxDSparkRequestDraft(
        proposer=proposer,
        context_cache=[_FakeContextCache(), _FakeContextCache()],
        verify_width=3,
        evaluate=lambda *_values: None,
    )
    runtime = KimiK3DSparkRequestRuntime(
        loaded=LoadedMlxDSpark(
            config=_config(tmp_path, 3),
            target_model=target_model,
            drafter=object(),
            proposer=proposer,
            evaluate=lambda *_values: None,
        ),
        target_model=target_model,
        target_cache=target_cache,
        draft=draft,
        collective=_FakeAgreement([], stage_outcomes=[True, True, None]),
        evaluate=lambda *_values: None,
    )
    materializations: list[None] = []

    def build(
        chunk: _FakePromptBatch,
        *,
        initial_offset: int,
        speculative_width: int | None = None,
    ) -> _TrackingPendingForward:
        assert initial_offset == 0
        assert speculative_width is None
        target_cache[0].offset += len(chunk.values)
        target_cache[1].cache = [object(), object()]
        return _TrackingPendingForward(
            SimpleNamespace(
                aux_hidden_states=tuple(
                    _FakeHidden((1, len(chunk.values), 7168), f"tap-{index}")
                    for index in range(5)
                )
            ),
            materializations,
        )

    monkeypatch.setattr(runtime, "_build_forward_with_taps", build)

    with pytest.raises(DSparkDistributedStateError, match="graph build"):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=None,
        )

    assert materializations == []


@dataclass(frozen=True)
class _FaithfulVerifyInput:
    values: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))


@dataclass
class _FaithfulTransaction:
    width: int
    initial_offset: int
    initial_kda: list[object]
    active: bool = True


@dataclass(frozen=True)
class _FakeCompactToken:
    value: int
    shape: tuple[int, ...] = (1,)

    def item(self) -> int:
        return self.value


@dataclass
class _FaithfulReplayTarget:
    posterior_tokens: tuple[int, ...]
    layers: tuple[object, ...] = field(
        default_factory=lambda: (object(), object())
    )
    events: list[tuple[str, int]] = field(default_factory=list)
    ordinary_token: int = 77
    compact_verifier_banned: list[tuple[int, ...]] = field(default_factory=list)

    def _ordinary_cache_step(self, cache: object, event: str) -> None:
        entries = cast(list[object], cache)
        mla = cast(_FakeTargetCache, entries[0])
        kda = cast(_FakeKDATargetCache, entries[1])
        assert kda.speculative_width == 0
        assert not kda.speculative_ready
        mla.offset += 1
        kda.cache = [f"{event}-conv", f"{event}-ssm"]
        self.events.append((event, 1))

    def __call__(self, inputs: _FaithfulVerifyInput, *, cache: object) -> object:
        self._ordinary_cache_step(cache, "full")
        return _FakeTargetLogits(_FakeTokenLogits((inputs.shape[1], 128)))

    def supports_vocab_parallel_greedy(self) -> bool:
        return True

    def vocab_parallel_greedy(
        self,
        inputs: _FaithfulVerifyInput,
        cache: object,
    ) -> _FakeCompactToken:
        self._ordinary_cache_step(cache, "compact")
        assert inputs.shape == (1, 1)
        return _FakeCompactToken(self.ordinary_token)

    def begin_speculative_cache(
        self,
        cache: object,
        width: int,
    ) -> _FaithfulTransaction:
        entries = cast(list[object], cache)
        mla = cast(_FakeTargetCache, entries[0])
        kda = cast(_FakeKDATargetCache, entries[1])
        assert kda.speculative_width == 0
        assert not kda.speculative_ready
        assert all(value is not None for value in kda.cache)
        kda.speculative_width = width
        self.events.append(("begin", width))
        return _FaithfulTransaction(width, mla.offset, cast(list[object], kda.cache[:]))

    def forward_with_aux_hidden_states(
        self,
        inputs: _FaithfulVerifyInput,
        cache: object,
        layer_ids: tuple[int, ...],
    ) -> object:
        entries = cast(list[object], cache)
        mla = cast(_FakeTargetCache, entries[0])
        kda = cast(_FakeKDATargetCache, entries[1])
        width = inputs.shape[1]
        assert layer_ids == (7, 23, 51, 67, 83)
        assert kda.speculative_width == width
        assert not kda.speculative_ready
        mla.offset += width
        kda.cache = [f"wide-{width}-conv", f"wide-{width}-ssm"]
        kda.speculative_ready = True
        self.events.append(("forward", width))
        return SimpleNamespace(
            logits=_FakeTargetLogits(_FakeTokenLogits((width, 128))),
            aux_hidden_states=tuple(
                _FakeHidden((1, width, 7168), f"tap-{index}")
                for index in range(5)
            ),
        )

    def forward_with_aux_hidden_states_greedy(
        self,
        inputs: _FaithfulVerifyInput,
        cache: object,
        layer_ids: tuple[int, ...],
        banned_token_ids: tuple[int, ...] = (),
    ) -> object:
        entries = cast(list[object], cache)
        mla = cast(_FakeTargetCache, entries[0])
        kda = cast(_FakeKDATargetCache, entries[1])
        width = inputs.shape[1]
        assert layer_ids == (7, 23, 51, 67, 83)
        assert kda.speculative_width == width
        assert not kda.speculative_ready
        mla.offset += width
        kda.cache = [f"compact-wide-{width}-conv", f"compact-wide-{width}-ssm"]
        kda.speculative_ready = True
        self.compact_verifier_banned.append(banned_token_ids)
        self.events.append(("compact_verify", width))
        return SimpleNamespace(
            tokens=_FakeTokenArray([list(self.posterior_tokens)]),
            aux_hidden_states=tuple(
                _FakeHidden((1, width, 7168), f"tap-{index}")
                for index in range(5)
            ),
        )

    def resolve_speculative_cache(
        self,
        transaction: object,
        consumed: int,
    ) -> None:
        assert isinstance(transaction, _FaithfulTransaction)
        assert transaction.active
        # The real hook commits the selected KDA checkpoint and rewinds the MLA
        # logical offset from the full verification width to the consumed width.
        transaction_cache = self._active_cache
        mla = transaction_cache[0]
        kda = transaction_cache[1]
        mla.offset = transaction.initial_offset + consumed
        kda.cache = [f"commit-{consumed}-conv", f"commit-{consumed}-ssm"]
        kda.speculative_width = 0
        kda.speculative_ready = False
        transaction.active = False
        self.events.append(("resolve", consumed))

    def cancel_speculative_cache(self, transaction: object) -> None:
        assert isinstance(transaction, _FaithfulTransaction)
        if not transaction.active:
            return
        mla = self._active_cache[0]
        kda = self._active_cache[1]
        mla.offset = transaction.initial_offset
        kda.cache = transaction.initial_kda[:]
        kda.speculative_width = 0
        kda.speculative_ready = False
        transaction.active = False
        self.events.append(("cancel", 0))

    @property
    def _active_cache(self) -> list[_FakeTargetCache | _FakeKDATargetCache]:
        return self.__dict__["active_cache"]

    @_active_cache.setter
    def _active_cache(
        self,
        value: list[_FakeTargetCache | _FakeKDATargetCache],
    ) -> None:
        self.__dict__["active_cache"] = value


def _faithful_request_runtime(
    tmp_path: Path,
    posterior: tuple[int, ...],
    *,
    compact_greedy: bool = False,
    banned_token_ids: tuple[int, ...] = (),
) -> tuple[
    KimiK3DSparkRequestRuntime,
    _FaithfulReplayTarget,
    list[_FakeTargetCache | _FakeKDATargetCache],
    list[_FakeContextCache],
]:
    calls: list[tuple[object, ...]] = []
    target = _FaithfulReplayTarget(posterior)
    target_cache: list[_FakeTargetCache | _FakeKDATargetCache] = [
        _FakeTargetCache(),
        _FakeKDATargetCache(),
    ]
    target._active_cache = target_cache
    proposer = _FakeMlxProposer(3, calls, [[11, 12]])
    context_cache = [_FakeContextCache(), _FakeContextCache()]
    draft = MlxDSparkRequestDraft(
        proposer=proposer,
        context_cache=context_cache,
        verify_width=3,
        evaluate=lambda *_values: None,
    )
    loaded = LoadedMlxDSpark(
        config=_config(tmp_path, 3),
        target_model=target,
        drafter=object(),
        proposer=proposer,
        evaluate=lambda *_values: None,
    )
    runtime = KimiK3DSparkRequestRuntime(
        loaded=loaded,
        target_model=target,
        target_cache=target_cache,
        draft=draft,
        collective=_FakeAgreement([]),
        banned_token_ids=banned_token_ids,
        compact_greedy=compact_greedy,
        evaluate=lambda *_values: None,
    )
    target_cache[0].offset = 5
    target_cache[1].cache = ["initial-conv", "initial-ssm"]
    for entry in context_cache:
        entry.length = 5
        entry.keys = object()
        entry.values = object()
    return runtime, target, target_cache, context_cache


def _patch_faithful_verification_mlx(
    monkeypatch: pytest.MonkeyPatch,
    target: _FaithfulReplayTarget,
) -> None:
    monkeypatch.setattr(
        dspark_module.mx,
        "array",
        lambda rows, **_kwargs: _FaithfulVerifyInput(tuple(rows[0])),
        raising=False,
    )
    monkeypatch.setattr(dspark_module.mx, "int32", object(), raising=False)
    monkeypatch.setattr(
        dspark_module,
        "_build_greedy_dspark_posterior_tokens",
        lambda _logits, **_kwargs: _FakeGreedyTokens(target.posterior_tokens),
    )


@pytest.mark.parametrize(
    ("posterior", "expected_tokens", "consumed"),
    [
        ((11, 12, 13), (11, 12, 13), 3),
        ((99, 100, 101), (99,), 1),
    ],
)
def test_request_runtime_verification_allows_only_exact_staged_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    posterior: tuple[int, ...],
    expected_tokens: tuple[int, ...],
    consumed: int,
) -> None:
    runtime, target, target_cache, context_cache = _faithful_request_runtime(
        tmp_path,
        posterior,
    )
    monkeypatch.setenv(MLX_DSPARK_PROPOSER_ENV, "1")
    _patch_faithful_verification_mlx(monkeypatch, target)

    result = runtime.make_round_engine().decode_round(10)

    assert result.emitted_tokens == expected_tokens
    assert target.events == [("begin", 3), ("forward", 3), ("resolve", consumed)]
    assert target_cache[0].offset == 5 + consumed
    assert target_cache[1].speculative_width == 0
    assert target_cache[1].speculative_ready is False
    assert [entry.length for entry in context_cache] == [5 + consumed, 5 + consumed]


def test_request_runtime_uses_compact_verifier_with_exact_banned_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, target, target_cache, _context_cache = _faithful_request_runtime(
        tmp_path,
        (11, 12, 13),
        compact_greedy=True,
        banned_token_ids=(2, 7),
    )
    _patch_faithful_verification_mlx(monkeypatch, target)
    monkeypatch.setenv(MLX_DSPARK_PROPOSER_ENV, "1")

    result = runtime.make_round_engine().decode_round(10)

    assert result.emitted_tokens == (11, 12, 13)
    assert target.events == [
        ("begin", 3),
        ("compact_verify", 3),
        ("resolve", 3),
    ]
    assert target.compact_verifier_banned == [(2, 7)]
    assert target_cache[0].offset == 8


def test_requested_compact_verifier_missing_capability_never_opens_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, target, _target_cache, _context_cache = _faithful_request_runtime(
        tmp_path,
        (11, 12, 13),
        compact_greedy=True,
    )
    _patch_faithful_verification_mlx(monkeypatch, target)
    monkeypatch.setenv(MLX_DSPARK_PROPOSER_ENV, "1")
    monkeypatch.setattr(
        target,
        "forward_with_aux_hidden_states_greedy",
        None,
    )

    result = runtime.make_round_engine().decode_round(10)

    assert result.emitted_tokens == (77,)
    assert ("begin", 3) not in target.events
    assert ("forward", 3) not in target.events
    assert target.events == [("compact", 1)]


def test_request_runtime_cancel_restores_clean_target_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, target, target_cache, _context_cache = _faithful_request_runtime(
        tmp_path,
        (11, 12, 13),
    )
    _patch_faithful_verification_mlx(monkeypatch, target)
    target_adapter = runtime.make_round_engine().target

    prepared = target_adapter.prepare_verification((10, 11, 12))
    transaction = prepared.build().materialize()
    assert target_cache[0].offset == 8
    assert target_cache[1].speculative_width == 3
    assert target_cache[1].speculative_ready is True
    transaction.cancel()

    assert target.events == [("begin", 3), ("forward", 3), ("cancel", 0)]
    assert target_cache[0].offset == 5
    assert target_cache[1].cache == ["initial-conv", "initial-ssm"]
    assert target_cache[1].speculative_width == 0
    assert target_cache[1].speculative_ready is False


@pytest.mark.parametrize(
    ("compact_greedy", "banned_token_ids", "expected_path"),
    [
        (False, (), "full"),
        (True, (), "compact"),
        (True, (2,), "full"),
    ],
)
def test_target_only_fallback_compact_greedy_matches_full_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compact_greedy: bool,
    banned_token_ids: tuple[int, ...],
    expected_path: str,
) -> None:
    runtime, target, target_cache, _context_cache = _faithful_request_runtime(
        tmp_path,
        (11, 12, 13),
        compact_greedy=compact_greedy,
        banned_token_ids=banned_token_ids,
    )
    _patch_faithful_verification_mlx(monkeypatch, target)
    monkeypatch.setattr(
        dspark_module,
        "_build_greedy_dspark_posterior_tokens",
        lambda _logits, **_kwargs: _FakeGreedyTokens((target.ordinary_token,)),
    )

    result = runtime.make_round_engine().decode_ordinary_tail(10)

    assert result.emitted_tokens == (77,)
    assert target.events == [(expected_path, 1)]
    assert target_cache[0].offset == 6
    assert target_cache[1].speculative_width == 0
    assert target_cache[1].speculative_ready is False
