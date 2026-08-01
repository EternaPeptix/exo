from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exo.worker.engines.mlx.generator.kimi_k3_dspark import (
    DSPARK_CHECKPOINT_ENV,
    DSPARK_CONSERVATIVE_VERIFY_WIDTH,
    DSPARK_ENABLE_ENV,
    DSPARK_MODEL_NATIVE_VERIFY_WIDTH,
    DSPARK_TELEMETRY_ENV,
    DSPARK_VERIFY_WIDTH_ENV,
    MLX_DSPARK_PROPOSER_ENV,
    MLX_REPLAYSSM_ENV,
    DSparkConfigurationError,
    DSparkFeatureUnavailableError,
    KimiK3DSparkConfig,
    KimiK3DSparkRoundEngine,
    MlxDSparkFeatures,
    ReplaySSMTargetAdapter,
    TargetPosterior,
    accepted_draft_prefix,
    has_replayssm_target_hooks,
    kimi_k3_dspark_config,
    load_replicated_mlx_dspark,
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
    assert config.target_hidden_state_indices == (8, 24, 52, 68, 84)
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

    def commit(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> None:
        self.events.append("draft_commit")
        self.commits.append(
            (
                accepted_draft_tokens,
                next_anchor_token,
                target_posterior.tokens,
            )
        )

    def cancel(self) -> None:
        self.events.append("draft_cancel")
        self.cancelled = True


@dataclass
class _FakeDraft:
    proposal_tokens: Sequence[int]
    verify_width: int
    events: list[str]
    placement: str = "replicated"
    rounds: list[_FakeDraftRound] = field(default_factory=list)

    def begin_round(self, anchor_token: int, num_proposals: int) -> _FakeDraftRound:
        self.events.append("draft")
        assert num_proposals == self.verify_width - 1
        round_state = _FakeDraftRound(self.proposal_tokens, self.events)
        self.rounds.append(round_state)
        return round_state


@dataclass
class _FakeTargetRound:
    posterior: TargetPosterior
    events: list[str]
    commits: list[int] = field(default_factory=list)
    cancelled: bool = False

    def commit(self, consumed_input_tokens: int) -> None:
        self.events.append("target_commit")
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

    def begin_verification(self, proposal_block: tuple[int, ...]) -> _FakeTargetRound:
        self.events.append("target_verify")
        round_state = _FakeTargetRound(
            TargetPosterior(tuple(self.posterior_tokens), ("hidden-taps",)),
            self.events,
        )
        self.rounds.append(round_state)
        return round_state

    def ordinary_decode(self, anchor_token: int) -> int:
        self.events.append("ordinary_decode")
        self.ordinary_anchors.append(anchor_token)
        return self.ordinary_token


@dataclass
class _FakeAgreement:
    events: list[str]
    reject_proposal: bool = False
    reject_acceptance: bool = False
    rank: int = 0
    size: int = 2

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
        "draft",
        "agree_proposal",
        "target_verify",
        "agree_acceptance",
        "target_commit",
        "draft_commit",
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
        "draft",
        "agree_proposal",
        "target_verify",
        "agree_acceptance",
        "target_cancel",
        "draft_cancel",
        "ordinary_decode",
    ]


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


def test_replayssm_adapter_feature_detects_and_commits_accepted_prefix() -> None:
    target = _FakeHookTarget()
    adapter = ReplaySSMTargetAdapter(
        target,
        target_cache=object(),
        verify=lambda block: TargetPosterior(tuple(range(20, 20 + len(block)))),
        ordinary_decode=lambda anchor: anchor + 1,
    )

    assert has_replayssm_target_hooks(target)
    transaction = adapter.begin_verification((10, 11, 12))
    transaction.commit(2)

    assert transaction.posterior.tokens == (20, 21, 22)
    assert target.events == [("begin", 3), ("resolve", 2)]
    assert adapter.ordinary_decode(30) == 31


def test_replayssm_adapter_cancels_when_target_verification_raises() -> None:
    target = _FakeHookTarget()

    def fail(_block: tuple[int, ...]) -> TargetPosterior:
        raise RuntimeError("verification failed")

    adapter = ReplaySSMTargetAdapter(
        target,
        target_cache=object(),
        verify=fail,
        ordinary_decode=lambda anchor: anchor + 1,
    )

    with pytest.raises(RuntimeError, match="verification failed"):
        adapter.begin_verification((10, 11, 12))
    assert target.events == [("begin", 3), ("cancel", 0)]


def test_replayssm_adapter_rejects_incomplete_target_api() -> None:
    with pytest.raises(DSparkFeatureUnavailableError, match="missing hidden-tap"):
        ReplaySSMTargetAdapter(
            object(),
            target_cache=object(),
            verify=lambda _block: TargetPosterior((1, 2, 3)),
            ordinary_decode=lambda anchor: anchor + 1,
        )


def test_replicated_loader_uses_local_path_hash_check_and_explicit_width(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, ...]] = []

    class _FakeProposer:
        def propose(self) -> None:
            return None

    proposer = _FakeProposer()

    def load(
        checkpoint_path: Path,
        target_model: object,
        *,
        verify_sha256: bool,
    ) -> object:
        calls.append(("load", checkpoint_path, target_model, verify_sha256))
        return "draft"

    def make_proposer(
        drafter: object,
        *,
        verify_width: int,
    ) -> object:
        calls.append(("proposer", drafter, verify_width))
        return proposer

    def append_context(*_args: object) -> None:
        return None

    features = MlxDSparkFeatures(
        load_kimi_k3_dspark=load,
        proposer_type=make_proposer,
        make_context_cache=lambda: "context-cache",
        append_target_context=append_context,
    )
    config = _config(tmp_path, 8)
    target_model = object()

    loaded = load_replicated_mlx_dspark(
        config,
        target_model,
        features=features,
    )

    assert loaded.placement == "replicated"
    assert loaded.context_cache == "context-cache"
    assert calls == [
        ("load", tmp_path, target_model, True),
        ("proposer", "draft", 8),
    ]


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
