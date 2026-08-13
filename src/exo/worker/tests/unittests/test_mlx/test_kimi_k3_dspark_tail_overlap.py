"""Engine-level tests for the DSpark tail-overlap prelaunch path.

Tail overlap commits the draft context and submits the next round's proposal
graph between the acceptance agreement and the CPU-bound target commit, so the
GPU executes the next draft chain while python resolves speculative target
checkpoints.  These tests validate the rank-agreed eligibility gate, the
relocated agreements, prelaunch reuse in the following round, and every
failure path's cleanup semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exo.worker.engines.mlx.generator.kimi_k3_dspark import (
    DSparkCollectivePoisonError,
    DSparkDistributedStateError,
    KimiK3DSparkConfig,
    KimiK3DSparkRoundEngine,
    TargetPosterior,
    dspark_decode_tokens,
)
from exo.worker.tests.unittests.test_mlx.test_kimi_k3_dspark import (
    _FakeAgreement,  # pyright: ignore[reportPrivateUsage]
    _FakeDraft,  # pyright: ignore[reportPrivateUsage]
    _FakeTarget,  # pyright: ignore[reportPrivateUsage]
)


@dataclass
class _OverlapDraftRound:
    """Draft round exposing the split-commit surface the prelaunch requires."""

    proposal_tokens: Sequence[int]
    events: list[str]
    owner: "_OverlapDraft"
    confidence_logits: Sequence[float] | None = None
    commits: list[tuple[int, int, tuple[int, ...]]] = field(default_factory=list)
    finalizes: list[bool] = field(default_factory=list)
    cancelled: bool = False

    def commit_build(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> tuple[int, tuple[object, ...]]:
        self.events.append("draft_commit_build")
        if self.owner.fail_commit_build:
            raise RuntimeError("injected draft commit-build failure")
        self.commits.append(
            (
                accepted_draft_tokens,
                next_anchor_token,
                tuple(target_posterior.tokens),
            )
        )
        return 6, ("ctx-k", "ctx-v")

    def commit_finalize(self, *, evaluate: bool) -> None:
        self.events.append(
            "draft_commit_finalize_eval" if evaluate else "draft_commit_finalize_lazy"
        )
        if self.owner.fail_finalize:
            raise RuntimeError("injected draft finalize failure")
        self.finalizes.append(evaluate)

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
                tuple(target_posterior.tokens),
            )
        )

    def cancel(self) -> None:
        self.events.append("draft_cancel")
        self.cancelled = True


@dataclass
class _OverlapPrepared:
    """Prepared draft graph that supports asynchronous submission."""

    owner: "_OverlapDraft"
    proposals: tuple[int, ...]
    submitted: bool = False
    cancelled: bool = False

    def materialize(self) -> _OverlapDraftRound:
        self.owner.events.append("draft_materialize")
        round_state = _OverlapDraftRound(self.proposals, self.owner.events, self.owner)
        self.owner.rounds.append(round_state)
        return round_state

    def submit(self, context_arrays: Sequence[object]) -> None:
        self.owner.events.append("draft_submit")
        if self.owner.fail_submit:
            raise RuntimeError("injected draft submit failure")
        self.owner.submitted_context.append(tuple(context_arrays))
        self.submitted = True

    def cancel(self) -> None:
        self.owner.events.append("draft_graph_cancel")
        self.cancelled = True


@dataclass
class _OverlapDraft:
    """Replicated draft whose successive graphs consume scripted proposals."""

    proposal_rounds: list[tuple[int, ...]]
    verify_width: int
    events: list[str]
    placement: str = "replicated"
    fail_commit_build: bool = False
    fail_submit: bool = False
    fail_finalize: bool = False
    fail_build_at_call: int | None = None
    build_calls: int = field(default=0, init=False)
    rounds: list[_OverlapDraftRound] = field(default_factory=list)
    prepared: list[_OverlapPrepared] = field(default_factory=list)
    submitted_context: list[tuple[object, ...]] = field(default_factory=list)

    def preflight_round(self, anchor_token: int, num_proposals: int) -> int:
        self.events.append("draft_preflight")
        assert num_proposals == self.verify_width - 1
        return 5

    def prepare_round(
        self,
        anchor_token: int,
        num_proposals: int,
    ) -> _OverlapPrepared:
        self.build_calls += 1
        self.events.append("draft_build")
        assert num_proposals == self.verify_width - 1
        if self.build_calls == self.fail_build_at_call:
            raise RuntimeError("injected draft graph-build failure")
        prepared = _OverlapPrepared(self, tuple(self.proposal_rounds.pop(0)))
        self.prepared.append(prepared)
        return prepared


def _overlap_engine(
    tmp_path: Path,
    *,
    proposal_rounds: Sequence[Sequence[int]] = ((11, 12), (11, 12), (11, 12)),
    posterior: Sequence[int] = (11, 12, 13),
    tail_overlap: bool = True,
    terminal_token_ids: tuple[int, ...] = (),
    target_fail_commit: bool = False,
    fail_commit_build: bool = False,
    fail_submit: bool = False,
    fail_finalize: bool = False,
    fail_build_at_call: int | None = None,
) -> tuple[
    KimiK3DSparkRoundEngine,
    _OverlapDraft,
    _FakeTarget,
    _FakeAgreement,
    list[str],
]:
    events: list[str] = []
    width = len(posterior)
    assert width in (3, 4, 8)
    draft = _OverlapDraft(
        [tuple(proposals) for proposals in proposal_rounds],
        width,
        events,
        fail_commit_build=fail_commit_build,
        fail_submit=fail_submit,
        fail_finalize=fail_finalize,
        fail_build_at_call=fail_build_at_call,
    )
    target = _FakeTarget(tuple(posterior), events, fail_commit=target_fail_commit)
    collective = _FakeAgreement(events)
    engine = KimiK3DSparkRoundEngine(
        config=KimiK3DSparkConfig(
            checkpoint_path=tmp_path,
            verify_width=width,
            round_telemetry=False,
            tail_overlap=tail_overlap,
        ),
        draft=draft,
        target=target,
        collective=collective,
        terminal_token_ids=terminal_token_ids,
    )
    return engine, draft, target, collective, events


def test_tail_overlap_prelaunches_next_draft_and_reuses_it(tmp_path: Path) -> None:
    engine, draft, target, _collective, events = _overlap_engine(tmp_path)

    first = engine.decode_round(10, remaining=100)

    assert first.emitted_tokens == (11, 12, 13)
    assert first.telemetry.error is None
    assert first.telemetry.prelaunch_submitted is True
    assert first.telemetry.prelaunch_used is False
    assert first.telemetry.prelaunch_ms >= 0.0
    assert target.rounds[0].commits == [3]
    assert draft.rounds[0].commits == [(2, 13, (11, 12, 13))]
    assert draft.rounds[0].finalizes == [False]
    assert draft.submitted_context == [("ctx-k", "ctx-v")]
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
        "draft_commit_build",
        "agree_stage_1",
        "draft_build",
        "agree_stage_1",
        "draft_submit",
        "draft_commit_finalize_lazy",
        "target_commit",
        "agree_stage_1",
        "agree_stage_1",
    ]

    events.clear()
    second = engine.decode_round(13, remaining=5)

    assert second.emitted_tokens == (11, 12, 13)
    assert second.telemetry.error is None
    assert second.telemetry.prelaunch_used is True
    assert second.telemetry.prelaunch_submitted is False
    assert events == [
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


def test_tail_overlap_skips_prelaunch_when_budget_exhausted(tmp_path: Path) -> None:
    engine, _draft, _target, _collective, events = _overlap_engine(tmp_path)

    result = engine.decode_round(10, remaining=3)

    assert result.emitted_tokens == (11, 12, 13)
    assert result.telemetry.prelaunch_submitted is False
    assert "draft_commit_build" not in events
    assert "draft_commit" in events
    assert engine._prelaunched is None  # pyright: ignore[reportPrivateUsage]


def test_tail_overlap_skips_prelaunch_on_terminal_token(tmp_path: Path) -> None:
    engine, _draft, _target, _collective, events = _overlap_engine(
        tmp_path,
        terminal_token_ids=(13,),
    )

    result = engine.decode_round(10, remaining=100)

    assert result.emitted_tokens == (11, 12, 13)
    assert result.telemetry.prelaunch_submitted is False
    assert "draft_commit_build" not in events
    assert "draft_commit" in events


def test_tail_overlap_requires_split_commit_capable_draft(tmp_path: Path) -> None:
    events: list[str] = []
    draft = _FakeDraft((11, 12), 3, events)
    target = _FakeTarget((11, 12, 13), events)
    engine = KimiK3DSparkRoundEngine(
        config=KimiK3DSparkConfig(
            checkpoint_path=tmp_path,
            verify_width=3,
            round_telemetry=False,
            tail_overlap=True,
        ),
        draft=draft,
        target=target,
        collective=_FakeAgreement(events),
    )

    result = engine.decode_round(10, remaining=100)

    assert result.emitted_tokens == (11, 12, 13)
    assert result.telemetry.error is None
    assert result.telemetry.prelaunch_submitted is False
    assert "draft_commit" in events


def test_tail_overlap_disabled_by_default_config(tmp_path: Path) -> None:
    engine, _draft, _target, _collective, events = _overlap_engine(
        tmp_path,
        tail_overlap=False,
    )

    result = engine.decode_round(10, remaining=100)

    assert result.telemetry.prelaunch_submitted is False
    assert "draft_commit_build" not in events
    assert "draft_commit" in events


def test_tail_overlap_composes_with_width_four(tmp_path: Path) -> None:
    engine, draft, target, _collective, events = _overlap_engine(
        tmp_path,
        proposal_rounds=((11, 12, 13), (11, 12, 13), (11, 12, 13)),
        posterior=(11, 12, 13, 14),
    )

    first = engine.decode_round(10, remaining=100)
    second = engine.decode_round(14, remaining=96)

    assert engine.verify_width == 4
    assert first.emitted_tokens == (11, 12, 13, 14)
    assert first.telemetry.prelaunch_submitted is True
    assert second.telemetry.prelaunch_used is True
    assert target.rounds[0].commits == [4]
    assert draft.rounds[0].commits == [(3, 14, (11, 12, 13, 14))]
    assert "draft_commit_finalize_lazy" in events


def test_tail_overlap_commit_build_failure_disables_dspark(tmp_path: Path) -> None:
    engine, draft, target, _collective, _events = _overlap_engine(
        tmp_path,
        fail_commit_build=True,
    )

    result = engine.decode_round(10, remaining=100)

    # The target commit stays authoritative: the round's tokens are kept.
    assert result.emitted_tokens == (11, 12, 13)
    assert target.rounds[0].commits == [3]
    assert result.telemetry.error is not None
    assert "draft commit" in result.telemetry.error
    assert result.telemetry.prelaunch_submitted is False
    assert draft.build_calls == 1
    assert draft.rounds[0].finalizes == []
    assert engine._prelaunched is None  # pyright: ignore[reportPrivateUsage]

    ordinary = engine.decode_round(13, remaining=97)

    assert ordinary.telemetry.fallback is True
    assert ordinary.emitted_tokens == (999,)


def test_tail_overlap_graph_build_failure_recovers(tmp_path: Path) -> None:
    engine, draft, _target, _collective, events = _overlap_engine(
        tmp_path,
        fail_build_at_call=2,
    )

    first = engine.decode_round(10, remaining=100)

    assert first.emitted_tokens == (11, 12, 13)
    assert first.telemetry.error is None
    assert first.telemetry.prelaunch_submitted is False
    # The context append is still committed, synchronously this time.
    assert draft.rounds[0].finalizes == [True]
    assert engine._prelaunched is None  # pyright: ignore[reportPrivateUsage]

    events.clear()
    second = engine.decode_round(13, remaining=5)

    assert second.emitted_tokens == (11, 12, 13)
    assert second.telemetry.error is None
    assert second.telemetry.prelaunch_used is False
    assert events[0] == "draft_preflight"


def test_tail_overlap_submit_failure_poison_stops_before_later_collective(
    tmp_path: Path,
) -> None:
    engine, draft, _target, _collective, _events = _overlap_engine(
        tmp_path,
        fail_submit=True,
    )

    with pytest.raises(
        DSparkCollectivePoisonError, match="worker/ring must be recycled"
    ):
        engine.decode_round(10, remaining=100)

    assert draft.prepared[1].cancelled is True
    assert draft.rounds[0].cancelled is True
    assert "target_commit" not in _events
    assert _events[-4:] == [
        "draft_submit",
        "draft_graph_cancel",
        "draft_cancel",
        "target_cancel",
    ]
    assert engine._prelaunched is None  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(DSparkCollectivePoisonError, match="earlier"):
        engine.decode_ordinary_tail(13)


def test_tail_overlap_finalize_failure_after_submit_is_poison_fail_stop(
    tmp_path: Path,
) -> None:
    engine, draft, _target, _collective, events = _overlap_engine(
        tmp_path,
        fail_finalize=True,
    )

    with pytest.raises(
        DSparkCollectivePoisonError, match="worker/ring must be recycled"
    ):
        engine.decode_round(10, remaining=100)

    assert draft.prepared[1].submitted is True
    assert draft.prepared[1].cancelled is True
    assert draft.rounds[0].cancelled is True
    assert "target_commit" not in events
    assert events[-5:] == [
        "draft_submit",
        "draft_commit_finalize_lazy",
        "draft_graph_cancel",
        "draft_cancel",
        "target_cancel",
    ]


def test_tail_overlap_discards_prelaunch_on_anchor_mismatch(tmp_path: Path) -> None:
    engine, draft, _target, _collective, events = _overlap_engine(tmp_path)
    engine.decode_round(10, remaining=100)
    assert engine._prelaunched is not None  # pyright: ignore[reportPrivateUsage]

    events.clear()
    result = engine.decode_round(14, remaining=5)

    assert draft.prepared[1].cancelled is True
    assert result.telemetry.prelaunch_used is False
    assert result.emitted_tokens == (11, 12, 13)
    assert events[0] == "draft_graph_cancel"
    assert "draft_preflight" in events


def test_tail_overlap_fatal_target_commit_cancels_prelaunched_draft(
    tmp_path: Path,
) -> None:
    engine, draft, _target, _collective, events = _overlap_engine(
        tmp_path,
        target_fail_commit=True,
    )

    with pytest.raises(DSparkDistributedStateError):
        engine.decode_round(10, remaining=100)

    assert draft.prepared[1].cancelled is True
    assert "draft_cancel" in events
    assert engine._prelaunched is None  # pyright: ignore[reportPrivateUsage]


def test_ordinary_tail_discards_pending_prelaunch(tmp_path: Path) -> None:
    engine, draft, _target, _collective, _events = _overlap_engine(tmp_path)
    engine.decode_round(10, remaining=100)
    assert engine._prelaunched is not None  # pyright: ignore[reportPrivateUsage]

    result = engine.decode_ordinary_tail(13)

    assert engine._prelaunched is None  # pyright: ignore[reportPrivateUsage]
    assert draft.prepared[1].cancelled is True
    assert result.emitted_tokens == (999,)
    # A planned max-token tail is an ordinary round, not an error fallback.
    assert result.telemetry.fallback is False
    assert result.telemetry.error is None


def test_decode_generator_close_discards_pending_prelaunch(tmp_path: Path) -> None:
    engine, draft, _target, _collective, _events = _overlap_engine(tmp_path)
    decoded = dspark_decode_tokens(
        engine,
        anchor_token=10,
        max_tokens=100,
        eos_token_ids=(),
    )

    assert next(decoded).token == 11
    assert engine._prelaunched is not None  # pyright: ignore[reportPrivateUsage]

    decoded.close()

    assert engine._prelaunched is None  # pyright: ignore[reportPrivateUsage]
    assert draft.prepared[1].cancelled is True
