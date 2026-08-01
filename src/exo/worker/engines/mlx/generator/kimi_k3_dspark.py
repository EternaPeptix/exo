"""Strict, opt-in EXO orchestration for Kimi K3 DSpark speculation.

The DSpark checkpoint is replicated on every target tensor-parallel rank.  A
round therefore drafts locally on every rank, agrees on the complete proposal
block, verifies the width-N block with the target, and agrees on the acceptance
boundary before either cache may commit.  The target cache adapter deliberately
uses MLX-LM's feature-detected speculative transaction hooks so accepted-prefix
commit can use ReplaySSM without binding EXO to MLX-LM implementation types.

This module does not download or discover a checkpoint.  Enabling it requires
an absolute local checkpoint directory whose config hash matches the pinned
RadixArk artifact.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast, final

import mlx.core as mx  # pyright: ignore[reportMissingModuleSource]

from exo.worker.runner.bootstrap import logger

DSPARK_ENABLE_ENV = "EXO_MLX_KIMI_K3_DSPARK_SPECULATIVE"
DSPARK_CHECKPOINT_ENV = "EXO_MLX_KIMI_K3_DSPARK_CHECKPOINT"
DSPARK_VERIFY_WIDTH_ENV = "EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH"
DSPARK_TELEMETRY_ENV = "EXO_MLX_KIMI_K3_DSPARK_ROUND_TELEMETRY"

MLX_DSPARK_PROPOSER_ENV = "MLX_LM_KIMI_K3_DSPARK_PROPOSER"
MLX_REPLAYSSM_ENV = "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE"

RADIXARK_KIMI_K3_DSPARK_MODEL = "RadixArk/Kimi-K3-DSpark"
RADIXARK_KIMI_K3_DSPARK_REVISION = "eb03982e58d4fb79bcfc099e902158f562e2e27b"
RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256 = (
    "6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f"
)
RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES = 4_498_585_858
RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256 = (
    "29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495"
)
RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS = (7, 23, 51, 67, 83)
RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE = 7

# SGLang's published K3 deployment maps checkpoint block_size directly to
# gamma, so seven proposals plus the current anchor are verified at once.
DSPARK_MODEL_NATIVE_GAMMA = RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE
DSPARK_MODEL_NATIVE_VERIFY_WIDTH = DSPARK_MODEL_NATIVE_GAMMA + 1

# MLX-LM's initial rollout uses a deliberately conservative two proposals.
DSPARK_CONSERVATIVE_GAMMA = 2
DSPARK_CONSERVATIVE_VERIFY_WIDTH = DSPARK_CONSERVATIVE_GAMMA + 1
DSPARK_ALLOWED_VERIFY_WIDTHS = (
    DSPARK_CONSERVATIVE_VERIFY_WIDTH,
    DSPARK_MODEL_NATIVE_VERIFY_WIDTH,
)

_EXO_COMPANION_ENVS = (
    DSPARK_CHECKPOINT_ENV,
    DSPARK_VERIFY_WIDTH_ENV,
    DSPARK_TELEMETRY_ENV,
)


class DSparkConfigurationError(ValueError):
    """An explicit DSpark deployment configuration is invalid."""


class DSparkFeatureUnavailableError(RuntimeError):
    """The installed MLX-LM build does not expose the required DSpark API."""


class DSparkDistributedStateError(RuntimeError):
    """A distributed state transition cannot be recovered safely."""


@dataclass(frozen=True)
class KimiK3DSparkConfig:
    """Validated local checkpoint and replicated placement contract."""

    checkpoint_path: Path
    verify_width: Literal[3, 8]
    round_telemetry: bool
    placement: Literal["replicated"] = "replicated"
    model_id: str = RADIXARK_KIMI_K3_DSPARK_MODEL
    revision: str = RADIXARK_KIMI_K3_DSPARK_REVISION
    config_sha256: str = RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256
    model_bytes: int = RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES
    model_sha256: str = RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256
    target_layer_ids: tuple[int, ...] = RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS

    @property
    def gamma(self) -> int:
        return self.verify_width - 1

    @property
    def target_hidden_state_indices(self) -> tuple[int, ...]:
        """Direct MLX-LM layer ids for post-layer target taps."""

        return self.target_layer_ids


def _strict_flag(name: str, raw: str) -> bool:
    if raw not in {"0", "1"}:
        raise DSparkConfigurationError(f"{name} must be 0 or 1")
    return raw == "1"


def _strict_verify_width(raw: str) -> Literal[3, 8]:
    if not raw.isascii() or not raw.isdecimal():
        raise DSparkConfigurationError(f"{DSPARK_VERIFY_WIDTH_ENV} must be 3 or 8")
    parsed = int(raw)
    if parsed not in DSPARK_ALLOWED_VERIFY_WIDTHS:
        raise DSparkConfigurationError(f"{DSPARK_VERIFY_WIDTH_ENV} must be 3 or 8")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as config_file:
        for chunk in iter(lambda: config_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_dspark_checkpoint(
    checkpoint_path: Path,
    *,
    expected_config_sha256: str = RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256,
    expected_model_bytes: int = RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES,
    sha256: Callable[[Path], str] = _sha256,
) -> None:
    """Validate the pinned config before any code can allocate draft weights."""

    if not checkpoint_path.is_absolute():
        raise DSparkConfigurationError(
            f"{DSPARK_CHECKPOINT_ENV} must be an absolute local directory"
        )
    if not checkpoint_path.is_dir():
        raise DSparkConfigurationError(
            f"{DSPARK_CHECKPOINT_ENV} must be an existing local directory"
        )
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise DSparkConfigurationError(
            f"{DSPARK_CHECKPOINT_ENV} is missing config.json"
        )
    actual_hash = sha256(config_path)
    if actual_hash != expected_config_sha256:
        raise DSparkConfigurationError(
            "Kimi K3 DSpark config.json does not match the pinned config hash"
        )
    model_path = checkpoint_path / "model.safetensors"
    if not model_path.is_file():
        raise DSparkConfigurationError(
            f"{DSPARK_CHECKPOINT_ENV} is missing model.safetensors"
        )
    if model_path.stat().st_size != expected_model_bytes:
        raise DSparkConfigurationError(
            "Kimi K3 DSpark model.safetensors does not match the pinned byte size"
        )


def kimi_k3_dspark_config(
    *,
    is_pipeline: bool,
    is_batch: bool,
    environ: Mapping[str, str] | None = None,
    warning: Callable[[str], None] = logger.warning,
    checkpoint_validator: Callable[[Path], None] = validate_local_dspark_checkpoint,
) -> KimiK3DSparkConfig | None:
    """Parse the fail-closed EXO and MLX-LM DSpark opt-ins.

    Configuration mistakes raise instead of silently selecting ordinary decode.
    :class:`KimiK3DSparkRoundEngine` falls back only while both caches are still
    recoverable; an uncertain target commit is a distributed fail-stop.
    """

    values = os.environ if environ is None else environ
    enabled_raw = values.get(DSPARK_ENABLE_ENV, "0")
    enabled = _strict_flag(DSPARK_ENABLE_ENV, enabled_raw)
    configured_companions = [name for name in _EXO_COMPANION_ENVS if name in values]
    if not enabled:
        if configured_companions:
            raise DSparkConfigurationError(
                f"{', '.join(configured_companions)} requires {DSPARK_ENABLE_ENV}=1"
            )
        return None

    if values.get(MLX_DSPARK_PROPOSER_ENV) != "1":
        raise DSparkConfigurationError(
            f"{MLX_DSPARK_PROPOSER_ENV}=1 is required when {DSPARK_ENABLE_ENV}=1"
        )
    if values.get(MLX_REPLAYSSM_ENV) != "1":
        raise DSparkConfigurationError(
            f"{MLX_REPLAYSSM_ENV}=1 is required when {DSPARK_ENABLE_ENV}=1"
        )
    if is_pipeline:
        raise DSparkConfigurationError(
            "Kimi K3 DSpark speculation does not support pipeline parallelism"
        )
    if is_batch:
        raise DSparkConfigurationError(
            "Kimi K3 DSpark speculation does not support batch generation"
        )

    checkpoint_raw = values.get(DSPARK_CHECKPOINT_ENV)
    if checkpoint_raw is None or checkpoint_raw == "":
        raise DSparkConfigurationError(
            f"{DSPARK_CHECKPOINT_ENV} is required when {DSPARK_ENABLE_ENV}=1"
        )
    checkpoint_path = Path(checkpoint_raw)
    checkpoint_validator(checkpoint_path)

    verify_width = _strict_verify_width(
        values.get(
            DSPARK_VERIFY_WIDTH_ENV,
            str(DSPARK_MODEL_NATIVE_VERIFY_WIDTH),
        )
    )
    if verify_width == DSPARK_CONSERVATIVE_VERIFY_WIDTH:
        warning(
            "Kimi K3 DSpark verify width 3 (gamma=2) overrides the "
            "model-native width 8 (gamma=7); use only for conservative rollout"
        )

    round_telemetry = _strict_flag(
        DSPARK_TELEMETRY_ENV,
        values.get(DSPARK_TELEMETRY_ENV, "0"),
    )
    return KimiK3DSparkConfig(
        checkpoint_path=checkpoint_path,
        verify_width=verify_width,
        round_telemetry=round_telemetry,
    )


@dataclass(frozen=True)
class TargetPosterior:
    """Target posterior tokens and ordered post-layer hidden-state taps."""

    tokens: tuple[int, ...]
    aux_hidden_states: tuple[object, ...] = ()


class DraftRound(Protocol):
    """One speculative mutation of a rank-local replicated draft cache."""

    @property
    def proposal_tokens(self) -> Sequence[int]: ...

    def commit(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> None: ...

    def cancel(self) -> None: ...


class ReplicatedDraft(Protocol):
    """Adapter implemented independently by every target TP rank."""

    placement: str
    verify_width: int

    def begin_round(self, anchor_token: int, num_proposals: int) -> DraftRound: ...


class TargetRound(Protocol):
    """One target width-N verification transaction."""

    @property
    def posterior(self) -> TargetPosterior: ...

    def commit(self, consumed_input_tokens: int) -> None: ...

    def cancel(self) -> None: ...


class WidthNTarget(Protocol):
    """Transactional target verification plus the unchanged one-token path."""

    def begin_verification(self, proposal_block: tuple[int, ...]) -> TargetRound: ...

    def ordinary_decode(self, anchor_token: int) -> int: ...


class RankAgreement(Protocol):
    """Fixed-width collectives used before distributed state transitions."""

    @property
    def rank(self) -> int: ...

    @property
    def size(self) -> int: ...

    def agree_proposal_block(
        self,
        local_block: tuple[int, ...] | None,
        expected_width: int,
    ) -> tuple[int, ...] | None: ...

    def agree_acceptance(
        self,
        local_boundary: int | None,
        local_next_token: int | None,
        maximum_boundary: int,
    ) -> tuple[int, int] | None: ...

    def agree_stage_success(self, local_success: bool) -> bool | None: ...


@final
class MlxRankAgreement:
    """Exact token/boundary agreement over an MLX distributed group."""

    def __init__(self, group: mx.distributed.Group | None):
        self._group = group

    @property
    def rank(self) -> int:
        return 0 if self._group is None else self._group.rank()

    @property
    def size(self) -> int:
        return 1 if self._group is None else self._group.size()

    def _all_gather_rows(self, row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        if self._group is None:
            return (row,)
        gathered = mx.distributed.all_gather(
            mx.array(row, dtype=mx.int32),
            group=self._group,
        )
        mx.eval(gathered)
        flat_values = cast(list[int], gathered.tolist())
        row_width = len(row)
        return tuple(
            tuple(flat_values[offset : offset + row_width])
            for offset in range(0, len(flat_values), row_width)
        )

    def agree_proposal_block(
        self,
        local_block: tuple[int, ...] | None,
        expected_width: int,
    ) -> tuple[int, ...] | None:
        if expected_width < 2:
            raise ValueError("proposal agreement requires width >= 2")
        valid = local_block is not None and len(local_block) == expected_width
        payload = (
            local_block if valid and local_block is not None else (-1,) * expected_width
        )
        rows = self._all_gather_rows((int(valid), *payload))
        first = rows[0]
        if first[0] != 1 or any(row != first for row in rows[1:]):
            return None
        return first[1:]

    def agree_acceptance(
        self,
        local_boundary: int | None,
        local_next_token: int | None,
        maximum_boundary: int,
    ) -> tuple[int, int] | None:
        valid = (
            local_boundary is not None
            and local_next_token is not None
            and 0 <= local_boundary <= maximum_boundary
            and local_next_token >= 0
        )
        row = (
            int(valid),
            local_boundary if valid and local_boundary is not None else -1,
            local_next_token if valid and local_next_token is not None else -1,
        )
        rows = self._all_gather_rows(row)
        first = rows[0]
        if first[0] != 1 or any(peer != first for peer in rows[1:]):
            return None
        return first[1], first[2]

    def agree_stage_success(self, local_success: bool) -> bool | None:
        """Return a unanimous result, or ``None`` when rank outcomes differ."""

        rows = self._all_gather_rows((int(local_success),))
        first = rows[0][0]
        if any(row[0] != first for row in rows[1:]):
            return None
        return first == 1


@dataclass(frozen=True)
class DSparkRoundTelemetry:
    """Timing and acceptance outcome for one speculative/fallback round."""

    round_index: int
    rank: int
    draft_ms: float
    target_verify_ms: float
    proposed: int
    accepted: int
    emitted: int
    fallback: bool
    error: str | None


@dataclass(frozen=True)
class DSparkRoundResult:
    emitted_tokens: tuple[int, ...]
    telemetry: DSparkRoundTelemetry


def log_dspark_round(telemetry: DSparkRoundTelemetry) -> None:
    logger.info(
        "MLX Kimi K3 DSpark round: "
        f"rank={telemetry.rank}, "
        f"round={telemetry.round_index}, "
        f"draft_ms={telemetry.draft_ms:.3f}, "
        f"target_verify_ms={telemetry.target_verify_ms:.3f}, "
        f"proposed={telemetry.proposed}, "
        f"accepted={telemetry.accepted}, "
        f"emitted={telemetry.emitted}, "
        f"fallback={telemetry.fallback}, "
        f"error={telemetry.error!r}"
    )


def _token_tuple(tokens: Sequence[int], *, expected: int, name: str) -> tuple[int, ...]:
    values = tuple(tokens)
    if len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} tokens")
    if any(type(token) is not int or token < 0 for token in values):
        raise ValueError(f"{name} must contain non-negative integer token ids")
    return values


def accepted_draft_prefix(
    proposal_block: tuple[int, ...],
    target_posterior: tuple[int, ...],
) -> int:
    """Return cumprod-style accepted proposals for a width-N verification."""

    if len(proposal_block) < 2 or len(target_posterior) != len(proposal_block):
        raise ValueError("proposal block and target posterior widths must match")
    accepted = 0
    for proposed, target_token in zip(
        proposal_block[1:],
        target_posterior[:-1],
        strict=True,
    ):
        if proposed != target_token:
            break
        accepted += 1
    return accepted


@dataclass
class KimiK3DSparkRoundEngine:
    """Run rank-agreed rounds with pre-commit fallback and commit fail-stop."""

    config: KimiK3DSparkConfig
    draft: ReplicatedDraft
    target: WidthNTarget
    collective: RankAgreement
    telemetry_sink: Callable[[DSparkRoundTelemetry], None] | None = None
    clock: Callable[[], float] = time.perf_counter
    _round_index: int = field(default=0, init=False)
    _disabled_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.draft.placement != "replicated":
            raise DSparkConfigurationError(
                "Kimi K3 DSpark must be replicated on every target TP rank"
            )
        if self.draft.verify_width != self.config.verify_width:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark proposer and EXO verify widths do not match"
            )

    def _publish(self, telemetry: DSparkRoundTelemetry) -> None:
        if self.config.round_telemetry:
            log_dspark_round(telemetry)
        if self.telemetry_sink is not None:
            try:
                self.telemetry_sink(telemetry)
            except Exception:
                logger.opt(exception=True).warning(
                    "Kimi K3 DSpark telemetry callback failed"
                )

    def _ordinary_fallback(
        self,
        anchor_token: int,
        *,
        draft_ms: float,
        target_verify_ms: float,
        proposed: int,
        error: str,
    ) -> DSparkRoundResult:
        self._disabled_reason = error
        fallback_error = error
        try:
            token = self.target.ordinary_decode(anchor_token)
            emitted_tokens = (token,)
        except Exception as decode_error:
            emitted_tokens = ()
            fallback_error = (
                f"{error}; ordinary target fallback failed: "
                f"{type(decode_error).__name__}: {decode_error}"
            )
            telemetry = DSparkRoundTelemetry(
                round_index=self._round_index,
                rank=self.collective.rank,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                proposed=proposed,
                accepted=0,
                emitted=0,
                fallback=True,
                error=fallback_error,
            )
            self._publish(telemetry)
            self._round_index += 1
            raise

        telemetry = DSparkRoundTelemetry(
            round_index=self._round_index,
            rank=self.collective.rank,
            draft_ms=draft_ms,
            target_verify_ms=target_verify_ms,
            proposed=proposed,
            accepted=0,
            emitted=len(emitted_tokens),
            fallback=True,
            error=fallback_error,
        )
        self._publish(telemetry)
        self._round_index += 1
        return DSparkRoundResult(emitted_tokens=emitted_tokens, telemetry=telemetry)

    def decode_round(self, anchor_token: int) -> DSparkRoundResult:
        """Decode one speculative round, or one ordinary token after rollback."""

        if type(anchor_token) is not int or anchor_token < 0:
            raise ValueError("anchor_token must be a non-negative integer")
        if self._disabled_reason is not None:
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=0.0,
                target_verify_ms=0.0,
                proposed=0,
                error=f"DSpark disabled after earlier error: {self._disabled_reason}",
            )

        gamma = self.config.gamma
        draft_round: DraftRound | None = None
        local_block: tuple[int, ...] | None = None
        local_error: str | None = None
        draft_start = self.clock()
        try:
            draft_round = self.draft.begin_round(anchor_token, gamma)
            proposals = _token_tuple(
                draft_round.proposal_tokens,
                expected=gamma,
                name="DSpark proposal",
            )
            local_block = (anchor_token, *proposals)
        except Exception as error:
            local_error = f"draft failed: {type(error).__name__}: {error}"
        draft_ms = (self.clock() - draft_start) * 1000.0

        agreed_block = self.collective.agree_proposal_block(
            local_block,
            self.config.verify_width,
        )
        if agreed_block is None:
            if draft_round is not None:
                draft_round.cancel()
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=draft_ms,
                target_verify_ms=0.0,
                proposed=0 if local_block is None else len(local_block) - 1,
                error=local_error or "DSpark proposal tokens disagreed across ranks",
            )

        target_round: TargetRound | None = None
        posterior: TargetPosterior | None = None
        local_boundary: int | None = None
        local_next_token: int | None = None
        target_error: str | None = None
        target_start = self.clock()
        try:
            target_round = self.target.begin_verification(agreed_block)
            posterior_tokens = _token_tuple(
                target_round.posterior.tokens,
                expected=self.config.verify_width,
                name="target posterior",
            )
            posterior = TargetPosterior(
                tokens=posterior_tokens,
                aux_hidden_states=target_round.posterior.aux_hidden_states,
            )
            local_boundary = accepted_draft_prefix(agreed_block, posterior_tokens)
            local_next_token = posterior_tokens[local_boundary]
        except Exception as error:
            target_error = f"target verify failed: {type(error).__name__}: {error}"
        target_verify_ms = (self.clock() - target_start) * 1000.0

        agreed_acceptance = self.collective.agree_acceptance(
            local_boundary,
            local_next_token,
            gamma,
        )
        if agreed_acceptance is None:
            if target_round is not None:
                target_round.cancel()
            if draft_round is not None:
                draft_round.cancel()
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                proposed=gamma,
                error=target_error
                or "DSpark acceptance boundary disagreed across ranks",
            )

        accepted, next_anchor_token = agreed_acceptance
        assert target_round is not None
        assert posterior is not None
        emitted_tokens = (
            *agreed_block[1 : accepted + 1],
            next_anchor_token,
        )

        # Acceptance is collective before either state commit.  The target is
        # authoritative; its consumed input count is anchor + accepted drafts.
        target_commit_error: str | None = None
        try:
            target_round.commit(accepted + 1)
        except Exception as error:
            target_commit_error = (
                f"target commit failed: {type(error).__name__}: {error}"
            )
            try:
                target_round.cancel()
            except Exception as cancel_error:
                target_commit_error += (
                    "; target cancellation also failed: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )

        target_commit_consensus = self.collective.agree_stage_success(
            target_commit_error is None
        )
        if target_commit_consensus is not True:
            if draft_round is not None:
                try:
                    draft_round.cancel()
                except Exception:
                    logger.opt(exception=True).warning(
                        "Kimi K3 DSpark draft cancellation failed during "
                        "fatal target commit handling"
                    )
            outcome = (
                "disagreed across ranks"
                if target_commit_consensus is None
                else "failed on every rank"
            )
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark target commit "
                f"{outcome}; target state cannot be recovered safely"
            ) from None

        draft_commit_error: str | None = None
        assert draft_round is not None
        try:
            draft_round.commit(accepted, next_anchor_token, posterior)
        except Exception as error:
            # The target commit is already authoritative.  Keep its emitted
            # tokens, discard this draft for future rounds, and use ordinary
            # target decode from the next anchor.
            draft_commit_error = (
                f"draft commit failed: {type(error).__name__}: {error}; "
                "DSpark disabled for subsequent rounds"
            )

        draft_commit_consensus = self.collective.agree_stage_success(
            draft_commit_error is None
        )
        if draft_commit_consensus is not True:
            outcome = (
                "outcome disagreed across ranks"
                if draft_commit_consensus is None
                else "failed on every rank"
            )
            draft_commit_error = (
                f"draft commit {outcome}; DSpark disabled on every rank"
            )
            self._disabled_reason = draft_commit_error

        telemetry = DSparkRoundTelemetry(
            round_index=self._round_index,
            rank=self.collective.rank,
            draft_ms=draft_ms,
            target_verify_ms=target_verify_ms,
            proposed=gamma,
            accepted=accepted,
            emitted=len(emitted_tokens),
            fallback=False,
            error=draft_commit_error,
        )
        self._publish(telemetry)
        self._round_index += 1
        return DSparkRoundResult(emitted_tokens=emitted_tokens, telemetry=telemetry)


class _SpeculativeCacheHooks(Protocol):
    def begin_speculative_cache(self, cache: object, width: int) -> object: ...

    def resolve_speculative_cache(self, transaction: object, consumed: int) -> None: ...

    def cancel_speculative_cache(self, transaction: object) -> None: ...


def has_replayssm_target_hooks(target_model: object) -> bool:
    """Feature-detect the exact MLX-LM accepted-prefix transaction surface."""

    required = (
        "forward_with_aux_hidden_states",
        "begin_speculative_cache",
        "resolve_speculative_cache",
        "cancel_speculative_cache",
    )
    return all(callable(getattr(target_model, name, None)) for name in required)


@final
class _ReplaySSMTargetRound:
    def __init__(
        self,
        hooks: _SpeculativeCacheHooks,
        transaction: object,
        posterior: TargetPosterior,
    ):
        self._hooks = hooks
        self._transaction = transaction
        self._posterior = posterior
        self._active = True

    @property
    def posterior(self) -> TargetPosterior:
        return self._posterior

    def commit(self, consumed_input_tokens: int) -> None:
        if not self._active:
            raise RuntimeError("target speculative transaction is no longer active")
        try:
            self._hooks.resolve_speculative_cache(
                self._transaction,
                consumed_input_tokens,
            )
        finally:
            # MLX-LM's hook either commits or cancels on failure.
            self._active = bool(getattr(self._transaction, "active", False))

    def cancel(self) -> None:
        if not self._active:
            return
        if bool(getattr(self._transaction, "active", True)):
            self._hooks.cancel_speculative_cache(self._transaction)
        self._active = False


@final
class ReplaySSMTargetAdapter:
    """Feature-detected target width-N / ReplaySSM commit-hook adapter."""

    def __init__(
        self,
        target_model: object,
        target_cache: object,
        verify: Callable[[tuple[int, ...]], TargetPosterior],
        ordinary_decode: Callable[[int], int],
    ):
        if not has_replayssm_target_hooks(target_model):
            raise DSparkFeatureUnavailableError(
                "MLX-LM Kimi K3 target is missing hidden-tap or ReplaySSM hooks"
            )
        self._hooks = cast(_SpeculativeCacheHooks, target_model)
        self._target_cache = target_cache
        self._verify = verify
        self._ordinary_decode = ordinary_decode

    def begin_verification(self, proposal_block: tuple[int, ...]) -> TargetRound:
        transaction = self._hooks.begin_speculative_cache(
            self._target_cache,
            len(proposal_block),
        )
        try:
            posterior = self._verify(proposal_block)
        except BaseException:
            if bool(getattr(transaction, "active", True)):
                self._hooks.cancel_speculative_cache(transaction)
            raise
        return _ReplaySSMTargetRound(self._hooks, transaction, posterior)

    def ordinary_decode(self, anchor_token: int) -> int:
        return self._ordinary_decode(anchor_token)


@dataclass(frozen=True)
class MlxDSparkFeatures:
    """Feature-detected, unstable MLX-LM draft construction callables."""

    load_kimi_k3_dspark: Callable[..., object]
    proposer_type: Callable[..., object]


def detect_mlx_dspark_features() -> MlxDSparkFeatures | None:
    """Detect the provisional MLX-LM API without importing its concrete types."""

    try:
        module = importlib.import_module("mlx_lm.models.kimi_k3_dspark")
    except ImportError:
        return None
    required = {
        "load_kimi_k3_dspark": getattr(module, "load_kimi_k3_dspark", None),
        "proposer_type": getattr(module, "KimiK3DSparkProposer", None),
    }
    if not all(callable(value) for value in required.values()):
        return None
    return MlxDSparkFeatures(
        load_kimi_k3_dspark=cast(
            Callable[..., object], required["load_kimi_k3_dspark"]
        ),
        proposer_type=cast(Callable[..., object], required["proposer_type"]),
    )


class _ProposalTokenArray(Protocol):
    def tolist(self) -> object: ...


class _AuxHiddenState(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def __getitem__(self, key: tuple[object, ...]) -> object: ...


class _MlxDSparkProposer(Protocol):
    verify_width: int

    def make_context_cache(self) -> object: ...

    def append_target_context(
        self,
        aux_hidden_states: Sequence[object],
        context_offset: int,
        context_cache: object,
    ) -> None: ...

    def propose(self, anchor_token: int, context_cache: object) -> object: ...


def _flatten_mlx_proposal_tokens(
    proposal: object,
    *,
    expected: int,
    verify_width: int,
) -> tuple[int, ...]:
    proposal_width = getattr(proposal, "verify_width", None)
    if type(proposal_width) is not int or proposal_width != verify_width:
        raise ValueError("MLX-LM DSpark proposal verify width does not match")
    raw_tokens = getattr(proposal, "tokens", None)
    tolist = getattr(raw_tokens, "tolist", None)
    if not callable(tolist):
        raise TypeError("MLX-LM DSpark proposal tokens must be an MLX array")
    rows = cast(_ProposalTokenArray, raw_tokens).tolist()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("MLX-LM DSpark proposal tokens must have batch size one")
    row_values = cast(Sequence[object], rows)
    if (
        len(row_values) != 1
        or not isinstance(row_values[0], Sequence)
        or isinstance(row_values[0], (str, bytes))
    ):
        raise ValueError("MLX-LM DSpark proposal tokens must have batch size one")
    return _token_tuple(
        cast(Sequence[int], row_values[0]),
        expected=expected,
        name="MLX-LM DSpark proposal",
    )


def _context_cache_offset(context_cache: object) -> int:
    if (
        not isinstance(context_cache, Sequence)
        or isinstance(context_cache, (str, bytes))
        or not context_cache
    ):
        raise ValueError("MLX-LM DSpark context cache must be a non-empty sequence")
    entries = cast(Sequence[object], context_cache)
    lengths = tuple(getattr(entry, "length", None) for entry in entries)
    if any(type(length) is not int or length < 0 for length in lengths):
        raise ValueError("MLX-LM DSpark context cache lengths are invalid")
    if len(set(lengths)) != 1:
        raise ValueError("MLX-LM DSpark context cache lengths disagree")
    return cast(int, lengths[0])


def _committed_aux_hidden_states(
    aux_hidden_states: tuple[object, ...],
    *,
    consumed: int,
    verify_width: int,
) -> tuple[object, ...]:
    if not aux_hidden_states:
        raise ValueError("target posterior is missing DSpark auxiliary hidden states")
    committed: list[object] = []
    for hidden in aux_hidden_states:
        shape = getattr(hidden, "shape", None)
        if not isinstance(shape, Sequence):
            raise ValueError(
                "target DSpark auxiliary hidden-state shape does not match"
            )
        dimensions = cast(Sequence[object], shape)
        if len(dimensions) != 3:
            raise ValueError(
                "target DSpark auxiliary hidden-state shape does not match"
            )
        batch_size, token_count = dimensions[:2]
        if (
            type(batch_size) is not int
            or batch_size != 1
            or type(token_count) is not int
            or token_count != verify_width
        ):
            raise ValueError(
                "target DSpark auxiliary hidden-state shape does not match"
            )
        hidden_state = cast(_AuxHiddenState, hidden)
        committed.append(hidden_state[:, :consumed, :])
    return tuple(committed)


@final
class _MlxDSparkDraftRound:
    """Non-mutating proposal followed by append-only committed target context."""

    def __init__(
        self,
        proposer: _MlxDSparkProposer,
        context_cache: object,
        proposal_tokens: tuple[int, ...],
        verify_width: int,
    ):
        self._proposer = proposer
        self._context_cache = context_cache
        self._proposal_tokens = proposal_tokens
        self._verify_width = verify_width
        self._active = True

    @property
    def proposal_tokens(self) -> tuple[int, ...]:
        return self._proposal_tokens

    def commit(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> None:
        if not self._active:
            raise RuntimeError("MLX-LM DSpark draft round is no longer active")
        if type(
            accepted_draft_tokens
        ) is not int or not 0 <= accepted_draft_tokens <= len(self._proposal_tokens):
            raise ValueError("accepted DSpark draft-token count is invalid")
        posterior_tokens = _token_tuple(
            target_posterior.tokens,
            expected=self._verify_width,
            name="target posterior",
        )
        if (
            type(next_anchor_token) is not int
            or next_anchor_token != posterior_tokens[accepted_draft_tokens]
        ):
            raise ValueError("next DSpark anchor does not match the target posterior")
        consumed = accepted_draft_tokens + 1
        committed_hidden = _committed_aux_hidden_states(
            target_posterior.aux_hidden_states,
            consumed=consumed,
            verify_width=self._verify_width,
        )
        context_offset = _context_cache_offset(self._context_cache)
        try:
            self._proposer.append_target_context(
                committed_hidden,
                context_offset,
                self._context_cache,
            )
        finally:
            # MLX-LM exposes append-only context, not rollback. Never retry a
            # possibly partial append; rank consensus disables the draft.
            self._active = False

    def cancel(self) -> None:
        # ``propose`` does not mutate MLX-LM's target-context cache.
        self._active = False


@dataclass(frozen=True)
class LoadedMlxDSpark:
    """Concrete rank-local replicated draft backed by MLX-LM's proposer."""

    drafter: object
    proposer: object
    context_cache: object
    verify_width: int
    placement: Literal["replicated"] = "replicated"

    def begin_round(self, anchor_token: int, num_proposals: int) -> DraftRound:
        if type(anchor_token) is not int or anchor_token < 0:
            raise ValueError("anchor_token must be a non-negative integer")
        if num_proposals != self.verify_width - 1:
            raise ValueError("requested DSpark proposal count does not match its width")
        proposer = cast(_MlxDSparkProposer, self.proposer)
        proposal = proposer.propose(anchor_token, self.context_cache)
        proposal_tokens = _flatten_mlx_proposal_tokens(
            proposal,
            expected=num_proposals,
            verify_width=self.verify_width,
        )
        return _MlxDSparkDraftRound(
            proposer,
            self.context_cache,
            proposal_tokens,
            self.verify_width,
        )


def load_replicated_mlx_dspark(
    config: KimiK3DSparkConfig,
    target_model: object,
    *,
    features: MlxDSparkFeatures | None = None,
) -> LoadedMlxDSpark:
    """Load the pinned local draft independently on the calling target rank.

    EXO intentionally passes ``verify_weights_sha256=True`` and an explicit
    width. No model identifier is accepted here, so this adapter cannot trigger
    a remote checkpoint download.
    """

    detected = detect_mlx_dspark_features() if features is None else features
    if detected is None:
        raise DSparkFeatureUnavailableError(
            "installed MLX-LM does not expose the Kimi K3 DSpark proposer API"
        )
    drafter = detected.load_kimi_k3_dspark(
        config.checkpoint_path,
        target_model,
        verify_weights_sha256=True,
    )
    proposer = detected.proposer_type(
        drafter,
        verify_width=config.verify_width,
        screening_override=(config.verify_width == DSPARK_CONSERVATIVE_VERIFY_WIDTH),
    )
    required_methods = (
        "propose",
        "make_context_cache",
        "append_target_context",
    )
    if not all(callable(getattr(proposer, name, None)) for name in required_methods):
        raise DSparkFeatureUnavailableError(
            "MLX-LM DSpark proposer is missing proposal or context-cache methods"
        )
    proposer_width = getattr(proposer, "verify_width", None)
    if type(proposer_width) is not int or proposer_width != config.verify_width:
        raise DSparkFeatureUnavailableError(
            "MLX-LM DSpark proposer returned an unexpected verify width"
        )
    context_cache = cast(_MlxDSparkProposer, proposer).make_context_cache()
    _context_cache_offset(context_cache)
    return LoadedMlxDSpark(
        drafter=drafter,
        proposer=proposer,
        context_cache=context_cache,
        verify_width=config.verify_width,
    )
