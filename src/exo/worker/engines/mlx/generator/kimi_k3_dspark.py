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

import contextlib
import fcntl
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import stat
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast, final

import mlx.core as mx

from exo.worker.engines.mlx.generator.kimi_k3_width4_receipt import (
    width4_receipt_log_enabled,
)
from exo.worker.runner.bootstrap import logger

DSPARK_ENABLE_ENV = "EXO_MLX_KIMI_K3_DSPARK_SPECULATIVE"
DSPARK_CHECKPOINT_ENV = "EXO_MLX_KIMI_K3_DSPARK_CHECKPOINT"
DSPARK_VERIFY_WIDTH_ENV = "EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH"
DSPARK_TELEMETRY_ENV = "EXO_MLX_KIMI_K3_DSPARK_ROUND_TELEMETRY"
DSPARK_AUX_ONLY_PREFILL_ENV = "EXO_MLX_KIMI_K3_DSPARK_AUX_ONLY_PREFILL"
DSPARK_CONFIDENCE_JSONL_ENV = "EXO_MLX_KIMI_K3_DSPARK_CONFIDENCE_JSONL"
DSPARK_CONFIDENCE_SESSION_ENV = "EXO_MLX_KIMI_K3_DSPARK_CONFIDENCE_SESSION"
DSPARK_DUAL_PROPOSER_ENV = "EXO_MLX_KIMI_K3_DSPARK_DUAL_PROPOSER"
DSPARK_YARN_CHECKPOINT_ENV = "EXO_MLX_KIMI_K3_DSPARK_YARN_CHECKPOINT"
DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV = (
    "EXO_MLX_KIMI_K3_DSPARK_RANK_ZERO_PROPOSAL_RECOVERY"
)
DSPARK_PACKED_AGREEMENTS_ENV = "EXO_MLX_KIMI_K3_DSPARK_PACKED_AGREEMENTS"
DSPARK_TAIL_OVERLAP_ENV = "EXO_MLX_KIMI_K3_DSPARK_TAIL_OVERLAP"

# These controls live on separate experimental branches.  The first dual
# proposer deliberately rejects them instead of silently composing untested
# request-state machines when those branches are integrated.
DSPARK_PREFIX_CACHE_ENV = "EXO_MLX_KIMI_K3_DSPARK_PREFIX_CACHE"
DSPARK_ADAPTIVE_GATE_ENV = "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_W3_GATE"
DSPARK_ADAPTIVE_GATE_POLICY_ENV = "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_W3_GATE_POLICY"
DSPARK_ADAPTIVE_GATE_POLICY_SHA256_ENV = (
    "EXO_MLX_KIMI_K3_DSPARK_ORDINARY_W3_GATE_POLICY_SHA256"
)

MLX_DSPARK_PROPOSER_ENV = "MLX_LM_KIMI_K3_DSPARK_PROPOSER"
MLX_REPLAYSSM_ENV = "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE"
MLX_DSPARK_SEGMENTED_SDPA_ENV = "MLX_LM_KIMI_K3_DSPARK_SEGMENTED_SDPA"
EXO_VOCAB_PARALLEL_GREEDY_ENV = "EXO_MLX_K3_VOCAB_PARALLEL_GREEDY"

RADIXARK_KIMI_K3_DSPARK_MODEL = "RadixArk/Kimi-K3-DSpark"
RADIXARK_KIMI_K3_DSPARK_REVISION = "eb03982e58d4fb79bcfc099e902158f562e2e27b"
RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256 = (
    "6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f"
)
RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES = 4_498_585_858
RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256 = (
    "29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495"
)
RADIXARK_KIMI_K3_DSPARK_YARN_REVISION = "9c4b2577dacb572ce88e8aad4357dffb4f6c9796"
RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256 = (
    "410dd228c75ff91b57af8a1581d44d2ea096d5604f0d37fbd400470e90d961d3"
)
RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256 = (
    "ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca"
)
RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS = (7, 23, 51, 67, 83)
RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE = 7
KIMI_K3_TARGET_HIDDEN_SIZE = 7168
KIMI_K3_TARGET_TAP_COUNT = len(RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS)
KIMI_K3_MAX_CONTEXT_LENGTH = 1_048_576
DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS = 8_192

# SGLang's published K3 deployment maps checkpoint block_size directly to
# gamma, so seven proposals plus the current anchor are verified at once.
DSPARK_MODEL_NATIVE_GAMMA = RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE
DSPARK_MODEL_NATIVE_VERIFY_WIDTH = DSPARK_MODEL_NATIVE_GAMMA + 1

# MLX-LM's initial rollout uses a deliberately conservative two proposals.
DSPARK_CONSERVATIVE_GAMMA = 2
DSPARK_CONSERVATIVE_VERIFY_WIDTH = DSPARK_CONSERVATIVE_GAMMA + 1
DSPARK_INTERMEDIATE_GAMMA = 3
DSPARK_INTERMEDIATE_VERIFY_WIDTH = DSPARK_INTERMEDIATE_GAMMA + 1
DSPARK_ALLOWED_VERIFY_WIDTHS = (
    DSPARK_CONSERVATIVE_VERIFY_WIDTH,
    DSPARK_INTERMEDIATE_VERIFY_WIDTH,
    DSPARK_MODEL_NATIVE_VERIFY_WIDTH,
)
KimiK3DSparkVerifyWidth = Literal[3, 4, 8]

_EXO_COMPANION_ENVS = (
    DSPARK_CHECKPOINT_ENV,
    DSPARK_VERIFY_WIDTH_ENV,
    DSPARK_TELEMETRY_ENV,
    DSPARK_AUX_ONLY_PREFILL_ENV,
    DSPARK_CONFIDENCE_JSONL_ENV,
    DSPARK_CONFIDENCE_SESSION_ENV,
    DSPARK_DUAL_PROPOSER_ENV,
    DSPARK_YARN_CHECKPOINT_ENV,
    DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV,
    DSPARK_PACKED_AGREEMENTS_ENV,
    DSPARK_TAIL_OVERLAP_ENV,
)


class DSparkConfigurationError(ValueError):
    """An explicit DSpark deployment configuration is invalid."""


class DSparkFeatureUnavailableError(RuntimeError):
    """The installed MLX-LM build does not expose the required DSpark API."""


class DSparkDistributedStateError(RuntimeError):
    """A distributed state transition cannot be recovered safely."""


class DSparkCancellationError(RuntimeError):
    """A speculative transaction could not be cancelled safely."""


@dataclass(frozen=True)
class KimiK3DSparkCheckpointContract:
    """One exact, audited DSpark config/weight identity pair."""

    revision: str
    config_sha256: str
    model_bytes: int
    model_sha256: str


_DSPARK_CHECKPOINT_CONTRACTS_BY_CONFIG_SHA256 = {
    RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256: KimiK3DSparkCheckpointContract(
        revision=RADIXARK_KIMI_K3_DSPARK_REVISION,
        config_sha256=RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256,
        model_bytes=RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES,
        model_sha256=RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256,
    ),
    RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256: KimiK3DSparkCheckpointContract(
        revision=RADIXARK_KIMI_K3_DSPARK_YARN_REVISION,
        config_sha256=RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256,
        model_bytes=RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES,
        model_sha256=RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256,
    ),
}


@dataclass(frozen=True)
class DSparkConfidenceCaptureConfig:
    """Strict operator metadata for one conservative-width confidence capture."""

    jsonl_path: Path
    session_id: str


@dataclass(frozen=True)
class KimiK3DSparkConfig:
    """Validated local checkpoint and replicated placement contract."""

    checkpoint_path: Path
    verify_width: KimiK3DSparkVerifyWidth
    round_telemetry: bool
    placement: Literal["replicated"] = "replicated"
    model_id: str = RADIXARK_KIMI_K3_DSPARK_MODEL
    revision: str = RADIXARK_KIMI_K3_DSPARK_REVISION
    config_sha256: str = RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256
    model_bytes: int = RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES
    model_sha256: str = RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256
    target_layer_ids: tuple[int, ...] = RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS
    aux_only_prefill: bool = False
    confidence_capture: DSparkConfidenceCaptureConfig | None = None
    rank_zero_proposal_recovery: bool = False
    packed_agreements: bool = False
    tail_overlap: bool = False

    @property
    def gamma(self) -> int:
        return self.verify_width - 1

    @property
    def target_hidden_state_indices(self) -> tuple[int, ...]:
        """Direct MLX-LM layer ids for post-layer target taps."""

        return self.target_layer_ids


@dataclass(frozen=True)
class KimiK3DSparkDualConfig:
    """Two exact proposer identities selected once from initial prompt length."""

    old: KimiK3DSparkConfig
    yarn: KimiK3DSparkConfig
    threshold_tokens: int = DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS

    def __post_init__(self) -> None:
        if self.threshold_tokens != DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS:
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark threshold must be exactly 8192 tokens"
            )
        if (
            self.old.revision != RADIXARK_KIMI_K3_DSPARK_REVISION
            or self.old.config_sha256 != RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256
            or self.old.model_sha256 != RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256
        ):
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark old checkpoint identity does not match"
            )
        if (
            self.yarn.revision != RADIXARK_KIMI_K3_DSPARK_YARN_REVISION
            or self.yarn.config_sha256 != RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256
            or self.yarn.model_sha256 != RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256
        ):
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark YaRN checkpoint identity does not match"
            )
        if self.old.checkpoint_path.resolve() == self.yarn.checkpoint_path.resolve():
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark checkpoints must be distinct directories"
            )
        shared_contract = (
            "verify_width",
            "round_telemetry",
            "placement",
            "model_id",
            "model_bytes",
            "target_layer_ids",
            "aux_only_prefill",
            "confidence_capture",
            "rank_zero_proposal_recovery",
            "packed_agreements",
            "tail_overlap",
        )
        if any(
            getattr(self.old, name) != getattr(self.yarn, name)
            for name in shared_contract
        ):
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark checkpoint runtime contracts disagree"
            )
        if self.old.confidence_capture is not None:
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark cannot be combined with confidence capture"
            )

    @property
    def verify_width(self) -> KimiK3DSparkVerifyWidth:
        return self.old.verify_width

    @property
    def rank_zero_proposal_recovery(self) -> bool:
        return self.old.rank_zero_proposal_recovery

    @property
    def packed_agreements(self) -> bool:
        return self.old.packed_agreements


KimiK3DSparkDeploymentConfig = KimiK3DSparkConfig | KimiK3DSparkDualConfig


def _strict_flag(name: str, raw: str) -> bool:
    if raw not in {"0", "1"}:
        raise DSparkConfigurationError(f"{name} must be 0 or 1")
    return raw == "1"


def _strict_verify_width(raw: str) -> KimiK3DSparkVerifyWidth:
    if not raw.isascii() or not raw.isdecimal():
        raise DSparkConfigurationError(f"{DSPARK_VERIFY_WIDTH_ENV} must be 3, 4, or 8")
    parsed = int(raw)
    if parsed not in DSPARK_ALLOWED_VERIFY_WIDTHS:
        raise DSparkConfigurationError(f"{DSPARK_VERIFY_WIDTH_ENV} must be 3, 4, or 8")
    return cast(KimiK3DSparkVerifyWidth, parsed)


def _strict_capture_label(name: str, raw: str) -> str:
    if not 1 <= len(raw) <= 128 or any(
        not (character.isascii() and (character.isalnum() or character in "._-"))
        for character in raw
    ):
        raise DSparkConfigurationError(
            f"{name} must be 1-128 ASCII letters, digits, dots, underscores, or dashes"
        )
    return raw


def _validate_confidence_jsonl_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise DSparkConfigurationError(
            f"{DSPARK_CONFIDENCE_JSONL_ENV} must be an absolute path"
        )
    if not path.parent.is_dir():
        raise DSparkConfigurationError(
            f"{DSPARK_CONFIDENCE_JSONL_ENV} parent must be an existing directory"
        )
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return path
    except OSError as error:
        raise DSparkConfigurationError(
            f"{DSPARK_CONFIDENCE_JSONL_ENV} cannot be inspected: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise DSparkConfigurationError(
            f"{DSPARK_CONFIDENCE_JSONL_ENV} must be a regular file"
        )
    if metadata.st_nlink != 1:
        raise DSparkConfigurationError(
            f"{DSPARK_CONFIDENCE_JSONL_ENV} must not be hard-linked"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise DSparkConfigurationError(
            f"{DSPARK_CONFIDENCE_JSONL_ENV} must not grant group or other access"
        )
    return path


def _confidence_capture_config(
    values: Mapping[str, str],
    *,
    verify_width: int,
) -> DSparkConfidenceCaptureConfig | None:
    configured = tuple(
        name
        for name in (DSPARK_CONFIDENCE_JSONL_ENV, DSPARK_CONFIDENCE_SESSION_ENV)
        if name in values
    )
    if not configured:
        return None
    if len(configured) != 2:
        missing = (
            DSPARK_CONFIDENCE_SESSION_ENV
            if DSPARK_CONFIDENCE_JSONL_ENV in configured
            else DSPARK_CONFIDENCE_JSONL_ENV
        )
        raise DSparkConfigurationError(
            f"{missing} is required for DSpark confidence capture"
        )
    if verify_width != DSPARK_CONSERVATIVE_VERIFY_WIDTH:
        raise DSparkConfigurationError(
            "DSpark confidence capture requires conservative verify width 3"
        )
    return DSparkConfidenceCaptureConfig(
        jsonl_path=_validate_confidence_jsonl_path(values[DSPARK_CONFIDENCE_JSONL_ENV]),
        session_id=_strict_capture_label(
            DSPARK_CONFIDENCE_SESSION_ENV,
            values[DSPARK_CONFIDENCE_SESSION_ENV],
        ),
    )


def validate_dspark_greedy_sampling(
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    logprobs: bool,
    top_logprobs: int | None,
    repetition_penalty: float | None,
    repetition_context_size: int | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
) -> None:
    """Reject request controls the deterministic DSpark decoder cannot honor."""

    if logprobs or top_logprobs is not None:
        raise ValueError("Kimi K3 DSpark does not support logprobs")
    if temperature != 0.0:
        raise ValueError(
            "Kimi K3 DSpark first canary supports greedy temperature=0 only"
        )
    if (
        top_p not in (None, 1.0)
        or top_k not in (None, 0)
        or min_p
        not in (
            None,
            0.05,
        )
    ):
        raise ValueError(
            "Kimi K3 DSpark does not support nondefault top_p, top_k, or min_p"
        )
    if (
        repetition_penalty not in (None, 1.0)
        or repetition_context_size is not None
        or presence_penalty not in (None, 0.0)
        or frequency_penalty not in (None, 0.0)
    ):
        raise ValueError("Kimi K3 DSpark does not support logit processors")


def dspark_context_capacity_hint(
    *,
    prompt_tokens: int,
    max_tokens: int,
    verify_width: int,
) -> int:
    """Return the exact maximum logical cache offset for this request.

    Prompt seeding consumes ``prompt_tokens - 1`` because the final prompt
    token is the decode anchor. A width-N verifier starts only when at least N
    output slots remain, so neither its temporary nor committed cache offset
    can exceed that prefix plus ``max_tokens``.
    """

    if type(prompt_tokens) is not int or prompt_tokens < 2:
        raise ValueError("Kimi K3 DSpark requires at least two prompt tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("Kimi K3 DSpark max tokens must be positive")
    if verify_width not in DSPARK_ALLOWED_VERIFY_WIDTHS:
        raise ValueError("Kimi K3 DSpark verify width must be 3, 4, or 8")
    capacity_hint = prompt_tokens - 1 + max_tokens
    if capacity_hint > KIMI_K3_MAX_CONTEXT_LENGTH:
        raise ValueError(
            "Kimi K3 DSpark prompt prefix and output exceed the "
            f"{KIMI_K3_MAX_CONTEXT_LENGTH}-token context limit"
        )
    return capacity_hint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as config_file:
        for chunk in iter(lambda: config_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_dspark_checkpoint(
    checkpoint_path: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_model_bytes: int = RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES,
    sha256: Callable[[Path], str] = _sha256,
) -> KimiK3DSparkCheckpointContract | None:
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
    if expected_config_sha256 is not None and actual_hash != expected_config_sha256:
        raise DSparkConfigurationError(
            "Kimi K3 DSpark config.json does not match the pinned config hash"
        )
    checkpoint_contract = _DSPARK_CHECKPOINT_CONTRACTS_BY_CONFIG_SHA256.get(actual_hash)
    if expected_config_sha256 is None and checkpoint_contract is None:
        raise DSparkConfigurationError(
            "Kimi K3 DSpark config.json does not match an audited config hash"
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
    return checkpoint_contract


def kimi_k3_dspark_config(
    *,
    is_pipeline: bool,
    is_batch: bool,
    environ: Mapping[str, str] | None = None,
    warning: Callable[[str], None] = logger.warning,
    checkpoint_validator: Callable[[Path], object] = validate_local_dspark_checkpoint,
) -> KimiK3DSparkDeploymentConfig | None:
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

    dual_enabled = _strict_flag(
        DSPARK_DUAL_PROPOSER_ENV,
        values.get(DSPARK_DUAL_PROPOSER_ENV, "0"),
    )
    yarn_checkpoint_raw = values.get(DSPARK_YARN_CHECKPOINT_ENV)
    if not dual_enabled and yarn_checkpoint_raw is not None:
        raise DSparkConfigurationError(
            f"{DSPARK_YARN_CHECKPOINT_ENV} requires {DSPARK_DUAL_PROPOSER_ENV}=1"
        )
    if dual_enabled:
        incompatible_enabled: list[str] = []
        for name in (DSPARK_PREFIX_CACHE_ENV, DSPARK_ADAPTIVE_GATE_ENV):
            raw = values.get(name)
            if raw is not None and _strict_flag(name, raw):
                incompatible_enabled.append(name)
        incompatible_configured = [
            name
            for name in (
                DSPARK_ADAPTIVE_GATE_POLICY_ENV,
                DSPARK_ADAPTIVE_GATE_POLICY_SHA256_ENV,
                DSPARK_CONFIDENCE_JSONL_ENV,
                DSPARK_CONFIDENCE_SESSION_ENV,
            )
            if name in values
        ]
        incompatible = incompatible_enabled + incompatible_configured
        if incompatible:
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark cannot be combined with " + ", ".join(incompatible)
            )
        if yarn_checkpoint_raw is None or yarn_checkpoint_raw == "":
            raise DSparkConfigurationError(
                f"{DSPARK_YARN_CHECKPOINT_ENV} is required when "
                f"{DSPARK_DUAL_PROPOSER_ENV}=1"
            )

    checkpoint_raw = values.get(DSPARK_CHECKPOINT_ENV)
    if checkpoint_raw is None or checkpoint_raw == "":
        raise DSparkConfigurationError(
            f"{DSPARK_CHECKPOINT_ENV} is required when {DSPARK_ENABLE_ENV}=1"
        )
    checkpoint_path = Path(checkpoint_raw)
    raw_checkpoint_contract: object = checkpoint_validator(checkpoint_path)
    if raw_checkpoint_contract is not None and not isinstance(
        raw_checkpoint_contract, KimiK3DSparkCheckpointContract
    ):
        raise DSparkConfigurationError(
            "Kimi K3 DSpark checkpoint validator returned an invalid contract"
        )
    checkpoint_contract = raw_checkpoint_contract

    verify_width = _strict_verify_width(
        values.get(
            DSPARK_VERIFY_WIDTH_ENV,
            str(DSPARK_MODEL_NATIVE_VERIFY_WIDTH),
        )
    )
    if verify_width != DSPARK_MODEL_NATIVE_VERIFY_WIDTH:
        warning(
            f"Kimi K3 DSpark verify width {verify_width} "
            f"(gamma={verify_width - 1}) overrides the model-native width 8 "
            "(gamma=7); use only for an explicitly screened rollout"
        )

    round_telemetry = _strict_flag(
        DSPARK_TELEMETRY_ENV,
        values.get(DSPARK_TELEMETRY_ENV, "0"),
    )
    aux_only_prefill = _strict_flag(
        DSPARK_AUX_ONLY_PREFILL_ENV,
        values.get(DSPARK_AUX_ONLY_PREFILL_ENV, "0"),
    )
    confidence_capture = _confidence_capture_config(
        values,
        verify_width=verify_width,
    )
    rank_zero_proposal_recovery = _strict_flag(
        DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV,
        values.get(DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV, "0"),
    )
    packed_agreements = _strict_flag(
        DSPARK_PACKED_AGREEMENTS_ENV,
        values.get(DSPARK_PACKED_AGREEMENTS_ENV, "0"),
    )
    tail_overlap = _strict_flag(
        DSPARK_TAIL_OVERLAP_ENV,
        values.get(DSPARK_TAIL_OVERLAP_ENV, "0"),
    )
    revision = RADIXARK_KIMI_K3_DSPARK_REVISION
    config_sha256 = RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256
    model_bytes = RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES
    model_sha256 = RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256
    if checkpoint_contract is not None:
        revision = checkpoint_contract.revision
        config_sha256 = checkpoint_contract.config_sha256
        model_bytes = checkpoint_contract.model_bytes
        model_sha256 = checkpoint_contract.model_sha256
    primary_config = KimiK3DSparkConfig(
        checkpoint_path=checkpoint_path,
        verify_width=verify_width,
        round_telemetry=round_telemetry,
        revision=revision,
        config_sha256=config_sha256,
        model_bytes=model_bytes,
        model_sha256=model_sha256,
        aux_only_prefill=aux_only_prefill,
        confidence_capture=confidence_capture,
        rank_zero_proposal_recovery=rank_zero_proposal_recovery,
        packed_agreements=packed_agreements,
        tail_overlap=tail_overlap,
    )
    if not dual_enabled:
        return primary_config

    if checkpoint_contract is None or (
        checkpoint_contract.revision != RADIXARK_KIMI_K3_DSPARK_REVISION
        or checkpoint_contract.config_sha256 != RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256
        or checkpoint_contract.model_sha256 != RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256
    ):
        raise DSparkConfigurationError(
            "Kimi K3 dual DSpark primary checkpoint must be the pinned old revision"
        )
    assert yarn_checkpoint_raw is not None
    yarn_checkpoint_path = Path(yarn_checkpoint_raw)
    yarn_contract = checkpoint_validator(yarn_checkpoint_path)
    if not isinstance(yarn_contract, KimiK3DSparkCheckpointContract) or (
        yarn_contract.revision != RADIXARK_KIMI_K3_DSPARK_YARN_REVISION
        or yarn_contract.config_sha256 != RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256
        or yarn_contract.model_sha256 != RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256
    ):
        raise DSparkConfigurationError(
            "Kimi K3 dual DSpark YaRN checkpoint must be the pinned 9c4 revision"
        )
    yarn_config = KimiK3DSparkConfig(
        checkpoint_path=yarn_checkpoint_path,
        verify_width=verify_width,
        round_telemetry=round_telemetry,
        revision=yarn_contract.revision,
        config_sha256=yarn_contract.config_sha256,
        model_bytes=yarn_contract.model_bytes,
        model_sha256=yarn_contract.model_sha256,
        aux_only_prefill=aux_only_prefill,
        rank_zero_proposal_recovery=rank_zero_proposal_recovery,
        packed_agreements=packed_agreements,
        tail_overlap=tail_overlap,
    )
    return KimiK3DSparkDualConfig(old=primary_config, yarn=yarn_config)


@dataclass(frozen=True)
class TargetPosterior:
    """Target posterior tokens and ordered post-layer hidden-state taps."""

    tokens: tuple[int, ...]
    aux_hidden_states: tuple[object, ...] = ()


class DraftRound(Protocol):
    """One speculative mutation of a rank-local replicated draft cache."""

    @property
    def proposal_tokens(self) -> Sequence[int]: ...

    @property
    def confidence_logits(self) -> Sequence[float] | None: ...

    def commit(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> None: ...

    def cancel(self) -> None: ...


class PreparedDraftRound(Protocol):
    """A rank-local lazy proposal graph that has not entered TP collectives."""

    def materialize(self) -> DraftRound: ...

    def cancel(self) -> None: ...


class _SplitCommitDraftRound(Protocol):
    """Draft round whose commit can be split around an asynchronous submit."""

    def commit_build(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> tuple[int, tuple[object, ...]]: ...

    def commit_finalize(self, *, evaluate: bool) -> None: ...


class _SubmittablePreparedDraftRound(Protocol):
    """Prepared draft round that can enqueue its graphs without blocking."""

    def submit(self, context_arrays: Sequence[object]) -> None: ...


class ReplicatedDraft(Protocol):
    """Adapter implemented independently by every target TP rank."""

    @property
    def placement(self) -> str: ...

    @property
    def verify_width(self) -> int: ...

    def preflight_round(self, anchor_token: int, num_proposals: int) -> int: ...

    def prepare_round(
        self,
        anchor_token: int,
        num_proposals: int,
    ) -> PreparedDraftRound: ...


class TargetRound(Protocol):
    """One target width-N verification transaction."""

    @property
    def posterior(self) -> TargetPosterior: ...

    def commit(self, consumed_input_tokens: int) -> None: ...

    def cancel(self) -> None: ...


class PreparedTargetVerification(Protocol):
    """Locally opened target transaction, before any target TP forward."""

    @property
    def initial_offset(self) -> int: ...

    @property
    def mode_code(self) -> int: ...

    def build(self) -> BuiltTargetVerification: ...

    def cancel(self) -> None: ...


class BuiltTargetVerification(Protocol):
    """A shape-checked target graph whose TP collectives are still lazy."""

    def materialize(self) -> TargetRound: ...

    def cancel(self) -> None: ...


@dataclass(frozen=True)
class TargetVerificationPlan:
    """Feature-detected target verifier payload agreed before graph build."""

    mode: Literal["full", "compact"]

    @property
    def agreement_code(self) -> int:
        return int(self.mode == "compact")


@dataclass(frozen=True)
class OrdinaryDecodePlan:
    """Rank-agreed target-only cache boundary and collective payload mode."""

    initial_offset: int
    mode: Literal["full", "compact"]

    @property
    def agreement_code(self) -> int:
        return self.initial_offset * 2 + int(self.mode == "compact")


class PreparedOrdinaryDecode(Protocol):
    """A shape-checked ordinary target graph before TP materialization."""

    def materialize(self) -> int: ...


class WidthNTarget(Protocol):
    """Transactional target verification plus the unchanged one-token path."""

    def prepare_verification(
        self,
        proposal_block: tuple[int, ...],
    ) -> PreparedTargetVerification: ...

    def preflight_ordinary(self, anchor_token: int) -> OrdinaryDecodePlan: ...

    def prepare_ordinary(
        self,
        anchor_token: int,
        plan: OrdinaryDecodePlan,
    ) -> PreparedOrdinaryDecode: ...


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

    def agree_token(self, local_token: int | None) -> int | None: ...

    def agree_packed(
        self,
        name: str,
        *,
        local_success: bool,
        error_fingerprint: int,
        payload: tuple[int, ...] = (),
    ) -> "PackedRankAgreement": ...


PACKED_AGREEMENT_PAYLOAD_WIDTH = 8
PACKED_AGREEMENT_ROW_WIDTH = 4 + PACKED_AGREEMENT_PAYLOAD_WIDTH


@dataclass(frozen=True)
class PackedRankAgreement:
    """One unanimous fixed-row agreement, or an all-``None`` mismatch."""

    success: bool | None
    error_fingerprint: int | None
    payload: tuple[int, ...] | None


@dataclass(frozen=True)
class PackedAgreementAttestation:
    """Request-local accounting for agreement rows and physical collectives."""

    enabled: bool
    row_calls: int
    packed_row_calls: int
    legacy_row_calls: int
    physical_all_gathers: int


def _packed_operation_tag(name: str) -> int:
    if not name or len(name) > 256:
        raise ValueError("packed agreement name must contain 1-256 characters")
    digest = hashlib.sha256(b"exo-kimi-k3-packed-agreement/v1\0")
    digest.update(name.encode("utf-8"))
    return int.from_bytes(digest.digest()[:4], "big") & 0x7FFFFFFF


@final
class MlxRankAgreement:
    """Exact token/boundary agreement over an MLX distributed group."""

    def __init__(
        self,
        group: mx.distributed.Group | None,
        *,
        rank_zero_proposal_recovery: bool = False,
    ):
        self._group = group
        self._rank_zero_proposal_recovery = rank_zero_proposal_recovery
        self._packed_enabled = False
        self._row_calls = 0
        self._packed_row_calls = 0
        self._physical_all_gathers = 0

    @property
    def rank(self) -> int:
        return 0 if self._group is None else self._group.rank()

    @property
    def size(self) -> int:
        return 1 if self._group is None else self._group.size()

    @property
    def packed_agreements_enabled(self) -> bool:
        return self._packed_enabled

    @property
    def packed_agreement_attestation(self) -> PackedAgreementAttestation:
        return PackedAgreementAttestation(
            enabled=self._packed_enabled,
            row_calls=self._row_calls,
            packed_row_calls=self._packed_row_calls,
            legacy_row_calls=self._row_calls - self._packed_row_calls,
            physical_all_gathers=self._physical_all_gathers,
        )

    def activate_packed_agreements(self) -> None:
        """Enable fixed-row packing only after legacy request setup agrees."""

        self._packed_enabled = True

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

    def _gather_rows(self, row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        self._row_calls += 1
        if self._group is not None:
            self._physical_all_gathers += 1
        return self._all_gather_rows(row)

    def agree_packed(
        self,
        name: str,
        *,
        local_success: bool,
        error_fingerprint: int,
        payload: tuple[int, ...] = (),
    ) -> PackedRankAgreement:
        """Agree status, error, and payload in one universal int32 row."""

        if not self._packed_enabled:
            raise DSparkConfigurationError(
                "Kimi K3 packed agreements were not rank-agreed during setup"
            )
        if type(local_success) is not bool:
            raise TypeError("packed agreement success must be boolean")
        if (
            type(error_fingerprint) is not int
            or not 0 <= error_fingerprint <= 0x7FFFFFFF
        ):
            raise ValueError("packed agreement error fingerprint must fit int32")
        if len(payload) > PACKED_AGREEMENT_PAYLOAD_WIDTH:
            raise ValueError("packed agreement payload exceeds the fixed row")
        if any(
            type(value) is not int or not -0x80000000 <= value <= 0x7FFFFFFF
            for value in payload
        ):
            raise ValueError("packed agreement payload values must fit int32")

        tag = _packed_operation_tag(name)
        row = (
            tag,
            int(local_success),
            error_fingerprint,
            len(payload),
            *payload,
            *((0,) * (PACKED_AGREEMENT_PAYLOAD_WIDTH - len(payload))),
        )
        assert len(row) == PACKED_AGREEMENT_ROW_WIDTH
        self._packed_row_calls += 1
        rows = self._gather_rows(row)
        first = rows[0]
        if (
            len(first) != PACKED_AGREEMENT_ROW_WIDTH
            or first[0] != tag
            or first[1] not in (0, 1)
            or not 0 <= first[2] <= 0x7FFFFFFF
            or not 0 <= first[3] <= PACKED_AGREEMENT_PAYLOAD_WIDTH
            or any(len(peer) != PACKED_AGREEMENT_ROW_WIDTH for peer in rows[1:])
            or any(peer != first for peer in rows[1:])
        ):
            return PackedRankAgreement(None, None, None)
        payload_width = first[3]
        if any(first[4 + payload_width :]):
            return PackedRankAgreement(None, None, None)
        return PackedRankAgreement(
            success=first[1] == 1,
            error_fingerprint=first[2],
            payload=first[4 : 4 + payload_width],
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
        rows = self._gather_rows((int(valid), *payload))
        rank_zero = rows[0]
        if not self._rank_zero_proposal_recovery:
            if rank_zero[0] != 1 or any(row != rank_zero for row in rows[1:]):
                return None
            return rank_zero[1:]
        # Proposal evaluation is non-mutating, and every draft commit later
        # appends the target posterior rather than its rank-local proposal.
        # Rank zero may therefore select the target verification block even
        # when valid peer proposals differ.  Every peer must still own a valid
        # materialized draft round so the post-target draft commit is safe.
        if rank_zero[0] != 1 or any(row[0] != 1 for row in rows[1:]):
            return None
        return rank_zero[1:]

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
        rows = self._gather_rows(row)
        first = rows[0]
        if first[0] != 1 or any(peer != first for peer in rows[1:]):
            return None
        return first[1], first[2]

    def agree_stage_success(self, local_success: bool) -> bool | None:
        """Return a unanimous result, or ``None`` when rank outcomes differ."""

        rows = self._gather_rows((int(local_success),))
        first = rows[0][0]
        if any(row[0] != first for row in rows[1:]):
            return None
        return first == 1

    def agree_token(self, local_token: int | None) -> int | None:
        valid = type(local_token) is int and local_token >= 0
        rows = self._gather_rows(
            (int(valid), local_token if valid and local_token is not None else -1)
        )
        first = rows[0]
        if first[0] != 1 or any(row != first for row in rows[1:]):
            return None
        return first[1]


@dataclass(frozen=True)
class DSparkRoundTelemetry:
    """Timing and acceptance outcome for one speculative/fallback round."""

    round_index: int
    rank: int
    draft_ms: float
    target_verify_ms: float
    target_commit_ms: float
    draft_commit_ms: float
    collective_ms: float
    proposed: int
    accepted: int
    emitted: int
    fallback: bool
    error: str | None
    prelaunch_ms: float = 0.0
    prelaunch_submitted: bool = False
    prelaunch_used: bool = False


@dataclass(frozen=True)
class DSparkConfidenceCaptureRequest:
    """Prompt-safe request identity shared by the rank-zero recorder."""

    prompt_id: str
    context_tokens: int
    context_bucket: str


@dataclass(frozen=True)
class DSparkConfidenceObservation:
    """Rank-zero copy of one fully materialized and agreed proposal."""

    confidence_logits: tuple[float, ...]
    proposal_sha256: str


class DSparkConfidenceRecorder(Protocol):
    def record(
        self,
        telemetry: DSparkRoundTelemetry,
        observation: DSparkConfidenceObservation | None,
        *,
        total_step_ms: float,
    ) -> None: ...

    def finalize(self, *, complete: bool) -> None: ...


def _token_sequence_sha256(tokens: Sequence[int], *, domain: bytes) -> str:
    digest = hashlib.sha256(domain)
    digest.update(len(tokens).to_bytes(8, "big"))
    for token in tokens:
        if type(token) is not int or not 0 <= token <= 0x7FFFFFFF:
            raise ValueError(
                "Kimi K3 DSpark identity tokens must fit non-negative int32"
            )
        digest.update(token.to_bytes(4, "big"))
    return digest.hexdigest()


def dspark_prompt_identity(tokens: Sequence[int]) -> str:
    """Match the K3 TP2 uint32-little-endian token digest contract."""

    if not tokens:
        raise ValueError("Kimi K3 DSpark prompt identity requires at least one token")
    digest = hashlib.sha256()
    for token in tokens:
        value = int(token)
        if not 0 <= value < 2**32:
            raise ValueError("Kimi K3 DSpark prompt token id is outside uint32")
        digest.update(value.to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def dspark_context_bucket(context_tokens: int) -> str:
    """Return a deterministic power-of-two context bucket through 1M."""

    if (
        type(context_tokens) is not int
        or not 1 <= context_tokens <= KIMI_K3_MAX_CONTEXT_LENGTH
    ):
        raise ValueError("Kimi K3 DSpark context token count is out of range")
    ceiling = 512
    while ceiling < context_tokens and ceiling < KIMI_K3_MAX_CONTEXT_LENGTH:
        ceiling *= 2
    return f"le_{ceiling}"


def dspark_confidence_capture_request(
    tokens: Sequence[int],
) -> DSparkConfidenceCaptureRequest:
    logical_tokens = tuple(tokens)
    return DSparkConfidenceCaptureRequest(
        prompt_id=dspark_prompt_identity(logical_tokens),
        context_tokens=len(logical_tokens),
        context_bucket=dspark_context_bucket(len(logical_tokens)),
    )


@final
class DSparkConfidenceJSONLRecorder:
    """Buffer labeled rank-zero evidence and append once at request end."""

    def __init__(
        self,
        config: DSparkConfidenceCaptureConfig,
        request: DSparkConfidenceCaptureRequest,
        *,
        verify_width: int,
        target_route_top_k: int,
    ):
        if verify_width != DSPARK_CONSERVATIVE_VERIFY_WIDTH:
            raise DSparkConfigurationError(
                "DSpark confidence recorder requires conservative verify width 3"
            )
        if target_route_top_k != 8:
            raise DSparkConfigurationError(
                "DSpark confidence recorder requires an attested target route top-k of 8"
            )
        self._config = config
        self._request = request
        self._verify_width = verify_width
        self._target_route_top_k = target_route_top_k
        self._run_id = uuid.uuid4().hex
        self._disabled = False
        self._finalized = False
        self._rounds_seen = 0
        self._ignored_tail_rounds = 0
        self._dropped_rounds = 0
        self._records: list[
            tuple[DSparkRoundTelemetry, DSparkConfidenceObservation, float]
        ] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    def _append(self, payload: bytes) -> None:
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._config.jsonl_path, flags, 0o600)
        locked = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("confidence capture target is not a regular file")
            if metadata.st_nlink != 1:
                raise OSError("confidence capture target is hard-linked")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise OSError("confidence capture target grants group or other access")
            initial_size = metadata.st_size
            try:
                view = memoryview(payload)
                offset = 0
                while offset < len(view):
                    written = os.write(descriptor, view[offset:])
                    if written <= 0:
                        raise OSError("confidence capture append made no progress")
                    offset += written
            except BaseException:
                try:
                    os.ftruncate(descriptor, initial_size)
                except OSError as rollback_error:
                    raise OSError(
                        "confidence capture append failed and could not be rolled back"
                    ) from rollback_error
                raise
        finally:
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def record(
        self,
        telemetry: DSparkRoundTelemetry,
        observation: DSparkConfidenceObservation | None,
        *,
        total_step_ms: float,
    ) -> None:
        if self._disabled or self._finalized:
            return
        self._rounds_seen += 1
        if telemetry.rank != 0:
            self._dropped_rounds += 1
            return
        if (
            telemetry.proposed == 0
            and not telemetry.fallback
            and telemetry.error is None
        ):
            self._ignored_tail_rounds += 1
            return
        gamma = self._verify_width - 1
        if (
            observation is None
            or telemetry.proposed != gamma
            or telemetry.fallback
            or telemetry.error is not None
        ):
            self._dropped_rounds += 1
            return
        try:
            _confidence_tuple(observation.confidence_logits, expected=gamma)
            if not math.isfinite(total_step_ms) or total_step_ms < 0.0:
                raise ValueError("confidence capture total step time is invalid")
            self._records.append((telemetry, observation, total_step_ms))
        except Exception:
            self._dropped_rounds += 1
            _log_nonfatal_warning(
                "Kimi K3 DSpark confidence row was dropped; inference continues"
            )

    def _round_payload(
        self,
        telemetry: DSparkRoundTelemetry,
        observation: DSparkConfidenceObservation,
        total_step_ms: float,
    ) -> dict[str, object]:
        return {
            "schema": "k3-dspark-confidence-capture-v3",
            "session_id": self._config.session_id,
            "run_id": self._run_id,
            "prompt_id": self._request.prompt_id,
            "context_bucket": self._request.context_bucket,
            "context_tokens": self._request.context_tokens,
            "round_index": telemetry.round_index,
            "verify_width": self._verify_width,
            "target_route_top_k": self._target_route_top_k,
            "confidence_logits": list(observation.confidence_logits),
            "proposal_sha256": observation.proposal_sha256,
            "proposed": telemetry.proposed,
            "accepted_prefix": telemetry.accepted,
            "emitted": telemetry.emitted,
            "draft_ms": telemetry.draft_ms,
            "target_verify_ms": telemetry.target_verify_ms,
            "target_commit_ms": telemetry.target_commit_ms,
            "draft_commit_ms": telemetry.draft_commit_ms,
            "collective_ms": telemetry.collective_ms,
            "total_step_ms": total_step_ms,
            "fallback": False,
            "error": None,
        }

    def finalize(self, *, complete: bool) -> None:
        if self._disabled or self._finalized:
            return
        self._finalized = True
        end: dict[str, object] = {
            "schema": "k3-dspark-confidence-request-end-v3",
            "session_id": self._config.session_id,
            "run_id": self._run_id,
            "prompt_id": self._request.prompt_id,
            "context_bucket": self._request.context_bucket,
            "context_tokens": self._request.context_tokens,
            "verify_width": self._verify_width,
            "target_route_top_k": self._target_route_top_k,
            "complete": bool(complete),
            "rounds_seen": self._rounds_seen,
            "captured_rounds": len(self._records),
            "ignored_tail_rounds": self._ignored_tail_rounds,
            "dropped_rounds": self._dropped_rounds,
            "request_buffered_locked_append": True,
            "round_timing_excludes_request_flush": True,
        }
        payloads = [
            self._round_payload(telemetry, observation, total_step_ms)
            for telemetry, observation, total_step_ms in self._records
        ]
        payloads.append(end)
        try:
            encoded = b"".join(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
                for payload in payloads
            )
            self._append(encoded)
        except Exception:
            self._disabled = True
            _log_nonfatal_warning(
                "Kimi K3 DSpark confidence request flush failed and was disabled"
            )


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
        f"target_commit_ms={telemetry.target_commit_ms:.3f}, "
        f"draft_commit_ms={telemetry.draft_commit_ms:.3f}, "
        f"collective_ms={telemetry.collective_ms:.3f}, "
        f"proposed={telemetry.proposed}, "
        f"accepted={telemetry.accepted}, "
        f"emitted={telemetry.emitted}, "
        f"fallback={telemetry.fallback}, "
        f"error={telemetry.error!r}, "
        f"prelaunch_ms={telemetry.prelaunch_ms:.3f}, "
        f"prelaunch_submitted={telemetry.prelaunch_submitted}, "
        f"prelaunch_used={telemetry.prelaunch_used}"
    )


def _log_nonfatal_warning(message: str) -> None:
    with contextlib.suppress(Exception):
        logger.opt(exception=True).warning(message)


def _nonthrowing_telemetry_clock(
    clock: Callable[[], float],
) -> Callable[[], float]:
    """Keep diagnostic timing failures out of distributed control flow."""

    def read() -> float:
        try:
            return float(clock())
        except Exception:
            _log_nonfatal_warning("Kimi K3 DSpark telemetry clock failed")
            return 0.0

    return read


def _token_tuple(tokens: Sequence[int], *, expected: int, name: str) -> tuple[int, ...]:
    values = tuple(tokens)
    if len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} tokens")
    if any(type(token) is not int or token < 0 for token in values):
        raise ValueError(f"{name} must contain non-negative integer token ids")
    return values


def _confidence_tuple(
    logits: Sequence[object] | None,
    *,
    expected: int,
) -> tuple[float, ...]:
    if logits is None:
        raise ValueError("MLX-LM DSpark confidence logits are unavailable")
    values = tuple(logits)
    if len(values) != expected:
        raise ValueError(
            f"DSpark confidence logits must contain exactly {expected} values"
        )
    validated: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("DSpark confidence logits must be finite numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("DSpark confidence logits must be finite numbers")
        validated.append(numeric)
    return tuple(validated)


def _shape_tuple(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(cast(Sequence[object], value))


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


def _error_fingerprint(error: str | None) -> int:
    if error is None:
        return 0
    return (
        int.from_bytes(hashlib.sha256(error.encode()).digest()[:4], "big") & 0x7FFFFFFF
    )


def _token_contract_fingerprint(*sequences: Sequence[int]) -> tuple[int, ...]:
    """Return four int32-safe words binding ordered request token contracts."""

    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(len(sequence).to_bytes(8, "big"))
        for token in sequence:
            if type(token) is not int or not 0 <= token <= 0x7FFFFFFF:
                raise ValueError(
                    "Kimi K3 request token ids must fit non-negative int32"
                )
            digest.update(token.to_bytes(4, "big"))
    raw = digest.digest()
    return tuple(
        int.from_bytes(raw[offset : offset + 4], "big") & 0x7FFFFFFF
        for offset in range(0, 16, 4)
    )


def _request_control_fingerprint(
    *,
    max_tokens: int,
    prefill_step_size: int,
    stop_sequences: Sequence[str],
    distributed_progress: bool,
) -> tuple[int, ...]:
    """Bind controls that can change TP graph count or response termination."""

    if type(max_tokens) is not int or not 0 < max_tokens <= KIMI_K3_MAX_CONTEXT_LENGTH:
        raise ValueError("Kimi K3 DSpark max tokens are out of range")
    if (
        type(prefill_step_size) is not int
        or not 0 < prefill_step_size <= KIMI_K3_MAX_CONTEXT_LENGTH
    ):
        raise ValueError("Kimi K3 DSpark prefill step size is out of range")
    if isinstance(stop_sequences, (str, bytes)):
        raise TypeError("Kimi K3 DSpark stop sequences must be a sequence of strings")

    digest = hashlib.sha256(b"kimi-k3-dspark-request-control/v1\0")
    digest.update(max_tokens.to_bytes(8, "big"))
    digest.update(prefill_step_size.to_bytes(8, "big"))
    digest.update(int(distributed_progress).to_bytes(1, "big"))
    digest.update(len(stop_sequences).to_bytes(8, "big"))
    for stop in stop_sequences:
        encoded = stop.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    raw = digest.digest()
    return tuple(
        int.from_bytes(raw[offset : offset + 4], "big") & 0x7FFFFFFF
        for offset in range(0, 16, 4)
    )


@dataclass(frozen=True)
class _PrelaunchedDraftRound:
    """A rank-agreed draft for the next round, already submitted for execution."""

    anchor_token: int
    context_offset: int | None
    prepared: PreparedDraftRound


@dataclass(frozen=True)
class _TailPrelaunch:
    """Outcome of one tail-overlap attempt between acceptance and target commit.

    ``error`` carries a draft-commit failure that must surface through the
    ``draft commit`` agreement at its usual slot.  Otherwise the lazy context
    append succeeded and :meth:`_SplitCommitDraftRound.commit_finalize` still
    has to run there.  ``submitted`` means every rank asynchronously enqueued
    the appended context together with the next proposal graph, so the target
    commit that follows overlaps the executing draft chain.
    """

    error: str | None
    prepared: PreparedDraftRound | None
    submitted: bool
    context_offset: int | None
    next_anchor_token: int
    collective_ms: float


@dataclass
class KimiK3DSparkRoundEngine:
    """Run rank-agreed rounds with pre-commit fallback and commit fail-stop."""

    config: KimiK3DSparkConfig
    draft: ReplicatedDraft
    target: WidthNTarget
    collective: RankAgreement
    terminal_token_ids: tuple[int, ...] = ()
    telemetry_sink: Callable[[DSparkRoundTelemetry], None] | None = None
    confidence_recorder: DSparkConfidenceRecorder | None = None
    clock: Callable[[], float] = time.perf_counter
    _round_index: int = field(default=0, init=False)
    _disabled_reason: str | None = field(default=None, init=False)
    _confidence_observation: DSparkConfidenceObservation | None = field(
        default=None,
        init=False,
    )
    _capture_round_started: float | None = field(default=None, init=False)
    _prelaunched: _PrelaunchedDraftRound | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.clock = _nonthrowing_telemetry_clock(self.clock)
        if self.draft.placement != "replicated":
            raise DSparkConfigurationError(
                "Kimi K3 DSpark must be replicated on every target TP rank"
            )
        if self.draft.verify_width != self.config.verify_width:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark proposer and EXO verify widths do not match"
            )
        if self.confidence_recorder is not None and (
            self.config.confidence_capture is None or self.collective.rank != 0
        ):
            raise DSparkConfigurationError(
                "Kimi K3 DSpark confidence recorder must be rank zero and configured"
            )
        if any(
            type(token_id) is not int or token_id < 0
            for token_id in self.terminal_token_ids
        ):
            raise DSparkConfigurationError(
                "Kimi K3 DSpark terminal token ids must be non-negative integers"
            )

    @property
    def verify_width(self) -> int:
        return self.config.verify_width

    def _publish(self, telemetry: DSparkRoundTelemetry) -> None:
        if self.config.round_telemetry:
            try:
                log_dspark_round(telemetry)
            except Exception:
                _log_nonfatal_warning("Kimi K3 DSpark round logging failed")
        if self.telemetry_sink is not None:
            try:
                self.telemetry_sink(telemetry)
            except Exception:
                _log_nonfatal_warning("Kimi K3 DSpark telemetry callback failed")
        if self.confidence_recorder is not None:
            started = self._capture_round_started
            total_step_ms = (
                0.0 if started is None else max(0.0, (self.clock() - started) * 1000.0)
            )
            try:
                self.confidence_recorder.record(
                    telemetry,
                    self._confidence_observation,
                    total_step_ms=total_step_ms,
                )
            except Exception:
                _log_nonfatal_warning("Kimi K3 DSpark confidence recorder failed")
            finally:
                self._confidence_observation = None
                self._capture_round_started = None

    def _begin_capture_round(self) -> None:
        self._confidence_observation = None
        self._capture_round_started = (
            None if self.confidence_recorder is None else self.clock()
        )

    def _agree_stage(
        self,
        name: str,
        local_success: bool,
        local_error: str | None,
    ) -> tuple[bool | None, int | None, float]:
        stage, fingerprint, _payload, collective_ms = self._agree_stage_payload(
            name,
            local_success,
            local_error,
            (),
        )
        return stage, fingerprint, collective_ms

    def _agree_stage_payload(
        self,
        name: str,
        local_success: bool,
        local_error: str | None,
        payload: tuple[int | None, ...],
    ) -> tuple[
        bool | None,
        int | None,
        tuple[int | None, ...] | None,
        float,
    ]:
        started = self.clock()
        if self.config.packed_agreements:
            agreement = self.collective.agree_packed(
                f"stage:{name}",
                local_success=local_success,
                error_fingerprint=_error_fingerprint(local_error),
                payload=tuple(-1 if value is None else value for value in payload),
            )
            agreed_payload = (
                None
                if agreement.payload is None
                else tuple(
                    None if value == -1 else value for value in agreement.payload
                )
            )
            return (
                agreement.success,
                agreement.error_fingerprint,
                agreed_payload,
                (self.clock() - started) * 1000.0,
            )
        stage = self.collective.agree_stage_success(local_success)
        fingerprint = self.collective.agree_token(_error_fingerprint(local_error))
        agreed_payload = tuple(self.collective.agree_token(value) for value in payload)
        return (
            stage,
            fingerprint,
            agreed_payload,
            (self.clock() - started) * 1000.0,
        )

    def _discard_prelaunched(
        self,
        prelaunched: _PrelaunchedDraftRound | None = None,
    ) -> None:
        """Cancel a prelaunched draft locally; already-enqueued GPU work drains.

        Discard decisions derive exclusively from rank-agreed integers, so all
        ranks discard identically without an extra collective.
        """

        if prelaunched is None:
            prelaunched = self._prelaunched
            self._prelaunched = None
        if prelaunched is None:
            return
        try:
            prelaunched.prepared.cancel()
        except Exception:
            _log_nonfatal_warning("Kimi K3 DSpark prelaunched draft cancel failed")

    def _take_prelaunched(self, anchor_token: int) -> _PrelaunchedDraftRound | None:
        prelaunched = self._prelaunched
        if prelaunched is None:
            return None
        self._prelaunched = None
        if prelaunched.anchor_token != anchor_token:
            # The driver anchor and the prelaunched anchor were both
            # rank-agreed, so every rank observes the same mismatch.
            self._discard_prelaunched(prelaunched)
            return None
        return prelaunched

    def _cancel_before_fallback(
        self,
        *,
        draft_round: DraftRound | PreparedDraftRound | None,
        target_round: (
            TargetRound | PreparedTargetVerification | BuiltTargetVerification | None
        ),
        already_uncertain: bool,
    ) -> float:
        errors: list[str] = []
        if already_uncertain:
            errors.append("target verification cancellation was uncertain")
        if target_round is not None:
            try:
                target_round.cancel()
            except Exception as error:
                errors.append(f"target cancel failed: {type(error).__name__}: {error}")
        if draft_round is not None:
            try:
                draft_round.cancel()
            except Exception as error:
                errors.append(f"draft cancel failed: {type(error).__name__}: {error}")

        local_error = "; ".join(errors) if errors else None
        outcome, fingerprint, collective_ms = self._agree_stage(
            "pre-commit cancellation",
            local_error is None,
            local_error,
        )
        if outcome is not True or fingerprint != 0:
            detail = (
                "outcomes disagreed across ranks"
                if outcome is None or fingerprint is None
                else "failed on every rank"
            )
            raise DSparkDistributedStateError(
                f"Kimi K3 DSpark pre-commit cancellation {detail}; "
                "cache state cannot be recovered safely"
            ) from None
        return collective_ms

    def _cancel_draft_after_fatal_target_commit(
        self,
        draft_round: DraftRound | None,
    ) -> float:
        local_error: str | None = None
        if draft_round is not None:
            try:
                draft_round.cancel()
            except Exception as error:
                local_error = f"draft cancel failed: {type(error).__name__}: {error}"
        _, _, collective_ms = self._agree_stage(
            "fatal target-commit draft cancellation",
            local_error is None,
            local_error,
        )
        return collective_ms

    def _ordinary_fallback(
        self,
        anchor_token: int,
        *,
        draft_ms: float,
        target_verify_ms: float,
        target_commit_ms: float,
        draft_commit_ms: float,
        collective_ms: float,
        proposed: int,
        error: str,
        planned_tail: bool = False,
    ) -> DSparkRoundResult:
        self._discard_prelaunched()
        self._disabled_reason = error
        fallback_error = None if planned_tail else error

        anchor_started = self.clock()
        agreed_anchor = self.collective.agree_token(anchor_token)
        collective_ms += (self.clock() - anchor_started) * 1000.0
        if agreed_anchor != anchor_token:
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark ordinary fallback anchor disagreed across ranks; "
                "no target TP graph was built"
            ) from None

        local_plan: OrdinaryDecodePlan | None = None
        local_preflight_error: str | None = None
        try:
            local_plan = self.target.preflight_ordinary(anchor_token)
        except Exception as preflight_error:
            local_preflight_error = (
                "ordinary target preflight failed: "
                f"{type(preflight_error).__name__}: {preflight_error}"
            )

        (
            preflight_outcome,
            preflight_fingerprint,
            agreed_preflight_payload,
            agreement_ms,
        ) = self._agree_stage_payload(
            "ordinary target preflight",
            local_preflight_error is None,
            local_preflight_error,
            (None if local_plan is None else local_plan.agreement_code,),
        )
        collective_ms += agreement_ms
        agreed_plan = (
            None if agreed_preflight_payload is None else agreed_preflight_payload[0]
        )
        if (
            preflight_outcome is not True
            or preflight_fingerprint != 0
            or agreed_plan is None
            or local_plan is None
            or agreed_plan != local_plan.agreement_code
        ):
            fallback_error = (
                f"{error}; "
                f"{local_preflight_error or 'ordinary target plan disagreed across ranks'}"
            )
            telemetry = DSparkRoundTelemetry(
                round_index=self._round_index,
                rank=self.collective.rank,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                target_commit_ms=target_commit_ms,
                draft_commit_ms=draft_commit_ms,
                collective_ms=collective_ms,
                proposed=proposed,
                accepted=0,
                emitted=0,
                fallback=not planned_tail,
                error=fallback_error,
            )
            self._publish(telemetry)
            self._round_index += 1
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark ordinary fallback readiness disagreed or failed; "
                "no target TP graph was executed"
            ) from None

        prepared: PreparedOrdinaryDecode | None = None
        local_build_error: str | None = None
        try:
            prepared = self.target.prepare_ordinary(anchor_token, local_plan)
        except Exception as build_error:
            local_build_error = (
                "ordinary target graph build failed: "
                f"{type(build_error).__name__}: {build_error}"
            )

        build_outcome, build_fingerprint, agreement_ms = self._agree_stage(
            "ordinary target graph build",
            local_build_error is None,
            local_build_error,
        )
        collective_ms += agreement_ms
        if build_outcome is not True or build_fingerprint != 0 or prepared is None:
            fallback_error = (
                f"{error}; "
                f"{local_build_error or 'ordinary target graph build disagreed'}"
            )
            telemetry = DSparkRoundTelemetry(
                round_index=self._round_index,
                rank=self.collective.rank,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                target_commit_ms=target_commit_ms,
                draft_commit_ms=draft_commit_ms,
                collective_ms=collective_ms,
                proposed=proposed,
                accepted=0,
                emitted=0,
                fallback=not planned_tail,
                error=fallback_error,
            )
            self._publish(telemetry)
            self._round_index += 1
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark ordinary fallback graph build disagreed or failed; "
                "request-local target cache state cannot continue safely"
            ) from None

        local_token: int | None = None
        local_decode_error: str | None = None
        try:
            # Every rank has now built and shape-checked the same collective
            # graph. Materialization is the first point at which MLX may enter
            # the target TP all-gather.
            local_token = prepared.materialize()
        except Exception as decode_error:
            local_decode_error = (
                "ordinary target fallback failed: "
                f"{type(decode_error).__name__}: {decode_error}"
            )

        decode_outcome, error_fingerprint, agreement_ms = self._agree_stage(
            "ordinary target materialization",
            local_decode_error is None,
            local_decode_error,
        )
        collective_ms += agreement_ms
        if decode_outcome is not True or error_fingerprint != 0:
            fallback_error = (
                f"{error}; {local_decode_error or 'rank outcome disagreed'}"
            )
            telemetry = DSparkRoundTelemetry(
                round_index=self._round_index,
                rank=self.collective.rank,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                target_commit_ms=target_commit_ms,
                draft_commit_ms=draft_commit_ms,
                collective_ms=collective_ms,
                proposed=proposed,
                accepted=0,
                emitted=0,
                fallback=not planned_tail,
                error=fallback_error,
            )
            self._publish(telemetry)
            self._round_index += 1
            outcome = (
                "disagreed across ranks"
                if decode_outcome is None or error_fingerprint is None
                else "failed on every rank"
            )
            raise DSparkDistributedStateError(
                f"Kimi K3 DSpark ordinary fallback {outcome}; "
                "target cache state cannot continue safely"
            ) from None

        agreed_token_started = self.clock()
        agreed_token = self.collective.agree_token(local_token)
        collective_ms += (self.clock() - agreed_token_started) * 1000.0
        if agreed_token is None:
            fallback_error = f"{error}; ordinary fallback token disagreed across ranks"
            telemetry = DSparkRoundTelemetry(
                round_index=self._round_index,
                rank=self.collective.rank,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                target_commit_ms=target_commit_ms,
                draft_commit_ms=draft_commit_ms,
                collective_ms=collective_ms,
                proposed=proposed,
                accepted=0,
                emitted=0,
                fallback=True,
                error=fallback_error,
            )
            self._publish(telemetry)
            self._round_index += 1
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark ordinary fallback token disagreed across ranks"
            ) from None

        emitted_tokens = (agreed_token,)

        telemetry = DSparkRoundTelemetry(
            round_index=self._round_index,
            rank=self.collective.rank,
            draft_ms=draft_ms,
            target_verify_ms=target_verify_ms,
            target_commit_ms=target_commit_ms,
            draft_commit_ms=draft_commit_ms,
            collective_ms=collective_ms,
            proposed=proposed,
            accepted=0,
            emitted=len(emitted_tokens),
            fallback=not planned_tail,
            error=fallback_error,
        )
        self._publish(telemetry)
        self._round_index += 1
        return DSparkRoundResult(emitted_tokens=emitted_tokens, telemetry=telemetry)

    def _maybe_prelaunch_next_draft(
        self,
        *,
        draft_round: DraftRound,
        accepted: int,
        next_anchor_token: int,
        posterior: TargetPosterior,
        remaining: int,
        emitted_tokens: tuple[int, ...],
    ) -> _TailPrelaunch | None:
        """Commit the draft context and submit the next proposal graph early.

        Runs between the acceptance agreement and the target commit so the
        asynchronously submitted draft chain executes on the GPU while the
        CPU resolves speculative target checkpoints.  Returns ``None`` when
        this round cannot legally feed a speculative next round; that decision
        uses only rank-agreed state (emitted tokens, config, remaining budget)
        so every rank takes the same branch without extra collectives.
        """

        remaining_after = remaining - len(emitted_tokens)
        if remaining_after < self.config.verify_width:
            return None
        if self.terminal_token_ids:
            terminal_ids = frozenset(self.terminal_token_ids)
            if any(token in terminal_ids for token in emitted_tokens):
                return None
        if not (
            callable(getattr(draft_round, "commit_build", None))
            and callable(getattr(draft_round, "commit_finalize", None))
        ):
            return None
        split = cast(_SplitCommitDraftRound, cast(object, draft_round))

        gamma = self.config.gamma
        collective_ms = 0.0
        prelaunch_error: str | None = None
        context_offset: int | None = None
        context_arrays: tuple[object, ...] = ()
        try:
            context_offset, context_arrays = split.commit_build(
                accepted,
                next_anchor_token,
                posterior,
            )
        except Exception as error:
            prelaunch_error = (
                f"draft commit failed: {type(error).__name__}: {error}; "
                "DSpark disabled for subsequent rounds"
            )

        (
            preflight_outcome,
            preflight_fingerprint,
            agreed_payload,
            agreement_ms,
        ) = self._agree_stage_payload(
            "draft preflight",
            prelaunch_error is None,
            prelaunch_error,
            (context_offset,),
        )
        collective_ms += agreement_ms
        agreed_offset = None if agreed_payload is None else agreed_payload[0]
        if (
            preflight_outcome is not True
            or preflight_fingerprint != 0
            or agreed_offset is None
            or context_offset is None
            or agreed_offset != context_offset
        ):
            if prelaunch_error is None:
                try:
                    split.commit_finalize(evaluate=True)
                except Exception as error:
                    prelaunch_error = (
                        f"draft commit failed: {type(error).__name__}: {error}; "
                        "DSpark disabled for subsequent rounds"
                    )
            return _TailPrelaunch(
                error=prelaunch_error,
                prepared=None,
                submitted=False,
                context_offset=context_offset,
                next_anchor_token=next_anchor_token,
                collective_ms=collective_ms,
            )

        prepared: PreparedDraftRound | None = None
        graph_error: str | None = None
        try:
            # MLX is lazy: build the next proposal graph over the appended
            # (still unevaluated) draft context without materializing anything.
            prepared = self.draft.prepare_round(next_anchor_token, gamma)
        except Exception as error:
            graph_error = f"draft graph build failed: {type(error).__name__}: {error}"
        if graph_error is None and not callable(getattr(prepared, "submit", None)):
            graph_error = "draft graph build failed: prelaunch submit unsupported"

        graph_outcome, graph_fingerprint, agreement_ms = self._agree_stage(
            "draft graph build",
            graph_error is None,
            graph_error,
        )
        collective_ms += agreement_ms
        if graph_outcome is not True or graph_fingerprint != 0 or prepared is None:
            if prepared is not None:
                try:
                    prepared.cancel()
                except Exception:
                    _log_nonfatal_warning(
                        "Kimi K3 DSpark prelaunched draft cancellation failed"
                    )
            finalize_error: str | None = None
            try:
                split.commit_finalize(evaluate=True)
            except Exception as error:
                finalize_error = (
                    f"draft commit failed: {type(error).__name__}: {error}; "
                    "DSpark disabled for subsequent rounds"
                )
            # A failed next-graph build is recoverable: the draft context is
            # fully committed, so the next round simply rebuilds legacy-style.
            return _TailPrelaunch(
                error=finalize_error,
                prepared=None,
                submitted=False,
                context_offset=context_offset,
                next_anchor_token=next_anchor_token,
                collective_ms=collective_ms,
            )

        submit_error: str | None = None
        try:
            cast(
                _SubmittablePreparedDraftRound,
                cast(object, prepared),
            ).submit(context_arrays)
        except Exception as error:
            submit_error = (
                f"draft commit failed: {type(error).__name__}: {error}; "
                "DSpark disabled for subsequent rounds"
            )
        if submit_error is None:
            try:
                split.commit_finalize(evaluate=False)
            except Exception as error:
                submit_error = (
                    f"draft commit failed: {type(error).__name__}: {error}; "
                    "DSpark disabled for subsequent rounds"
                )
        if submit_error is not None:
            try:
                prepared.cancel()
            except Exception:
                _log_nonfatal_warning(
                    "Kimi K3 DSpark prelaunched draft cancellation failed"
                )
            try:
                split.commit_finalize(evaluate=True)
            except Exception:
                _log_nonfatal_warning(
                    "Kimi K3 DSpark draft finalize after failed submit failed"
                )
            return _TailPrelaunch(
                error=submit_error,
                prepared=None,
                submitted=False,
                context_offset=context_offset,
                next_anchor_token=next_anchor_token,
                collective_ms=collective_ms,
            )

        return _TailPrelaunch(
            error=None,
            prepared=prepared,
            submitted=True,
            context_offset=context_offset,
            next_anchor_token=next_anchor_token,
            collective_ms=collective_ms,
        )

    def decode_ordinary_tail(self, anchor_token: int) -> DSparkRoundResult:
        """Commit one target token when a full verifier round cannot fit."""

        if type(anchor_token) is not int or anchor_token < 0:
            raise ValueError("anchor_token must be a non-negative integer")
        self._begin_capture_round()
        return self._ordinary_fallback(
            anchor_token,
            draft_ms=0.0,
            target_verify_ms=0.0,
            target_commit_ms=0.0,
            draft_commit_ms=0.0,
            collective_ms=0.0,
            proposed=0,
            error="target-only max-token tail",
            planned_tail=True,
        )

    def decode_round(
        self,
        anchor_token: int,
        *,
        remaining: int | None = None,
    ) -> DSparkRoundResult:
        """Decode one speculative round, or one ordinary token after rollback."""

        if type(anchor_token) is not int or anchor_token < 0:
            raise ValueError("anchor_token must be a non-negative integer")
        self._begin_capture_round()
        if self._disabled_reason is not None:
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=0.0,
                target_verify_ms=0.0,
                target_commit_ms=0.0,
                draft_commit_ms=0.0,
                collective_ms=0.0,
                proposed=0,
                error=f"DSpark disabled after earlier error: {self._disabled_reason}",
            )

        gamma = self.config.gamma
        collective_ms = 0.0
        anchor_started = self.clock()
        agreed_anchor = self.collective.agree_token(anchor_token)
        collective_ms += (self.clock() - anchor_started) * 1000.0
        if agreed_anchor != anchor_token:
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark decode anchor disagreed across ranks; "
                "no proposer or target TP graph was built"
            ) from None
        draft_start = self.clock()
        prelaunched = self._take_prelaunched(anchor_token)
        prelaunch_used = prelaunched is not None
        prepared_draft: PreparedDraftRound | None = None
        local_error: str | None = None
        agreed_context_offset: int | None = None
        if prelaunched is not None:
            # The previous round rank-agreed this draft's readiness and graph
            # build at its tail and asynchronously submitted the graphs;
            # materialization below only waits on the in-flight evaluation.
            prepared_draft = prelaunched.prepared
            agreed_context_offset = prelaunched.context_offset
        else:
            local_context_offset: int | None = None
            try:
                local_context_offset = self.draft.preflight_round(
                    anchor_token, gamma
                )
            except Exception as error:
                local_error = (
                    f"draft preflight failed: {type(error).__name__}: {error}"
                )

            (
                preflight_outcome,
                preflight_fingerprint,
                agreed_preflight_payload,
                agreement_ms,
            ) = self._agree_stage_payload(
                "draft preflight",
                local_error is None,
                local_error,
                (local_context_offset,),
            )
            collective_ms += agreement_ms
            agreed_context_offset = (
                None
                if agreed_preflight_payload is None
                else agreed_preflight_payload[0]
            )
            if (
                preflight_outcome is not True
                or preflight_fingerprint != 0
                or agreed_context_offset is None
                or local_context_offset is None
                or agreed_context_offset != local_context_offset
            ):
                draft_ms = (self.clock() - draft_start) * 1000.0
                return self._ordinary_fallback(
                    anchor_token,
                    draft_ms=draft_ms,
                    target_verify_ms=0.0,
                    target_commit_ms=0.0,
                    draft_commit_ms=0.0,
                    collective_ms=collective_ms,
                    proposed=0,
                    error=local_error
                    or "DSpark draft readiness disagreed across ranks",
                )

            graph_error: str | None = None
            try:
                # MLX is lazy: this constructs the proposer and borrowed
                # target-head graph, but must not call mx.eval()/tolist() yet.
                prepared_draft = self.draft.prepare_round(anchor_token, gamma)
            except Exception as error:
                graph_error = (
                    f"draft graph build failed: {type(error).__name__}: {error}"
                )

            graph_outcome, graph_fingerprint, agreement_ms = self._agree_stage(
                "draft graph build",
                graph_error is None,
                graph_error,
            )
            collective_ms += agreement_ms
            if (
                graph_outcome is not True
                or graph_fingerprint != 0
                or prepared_draft is None
            ):
                collective_ms += self._cancel_before_fallback(
                    draft_round=prepared_draft,
                    target_round=None,
                    already_uncertain=False,
                )
                draft_ms = (self.clock() - draft_start) * 1000.0
                return self._ordinary_fallback(
                    anchor_token,
                    draft_ms=draft_ms,
                    target_verify_ms=0.0,
                    target_commit_ms=0.0,
                    draft_commit_ms=0.0,
                    collective_ms=collective_ms,
                    proposed=0,
                    error=graph_error
                    or "DSpark draft graph build disagreed across ranks",
                )
        assert prepared_draft is not None

        draft_round: DraftRound | None = None
        local_block: tuple[int, ...] | None = None
        local_confidence: tuple[float, ...] | None = None
        local_error = None
        try:
            # This is the first operation allowed to materialize the borrowed
            # target vocabulary head and therefore enter its TP all-gather.
            draft_round = prepared_draft.materialize()
            proposals = _token_tuple(
                draft_round.proposal_tokens,
                expected=gamma,
                name="DSpark proposal",
            )
            local_block = (anchor_token, *proposals)
        except Exception as error:
            local_error = f"draft failed: {type(error).__name__}: {error}"
        if local_block is not None and self.confidence_recorder is not None:
            raw_confidence = (
                None if draft_round is None else draft_round.confidence_logits
            )
            if raw_confidence is not None:
                try:
                    local_confidence = _confidence_tuple(
                        raw_confidence,
                        expected=gamma,
                    )
                except Exception:
                    _log_nonfatal_warning(
                        "Kimi K3 DSpark confidence conversion failed; "
                        "inference continues"
                    )
        draft_ms = (self.clock() - draft_start) * 1000.0

        proposal_agreement_started = self.clock()
        agreed_block = self.collective.agree_proposal_block(
            local_block,
            self.config.verify_width,
        )
        collective_ms += (self.clock() - proposal_agreement_started) * 1000.0
        recovered_mismatch = (
            local_block is not None
            and agreed_block is not None
            and local_block != agreed_block
        )
        recovery_disabled = (
            recovered_mismatch and not self.config.rank_zero_proposal_recovery
        )
        if agreed_block is None or recovery_disabled:
            collective_ms += self._cancel_before_fallback(
                draft_round=draft_round or prepared_draft,
                target_round=None,
                already_uncertain=False,
            )
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=draft_ms,
                target_verify_ms=0.0,
                target_commit_ms=0.0,
                draft_commit_ms=0.0,
                collective_ms=collective_ms,
                proposed=0 if local_block is None else len(local_block) - 1,
                error=local_error
                or (
                    "DSpark proposal recovery is disabled"
                    if recovery_disabled
                    else "DSpark proposal tokens disagreed across ranks"
                ),
            )
        if recovered_mismatch:
            assert local_block is not None
            _log_nonfatal_warning(
                "Kimi K3 DSpark recovered rank-local proposal mismatch: "
                f"rank={self.collective.rank}, "
                f"context_offset={agreed_context_offset}, "
                "local_proposal_sha256="
                f"{_token_sequence_sha256(local_block[1:], domain=b'exo-k3-dspark-local-proposal/v1\0')}, "
                "authoritative_proposal_sha256="
                f"{_token_sequence_sha256(agreed_block[1:], domain=b'exo-k3-dspark-authoritative-proposal/v1\0')}"
            )

        target_start = self.clock()
        prepared_target: PreparedTargetVerification | None = None
        target_error: str | None = None
        target_cancel_uncertain = False
        try:
            prepared_target = self.target.prepare_verification(agreed_block)
        except Exception as error:
            target_error = (
                f"target transaction prepare failed: {type(error).__name__}: {error}"
            )
            target_cancel_uncertain = isinstance(error, DSparkCancellationError)

        (
            prepare_outcome,
            prepare_fingerprint,
            agreed_prepare_payload,
            agreement_ms,
        ) = self._agree_stage_payload(
            "target transaction prepare",
            target_error is None,
            target_error,
            (
                None if prepared_target is None else prepared_target.initial_offset,
                None if prepared_target is None else prepared_target.mode_code,
            ),
        )
        collective_ms += agreement_ms
        agreed_target_offset = (
            None if agreed_prepare_payload is None else agreed_prepare_payload[0]
        )
        agreed_target_mode = (
            None if agreed_prepare_payload is None else agreed_prepare_payload[1]
        )
        if (
            prepare_outcome is not True
            or prepare_fingerprint != 0
            or agreed_target_offset is None
            or prepared_target is None
            or agreed_target_offset != prepared_target.initial_offset
            or agreed_target_mode != prepared_target.mode_code
        ):
            collective_ms += self._cancel_before_fallback(
                draft_round=draft_round,
                target_round=prepared_target,
                already_uncertain=target_cancel_uncertain,
            )
            target_verify_ms = (self.clock() - target_start) * 1000.0
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                target_commit_ms=0.0,
                draft_commit_ms=0.0,
                collective_ms=collective_ms,
                proposed=gamma,
                error=target_error
                or "DSpark target transaction readiness disagreed across ranks",
            )

        built_target: BuiltTargetVerification | None = None
        target_error = None
        try:
            # Build the lazy target graph and validate all output shapes before
            # any rank is allowed to materialize its target TP collectives.
            built_target = prepared_target.build()
        except Exception as error:
            target_error = (
                f"target verification graph build failed: "
                f"{type(error).__name__}: {error}"
            )
            target_cancel_uncertain = isinstance(error, DSparkCancellationError)

        build_outcome, build_fingerprint, agreement_ms = self._agree_stage(
            "target verification graph build",
            target_error is None,
            target_error,
        )
        collective_ms += agreement_ms
        if build_outcome is not True or build_fingerprint != 0 or built_target is None:
            collective_ms += self._cancel_before_fallback(
                draft_round=draft_round,
                target_round=built_target or prepared_target,
                already_uncertain=target_cancel_uncertain,
            )
            target_verify_ms = (self.clock() - target_start) * 1000.0
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                target_commit_ms=0.0,
                draft_commit_ms=0.0,
                collective_ms=collective_ms,
                proposed=gamma,
                error=target_error
                or "DSpark target graph build disagreed across ranks",
            )

        target_round: TargetRound | None = None
        posterior: TargetPosterior | None = None
        local_boundary: int | None = None
        local_next_token: int | None = None
        target_error = None
        try:
            # All ranks have agreed on a valid graph. This materialization is
            # the first target-verification TP collective boundary.
            target_round = built_target.materialize()
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
            if self.terminal_token_ids:
                provisional_tokens = (
                    *agreed_block[1 : local_boundary + 1],
                    posterior_tokens[local_boundary],
                )
                terminal_ids = frozenset(self.terminal_token_ids)
                terminal_index = next(
                    (
                        index
                        for index, token in enumerate(provisional_tokens)
                        if token in terminal_ids
                    ),
                    None,
                )
                if terminal_index is not None:
                    # Reinterpret an accepted terminal proposal as the bonus at
                    # that boundary so neither target nor draft commits beyond
                    # the first visible EOS token.
                    local_boundary = terminal_index
            local_next_token = posterior_tokens[local_boundary]
        except Exception as error:
            target_error = f"target verify failed: {type(error).__name__}: {error}"
            target_cancel_uncertain = isinstance(error, DSparkCancellationError)
        target_verify_ms = (self.clock() - target_start) * 1000.0

        acceptance_started = self.clock()
        agreed_acceptance = self.collective.agree_acceptance(
            local_boundary,
            local_next_token,
            gamma,
        )
        collective_ms += (self.clock() - acceptance_started) * 1000.0
        if agreed_acceptance is None:
            collective_ms += self._cancel_before_fallback(
                draft_round=draft_round,
                target_round=target_round or built_target,
                already_uncertain=target_cancel_uncertain,
            )
            return self._ordinary_fallback(
                anchor_token,
                draft_ms=draft_ms,
                target_verify_ms=target_verify_ms,
                target_commit_ms=0.0,
                draft_commit_ms=0.0,
                collective_ms=collective_ms,
                proposed=gamma,
                error=target_error
                or "DSpark acceptance boundary disagreed across ranks",
            )

        accepted, next_anchor_token = agreed_acceptance
        assert target_round is not None
        assert posterior is not None
        if local_confidence is not None:
            self._confidence_observation = DSparkConfidenceObservation(
                confidence_logits=local_confidence,
                proposal_sha256=_token_sequence_sha256(
                    agreed_block[1:],
                    domain=b"exo-k3-dspark-proposal/v1\0",
                ),
            )
        emitted_tokens = (
            *agreed_block[1 : accepted + 1],
            next_anchor_token,
        )

        # Before the CPU-bound target commit, optionally commit the draft
        # context and submit the next round's proposal graph so the GPU is
        # busy while python resolves speculative target checkpoints.  The
        # eligibility branch is a pure function of rank-agreed state.
        tail: _TailPrelaunch | None = None
        prelaunch_ms = 0.0
        assert draft_round is not None
        if self.config.tail_overlap and remaining is not None:
            prelaunch_started = self.clock()
            tail = self._maybe_prelaunch_next_draft(
                draft_round=draft_round,
                accepted=accepted,
                next_anchor_token=next_anchor_token,
                posterior=posterior,
                remaining=remaining,
                emitted_tokens=emitted_tokens,
            )
            prelaunch_ms = (self.clock() - prelaunch_started) * 1000.0
            if tail is not None:
                collective_ms += tail.collective_ms

        # Acceptance is collective before either state commit.  The target is
        # authoritative; its consumed input count is anchor + accepted drafts.
        target_commit_error: str | None = None
        target_commit_started = self.clock()
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
        target_commit_ms = (self.clock() - target_commit_started) * 1000.0

        (
            target_commit_consensus,
            target_commit_fingerprint,
            agreement_ms,
        ) = self._agree_stage(
            "target commit",
            target_commit_error is None,
            target_commit_error,
        )
        collective_ms += agreement_ms
        if target_commit_consensus is not True or target_commit_fingerprint != 0:
            if tail is not None and tail.prepared is not None:
                try:
                    tail.prepared.cancel()
                except Exception:
                    _log_nonfatal_warning(
                        "Kimi K3 DSpark prelaunched draft cancellation failed"
                    )
            collective_ms += self._cancel_draft_after_fatal_target_commit(draft_round)
            outcome = (
                "disagreed across ranks"
                if target_commit_consensus is None or target_commit_fingerprint is None
                else "failed on every rank"
            )
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark target commit "
                f"{outcome}; target state cannot be recovered safely"
            ) from None

        draft_commit_error: str | None = None
        draft_commit_ms = 0.0
        if tail is not None:
            # The draft context was already committed (and possibly submitted)
            # by the tail prelaunch; surface its outcome through the same
            # "draft commit" agreement the legacy path uses.
            draft_commit_error = tail.error
        else:
            draft_commit_started = self.clock()
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
            draft_commit_ms = (self.clock() - draft_commit_started) * 1000.0

        (
            draft_commit_consensus,
            draft_commit_fingerprint,
            agreement_ms,
        ) = self._agree_stage(
            "draft commit",
            draft_commit_error is None,
            draft_commit_error,
        )
        collective_ms += agreement_ms
        if draft_commit_consensus is not True or draft_commit_fingerprint != 0:
            outcome = (
                "outcome disagreed across ranks"
                if draft_commit_consensus is None or draft_commit_fingerprint is None
                else "failed on every rank"
            )
            draft_commit_error = (
                f"draft commit {outcome}; DSpark disabled on every rank"
            )
            self._disabled_reason = draft_commit_error
            if tail is not None and tail.prepared is not None:
                try:
                    tail.prepared.cancel()
                except Exception:
                    _log_nonfatal_warning(
                        "Kimi K3 DSpark prelaunched draft cancellation failed"
                    )
        elif tail is not None and tail.submitted and tail.prepared is not None:
            self._prelaunched = _PrelaunchedDraftRound(
                anchor_token=tail.next_anchor_token,
                context_offset=tail.context_offset,
                prepared=tail.prepared,
            )

        telemetry = DSparkRoundTelemetry(
            round_index=self._round_index,
            rank=self.collective.rank,
            draft_ms=draft_ms,
            target_verify_ms=target_verify_ms,
            target_commit_ms=target_commit_ms,
            draft_commit_ms=draft_commit_ms,
            collective_ms=collective_ms,
            proposed=gamma,
            accepted=accepted,
            emitted=len(emitted_tokens),
            fallback=False,
            error=draft_commit_error,
            prelaunch_ms=prelaunch_ms,
            prelaunch_submitted=tail is not None and tail.submitted,
            prelaunch_used=prelaunch_used,
        )
        self._publish(telemetry)
        self._round_index += 1
        return DSparkRoundResult(emitted_tokens=emitted_tokens, telemetry=telemetry)


class _SpeculativeCacheHooks(Protocol):
    def begin_speculative_cache(self, cache: object, width: int) -> object: ...

    def resolve_speculative_cache(self, transaction: object, consumed: int) -> None: ...

    def cancel_speculative_cache(self, transaction: object) -> None: ...


class _TargetPosteriorGraph(Protocol):
    """Lazy target verification graph, validated but not materialized."""

    def materialize(self) -> TargetPosterior: ...


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
        initial_offset: int | None,
        validate_closed: Callable[[int], None] | None,
    ):
        self._hooks = hooks
        self._transaction = transaction
        self._posterior = posterior
        self._initial_offset = initial_offset
        self._validate_closed = validate_closed
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
            if self._validate_closed is not None:
                assert self._initial_offset is not None
                self._validate_closed(self._initial_offset + consumed_input_tokens)
        finally:
            # MLX-LM's hook either commits or cancels on failure.
            self._active = bool(getattr(self._transaction, "active", False))

    def cancel(self) -> None:
        if not self._active:
            return
        try:
            if bool(getattr(self._transaction, "active", True)):
                self._hooks.cancel_speculative_cache(self._transaction)
            if self._validate_closed is not None:
                assert self._initial_offset is not None
                self._validate_closed(self._initial_offset)
        finally:
            self._active = False


def _cancel_replayssm_transaction(
    hooks: _SpeculativeCacheHooks,
    transaction: object,
    *,
    initial_offset: int,
    validate_closed: Callable[[int], None] | None,
) -> None:
    if bool(getattr(transaction, "active", True)):
        hooks.cancel_speculative_cache(transaction)
    if validate_closed is not None:
        validate_closed(initial_offset)


@final
class _BuiltReplaySSMVerification:
    """Own an open transaction and a rank-agreed lazy target graph."""

    def __init__(
        self,
        hooks: _SpeculativeCacheHooks,
        transaction: object,
        graph: _TargetPosteriorGraph,
        initial_offset: int,
        validate_closed: Callable[[int], None] | None,
    ):
        self._hooks = hooks
        self._transaction = transaction
        self._graph = graph
        self._initial_offset = initial_offset
        self._validate_closed = validate_closed
        self._active = True

    def materialize(self) -> TargetRound:
        if not self._active:
            raise RuntimeError("target verification graph is no longer active")
        try:
            posterior = self._graph.materialize()
        except BaseException as verify_error:
            try:
                _cancel_replayssm_transaction(
                    self._hooks,
                    self._transaction,
                    initial_offset=self._initial_offset,
                    validate_closed=self._validate_closed,
                )
            except BaseException as cancel_error:
                raise DSparkCancellationError(
                    "target verification failed and its speculative transaction "
                    f"could not be cancelled: {type(cancel_error).__name__}: "
                    f"{cancel_error}"
                ) from verify_error
            finally:
                self._active = False
            raise
        self._active = False
        return _ReplaySSMTargetRound(
            self._hooks,
            self._transaction,
            posterior,
            self._initial_offset,
            self._validate_closed,
        )

    def cancel(self) -> None:
        if not self._active:
            return
        try:
            _cancel_replayssm_transaction(
                self._hooks,
                self._transaction,
                initial_offset=self._initial_offset,
                validate_closed=self._validate_closed,
            )
        finally:
            self._active = False


@final
class _PreparedReplaySSMVerification:
    """Own an open local transaction before the target graph is constructed."""

    def __init__(
        self,
        hooks: _SpeculativeCacheHooks,
        transaction: object,
        proposal_block: tuple[int, ...],
        plan: TargetVerificationPlan,
        build_verify: Callable[
            [tuple[int, ...], TargetVerificationPlan], _TargetPosteriorGraph
        ],
        initial_offset: int,
        validate_closed: Callable[[int], None] | None,
    ):
        self._hooks = hooks
        self._transaction = transaction
        self._proposal_block = proposal_block
        self._plan = plan
        self._build_verify = build_verify
        self._initial_offset = initial_offset
        self._validate_closed = validate_closed
        self._active = True

    @property
    def initial_offset(self) -> int:
        return self._initial_offset

    @property
    def mode_code(self) -> int:
        return self._plan.agreement_code

    def build(self) -> BuiltTargetVerification:
        if not self._active:
            raise RuntimeError("target speculative transaction is no longer active")
        try:
            graph = self._build_verify(self._proposal_block, self._plan)
        except BaseException as build_error:
            try:
                _cancel_replayssm_transaction(
                    self._hooks,
                    self._transaction,
                    initial_offset=self._initial_offset,
                    validate_closed=self._validate_closed,
                )
            except BaseException as cancel_error:
                raise DSparkCancellationError(
                    "target graph build failed and its speculative transaction "
                    f"could not be cancelled: {type(cancel_error).__name__}: "
                    f"{cancel_error}"
                ) from build_error
            finally:
                self._active = False
            raise
        self._active = False
        return _BuiltReplaySSMVerification(
            self._hooks,
            self._transaction,
            graph,
            self._initial_offset,
            self._validate_closed,
        )

    def cancel(self) -> None:
        if not self._active:
            return
        try:
            _cancel_replayssm_transaction(
                self._hooks,
                self._transaction,
                initial_offset=self._initial_offset,
                validate_closed=self._validate_closed,
            )
        finally:
            self._active = False


@final
class ReplaySSMTargetAdapter:
    """Feature-detected target width-N / ReplaySSM commit-hook adapter."""

    def __init__(
        self,
        target_model: object,
        target_cache: object,
        verification_plan: Callable[[tuple[int, ...]], TargetVerificationPlan],
        build_verify: Callable[
            [tuple[int, ...], TargetVerificationPlan], _TargetPosteriorGraph
        ],
        preflight_ordinary: Callable[[int], OrdinaryDecodePlan],
        prepare_ordinary: Callable[[int, OrdinaryDecodePlan], PreparedOrdinaryDecode],
        validate_closed: Callable[[int], None] | None = None,
        validate_open: Callable[[int, int], None] | None = None,
    ):
        if not has_replayssm_target_hooks(target_model):
            raise DSparkFeatureUnavailableError(
                "MLX-LM Kimi K3 target is missing hidden-tap or ReplaySSM hooks"
            )
        self._hooks = cast(_SpeculativeCacheHooks, target_model)
        self._target_cache = target_cache
        self._verification_plan = verification_plan
        self._build_verify = build_verify
        self._preflight_ordinary = preflight_ordinary
        self._prepare_ordinary = prepare_ordinary
        self._validate_closed = validate_closed
        self._validate_open = validate_open

    def prepare_verification(
        self,
        proposal_block: tuple[int, ...],
    ) -> PreparedTargetVerification:
        plan = self._verification_plan(proposal_block)
        initial_offset = (
            _target_cache_offset(self._target_cache)
            if self._validate_closed is not None
            else 0
        )
        if self._validate_closed is not None:
            self._validate_closed(initial_offset)
        transaction: object | None = None
        try:
            transaction = self._hooks.begin_speculative_cache(
                self._target_cache,
                len(proposal_block),
            )
            if self._validate_open is not None:
                self._validate_open(initial_offset, len(proposal_block))
        except BaseException as begin_error:
            try:
                if transaction is not None:
                    _cancel_replayssm_transaction(
                        self._hooks,
                        transaction,
                        initial_offset=initial_offset,
                        validate_closed=self._validate_closed,
                    )
                elif self._validate_closed is not None:
                    self._validate_closed(initial_offset)
            except BaseException as cancel_error:
                raise DSparkCancellationError(
                    "target speculative begin failed and cache cleanup could "
                    f"not be attested: {type(cancel_error).__name__}: {cancel_error}"
                ) from begin_error
            raise
        return _PreparedReplaySSMVerification(
            self._hooks,
            transaction,
            proposal_block,
            plan,
            self._build_verify,
            initial_offset,
            self._validate_closed,
        )

    def preflight_ordinary(self, anchor_token: int) -> OrdinaryDecodePlan:
        return self._preflight_ordinary(anchor_token)

    def prepare_ordinary(
        self,
        anchor_token: int,
        plan: OrdinaryDecodePlan,
    ) -> PreparedOrdinaryDecode:
        return self._prepare_ordinary(
            anchor_token,
            plan,
        )


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


def preflight_mlx_dspark_segmented_sdpa(
    *,
    environ: Mapping[str, str] | None = None,
    module_loader: Callable[[str], object] | None = None,
) -> None:
    """Reject segmented DSpark unless MLX advertises bounded Metal memory.

    The accepted runtime keeps this experimental path disabled.  If an operator
    explicitly enables it, EXO delegates to MLX-LM's public capability gate
    before allocating either target or draft weights.
    """

    values = os.environ if environ is None else environ
    if not _strict_flag(
        MLX_DSPARK_SEGMENTED_SDPA_ENV,
        values.get(MLX_DSPARK_SEGMENTED_SDPA_ENV, "0"),
    ):
        return
    load_module = importlib.import_module if module_loader is None else module_loader
    try:
        module = load_module("mlx_lm.models.kimi_k3_dspark")
    except ImportError as error:
        raise DSparkFeatureUnavailableError(
            "segmented Kimi K3 DSpark requires the pinned MLX-LM capability gate"
        ) from error
    preflight = getattr(module, "require_kimi_k3_dspark_segmented_sdpa", None)
    if not callable(preflight):
        raise DSparkFeatureUnavailableError(
            "segmented Kimi K3 DSpark requires the pinned MLX-LM capability gate"
        )
    try:
        preflight()
    except Exception as error:
        raise DSparkFeatureUnavailableError(
            "segmented Kimi K3 DSpark requires MLX bounded_memory_metal_v1; "
            "the accepted runtime must keep it disabled"
        ) from error


class _ProposalTokenArray(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def tolist(self) -> object: ...


class _ConfidenceLogitArray(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def tolist(self) -> object: ...


class _AuxHiddenState(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def __getitem__(self, key: tuple[object, ...]) -> object: ...


class _MlxDSparkProposer(Protocol):
    verify_width: int

    def make_context_cache(self, **kwargs: object) -> object: ...

    def append_target_context(
        self,
        aux_hidden_states: Sequence[object],
        context_offset: int,
        context_cache: object,
    ) -> None: ...

    def propose(self, anchor_token: int, context_cache: object) -> object: ...


def _validate_mlx_proposal_graph(
    proposal: object,
    *,
    expected: int,
    verify_width: int,
) -> _ProposalTokenArray:
    proposal_width = getattr(proposal, "verify_width", None)
    if type(proposal_width) is not int or proposal_width != verify_width:
        raise ValueError("MLX-LM DSpark proposal verify width does not match")
    raw_tokens = getattr(proposal, "tokens", None)
    shape = getattr(raw_tokens, "shape", None)
    if _shape_tuple(shape) != (1, expected):
        raise ValueError(
            f"MLX-LM DSpark proposal token graph must have shape [1, {expected}]"
        )
    tolist = getattr(raw_tokens, "tolist", None)
    if not callable(tolist):
        raise TypeError("MLX-LM DSpark proposal tokens must be an MLX array")
    return cast(_ProposalTokenArray, raw_tokens)


def _validate_mlx_confidence_graph(
    proposal: object,
    *,
    expected: int,
) -> _ConfidenceLogitArray:
    candidate = getattr(proposal, "confidence_logits", None)
    confidence_shape = getattr(candidate, "shape", None)
    if _shape_tuple(confidence_shape) != (1, expected):
        raise ValueError(
            f"MLX-LM DSpark confidence graph must have shape [1, {expected}]"
        )
    confidence_tolist = getattr(candidate, "tolist", None)
    if not callable(confidence_tolist):
        raise TypeError("MLX-LM DSpark confidence logits must be an MLX array")
    return cast(_ConfidenceLogitArray, candidate)


@dataclass
class _ConfidenceCaptureState:
    enabled: bool = True

    def disable(self) -> None:
        if self.enabled:
            self.enabled = False
            _log_nonfatal_warning(
                "Kimi K3 DSpark confidence tensor capture failed and was disabled"
            )


def _materialize_mlx_proposal_tokens(
    raw_tokens: _ProposalTokenArray,
    *,
    expected: int,
) -> tuple[int, ...]:
    rows = raw_tokens.tolist()
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


def _materialize_mlx_confidence_logits(
    raw_logits: _ConfidenceLogitArray,
    *,
    expected: int,
) -> tuple[float, ...]:
    rows = raw_logits.tolist()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("MLX-LM DSpark confidence logits must have batch size one")
    row_values = cast(Sequence[object], rows)
    if (
        len(row_values) != 1
        or not isinstance(row_values[0], Sequence)
        or isinstance(row_values[0], (str, bytes))
    ):
        raise ValueError("MLX-LM DSpark confidence logits must have batch size one")
    return _confidence_tuple(
        cast(Sequence[object], row_values[0]),
        expected=expected,
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
    validated = _validated_aux_hidden_states(
        aux_hidden_states,
        expected_width=verify_width,
    )
    committed: list[object] = []
    for hidden in validated:
        hidden_state = cast(_AuxHiddenState, hidden)
        committed.append(hidden_state[:, :consumed, :])
    return tuple(committed)


def _validated_aux_hidden_states(
    aux_hidden_states: Sequence[object],
    *,
    expected_width: int,
) -> tuple[object, ...]:
    if len(aux_hidden_states) != KIMI_K3_TARGET_TAP_COUNT:
        raise ValueError(
            f"target DSpark must provide exactly {KIMI_K3_TARGET_TAP_COUNT} "
            "auxiliary hidden-state taps"
        )
    validated: list[object] = []
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
        batch_size, token_count, hidden_size = dimensions
        if (
            type(batch_size) is not int
            or batch_size != 1
            or type(token_count) is not int
            or token_count != expected_width
            or type(hidden_size) is not int
            or hidden_size != KIMI_K3_TARGET_HIDDEN_SIZE
        ):
            raise ValueError(
                "target DSpark auxiliary hidden-state shape must be "
                f"[1, {expected_width}, {KIMI_K3_TARGET_HIDDEN_SIZE}]"
            )
        validated.append(hidden)
    return tuple(validated)


def _materialize_context_cache(
    context_cache: object,
    evaluate: Callable[..., None],
) -> None:
    entries = cast(Sequence[object], context_cache)
    arrays: list[object] = []
    for entry in entries:
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            raise ValueError("MLX-LM DSpark context cache is not populated")
        arrays.extend((keys, values))
    evaluate(arrays)


def _context_cache_arrays(context_cache: object) -> tuple[object, ...]:
    """Collect the context-cache arrays without evaluating them."""

    entries = cast(Sequence[object], context_cache)
    arrays: list[object] = []
    for entry in entries:
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            raise ValueError("MLX-LM DSpark context cache is not populated")
        arrays.extend((keys, values))
    return tuple(arrays)


@final
class _MlxDSparkDraftRound:
    """Non-mutating proposal followed by append-only committed target context."""

    def __init__(
        self,
        proposer: _MlxDSparkProposer,
        context_cache: object,
        proposal_tokens: tuple[int, ...],
        confidence_logits: tuple[float, ...] | None,
        verify_width: int,
        evaluate: Callable[..., None],
    ):
        self._proposer = proposer
        self._context_cache = context_cache
        self._proposal_tokens = proposal_tokens
        self._confidence_logits = confidence_logits
        self._verify_width = verify_width
        self._evaluate = evaluate
        self._active = True
        self._pending_offset: int | None = None

    @property
    def proposal_tokens(self) -> tuple[int, ...]:
        return self._proposal_tokens

    @property
    def confidence_logits(self) -> tuple[float, ...] | None:
        return self._confidence_logits

    def commit(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> None:
        self.commit_build(accepted_draft_tokens, next_anchor_token, target_posterior)
        self.commit_finalize(evaluate=True)

    def commit_build(
        self,
        accepted_draft_tokens: int,
        next_anchor_token: int,
        target_posterior: TargetPosterior,
    ) -> tuple[int, tuple[object, ...]]:
        """Validate and lazily append committed context; defer materialization.

        Returns the expected post-commit context offset together with the
        appended cache arrays so a tail-overlap caller can enqueue them
        asynchronously.  :meth:`commit_finalize` must run afterwards.
        """

        if not self._active:
            raise RuntimeError("MLX-LM DSpark draft round is no longer active")
        if self._pending_offset is not None:
            raise RuntimeError("MLX-LM DSpark draft commit was already built")
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
        except Exception:
            # MLX-LM exposes append-only context, not rollback. Never retry a
            # possibly partial append; rank consensus disables the draft.
            self._active = False
            raise
        self._pending_offset = context_offset + consumed
        return self._pending_offset, _context_cache_arrays(self._context_cache)

    def commit_finalize(self, *, evaluate: bool) -> None:
        """Materialize (unless already submitted) and assert the new offset."""

        if not self._active:
            raise RuntimeError("MLX-LM DSpark draft round is no longer active")
        expected_offset = self._pending_offset
        if expected_offset is None:
            raise RuntimeError("MLX-LM DSpark draft commit was not built")
        try:
            if evaluate:
                _materialize_context_cache(self._context_cache, self._evaluate)
            if _context_cache_offset(self._context_cache) != expected_offset:
                raise ValueError(
                    "MLX-LM DSpark committed context offset does not match"
                )
        finally:
            # MLX-LM exposes append-only context, not rollback. Never retry a
            # possibly partial append; rank consensus disables the draft.
            self._active = False

    def cancel(self) -> None:
        # ``propose`` does not mutate MLX-LM's target-context cache.
        self._active = False


@final
class _PreparedMlxDSparkRound:
    """Hold a validated lazy proposal until rank agreement permits evaluation."""

    def __init__(
        self,
        proposer: _MlxDSparkProposer,
        context_cache: object,
        proposal_tokens: _ProposalTokenArray,
        confidence_logits: _ConfidenceLogitArray | None,
        confidence_state: _ConfidenceCaptureState | None,
        proposal_count: int,
        verify_width: int,
        evaluate: Callable[..., None],
    ):
        self._proposer = proposer
        self._context_cache = context_cache
        self._proposal_tokens = proposal_tokens
        self._confidence_logits = confidence_logits
        self._confidence_state = confidence_state
        self._proposal_count = proposal_count
        self._verify_width = verify_width
        self._evaluate = evaluate
        self._active = True
        self._submitted = False

    def submit(self, context_arrays: Sequence[object]) -> None:
        """Asynchronously enqueue the appended context plus proposal graphs.

        The proposal graph closes over the freshly appended context arrays, so
        one ``mx.async_eval`` schedules the exact kernel sequence the serial
        path would run, without blocking the caller.  ``materialize`` keeps its
        token-only contract afterwards.
        """

        if not self._active:
            raise RuntimeError("MLX-LM DSpark proposal graph is no longer active")
        if self._submitted:
            raise RuntimeError("MLX-LM DSpark proposal graph was already submitted")
        roots: list[object] = [*context_arrays, self._proposal_tokens]
        if self._confidence_logits is not None:
            roots.append(self._confidence_logits)
        cast(Callable[..., None], mx.async_eval)(roots)
        self._submitted = True

    def materialize(self) -> DraftRound:
        if not self._active:
            raise RuntimeError("MLX-LM DSpark proposal graph is no longer active")
        try:
            # Preserve the exact token-only inference path as authoritative.
            proposal_tokens = _materialize_mlx_proposal_tokens(
                self._proposal_tokens,
                expected=self._proposal_count,
            )
            confidence_logits: tuple[float, ...] | None = None
            if self._confidence_logits is not None:
                try:
                    # Proposal tokens are already materialized and usable. The
                    # rank-zero diagnostic tensor cannot affect their value.
                    self._evaluate(self._confidence_logits)
                    confidence_logits = _materialize_mlx_confidence_logits(
                        self._confidence_logits,
                        expected=self._proposal_count,
                    )
                except Exception:
                    if self._confidence_state is not None:
                        self._confidence_state.disable()
        finally:
            self._active = False
        return _MlxDSparkDraftRound(
            self._proposer,
            self._context_cache,
            proposal_tokens,
            confidence_logits,
            self._verify_width,
            self._evaluate,
        )

    def cancel(self) -> None:
        # Proposal construction is lazy and does not mutate target context.
        self._active = False


@dataclass(frozen=True)
class MlxDSparkRequestDraft:
    """Fresh per-request context bound to one rank-local loaded proposer."""

    proposer: object
    context_cache: object
    verify_width: int
    confidence_state: _ConfidenceCaptureState | None = None
    evaluate: Callable[..., None] = mx.eval
    placement: Literal["replicated"] = "replicated"

    def preflight_round(self, anchor_token: int, num_proposals: int) -> int:
        """Validate replicated draft state before its borrowed TP head runs."""

        if type(anchor_token) is not int or anchor_token < 0:
            raise ValueError("anchor_token must be a non-negative integer")
        if num_proposals != self.verify_width - 1:
            raise ValueError("requested DSpark proposal count does not match its width")
        if os.environ.get(MLX_DSPARK_PROPOSER_ENV) != "1":
            raise ValueError(
                "MLX-LM DSpark proposer became disabled during the request"
            )
        context_offset = _context_cache_offset(self.context_cache)
        if context_offset <= 0:
            raise ValueError("MLX-LM DSpark proposal requires populated context")
        _materialize_context_cache(self.context_cache, self.evaluate)
        return context_offset

    def seed_target_context(
        self,
        aux_hidden_states: Sequence[object],
        *,
        expected_width: int,
    ) -> None:
        validated = _validated_aux_hidden_states(
            aux_hidden_states,
            expected_width=expected_width,
        )
        context_offset = _context_cache_offset(self.context_cache)
        proposer = cast(_MlxDSparkProposer, self.proposer)
        proposer.append_target_context(
            validated,
            context_offset,
            self.context_cache,
        )
        _materialize_context_cache(self.context_cache, self.evaluate)
        if _context_cache_offset(self.context_cache) != context_offset + expected_width:
            raise ValueError("MLX-LM DSpark seeded context offset does not match")

    def prepare_round(
        self,
        anchor_token: int,
        num_proposals: int,
    ) -> PreparedDraftRound:
        if type(anchor_token) is not int or anchor_token < 0:
            raise ValueError("anchor_token must be a non-negative integer")
        if num_proposals != self.verify_width - 1:
            raise ValueError("requested DSpark proposal count does not match its width")
        proposer = cast(_MlxDSparkProposer, self.proposer)
        proposal = proposer.propose(anchor_token, self.context_cache)
        proposal_tokens = _validate_mlx_proposal_graph(
            proposal,
            expected=num_proposals,
            verify_width=self.verify_width,
        )
        confidence_logits: _ConfidenceLogitArray | None = None
        if self.confidence_state is not None and self.confidence_state.enabled:
            try:
                confidence_logits = _validate_mlx_confidence_graph(
                    proposal,
                    expected=num_proposals,
                )
            except Exception:
                self.confidence_state.disable()
        return _PreparedMlxDSparkRound(
            proposer,
            self.context_cache,
            proposal_tokens,
            confidence_logits,
            self.confidence_state,
            num_proposals,
            self.verify_width,
            self.evaluate,
        )


def attest_kimi_k3_target_route_top_k(
    target_model: object,
    *,
    expected: int | None = None,
) -> int:
    """Attest one routing width across every sparse target layer."""

    layers: object = getattr(target_model, "layers", None)
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise DSparkConfigurationError("Kimi K3 target layers are unavailable")
    route_top_ks: list[int] = []
    for index, layer in enumerate(cast(Sequence[object], layers)):
        mlp = getattr(layer, "mlp", None)
        is_sparse = any(
            hasattr(mlp, marker)
            for marker in ("switch_mlp", "e_score_correction_bias", "expert_top_k")
        )
        if not is_sparse:
            continue
        route_top_k = getattr(mlp, "expert_top_k", None)
        if type(route_top_k) is not int or route_top_k <= 0:
            raise DSparkConfigurationError(
                f"Kimi K3 sparse target layer {index} has no valid route top-k"
            )
        route_top_ks.append(route_top_k)
    if not route_top_ks:
        raise DSparkConfigurationError("Kimi K3 target has no attested sparse layers")
    unique = set(route_top_ks)
    if len(unique) != 1:
        raise DSparkConfigurationError(
            "Kimi K3 sparse target layers disagree on route top-k"
        )
    route_top_k = next(iter(unique))
    if expected is not None and route_top_k != expected:
        raise DSparkConfigurationError(
            f"Kimi K3 target route top-k is {route_top_k}, expected {expected}"
        )
    return route_top_k


@dataclass(frozen=True)
class LoadedMlxDSpark:
    """Rank-local immutable weights/proposer factory; no request context lives here."""

    config: KimiK3DSparkConfig
    target_model: object
    drafter: object
    proposer: object
    target_route_top_k: int | None = None
    evaluate: Callable[..., None] = mx.eval
    placement: Literal["replicated"] = "replicated"

    @property
    def verify_width(self) -> int:
        return self.config.verify_width

    def new_request(
        self,
        *,
        capacity_hint: int,
        capture_confidence: bool = False,
    ) -> MlxDSparkRequestDraft:
        if type(capacity_hint) is not int or capacity_hint <= 0:
            raise ValueError("Kimi K3 DSpark context capacity hint must be positive")
        if capture_confidence and self.config.confidence_capture is None:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark confidence materialization requires capture config"
            )
        proposer = cast(_MlxDSparkProposer, self.proposer)
        make_context_cache = proposer.make_context_cache
        parameters = inspect.signature(make_context_cache).parameters
        context_cache = (
            make_context_cache(capacity_hint=capacity_hint)
            if "capacity_hint" in parameters
            else make_context_cache()
        )
        if _context_cache_offset(context_cache) != 0:
            raise ValueError("MLX-LM DSpark request context must start empty")
        return MlxDSparkRequestDraft(
            proposer=self.proposer,
            context_cache=context_cache,
            verify_width=self.verify_width,
            confidence_state=(
                _ConfidenceCaptureState() if capture_confidence else None
            ),
            evaluate=self.evaluate,
        )


DSparkProposerRole = Literal["old", "yarn"]


@dataclass(frozen=True)
class KimiK3DSparkProposerSelection:
    """Immutable request-scoped identity chosen from initial encoded tokens."""

    role: DSparkProposerRole
    initial_prompt_tokens: int
    threshold_tokens: int
    revision: str
    config_sha256: str
    model_sha256: str

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(
            {
                "role": self.role,
                "initial_prompt_tokens": self.initial_prompt_tokens,
                "threshold_tokens": self.threshold_tokens,
                "revision": self.revision,
                "config_sha256": self.config_sha256,
                "model_sha256": self.model_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(b"exo-kimi-k3-dspark-selection/v1\0")
        digest.update(payload)
        return digest.hexdigest()


@dataclass
class LoadedMlxDSparkDual:
    """Two fully loaded proposers with no shared request-context cache."""

    config: KimiK3DSparkDualConfig
    old: LoadedMlxDSpark | None
    yarn: LoadedMlxDSpark | None
    placement: Literal["replicated"] = "replicated"
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.old is None or self.yarn is None:
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark requires both loaded proposers"
            )
        if self.old.config != self.config.old or self.yarn.config != self.config.yarn:
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark loaded proposer identities do not match config"
            )
        if self.old.target_model is not self.yarn.target_model:
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark proposers must share one target model"
            )
        if self.old.verify_width != self.yarn.verify_width:
            raise DSparkConfigurationError(
                "Kimi K3 dual DSpark loaded proposer widths disagree"
            )

    @property
    def verify_width(self) -> int:
        return self.config.verify_width

    @property
    def target_model(self) -> object:
        old = self._require_loaded(self.old)
        return old.target_model

    @staticmethod
    def _require_loaded(loaded: LoadedMlxDSpark | None) -> LoadedMlxDSpark:
        if loaded is None:
            raise DSparkConfigurationError("Kimi K3 dual DSpark is closed")
        return loaded

    def select(
        self,
        initial_prompt_tokens: int,
    ) -> tuple[LoadedMlxDSpark, KimiK3DSparkProposerSelection]:
        if type(initial_prompt_tokens) is not int or initial_prompt_tokens < 2:
            raise ValueError(
                "Kimi K3 dual DSpark initial prompt must contain at least two tokens"
            )
        if initial_prompt_tokens < self.config.threshold_tokens:
            role: DSparkProposerRole = "old"
            loaded = self._require_loaded(self.old)
        else:
            role = "yarn"
            loaded = self._require_loaded(self.yarn)
        selected_config = loaded.config
        return loaded, KimiK3DSparkProposerSelection(
            role=role,
            initial_prompt_tokens=initial_prompt_tokens,
            threshold_tokens=self.config.threshold_tokens,
            revision=selected_config.revision,
            config_sha256=selected_config.config_sha256,
            model_sha256=selected_config.model_sha256,
        )

    def close(self) -> None:
        """Drop both rank-local weight graphs and release cached MLX allocations."""

        if self._closed:
            return
        self._closed = True
        self.old = None
        self.yarn = None
        _release_mlx_memory()


LoadedKimiK3DSpark = LoadedMlxDSpark | LoadedMlxDSparkDual


def select_loaded_mlx_dspark(
    loaded: LoadedKimiK3DSpark,
    *,
    initial_prompt_tokens: int,
) -> tuple[LoadedMlxDSpark, KimiK3DSparkProposerSelection | None]:
    """Select once; the returned ordinary loaded object owns the only request cache."""

    if isinstance(loaded, LoadedMlxDSparkDual):
        return loaded.select(initial_prompt_tokens)
    return loaded, None


def _release_mlx_memory() -> None:
    with contextlib.suppress(Exception):
        mx.synchronize()
    gc.collect()
    with contextlib.suppress(Exception):
        mx.clear_cache()


def load_replicated_mlx_dspark(
    config: KimiK3DSparkConfig,
    target_model: object,
    *,
    features: MlxDSparkFeatures | None = None,
    evaluate: Callable[..., None] = mx.eval,
) -> LoadedMlxDSpark:
    """Load the pinned local draft independently on the calling target rank.

    EXO intentionally passes ``verify_weights_sha256=True`` and an explicit
    width. No model identifier is accepted here, so this adapter cannot trigger
    a remote checkpoint download.
    """

    target_route_top_k = None
    if config.confidence_capture is not None or width4_receipt_log_enabled():
        target_route_top_k = attest_kimi_k3_target_route_top_k(
            target_model,
            expected=8,
        )
    preflight_mlx_dspark_segmented_sdpa()
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
        screening_override=(config.verify_width != DSPARK_MODEL_NATIVE_VERIFY_WIDTH),
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
    return LoadedMlxDSpark(
        config=config,
        target_model=target_model,
        drafter=drafter,
        proposer=proposer,
        target_route_top_k=target_route_top_k,
        evaluate=evaluate,
    )


def load_replicated_mlx_dspark_dual(
    config: KimiK3DSparkDualConfig,
    target_model: object,
    *,
    features: MlxDSparkFeatures | None = None,
    evaluate: Callable[..., None] = mx.eval,
    loader: Callable[[KimiK3DSparkConfig, object], LoadedMlxDSpark] | None = None,
) -> LoadedMlxDSparkDual:
    """Load and fully attest both pinned checkpoints exactly once on this rank."""

    def load_one(
        checkpoint_config: KimiK3DSparkConfig,
        model: object,
    ) -> LoadedMlxDSpark:
        if loader is not None:
            return loader(checkpoint_config, model)
        return load_replicated_mlx_dspark(
            checkpoint_config,
            model,
            features=features,
            evaluate=evaluate,
        )

    old = load_one(config.old, target_model)
    try:
        yarn = load_one(config.yarn, target_model)
    except BaseException:
        old = None
        _release_mlx_memory()
        raise
    try:
        return LoadedMlxDSparkDual(config=config, old=old, yarn=yarn)
    except BaseException:
        old = None
        yarn = None
        _release_mlx_memory()
        raise


class _TargetForwardResult(Protocol):
    logits: object
    aux_hidden_states: Sequence[object]


class _AuxPrefillResult(Protocol):
    final_hidden_state: object
    aux_hidden_states: Sequence[object]


class _CompactTargetForwardResult(Protocol):
    tokens: object
    aux_hidden_states: Sequence[object]


class _TargetCacheEntry(Protocol):
    @property
    def state(self) -> object: ...


class _TargetWithAuxForward(Protocol):
    def forward_with_aux_hidden_states(
        self,
        inputs: mx.array,
        cache: object,
        layer_ids: tuple[int, ...],
    ) -> object: ...


class _TargetWithAuxGreedyForward(Protocol):
    def forward_with_aux_hidden_states_greedy(
        self,
        inputs: mx.array,
        cache: object,
        layer_ids: tuple[int, ...],
        banned_token_ids: tuple[int, ...] = (),
    ) -> object: ...


class _CompactGreedyTarget(Protocol):
    def supports_vocab_parallel_greedy(self) -> bool: ...

    def vocab_parallel_greedy(
        self,
        inputs: mx.array,
        cache: object,
    ) -> object: ...


class _GreedyTokenArray(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def tolist(self) -> object: ...


class _BatchedGreedyTokenArray(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def tolist(self) -> object: ...


def _target_cache_offset(target_cache: object) -> int:
    if (
        not isinstance(target_cache, Sequence)
        or isinstance(target_cache, (str, bytes))
        or not target_cache
    ):
        raise ValueError("Kimi K3 target cache must be a non-empty sequence")
    offsets: list[int] = []
    for entry in cast(Sequence[object], target_cache):
        class_name = type(entry).__name__
        if any(
            unsupported in class_name
            for unsupported in ("Batch", "Quantized", "Rotating", "CacheList")
        ):
            raise ValueError(
                f"Kimi K3 DSpark does not support target cache type {class_name}"
            )
        offset: object = getattr(entry, "offset", None)
        if offset is not None:
            if type(offset) is not int or offset < 0:
                raise ValueError("Kimi K3 target cache offset is invalid")
            offsets.append(offset)
            continue
        values: object = getattr(entry, "cache", None)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(
                f"Kimi K3 DSpark does not recognize target cache type {class_name}"
            )
    if not offsets or len(set(offsets)) != 1:
        raise ValueError("Kimi K3 target MLA cache offsets disagree")
    return offsets[0]


def _validate_target_cache(
    target_cache: object,
    *,
    expected_offset: int,
    require_kda_state: bool,
    speculative_phase: Literal["closed", "open", "staged"] = "closed",
    speculative_width: int | None = None,
) -> None:
    if speculative_phase == "closed":
        if speculative_width is not None:
            raise ValueError("closed Kimi K3 target cache cannot have a width")
    elif type(speculative_width) is not int or speculative_width <= 1:
        raise ValueError("active Kimi K3 target cache requires an exact width")
    if _target_cache_offset(target_cache) != expected_offset:
        raise ValueError(f"Kimi K3 target cache must be at offset {expected_offset}")
    kda_entries = 0
    for entry in cast(Sequence[object], target_cache):
        values: object = getattr(entry, "cache", None)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        kda_entries += 1
        cache_values = cast(Sequence[object], values)
        populated = tuple(value is not None for value in cache_values)
        if require_kda_state and (not populated or not all(populated)):
            raise ValueError("Kimi K3 target KDA cache is not fully populated")
        if not require_kda_state and any(populated):
            raise ValueError("Kimi K3 target cache must be fresh for each request")
        width = getattr(entry, "speculative_width", 0)
        ready = getattr(entry, "speculative_ready", False)
        if type(width) is not int or type(ready) is not bool:
            raise ValueError("Kimi K3 target speculative cache markers are invalid")
        if speculative_phase == "closed":
            if width != 0 or ready:
                raise ValueError(
                    "Kimi K3 target cache has a stale speculative transaction"
                )
        elif width != speculative_width:
            raise ValueError("Kimi K3 target speculative width does not match")
        elif speculative_phase == "open" and ready:
            raise ValueError("Kimi K3 target speculative cache staged too early")
        elif speculative_phase == "staged" and not ready:
            raise ValueError("Kimi K3 target speculative checkpoints are incomplete")
    if kda_entries == 0:
        raise ValueError("Kimi K3 target cache contains no KDA state")


def _target_cache_states(target_cache: object) -> tuple[object, ...]:
    return tuple(
        cast(_TargetCacheEntry, entry).state
        for entry in cast(Sequence[object], target_cache)
    )


def _target_cache_materialization_roots(target_cache: object) -> tuple[object, ...]:
    """Flatten every nested cache state before any rank can enter evaluation."""

    roots: list[object] = []

    def append_state(value: object) -> None:
        if value is None:
            raise ValueError("Kimi K3 target cache contains an empty state root")
        if isinstance(value, Mapping):
            if not value:
                raise ValueError("Kimi K3 target cache contains an empty state root")
            for nested in cast(Mapping[object, object], value).values():
                append_state(nested)
            return
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Kimi K3 target cache contains an empty state root")
            for nested in cast(Sequence[object], value):
                append_state(nested)
            return
        roots.append(value)

    for state in _target_cache_states(target_cache):
        append_state(state)
    if not roots:
        raise ValueError("Kimi K3 target cache has no materialization roots")
    return tuple(roots)


def _validate_target_final_hidden_state(
    final_hidden_state: object,
    *,
    expected_width: int,
) -> None:
    shape = _shape_tuple(getattr(final_hidden_state, "shape", None))
    expected = (1, expected_width, KIMI_K3_TARGET_HIDDEN_SIZE)
    if shape != expected:
        raise ValueError(
            "Kimi K3 target final hidden state must have shape "
            f"[1, {expected_width}, {KIMI_K3_TARGET_HIDDEN_SIZE}]"
        )


def _validate_target_logits(logits: object, *, expected_width: int) -> None:
    shape: object = getattr(logits, "shape", None)
    if not isinstance(shape, Sequence):
        raise ValueError("Kimi K3 target logits have no shape")
    dimensions = tuple(cast(Sequence[object], shape))
    if (
        len(dimensions) != 3
        or dimensions[0] != 1
        or dimensions[1] != expected_width
        or type(dimensions[2]) is not int
        or dimensions[2] <= 0
    ):
        raise ValueError(
            f"Kimi K3 target logits must have shape [1, {expected_width}, vocab]"
        )


def _build_greedy_dspark_posterior_tokens(
    logits: object,
    *,
    expected_width: int,
    banned_token_ids: Sequence[int],
) -> _GreedyTokenArray:
    _validate_target_logits(logits, expected_width=expected_width)
    logits_array = cast(mx.array, logits)
    token_logits = logits_array[0]
    vocab_size = int(token_logits.shape[-1])
    for token_id in banned_token_ids:
        if type(token_id) is not int or not 0 <= token_id < vocab_size:
            raise ValueError("Kimi K3 DSpark banned token id is out of range")
        token_logits[..., token_id] = float("-inf")
    tokens = mx.argmax(token_logits, axis=-1).astype(mx.int32)
    shape = getattr(tokens, "shape", None)
    if _shape_tuple(shape) != (expected_width,):
        raise ValueError("Kimi K3 target posterior graph has an invalid shape")
    if not callable(getattr(tokens, "tolist", None)):
        raise TypeError("Kimi K3 target posterior must be an MLX array")
    return cast(_GreedyTokenArray, tokens)


def _materialize_greedy_dspark_posterior_tokens(
    tokens: _GreedyTokenArray,
    *,
    expected_width: int,
) -> tuple[int, ...]:
    values = cast(Sequence[int], tokens.tolist())
    return _token_tuple(
        values,
        expected=expected_width,
        name="Kimi K3 target posterior",
    )


def _validate_batched_greedy_tokens(
    tokens: object,
    *,
    expected_width: int,
) -> _BatchedGreedyTokenArray:
    shape = getattr(tokens, "shape", None)
    if _shape_tuple(shape) != (1, expected_width):
        raise ValueError(
            f"Kimi K3 compact verifier tokens must have shape [1, {expected_width}]"
        )
    if not callable(getattr(tokens, "tolist", None)):
        raise TypeError("Kimi K3 compact verifier tokens must be an MLX array")
    return cast(_BatchedGreedyTokenArray, tokens)


def _materialize_batched_greedy_tokens(
    tokens: _BatchedGreedyTokenArray,
    *,
    expected_width: int,
) -> tuple[int, ...]:
    rows = tokens.tolist()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Kimi K3 compact verifier tokens must have batch size one")
    row_values = cast(Sequence[object], rows)
    if len(row_values) != 1:
        raise ValueError("Kimi K3 compact verifier tokens must have batch size one")
    first_row = row_values[0]
    if not isinstance(first_row, Sequence) or isinstance(first_row, (str, bytes)):
        raise ValueError("Kimi K3 compact verifier tokens must have batch size one")
    return _token_tuple(
        cast(Sequence[int], first_row),
        expected=expected_width,
        name="Kimi K3 compact target posterior",
    )


def greedy_dspark_posterior_tokens(
    logits: object,
    *,
    expected_width: int,
    banned_token_ids: Sequence[int],
    evaluate: Callable[..., None],
) -> tuple[int, ...]:
    """Build and materialize a greedy posterior outside rank-gated paths."""

    tokens = _build_greedy_dspark_posterior_tokens(
        logits,
        expected_width=expected_width,
        banned_token_ids=banned_token_ids,
    )
    evaluate(tokens)
    return _materialize_greedy_dspark_posterior_tokens(
        tokens,
        expected_width=expected_width,
    )


@final
class _BuiltKimiK3TargetForward:
    """Shape-checked lazy target forward, before any MLX evaluation."""

    def __init__(
        self,
        *,
        forward: _TargetForwardResult,
        validated_aux_hidden_states: tuple[object, ...],
        target_cache: object,
        initial_offset: int,
        width: int,
        speculative_width: int | None,
        evaluate: Callable[..., None],
    ):
        self.forward = forward
        self.validated_aux_hidden_states = validated_aux_hidden_states
        self._target_cache = target_cache
        self._initial_offset = initial_offset
        self._width = width
        self._speculative_width = speculative_width
        self._evaluate = evaluate
        self._materialized = False

    def materialize(self, *extra_values: object) -> _TargetForwardResult:
        if self._materialized:
            raise RuntimeError("Kimi K3 target graph was already materialized")
        self._evaluate(
            self.forward.logits,
            self.validated_aux_hidden_states,
            _target_cache_states(self._target_cache),
            *extra_values,
        )
        _validate_target_cache(
            self._target_cache,
            expected_offset=self._initial_offset + self._width,
            require_kda_state=True,
            speculative_phase=(
                "staged" if self._speculative_width is not None else "closed"
            ),
            speculative_width=self._speculative_width,
        )
        self._materialized = True
        return self.forward


@final
class _BuiltKimiK3AuxPrefill:
    """Shape-checked prompt graph with no vocabulary-logit evaluation root."""

    def __init__(
        self,
        *,
        final_hidden_state: object,
        validated_aux_hidden_states: tuple[object, ...],
        target_cache: object,
        cache_roots: tuple[object, ...],
        initial_offset: int,
        width: int,
        evaluate: Callable[..., None],
    ):
        self.final_hidden_state = final_hidden_state
        self.validated_aux_hidden_states = validated_aux_hidden_states
        self._target_cache = target_cache
        self._cache_roots = cache_roots
        self._initial_offset = initial_offset
        self._width = width
        self._evaluate = evaluate
        self._materialized = False

    def materialize(self) -> tuple[object, ...]:
        if self._materialized:
            raise RuntimeError(
                "Kimi K3 auxiliary prefill graph was already materialized"
            )
        self._evaluate(
            self.final_hidden_state,
            *self.validated_aux_hidden_states,
            *self._cache_roots,
        )
        _validate_target_cache(
            self._target_cache,
            expected_offset=self._initial_offset + self._width,
            require_kda_state=True,
        )
        self._materialized = True
        return self.validated_aux_hidden_states


@final
class _BuiltKimiK3TargetPosterior:
    """Lazy verifier graph whose materialization may enter target TP."""

    def __init__(
        self,
        forward: _BuiltKimiK3TargetForward,
        tokens: _GreedyTokenArray,
        expected_width: int,
    ):
        self._forward = forward
        self._tokens = tokens
        self._expected_width = expected_width

    def materialize(self) -> TargetPosterior:
        self._forward.materialize(self._tokens)
        tokens = _materialize_greedy_dspark_posterior_tokens(
            self._tokens,
            expected_width=self._expected_width,
        )
        return TargetPosterior(
            tokens,
            tuple(self._forward.validated_aux_hidden_states),
        )


@final
class _BuiltKimiK3CompactTargetPosterior:
    """Compact verifier graph carrying tokens instead of full-vocab logits."""

    def __init__(
        self,
        *,
        tokens: _BatchedGreedyTokenArray,
        aux_hidden_states: tuple[object, ...],
        target_cache: object,
        initial_offset: int,
        width: int,
        evaluate: Callable[..., None],
    ):
        self._tokens = tokens
        self._aux_hidden_states = aux_hidden_states
        self._target_cache = target_cache
        self._initial_offset = initial_offset
        self._width = width
        self._evaluate = evaluate
        self._materialized = False

    def materialize(self) -> TargetPosterior:
        if self._materialized:
            raise RuntimeError(
                "Kimi K3 compact verifier graph was already materialized"
            )
        self._evaluate(
            self._tokens,
            self._aux_hidden_states,
            _target_cache_states(self._target_cache),
        )
        _validate_target_cache(
            self._target_cache,
            expected_offset=self._initial_offset + self._width,
            require_kda_state=True,
            speculative_phase="staged",
            speculative_width=self._width,
        )
        self._materialized = True
        return TargetPosterior(
            _materialize_batched_greedy_tokens(
                self._tokens,
                expected_width=self._width,
            ),
            self._aux_hidden_states,
        )


@final
class _BuiltKimiK3OrdinaryDecode:
    """Lazy one-token full/compact target graph after rank-agreed mode."""

    def __init__(
        self,
        *,
        output: object,
        mode: Literal["full", "compact"],
        target_cache: object,
        expected_offset: int,
        evaluate: Callable[..., None],
    ):
        self._output = output
        self._mode = mode
        self._target_cache = target_cache
        self._expected_offset = expected_offset
        self._evaluate = evaluate
        self._materialized = False

    def materialize(self) -> int:
        if self._materialized:
            raise RuntimeError("Kimi K3 ordinary target graph was already materialized")
        self._evaluate(self._output, _target_cache_states(self._target_cache))
        _validate_target_cache(
            self._target_cache,
            expected_offset=self._expected_offset,
            require_kda_state=True,
        )
        self._materialized = True
        if self._mode == "compact":
            item = getattr(self._output, "item", None)
            if not callable(item):
                raise TypeError("Kimi K3 compact greedy token is not materializable")
            token = item()
            if type(token) is not int or token < 0:
                raise ValueError("Kimi K3 compact greedy target token is invalid")
            return token
        return _materialize_greedy_dspark_posterior_tokens(
            cast(_GreedyTokenArray, self._output),
            expected_width=1,
        )[0]


@dataclass
class KimiK3DSparkRequestRuntime:
    """One fresh target/draft cache pair for sequential greedy TP2 generation."""

    loaded: LoadedMlxDSpark
    target_model: object
    target_cache: object
    draft: MlxDSparkRequestDraft
    collective: RankAgreement
    banned_token_ids: tuple[int, ...] = ()
    terminal_token_ids: tuple[int, ...] = ()
    compact_greedy: bool = False
    confidence_request: DSparkConfidenceCaptureRequest | None = None
    evaluate: Callable[..., None] = mx.eval
    clock: Callable[[], float] = time.perf_counter
    _confidence_recorder: DSparkConfidenceRecorder | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        self.clock = _nonthrowing_telemetry_clock(self.clock)
        if self.target_model is not self.loaded.target_model:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark weights are bound to a different target model"
            )
        if self.collective.size != 2:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark first canary requires exactly two tensor ranks"
            )
        if self.draft.verify_width != self.loaded.verify_width:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark request and loaded proposer widths do not match"
            )
        if (self.loaded.config.confidence_capture is None) != (
            self.confidence_request is None
        ):
            raise DSparkConfigurationError(
                "Kimi K3 DSpark confidence request metadata does not match config"
            )
        if (
            self.loaded.config.confidence_capture is not None
            and self.loaded.target_route_top_k != 8
        ):
            raise DSparkConfigurationError(
                "Kimi K3 DSpark confidence capture target route top-k is not 8"
            )
        if not has_replayssm_target_hooks(self.target_model):
            raise DSparkFeatureUnavailableError(
                "MLX-LM Kimi K3 target is missing hidden-tap or ReplaySSM hooks"
            )
        layers: object = getattr(self.target_model, "layers", None)
        if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
            raise DSparkConfigurationError("Kimi K3 target layers are unavailable")
        target_cache: object = self.target_cache
        if not isinstance(target_cache, Sequence) or isinstance(
            target_cache, (str, bytes)
        ):
            raise DSparkConfigurationError(
                "Kimi K3 target cache must be a non-empty sequence"
            )
        target_entries = cast(Sequence[object], target_cache)
        target_layers = cast(Sequence[object], layers)
        if len(target_entries) != len(target_layers):
            raise DSparkConfigurationError(
                "Kimi K3 fresh target cache does not match the target layers"
            )
        target_cache_object: object = target_entries
        _validate_target_cache(
            target_cache_object,
            expected_offset=0,
            require_kda_state=False,
        )
        if _context_cache_offset(self.draft.context_cache) != 0:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark request context must begin empty"
            )

    @classmethod
    def create(
        cls,
        loaded: LoadedMlxDSpark,
        target_model: object,
        target_cache: object,
        collective: RankAgreement,
        *,
        capacity_hint: int,
        banned_token_ids: Sequence[int] = (),
        terminal_token_ids: Sequence[int] = (),
        compact_greedy: bool = False,
        confidence_request: DSparkConfidenceCaptureRequest | None = None,
        evaluate: Callable[..., None] = mx.eval,
    ) -> "KimiK3DSparkRequestRuntime":
        draft = loaded.new_request(
            capacity_hint=capacity_hint,
            capture_confidence=(
                loaded.config.confidence_capture is not None and collective.rank == 0
            ),
        )
        return cls(
            loaded=loaded,
            target_model=target_model,
            target_cache=target_cache,
            draft=draft,
            collective=collective,
            banned_token_ids=tuple(banned_token_ids),
            terminal_token_ids=tuple(terminal_token_ids),
            compact_greedy=compact_greedy,
            confidence_request=confidence_request,
            evaluate=evaluate,
        )

    def _agreed_operation[T](self, name: str, operation: Callable[[], T]) -> T:
        result: T | None = None
        local_error: str | None = None
        try:
            result = operation()
        except Exception as error:
            local_error = f"{name} failed: {type(error).__name__}: {error}"

        if self.loaded.config.packed_agreements:
            agreement = self.collective.agree_packed(
                f"value:{name}",
                local_success=local_error is None,
                error_fingerprint=_error_fingerprint(local_error),
            )
            outcome = agreement.success
            fingerprint = agreement.error_fingerprint
        else:
            outcome = self.collective.agree_stage_success(local_error is None)
            fingerprint = self.collective.agree_token(_error_fingerprint(local_error))
        if outcome is not True or fingerprint != 0:
            detail = (
                "outcomes disagreed across ranks"
                if outcome is None or fingerprint is None
                else "failed on every rank"
            )
            raise DSparkDistributedStateError(
                f"Kimi K3 DSpark {name} {detail}; request caches cannot continue"
            ) from None
        return cast(T, result)

    def agree_local_value[T](self, name: str, operation: Callable[[], T]) -> T:
        """Agree local success before a peer can enter another TP graph."""

        return self._agreed_operation(name, operation)

    def agree_local_side_effect(self, name: str, operation: Callable[[], None]) -> None:
        """Agree a callback, preserving only a unanimous callback exception."""

        local_error: Exception | None = None
        error_text: str | None = None
        try:
            operation()
        except Exception as error:
            local_error = error
            error_text = f"{name} failed: {type(error).__name__}: {error}"

        local_fingerprint = _error_fingerprint(error_text)
        if self.loaded.config.packed_agreements:
            agreement = self.collective.agree_packed(
                f"side-effect:{name}",
                local_success=local_error is None,
                error_fingerprint=local_fingerprint,
            )
            outcome = agreement.success
            agreed_fingerprint = agreement.error_fingerprint
        else:
            outcome = self.collective.agree_stage_success(local_error is None)
            agreed_fingerprint = self.collective.agree_token(local_fingerprint)
        if outcome is True and agreed_fingerprint == 0:
            return
        if (
            outcome is False
            and local_error is not None
            and agreed_fingerprint == local_fingerprint
        ):
            raise local_error
        raise DSparkDistributedStateError(
            f"Kimi K3 DSpark {name} outcomes disagreed across ranks; "
            "request caches cannot continue"
        ) from None

    def agree_text(
        self,
        name: str,
        operation: Callable[[], str],
        *,
        preserve_unanimous_error: bool = False,
    ) -> str:
        """Agree successful local text production and its exact UTF-8 digest."""

        def render_and_fingerprint() -> tuple[str, tuple[int, ...]]:
            text = operation()
            return text, _token_contract_fingerprint(tuple(text.encode("utf-8")))

        if self.loaded.config.packed_agreements:
            text: str | None = None
            fingerprint = (0, 0, 0, 0)
            local_error: str | None = None
            try:
                text, fingerprint = render_and_fingerprint()
            except Exception as error:
                local_error = f"{name} failed: {type(error).__name__}: {error}"
                if preserve_unanimous_error:
                    fingerprint = _token_contract_fingerprint(
                        tuple(local_error.encode("utf-8"))
                    )
            agreement = self.collective.agree_packed(
                f"text:{name}",
                local_success=local_error is None,
                error_fingerprint=_error_fingerprint(local_error),
                payload=fingerprint,
            )
            if (
                preserve_unanimous_error
                and agreement.success is False
                and local_error is not None
                and agreement.error_fingerprint == _error_fingerprint(local_error)
                and agreement.payload == fingerprint
            ):
                raise DSparkDistributedStateError(
                    f"Kimi K3 DSpark {name} failed on every rank: {local_error}; "
                    "request caches cannot continue"
                ) from None
            if (
                agreement.success is not True
                or agreement.error_fingerprint != 0
                or agreement.payload != fingerprint
                or text is None
            ):
                detail = (
                    "outcomes disagreed across ranks"
                    if (preserve_unanimous_error and local_error is not None)
                    or agreement.success is None
                    or agreement.error_fingerprint is None
                    or agreement.payload is None
                    else "failed on every rank"
                )
                raise DSparkDistributedStateError(
                    f"Kimi K3 DSpark {name} {detail}; request caches cannot continue"
                ) from None
            return text

        text, fingerprint = self._agreed_operation(name, render_and_fingerprint)
        for word in fingerprint:
            if self.collective.agree_token(word) != word:
                raise DSparkDistributedStateError(
                    f"Kimi K3 DSpark {name} disagreed across ranks; "
                    "request caches cannot continue"
                ) from None
        return text

    def agree_response_control(
        self,
        operation: Callable[[], tuple[int, str | None, bool, str, str]],
    ) -> tuple[int, str | None, bool, str, str]:
        """Agree text-derived stop control before callbacks or another round."""

        def resolve() -> tuple[
            tuple[int, str | None, bool, str, str],
            tuple[int, ...],
        ]:
            result = operation()
            token, finish_reason, stop_matched, accumulated_text, visible_text = result
            if type(token) is not int or not 0 <= token <= 0x7FFFFFFF:
                raise ValueError("Kimi K3 DSpark response token is invalid")
            finish_codes = {None: 0, "stop": 1, "length": 2}
            if finish_reason not in finish_codes:
                raise ValueError("Kimi K3 DSpark finish reason is invalid")
            if type(stop_matched) is not bool:
                raise TypeError("Kimi K3 DSpark stop-match flag must be boolean")
            text_fingerprint = _token_contract_fingerprint(
                tuple(accumulated_text.encode("utf-8")),
                tuple(visible_text.encode("utf-8")),
            )
            control = (
                token,
                finish_codes[finish_reason],
                int(stop_matched),
                *text_fingerprint,
            )
            return result, control

        if self.loaded.config.packed_agreements:
            result: tuple[int, str | None, bool, str, str] | None = None
            control = (0, 0, 0, 0, 0, 0, 0)
            local_error: str | None = None
            try:
                result, control = resolve()
            except Exception as error:
                local_error = (
                    f"response control failed: {type(error).__name__}: {error}"
                )
            agreement = self.collective.agree_packed(
                "response-control",
                local_success=local_error is None,
                error_fingerprint=_error_fingerprint(local_error),
                payload=control,
            )
            if (
                agreement.success is not True
                or agreement.error_fingerprint != 0
                or agreement.payload != control
                or result is None
            ):
                detail = (
                    "outcomes disagreed across ranks"
                    if agreement.success is None
                    or agreement.error_fingerprint is None
                    or agreement.payload is None
                    else "failed on every rank"
                )
                raise DSparkDistributedStateError(
                    "Kimi K3 DSpark response control "
                    f"{detail}; request caches cannot continue"
                ) from None
            return result

        result, control = self._agreed_operation("response control", resolve)
        for value in control:
            if self.collective.agree_token(value) != value:
                raise DSparkDistributedStateError(
                    "Kimi K3 DSpark response control disagreed across ranks; "
                    "request caches cannot continue"
                ) from None
        return result

    def _target_aux_prefill_forward(
        self,
    ) -> Callable[[mx.array, object, tuple[int, ...]], object]:
        forward = getattr(
            self.target_model,
            "forward_aux_hidden_states_for_cache",
            None,
        )
        if not callable(forward):
            raise DSparkFeatureUnavailableError(
                "Kimi K3 auxiliary prompt prefill was requested but the target "
                "rank does not expose forward_aux_hidden_states_for_cache"
            )
        return cast(Callable[[mx.array, object, tuple[int, ...]], object], forward)

    def _validate_forward_readiness(
        self,
        inputs: mx.array,
        *,
        speculative_width: int | None = None,
    ) -> int:
        if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] <= 0:
            raise ValueError("Kimi K3 DSpark target input must be non-empty batch one")
        width = int(inputs.shape[1])
        if speculative_width is not None and speculative_width != width:
            raise ValueError(
                "Kimi K3 target verification width does not match its input"
            )
        initial_offset = _target_cache_offset(self.target_cache)
        _validate_target_cache(
            self.target_cache,
            expected_offset=initial_offset,
            require_kda_state=speculative_width is not None or initial_offset > 0,
            speculative_phase="open" if speculative_width is not None else "closed",
            speculative_width=speculative_width,
        )
        return initial_offset

    def _build_forward_with_taps(
        self,
        inputs: mx.array,
        *,
        initial_offset: int,
        speculative_width: int | None = None,
    ) -> _BuiltKimiK3TargetForward:
        if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] <= 0:
            raise ValueError("Kimi K3 DSpark target input must be non-empty batch one")
        width = int(inputs.shape[1])
        if speculative_width is not None and speculative_width != width:
            raise ValueError(
                "Kimi K3 target verification width does not match its input"
            )
        if _target_cache_offset(self.target_cache) != initial_offset:
            raise ValueError("Kimi K3 target cache moved after its readiness gate")
        forward = cast(
            _TargetWithAuxForward,
            self.target_model,
        ).forward_with_aux_hidden_states(
            inputs,
            self.target_cache,
            self.loaded.config.target_hidden_state_indices,
        )
        logits = getattr(forward, "logits", None)
        aux_hidden_states = getattr(forward, "aux_hidden_states", None)
        if not isinstance(aux_hidden_states, Sequence) or isinstance(
            aux_hidden_states, (str, bytes)
        ):
            raise ValueError("Kimi K3 target did not return auxiliary hidden states")
        validated = _validated_aux_hidden_states(
            cast(Sequence[object], aux_hidden_states),
            expected_width=width,
        )
        _validate_target_logits(logits, expected_width=width)
        return _BuiltKimiK3TargetForward(
            forward=cast(_TargetForwardResult, forward),
            validated_aux_hidden_states=validated,
            target_cache=self.target_cache,
            initial_offset=initial_offset,
            width=width,
            speculative_width=speculative_width,
            evaluate=self.evaluate,
        )

    def _build_aux_prefill_with_taps(
        self,
        inputs: mx.array,
        *,
        initial_offset: int,
        forward: Callable[[mx.array, object, tuple[int, ...]], object],
    ) -> _BuiltKimiK3AuxPrefill:
        if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] <= 0:
            raise ValueError("Kimi K3 DSpark target input must be non-empty batch one")
        width = int(inputs.shape[1])
        if _target_cache_offset(self.target_cache) != initial_offset:
            raise ValueError("Kimi K3 target cache moved after its readiness gate")
        result = cast(
            _AuxPrefillResult,
            forward(
                inputs,
                self.target_cache,
                self.loaded.config.target_hidden_state_indices,
            ),
        )
        final_hidden_state = getattr(result, "final_hidden_state", None)
        aux_hidden_states = getattr(result, "aux_hidden_states", None)
        if not isinstance(aux_hidden_states, Sequence) or isinstance(
            aux_hidden_states, (str, bytes)
        ):
            raise TypeError("Kimi K3 target did not return auxiliary hidden states")
        _validate_target_final_hidden_state(
            final_hidden_state,
            expected_width=width,
        )
        validated = _validated_aux_hidden_states(
            cast(Sequence[object], aux_hidden_states),
            expected_width=width,
        )
        cache_roots = _target_cache_materialization_roots(self.target_cache)
        return _BuiltKimiK3AuxPrefill(
            final_hidden_state=final_hidden_state,
            validated_aux_hidden_states=validated,
            target_cache=self.target_cache,
            cache_roots=cache_roots,
            initial_offset=initial_offset,
            width=width,
            evaluate=self.evaluate,
        )

    def _prompt_contract(
        self,
        prompt_prefix: mx.array,
        *,
        max_tokens: int,
        prefill_step_size: int,
        stop_sequences: Sequence[str],
        distributed_progress: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if prompt_prefix.ndim != 1 or len(prompt_prefix) == 0:
            raise DSparkConfigurationError(
                "Kimi K3 DSpark requires at least two logical prompt tokens"
            )
        tolist = getattr(prompt_prefix, "tolist", None)
        if not callable(tolist):
            raise TypeError("Kimi K3 prompt prefix must be an MLX token array")
        raw_tokens = tolist()
        if not isinstance(raw_tokens, Sequence) or isinstance(raw_tokens, (str, bytes)):
            raise ValueError("Kimi K3 prompt prefix must be one-dimensional")
        tokens = _token_tuple(
            cast(Sequence[int], raw_tokens),
            expected=len(prompt_prefix),
            name="Kimi K3 prompt prefix",
        )
        feature_contract = (
            int(self.compact_greedy),
            int(self.loaded.config.aux_only_prefill),
        )
        if self.loaded.config.packed_agreements:
            feature_contract = (*feature_contract, 1)
        token_fingerprint = _token_contract_fingerprint(
            tokens,
            self.banned_token_ids,
            self.terminal_token_ids,
            feature_contract,
        )
        request_fingerprint = _request_control_fingerprint(
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
            stop_sequences=stop_sequences,
            distributed_progress=distributed_progress,
        )
        return tokens, (*token_fingerprint, *request_fingerprint)

    def seed_prompt(
        self,
        prompt_prefix: mx.array,
        *,
        prefill_step_size: int,
        max_tokens: int,
        stop_sequences: Sequence[str],
        progress_callback: Callable[[int, int], None],
        distributed_progress_callback: Callable[[], None] | None,
    ) -> tuple[float, int]:
        prompt_tokens, prompt_fingerprint = self._agreed_operation(
            "prompt contract validation",
            lambda: self._prompt_contract(
                prompt_prefix,
                max_tokens=max_tokens,
                prefill_step_size=prefill_step_size,
                stop_sequences=stop_sequences,
                distributed_progress=distributed_progress_callback is not None,
            ),
        )
        total = len(prompt_tokens)
        contract_values = (
            total,
            prefill_step_size,
            max_tokens,
            *prompt_fingerprint,
        )
        for contract_value in contract_values:
            if self.collective.agree_token(contract_value) != contract_value:
                raise DSparkDistributedStateError(
                    "Kimi K3 DSpark prompt/request contract disagreed across ranks; "
                    "no target TP graph was built"
                ) from None
        aux_prefill_forward: (
            Callable[[mx.array, object, tuple[int, ...]], object] | None
        ) = None
        if self.loaded.config.aux_only_prefill:
            # The opt-in bit is already bound into the agreed prompt digest, so
            # every rank enters this preflight or none do. Feature asymmetry is
            # agreed before a target graph capable of TP collective entry exists.
            aux_prefill_forward = self._agreed_operation(
                "target auxiliary prompt prefill API preflight",
                self._target_aux_prefill_forward,
            )
        processed = 0
        started = self.clock()
        self.agree_local_side_effect(
            "initial prompt progress publication",
            lambda: progress_callback(0, total),
        )
        while processed < total:
            chunk_size = min(prefill_step_size, total - processed)
            next_processed = processed + chunk_size
            chunk_tokens = prompt_tokens[processed:next_processed]
            chunk_contract = (
                processed,
                next_processed,
                chunk_size,
                *_token_contract_fingerprint(chunk_tokens),
            )
            for contract_value in chunk_contract:
                if self.collective.agree_token(contract_value) != contract_value:
                    raise DSparkDistributedStateError(
                        "Kimi K3 DSpark prompt chunk contract disagreed across "
                        "ranks; no target TP graph was built"
                    ) from None
            chunk = self._agreed_operation(
                "prompt chunk construction",
                lambda processed=processed, chunk_size=chunk_size: prompt_prefix[
                    processed : processed + chunk_size
                ][None],
            )
            initial_offset = self._agreed_operation(
                "target prompt readiness",
                lambda chunk=chunk: self._validate_forward_readiness(chunk),
            )
            if self.collective.agree_token(initial_offset) != initial_offset:
                raise DSparkDistributedStateError(
                    "Kimi K3 DSpark target prompt offsets disagreed across ranks; "
                    "no target TP graph was built"
                ) from None
            prompt_aux_hidden_states: Sequence[object]
            if aux_prefill_forward is None:
                pending_forward = self._agreed_operation(
                    "target prompt graph build",
                    lambda chunk=chunk, initial_offset=initial_offset: (
                        self._build_forward_with_taps(
                            chunk,
                            initial_offset=initial_offset,
                        )
                    ),
                )
                forward_result = self._agreed_operation(
                    "target prompt materialization",
                    pending_forward.materialize,
                )
                prompt_aux_hidden_states = forward_result.aux_hidden_states
            else:
                pending_aux_prefill = self._agreed_operation(
                    "target auxiliary prompt prefill graph build",
                    lambda chunk=chunk,
                    initial_offset=initial_offset,
                    aux_prefill_forward=aux_prefill_forward: (
                        self._build_aux_prefill_with_taps(
                            chunk,
                            initial_offset=initial_offset,
                            forward=aux_prefill_forward,
                        )
                    ),
                )
                prompt_aux_hidden_states = self._agreed_operation(
                    "target auxiliary prompt prefill materialization",
                    pending_aux_prefill.materialize,
                )
            self._agreed_operation(
                "draft prompt projection",
                lambda prompt_aux_hidden_states=prompt_aux_hidden_states,
                chunk_size=chunk_size: (
                    self.draft.seed_target_context(
                        prompt_aux_hidden_states,
                        expected_width=chunk_size,
                    )
                ),
            )
            processed = next_processed
            if distributed_progress_callback is not None:
                self.agree_local_side_effect(
                    "distributed prompt progress publication",
                    distributed_progress_callback,
                )
            self.agree_local_side_effect(
                "prompt progress publication",
                lambda processed=processed: progress_callback(processed, total),
            )
            try:
                mx.clear_cache()
            except Exception:
                _log_nonfatal_warning("Kimi K3 DSpark prompt cache cleanup failed")

        def validate_complete_prompt() -> None:
            if _target_cache_offset(self.target_cache) != total:
                raise DSparkDistributedStateError(
                    "Kimi K3 target prompt cache did not reach the decode boundary"
                )
            if _context_cache_offset(self.draft.context_cache) != total:
                raise DSparkDistributedStateError(
                    "Kimi K3 draft prompt context did not reach the decode boundary"
                )

        self._agreed_operation(
            "prompt completion validation",
            validate_complete_prompt,
        )
        elapsed = self.clock() - started
        return (total / elapsed if elapsed > 0 else 0.0), total

    def _verification_plan(
        self,
        proposal_block: tuple[int, ...],
    ) -> TargetVerificationPlan:
        _token_tuple(
            proposal_block,
            expected=self.loaded.verify_width,
            name="Kimi K3 target verification input",
        )
        supports_compact = getattr(
            self.target_model,
            "supports_vocab_parallel_greedy",
            None,
        )
        compact_verify = getattr(
            self.target_model,
            "forward_with_aux_hidden_states_greedy",
            None,
        )
        compact_available = (
            callable(supports_compact)
            and bool(supports_compact())
            and callable(compact_verify)
        )
        if self.compact_greedy and not compact_available:
            raise DSparkFeatureUnavailableError(
                "Kimi K3 compact verifier was requested but the target rank "
                "does not expose its exact vocab-parallel greedy API"
            )
        return TargetVerificationPlan("compact" if self.compact_greedy else "full")

    def _build_verification(
        self,
        proposal_block: tuple[int, ...],
        plan: TargetVerificationPlan,
    ) -> _TargetPosteriorGraph:
        proposal_block = _token_tuple(
            proposal_block,
            expected=self.loaded.verify_width,
            name="Kimi K3 target verification input",
        )
        if self._verification_plan(proposal_block) != plan:
            raise ValueError("Kimi K3 target verifier mode changed after agreement")
        input_ids = mx.array([proposal_block], dtype=mx.int32)
        initial_offset = self._validate_forward_readiness(
            input_ids,
            speculative_width=len(proposal_block),
        )
        if plan.mode == "compact":
            forward = cast(
                _CompactTargetForwardResult,
                cast(
                    _TargetWithAuxGreedyForward,
                    self.target_model,
                ).forward_with_aux_hidden_states_greedy(
                    input_ids,
                    self.target_cache,
                    self.loaded.config.target_hidden_state_indices,
                    self.banned_token_ids,
                ),
            )
            validated_aux = _validated_aux_hidden_states(
                forward.aux_hidden_states,
                expected_width=len(proposal_block),
            )
            tokens = _validate_batched_greedy_tokens(
                forward.tokens,
                expected_width=len(proposal_block),
            )
            return _BuiltKimiK3CompactTargetPosterior(
                tokens=tokens,
                aux_hidden_states=validated_aux,
                target_cache=self.target_cache,
                initial_offset=initial_offset,
                width=len(proposal_block),
                evaluate=self.evaluate,
            )
        forward = self._build_forward_with_taps(
            input_ids,
            initial_offset=initial_offset,
            speculative_width=len(proposal_block),
        )
        tokens = _build_greedy_dspark_posterior_tokens(
            forward.forward.logits,
            expected_width=len(proposal_block),
            banned_token_ids=self.banned_token_ids,
        )
        return _BuiltKimiK3TargetPosterior(
            forward,
            tokens,
            len(proposal_block),
        )

    def _preflight_ordinary(self, anchor_token: int) -> OrdinaryDecodePlan:
        if type(anchor_token) is not int or not 0 <= anchor_token <= 0x7FFFFFFF:
            raise ValueError("Kimi K3 ordinary anchor must fit non-negative int32")
        initial_offset = _target_cache_offset(self.target_cache)
        _validate_target_cache(
            self.target_cache,
            expected_offset=initial_offset,
            require_kda_state=True,
        )
        supports_compact = getattr(
            self.target_model,
            "supports_vocab_parallel_greedy",
            None,
        )
        compact_greedy = getattr(self.target_model, "vocab_parallel_greedy", None)
        compact_requested = self.compact_greedy and not self.banned_token_ids
        if compact_requested and (
            not callable(supports_compact)
            or not bool(supports_compact())
            or not callable(compact_greedy)
        ):
            raise DSparkFeatureUnavailableError(
                "Kimi K3 compact vocab-parallel greedy was requested but the "
                "target rank does not support it"
            )
        mode: Literal["full", "compact"] = "compact" if compact_requested else "full"
        if mode == "full" and not callable(self.target_model):
            raise DSparkFeatureUnavailableError(
                "Kimi K3 target does not expose the ordinary full-logits path"
            )
        return OrdinaryDecodePlan(initial_offset, mode)

    def _prepare_ordinary(
        self,
        anchor_token: int,
        plan: OrdinaryDecodePlan,
    ) -> PreparedOrdinaryDecode:
        local_plan = self._preflight_ordinary(anchor_token)
        if local_plan != plan:
            raise ValueError("Kimi K3 ordinary decode plan changed after agreement")
        input_ids = mx.array([[anchor_token]], dtype=mx.int32)
        if plan.mode == "compact":
            sampled = cast(
                _CompactGreedyTarget, self.target_model
            ).vocab_parallel_greedy(
                input_ids,
                cache=self.target_cache,
            )
            shape = getattr(sampled, "shape", None)
            item = getattr(sampled, "item", None)
            if _shape_tuple(shape) != (1,) or not callable(item):
                raise ValueError("Kimi K3 compact greedy target must return one token")
            return _BuiltKimiK3OrdinaryDecode(
                output=sampled,
                mode="compact",
                target_cache=self.target_cache,
                expected_offset=plan.initial_offset + 1,
                evaluate=self.evaluate,
            )
        logits = cast(Callable[..., object], self.target_model)(
            input_ids,
            cache=self.target_cache,
        )
        _validate_target_logits(logits, expected_width=1)
        tokens = _build_greedy_dspark_posterior_tokens(
            logits,
            expected_width=1,
            banned_token_ids=self.banned_token_ids,
        )
        return _BuiltKimiK3OrdinaryDecode(
            output=tokens,
            mode="full",
            target_cache=self.target_cache,
            expected_offset=plan.initial_offset + 1,
            evaluate=self.evaluate,
        )

    def make_round_engine(
        self,
        telemetry_sink: Callable[[DSparkRoundTelemetry], None] | None = None,
    ) -> KimiK3DSparkRoundEngine:
        target = ReplaySSMTargetAdapter(
            self.target_model,
            self.target_cache,
            self._verification_plan,
            self._build_verification,
            self._preflight_ordinary,
            self._prepare_ordinary,
            validate_closed=lambda expected_offset: _validate_target_cache(
                self.target_cache,
                expected_offset=expected_offset,
                require_kda_state=True,
            ),
            validate_open=lambda expected_offset, width: _validate_target_cache(
                self.target_cache,
                expected_offset=expected_offset,
                require_kda_state=True,
                speculative_phase="open",
                speculative_width=width,
            ),
        )
        recorder: DSparkConfidenceRecorder | None = None
        capture_config = self.loaded.config.confidence_capture
        if capture_config is not None and self.collective.rank == 0:
            assert self.confidence_request is not None
            if self._confidence_recorder is None:
                self._confidence_recorder = DSparkConfidenceJSONLRecorder(
                    capture_config,
                    self.confidence_request,
                    verify_width=self.loaded.verify_width,
                    target_route_top_k=cast(int, self.loaded.target_route_top_k),
                )
            recorder = self._confidence_recorder
        return KimiK3DSparkRoundEngine(
            config=self.loaded.config,
            draft=self.draft,
            target=target,
            collective=self.collective,
            terminal_token_ids=self.terminal_token_ids,
            telemetry_sink=telemetry_sink,
            confidence_recorder=recorder,
            clock=self.clock,
        )

    @property
    def confidence_capture_enabled(self) -> bool:
        return self.loaded.config.confidence_capture is not None

    def finalize_confidence_capture(self, *, complete: bool) -> None:
        if self._confidence_recorder is not None:
            self._confidence_recorder.finalize(complete=complete)

    def log_packed_agreement_attestation(self, *, required: bool = False) -> None:
        """Publish cumulative request-local row counts without a collective."""

        if not self.loaded.config.packed_agreements:
            if required:
                raise RuntimeError(
                    "Kimi K3 packed agreement attestation is required but disabled"
                )
            return
        try:
            attestation = cast(
                MlxRankAgreement,
                self.collective,
            ).packed_agreement_attestation
            logger.info(
                "MLX Kimi K3 packed agreement attestation: "
                f"rank={self.collective.rank}, "
                f"row_calls={attestation.row_calls}, "
                f"packed_row_calls={attestation.packed_row_calls}, "
                f"legacy_row_calls={attestation.legacy_row_calls}, "
                f"physical_all_gathers={attestation.physical_all_gathers}"
            )
        except Exception as error:
            if required:
                raise RuntimeError(
                    "Kimi K3 packed agreement attestation logging failed"
                ) from error
            _log_nonfatal_warning("Kimi K3 packed agreement attestation logging failed")


@dataclass(frozen=True)
class DSparkDecodedToken:
    token: int
    from_draft: bool
    finish_reason: Literal["stop", "length"] | None


class DSparkRoundDecoder(Protocol):
    @property
    def verify_width(self) -> int: ...

    def decode_round(
        self,
        anchor_token: int,
        *,
        remaining: int | None = None,
    ) -> DSparkRoundResult: ...

    def decode_ordinary_tail(self, anchor_token: int) -> DSparkRoundResult: ...


def dspark_decode_tokens(
    engine: DSparkRoundDecoder,
    *,
    anchor_token: int,
    max_tokens: int,
    eos_token_ids: Sequence[int],
    force_ordinary: bool = False,
    round_observer: Callable[[DSparkRoundTelemetry], None] | None = None,
    token_observer: Callable[[bool], None] | None = None,
) -> Iterator[DSparkDecodedToken]:
    """Flatten committed rounds while preserving earliest EOS and length limits."""

    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("Kimi K3 DSpark max tokens must be positive")
    eos = frozenset(eos_token_ids)
    emitted = 0
    next_anchor = anchor_token
    while emitted < max_tokens:
        remaining = max_tokens - emitted
        result = (
            engine.decode_ordinary_tail(next_anchor)
            if force_ordinary or remaining < engine.verify_width
            else engine.decode_round(next_anchor, remaining=remaining)
        )
        if round_observer is not None:
            try:
                round_observer(result.telemetry)
            except Exception:
                _log_nonfatal_warning("Kimi K3 DSpark round observer failed")
        if not result.emitted_tokens:
            raise DSparkDistributedStateError(
                "Kimi K3 DSpark round committed no output token"
            )
        for round_token_index, token in enumerate(result.emitted_tokens):
            emitted += 1
            next_anchor = token
            from_draft = round_token_index < result.telemetry.accepted
            if token_observer is not None:
                try:
                    token_observer(from_draft)
                except Exception:
                    _log_nonfatal_warning("Kimi K3 DSpark token observer failed")
            if token in eos:
                yield DSparkDecodedToken(token, from_draft, "stop")
                return
            if emitted == max_tokens:
                yield DSparkDecodedToken(token, from_draft, "length")
                return
            yield DSparkDecodedToken(token, from_draft, None)
