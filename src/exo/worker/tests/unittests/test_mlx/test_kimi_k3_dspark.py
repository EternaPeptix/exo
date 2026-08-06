from __future__ import annotations

import hashlib
import json
import stat
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.generator import kimi_k3_dspark as dspark_module
from exo.worker.engines.mlx.generator.kimi_k3_dspark import (
    DSPARK_ADAPTIVE_GATE_ENV,
    DSPARK_ADAPTIVE_GATE_POLICY_ENV,
    DSPARK_ADAPTIVE_GATE_POLICY_SHA256_ENV,
    DSPARK_AUX_ONLY_PREFILL_ENV,
    DSPARK_CHECKPOINT_ENV,
    DSPARK_CONFIDENCE_JSONL_ENV,
    DSPARK_CONFIDENCE_SESSION_ENV,
    DSPARK_CONSERVATIVE_VERIFY_WIDTH,
    DSPARK_DUAL_PROPOSER_ENV,
    DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS,
    DSPARK_ENABLE_ENV,
    DSPARK_MODEL_NATIVE_VERIFY_WIDTH,
    DSPARK_PACKED_AGREEMENTS_ENV,
    DSPARK_PREFIX_CACHE_ENV,
    DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV,
    DSPARK_TELEMETRY_ENV,
    DSPARK_VERIFY_WIDTH_ENV,
    DSPARK_YARN_CHECKPOINT_ENV,
    MLX_DSPARK_PROPOSER_ENV,
    MLX_DSPARK_SEGMENTED_SDPA_ENV,
    MLX_REPLAYSSM_ENV,
    PACKED_AGREEMENT_ROW_WIDTH,
    DSparkCancellationError,
    DSparkConfidenceCaptureConfig,
    DSparkConfidenceCaptureRequest,
    DSparkConfidenceJSONLRecorder,
    DSparkConfidenceObservation,
    DSparkConfigurationError,
    DSparkDistributedStateError,
    DSparkFeatureUnavailableError,
    DSparkRoundResult,
    DSparkRoundTelemetry,
    KimiK3DSparkCheckpointContract,
    KimiK3DSparkConfig,
    KimiK3DSparkDualConfig,
    KimiK3DSparkPrefixCache,
    KimiK3DSparkRequestRuntime,
    KimiK3DSparkRoundEngine,
    LoadedMlxDSpark,
    LoadedMlxDSparkDual,
    MlxDSparkFeatures,
    MlxDSparkRequestDraft,
    MlxRankAgreement,
    OrdinaryDecodePlan,
    PackedRankAgreement,
    ReplaySSMTargetAdapter,
    TargetPosterior,
    TargetVerificationPlan,
    accepted_draft_prefix,
    attest_kimi_k3_target_route_top_k,
    detect_mlx_dspark_features,
    dspark_context_bucket,
    dspark_context_capacity_hint,
    dspark_decode_tokens,
    dspark_prompt_identity,
    has_replayssm_target_hooks,
    kimi_k3_dspark_config,
    load_replicated_mlx_dspark,
    load_replicated_mlx_dspark_dual,
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


def _checkpoint_contract(*, yarn: bool) -> KimiK3DSparkCheckpointContract:
    return KimiK3DSparkCheckpointContract(
        revision=(
            dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_REVISION
            if yarn
            else dspark_module.RADIXARK_KIMI_K3_DSPARK_REVISION
        ),
        config_sha256=(
            dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256
            if yarn
            else dspark_module.RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256
        ),
        model_bytes=dspark_module.RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES,
        model_sha256=(
            dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256
            if yarn
            else dspark_module.RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256
        ),
    )


def _dual_environment(old: Path, yarn: Path) -> dict[str, str]:
    environment = _enabled_environment(old)
    environment.update(
        {
            DSPARK_DUAL_PROPOSER_ENV: "1",
            DSPARK_YARN_CHECKPOINT_ENV: str(yarn),
        }
    )
    return environment


def _dual_config(
    tmp_path: Path,
    *,
    width: Literal[3, 8] = 8,
) -> KimiK3DSparkDualConfig:
    old_path = tmp_path / "old"
    yarn_path = tmp_path / "yarn"
    old = KimiK3DSparkConfig(
        checkpoint_path=old_path,
        verify_width=width,
        round_telemetry=False,
    )
    yarn = KimiK3DSparkConfig(
        checkpoint_path=yarn_path,
        verify_width=width,
        round_telemetry=False,
        revision=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_REVISION,
        config_sha256=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256,
        model_sha256=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256,
    )
    return KimiK3DSparkDualConfig(old=old, yarn=yarn)


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

    with pytest.raises(
        DSparkConfigurationError,
        match=(
            f"{DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV} requires {DSPARK_ENABLE_ENV}=1"
        ),
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ={DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV: "1"},
        )

    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_PACKED_AGREEMENTS_ENV} requires {DSPARK_ENABLE_ENV}=1",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ={DSPARK_PACKED_AGREEMENTS_ENV: "1"},
        )


def test_dual_proposer_is_default_off_and_preserves_single_config(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old"
    calls: list[Path] = []

    def validate(path: Path) -> KimiK3DSparkCheckpointContract:
        calls.append(path)
        return _checkpoint_contract(yarn=False)

    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=_enabled_environment(old_path),
        checkpoint_validator=validate,
    )

    assert type(config) is KimiK3DSparkConfig
    assert config.checkpoint_path == old_path
    assert config.revision == dspark_module.RADIXARK_KIMI_K3_DSPARK_REVISION
    assert config.rank_zero_proposal_recovery is False
    assert config.packed_agreements is False
    assert calls == [old_path]

    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_YARN_CHECKPOINT_ENV} requires {DSPARK_DUAL_PROPOSER_ENV}=1",
    ):
        environment = _enabled_environment(old_path)
        environment[DSPARK_YARN_CHECKPOINT_ENV] = str(tmp_path / "yarn")
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=validate,
        )


def test_rank_zero_proposal_recovery_requires_explicit_selector(
    tmp_path: Path,
) -> None:
    environment = _enabled_environment(tmp_path)
    environment[DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV] = "1"

    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=environment,
        checkpoint_validator=lambda _path: None,
    )

    assert type(config) is KimiK3DSparkConfig
    assert config.rank_zero_proposal_recovery is True


def test_packed_agreements_require_strict_explicit_selector(tmp_path: Path) -> None:
    environment = _enabled_environment(tmp_path)
    environment[DSPARK_PACKED_AGREEMENTS_ENV] = "1"
    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=environment,
        checkpoint_validator=lambda _path: None,
    )

    assert type(config) is KimiK3DSparkConfig
    assert config.packed_agreements is True

    environment[DSPARK_PACKED_AGREEMENTS_ENV] = "true"
    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_PACKED_AGREEMENTS_ENV} must be 0 or 1",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )


def test_dual_proposer_requires_exact_old_and_yarn_checkpoint_pair(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old"
    yarn_path = tmp_path / "yarn"
    missing_yarn = _enabled_environment(old_path)
    missing_yarn[DSPARK_DUAL_PROPOSER_ENV] = "1"
    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_YARN_CHECKPOINT_ENV} is required",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=missing_yarn,
            checkpoint_validator=lambda _path: _checkpoint_contract(yarn=False),
        )

    contracts = {
        old_path: _checkpoint_contract(yarn=False),
        yarn_path: _checkpoint_contract(yarn=True),
    }
    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=_dual_environment(old_path, yarn_path),
        checkpoint_validator=contracts.__getitem__,
    )

    assert isinstance(config, KimiK3DSparkDualConfig)
    assert config.old.checkpoint_path == old_path
    assert config.yarn.checkpoint_path == yarn_path
    assert config.threshold_tokens == DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS

    swapped = {
        old_path: _checkpoint_contract(yarn=True),
        yarn_path: _checkpoint_contract(yarn=False),
    }
    with pytest.raises(DSparkConfigurationError, match="pinned old revision"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=_dual_environment(old_path, yarn_path),
            checkpoint_validator=swapped.__getitem__,
        )

    hybrid_old = KimiK3DSparkCheckpointContract(
        revision=dspark_module.RADIXARK_KIMI_K3_DSPARK_REVISION,
        config_sha256=dspark_module.RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256,
        model_bytes=dspark_module.RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES,
        model_sha256=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256,
    )
    with pytest.raises(DSparkConfigurationError, match="pinned old revision"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=_dual_environment(old_path, yarn_path),
            checkpoint_validator=lambda path: (
                hybrid_old if path == old_path else _checkpoint_contract(yarn=True)
            ),
        )

    with pytest.raises(DSparkConfigurationError, match="pinned old revision"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=_dual_environment(old_path, yarn_path),
            checkpoint_validator=lambda _path: None,
        )

    with pytest.raises(DSparkConfigurationError, match="distinct directories"):
        same_path_contracts = iter(
            (_checkpoint_contract(yarn=False), _checkpoint_contract(yarn=True))
        )
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=_dual_environment(old_path, old_path),
            checkpoint_validator=lambda _path: next(same_path_contracts),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (DSPARK_PREFIX_CACHE_ENV, "1"),
        (DSPARK_ADAPTIVE_GATE_ENV, "1"),
        (DSPARK_ADAPTIVE_GATE_POLICY_ENV, "/tmp/policy.json"),
        (DSPARK_ADAPTIVE_GATE_POLICY_SHA256_ENV, "0" * 64),
        (DSPARK_CONFIDENCE_JSONL_ENV, "/tmp/capture.jsonl"),
    ],
)
def test_dual_proposer_rejects_unvalidated_feature_combinations(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    old_path = tmp_path / "old"
    yarn_path = tmp_path / "yarn"
    environment = _dual_environment(old_path, yarn_path)
    environment[name] = value

    with pytest.raises(DSparkConfigurationError, match="cannot be combined"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda path: _checkpoint_contract(
                yarn=path == yarn_path
            ),
        )


def test_dual_proposer_threshold_is_fixed_not_operator_tunable(
    tmp_path: Path,
) -> None:
    config = _dual_config(tmp_path)

    with pytest.raises(DSparkConfigurationError, match="exactly 8192"):
        KimiK3DSparkDualConfig(
            old=config.old,
            yarn=config.yarn,
            threshold_tokens=DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS + 1,
        )


@pytest.mark.parametrize(
    ("old_prefix", "yarn_prefix"),
    [(True, False), (False, True)],
)
def test_dual_config_rejects_asymmetric_prefix_cache_contract(
    tmp_path: Path,
    old_prefix: bool,
    yarn_prefix: bool,
) -> None:
    config = _dual_config(tmp_path)

    with pytest.raises(DSparkConfigurationError, match="runtime contracts disagree"):
        KimiK3DSparkDualConfig(
            old=replace(config.old, prefix_cache=old_prefix),
            yarn=replace(config.yarn, prefix_cache=yarn_prefix),
        )


def test_dual_config_rejects_paired_prefix_even_when_both_configs_match(
    tmp_path: Path,
) -> None:
    config = _dual_config(tmp_path)

    with pytest.raises(DSparkConfigurationError, match="paired prefix cache"):
        KimiK3DSparkDualConfig(
            old=replace(config.old, prefix_cache=True),
            yarn=replace(config.yarn, prefix_cache=True),
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
    assert not config.aux_only_prefill
    assert validated == [tmp_path]
    assert warnings == []


def test_conservative_confidence_capture_requires_complete_strict_metadata(
    tmp_path: Path,
) -> None:
    environment = _enabled_environment(tmp_path, width="3")
    environment.update(
        {
            DSPARK_AUX_ONLY_PREFILL_ENV: "1",
            DSPARK_CONFIDENCE_JSONL_ENV: str(tmp_path / "rounds.jsonl"),
            DSPARK_CONFIDENCE_SESSION_ENV: "top8-canary-20260806",
        }
    )

    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=environment,
        checkpoint_validator=lambda _path: None,
    )

    assert config is not None
    assert config.verify_width == 3
    assert config.aux_only_prefill is True
    assert config.confidence_capture == DSparkConfidenceCaptureConfig(
        jsonl_path=tmp_path / "rounds.jsonl",
        session_id="top8-canary-20260806",
    )

    del environment[DSPARK_CONFIDENCE_SESSION_ENV]
    with pytest.raises(DSparkConfigurationError, match="is required"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )


def test_confidence_capture_rejects_width_eight_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    environment = _enabled_environment(tmp_path, width="8")
    environment.update(
        {
            DSPARK_CONFIDENCE_JSONL_ENV: str(tmp_path / "rounds.jsonl"),
            DSPARK_CONFIDENCE_SESSION_ENV: "capture-1",
        }
    )
    with pytest.raises(DSparkConfigurationError, match="conservative verify width 3"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )

    environment[DSPARK_VERIFY_WIDTH_ENV] = "3"
    environment[DSPARK_CONFIDENCE_JSONL_ENV] = "relative.jsonl"
    with pytest.raises(DSparkConfigurationError, match="absolute path"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )

    capture_path = tmp_path / "world-readable.jsonl"
    capture_path.write_text("", encoding="utf-8")
    capture_path.chmod(0o644)
    environment[DSPARK_CONFIDENCE_JSONL_ENV] = str(capture_path)
    with pytest.raises(DSparkConfigurationError, match="group or other access"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )


@pytest.mark.parametrize("raw", ["", "2", "true", " 1", "1 "])
def test_aux_only_prefill_flag_is_strict_and_default_off(
    tmp_path: Path,
    raw: str,
) -> None:
    environment = _enabled_environment(tmp_path)
    environment[DSPARK_AUX_ONLY_PREFILL_ENV] = raw
    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_AUX_ONLY_PREFILL_ENV} must be 0 or 1",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )

    environment[DSPARK_AUX_ONLY_PREFILL_ENV] = "1"
    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=environment,
        checkpoint_validator=lambda _path: None,
    )
    assert config is not None
    assert config.aux_only_prefill


@pytest.mark.parametrize("raw", ["", "2", "true", " 1", "1 "])
def test_paired_prefix_cache_flag_is_strict_and_default_off(
    tmp_path: Path,
    raw: str,
) -> None:
    environment = _enabled_environment(tmp_path)
    default_config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=environment,
        checkpoint_validator=lambda _path: None,
    )
    assert default_config is not None
    assert not default_config.prefix_cache

    environment[DSPARK_PREFIX_CACHE_ENV] = raw
    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_PREFIX_CACHE_ENV} must be 0 or 1",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )


def test_paired_prefix_cache_requires_dspark_and_explicitly_enables(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DSparkConfigurationError,
        match=f"{DSPARK_PREFIX_CACHE_ENV} requires {DSPARK_ENABLE_ENV}=1",
    ):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ={DSPARK_PREFIX_CACHE_ENV: "1"},
        )

    environment = _enabled_environment(tmp_path)
    environment[DSPARK_PREFIX_CACHE_ENV] = "1"
    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=environment,
        checkpoint_validator=lambda _path: None,
    )
    assert config is not None
    assert config.prefix_cache


def test_paired_prefix_cache_rejects_confidence_capture(
    tmp_path: Path,
) -> None:
    environment = _enabled_environment(
        tmp_path,
        width=str(DSPARK_CONSERVATIVE_VERIFY_WIDTH),
    )
    environment.update(
        {
            DSPARK_PREFIX_CACHE_ENV: "1",
            DSPARK_CONFIDENCE_JSONL_ENV: str(tmp_path / "capture.jsonl"),
            DSPARK_CONFIDENCE_SESSION_ENV: "prefix-capture",
        }
    )

    with pytest.raises(DSparkConfigurationError, match="confidence capture"):
        kimi_k3_dspark_config(
            is_pipeline=False,
            is_batch=False,
            environ=environment,
            checkpoint_validator=lambda _path: None,
        )


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


@pytest.mark.parametrize(
    ("config_sha256", "revision", "model_sha256"),
    [
        (
            dspark_module.RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256,
            dspark_module.RADIXARK_KIMI_K3_DSPARK_REVISION,
            dspark_module.RADIXARK_KIMI_K3_DSPARK_MODEL_SHA256,
        ),
        (
            dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256,
            dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_REVISION,
            dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256,
        ),
    ],
)
def test_local_checkpoint_validation_selects_only_audited_pairs(
    tmp_path: Path,
    config_sha256: str,
    revision: str,
    model_sha256: str,
) -> None:
    (tmp_path / "config.json").write_bytes(b"audited config")
    (tmp_path / "model.safetensors").write_bytes(b"x")

    contract = validate_local_dspark_checkpoint(
        tmp_path,
        expected_model_bytes=1,
        sha256=lambda _path: config_sha256,
    )

    assert contract is not None
    assert contract.revision == revision
    assert contract.config_sha256 == config_sha256
    assert contract.model_sha256 == model_sha256


def test_local_checkpoint_validation_rejects_unknown_pair(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_bytes(b"unknown config")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(DSparkConfigurationError, match="audited config hash"):
        validate_local_dspark_checkpoint(
            tmp_path,
            expected_model_bytes=1,
            sha256=lambda _path: "f" * 64,
        )


def test_yarn_checkpoint_identity_is_bound_into_runtime_config(tmp_path: Path) -> None:
    yarn_contract = dspark_module.KimiK3DSparkCheckpointContract(
        revision=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_REVISION,
        config_sha256=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256,
        model_bytes=dspark_module.RADIXARK_KIMI_K3_DSPARK_MODEL_BYTES,
        model_sha256=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256,
    )
    config = kimi_k3_dspark_config(
        is_pipeline=False,
        is_batch=False,
        environ=_enabled_environment(tmp_path, width="3"),
        checkpoint_validator=lambda _path: yarn_contract,
    )

    assert config is not None
    assert config.revision == yarn_contract.revision
    assert config.config_sha256 == yarn_contract.config_sha256
    assert config.model_bytes == yarn_contract.model_bytes
    assert config.model_sha256 == yarn_contract.model_sha256


@dataclass
class _FakeDraftRound:
    proposal_tokens: Sequence[int]
    events: list[str]
    confidence_logits: Sequence[float] | None = None
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
    confidence_logits: Sequence[float] | None = None
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
            confidence_logits=self.owner.confidence_logits,
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
    reject_token_call: int | None = None
    reject_packed_call: int | None = None
    token_calls: int = field(default=0, init=False)
    packed_calls: list[tuple[str, bool, int, tuple[int, ...]]] = field(
        default_factory=list,
        init=False,
    )

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
        self.token_calls += 1
        if self.token_calls == self.reject_token_call:
            return None
        if self.agreed_tokens:
            return self.agreed_tokens.pop(0)
        return local_token

    def agree_packed(
        self,
        name: str,
        *,
        local_success: bool,
        error_fingerprint: int,
        payload: tuple[int, ...] = (),
    ) -> PackedRankAgreement:
        self.packed_calls.append((name, local_success, error_fingerprint, payload))
        if len(self.packed_calls) == self.reject_packed_call:
            return PackedRankAgreement(None, None, None)
        return PackedRankAgreement(local_success, error_fingerprint, payload)


def _config(
    tmp_path: Path,
    width: int,
    *,
    aux_only_prefill: bool = False,
    rank_zero_proposal_recovery: bool = False,
    packed_agreements: bool = False,
    prefix_cache: bool = False,
) -> KimiK3DSparkConfig:
    assert width in (3, 8)
    return KimiK3DSparkConfig(
        checkpoint_path=tmp_path,
        verify_width=width,
        round_telemetry=False,
        aux_only_prefill=aux_only_prefill,
        rank_zero_proposal_recovery=rank_zero_proposal_recovery,
        packed_agreements=packed_agreements,
        prefix_cache=prefix_cache,
    )


def _engine(
    tmp_path: Path,
    *,
    proposals: Sequence[int],
    posterior: Sequence[int],
    agreement: _FakeAgreement | None = None,
    terminal_token_ids: tuple[int, ...] = (),
    packed_agreements: bool = False,
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
        config=_config(tmp_path, width, packed_agreements=packed_agreements),
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
    engine, draft, target, collective, events = _engine(
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
    assert collective.token_calls == 10
    assert len([event for event in events if event.startswith("agree_stage_")]) == 6
    assert collective.packed_calls == []


def test_packed_width_three_round_uses_nine_agreement_rows(tmp_path: Path) -> None:
    agreement = _FakeAgreement([])
    engine, _draft, _target, _collective, events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
        packed_agreements=True,
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (11, 12, 13)
    assert agreement.token_calls == 1
    assert len(agreement.packed_calls) == 6
    assert [call[0] for call in agreement.packed_calls] == [
        "stage:draft preflight",
        "stage:draft graph build",
        "stage:target transaction prepare",
        "stage:target verification graph build",
        "stage:target commit",
        "stage:draft commit",
    ]
    assert events.count("agree_proposal") == 1
    assert events.count("agree_acceptance") == 1
    assert agreement.token_calls + len(agreement.packed_calls) + 2 == 9


def test_packed_ordinary_tail_uses_five_agreement_rows(tmp_path: Path) -> None:
    agreement = _FakeAgreement([])
    engine, _draft, _target, _collective, _events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
        packed_agreements=True,
    )

    result = engine.decode_ordinary_tail(10)

    assert result.emitted_tokens == (999,)
    assert agreement.token_calls == 2
    assert [call[0] for call in agreement.packed_calls] == [
        "stage:ordinary target preflight",
        "stage:ordinary target graph build",
        "stage:ordinary target materialization",
    ]
    assert agreement.token_calls + len(agreement.packed_calls) == 5


def test_packed_ordinary_preflight_mismatch_prevents_graph_build(
    tmp_path: Path,
) -> None:
    agreement = _FakeAgreement([], reject_packed_call=1)
    engine, _draft, _target, _collective, events = _engine(
        tmp_path,
        proposals=(11, 12),
        posterior=(11, 12, 13),
        agreement=agreement,
        packed_agreements=True,
    )

    with pytest.raises(DSparkDistributedStateError, match="readiness disagreed"):
        engine.decode_ordinary_tail(10)

    assert "ordinary_build" not in events
    assert "ordinary_decode" not in events


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


@dataclass
class _FakeConfidenceRecorder:
    records: list[
        tuple[DSparkRoundTelemetry, DSparkConfidenceObservation | None, float]
    ] = field(default_factory=list)
    events: list[str] | None = None

    def record(
        self,
        telemetry: DSparkRoundTelemetry,
        observation: DSparkConfidenceObservation | None,
        *,
        total_step_ms: float,
    ) -> None:
        if self.events is not None:
            self.events.append("confidence_record")
        self.records.append((telemetry, observation, total_step_ms))

    def finalize(self, *, complete: bool) -> None:
        del complete


def test_rank_zero_capture_labels_conservative_agreed_prefix_after_commits(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    confidence = (-1.5, 0.75)
    draft = _FakeDraft(
        (21, 22),
        3,
        events,
        confidence_logits=confidence,
    )
    target = _FakeTarget((21, 99, 100), events)
    recorder = _FakeConfidenceRecorder(events=events)
    config = KimiK3DSparkConfig(
        checkpoint_path=tmp_path,
        verify_width=3,
        round_telemetry=False,
        aux_only_prefill=True,
        confidence_capture=DSparkConfidenceCaptureConfig(
            tmp_path / "rounds.jsonl",
            "capture-1",
        ),
    )
    engine = KimiK3DSparkRoundEngine(
        config=config,
        draft=draft,
        target=target,
        collective=_FakeAgreement(events),
        confidence_recorder=recorder,
    )

    result = engine.decode_round(20)

    assert result.emitted_tokens == (21, 99)
    assert result.telemetry.accepted == 1
    telemetry, observation, total_step_ms = recorder.records[0]
    assert telemetry is result.telemetry
    assert observation is not None
    assert observation.confidence_logits == confidence
    assert len(observation.proposal_sha256) == 64
    assert total_step_ms >= 0.0
    assert events[-3:] == ["draft_commit", "agree_stage_1", "confidence_record"]


def test_confidence_capture_absence_never_changes_conservative_tokens(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    draft = _FakeDraft((21, 22), 3, events)
    target = _FakeTarget((21, 22, 23), events)
    recorder = _FakeConfidenceRecorder()
    engine = KimiK3DSparkRoundEngine(
        config=KimiK3DSparkConfig(
            checkpoint_path=tmp_path,
            verify_width=3,
            round_telemetry=False,
            confidence_capture=DSparkConfidenceCaptureConfig(
                tmp_path / "rounds.jsonl",
                "capture-1",
            ),
        ),
        draft=draft,
        target=target,
        collective=_FakeAgreement(events),
        confidence_recorder=recorder,
    )

    result = engine.decode_round(20)

    assert result.emitted_tokens == (21, 22, 23)
    assert result.telemetry.fallback is False
    assert recorder.records[0][1] is None


def test_confidence_recorder_is_rejected_on_nonzero_rank(tmp_path: Path) -> None:
    events: list[str] = []
    with pytest.raises(DSparkConfigurationError, match="must be rank zero"):
        KimiK3DSparkRoundEngine(
            config=KimiK3DSparkConfig(
                checkpoint_path=tmp_path,
                verify_width=3,
                round_telemetry=False,
                confidence_capture=DSparkConfidenceCaptureConfig(
                    tmp_path / "rounds.jsonl",
                    "capture-1",
                ),
            ),
            draft=_FakeDraft((21, 22), 3, events),
            target=_FakeTarget((21, 22, 23), events),
            collective=_FakeAgreement(events, rank=1),
            confidence_recorder=_FakeConfidenceRecorder(),
        )


def test_confidence_jsonl_has_labels_timings_prompt_identity_and_end_marker(
    tmp_path: Path,
) -> None:
    capture_path = tmp_path / "rounds.jsonl"
    request = DSparkConfidenceCaptureRequest(
        prompt_id=dspark_prompt_identity((1, 2, 3)),
        context_tokens=3,
        context_bucket=dspark_context_bucket(3),
    )
    recorder = DSparkConfidenceJSONLRecorder(
        DSparkConfidenceCaptureConfig(capture_path, "capture-1"),
        request,
        verify_width=3,
        target_route_top_k=8,
    )
    telemetry = DSparkRoundTelemetry(
        round_index=4,
        rank=0,
        draft_ms=1.0,
        target_verify_ms=2.0,
        target_commit_ms=3.0,
        draft_commit_ms=4.0,
        collective_ms=5.0,
        proposed=2,
        accepted=1,
        emitted=2,
        fallback=False,
        error=None,
    )
    observation = DSparkConfidenceObservation(
        confidence_logits=(-1.0, 2.0),
        proposal_sha256="a" * 64,
    )

    recorder.record(telemetry, observation, total_step_ms=16.0)
    recorder.finalize(complete=True)

    lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["schema"] == "k3-dspark-confidence-capture-v3"
    assert payload["prompt_id"] == request.prompt_id
    assert payload["context_tokens"] == 3
    assert payload["context_bucket"] == "le_512"
    assert payload["verify_width"] == 3
    assert payload["target_route_top_k"] == 8
    assert payload["confidence_logits"] == [-1.0, 2.0]
    assert payload["accepted_prefix"] == 1
    assert payload["total_step_ms"] == 16.0
    assert "prompt_tokens" not in payload
    assert "prompt_text" not in payload
    end = json.loads(lines[1])
    assert end["schema"] == "k3-dspark-confidence-request-end-v3"
    assert end["complete"] is True
    assert end["captured_rounds"] == 1
    assert end["target_route_top_k"] == 8
    assert end["request_buffered_locked_append"] is True
    assert end["round_timing_excludes_request_flush"] is True
    assert stat.S_IMODE(capture_path.stat().st_mode) == 0o600


def test_confidence_recorder_retries_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_path = tmp_path / "rounds.jsonl"
    recorder = DSparkConfidenceJSONLRecorder(
        DSparkConfidenceCaptureConfig(capture_path, "capture-1"),
        DSparkConfidenceCaptureRequest("a" * 64, 3, "le_512"),
        verify_width=3,
        target_route_top_k=8,
    )
    telemetry = DSparkRoundTelemetry(
        round_index=0,
        rank=0,
        draft_ms=1.0,
        target_verify_ms=2.0,
        target_commit_ms=3.0,
        draft_commit_ms=4.0,
        collective_ms=5.0,
        proposed=2,
        accepted=1,
        emitted=2,
        fallback=False,
        error=None,
    )
    observation = DSparkConfidenceObservation(
        confidence_logits=(-1.0, 2.0),
        proposal_sha256="b" * 64,
    )
    real_write = dspark_module.os.write
    write_sizes: list[int] = []

    def short_write(descriptor: int, payload: bytes | memoryview) -> int:
        chunk = payload[:17]
        write_sizes.append(len(chunk))
        return real_write(descriptor, chunk)

    monkeypatch.setattr(dspark_module.os, "write", short_write)
    recorder.record(telemetry, observation, total_step_ms=16.0)
    recorder.finalize(complete=True)

    lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["schema"] == "k3-dspark-confidence-capture-v3"
    assert json.loads(lines[1])["schema"] == "k3-dspark-confidence-request-end-v3"
    assert len(write_sizes) > 2


def test_confidence_recorder_serializes_concurrent_short_write_transactions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_path = tmp_path / "rounds.jsonl"
    telemetry = DSparkRoundTelemetry(
        round_index=0,
        rank=0,
        draft_ms=1.0,
        target_verify_ms=2.0,
        target_commit_ms=3.0,
        draft_commit_ms=4.0,
        collective_ms=5.0,
        proposed=2,
        accepted=1,
        emitted=2,
        fallback=False,
        error=None,
    )
    recorders = [
        DSparkConfidenceJSONLRecorder(
            DSparkConfidenceCaptureConfig(capture_path, "capture-1"),
            DSparkConfidenceCaptureRequest(character * 64, 3, "le_512"),
            verify_width=3,
            target_route_top_k=8,
        )
        for character in ("a", "b")
    ]
    for index, recorder in enumerate(recorders):
        recorder.record(
            telemetry,
            DSparkConfidenceObservation(
                confidence_logits=(-1.0, 2.0),
                proposal_sha256=("c" if index == 0 else "d") * 64,
            ),
            total_step_ms=16.0,
        )

    real_write = dspark_module.os.write

    def slow_short_write(descriptor: int, payload: bytes | memoryview) -> int:
        chunk = payload[:11]
        written = real_write(descriptor, chunk)
        time.sleep(0.0001)
        return written

    monkeypatch.setattr(dspark_module.os, "write", slow_short_write)
    threads = [
        threading.Thread(target=recorder.finalize, kwargs={"complete": True})
        for recorder in recorders
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    payloads = [
        json.loads(line)
        for line in capture_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [payload["schema"] for payload in payloads] == [
        "k3-dspark-confidence-capture-v3",
        "k3-dspark-confidence-request-end-v3",
        "k3-dspark-confidence-capture-v3",
        "k3-dspark-confidence-request-end-v3",
    ]
    assert payloads[0]["prompt_id"] == payloads[1]["prompt_id"]
    assert payloads[2]["prompt_id"] == payloads[3]["prompt_id"]
    assert payloads[0]["prompt_id"] != payloads[2]["prompt_id"]


def test_confidence_recorder_rolls_back_partial_failed_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_path = tmp_path / "rounds.jsonl"
    capture_path.write_bytes(b'{"existing":"valid"}\n')
    capture_path.chmod(0o600)
    before = capture_path.read_bytes()
    recorder = DSparkConfidenceJSONLRecorder(
        DSparkConfidenceCaptureConfig(capture_path, "capture-1"),
        DSparkConfidenceCaptureRequest("a" * 64, 3, "le_512"),
        verify_width=3,
        target_route_top_k=8,
    )
    telemetry = DSparkRoundTelemetry(
        round_index=0,
        rank=0,
        draft_ms=1.0,
        target_verify_ms=2.0,
        target_commit_ms=3.0,
        draft_commit_ms=4.0,
        collective_ms=5.0,
        proposed=2,
        accepted=1,
        emitted=2,
        fallback=False,
        error=None,
    )
    recorder.record(
        telemetry,
        DSparkConfidenceObservation(
            confidence_logits=(-1.0, 2.0),
            proposal_sha256="b" * 64,
        ),
        total_step_ms=16.0,
    )
    real_write = dspark_module.os.write
    calls = 0

    def fail_after_partial_write(
        descriptor: int,
        payload: bytes | memoryview,
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            chunk = payload[:19]
            return real_write(descriptor, chunk)
        raise OSError("injected partial append failure")

    monkeypatch.setattr(dspark_module.os, "write", fail_after_partial_write)
    recorder.finalize(complete=True)

    assert calls == 2
    assert capture_path.read_bytes() == before


def test_confidence_recorder_closes_descriptor_when_unlock_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_path = tmp_path / "rounds.jsonl"
    recorders = [
        DSparkConfidenceJSONLRecorder(
            DSparkConfidenceCaptureConfig(capture_path, "capture-1"),
            DSparkConfidenceCaptureRequest(character * 64, 3, "le_512"),
            verify_width=3,
            target_route_top_k=8,
        )
        for character in ("a", "b")
    ]
    real_flock = dspark_module.fcntl.flock
    real_close = dspark_module.os.close
    unlock_failed = False
    closed: list[int] = []

    def fail_first_unlock(descriptor: int, operation: int) -> None:
        nonlocal unlock_failed
        if operation == dspark_module.fcntl.LOCK_UN and not unlock_failed:
            unlock_failed = True
            raise OSError("injected unlock failure")
        real_flock(descriptor, operation)

    def track_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(dspark_module.fcntl, "flock", fail_first_unlock)
    monkeypatch.setattr(dspark_module.os, "close", track_close)
    recorders[0].finalize(complete=False)

    assert unlock_failed is True
    assert len(closed) == 1

    recorders[1].finalize(complete=False)

    assert len(closed) == 2
    assert len(capture_path.read_text(encoding="utf-8").splitlines()) == 2


def test_confidence_recorder_flush_failure_is_nonthrowing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = DSparkConfidenceJSONLRecorder(
        DSparkConfidenceCaptureConfig(tmp_path / "rounds.jsonl", "capture-1"),
        DSparkConfidenceCaptureRequest("a" * 64, 3, "le_512"),
        verify_width=3,
        target_route_top_k=8,
    )
    attempts: list[bytes] = []

    def reject(payload: bytes) -> None:
        attempts.append(payload)
        raise OSError("injected recorder failure")

    monkeypatch.setattr(recorder, "_append", reject)
    recorder.finalize(complete=False)
    recorder.finalize(complete=False)

    assert len(attempts) == 1


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


def test_packed_agreement_uses_one_universal_row_and_attests_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    agreement = MlxRankAgreement(cast(object, group))  # type: ignore[arg-type]
    rows: list[tuple[int, ...]] = []

    def gathered(row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        rows.append(row)
        return (row, row)

    monkeypatch.setattr(agreement, "_all_gather_rows", gathered)
    agreement.activate_packed_agreements()

    assert agreement.agree_packed(
        "stage:a",
        local_success=True,
        error_fingerprint=0,
    ) == PackedRankAgreement(True, 0, ())
    assert agreement.agree_packed(
        "response-control",
        local_success=True,
        error_fingerprint=0,
        payload=(11, 0, 0, 1, 2, 3, 4),
    ) == PackedRankAgreement(True, 0, (11, 0, 0, 1, 2, 3, 4))

    assert [len(row) for row in rows] == [
        PACKED_AGREEMENT_ROW_WIDTH,
        PACKED_AGREEMENT_ROW_WIDTH,
    ]
    attestation = agreement.packed_agreement_attestation
    assert attestation.row_calls == 2
    assert attestation.packed_row_calls == 2
    assert attestation.legacy_row_calls == 0
    assert attestation.physical_all_gathers == 2


@pytest.mark.parametrize("field", [0, 1, 2, 3, 4])
def test_packed_agreement_rejects_peer_tag_status_error_or_payload_mismatch(
    field: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreement = MlxRankAgreement(None)
    agreement.activate_packed_agreements()

    def gathered(row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        peer = list(row)
        peer[field] = peer[field] + 1
        return (row, tuple(peer))

    monkeypatch.setattr(agreement, "_all_gather_rows", gathered)

    assert agreement.agree_packed(
        "ordered-op",
        local_success=True,
        error_fingerprint=0,
        payload=(7,),
    ) == PackedRankAgreement(None, None, None)


def test_rank_zero_valid_proposal_is_authoritative_when_peer_tokens_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gathered(_row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        return ((1, 10, 11, 12), (1, 10, 21, 22))

    agreement = MlxRankAgreement(None, rank_zero_proposal_recovery=True)
    monkeypatch.setattr(agreement, "_all_gather_rows", gathered)

    assert agreement.agree_proposal_block((10, 21, 22), 3) == (10, 11, 12)


def test_rank_zero_proposal_mismatch_is_rejected_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gathered(_row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        return ((1, 10, 11, 12), (1, 10, 21, 22))

    agreement = MlxRankAgreement(None)
    monkeypatch.setattr(agreement, "_all_gather_rows", gathered)

    assert agreement.agree_proposal_block((10, 21, 22), 3) is None


def test_rank_zero_proposal_is_rejected_when_peer_has_no_draft_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gathered(_row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        return ((1, 10, 11, 12), (0, -1, -1, -1))

    agreement = MlxRankAgreement(None, rank_zero_proposal_recovery=True)
    monkeypatch.setattr(agreement, "_all_gather_rows", gathered)

    assert agreement.agree_proposal_block((10, 11, 12), 3) is None


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


def test_decode_anchor_disagreement_prevents_every_lazy_tp_graph(
    tmp_path: Path,
) -> None:
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
class _FakeConfidenceArray:
    rows: list[list[float]]

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
    confidence_rows: list[list[float]] | None = None

    def make_context_cache(self, *, capacity_hint: int = 0) -> object:
        self.calls.append(("make_context_cache", capacity_hint))
        return [_FakeContextCache(), _FakeContextCache()]

    def propose(self, anchor_token: int, context_cache: object) -> object:
        self.calls.append(("propose", anchor_token, context_cache))
        proposal = SimpleNamespace(
            tokens=_FakeTokenArray(self.proposal_rows),
            verify_width=self.verify_width,
        )
        if self.confidence_rows is not None:
            proposal.confidence_logits = _FakeConfidenceArray(  # type: ignore[attr-defined]
                self.confidence_rows
            )
        return proposal

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


@dataclass
class _RankZeroBroadcastAgreement:
    rank: int
    authoritative_blocks: tuple[tuple[int, ...], ...]
    size: int = 2
    local_blocks: list[tuple[int, ...] | None] = field(default_factory=list)
    acceptances: list[tuple[int | None, int | None]] = field(default_factory=list)

    def agree_proposal_block(
        self,
        local_block: tuple[int, ...] | None,
        expected_width: int,
    ) -> tuple[int, ...] | None:
        authoritative = self.authoritative_blocks[len(self.local_blocks)]
        assert local_block is not None
        assert len(local_block) == expected_width
        assert len(authoritative) == expected_width
        self.local_blocks.append(local_block)
        return authoritative

    def agree_acceptance(
        self,
        local_boundary: int | None,
        local_next_token: int | None,
        maximum_boundary: int,
    ) -> tuple[int, int] | None:
        assert local_boundary is not None
        assert local_next_token is not None
        assert 0 <= local_boundary <= maximum_boundary
        self.acceptances.append((local_boundary, local_next_token))
        return local_boundary, local_next_token

    def agree_stage_success(self, local_success: bool) -> bool | None:
        return local_success

    def agree_token(self, local_token: int | None) -> int | None:
        return local_token


@dataclass
class _OffsetTrackingTarget:
    offset: int
    proposal_blocks: list[tuple[int, ...]] = field(default_factory=list)
    commit_widths: list[int] = field(default_factory=list)

    def prepare_verification(
        self,
        proposal_block: tuple[int, ...],
    ) -> _OffsetTrackingPreparedTarget:
        self.proposal_blocks.append(proposal_block)
        return _OffsetTrackingPreparedTarget(
            owner=self,
            proposal_block=proposal_block,
            initial_offset=self.offset,
        )

    def preflight_ordinary(self, anchor_token: int) -> OrdinaryDecodePlan:
        raise AssertionError(f"unexpected ordinary fallback for anchor {anchor_token}")

    def prepare_ordinary(
        self,
        anchor_token: int,
        plan: OrdinaryDecodePlan,
    ) -> _FakePreparedOrdinary:
        del plan
        raise AssertionError(f"unexpected ordinary fallback for anchor {anchor_token}")


@dataclass
class _OffsetTrackingPreparedTarget:
    owner: _OffsetTrackingTarget
    proposal_block: tuple[int, ...]
    initial_offset: int
    mode_code: int = 0

    def build(self) -> _OffsetTrackingBuiltTarget:
        return _OffsetTrackingBuiltTarget(
            self.owner,
            self.proposal_block,
            self.initial_offset,
        )

    def cancel(self) -> None:
        return None


@dataclass
class _OffsetTrackingBuiltTarget:
    owner: _OffsetTrackingTarget
    proposal_block: tuple[int, ...]
    initial_offset: int

    def materialize(self) -> _OffsetTrackingTargetRound:
        if self.proposal_block == (10, 11, 12):
            posterior_tokens = (11, 99, 100)
        elif self.proposal_block == (99, 31, 32):
            posterior_tokens = (31, 32, 33)
        else:
            raise AssertionError(f"unexpected target block {self.proposal_block}")
        hidden_states = tuple(
            _FakeHidden((1, 3, 7168), f"tap-{index}") for index in range(5)
        )
        return _OffsetTrackingTargetRound(
            self.owner,
            TargetPosterior(posterior_tokens, hidden_states),
            self.initial_offset,
        )

    def cancel(self) -> None:
        return None


@dataclass
class _OffsetTrackingTargetRound:
    owner: _OffsetTrackingTarget
    posterior: TargetPosterior
    initial_offset: int

    def commit(self, consumed_input_tokens: int) -> None:
        assert self.owner.offset == self.initial_offset
        self.owner.offset += consumed_input_tokens
        self.owner.commit_widths.append(consumed_input_tokens)

    def cancel(self) -> None:
        return None


def _mock_mlx_features(
    calls: list[tuple[object, ...]],
    proposal_rows: list[list[int]],
    confidence_rows: list[list[float]] | None = None,
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
        return _FakeMlxProposer(
            verify_width,
            calls,
            proposal_rows,
            confidence_rows,
        )

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


def _fake_sparse_target(*route_top_ks: int) -> object:
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                mlp=SimpleNamespace(
                    switch_mlp=object(),
                    expert_top_k=route_top_k,
                )
            )
            for route_top_k in route_top_ks
        ]
    )


def test_confidence_capture_rejects_native_top16_target_before_loading(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, ...]] = []
    config = KimiK3DSparkConfig(
        checkpoint_path=tmp_path,
        verify_width=3,
        round_telemetry=False,
        confidence_capture=DSparkConfidenceCaptureConfig(
            tmp_path / "rounds.jsonl",
            "capture-1",
        ),
    )

    with pytest.raises(
        DSparkConfigurationError,
        match="target route top-k is 16, expected 8",
    ):
        load_replicated_mlx_dspark(
            config,
            _fake_sparse_target(16, 16),
            features=_mock_mlx_features(calls, [[11, 12]]),
            evaluate=lambda *_values: None,
        )

    assert calls == []


def test_target_route_top_k_attestation_rejects_mixed_sparse_layers() -> None:
    assert (
        attest_kimi_k3_target_route_top_k(
            _fake_sparse_target(8, 8),
            expected=8,
        )
        == 8
    )
    with pytest.raises(DSparkConfigurationError, match="layers disagree"):
        attest_kimi_k3_target_route_top_k(_fake_sparse_target(8, 16))


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
    with pytest.raises(DSparkConfigurationError, match="requires capture config"):
        loaded.new_request(capacity_hint=300, capture_confidence=True)


def test_dual_loader_attests_each_checkpoint_once_and_selects_exact_boundary(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, ...]] = []
    target_model = object()
    config = _dual_config(tmp_path)

    loaded = load_replicated_mlx_dspark_dual(
        config,
        target_model,
        features=_mock_mlx_features(calls, [[11, 12, 13, 14, 15, 16, 17]]),
        evaluate=lambda *_values: None,
    )

    assert calls[:4] == [
        ("load", config.old.checkpoint_path, target_model, True),
        ("proposer", "draft", 8, False),
        ("load", config.yarn.checkpoint_path, target_model, True),
        ("proposer", "draft", 8, False),
    ]
    old, old_selection = loaded.select(DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS - 1)
    yarn, yarn_selection = loaded.select(DSPARK_DUAL_PROPOSER_THRESHOLD_TOKENS)
    minimum, minimum_selection = loaded.select(2)
    maximum, maximum_selection = loaded.select(1_048_576)
    assert old is loaded.old
    assert old_selection.role == "old"
    assert old_selection.revision == config.old.revision
    assert yarn is loaded.yarn
    assert yarn_selection.role == "yarn"
    assert yarn_selection.revision == config.yarn.revision
    assert minimum is loaded.old
    assert minimum_selection.role == "old"
    assert maximum is loaded.yarn
    assert maximum_selection.role == "yarn"

    with pytest.raises(FrozenInstanceError):
        old_selection.role = "yarn"


def test_dual_selection_allocates_only_the_selected_request_cache(
    tmp_path: Path,
) -> None:
    target_model = object()
    config = _dual_config(tmp_path)
    old_calls: list[tuple[object, ...]] = []
    yarn_calls: list[tuple[object, ...]] = []
    old_proposer = _FakeMlxProposer(8, old_calls, [[11, 12, 13, 14, 15, 16, 17]])
    yarn_proposer = _FakeMlxProposer(
        8,
        yarn_calls,
        [[21, 22, 23, 24, 25, 26, 27]],
    )
    loaded = LoadedMlxDSparkDual(
        config=config,
        old=LoadedMlxDSpark(
            config=config.old,
            target_model=target_model,
            drafter=object(),
            proposer=old_proposer,
            evaluate=lambda *_values: None,
        ),
        yarn=LoadedMlxDSpark(
            config=config.yarn,
            target_model=target_model,
            drafter=object(),
            proposer=yarn_proposer,
            evaluate=lambda *_values: None,
        ),
    )

    selected_old, _ = loaded.select(8_191)
    old_request = selected_old.new_request(capacity_hint=8_200)
    assert old_calls == [("make_context_cache", 8_200)]
    assert yarn_calls == []

    selected_yarn, _ = loaded.select(8_192)
    yarn_request = selected_yarn.new_request(capacity_hint=8_300)
    assert old_calls == [("make_context_cache", 8_200)]
    assert yarn_calls == [("make_context_cache", 8_300)]
    assert old_request.context_cache is not yarn_request.context_cache


def test_dual_loader_cleans_partial_first_load_when_yarn_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _dual_config(tmp_path)
    target_model = object()
    calls: list[Path] = []
    cleanups: list[str] = []
    monkeypatch.setattr(
        dspark_module,
        "_release_mlx_memory",
        lambda: cleanups.append("released"),
    )

    def load(
        checkpoint_config: KimiK3DSparkConfig,
        model: object,
    ) -> LoadedMlxDSpark:
        calls.append(checkpoint_config.checkpoint_path)
        if checkpoint_config is config.yarn:
            raise RuntimeError("injected YaRN load failure")
        return LoadedMlxDSpark(
            config=checkpoint_config,
            target_model=model,
            drafter=object(),
            proposer=_FakeMlxProposer(8, [], [[11, 12, 13, 14, 15, 16, 17]]),
            evaluate=lambda *_values: None,
        )

    with pytest.raises(RuntimeError, match="injected YaRN load failure"):
        load_replicated_mlx_dspark_dual(
            config,
            target_model,
            loader=load,
        )

    assert calls == [config.old.checkpoint_path, config.yarn.checkpoint_path]
    assert cleanups == ["released"]


def test_dual_loaded_cleanup_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _dual_config(tmp_path)
    target_model = object()
    cleanups: list[str] = []
    monkeypatch.setattr(
        dspark_module,
        "_release_mlx_memory",
        lambda: cleanups.append("released"),
    )
    loaded = LoadedMlxDSparkDual(
        config=config,
        old=LoadedMlxDSpark(
            config.old,
            target_model,
            object(),
            _FakeMlxProposer(8, [], [[11, 12, 13, 14, 15, 16, 17]]),
        ),
        yarn=LoadedMlxDSpark(
            config.yarn,
            target_model,
            object(),
            _FakeMlxProposer(8, [], [[21, 22, 23, 24, 25, 26, 27]]),
        ),
    )

    loaded.close()
    loaded.close()

    assert loaded.old is None
    assert loaded.yarn is None
    assert cleanups == ["released"]


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


def test_round_engine_rejects_recovered_block_when_config_gate_is_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(dspark_module, "_log_nonfatal_warning", warnings.append)
    events: list[str] = []
    draft = _FakeDraft((21, 22), 3, events)
    target = _FakeTarget((11, 12, 13), events)
    engine = KimiK3DSparkRoundEngine(
        config=_config(tmp_path, 3),
        draft=draft,
        target=target,
        collective=_RankZeroBroadcastAgreement(1, ((10, 11, 12),)),
    )

    result = engine.decode_round(10)

    assert result.emitted_tokens == (999,)
    assert result.telemetry.fallback is True
    assert result.telemetry.error == "DSpark proposal recovery is disabled"
    assert draft.rounds[0].cancelled is True
    assert target.rounds == []
    assert warnings == []


def test_rank_zero_block_resynchronizes_draft_context_for_the_next_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MLX_DSPARK_PROPOSER_ENV, "1")
    recovered_mismatches: list[str] = []
    monkeypatch.setattr(
        dspark_module,
        "_log_nonfatal_warning",
        recovered_mismatches.append,
    )
    authoritative_blocks = ((10, 11, 12), (99, 31, 32))
    proposer_calls: tuple[list[tuple[object, ...]], ...] = ([], [])
    proposers = (
        _FakeMlxProposer(3, proposer_calls[0], [[11, 12]]),
        _FakeMlxProposer(3, proposer_calls[1], [[21, 22]]),
    )
    context_caches = tuple(
        [
            _FakeContextCache(length=5, keys=object(), values=object()),
            _FakeContextCache(length=5, keys=object(), values=object()),
        ]
        for _rank in range(2)
    )
    agreements = tuple(
        _RankZeroBroadcastAgreement(rank, authoritative_blocks) for rank in range(2)
    )
    targets = tuple(_OffsetTrackingTarget(offset=5) for _rank in range(2))
    engines = tuple(
        KimiK3DSparkRoundEngine(
            config=_config(tmp_path, 3, rank_zero_proposal_recovery=True),
            draft=MlxDSparkRequestDraft(
                proposer=proposers[rank],
                context_cache=context_caches[rank],
                verify_width=3,
                evaluate=lambda *_values: None,
            ),
            target=targets[rank],
            collective=agreements[rank],
        )
        for rank in range(2)
    )

    first_results = tuple(engine.decode_round(10) for engine in engines)

    assert agreements[0].local_blocks == [(10, 11, 12)]
    assert agreements[1].local_blocks == [(10, 21, 22)]
    assert [target.proposal_blocks for target in targets] == [
        [(10, 11, 12)],
        [(10, 11, 12)],
    ]
    assert [agreement.acceptances for agreement in agreements] == [
        [(1, 99)],
        [(1, 99)],
    ]
    assert [result.emitted_tokens for result in first_results] == [
        (11, 99),
        (11, 99),
    ]
    assert [target.offset for target in targets] == [7, 7]
    assert [target.commit_widths for target in targets] == [[2], [2]]
    assert [
        [entry.length for entry in context_cache] for context_cache in context_caches
    ] == [[7, 7], [7, 7]]
    assert len(recovered_mismatches) == 1
    recovery_log = recovered_mismatches[0]
    assert "rank=1" in recovery_log
    assert "context_offset=5" in recovery_log
    assert "local_proposal_sha256=" in recovery_log
    assert "authoritative_proposal_sha256=" in recovery_log
    local_hash = recovery_log.split("local_proposal_sha256=", 1)[1].split(", ", 1)[0]
    authoritative_hash = recovery_log.split("authoritative_proposal_sha256=", 1)[1]
    assert len(local_hash) == 64
    assert len(authoritative_hash) == 64
    int(local_hash, 16)
    int(authoritative_hash, 16)
    assert "(10, 21, 22)" not in recovery_log
    assert "[21, 22]" not in recovery_log

    for proposer in proposers:
        proposer.proposal_rows = [[31, 32]]
    second_results = tuple(engine.decode_round(99) for engine in engines)

    assert [agreement.local_blocks[1] for agreement in agreements] == [
        (99, 31, 32),
        (99, 31, 32),
    ]
    assert [target.proposal_blocks[1] for target in targets] == [
        (99, 31, 32),
        (99, 31, 32),
    ]
    assert [agreement.acceptances[1] for agreement in agreements] == [
        (2, 33),
        (2, 33),
    ]
    assert [result.emitted_tokens for result in second_results] == [
        (31, 32, 33),
        (31, 32, 33),
    ]
    assert [target.offset for target in targets] == [10, 10]
    assert [target.commit_widths for target in targets] == [[2, 3], [2, 3]]
    assert [
        [entry.length for entry in context_cache] for context_cache in context_caches
    ] == [[10, 10], [10, 10]]
    assert len(recovered_mismatches) == 1


def test_conservative_capture_materializes_two_rank_zero_confidences_only(
    tmp_path: Path,
) -> None:
    evaluated: list[tuple[object, ...]] = []
    config = KimiK3DSparkConfig(
        checkpoint_path=tmp_path,
        verify_width=3,
        round_telemetry=False,
        confidence_capture=DSparkConfidenceCaptureConfig(
            tmp_path / "rounds.jsonl",
            "capture-1",
        ),
    )
    loaded = load_replicated_mlx_dspark(
        config,
        _fake_sparse_target(8, 8),
        features=_mock_mlx_features(
            [],
            [[11, 12]],
            [[-1.0, 2.0]],
        ),
        evaluate=lambda *values: evaluated.append(values),
    )

    rank_zero_round = (
        loaded.new_request(capacity_hint=100, capture_confidence=True)
        .prepare_round(10, 2)
        .materialize()
    )
    peer_round = (
        loaded.new_request(capacity_hint=100, capture_confidence=False)
        .prepare_round(10, 2)
        .materialize()
    )

    assert tuple(rank_zero_round.proposal_tokens) == (11, 12)
    assert tuple(rank_zero_round.confidence_logits or ()) == (-1.0, 2.0)
    assert tuple(peer_round.proposal_tokens) == (11, 12)
    assert peer_round.confidence_logits is None
    assert len(evaluated) == 1


def test_missing_confidence_graph_preserves_exact_conservative_proposals(
    tmp_path: Path,
) -> None:
    loaded = load_replicated_mlx_dspark(
        KimiK3DSparkConfig(
            checkpoint_path=tmp_path,
            verify_width=3,
            round_telemetry=False,
            confidence_capture=DSparkConfidenceCaptureConfig(
                tmp_path / "rounds.jsonl",
                "capture-1",
            ),
        ),
        _fake_sparse_target(8, 8),
        features=_mock_mlx_features([], [[11, 12]]),
        evaluate=lambda *_values: None,
    )

    round_state = (
        loaded.new_request(capacity_hint=100, capture_confidence=True)
        .prepare_round(10, 2)
        .materialize()
    )

    assert tuple(round_state.proposal_tokens) == (11, 12)
    assert round_state.confidence_logits is None


def test_confidence_evaluation_failure_preserves_exact_conservative_proposals(
    tmp_path: Path,
) -> None:
    def reject_confidence(*_values: object) -> None:
        raise RuntimeError("injected confidence evaluation failure")

    loaded = load_replicated_mlx_dspark(
        KimiK3DSparkConfig(
            checkpoint_path=tmp_path,
            verify_width=3,
            round_telemetry=False,
            confidence_capture=DSparkConfidenceCaptureConfig(
                tmp_path / "rounds.jsonl",
                "capture-1",
            ),
        ),
        _fake_sparse_target(8, 8),
        features=_mock_mlx_features([], [[11, 12]], [[-1.0, 2.0]]),
        evaluate=reject_confidence,
    )

    round_state = (
        loaded.new_request(capacity_hint=100, capture_confidence=True)
        .prepare_round(10, 2)
        .materialize()
    )

    assert tuple(round_state.proposal_tokens) == (11, 12)
    assert round_state.confidence_logits is None


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
        (DSPARK_DUAL_PROPOSER_ENV, "true"),
        (DSPARK_RANK_ZERO_PROPOSAL_RECOVERY_ENV, "true"),
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


def test_decode_token_stream_can_force_aligned_target_only_control() -> None:
    engine = _FakeRoundEngine(
        [(11, 12, 13)],
        verify_width=3,
        ordinary_tokens=[91, 92, 93],
    )

    decoded = list(
        dspark_decode_tokens(
            engine,
            anchor_token=10,
            max_tokens=3,
            eos_token_ids=(),
            force_ordinary=True,
        )
    )

    assert [item.token for item in decoded] == [91, 92, 93]
    assert [item.from_draft for item in decoded] == [False, False, False]
    assert engine.anchors == []
    assert engine.ordinary_anchors == [10, 91, 92]


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
    layers: tuple[object, ...] = field(default_factory=lambda: (object(), object()))


@dataclass
class _FakeAuxPrefillRuntimeTarget(_FakeRuntimeTarget):
    aux_prefill_calls: list[int] = field(default_factory=list)
    final_hidden_size: int = 7168
    tap_count: int = 5

    def forward_aux_hidden_states_for_cache(
        self,
        inputs: mx.array,
        cache: object,
        layer_ids: tuple[int, ...],
    ) -> object:
        assert layer_ids == (7, 23, 51, 67, 83)
        width = int(inputs.shape[1])
        self.aux_prefill_calls.append(width)
        entries = cast(list[object], cache)
        mla_cache = cast(_FakeTargetCache, entries[0])
        kda_cache = cast(_FakeKDATargetCache, entries[1])
        mla_cache.offset += width
        mla_cache.state = ("mla-keys", ["mla-values"])
        kda_cache.cache = [("kda-conv",), {"ssm": "kda-ssm"}]
        return SimpleNamespace(
            final_hidden_state=_FakeHidden(
                (1, width, self.final_hidden_size),
                "final-hidden",
            ),
            aux_hidden_states=tuple(
                _FakeHidden((1, width, 7168), f"tap-{index}")
                for index in range(self.tap_count)
            ),
        )


def _prompt_runtime(
    tmp_path: Path,
    agreement: _FakeAgreement,
    *,
    packed_agreements: bool = False,
) -> tuple[KimiK3DSparkRequestRuntime, list[object]]:
    target_model = _FakeRuntimeTarget()
    target_cache: list[object] = [_FakeTargetCache(), _FakeKDATargetCache()]
    proposer = _FakeMlxProposer(3, [], [[11, 12]])
    draft = MlxDSparkRequestDraft(
        proposer=proposer,
        context_cache=[_FakeContextCache(), _FakeContextCache()],
        verify_width=3,
        evaluate=lambda *_values: None,
    )
    return (
        KimiK3DSparkRequestRuntime(
            loaded=LoadedMlxDSpark(
                config=_config(
                    tmp_path,
                    3,
                    packed_agreements=packed_agreements,
                ),
                target_model=target_model,
                drafter=object(),
                proposer=proposer,
                evaluate=lambda *_values: None,
            ),
            target_model=target_model,
            target_cache=target_cache,
            draft=draft,
            collective=agreement,
            evaluate=lambda *_values: None,
        ),
        target_cache,
    )


def _discard_evaluation(*_values: object) -> None:
    return None


def _aux_prompt_runtime(
    tmp_path: Path,
    agreement: _FakeAgreement,
    *,
    target_model: _FakeRuntimeTarget | None = None,
    evaluate: Callable[..., None] | None = None,
) -> tuple[KimiK3DSparkRequestRuntime, list[object], _FakeRuntimeTarget]:
    target = target_model or _FakeAuxPrefillRuntimeTarget()
    target_evaluate: Callable[..., None] = (
        _discard_evaluation if evaluate is None else evaluate
    )
    target_cache: list[object] = [_FakeTargetCache(), _FakeKDATargetCache()]
    proposer = _FakeMlxProposer(3, [], [[11, 12]])
    draft = MlxDSparkRequestDraft(
        proposer=proposer,
        context_cache=[_FakeContextCache(), _FakeContextCache()],
        verify_width=3,
        evaluate=lambda *_values: None,
    )
    runtime = KimiK3DSparkRequestRuntime(
        loaded=LoadedMlxDSpark(
            config=_config(tmp_path, 3, aux_only_prefill=True),
            target_model=target,
            drafter=object(),
            proposer=proposer,
            evaluate=target_evaluate,
        ),
        target_model=target,
        target_cache=target_cache,
        draft=draft,
        collective=agreement,
        evaluate=target_evaluate,
    )
    return runtime, target_cache, target


def test_aux_only_prefill_missing_api_fails_before_graph_or_progress(
    tmp_path: Path,
) -> None:
    runtime, target_cache, _target = _aux_prompt_runtime(
        tmp_path,
        _FakeAgreement([]),
        target_model=_FakeRuntimeTarget(),
    )
    progress: list[tuple[int, int]] = []

    with pytest.raises(
        DSparkDistributedStateError,
        match="target auxiliary prompt prefill API preflight failed on every rank",
    ):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=(),
            progress_callback=lambda done, total: progress.append((done, total)),
            distributed_progress_callback=None,
        )

    assert cast(_FakeTargetCache, target_cache[0]).offset == 0
    assert cast(_FakeKDATargetCache, target_cache[1]).cache == [None, None]
    assert progress == []


def test_aux_only_prefill_skips_logits_and_materializes_every_root_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    materializations: list[tuple[object, ...]] = []

    def evaluate(*values: object) -> None:
        trace.append("target_materialization")
        materializations.append(values)

    target = _FakeAuxPrefillRuntimeTarget()
    runtime, target_cache, _target = _aux_prompt_runtime(
        tmp_path,
        _FakeAgreement([]),
        target_model=target,
        evaluate=evaluate,
    )
    proposer = cast(_FakeMlxProposer, runtime.draft.proposer)
    append_target_context = proposer.append_target_context
    full_logit_graph_builds: list[int] = []

    def unexpected_full_logit_graph(*_args: object, **_kwargs: object) -> object:
        full_logit_graph_builds.append(1)
        raise AssertionError("opt-in prompt path must not build target logits")

    def track_draft_projection(
        aux_hidden_states: Sequence[object],
        context_offset: int,
        context_cache: object,
    ) -> None:
        trace.append("draft_projection")
        append_target_context(aux_hidden_states, context_offset, context_cache)

    monkeypatch.setattr(proposer, "append_target_context", track_draft_projection)
    monkeypatch.setattr(
        runtime,
        "_build_forward_with_taps",
        unexpected_full_logit_graph,
    )
    monkeypatch.setattr(dspark_module.mx, "clear_cache", lambda: None, raising=False)

    prompt_tps, prompt_tokens = runtime.seed_prompt(
        cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
        prefill_step_size=2,
        max_tokens=16,
        stop_sequences=(),
        progress_callback=lambda done, _total: trace.append(f"progress:{done}"),
        distributed_progress_callback=None,
    )

    assert prompt_tps > 0
    assert prompt_tokens == 2
    assert target.aux_prefill_calls == [2]
    assert full_logit_graph_builds == []
    assert len(materializations) == 1
    final_hidden_state, *other_roots = materializations[0]
    assert cast(_FakeHidden, final_hidden_state).name == "final-hidden"
    assert [cast(_FakeHidden, value).name for value in other_roots[:5]] == [
        "tap-0",
        "tap-1",
        "tap-2",
        "tap-3",
        "tap-4",
    ]
    assert tuple(other_roots[5:]) == (
        "mla-keys",
        "mla-values",
        "kda-conv",
        "kda-ssm",
    )
    assert trace == [
        "progress:0",
        "target_materialization",
        "draft_projection",
        "progress:2",
    ]
    assert cast(_FakeTargetCache, target_cache[0]).offset == 2
    assert [
        entry.length
        for entry in cast(Sequence[_FakeContextCache], runtime.draft.context_cache)
    ] == [2, 2]


@pytest.mark.parametrize(
    ("final_hidden_size", "tap_count"),
    [
        pytest.param(1024, 5, id="final-hidden-shape"),
        pytest.param(7168, 4, id="tap-count"),
    ],
)
def test_aux_only_prefill_rejects_malformed_result_before_evaluation(
    tmp_path: Path,
    final_hidden_size: int,
    tap_count: int,
) -> None:
    evaluations: list[tuple[object, ...]] = []
    target = _FakeAuxPrefillRuntimeTarget(
        final_hidden_size=final_hidden_size,
        tap_count=tap_count,
    )
    runtime, _target_cache, _target = _aux_prompt_runtime(
        tmp_path,
        _FakeAgreement([]),
        target_model=target,
        evaluate=lambda *values: evaluations.append(values),
    )

    with pytest.raises(
        DSparkDistributedStateError,
        match="target auxiliary prompt prefill graph build failed on every rank",
    ):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=(),
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=None,
        )

    assert target.aux_prefill_calls == [2]
    assert evaluations == []


def test_aux_only_prefill_rank_asymmetry_fails_before_target_graph(
    tmp_path: Path,
) -> None:
    target = _FakeAuxPrefillRuntimeTarget()
    agreement = _FakeAgreement([], stage_outcomes=[True, None])
    evaluations: list[tuple[object, ...]] = []
    runtime, target_cache, _target = _aux_prompt_runtime(
        tmp_path,
        agreement,
        target_model=target,
        evaluate=lambda *values: evaluations.append(values),
    )
    progress: list[tuple[int, int]] = []

    with pytest.raises(
        DSparkDistributedStateError,
        match=(
            "target auxiliary prompt prefill API preflight outcomes disagreed "
            "across ranks"
        ),
    ):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=(),
            progress_callback=lambda done, total: progress.append((done, total)),
            distributed_progress_callback=None,
        )

    assert target.aux_prefill_calls == []
    assert evaluations == []
    assert cast(_FakeTargetCache, target_cache[0]).offset == 0
    assert cast(_FakeKDATargetCache, target_cache[1]).cache == [None, None]
    assert progress == []


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

    def unexpected_aux_prefill() -> None:
        raise AssertionError("default-off prompt path must not preflight auxiliary API")

    monkeypatch.setattr(runtime, "_target_aux_prefill_forward", unexpected_aux_prefill)
    monkeypatch.setattr(dspark_module.mx, "clear_cache", lambda: None, raising=False)
    progress: list[tuple[int, int]] = []
    distributed_progress: list[None] = []
    full_prompt = _FakePromptArray((1, 2, 3, 4, 5, 6))

    prompt_tps, prompt_tokens = runtime.seed_prompt(
        cast(mx.array, cast(object, full_prompt[:-1])),
        prefill_step_size=2,
        max_tokens=16,
        stop_sequences=("STOP",),
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
        # Validation, length, and controls agree; a prompt digest word does not.
        collective=_FakeAgreement([], reject_token_call=5),
        evaluate=lambda *_values: None,
    )

    with pytest.raises(DSparkDistributedStateError, match="prompt/request"):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=(),
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=None,
        )

    assert target_cache[0].offset == 0
    assert target_cache[1].cache == [None, None]


def test_request_control_fingerprint_binds_graph_and_termination_controls() -> None:
    fingerprint = dspark_module._request_control_fingerprint(
        max_tokens=16,
        prefill_step_size=2,
        stop_sequences=("STOP", "END"),
        distributed_progress=False,
    )

    assert fingerprint != dspark_module._request_control_fingerprint(
        max_tokens=17,
        prefill_step_size=2,
        stop_sequences=("STOP", "END"),
        distributed_progress=False,
    )
    assert fingerprint != dspark_module._request_control_fingerprint(
        max_tokens=16,
        prefill_step_size=3,
        stop_sequences=("STOP", "END"),
        distributed_progress=False,
    )
    assert fingerprint != dspark_module._request_control_fingerprint(
        max_tokens=16,
        prefill_step_size=2,
        stop_sequences=("END", "STOP"),
        distributed_progress=False,
    )
    assert fingerprint != dspark_module._request_control_fingerprint(
        max_tokens=16,
        prefill_step_size=2,
        stop_sequences=("STOP", "END"),
        distributed_progress=True,
    )


def test_aux_only_prefill_flag_is_bound_into_rank_prompt_contract(
    tmp_path: Path,
) -> None:
    control_runtime, _control_cache = _prompt_runtime(tmp_path, _FakeAgreement([]))
    candidate_runtime, _candidate_cache, _target = _aux_prompt_runtime(
        tmp_path,
        _FakeAgreement([]),
    )
    prompt = cast(mx.array, cast(object, _FakePromptArray((1, 2))))

    control_tokens, control_fingerprint = control_runtime._prompt_contract(
        prompt,
        max_tokens=16,
        prefill_step_size=2,
        stop_sequences=(),
        distributed_progress=False,
    )
    candidate_tokens, candidate_fingerprint = candidate_runtime._prompt_contract(
        prompt,
        max_tokens=16,
        prefill_step_size=2,
        stop_sequences=(),
        distributed_progress=False,
    )

    assert control_tokens == candidate_tokens == (1, 2)
    assert control_fingerprint != candidate_fingerprint


@pytest.mark.parametrize(
    ("reject_token_call", "control"),
    ((3, "prefill step"), (4, "max tokens"), (9, "stop sequence digest")),
)
def test_asymmetric_request_controls_prevent_prompt_graph_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reject_token_call: int,
    control: str,
) -> None:
    agreement = _FakeAgreement([], reject_token_call=reject_token_call)
    runtime, target_cache = _prompt_runtime(tmp_path, agreement)
    graph_builds: list[str] = []

    def unexpected_graph(*_args: object, **_kwargs: object) -> object:
        graph_builds.append(control)
        raise AssertionError("request disagreement must precede target graph build")

    monkeypatch.setattr(runtime, "_build_forward_with_taps", unexpected_graph)

    with pytest.raises(DSparkDistributedStateError, match="prompt/request"):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=("STOP",),
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=None,
        )

    assert graph_builds == []
    assert cast(_FakeTargetCache, target_cache[0]).offset == 0


def test_asymmetric_distributed_progress_presence_prevents_prompt_graph_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreement = _FakeAgreement([], reject_token_call=9)
    runtime, target_cache = _prompt_runtime(tmp_path, agreement)
    graph_builds: list[None] = []
    distributed_progress: list[None] = []

    def unexpected_graph(*_args: object, **_kwargs: object) -> object:
        graph_builds.append(None)
        raise AssertionError("callback disagreement must precede target graph build")

    monkeypatch.setattr(runtime, "_build_forward_with_taps", unexpected_graph)

    with pytest.raises(DSparkDistributedStateError, match="prompt/request"):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=(),
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=lambda: distributed_progress.append(None),
        )

    assert graph_builds == []
    assert distributed_progress == []
    assert cast(_FakeTargetCache, target_cache[0]).offset == 0


def test_asymmetric_prompt_chunk_contract_prevents_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Calls 1-12 bind validation plus the complete request; 13-15 bind the
    # first chunk's start, cumulative boundary, and width, respectively.
    agreement = _FakeAgreement([], reject_token_call=15)
    runtime, target_cache = _prompt_runtime(tmp_path, agreement)
    graph_builds: list[None] = []

    def unexpected_graph(*_args: object, **_kwargs: object) -> object:
        graph_builds.append(None)
        raise AssertionError("chunk disagreement must precede target graph build")

    monkeypatch.setattr(runtime, "_build_forward_with_taps", unexpected_graph)

    with pytest.raises(DSparkDistributedStateError, match="prompt chunk contract"):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=("STOP",),
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=None,
        )

    assert graph_builds == []
    assert cast(_FakeTargetCache, target_cache[0]).offset == 0


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
        collective=_FakeAgreement([], stage_outcomes=[True, True, True, True, None]),
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
            max_tokens=16,
            stop_sequences=(),
            progress_callback=lambda _done, _total: None,
            distributed_progress_callback=None,
        )

    assert materializations == []


def test_asymmetric_prompt_progress_failure_is_agreed_after_committed_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreement = _FakeAgreement(
        [],
        # prompt contract, initial publication, chunk construction, readiness,
        # graph build, materialization, draft projection, then asymmetric
        # publication.
        stage_outcomes=[True, True, True, True, True, True, True, None],
    )
    runtime, target_cache = _prompt_runtime(tmp_path, agreement)
    graph_builds: list[None] = []

    def build(
        chunk: _FakePromptBatch,
        *,
        initial_offset: int,
        speculative_width: int | None = None,
    ) -> _FakePendingForward:
        assert initial_offset == cast(_FakeTargetCache, target_cache[0]).offset
        assert speculative_width is None
        graph_builds.append(None)
        cast(_FakeTargetCache, target_cache[0]).offset += len(chunk.values)
        cast(_FakeKDATargetCache, target_cache[1]).cache = [object(), object()]
        return _FakePendingForward(
            SimpleNamespace(
                aux_hidden_states=tuple(
                    _FakeHidden((1, len(chunk.values), 7168), f"tap-{index}")
                    for index in range(5)
                )
            )
        )

    monkeypatch.setattr(runtime, "_build_forward_with_taps", build)
    monkeypatch.setattr(dspark_module.mx, "clear_cache", lambda: None, raising=False)

    def publish(done: int, _total: int) -> None:
        if done > 0:
            raise RuntimeError("rank-zero event sender failed")

    with pytest.raises(DSparkDistributedStateError, match="progress publication"):
        runtime.seed_prompt(
            cast(mx.array, cast(object, _FakePromptArray((1, 2)))),
            prefill_step_size=2,
            max_tokens=16,
            stop_sequences=(),
            progress_callback=publish,
            distributed_progress_callback=None,
        )

    assert graph_builds == [None]
    assert cast(_FakeTargetCache, target_cache[0]).offset == 2
    assert agreement.stage_outcomes == []


def test_unanimous_progress_cancellation_preserves_original_exception(
    tmp_path: Path,
) -> None:
    class ExpectedCancellationError(Exception):
        pass

    runtime, _target_cache = _prompt_runtime(tmp_path, _FakeAgreement([]))

    def cancel() -> None:
        raise ExpectedCancellationError("cancelled on every rank")

    with pytest.raises(ExpectedCancellationError, match="cancelled on every rank"):
        runtime.agree_local_side_effect("distributed progress", cancel)


def test_packed_callback_unanimous_failure_preserves_original_exception(
    tmp_path: Path,
) -> None:
    class ExpectedCancellationError(Exception):
        pass

    agreement = _FakeAgreement([])
    runtime, _target_cache = _prompt_runtime(
        tmp_path,
        agreement,
        packed_agreements=True,
    )

    def cancel() -> None:
        raise ExpectedCancellationError("cancelled on every rank")

    with pytest.raises(ExpectedCancellationError, match="cancelled on every rank"):
        runtime.agree_local_side_effect("distributed progress", cancel)

    assert len(agreement.packed_calls) == 1
    assert agreement.packed_calls[0][0] == "side-effect:distributed progress"


def test_response_agreement_sequence_collapses_from_twenty_one_to_five_rows(
    tmp_path: Path,
) -> None:
    legacy_agreement = _FakeAgreement([])
    legacy, _target_cache = _prompt_runtime(tmp_path, legacy_agreement)
    packed_agreement = _FakeAgreement([])
    packed, _target_cache = _prompt_runtime(
        tmp_path,
        packed_agreement,
        packed_agreements=True,
    )

    def run(runtime: KimiK3DSparkRequestRuntime) -> None:
        assert runtime.agree_text("detokenizer output", lambda: "piece") == "piece"
        assert runtime.agree_local_value("response construction", lambda: 1) == 1
        assert runtime.agree_response_control(
            lambda: (11, None, False, "tail", "piece")
        ) == (11, None, False, "tail", "piece")
        assert runtime.agree_local_value("public response construction", lambda: 2) == 2
        runtime.agree_local_side_effect("generation progress callback", lambda: None)

    run(legacy)
    run(packed)

    legacy_stage_calls = len(
        [event for event in legacy_agreement.events if event.startswith("agree_stage_")]
    )
    assert legacy_stage_calls == 5
    assert legacy_agreement.token_calls == 16
    assert legacy_stage_calls + legacy_agreement.token_calls == 21
    assert packed_agreement.token_calls == 0
    assert len(packed_agreement.packed_calls) == 5
    assert [call[0] for call in packed_agreement.packed_calls] == [
        "text:detokenizer output",
        "value:response construction",
        "response-control",
        "value:public response construction",
        "side-effect:generation progress callback",
    ]


def test_packed_text_and_control_mismatch_fail_before_next_side_effect(
    tmp_path: Path,
) -> None:
    text_agreement = _FakeAgreement([], reject_packed_call=1)
    text_runtime, _target_cache = _prompt_runtime(
        tmp_path,
        text_agreement,
        packed_agreements=True,
    )
    with pytest.raises(DSparkDistributedStateError, match="detokenizer output"):
        text_runtime.agree_text("detokenizer output", lambda: "rank-local text")

    control_agreement = _FakeAgreement([], reject_packed_call=1)
    control_runtime, _target_cache = _prompt_runtime(
        tmp_path,
        control_agreement,
        packed_agreements=True,
    )
    callbacks: list[None] = []
    with pytest.raises(DSparkDistributedStateError, match="response control"):
        control_runtime.agree_response_control(
            lambda: (11, None, False, "tail", "piece")
        )

    assert callbacks == []


def test_asymmetric_detokenizer_text_is_rejected_before_stop_control(
    tmp_path: Path,
) -> None:
    # Operation-success fingerprint is call 1; text digest begins at call 2.
    runtime, _target_cache = _prompt_runtime(
        tmp_path,
        _FakeAgreement([], reject_token_call=2),
    )

    with pytest.raises(DSparkDistributedStateError, match="detokenizer output"):
        runtime.agree_text("detokenizer output", lambda: "rank-local text")


def test_asymmetric_response_control_is_rejected_before_callback(
    tmp_path: Path,
) -> None:
    # Operation-success fingerprint is call 1; response token is call 2.
    runtime, _target_cache = _prompt_runtime(
        tmp_path,
        _FakeAgreement([], reject_token_call=2),
    )

    with pytest.raises(DSparkDistributedStateError, match="response control"):
        runtime.agree_response_control(lambda: (11, None, False, "tail", "piece"))


def test_decode_telemetry_observer_failures_are_nonfatal() -> None:
    engine = _FakeRoundEngine([(11, 12, 13)])

    def fail(*_args: object) -> None:
        raise RuntimeError("telemetry sink failed")

    decoded = list(
        dspark_decode_tokens(
            engine,
            anchor_token=10,
            max_tokens=3,
            eos_token_ids=(),
            round_observer=fail,
            token_observer=fail,
        )
    )

    assert [item.token for item in decoded] == [11, 12, 13]


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
    layers: tuple[object, ...] = field(default_factory=lambda: (object(), object()))
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
                _FakeHidden((1, width, 7168), f"tap-{index}") for index in range(5)
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
                _FakeHidden((1, width, 7168), f"tap-{index}") for index in range(5)
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


def _complete_prefix_pair(
    logical_prompt_tokens: tuple[int, ...],
) -> tuple[list[object], list[_FakeContextCache]]:
    prompt_prefix = logical_prompt_tokens[:-1]
    offset = len(prompt_prefix)
    target_cache: list[object] = [
        _FakeTargetCache(
            offset=offset,
            state=("history", tuple(prompt_prefix)),
        ),
        _FakeKDATargetCache(
            cache=[
                ("kda-conv", tuple(prompt_prefix)),
                {"ssm": tuple(prompt_prefix)},
            ]
        ),
    ]
    context_cache = [
        _FakeContextCache(offset, object(), object()),
        _FakeContextCache(offset, object(), object()),
    ]
    return target_cache, context_cache


def _target_with_prefix_identity(metadata_contract_sha256: str) -> SimpleNamespace:
    target = SimpleNamespace()
    setattr(
        target,
        dspark_module.RANK_LOCAL_METADATA_CONTRACT_MODEL_ATTRIBUTE,
        metadata_contract_sha256,
    )
    setattr(
        target,
        dspark_module.RANK_LOCAL_RUNTIME_MLX_LM_COMMIT_MODEL_ATTRIBUTE,
        dspark_module.KIMI_K3_DSPARK_PREFIX_CACHE_MLX_LM_COMMIT,
    )
    return target


def test_paired_prefix_binding_rejects_stale_target_checkpoint(
    tmp_path: Path,
) -> None:
    first_target = _target_with_prefix_identity("1" * 64)
    replacement_target = _target_with_prefix_identity("2" * 64)
    loaded = LoadedMlxDSpark(
        config=_config(tmp_path, 3, prefix_cache=True),
        target_model=first_target,
        drafter=object(),
        proposer=object(),
        target_route_top_k=8,
    )
    replacement_loaded = LoadedMlxDSpark(
        config=loaded.config,
        target_model=replacement_target,
        drafter=object(),
        proposer=object(),
        target_route_top_k=8,
    )
    first_binding = dspark_module.kimi_k3_dspark_prefix_model_binding(
        loaded,
        first_target,
        "kernelpool/Kimi-K3-2bit-UVMAX",
    )
    replacement_binding = dspark_module.kimi_k3_dspark_prefix_model_binding(
        replacement_loaded,
        replacement_target,
        "kernelpool/Kimi-K3-2bit-UVMAX",
    )
    assert first_binding != replacement_binding

    prompt = (1, 2, 3, 4)
    target_cache, context_cache = _complete_prefix_pair(prompt)
    cache = KimiK3DSparkPrefixCache()
    cache.commit(
        cache.stage(
            prompt,
            target_cache,
            context_cache,
            model_binding=first_binding,
            prefill_tps=1.0,
            evaluate=lambda *_values: None,
        )
    )

    lookup = cache.lookup(
        prompt,
        model_binding=replacement_binding,
        evaluate=lambda *_values: None,
    )

    assert lookup.hit_kind == "miss"
    assert lookup.target_cache is None
    assert lookup.draft_context_cache is None
    assert cache.entry is None


def test_paired_prefix_binding_retains_draft_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    target = _target_with_prefix_identity("1" * 64)
    old = LoadedMlxDSpark(
        config=_config(tmp_path, 3, prefix_cache=True),
        target_model=target,
        drafter=object(),
        proposer=object(),
        target_route_top_k=8,
    )
    yarn = LoadedMlxDSpark(
        config=KimiK3DSparkConfig(
            checkpoint_path=tmp_path,
            verify_width=3,
            round_telemetry=False,
            revision=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_REVISION,
            config_sha256=(dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_CONFIG_SHA256),
            model_sha256=dspark_module.RADIXARK_KIMI_K3_DSPARK_YARN_MODEL_SHA256,
            prefix_cache=True,
        ),
        target_model=target,
        drafter=object(),
        proposer=object(),
        target_route_top_k=8,
    )

    old_binding = dspark_module.kimi_k3_dspark_prefix_model_binding(
        old,
        target,
        "kernelpool/Kimi-K3-2bit-UVMAX",
    )
    yarn_binding = dspark_module.kimi_k3_dspark_prefix_model_binding(
        yarn,
        target,
        "kernelpool/Kimi-K3-2bit-UVMAX",
    )

    assert old_binding != yarn_binding


def test_paired_prefix_binding_requires_golden_target_identity(
    tmp_path: Path,
) -> None:
    target = SimpleNamespace()
    loaded = LoadedMlxDSpark(
        config=_config(tmp_path, 3, prefix_cache=True),
        target_model=target,
        drafter=object(),
        proposer=object(),
    )

    with pytest.raises(DSparkConfigurationError, match="metadata contract digest"):
        dspark_module.kimi_k3_dspark_prefix_model_binding(
            loaded,
            target,
            "kernelpool/Kimi-K3-2bit-UVMAX",
        )

    setattr(
        target,
        dspark_module.RANK_LOCAL_METADATA_CONTRACT_MODEL_ATTRIBUTE,
        "1" * 64,
    )
    setattr(
        target,
        dspark_module.RANK_LOCAL_RUNTIME_MLX_LM_COMMIT_MODEL_ATTRIBUTE,
        "unreviewed-runtime",
    )
    with pytest.raises(DSparkConfigurationError, match="pinned golden MLX-LM"):
        dspark_module.kimi_k3_dspark_prefix_model_binding(
            loaded,
            target,
            "kernelpool/Kimi-K3-2bit-UVMAX",
        )


def _restored_prefix_runtime(
    tmp_path: Path,
    *,
    target_model: _FakeRuntimeTarget,
    proposer: _FakeMlxProposer,
    target_cache: object,
    context_cache: object,
    agreement: _FakeAgreement,
    prefix_cache: KimiK3DSparkPrefixCache | None,
    hit_kind: Literal["miss", "exact", "append"],
    restored_offset: int,
    model_binding: tuple[int, ...] = (1, 2, 3, 4),
) -> KimiK3DSparkRequestRuntime:
    loaded = LoadedMlxDSpark(
        config=_config(
            tmp_path,
            3,
            prefix_cache=prefix_cache is not None,
        ),
        target_model=target_model,
        drafter=object(),
        proposer=proposer,
        evaluate=lambda *_values: None,
    )
    return KimiK3DSparkRequestRuntime(
        loaded=loaded,
        target_model=target_model,
        target_cache=target_cache,
        draft=MlxDSparkRequestDraft(
            proposer=proposer,
            context_cache=context_cache,
            verify_width=3,
            evaluate=lambda *_values: None,
        ),
        collective=agreement,
        prefix_cache=prefix_cache,
        prefix_hit_kind=hit_kind,
        restored_offset=restored_offset,
        prefix_model_binding=model_binding,
        evaluate=lambda *_values: None,
    )


def _patch_deterministic_prompt_forward(
    monkeypatch: pytest.MonkeyPatch,
    runtime: KimiK3DSparkRequestRuntime,
    history: list[int],
    graph_chunks: list[tuple[int, ...]],
) -> None:
    def build(
        chunk: _FakePromptBatch,
        *,
        initial_offset: int,
        speculative_width: int | None = None,
    ) -> _FakePendingForward:
        assert speculative_width is None
        assert initial_offset == len(history)
        values = tuple(chunk.values)
        graph_chunks.append(values)
        history.extend(values)
        entries = cast(list[object], runtime.target_cache)
        mla_cache = cast(_FakeTargetCache, entries[0])
        kda_cache = cast(_FakeKDATargetCache, entries[1])
        mla_cache.offset = len(history)
        mla_cache.state = ("history", tuple(history))
        kda_cache.cache = [
            ("kda-conv", tuple(history)),
            {"ssm": tuple(history)},
        ]
        return _FakePendingForward(
            SimpleNamespace(
                aux_hidden_states=tuple(
                    _FakeHidden((1, len(values), 7168), f"tap-{index}")
                    for index in range(5)
                )
            )
        )

    monkeypatch.setattr(runtime, "_build_forward_with_taps", build)
    monkeypatch.setattr(dspark_module.mx, "clear_cache", lambda: None, raising=False)


@pytest.mark.parametrize(
    ("target_offset", "draft_offset", "match"),
    [
        (0, 0, "offset 3"),
        (2, 2, "offset 3"),
        (3, 2, "target and draft offsets disagree"),
    ],
    ids=["zero", "nonterminal", "target-draft-mismatch"],
)
def test_paired_prefix_stage_rejects_invalid_terminal_boundaries(
    target_offset: int,
    draft_offset: int,
    match: str,
) -> None:
    prompt = (1, 2, 3, 4)
    target_cache: list[object] = [
        _FakeTargetCache(target_offset, ("state",)),
        _FakeKDATargetCache([("conv",), {"ssm": "state"}]),
    ]
    context_cache = [
        _FakeContextCache(draft_offset, object(), object()),
        _FakeContextCache(draft_offset, object(), object()),
    ]

    with pytest.raises(ValueError, match=match):
        KimiK3DSparkPrefixCache().stage(
            prompt,
            target_cache,
            context_cache,
            model_binding=(1, 2, 3, 4),
            prefill_tps=1.0,
            evaluate=lambda *_values: None,
        )


def test_disabled_prefix_inspection_is_nonmutating_then_restore_clears() -> None:
    prompt = (1, 2, 3, 4)
    target_cache, context_cache = _complete_prefix_pair(prompt)
    cache = KimiK3DSparkPrefixCache()
    cache.commit(
        cache.stage(
            prompt,
            target_cache,
            context_cache,
            model_binding=(1, 2, 3, 4),
            prefill_tps=1.0,
            evaluate=lambda *_values: None,
        )
    )
    retained = cache.entry

    inspection = cache.inspect(
        prompt,
        model_binding=(1, 2, 3, 4),
        enabled=False,
    )
    assert inspection.hit_kind == "disabled"
    assert cache.entry is retained

    lookup = cache.restore(inspection, evaluate=lambda *_values: None)
    assert lookup.hit_kind == "disabled"
    assert lookup.target_cache is None
    assert cache.entry is None


def test_paired_prefix_cache_exact_repeat_clones_both_halves_without_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = (1, 2, 3, 4)
    target_cache, context_cache = _complete_prefix_pair(prompt)
    cache = KimiK3DSparkPrefixCache()
    staged = cache.stage(
        prompt,
        target_cache,
        context_cache,
        model_binding=(1, 2, 3, 4),
        prefill_tps=12.5,
        evaluate=lambda *_values: None,
    )
    cache.commit(staged)
    retained_entry = cache.entry
    assert retained_entry is not None

    lookup = cache.lookup(
        prompt,
        model_binding=(1, 2, 3, 4),
        evaluate=lambda *_values: None,
    )
    assert lookup.hit_kind == "exact"
    assert lookup.restored_offset == 3
    assert lookup.target_cache is not retained_entry.target_cache
    assert lookup.draft_context_cache is not retained_entry.draft_context_cache
    assert staged.logical_prompt_tokens == prompt
    assert not hasattr(staged, "prompt")
    assert all(type(token) is int for token in staged.logical_prompt_tokens)

    target_model = _FakeRuntimeTarget()
    proposer = _FakeMlxProposer(3, [], [[11, 12]])
    runtime = _restored_prefix_runtime(
        tmp_path,
        target_model=target_model,
        proposer=proposer,
        target_cache=lookup.target_cache,
        context_cache=lookup.draft_context_cache,
        agreement=_FakeAgreement([]),
        prefix_cache=cache,
        hit_kind="exact",
        restored_offset=lookup.restored_offset,
    )
    graph_chunks: list[tuple[int, ...]] = []
    _patch_deterministic_prompt_forward(
        monkeypatch,
        runtime,
        [1, 2, 3],
        graph_chunks,
    )
    progress: list[tuple[int, int]] = []
    prefill_tps, newly_processed = runtime.seed_prompt(
        cast(mx.array, cast(object, _FakePromptArray(prompt[:-1]))),
        prefill_step_size=8,
        max_tokens=16,
        stop_sequences=(),
        progress_callback=lambda done, total: progress.append((done, total)),
        distributed_progress_callback=None,
    )
    runtime.commit_prompt_prefix_cache(prompt, prefill_tps=prefill_tps)

    assert graph_chunks == []
    assert newly_processed == 0
    assert prefill_tps == 0.0
    assert progress == [(3, 3)]
    assert cache.entry is retained_entry


def test_strict_append_reuses_l_minus_one_and_matches_cold_prefill_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_prompt = (1, 2, 3, 4)
    extended_prompt = (1, 2, 3, 4, 5, 6)
    base_target, base_context = _complete_prefix_pair(base_prompt)
    cache = KimiK3DSparkPrefixCache()
    cache.commit(
        cache.stage(
            base_prompt,
            base_target,
            base_context,
            model_binding=(1, 2, 3, 4),
            prefill_tps=9.0,
            evaluate=lambda *_values: None,
        )
    )
    lookup = cache.lookup(
        extended_prompt,
        model_binding=(1, 2, 3, 4),
        evaluate=lambda *_values: None,
    )
    assert lookup.hit_kind == "append"
    assert lookup.restored_offset == len(base_prompt) - 1

    append_model = _FakeRuntimeTarget()
    append_proposer = _FakeMlxProposer(3, [], [[11, 12]])
    append_runtime = _restored_prefix_runtime(
        tmp_path,
        target_model=append_model,
        proposer=append_proposer,
        target_cache=lookup.target_cache,
        context_cache=lookup.draft_context_cache,
        agreement=_FakeAgreement([]),
        prefix_cache=cache,
        hit_kind="append",
        restored_offset=lookup.restored_offset,
    )
    append_history = list(base_prompt[:-1])
    append_chunks: list[tuple[int, ...]] = []
    _patch_deterministic_prompt_forward(
        monkeypatch,
        append_runtime,
        append_history,
        append_chunks,
    )
    append_tps, append_work = append_runtime.seed_prompt(
        cast(mx.array, cast(object, _FakePromptArray(extended_prompt[:-1]))),
        prefill_step_size=8,
        max_tokens=16,
        stop_sequences=(),
        progress_callback=lambda _done, _total: None,
        distributed_progress_callback=None,
    )
    append_runtime.commit_prompt_prefix_cache(
        extended_prompt,
        prefill_tps=append_tps,
    )

    cold_target: list[object] = [_FakeTargetCache(), _FakeKDATargetCache()]
    cold_context = [_FakeContextCache(), _FakeContextCache()]
    cold_model = _FakeRuntimeTarget()
    cold_proposer = _FakeMlxProposer(3, [], [[11, 12]])
    cold_runtime = _restored_prefix_runtime(
        tmp_path,
        target_model=cold_model,
        proposer=cold_proposer,
        target_cache=cold_target,
        context_cache=cold_context,
        agreement=_FakeAgreement([]),
        prefix_cache=None,
        hit_kind="miss",
        restored_offset=0,
    )
    cold_history: list[int] = []
    cold_chunks: list[tuple[int, ...]] = []
    _patch_deterministic_prompt_forward(
        monkeypatch,
        cold_runtime,
        cold_history,
        cold_chunks,
    )
    _cold_tps, cold_work = cold_runtime.seed_prompt(
        cast(mx.array, cast(object, _FakePromptArray(extended_prompt[:-1]))),
        prefill_step_size=8,
        max_tokens=16,
        stop_sequences=(),
        progress_callback=lambda _done, _total: None,
        distributed_progress_callback=None,
    )

    assert append_chunks == [(4, 5)]
    assert cold_chunks == [(1, 2, 3, 4, 5)]
    assert append_work == 2
    assert cold_work == 5
    assert append_history == cold_history == list(extended_prompt[:-1])
    assert append_runtime.target_cache == cold_runtime.target_cache
    assert [entry.length for entry in append_runtime.draft.context_cache] == [
        entry.length for entry in cold_runtime.draft.context_cache
    ]
    assert cache.entry is not None
    assert cache.entry.logical_prompt_tokens == extended_prompt


def test_paired_prefix_cache_mismatch_eviction_and_validation_fail_closed() -> None:
    first_prompt = (1, 2, 3, 4)
    second_prompt = (5, 6, 7, 8)
    cache = KimiK3DSparkPrefixCache()
    first_target, first_context = _complete_prefix_pair(first_prompt)
    first_entry = cache.stage(
        first_prompt,
        first_target,
        first_context,
        model_binding=(1, 2, 3, 4),
        prefill_tps=1.0,
        evaluate=lambda *_values: None,
    )
    cache.commit(first_entry)

    mismatch = cache.lookup(
        first_prompt,
        model_binding=(4, 3, 2, 1),
        evaluate=lambda *_values: None,
    )
    assert mismatch.hit_kind == "miss"
    assert cache.entry is None

    with pytest.raises(ValueError, match="validated staged entry"):
        cache.commit(first_entry)
    cache.commit(
        cache.stage(
            first_prompt,
            first_target,
            first_context,
            model_binding=(1, 2, 3, 4),
            prefill_tps=1.0,
            evaluate=lambda *_values: None,
        )
    )
    second_target, second_context = _complete_prefix_pair(second_prompt)
    cache.commit(
        cache.stage(
            second_prompt,
            second_target,
            second_context,
            model_binding=(1, 2, 3, 4),
            prefill_tps=2.0,
            evaluate=lambda *_values: None,
        )
    )
    assert cache.entry is not None
    assert cache.entry.logical_prompt_tokens == second_prompt

    corrupted_target = cast(list[object], cache.entry.target_cache)
    cast(_FakeTargetCache, corrupted_target[0]).offset -= 1
    with pytest.raises(ValueError, match="offset"):
        cache.lookup(
            second_prompt,
            model_binding=(1, 2, 3, 4),
            evaluate=lambda *_values: None,
        )
    assert cache.entry is None


@pytest.mark.parametrize(
    "query",
    [
        (1, 2, 3),
        (1, 2, 9, 4),
        (8, 9, 10, 11),
    ],
    ids=["shortened", "edited", "unrelated"],
)
def test_paired_prefix_non_append_mismatch_clears_and_returns_cold(
    query: tuple[int, ...],
) -> None:
    prompt = (1, 2, 3, 4)
    target_cache, context_cache = _complete_prefix_pair(prompt)
    cache = KimiK3DSparkPrefixCache()
    cache.commit(
        cache.stage(
            prompt,
            target_cache,
            context_cache,
            model_binding=(1, 2, 3, 4),
            prefill_tps=1.0,
            evaluate=lambda *_values: None,
        )
    )

    lookup = cache.lookup(
        query,
        model_binding=(1, 2, 3, 4),
        evaluate=lambda *_values: None,
    )

    assert lookup.hit_kind == "miss"
    assert lookup.restored_offset == 0
    assert lookup.target_cache is None
    assert lookup.draft_context_cache is None
    assert cache.entry is None


@dataclass
class _TracingAgreement(_FakeAgreement):
    trace: list[str] = field(default_factory=list)

    def agree_stage_success(self, local_success: bool) -> bool | None:
        self.trace.append(f"agree_stage:{int(local_success)}")
        return super().agree_stage_success(local_success)

    def agree_token(self, local_token: int | None) -> int | None:
        self.trace.append("agree_token")
        return super().agree_token(local_token)


class _TracingPrefixCache(KimiK3DSparkPrefixCache):
    def __init__(self, trace: list[str]) -> None:
        super().__init__()
        self.trace = trace

    def stage(self, *args: object, **kwargs: object) -> object:
        self.trace.append("stage")
        return super().stage(*args, **kwargs)  # type: ignore[arg-type]

    def commit(self, entry: object) -> None:
        self.trace.append("commit")
        super().commit(entry)  # type: ignore[arg-type]


def test_paired_prefix_commit_order_and_peer_failure_roll_back(
    tmp_path: Path,
) -> None:
    prompt = (1, 2, 3, 4)
    target_cache, context_cache = _complete_prefix_pair(prompt)
    trace: list[str] = []
    cache = _TracingPrefixCache(trace)
    agreement = _TracingAgreement(
        [],
        stage_outcomes=[True, None],
        trace=trace,
    )
    runtime = _restored_prefix_runtime(
        tmp_path,
        target_model=_FakeRuntimeTarget(),
        proposer=_FakeMlxProposer(3, [], [[11, 12]]),
        target_cache=target_cache,
        context_cache=context_cache,
        agreement=agreement,
        prefix_cache=cache,
        hit_kind="append",
        restored_offset=len(prompt) - 1,
    )

    with pytest.raises(DSparkDistributedStateError, match="commit"):
        runtime.commit_prompt_prefix_cache(prompt, prefill_tps=1.0)

    assert trace[0] == "stage"
    assert "commit" in trace
    assert trace.index("stage") < trace.index("agree_token") < trace.index("commit")
    assert trace[-2:] == ["agree_stage:1", "agree_token"]
    assert cache.entry is None


def test_paired_prefix_stage_error_clears_prior_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = (1, 2, 3, 4)
    target_cache, context_cache = _complete_prefix_pair(prompt)
    cache = KimiK3DSparkPrefixCache()
    cache.commit(
        cache.stage(
            prompt,
            target_cache,
            context_cache,
            model_binding=(1, 2, 3, 4),
            prefill_tps=1.0,
            evaluate=lambda *_values: None,
        )
    )
    runtime = _restored_prefix_runtime(
        tmp_path,
        target_model=_FakeRuntimeTarget(),
        proposer=_FakeMlxProposer(3, [], [[11, 12]]),
        target_cache=target_cache,
        context_cache=context_cache,
        agreement=_FakeAgreement([]),
        prefix_cache=cache,
        hit_kind="append",
        restored_offset=len(prompt) - 1,
    )
    monkeypatch.setattr(
        cache,
        "stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected clone failure")
        ),
    )

    with pytest.raises(DSparkDistributedStateError, match="staging"):
        runtime.commit_prompt_prefix_cache(prompt, prefill_tps=1.0)
    assert cache.entry is None


def test_real_mlx_paired_prefix_clone_mutation_preserves_saved_arrays() -> None:
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.kimi_k3_dspark import KimiK3DSparkContextCache

    prompt = (1, 2, 3, 4)
    kv_cache = KVCache()
    initial_keys = mx.arange(6, dtype=mx.float32).reshape(1, 1, 3, 2)
    initial_values = initial_keys + 10
    kv_cache.update_and_fetch(initial_keys, initial_values)
    kda_cache = ArraysCache(2)
    kda_cache.cache = [
        mx.arange(4, dtype=mx.float32).reshape(1, 4),
        mx.arange(4, dtype=mx.float32).reshape(1, 4) + 20,
    ]
    target_cache = [kv_cache, kda_cache]
    draft_context = [
        KimiK3DSparkContextCache(capacity_hint=8, step=4),
        KimiK3DSparkContextCache(capacity_hint=8, step=4),
    ]
    for context in draft_context:
        context.append(initial_keys, initial_values)
    mx.eval(
        initial_keys,
        initial_values,
        *kda_cache.cache,
        *(context.keys for context in draft_context),
        *(context.values for context in draft_context),
    )

    cache = KimiK3DSparkPrefixCache()
    cache.commit(
        cache.stage(
            prompt,
            target_cache,
            draft_context,
            model_binding=(1, 2, 3, 4),
            prefill_tps=1.0,
            evaluate=mx.eval,
        )
    )
    retained = cache.entry
    assert retained is not None
    retained_target = cast(list[object], retained.target_cache)
    retained_draft = cast(
        list[KimiK3DSparkContextCache],
        retained.draft_context_cache,
    )
    retained_bytes = (
        np.asarray(cast(KVCache, retained_target[0]).keys).tobytes(),
        np.asarray(cast(ArraysCache, retained_target[1]).cache[0]).tobytes(),
        np.asarray(retained_draft[0].keys).tobytes(),
    )

    lookup = cache.lookup(
        prompt,
        model_binding=(1, 2, 3, 4),
        evaluate=mx.eval,
    )
    active_target = cast(list[object], lookup.target_cache)
    active_draft = cast(
        list[KimiK3DSparkContextCache],
        lookup.draft_context_cache,
    )
    extra_keys = mx.full((1, 1, 1, 2), 99, dtype=mx.float32)
    extra_values = mx.full((1, 1, 1, 2), 199, dtype=mx.float32)
    cast(KVCache, active_target[0]).update_and_fetch(extra_keys, extra_values)
    active_kda = cast(ArraysCache, active_target[1])
    active_kda.cache[0] = active_kda.cache[0] + 100
    for context in active_draft:
        context.append(extra_keys, extra_values)
    mx.eval(
        cast(KVCache, active_target[0]).keys,
        *active_kda.cache,
        *(context.keys for context in active_draft),
    )

    assert cast(KVCache, active_target[0]).offset == 4
    assert all(context.length == 4 for context in active_draft)
    assert cast(KVCache, retained_target[0]).offset == 3
    assert all(context.length == 3 for context in retained_draft)
    assert retained_bytes == (
        np.asarray(cast(KVCache, retained_target[0]).keys).tobytes(),
        np.asarray(cast(ArraysCache, retained_target[1]).cache[0]).tobytes(),
        np.asarray(retained_draft[0].keys).tobytes(),
    )
