from __future__ import annotations

import hashlib
from pathlib import Path

from exo.worker.engines.mlx.rank_local_checkpoint import (
    PINNED_RANK_LOCAL_METADATA_FILES,
    SUPPORTED_LOADER_SHA256,
)


def test_checked_in_rank_local_loader_matches_runtime_pin() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    loader = repo_root / "scripts/kimi_k3_tp2/rank_local_loader.py"
    assert loader.is_file()
    assert hashlib.sha256(loader.read_bytes()).hexdigest() == (SUPPORTED_LOADER_SHA256)


def test_rank_local_metadata_pin_matches_exact_edb511_runtime_inventory() -> None:
    assert set(PINNED_RANK_LOCAL_METADATA_FILES) == {
        "README.md",
        "added_tokens.json",
        "config.json",
        "configuration_kimi_k3.py",
        "encoding_k3.py",
        "generation_config.json",
        "kimi_k3_processor.py",
        "kimi_k3_vision_processing.py",
        "media_utils.py",
        "modeling_kimi_k3.py",
        "modeling_kimi_linear.py",
        "preprocessor_config.json",
        "tiktoken.model",
        "tokenization_kimi.py",
        "tokenizer_config.json",
    }
    assert PINNED_RANK_LOCAL_METADATA_FILES["tiktoken.model"] == {
        "bytes": 2_795_286,
        "sha256": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    }
    assert not any(
        name.startswith("LICENSE") for name in PINNED_RANK_LOCAL_METADATA_FILES
    )
