from __future__ import annotations

import hashlib
from pathlib import Path

from exo.worker.engines.mlx.rank_local_checkpoint import (
    SUPPORTED_LOADER_SHA256,
)


def test_checked_in_rank_local_loader_matches_runtime_pin() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    loader = repo_root / "scripts/kimi_k3_tp2/rank_local_loader.py"
    assert loader.is_file()
    assert hashlib.sha256(loader.read_bytes()).hexdigest() == (SUPPORTED_LOADER_SHA256)
