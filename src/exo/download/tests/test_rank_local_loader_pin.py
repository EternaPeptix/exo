from exo.worker.engines.mlx import rank_local_checkpoint


def test_width_four_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "7ae67eba9a64e4ca921697680d17c2060cd39a82d88eba724d89d7a72b7946ab"
    )
