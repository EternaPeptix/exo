from exo.worker.engines.mlx import rank_local_checkpoint


def test_top8_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "f3c7babccc96c1805e383d4d9016e6a88f497450b70f877e49cd1d0cf21c9117"
    )
