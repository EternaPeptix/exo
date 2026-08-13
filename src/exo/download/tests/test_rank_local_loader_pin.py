from exo.worker.engines.mlx import rank_local_checkpoint


def test_width_four_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "9f10d5572f43c2dac34a9dfb2969118dad9d42326c36eace28d7b63a80897639"
    )
