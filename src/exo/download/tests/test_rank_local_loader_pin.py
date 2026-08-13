from exo.worker.engines.mlx import rank_local_checkpoint


def test_width_four_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "fca455cb0143152473ea4c8fabca8c7660c519467bd370f8c96bb41283ca9afe"
    )
