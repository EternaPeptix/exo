from exo.worker.engines.mlx import rank_local_checkpoint


def test_sequential_q3_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "81ca44ddf2243d0178c34b46b664fa73ffb1b6ce766693573743c4d820b65a7c"
    )
