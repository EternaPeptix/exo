from exo.worker.engines.mlx import rank_local_checkpoint


def test_width_four_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "de9a611233eb70dbde747301bb2a07bbf9feac0b07dc843c7bc7656137cba5ec"
    )
