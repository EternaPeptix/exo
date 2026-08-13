from exo.worker.engines.mlx import rank_local_checkpoint


def test_width_four_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "ebda8aa3bd3844cd64b37149ad5099c9b517293803f224217357b4fd1c712ab2"
    )
