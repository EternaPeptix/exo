from exo.worker.engines.mlx import rank_local_checkpoint


def test_top8_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "e03fc84b252d15313e55bccd4f790b1473a9eab1685eee7647df5c97a4d9a67a"
    )
