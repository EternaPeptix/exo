from exo.worker.engines.mlx import rank_local_checkpoint


def test_current_offline_w3_composition_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "f7059a2d45cffedf7614174e44a326f4f14372951231eac4159094eea8b3e67c"
    )


def test_previous_rank_local_loaders_are_not_accepted() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 not in {
        "9f10d5572f43c2dac34a9dfb2969118dad9d42326c36eace28d7b63a80897639",
        "3619ab75364182e6fd6e961353943b4965a91ccd53dedc6621527ea16b7fd021",
        "499c0d013980e0ed4f53433acd7bf5e3a0c9f9bba2138545ae07d210c8cd78eb",
        "d5e532c1c053b8cced19f5e4d80fb41f6e2b5748e5097ea1895a6b9c88477141",
    }
