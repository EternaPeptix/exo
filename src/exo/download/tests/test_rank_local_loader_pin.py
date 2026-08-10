from exo.worker.engines.mlx import rank_local_checkpoint


def test_width_four_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "dadc21b38669eb70c3b3b90c7b15054cafa6e512d81f0e2aa472ad26c49d304b"
    )
