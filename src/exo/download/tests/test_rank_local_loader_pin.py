from exo.worker.engines.mlx import rank_local_checkpoint


def test_width_four_rank_local_loader_hash_is_pinned() -> None:
    assert rank_local_checkpoint.SUPPORTED_LOADER_SHA256 == (
        "1f7e344b0f99636f567394294304f9a5c7b9730b301380e173fddd1b921794f1"
    )
