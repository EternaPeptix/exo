"""Real two-rank asymmetric-failure coverage for the MLX fail-stop helper."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
from queue import Empty
from typing import Any

import mlx.core as mx

from exo.worker.engines.mlx.utils_mlx import rank_agreed_fail_stop


def _run_asymmetric_failure_rank(
    rank: int,
    hostfile: str,
    result_queue: Any,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)

        def local_operation() -> str:
            if rank == 0:
                raise ValueError("rank-zero parser failure")
            return "rank-one parsed"

        try:
            rank_agreed_fail_stop(
                "two-rank parser",
                group,
                local_operation,
            )
        except RuntimeError as error:
            result_queue.put((rank, True, str(error)))
        else:
            result_queue.put((rank, False, "rank advanced after peer failure"))
    except Exception as error:
        result_queue.put((rank, False, repr(error)))


def test_rank_agreed_fail_stop_coordinates_real_two_rank_asymmetry() -> None:
    context = mp.get_context("spawn")
    hosts = ["127.0.0.1:29960", "127.0.0.1:29961"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        json.dump(hosts, file)
        hostfile = file.name

    result_queue: Any = context.Queue()
    processes = [
        context.Process(
            target=_run_asymmetric_failure_rank,
            args=(rank, hostfile, result_queue),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)

        timed_out = [process.pid for process in processes if process.is_alive()]
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        assert timed_out == [], f"rank process timed out: {timed_out}"
        assert [process.exitcode for process in processes] == [0, 0]

        results: dict[int, tuple[bool, str]] = {}
        try:
            for _ in range(2):
                rank, coordinated, detail = result_queue.get(timeout=2)
                results[rank] = (coordinated, detail)
        except Empty:
            pass

        assert len(results) == 2, f"missing rank result(s): {results}"
        for rank in range(2):
            coordinated, detail = results[rank]
            assert coordinated, f"rank {rank} did not fail-stop: {detail}"
            assert "failed on at least one distributed rank" in detail
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        os.unlink(hostfile)
