"""Tests for _rotate_metal_to_middle: heterogeneous Metal+CUDA ring ordering.

The MLX 0.32.0 ring backend hangs on CUDA↔CUDA send/recv. The rotation must
arrange the cycle so that CUDA↔CUDA ring-neighbor pairs are minimised (and the
maximum CUDA-only run length is ceil(#CUDA / #Metal)). These cases cover the
3-node original target plus the 4/5/6-node heterogeneous clusters the rotation
was previously broken for.
"""
from __future__ import annotations

from itertools import permutations

import pytest

from exo.master.placement import _rotate_metal_to_middle
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.topology import Cycle


def _backends(spec: list[str]) -> tuple[dict[NodeId, list[Backend]], list[NodeId]]:
    """Build node_backends + an arbitrary cycle order from a spec like ['C','M','C']."""
    backends: dict[NodeId, list[Backend]] = {}
    order: list[NodeId] = []
    for i, kind in enumerate(spec):
        nid = NodeId(f"node{i}")
        order.append(nid)
        bs: list[Backend] = []
        if "C" in kind:
            bs.append(Backend.MlxCuda)
        if "M" in kind:
            bs.append(Backend.MlxMetal)
        backends[nid] = bs
    return backends, order


def _is_cuda_only(nid: NodeId, backends) -> bool:
    bs = backends[nid]
    return Backend.MlxCuda in bs and Backend.MlxMetal not in bs


def _cuda_cuda_pairs(order: list[NodeId], backends) -> int:
    """Count CUDA↔CUDA neighbour pairs in the cyclic ring (each is a hang risk)."""
    n = len(order)
    return sum(
        1
        for i in range(n)
        if _is_cuda_only(order[i], backends) and _is_cuda_only(order[(i + 1) % n], backends)
    )


def _max_cuda_run(order: list[NodeId], backends) -> int:
    """Longest run of consecutive CUDA-only nodes in the cyclic ring."""
    n = len(order)
    best = 0
    cur = 0
    for i in range(2 * n):
        if _is_cuda_only(order[i % n], backends):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return min(best, n)


def _brute_force_min_pairs(backends: dict[NodeId, list[Backend]]) -> int:
    """Exact minimum CUDA↔CUDA pairs over all cyclic orderings (small N only)."""
    nodes = list(backends.keys())
    if len(nodes) <= 1:
        return 0
    first = nodes[0]
    best = len(nodes) + 1
    for perm in permutations(nodes[1:]):
        best = min(best, _cuda_cuda_pairs([first, *perm], backends))
    return best


@pytest.mark.parametrize(
    "spec",
    [
        ["C", "M", "C"],            # 3-node 2C+1M (PR #2129 original target)
        ["C", "C", "M"],            # 3-node 2C+1M, Metal at end
        ["C", "C", "C", "M"],       # 4-node 3C+1M
        ["C", "C", "C", "C", "M"],  # 5-node 4C+1M
        ["C", "C", "C", "M", "M"],  # 5-node 3C+2M
        ["C", "C", "C", "C", "M", "M"],  # 6-node 4C+2M (4x Spark + 2x Mac)
        ["C", "C", "C", "C", "C", "M"],  # 6-node 5C+1M
        ["C", "C", "M", "M"],       # 4-node 2C+2M (zero achievable)
    ],
)
def test_rotate_minimises_cuda_to_cuda_ring_pairs(spec: list[str]) -> None:
    """Rotation must reach the brute-force minimum of CUDA↔CUDA ring pairs."""
    backends, order = _backends(spec)
    rotated = _rotate_metal_to_middle(Cycle(node_ids=order), backends)
    rotated_nodes = list(rotated)
    assert sorted(rotated_nodes) == sorted(order), "rotation must preserve node set"
    got = _cuda_cuda_pairs(rotated_nodes, backends)
    optimal = _brute_force_min_pairs(backends)
    assert got == optimal, (
        f"spec={spec} got {got} CUDA↔CUDA pairs, optimal is {optimal}; "
        f"rotated={[str(n) for n in rotated_nodes]}"
    )


@pytest.mark.parametrize(
    "spec",
    [
        ["C", "C", "C", "M"],
        ["C", "C", "C", "C", "M"],
        ["C", "C", "C", "C", "M", "M"],
        ["C", "C", "C", "C", "C", "M"],
    ],
)
def test_rotate_achieves_minimal_max_cuda_run(spec: list[str]) -> None:
    """The longest CUDA-only run must equal ceil(#CUDA / #Metal)."""
    backends, order = _backends(spec)
    n_cuda = sum(1 for nid in order if _is_cuda_only(nid, backends))
    n_metal = sum(1 for nid in order if Backend.MlxMetal in backends[nid])
    expected_max_run = -(-n_cuda // n_metal)  # ceil division
    rotated = _rotate_metal_to_middle(Cycle(node_ids=order), backends)
    got = _max_cuda_run(list(rotated), backends)
    assert got == expected_max_run, (
        f"spec={spec} max CUDA run {got}, expected ceil({n_cuda}/{n_metal})={expected_max_run}"
    )


def test_rotate_zero_cuda_pairs_when_feasible() -> None:
    """2 CUDA + 2 Metal must eliminate CUDA↔CUDA pairs entirely."""
    backends, order = _backends(["C", "C", "M", "M"])
    rotated = _rotate_metal_to_middle(Cycle(node_ids=order), backends)
    assert _cuda_cuda_pairs(list(rotated), backends) == 0


def test_rotate_preserves_order_when_no_metal() -> None:
    """All-CUDA cycle: nothing to optimise, return unchanged."""
    backends, order = _backends(["C", "C", "C"])
    rotated = _rotate_metal_to_middle(Cycle(node_ids=order), backends)
    assert list(rotated) == order


def test_rotate_preserves_order_when_all_metal() -> None:
    """All-Metal cycle: nothing to optimise, return unchanged."""
    backends, order = _backends(["M", "M", "M"])
    rotated = _rotate_metal_to_middle(Cycle(node_ids=order), backends)
    assert list(rotated) == order


def test_rotate_returns_cycle_unchanged_for_small_cycles() -> None:
    """<3 node cycles are returned unchanged (no rotation possible/needed)."""
    backends, order = _backends(["C", "M"])
    rotated = _rotate_metal_to_middle(Cycle(node_ids=order), backends)
    assert list(rotated) == order
