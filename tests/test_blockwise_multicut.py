"""Tests for block-wise (lifted) multicut.

The core guarantees, mirroring ``tests/test_runner_parity.py``:
- the block-wise (hierarchical) solve reproduces the whole-graph multicut partition on a clean
  problem, and
- ``local == subprocess`` for the distributed per-block map step (exact, across workers and levels).
"""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.graph import distributed_rag
from bioimage_py.segmentation import (blockwise_lifted_multicut, blockwise_multicut,
                                      lifted_multicut_kernighan_lin, multicut_kernighan_lin, relabel)

# 4x4 superpixels of 8x8 (a 32x32 volume); macro clusters are 2x2 blocks of superpixels (4 macros).
_SP_SIZE = 8
_GRID = 4
_BLOCK_SHAPE = (16, 16)


def _same_partition(a, b):
    """Whether two node labelings define the same partition (ids may differ)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])


def _build_problem():
    """Build ``(segmentation, macro, graph, uv_ids)`` for the macro-cluster test problem."""
    size = _SP_SIZE * _GRID
    sp = np.zeros((size, size), dtype="uint32")
    sid = 0
    for i in range(0, size, _SP_SIZE):
        for j in range(0, size, _SP_SIZE):
            sp[i:i + _SP_SIZE, j:j + _SP_SIZE] = sid
            sid += 1
    ids = np.arange(sid)
    macro = (ids // _GRID // 2) * (_GRID // 2) + (ids % _GRID) // 2
    graph = distributed_rag(sp, block_shape=_BLOCK_SHAPE)
    return sp, macro, graph, np.asarray(graph.uv_ids())


def _clean_costs(uv, macro, attractive=10.0, repulsive=-10.0):
    """Attractive within a macro cluster, repulsive across -> the multicut is exactly ``macro``."""
    return np.where(macro[uv[:, 0]] == macro[uv[:, 1]], attractive, repulsive)


@pytest.mark.parametrize("n_levels", [1, 2, 3])
def test_blockwise_multicut_matches_whole_graph(n_levels):
    sp, macro, graph, uv = _build_problem()
    costs = _clean_costs(uv, macro)

    whole = np.asarray(multicut_kernighan_lin(graph, costs))
    assert _same_partition(whole, macro)  # the clean problem recovers the macro clusters

    node_labels = np.asarray(blockwise_multicut(graph, costs, sp, block_shape=_BLOCK_SHAPE,
                                                n_levels=n_levels))
    assert len(node_labels) == graph.number_of_nodes
    assert _same_partition(node_labels, whole)
    assert _same_partition(node_labels, macro)


@pytest.mark.parametrize("n_levels", [1, 2])
@pytest.mark.parametrize("num_workers", [1, 3])
def test_blockwise_multicut_backend_parity(n_levels, num_workers, zarr_factory):
    sp, macro, graph, uv = _build_problem()
    costs = _clean_costs(uv, macro)
    seg = zarr_factory(sp, chunks=_BLOCK_SHAPE)

    results = {}
    for job in ("local", "subprocess"):
        results[job] = np.asarray(blockwise_multicut(
            graph, costs, seg, block_shape=_BLOCK_SHAPE, n_levels=n_levels,
            num_workers=num_workers, job_type=job))
    # The distributed map step is deterministic -> the node labelings are bit-identical.
    np.testing.assert_array_equal(results["local"], results["subprocess"])


def test_blockwise_multicut_halo():
    sp, macro, graph, uv = _build_problem()
    costs = _clean_costs(uv, macro)
    node_labels = np.asarray(blockwise_multicut(graph, costs, sp, block_shape=_BLOCK_SHAPE,
                                                n_levels=2, halo=(2, 2)))
    assert _same_partition(node_labels, macro)


def test_blockwise_multicut_then_relabel():
    # End to end: solve block-wise, then project the node labeling to pixels with relabel.
    sp, macro, graph, uv = _build_problem()
    costs = _clean_costs(uv, macro)
    node_labels = np.asarray(blockwise_multicut(graph, costs, sp, block_shape=_BLOCK_SHAPE,
                                                n_levels=2)).astype("uint32")
    seg = np.asarray(relabel(sp, node_labels))
    assert seg.shape == sp.shape
    # Projecting the node labeling to pixels is exactly node_labels indexed by the superpixel id.
    np.testing.assert_array_equal(seg, node_labels[sp])
    # The resulting pixel segmentation has exactly one label per macro cluster.
    assert len(np.unique(seg)) == len(np.unique(macro))


# --------------------------------------------------------------------------------------------------
# Lifted
# --------------------------------------------------------------------------------------------------

def _lifted_matters_problem():
    """Base weakly repulsive (plain MC = singletons); within-macro lifted edges (strongly attractive)."""
    sp, macro, graph, uv = _build_problem()
    costs = np.full(len(uv), -1.0)
    pairs = []
    for m in np.unique(macro):
        members = np.where(macro == m)[0]
        for x in members[1:]:
            pairs.append([members[0], x])
    lifted_uv = np.array(pairs, dtype="uint64")
    lifted_costs = np.full(len(lifted_uv), 50.0)
    return sp, macro, graph, costs, lifted_uv, lifted_costs


@pytest.mark.parametrize("n_levels", [1, 2, 3])
def test_blockwise_lifted_matches_whole_graph(n_levels):
    sp, macro, graph, costs, lifted_uv, lifted_costs = _lifted_matters_problem()

    whole = np.asarray(lifted_multicut_kernighan_lin(graph, costs, lifted_uv, lifted_costs))
    assert _same_partition(whole, macro)  # the lifted edges recover the macro clusters

    node_labels = np.asarray(blockwise_lifted_multicut(
        graph, costs, lifted_uv, lifted_costs, sp, block_shape=_BLOCK_SHAPE, n_levels=n_levels))
    assert _same_partition(node_labels, whole)
    assert _same_partition(node_labels, macro)


@pytest.mark.parametrize("num_workers", [1, 3])
def test_blockwise_lifted_backend_parity(num_workers, zarr_factory):
    sp, macro, graph, costs, lifted_uv, lifted_costs = _lifted_matters_problem()
    seg = zarr_factory(sp, chunks=_BLOCK_SHAPE)

    results = {}
    for job in ("local", "subprocess"):
        results[job] = np.asarray(blockwise_lifted_multicut(
            graph, costs, lifted_uv, lifted_costs, seg, block_shape=_BLOCK_SHAPE, n_levels=2,
            num_workers=num_workers, job_type=job))
    np.testing.assert_array_equal(results["local"], results["subprocess"])


# --------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------

def test_blockwise_multicut_invalid_args():
    sp, macro, graph, uv = _build_problem()
    costs = _clean_costs(uv, macro)
    with pytest.raises(ValueError):
        blockwise_multicut(graph, costs, sp, block_shape=_BLOCK_SHAPE, n_levels=0)
    with pytest.raises(ValueError):
        blockwise_multicut(graph, costs, sp, block_shape=(16, 16, 16), n_levels=1)


def test_blockwise_multicut_exported():
    assert hasattr(bp.segmentation, "blockwise_multicut")
    assert hasattr(bp.segmentation, "blockwise_lifted_multicut")
