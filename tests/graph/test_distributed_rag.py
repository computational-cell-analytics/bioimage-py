"""Parity tests for the distributed RAG / edge-feature orchestration.

The core guarantee mirrors ``tests/test_runner_parity.py`` and the bioimage-cpp reference
(``tests/graph/test_distributed_rag.py``): computing the graph / features block-wise and merging
reproduces the whole-volume result -- exactly for the region graph and the ``size/min/max`` feature
columns, and to floating-point tolerance for ``mean/std`` -- and ``direct == LocalRunner ==
SubprocessRunner``.
"""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp

LABEL_DTYPES = ["uint32", "uint64", "int32", "int64"]

# In-core complex feature columns are (mean, median, std, min, max, p5, p10, p25, p75, p90, p95,
# size); the distributed complex output is the moment subset (mean, std, min, max, size).
_COMPLEX_MOMENT_COLUMNS = [0, 2, 3, 4, 11]
# Distributed complex columns that are exact across backends / block order (min, max, size) and
# those that are only float-order stable (mean, std).
_EXACT_COMPLEX = [2, 3, 4]
_ALLCLOSE_COMPLEX = [0, 1]


def _assert_complex_matches_whole(feats, whole_complex):
    """Assert distributed complex features equal the in-core moment columns."""
    for dist_col, whole_col in zip(range(5), _COMPLEX_MOMENT_COLUMNS):
        if dist_col in _EXACT_COMPLEX:
            np.testing.assert_array_equal(feats[:, dist_col], whole_complex[:, whole_col])
        else:
            np.testing.assert_allclose(feats[:, dist_col], whole_complex[:, whole_col],
                                       rtol=1e-6, atol=1e-9)


def _assert_features_backend_equal(a, b, complex_features):
    """Assert two distributed feature arrays agree (exact on size/min/max, allclose on mean/std)."""
    if complex_features:
        for col in _EXACT_COMPLEX:
            np.testing.assert_array_equal(a[:, col], b[:, col])
        for col in _ALLCLOSE_COMPLEX:
            np.testing.assert_allclose(a[:, col], b[:, col], rtol=1e-6, atol=1e-9)
    else:  # (mean, size)
        np.testing.assert_array_equal(a[:, 1], b[:, 1])
        np.testing.assert_allclose(a[:, 0], b[:, 0], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# Region adjacency graph
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", LABEL_DTYPES)
def test_rag_equality_2d(dtype, rng):
    labels = rng.integers(0, 7, size=(18, 21)).astype(dtype)
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()
    graph = bp.graph.distributed_rag(labels)  # direct
    np.testing.assert_array_equal(graph.uv_ids(), whole)
    assert graph.number_of_nodes == int(labels.max()) + 1


def test_rag_equality_3d(rng):
    labels = rng.integers(0, 5, size=(7, 8, 9)).astype("uint64")
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()
    graph = bp.graph.distributed_rag(labels)
    np.testing.assert_array_equal(graph.uv_ids(), whole)


@pytest.mark.parametrize("block_shape", [(8, 8), (7, 5)])  # a divisor and a non-divisor of the shape
def test_rag_backend_parity(block_shape, zarr_factory, rng):
    labels = rng.integers(0, 6, size=(24, 20)).astype("uint64")
    z = zarr_factory(labels, chunks=(8, 8))
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()

    direct = bp.graph.distributed_rag(labels)  # local, 1 worker, no block shape
    np.testing.assert_array_equal(direct.uv_ids(), whole)
    for nw, job in [(4, "local"), (3, "subprocess")]:
        graph = bp.graph.distributed_rag(z, block_shape=block_shape, num_workers=nw, job_type=job)
        np.testing.assert_array_equal(graph.uv_ids(), whole,
                                      err_msg=f"rag mismatch nw={nw} job={job} bs={block_shape}")


# --------------------------------------------------------------------------------------------------
# Edge-map features
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["uint32", "int64"])
def test_edge_map_features_2d(dtype, rng):
    labels = rng.integers(0, 6, size=(12, 15)).astype(dtype)
    edge_map = rng.standard_normal((12, 15)).astype("float64")
    rag = bic.graph.region_adjacency_graph(labels)

    graph, simple = bp.graph.distributed_edge_features(labels, edge_map, complex_features=False)
    np.testing.assert_array_equal(graph.uv_ids(), rag.uv_ids())
    whole_simple = bic.graph.features.edge_map_features(rag, labels, edge_map)
    np.testing.assert_array_equal(simple[:, 1], whole_simple[:, 1])  # size exact
    np.testing.assert_allclose(simple[:, 0], whole_simple[:, 0], rtol=1e-6, atol=1e-9)  # mean

    _, complex_ = bp.graph.distributed_edge_features(labels, edge_map, complex_features=True)
    _assert_complex_matches_whole(complex_, bic.graph.features.edge_map_features_complex(
        rag, labels, edge_map))


def test_edge_map_features_3d(rng):
    labels = rng.integers(0, 5, size=(6, 7, 8)).astype("uint64")
    edge_map = rng.standard_normal((6, 7, 8)).astype("float64")
    rag = bic.graph.region_adjacency_graph(labels)
    _, complex_ = bp.graph.distributed_edge_features(labels, edge_map, complex_features=True)
    _assert_complex_matches_whole(complex_, bic.graph.features.edge_map_features_complex(
        rag, labels, edge_map))


@pytest.mark.parametrize("complex_features", [False, True])
def test_edge_map_backend_parity(complex_features, zarr_factory, rng):
    labels = rng.integers(0, 6, size=(24, 20)).astype("uint64")
    edge_map = rng.standard_normal((24, 20)).astype("float64")
    zl = zarr_factory(labels, chunks=(8, 8))
    ze = zarr_factory(edge_map, chunks=(8, 8))

    _, direct = bp.graph.distributed_edge_features(labels, edge_map, complex_features=complex_features)
    for nw, job in [(4, "local"), (3, "subprocess")]:
        _, feats = bp.graph.distributed_edge_features(
            zl, ze, complex_features=complex_features, block_shape=(8, 8), num_workers=nw,
            job_type=job)
        _assert_features_backend_equal(direct, feats, complex_features)


# --------------------------------------------------------------------------------------------------
# Affinity features
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("offsets", [[[1, 0], [0, 1]], [[1, 0], [0, 1], [3, 0], [0, 3]]])
def test_affinity_features_2d(offsets, rng):
    labels = rng.integers(0, 6, size=(13, 14)).astype("uint32")
    aff = rng.standard_normal((len(offsets),) + labels.shape).astype("float64")
    rag = bic.graph.region_adjacency_graph(labels)

    graph, complex_ = bp.graph.distributed_edge_features(
        labels, aff, feature_type="affinity", offsets=offsets, complex_features=True)
    # The graph stays the nearest-neighbor RAG even with long-range offsets.
    np.testing.assert_array_equal(graph.uv_ids(), rag.uv_ids())
    _assert_complex_matches_whole(complex_, bic.graph.features.affinity_features_complex(
        rag, labels, aff, offsets))


def test_affinity_features_3d(rng):
    labels = rng.integers(0, 5, size=(5, 6, 7)).astype("uint64")
    offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 2, 0]]
    aff = rng.standard_normal((len(offsets),) + labels.shape).astype("float64")
    rag = bic.graph.region_adjacency_graph(labels)
    graph, complex_ = bp.graph.distributed_edge_features(
        labels, aff, feature_type="affinity", offsets=offsets, complex_features=True)
    np.testing.assert_array_equal(graph.uv_ids(), rag.uv_ids())
    _assert_complex_matches_whole(complex_, bic.graph.features.affinity_features_complex(
        rag, labels, aff, offsets))


def test_affinity_backend_parity(zarr_factory, rng):
    offsets = [[1, 0], [0, 1], [3, 0], [0, 3]]  # includes long-range offsets (halo > block-face)
    labels = rng.integers(0, 6, size=(24, 20)).astype("uint64")
    aff = rng.standard_normal((len(offsets),) + labels.shape).astype("float64")
    zl = zarr_factory(labels, chunks=(8, 8))
    za = zarr_factory(aff, chunks=(len(offsets), 8, 8))
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()

    _, direct = bp.graph.distributed_edge_features(
        labels, aff, feature_type="affinity", offsets=offsets, complex_features=True)
    for nw, job in [(4, "local"), (3, "subprocess")]:
        graph, feats = bp.graph.distributed_edge_features(
            zl, za, feature_type="affinity", offsets=offsets, complex_features=True,
            block_shape=(8, 8), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(graph.uv_ids(), whole, err_msg=f"nw={nw} job={job}")
        _assert_features_backend_equal(direct, feats, True)


# --------------------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------------------

def test_single_label_empty_graph():
    labels = np.zeros((10, 12), dtype="uint32")
    graph = bp.graph.distributed_rag(labels)
    assert graph.number_of_nodes == 1
    assert graph.number_of_edges == 0

    edge_map = np.ones((10, 12), dtype="float64")
    graph2, simple = bp.graph.distributed_edge_features(labels, edge_map)
    assert graph2.number_of_edges == 0
    assert simple.shape == (0, 2)
    _, complex_ = bp.graph.distributed_edge_features(labels, edge_map, complex_features=True)
    assert complex_.shape == (0, 5)


def test_isolated_segments_kept_as_nodes():
    # Only ids 0 and 5 are present (adjacent), so nodes 1..4 are isolated; n_nodes must be max+1=6.
    labels = np.zeros((6, 6), dtype="uint32")
    labels[:, 3:] = 5
    graph = bp.graph.distributed_rag(labels)
    assert graph.number_of_nodes == 6
    np.testing.assert_array_equal(graph.uv_ids(), np.array([[0, 5]], dtype="uint64"))


def test_unsupported_dtype_is_cast(rng):
    labels = rng.integers(0, 6, size=(12, 12)).astype("uint16")
    whole = bic.graph.region_adjacency_graph(labels.astype("uint32")).uv_ids()
    graph = bp.graph.distributed_rag(labels)
    np.testing.assert_array_equal(graph.uv_ids(), whole)


def test_n_nodes_explicit_and_too_small(rng):
    labels = rng.integers(0, 6, size=(12, 12)).astype("uint32")
    graph = bp.graph.distributed_rag(labels, n_nodes=100)
    assert graph.number_of_nodes == 100
    with pytest.raises(ValueError, match="n_nodes"):
        bp.graph.distributed_rag(labels, n_nodes=3)


def test_invalid_arguments_raise(rng):
    labels = rng.integers(0, 4, size=(8, 8)).astype("uint32")
    data = rng.standard_normal((8, 8)).astype("float64")
    with pytest.raises(ValueError, match="feature_type"):
        bp.graph.distributed_edge_features(labels, data, feature_type="bogus")
    with pytest.raises(ValueError, match="offsets is required"):
        bp.graph.distributed_edge_features(labels, data, feature_type="affinity")
    with pytest.raises(ValueError, match="must match the labels shape"):
        bp.graph.distributed_edge_features(labels, rng.standard_normal((8, 7)).astype("float64"))
    with pytest.raises(ValueError, match="integer"):
        bp.graph.distributed_rag(data)
