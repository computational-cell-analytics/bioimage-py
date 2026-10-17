"""Tests for the whole-graph lifted multicut solvers and lifted-problem construction."""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp
from bioimage_py.segmentation import (get_lifted_multicut_solver, lifted_edges_from_node_labels,
                                      lifted_multicut_gaec, lifted_multicut_fusion_moves,
                                      lifted_multicut_kernighan_lin, lifted_problem_from_node_labels,
                                      multicut_kernighan_lin)


def _line_graph(n=4):
    """A simple path graph 0-1-...-(n-1) as a ``bic.graph.UndirectedGraph``."""
    uv = np.array([[i, i + 1] for i in range(n - 1)], dtype="uint64")
    return bic.graph.UndirectedGraph.from_edges(n, uv), uv


def test_lifted_edge_splits_attractive_blob():
    # Base edges are attractive -> plain multicut merges the whole path into one segment. A strong
    # repulsive lifted edge between the two ends (the canonical lifted use) must split them apart.
    graph, _ = _line_graph(4)
    costs = np.full(graph.number_of_edges, 5.0)
    lifted_uv = np.array([[0, 3]], dtype="uint64")
    lifted_costs = np.array([-50.0])

    plain = np.asarray(multicut_kernighan_lin(graph, costs))
    assert len(np.unique(plain)) == 1  # all merged

    for solver in (lifted_multicut_kernighan_lin, lifted_multicut_gaec, lifted_multicut_fusion_moves):
        node_labels = np.asarray(solver(graph, costs, lifted_uv, lifted_costs))
        assert len(node_labels) == graph.number_of_nodes
        assert node_labels[0] != node_labels[3], solver.__name__


def test_lifted_multicut_matches_plain_without_lifted_edges():
    # With no lifted edges the lifted solver must reproduce the plain multicut partition.
    graph, _ = _line_graph(5)
    costs = np.array([5.0, -5.0, 5.0, -5.0])
    empty_uv = np.zeros((0, 2), dtype="uint64")
    empty_costs = np.zeros(0)

    plain = np.asarray(multicut_kernighan_lin(graph, costs))
    lifted = np.asarray(lifted_multicut_kernighan_lin(graph, costs, empty_uv, empty_costs))
    # Compare as partitions (ids may differ).
    assert np.array_equal(plain[:, None] == plain[None, :], lifted[:, None] == lifted[None, :])


def test_get_lifted_multicut_solver_dispatch():
    graph, _ = _line_graph(4)
    costs = np.full(graph.number_of_edges, 5.0)
    lifted_uv = np.array([[0, 3]], dtype="uint64")
    lifted_costs = np.array([-50.0])

    solver = get_lifted_multicut_solver("kernighan-lin", warmstart=False)
    node_labels = np.asarray(solver(graph, costs, lifted_uv, lifted_costs))
    assert node_labels[0] != node_labels[3]

    with pytest.raises(ValueError):
        get_lifted_multicut_solver("not-a-solver")


def test_lifted_edges_from_node_labels():
    # Path 0-1-2-3-4; labels split ends (1 vs 2), middle ignored (0).
    graph, _ = _line_graph(5)
    node_labels = np.array([1, 0, 0, 0, 2], dtype="uint64")

    diff = np.asarray(lifted_edges_from_node_labels(graph, node_labels, graph_depth=4,
                                                    mode="different", ignore_label=0))
    # Only nodes 0 and 4 carry a (different) label -> exactly the (0, 4) lifted edge.
    assert diff.shape == (1, 2)
    assert set(diff[0].tolist()) == {0, 4}

    # ignore_label=0 excludes the labelled-0 nodes; "same" with no shared non-zero labels -> empty.
    same = np.asarray(lifted_edges_from_node_labels(graph, node_labels, graph_depth=4,
                                                    mode="same", ignore_label=0))
    assert same.shape[0] == 0


def test_lifted_problem_from_node_labels_costs():
    graph, _ = _line_graph(5)
    node_labels = np.array([1, 1, 2, 2, 2], dtype="uint64")
    lifted_uv, lifted_costs = lifted_problem_from_node_labels(
        graph, node_labels, graph_depth=4, same_segment_cost=7.0, different_segment_cost=-3.0,
        mode="all", ignore_label=None,
    )
    lifted_uv = np.asarray(lifted_uv)
    lifted_costs = np.asarray(lifted_costs)
    assert len(lifted_uv) == len(lifted_costs) > 0
    # Every cost is the same/different constant matching its endpoints' labels.
    for (u, v), c in zip(lifted_uv, lifted_costs):
        expected = 7.0 if node_labels[u] == node_labels[v] else -3.0
        assert c == expected


def test_lifted_problem_from_node_labels_empty():
    graph, _ = _line_graph(4)
    node_labels = np.zeros(4, dtype="uint64")  # all ignored
    lifted_uv, lifted_costs = lifted_problem_from_node_labels(
        graph, node_labels, graph_depth=3, same_segment_cost=1.0, different_segment_cost=-1.0,
        mode="different", ignore_label=0,
    )
    assert np.asarray(lifted_uv).shape == (0, 2)
    assert np.asarray(lifted_costs).shape == (0,)


def test_lifted_multicut_exported():
    for name in ("lifted_multicut_kernighan_lin", "lifted_multicut_gaec",
                 "lifted_multicut_fusion_moves", "get_lifted_multicut_solver",
                 "lifted_edges_from_node_labels", "lifted_problem_from_node_labels",
                 "blockwise_lifted_multicut"):
        assert hasattr(bp.segmentation, name)
