"""Lifted multicut solver functionality (backed by :mod:`bioimage_cpp.graph.lifted_multicut`).

The lifted multicut extends the multicut (see :mod:`bioimage_py.segmentation.multicut`) with an
additional set of *lifted* edges. A lifted edge connects two nodes that need not be adjacent in the
base graph and contributes its cost to the objective iff its endpoints land in different partition
elements -- exactly like a regular edge -- but it is not itself part of the graph whose cut is being
optimized. This lets long-range attraction / repulsion (e.g. from an external segmentation or class
probabilities) bias the partition without adding spurious short-range connectivity.

The problem is described by the base graph, the base ``costs`` (one per base edge), and a separate pair
of arrays for the lifted problem: ``lifted_uv_ids`` (``(n_lifted, 2)`` node-id pairs) and
``lifted_costs`` (one per lifted edge). All solvers return a node labeling (a dense 1D array indexed by
node id); positive costs are attractive, negative costs repulsive, as for the plain multicut.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional, Tuple

import bioimage_cpp as bic
import numpy as np

__all__ = [
    "lifted_multicut_kernighan_lin",
    "lifted_multicut_gaec",
    "lifted_multicut_fusion_moves",
    "get_lifted_multicut_solver",
    "lifted_edges_from_node_labels",
    "lifted_problem_from_node_labels",
]


def _to_lifted_objective(graph, costs: np.ndarray, lifted_uv_ids: np.ndarray,
                         lifted_costs: np.ndarray):
    """Build a ``bic.graph.lifted_multicut.LiftedMulticutObjective`` from a graph, costs and lifted arrays."""
    if isinstance(graph, bic.graph.UndirectedGraph):
        graph_ = graph
    else:
        graph_ = bic.graph.UndirectedGraph.from_edges(
            graph.number_of_nodes, np.asarray(graph.uv_ids(), dtype="uint64"),
        )
    return bic.graph.lifted_multicut.LiftedMulticutObjective(
        graph_, costs,
        lifted_uvs=np.asarray(lifted_uv_ids, dtype="uint64"),
        lifted_costs=np.asarray(lifted_costs),
    )


def _get_lifted_solver(internal_solver: str):
    """Return a fresh ``bic.graph.lifted_multicut`` solver instance for the given name."""
    if internal_solver == "kernighan-lin":
        return bic.graph.lifted_multicut.LiftedKernighanLinMulticut()
    elif internal_solver == "greedy-additive":
        return bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut()
    else:
        raise ValueError(f"{internal_solver} cannot be used as internal lifted solver.")


def lifted_multicut_kernighan_lin(
    graph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
    warmstart: bool = True,
) -> np.ndarray:
    """Solve the lifted multicut problem with the Kernighan-Lin solver.

    Args:
        graph: The base graph (or region adjacency graph) of the lifted multicut problem.
        costs: The base edge costs, aligned to the base edge ids.
        lifted_uv_ids: The lifted edges as a ``(n_lifted, 2)`` array of node-id pairs.
        lifted_costs: The lifted edge costs, aligned to ``lifted_uv_ids``.
        warmstart: Whether to warmstart with the lifted greedy-additive solution.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_lifted_objective(graph, costs, lifted_uv_ids, lifted_costs)
    if warmstart:
        solver = bic.graph.lifted_multicut.LiftedChainedSolvers([
            bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
            bic.graph.lifted_multicut.LiftedKernighanLinMulticut(),
        ])
    else:
        solver = bic.graph.lifted_multicut.LiftedKernighanLinMulticut()
    return solver.optimize(objective)


def lifted_multicut_gaec(
    graph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
) -> np.ndarray:
    """Solve the lifted multicut problem with the greedy-additive edge contraction solver.

    Args:
        graph: The base graph (or region adjacency graph) of the lifted multicut problem.
        costs: The base edge costs, aligned to the base edge ids.
        lifted_uv_ids: The lifted edges as a ``(n_lifted, 2)`` array of node-id pairs.
        lifted_costs: The lifted edge costs, aligned to ``lifted_uv_ids``.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_lifted_objective(graph, costs, lifted_uv_ids, lifted_costs)
    return bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)


def lifted_multicut_fusion_moves(
    graph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
    n_threads: int = 1,
    warmstart: bool = True,
    sigma: float = 2.0,
    seed_fraction: float = 0.05,
    num_it: int = 10,
    num_it_stop: int = 4,
) -> np.ndarray:
    """Solve the lifted multicut problem with the fusion-moves solver.

    Fusion moves generate proposal solutions (via a watershed proposal generator) and fuse them with
    the current best solution, escaping the local optima that the greedy / KL solvers can get stuck in.

    Args:
        graph: The base graph (or region adjacency graph) of the lifted multicut problem.
        costs: The base edge costs, aligned to the base edge ids.
        lifted_uv_ids: The lifted edges as a ``(n_lifted, 2)`` array of node-id pairs.
        lifted_costs: The lifted edge costs, aligned to ``lifted_uv_ids``.
        n_threads: The number of threads used to generate and fuse proposals in parallel.
        warmstart: Whether to warmstart with the lifted greedy-additive and Kernighan-Lin solutions.
        sigma: The noise sigma of the watershed proposal generator.
        seed_fraction: The fraction of nodes used as seeds by the watershed proposal generator.
        num_it: The number of fusion-move iterations.
        num_it_stop: Stop after this many iterations without improvement.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_lifted_objective(graph, costs, lifted_uv_ids, lifted_costs)
    proposal_generator = bic.graph.lifted_multicut.WatershedProposalGenerator(
        sigma=sigma, n_seeds_fraction=seed_fraction,
    )
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=proposal_generator,
        sub_solver=bic.graph.lifted_multicut.LiftedKernighanLinMulticut(),
        number_of_iterations=num_it,
        stop_if_no_improvement=num_it_stop,
        number_of_threads=n_threads,
    )
    if warmstart:
        solver = bic.graph.lifted_multicut.LiftedChainedSolvers([
            bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
            bic.graph.lifted_multicut.LiftedKernighanLinMulticut(),
            solver,
        ])
    return solver.optimize(objective)


# Registry of the lifted multicut solvers (name -> function).
_solvers = {
    "kernighan-lin": lifted_multicut_kernighan_lin,
    "greedy-additive": lifted_multicut_gaec,
    "fusion-moves": lifted_multicut_fusion_moves,
}


def get_lifted_multicut_solver(name: str, **kwargs) -> Callable[..., np.ndarray]:
    """Get a lifted multicut solver by name, with solver keyword arguments bound.

    Args:
        name: The name of the solver; one of ``"kernighan-lin"``, ``"greedy-additive"`` or
            ``"fusion-moves"``.
        kwargs: Keyword arguments bound to the solver (e.g. ``warmstart`` or ``n_threads``).

    Returns:
        A callable ``solver(graph, costs, lifted_uv_ids, lifted_costs)`` returning the node labeling.

    Raises:
        ValueError: If ``name`` is not a known lifted multicut solver.
    """
    if name not in _solvers:
        names = ", ".join(_solvers.keys())
        raise ValueError(f"Lifted multicut solver must be one of ({names}), got {name!r}.")
    return functools.partial(_solvers[name], **kwargs)


def lifted_edges_from_node_labels(
    graph,
    node_labels: np.ndarray,
    graph_depth: int,
    mode: str = "different",
    ignore_label: Optional[int] = 0,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Discover lifted edges from per-node labels within a graph neighborhood.

    A lifted edge is inserted between every pair of nodes that are within ``graph_depth`` hops of each
    other in the base graph (excluding the base edges themselves) and whose labels satisfy ``mode``.

    Args:
        graph: The base graph (or region adjacency graph).
        node_labels: A per-node label array (length ``graph.number_of_nodes``).
        graph_depth: The maximum number of graph hops between the endpoints of a lifted edge.
        mode: Which node-label pairs to connect; one of ``"all"``, ``"same"`` or ``"different"``.
        ignore_label: A node label to ignore; nodes with this label are not connected. Pass ``None``
            to disable.
        n_threads: The number of threads to use. Defaults to all available cores.

    Returns:
        The lifted edges as a ``(n_lifted, 2)`` array of node-id pairs.
    """
    if isinstance(graph, bic.graph.UndirectedGraph):
        graph_ = graph
    else:
        graph_ = bic.graph.UndirectedGraph.from_edges(
            graph.number_of_nodes, np.asarray(graph.uv_ids(), dtype="uint64"),
        )
    n_threads_ = 0 if n_threads is None else int(n_threads)
    return bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph_, np.asarray(node_labels), int(graph_depth),
        mode=mode, ignore_label=ignore_label, number_of_threads=n_threads_,
    )


def lifted_problem_from_node_labels(
    graph,
    node_labels: np.ndarray,
    graph_depth: int,
    same_segment_cost: float,
    different_segment_cost: float,
    mode: str = "all",
    ignore_label: Optional[int] = 0,
    n_threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a lifted problem (edges and costs) from per-node labels.

    Lifted edges are discovered with :func:`lifted_edges_from_node_labels`, then each edge is assigned a
    flat cost: ``same_segment_cost`` when its endpoints share a node label (attractive if positive),
    ``different_segment_cost`` when they differ (repulsive if negative). This is the node-label variant
    of the lifted-from-segmentation construction: derive ``node_labels`` externally (e.g. the dominant
    overlap of each superpixel with an external segmentation) and pass them here.

    Args:
        graph: The base graph (or region adjacency graph).
        node_labels: A per-node label array (length ``graph.number_of_nodes``).
        graph_depth: The maximum number of graph hops between the endpoints of a lifted edge.
        same_segment_cost: The cost assigned to lifted edges whose endpoints share a label.
        different_segment_cost: The cost assigned to lifted edges whose endpoints differ in label.
        mode: Which node-label pairs to connect; one of ``"all"``, ``"same"`` or ``"different"``.
        ignore_label: A node label to ignore; nodes with this label are not connected. Pass ``None``
            to disable.
        n_threads: The number of threads to use. Defaults to all available cores.

    Returns:
        A ``(lifted_uv_ids, lifted_costs)`` tuple: the ``(n_lifted, 2)`` lifted edges and their costs.
    """
    node_labels = np.asarray(node_labels)
    lifted_uv_ids = lifted_edges_from_node_labels(
        graph, node_labels, graph_depth, mode=mode, ignore_label=ignore_label, n_threads=n_threads,
    )
    lifted_uv_ids = np.asarray(lifted_uv_ids)
    if len(lifted_uv_ids) == 0:
        return lifted_uv_ids.reshape(0, 2).astype("uint64"), np.zeros(0, dtype="float64")
    same = node_labels[lifted_uv_ids[:, 0]] == node_labels[lifted_uv_ids[:, 1]]
    lifted_costs = np.where(same, float(same_segment_cost), float(different_segment_cost))
    return lifted_uv_ids, lifted_costs.astype("float64")
