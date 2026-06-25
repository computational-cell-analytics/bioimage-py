"""Multicut solver functionality (backed by :mod:`bioimage_cpp.graph.multicut`).

The multicut problem partitions the nodes of an (edge-weighted) graph by deciding, for every edge,
whether it is "cut" (its endpoints land in different partitions) or kept, minimizing the total cost of
cut edges. Positive edge costs attract their endpoints (likely same partition elelemt), negative costs repel them.
"""
from __future__ import annotations

from typing import List, Optional

import bioimage_cpp as bic
import numpy as np

__all__ = [
    "transform_probabilities_to_costs",
    "compute_edge_costs",
    "multicut_decomposition",
    "multicut_gaec",
    "multicut_kernighan_lin",
]


def _weight_edges(costs: np.ndarray, edge_sizes: np.ndarray, weighting_exponent: float) -> np.ndarray:
    """Scale ``costs`` by the (optionally exponentiated) normalized edge sizes, in place."""
    w = edge_sizes / float(edge_sizes.max())
    if weighting_exponent != 1.0:
        w = w ** weighting_exponent
    costs *= w
    return costs


def _weight_populations(costs: np.ndarray, edge_sizes: np.ndarray,
                        edge_populations: List[np.ndarray], weighting_exponent: float) -> np.ndarray:
    """Size-weight each edge population independently (populations must be disjoint)."""
    # Check that the population indices cover each edge at most once.
    covered = np.zeros(len(costs), dtype="uint8")
    for edge_pop in edge_populations:
        covered[edge_pop] += 1
    assert (covered <= 1).all()

    for edge_pop in edge_populations:
        costs[edge_pop] = _weight_edges(costs[edge_pop], edge_sizes[edge_pop], weighting_exponent)
    return costs


def transform_probabilities_to_costs(
    probs: np.ndarray,
    beta: float = 0.5,
    edge_sizes: Optional[np.ndarray] = None,
    edge_populations: Optional[List[np.ndarray]] = None,
    weighting_exponent: float = 1.0,
) -> np.ndarray:
    """Transform merge probabilities to multicut costs via the negative log-likelihood.

    Probabilities near ``1`` map to large positive costs (attractive, likely same segment) and
    probabilities near ``0`` to large negative costs (repulsive). The boundary bias ``beta``
    shifts the decision threshold.

    Args:
        probs: The input edge probabilities, in ``[0, 1]``.
        beta: The boundary bias term; ``> 0.5`` biases towards over-segmentation (more cuts),
            ``< 0.5`` towards under-segmentation. Must be in the exclusive range ``(0, 1)``.
        edge_sizes: The sizes of the edges, used for weighting if given.
        edge_populations: Disjoint edge populations (lists of masks or index arrays) that are
            size-weighted independently, e.g. flat superpixels in a 3d problem.
        weighting_exponent: The exponent applied to the normalized edge sizes when weighting.

    Returns:
        The edge costs.
    """
    p_min = 0.001
    p_max = 1.0 - p_min
    costs = (p_max - p_min) * probs + p_min
    # Probabilities to costs; the second term is the boundary bias.
    costs = np.log((1.0 - costs) / costs) + np.log((1.0 - beta) / beta)
    # Weight the costs with edge sizes, if they are given.
    if edge_sizes is not None:
        assert len(edge_sizes) == len(costs)
        if edge_populations is None:
            costs = _weight_edges(costs, edge_sizes, weighting_exponent)
        else:
            costs = _weight_populations(costs, edge_sizes, edge_populations, weighting_exponent)
    return costs


def compute_edge_costs(
    probs: np.ndarray,
    edge_sizes: Optional[np.ndarray] = None,
    z_edge_mask: Optional[np.ndarray] = None,
    beta: float = 0.5,
    weighting_scheme: Optional[str] = None,
    weighting_exponent: float = 1.0,
) -> np.ndarray:
    """Compute multicut edge costs from probabilities with a pre-defined weighting scheme.

    Args:
        probs: The input edge probabilities, in ``[0, 1]``.
        edge_sizes: The sizes of the edges; required for all weighting schemes except ``None`` /
            ``"none"``.
        z_edge_mask: A boolean mask of inter-slice edges; required for the ``"xyz"`` and ``"z"``
            schemes (for flat superpixels in a 3d problem).
        beta: The boundary bias (see `transform_probabilities_to_costs`).
        weighting_scheme: How to weight the costs by edge size; one of ``None``, ``"none"``,
            ``"all"``, ``"xyz"`` or ``"z"``.
        weighting_exponent: The exponent applied to the normalized edge sizes when weighting.

    Returns:
        The edge costs.

    Raises:
        ValueError: If ``weighting_scheme`` is unknown or a scheme's required inputs are missing.
    """
    schemes = (None, "all", "none", "xyz", "z")
    if weighting_scheme not in schemes:
        schemes_str = ", ".join([str(scheme) for scheme in schemes])
        raise ValueError(f"Weighting scheme must be one of {schemes_str}, got {weighting_scheme}.")

    if weighting_scheme is None or weighting_scheme == "none":
        edge_pop = edge_sizes_ = None

    elif weighting_scheme == "all":
        if edge_sizes is None:
            raise ValueError("Need edge sizes for weighting scheme 'all'.")
        if len(edge_sizes) != len(probs):
            raise ValueError("Invalid edge sizes.")
        edge_sizes_ = edge_sizes
        edge_pop = None

    elif weighting_scheme == "xyz":
        if edge_sizes is None or z_edge_mask is None:
            raise ValueError("Need edge sizes and z edge mask for weighting scheme 'xyz'.")
        if len(edge_sizes) != len(probs) or len(z_edge_mask) != len(probs):
            raise ValueError("Invalid edge sizes or z edge mask.")
        edge_pop = [z_edge_mask, np.logical_not(z_edge_mask)]
        edge_sizes_ = edge_sizes

    elif weighting_scheme == "z":
        if edge_sizes is None or z_edge_mask is None:
            raise ValueError("Need edge sizes and z edge mask for weighting scheme 'z'.")
        if len(edge_sizes) != len(probs) or len(z_edge_mask) != len(probs):
            raise ValueError("Invalid edge sizes or z edge mask.")
        edge_pop = [z_edge_mask, np.logical_not(z_edge_mask)]
        edge_sizes_ = edge_sizes.copy()
        edge_sizes_[edge_pop[1]] = 1.0

    return transform_probabilities_to_costs(
        probs, beta=beta, edge_sizes=edge_sizes_, edge_populations=edge_pop,
        weighting_exponent=weighting_exponent,
    )


def _to_objective(graph, costs: np.ndarray):
    """Build a ``bic.graph.multicut.MulticutObjective`` from a graph (or RAG) and edge costs."""
    if isinstance(graph, bic.graph.UndirectedGraph):
        graph_ = graph
    else:
        graph_ = bic.graph.UndirectedGraph.from_edges(
            graph.number_of_nodes, np.asarray(graph.uv_ids(), dtype="uint64"),
        )
    return bic.graph.multicut.MulticutObjective(graph_, costs)


def _get_solver(internal_solver: str):
    """Return a fresh ``bic.graph.multicut`` solver instance for the given name."""
    if internal_solver == "kernighan-lin":
        return bic.graph.multicut.KernighanLinMulticut()
    elif internal_solver == "greedy-additive":
        return bic.graph.multicut.GreedyAdditiveMulticut()
    elif internal_solver == "greedy-fixation":
        return bic.graph.multicut.GreedyFixationMulticut()
    else:
        raise ValueError(f"{internal_solver} cannot be used as internal solver.")


def multicut_kernighan_lin(graph, costs: np.ndarray, warmstart: bool = True) -> np.ndarray:
    """Solve the multicut problem with the Kernighan-Lin solver.

    Introduced in "An efficient heuristic procedure for partitioning graphs"
    (http://xilinx.asia/_hdl/4/eda.ee.ucla.edu/EE201A-04Spring/kl.pdf).

    Args:
        graph: The graph (or region adjacency graph) of the multicut problem.
        costs: The edge costs of the multicut problem.
        warmstart: Whether to warmstart with the greedy-additive solution.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    if warmstart:
        solver = bic.graph.multicut.ChainedMulticutSolvers([
            bic.graph.multicut.GreedyAdditiveMulticut(),
            bic.graph.multicut.KernighanLinMulticut(),
        ])
    else:
        solver = bic.graph.multicut.KernighanLinMulticut()
    return solver.optimize(objective)


def multicut_gaec(graph, costs: np.ndarray) -> np.ndarray:
    """Solve the multicut problem with the greedy-additive edge contraction solver.

    Introduced in "Fusion moves for correlation clustering"
    (http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf).

    Args:
        graph: The graph (or region adjacency graph) of the multicut problem.
        costs: The edge costs of the multicut problem.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    return bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)


def multicut_decomposition(
    graph,
    costs: np.ndarray,
    n_threads: int = 1,
    internal_solver: str = "kernighan-lin",
) -> np.ndarray:
    """Solve the multicut problem with the decomposition solver.

    The graph is split into its connected components after removing strongly repulsive edges, each
    component is solved independently (in parallel) with ``internal_solver``, and the solutions are
    combined. Introduced in "Break and Conquer: Efficient Correlation Clustering for Image
    Segmentation" (https://link.springer.com/chapter/10.1007/978-3-642-39140-8_9).

    Args:
        graph: The graph (or region adjacency graph) of the multicut problem.
        costs: The edge costs of the multicut problem.
        n_threads: The number of threads used to solve sub-problems in parallel.
        internal_solver: The name of the solver used for the sub-problems; one of
            ``"kernighan-lin"``, ``"greedy-additive"`` or ``"greedy-fixation"``.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    solver = bic.graph.multicut.MulticutDecomposer(
        sub_solver=_get_solver(internal_solver),
        fallthrough_solver=_get_solver(internal_solver),
        number_of_threads=n_threads,
    )
    return solver.optimize(objective)
