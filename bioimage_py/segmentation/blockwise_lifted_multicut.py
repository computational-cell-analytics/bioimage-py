"""Block-wise (hierarchical) lifted multicut with a distributed per-block solve.

:func:`blockwise_lifted_multicut` is the lifted counterpart of
:func:`bioimage_py.segmentation.blockwise_multicut`: it solves a global lifted multicut by hierarchical
domain decomposition, distributing the per-block sub-problem solves through the runner and reducing them
in the orchestrating process. The lifted edges/costs are carried alongside the base problem -- each
block additionally solves its interior lifted edges (falling back to the plain multicut solve for a
block with none), and the reduce step contracts the lifted edges through the same node labeling as the
base edges. It returns a node labeling; project it back to pixels with
:func:`bioimage_py.segmentation.relabel`.
"""
from __future__ import annotations

import shutil
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import SourceLike, as_source
from ._blockwise_common import (as_undirected_graph, lifted_solver_by_name, reduce_problem,
                                solve_subproblems_distributed)

__all__ = ["blockwise_lifted_multicut"]


def blockwise_lifted_multicut(
    graph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
    segmentation: SourceLike,
    *,
    block_shape: Tuple[int, ...],
    n_levels: int = 1,
    internal_solver: str = "kernighan-lin",
    halo: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
) -> np.ndarray:
    """Solve the lifted multicut problem block-wise (hierarchical domain decomposition).

    The per-block sub-problem solves are the distributed map step (via the runner); the union-find
    merge, edge (and lifted-edge) contraction and final global lifted solve run in the orchestrating
    process.

    Args:
        graph: The base graph (or region adjacency graph). Its node ids must equal the segment ids of
            ``segmentation`` (node ``i`` is segment ``i``).
        costs: The base edge costs (aligned to the base edge ids).
        lifted_uv_ids: The lifted edges as a ``(n_lifted, 2)`` array of node-id pairs.
        lifted_costs: The lifted edge costs, aligned to ``lifted_uv_ids``.
        segmentation: The segmentation whose ids are the graph nodes (a numpy/zarr/n5 array or a
            `Source`). It defines the blocking domain; for distributed backends it must be file-backed
            (reopenable). It is only read.
        block_shape: The block shape of the finest (level 0) decomposition. Required. The block size is
            doubled each level and clamped to the volume shape.
        n_levels: The number of hierarchy levels (>= 1).
        internal_solver: The solver used for the per-block sub-problems and the final global solve; one
            of ``"kernighan-lin"`` or ``"greedy-additive"``.
        halo: Optional per-axis halo added to each block when reading the segmentation. ``None`` reads
            the block region only.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).

    Returns:
        The node label solution (a dense 1D array indexed by node id == segment id).

    Raises:
        ValueError: If ``n_levels < 1`` or ``block_shape`` does not match the segmentation ndim.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}.")

    graph = as_undirected_graph(graph)
    costs = np.asarray(costs)
    lifted_uv_ids = np.asarray(lifted_uv_ids, dtype="uint64").reshape(-1, 2)
    lifted_costs = np.asarray(lifted_costs, dtype="float64")
    seg_src = as_source(segmentation)
    shape = tuple(int(s) for s in seg_src.shape)
    block_shape = tuple(int(b) for b in block_shape)
    if len(block_shape) != len(shape):
        raise ValueError(
            f"block_shape {block_shape} must match the segmentation ndim {len(shape)}."
        )

    runner = get_runner(job_type, job_config)
    tmp_dir: Optional[str] = None
    if job_type != "local":
        tmp_dir = tempfile.mkdtemp(prefix="bioimage_py_blockwise_lmc_", dir=runner.config.tmp_root)

    graph_, costs_ = graph, costs
    lifted_uv_, lifted_costs_ = lifted_uv_ids, lifted_costs
    labels: Optional[np.ndarray] = None
    for level in range(n_levels):
        level_block_shape = tuple(min(b * (2 ** level), s) for b, s in zip(block_shape, shape))
        merge_mask = solve_subproblems_distributed(
            runner, graph_, costs_, seg_src, labels,
            block_shape=level_block_shape, halo=halo, internal_solver=internal_solver,
            num_workers=num_workers, job_type=job_type, tmp_dir=tmp_dir,
            name=f"blockwise-lifted-multicut-l{level}",
            lifted_uv_ids=lifted_uv_, lifted_costs=lifted_costs_,
        )
        graph_, costs_, new_labels, lifted_uv_, lifted_costs_ = reduce_problem(
            graph_, costs_, merge_mask, lifted_uv_ids=lifted_uv_, lifted_costs=lifted_costs_,
        )
        labels = new_labels if labels is None else new_labels[labels]

    final_labels = np.asarray(
        lifted_solver_by_name(internal_solver)(graph_, costs_, lifted_uv_, lifted_costs_)
    )
    node_labels = final_labels[labels]

    if tmp_dir is not None:  # preserved on failure (an exception above skips this) for debugging.
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return node_labels
