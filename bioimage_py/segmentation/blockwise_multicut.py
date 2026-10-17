"""Block-wise (hierarchical) multicut with a distributed per-block solve.

:func:`blockwise_multicut` solves a global multicut by hierarchical domain decomposition: at each level
every block solves its induced sub-problem independently (the *map* step, distributed across the
``local`` / ``subprocess`` / ``slurm`` backends via the runner), and the per-block cut decisions are
merged into a coarser graph in the orchestrating process (the *reduce* step). The block size doubles
each level, so block-boundary edges that could not be decided locally become interior edges of the next
level; a final global solve on the fully contracted graph completes the partition.

This scales the whole-graph solvers in :mod:`bioimage_py.segmentation.multicut` to problems whose
per-block solves are too many for a single process, at the cost of an approximation (the decomposition
fixes block-boundary edges as cut at each level). It returns a node labeling; project it back to pixels
with :func:`bioimage_py.segmentation.relabel` (``relabel(segmentation, node_labels, output=...)``), the
canonical distributed node-label writer.
"""
from __future__ import annotations

import shutil
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import SourceLike, as_source
from ._blockwise_common import (as_undirected_graph, multicut_solver_by_name, reduce_problem,
                                solve_subproblems_distributed)

__all__ = ["blockwise_multicut"]


def blockwise_multicut(
    graph,
    costs: np.ndarray,
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
    """Solve the multicut problem block-wise (hierarchical domain decomposition).

    The per-block sub-problem solves are the distributed map step (via the runner); the union-find
    merge, edge contraction and final global solve run in the orchestrating process.

    Args:
        graph: The graph (or region adjacency graph) of the multicut problem. Its node ids must equal
            the segment ids of ``segmentation`` (node ``i`` is segment ``i``) -- the contract of
            :func:`bioimage_py.graph.distributed_rag`.
        costs: The edge costs of the multicut problem (aligned to the graph edge ids; positive costs
            attractive, negative repulsive).
        segmentation: The segmentation whose ids are the graph nodes (a numpy/zarr/n5 array or a
            `Source`). It defines the blocking domain; for distributed backends it must be file-backed
            (reopenable). It is only read.
        block_shape: The block shape of the finest (level 0) decomposition. Required. The block size is
            doubled each level and clamped to the volume shape.
        n_levels: The number of hierarchy levels (>= 1). More levels contract more block-boundary edges
            before the final global solve.
        internal_solver: The solver used for the per-block sub-problems and the final global solve; one
            of ``"kernighan-lin"``, ``"greedy-additive"`` or ``"decomposition"``.
        halo: Optional per-axis halo added to each block when reading the segmentation (block node
            membership then includes the neighborhood). ``None`` reads the block region only.
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
        tmp_dir = tempfile.mkdtemp(prefix="bioimage_py_blockwise_mc_", dir=runner.config.tmp_root)

    graph_, costs_ = graph, costs
    labels: Optional[np.ndarray] = None
    for level in range(n_levels):
        # The block size doubles each level and is clamped to the volume (a block never exceeds it).
        level_block_shape = tuple(min(b * (2 ** level), s) for b, s in zip(block_shape, shape))
        merge_mask = solve_subproblems_distributed(
            runner, graph_, costs_, seg_src, labels,
            block_shape=level_block_shape, halo=halo, internal_solver=internal_solver,
            num_workers=num_workers, job_type=job_type, tmp_dir=tmp_dir,
            name=f"blockwise-multicut-l{level}",
        )
        graph_, costs_, new_labels = reduce_problem(graph_, costs_, merge_mask)
        labels = new_labels if labels is None else new_labels[labels]

    # Final global solve on the fully contracted graph, then compose back to the original node ids.
    final_labels = np.asarray(multicut_solver_by_name(internal_solver)(graph_, costs_))
    node_labels = final_labels[labels]

    if tmp_dir is not None:  # preserved on failure (an exception above skips this) for debugging.
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return node_labels
