"""Rand-index derived scores: the adapted rand error and the rand index.

Both are pure reductions of a :class:`ContingencyTable`; the result is symmetric in the two
segmentations, so the table orientation does not matter.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..runner.config import RunnerConfig
from ..sources import SourceLike
from ._common import build_table
from .contingency_table import ContingencyTable

__all__ = ["rand_scores", "rand_index"]


def rand_scores(table: ContingencyTable) -> Tuple[float, float]:
    """Compute the adapted rand error and the rand index from a contingency table.

    Args:
        table: A contingency table for the two segmentations.

    Returns:
        The adapted rand error and the rand index.
    """
    n = table.n_points
    if n == 0:
        return 0.0, 1.0
    sum_a2 = float(np.sum(table.sizes_a.astype("float64") ** 2))
    sum_b2 = float(np.sum(table.sizes_b.astype("float64") ** 2))
    sum_ab2 = float(np.sum(table.counts.astype("float64") ** 2))

    precision = sum_ab2 / sum_a2
    recall = sum_ab2 / sum_b2
    adapted_rand_error = 1.0 - (2 * precision * recall) / (precision + recall)
    rand_index = 1.0 - (sum_a2 + sum_b2 - 2 * sum_ab2) / (n * n)
    return float(adapted_rand_error), float(rand_index)


def rand_index(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    ignore_seg: Optional[Sequence[int]] = None,
    ignore_gt: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> Tuple[float, float]:
    """Compute the adapted rand error and the rand index between two segmentations.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        ignore_seg: Labels to ignore in the segmentation (their voxels are excluded).
        ignore_gt: Labels to ignore in the groundtruth (their voxels are excluded).
        num_workers: Number of parallel workers used to build the contingency table.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        The adapted rand error and the rand index.
    """
    table = build_table(segmentation, groundtruth, ignore_seg=ignore_seg, ignore_gt=ignore_gt,
                        num_workers=num_workers, block_shape=block_shape, job_type=job_type,
                        job_config=job_config, mask=mask)
    return rand_scores(table)
