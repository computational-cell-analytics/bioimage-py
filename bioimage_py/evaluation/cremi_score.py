"""CREMI score: the geometric mean of the variation of information and the adapted rand error.

This was the evaluation metric of the CREMI challenge. It reuses a single contingency table to compute
both the variation of information and the adapted rand error.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..runner.config import RunnerConfig
from ..sources import SourceLike
from ._common import build_table
from .contingency_table import ContingencyTable
from .rand_index import rand_scores
from .variation_of_information import vi_scores

__all__ = ["cremi_scores", "cremi_score"]


def cremi_scores(table: ContingencyTable, *, use_log2: bool = True) -> Tuple[float, float, float, float]:
    """Compute the CREMI score and its components from a single contingency table.

    Args:
        table: A contingency table built as ``contingency_table(segmentation, groundtruth)``.
        use_log2: Whether to use ``log2`` (bits) or natural ``log`` (nats) for the VI part.

    Returns:
        The split variation of information, the merge variation of information, the adapted rand error,
        and the CREMI score (``sqrt(adapted_rand_error * (vi_split + vi_merge))``).
    """
    vi_split, vi_merge = vi_scores(table, use_log2=use_log2)
    adapted_rand_error, _ = rand_scores(table)
    cremi = float(np.sqrt(adapted_rand_error * (vi_split + vi_merge)))
    return vi_split, vi_merge, adapted_rand_error, cremi


def cremi_score(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    ignore_seg: Optional[Sequence[int]] = None,
    ignore_gt: Optional[Sequence[int]] = None,
    use_log2: bool = True,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> Tuple[float, float, float, float]:
    """Compute the CREMI score between two segmentations.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        ignore_seg: Labels to ignore in the segmentation (their voxels are excluded).
        ignore_gt: Labels to ignore in the groundtruth (their voxels are excluded).
        use_log2: Whether to use ``log2`` (bits) or natural ``log`` (nats) for the VI part.
        num_workers: Number of parallel workers used to build the contingency table.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        The split variation of information, the merge variation of information, the adapted rand error,
        and the CREMI score.
    """
    table = build_table(segmentation, groundtruth, ignore_seg=ignore_seg, ignore_gt=ignore_gt,
                        num_workers=num_workers, block_shape=block_shape, job_type=job_type,
                        job_config=job_config, mask=mask)
    return cremi_scores(table, use_log2=use_log2)
