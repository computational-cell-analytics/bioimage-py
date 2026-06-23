"""Dice scores: foreground (binary) dice and the symmetric best dice for instance segmentations.

``dice_score`` is a foreground overlap of two (optionally thresholded) images — a small additive
reduction of three scalars (``|a & b|``, ``|a|``, ``|b|``), so it has its own block function rather
than going through the contingency table. ``symmetric_best_dice_score`` is per-object and is a pure
reduction of a :class:`ContingencyTable`.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, check_direct, full_roi, to_roi
from ._common import build_table
from .contingency_table import ContingencyTable

__all__ = ["dice_score", "best_dice_scores", "symmetric_best_dice_score"]

_EPS = 1e-7


# --- foreground dice -------------------------------------------------------------------

def _binarize(data: np.ndarray, threshold: Optional[float]) -> np.ndarray:
    """Threshold ``data`` to a boolean foreground, or pass it through if ``threshold`` is ``None``."""
    return data if threshold is None else (data > threshold)


def _dice_sums(a: np.ndarray, b: np.ndarray, threshold_seg: Optional[float],
               threshold_gt: Optional[float]) -> Tuple[float, float, float]:
    """Return ``(sum(a*b), sum(a), sum(b))`` after optional thresholding (the dice sufficient stats)."""
    a = _binarize(a, threshold_seg).astype("float64")
    b = _binarize(b, threshold_gt).astype("float64")
    return float(np.sum(a * b)), float(np.sum(a)), float(np.sum(b))


def _make_dice_compute(threshold_seg: Optional[float],
                       threshold_gt: Optional[float]) -> Callable:
    """Build the per-block dice-sums function (captures only picklable thresholds)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[Tuple[float, float, float]]:
        roi = to_roi(block)
        a, b = inputs[0][roi], inputs[1][roi]
        if mask is not None:
            m = mask[roi].astype(bool)
            if not m.any():
                return None
            a, b = a[m], b[m]
        return _dice_sums(a, b, threshold_seg, threshold_gt)

    return _compute


def dice_score(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    threshold_seg: Optional[float] = 0,
    threshold_gt: Optional[float] = 0,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> float:
    """Compute the dice score between a (binarized) segmentation and groundtruth.

    To compare probability maps (values in ``[0, 1]``) pass ``threshold_seg=None`` /
    ``threshold_gt=None`` (a soft dice on the raw values); otherwise each input is binarized at its
    threshold.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth; same shape as ``segmentation``.
        threshold_seg: Threshold applied to the segmentation, or ``None`` to use it as-is.
        threshold_gt: Threshold applied to the groundtruth, or ``None`` to use it as-is.
        num_workers: Number of parallel workers.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        The dice score.
    """
    if check_direct(job_type, num_workers, block_shape, mask, None):
        src_a, src_b = as_source(segmentation), as_source(groundtruth)
        intersection, sum_a, sum_b = _dice_sums(src_a[full_roi(src_a.ndim)], src_b[full_roi(src_b.ndim)],
                                                threshold_seg, threshold_gt)
    else:
        runner = get_runner(job_type, job_config)
        results = runner.run(_make_dice_compute(threshold_seg, threshold_gt),
                             [segmentation, groundtruth], num_workers=num_workers,
                             block_shape=block_shape, mask=mask, has_return_val=True,
                             name="dice_score")
        results = [r for r in results if r is not None]
        if not results:
            return 0.0
        intersection, sum_a, sum_b = np.array(results, dtype="float64").sum(axis=0)
    return float(2.0 * intersection) / float(sum_a + sum_b + _EPS)


# --- symmetric best dice ---------------------------------------------------------------

def _best_dice_direction(labels: np.ndarray, idx: np.ndarray, dice_pair: np.ndarray,
                         valid: np.ndarray, ignore_label: Optional[int]) -> float:
    """Mean over the objects of ``labels`` of their best per-pair dice (0 if no valid overlap)."""
    best = np.zeros(labels.size, dtype="float64")
    if valid.any():
        np.maximum.at(best, idx[valid], dice_pair[valid])
    keep = np.ones(labels.size, dtype=bool) if ignore_label is None else (labels != ignore_label)
    if not keep.any():
        return 0.0
    return float(np.mean(best[keep]))


def best_dice_scores(table: ContingencyTable, *, ignore_label: Optional[int] = 0) -> float:
    """Compute the symmetric best dice score from a contingency table.

    For each object in one segmentation the best dice with any object in the other is taken; this is
    averaged per segmentation and the smaller of the two averages is returned.

    Args:
        table: A contingency table built as ``contingency_table(segmentation, groundtruth)``.
        ignore_label: Label excluded as an object on both sides (e.g. background). ``None`` keeps all.

    Returns:
        The symmetric best dice score.
    """
    if table.pairs.shape[0] == 0:
        return 0.0
    idx_a = np.searchsorted(table.labels_a, table.pairs[:, 0])
    idx_b = np.searchsorted(table.labels_b, table.pairs[:, 1])
    size_a = table.sizes_a.astype("float64")[idx_a]
    size_b = table.sizes_b.astype("float64")[idx_b]
    dice_pair = 2.0 * table.counts.astype("float64") / (size_a + size_b)

    if ignore_label is None:
        valid = np.ones(table.pairs.shape[0], dtype=bool)
    else:
        valid = (table.pairs[:, 0] != ignore_label) & (table.pairs[:, 1] != ignore_label)

    dir_seg = _best_dice_direction(table.labels_a, idx_a, dice_pair, valid, ignore_label)
    dir_gt = _best_dice_direction(table.labels_b, idx_b, dice_pair, valid, ignore_label)
    return min(dir_seg, dir_gt)


def symmetric_best_dice_score(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    ignore_label: Optional[int] = 0,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> float:
    """Compute the symmetric best dice score between two instance segmentations.

    This metric is used in the CVPPP instance segmentation challenge.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        ignore_label: Label excluded as an object on both sides (e.g. background). ``None`` keeps all.
        num_workers: Number of parallel workers used to build the contingency table.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        The symmetric best dice score.
    """
    table = build_table(segmentation, groundtruth, num_workers=num_workers, block_shape=block_shape,
                        job_type=job_type, job_config=job_config, mask=mask)
    return best_dice_scores(table, ignore_label=ignore_label)
