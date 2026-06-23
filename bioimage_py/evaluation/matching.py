"""Object matching scores (precision / recall / segmentation accuracy / f1) and mean segmentation accuracy.

These densify the contingency table into an overlap matrix (rows = segmentation objects, columns =
groundtruth objects), score every pair under a matching criterion, and find the optimal one-to-one
assignment with the Hungarian algorithm. Implementation follows
https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py.

Note: this is the only metric whose cost is not purely table-sized — the dense ``(n_seg, n_gt)`` matrix
plus the Hungarian solve scale with the number of objects (``O(n_obj^2)`` memory, ``O(n_obj^3)`` time),
not the number of voxels. The contingency-table build is still the parallel, voxel-scale step.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..runner.config import RunnerConfig
from ..sources import SourceLike
from ._common import build_table
from .contingency_table import ContingencyTable

__all__ = ["matching_scores", "mean_segmentation_accuracy_scores", "matching",
           "mean_segmentation_accuracy"]


def _intersection_over_union(overlap: np.ndarray) -> np.ndarray:
    """@private"""
    if overlap.sum() == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / np.maximum(n_pixels_pred + n_pixels_true - overlap, 1e-7)


def _intersection_over_true(overlap: np.ndarray) -> np.ndarray:
    """@private"""
    if overlap.sum() == 0:
        return overlap
    return overlap / np.sum(overlap, axis=1, keepdims=True)


def _intersection_over_pred(overlap: np.ndarray) -> np.ndarray:
    """@private"""
    if overlap.sum() == 0:
        return overlap
    return overlap / np.sum(overlap, axis=0, keepdims=True)


_MATCHING_CRITERIA = {"iou": _intersection_over_union,
                      "iot": _intersection_over_true,
                      "iop": _intersection_over_pred}


def _precision(tp: int, fp: int, fn: int) -> float:
    """@private"""
    return tp / (tp + fp) if tp > 0 else 0.0


def _recall(tp: int, fp: int, fn: int) -> float:
    """@private"""
    return tp / (tp + fn) if tp > 0 else 0.0


def _segmentation_accuracy(tp: int, fp: int, fn: int) -> float:
    """@private"""
    return tp / (tp + fp + fn) if tp > 0 else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    """@private"""
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0.0


def _label_index(labels: np.ndarray, label: int) -> Optional[int]:
    """Return the position of ``label`` in the sorted ``labels`` array, or ``None`` if absent."""
    idx = int(np.searchsorted(labels, label))
    if idx < labels.size and int(labels[idx]) == label:
        return idx
    return None


def _dense_overlap(table: ContingencyTable) -> np.ndarray:
    """Densify the contingency table into a ``(n_seg, n_gt)`` overlap matrix (rows = A, cols = B)."""
    overlap = np.zeros((table.labels_a.size, table.labels_b.size), dtype="float64")
    if table.pairs.shape[0]:
        ai = np.searchsorted(table.labels_a, table.pairs[:, 0])
        bi = np.searchsorted(table.labels_b, table.pairs[:, 1])
        overlap[ai, bi] = table.counts.astype("float64")
    return overlap


def _compute_scores(table: ContingencyTable, criterion: str,
                    ignore_label: Optional[int]) -> Tuple[int, int, int, np.ndarray]:
    """Build the matching-criterion score matrix and drop the ignore label's row/column."""
    if criterion not in _MATCHING_CRITERIA:
        raise ValueError(f"Unknown matching criterion {criterion!r}; expected one of "
                         f"{sorted(_MATCHING_CRITERIA)}.")
    overlap = _dense_overlap(table)
    scores = _MATCHING_CRITERIA[criterion](overlap)
    if scores.size:
        assert 0.0 <= float(np.min(scores)) <= float(np.max(scores)) <= 1.0, \
            f"{np.min(scores)}, {np.max(scores)}"

    if ignore_label is not None:
        ai = _label_index(table.labels_a, ignore_label)
        if ai is not None:
            scores = np.delete(scores, ai, axis=0)
        bi = _label_index(table.labels_b, ignore_label)
        if bi is not None:
            scores = np.delete(scores, bi, axis=1)

    n_pred, n_true = scores.shape
    n_matched = min(n_true, n_pred)
    return n_true, n_matched, n_pred, scores


def _compute_tps(scores: np.ndarray, n_matched: int, threshold: float) -> int:
    """The number of true positives: optimal assignment with the score as a tie-breaker."""
    if n_matched > 0 and np.any(scores >= threshold):
        costs = -(scores >= threshold).astype(float) - scores / (2 * n_matched)
        pred_ind, true_ind = linear_sum_assignment(costs)
        assert n_matched == len(true_ind) == len(pred_ind)
        return int(np.count_nonzero(scores[pred_ind, true_ind] >= threshold))
    return 0


def matching_scores(table: ContingencyTable, *, threshold: float = 0.5, criterion: str = "iou",
                    ignore_label: Optional[int] = 0) -> Dict[str, float]:
    """Compute object-matching scores from a contingency table.

    Args:
        table: A contingency table built as ``contingency_table(segmentation, groundtruth)``.
        threshold: Overlap threshold for a match.
        criterion: Matching criterion, one of ``"iou"``, ``"iot"`` or ``"iop"``.
        ignore_label: Object label removed from both axes before matching (e.g. background). ``None``
            keeps all objects.

    Returns:
        A mapping with keys ``precision``, ``recall``, ``segmentation_accuracy`` and ``f1``.
    """
    n_true, n_matched, n_pred, scores = _compute_scores(table, criterion, ignore_label)
    tp = _compute_tps(scores, n_matched, threshold)
    fp, fn = n_pred - tp, n_true - tp
    return {"precision": _precision(tp, fp, fn), "recall": _recall(tp, fp, fn),
            "segmentation_accuracy": _segmentation_accuracy(tp, fp, fn), "f1": _f1(tp, fp, fn)}


def mean_segmentation_accuracy_scores(
    table: ContingencyTable,
    *,
    thresholds: Optional[Sequence[float]] = None,
    ignore_label: Optional[int] = 0,
    return_accuracies: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute the mean segmentation accuracy (DSB-2018 style) from a contingency table.

    Args:
        table: A contingency table built as ``contingency_table(segmentation, groundtruth)``.
        thresholds: IoU thresholds to average over; defaults to ``np.arange(0.5, 1.0, 0.05)``.
        ignore_label: Object label removed from both axes before matching. ``None`` keeps all objects.
        return_accuracies: Whether to also return the per-threshold accuracies.

    Returns:
        The mean segmentation accuracy, and (only if ``return_accuracies``) the per-threshold accuracies.
    """
    n_true, n_matched, n_pred, scores = _compute_scores(table, "iou", ignore_label)
    thresholds = np.arange(0.5, 1.0, 0.05) if thresholds is None else np.asarray(thresholds, "float64")
    accuracies = np.array([
        _segmentation_accuracy(tp, n_pred - tp, n_true - tp)
        for tp in (_compute_tps(scores, n_matched, float(t)) for t in thresholds)
    ])
    mean_accuracy = float(np.mean(accuracies)) if accuracies.size else 0.0
    if return_accuracies:
        return mean_accuracy, accuracies
    return mean_accuracy


def matching(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    threshold: float = 0.5,
    criterion: str = "iou",
    ignore_label: Optional[int] = 0,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> Dict[str, float]:
    """Compute object-matching scores between two segmentations.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        threshold: Overlap threshold for a match.
        criterion: Matching criterion, one of ``"iou"``, ``"iot"`` or ``"iop"``.
        ignore_label: Object label removed from both axes before matching (e.g. background). ``None``
            keeps all objects.
        num_workers: Number of parallel workers used to build the contingency table.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        A mapping with keys ``precision``, ``recall``, ``segmentation_accuracy`` and ``f1``.
    """
    table = build_table(segmentation, groundtruth, num_workers=num_workers, block_shape=block_shape,
                        job_type=job_type, job_config=job_config, mask=mask)
    return matching_scores(table, threshold=threshold, criterion=criterion, ignore_label=ignore_label)


def mean_segmentation_accuracy(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    thresholds: Optional[Sequence[float]] = None,
    ignore_label: Optional[int] = 0,
    return_accuracies: bool = False,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute the mean segmentation accuracy between two segmentations.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        thresholds: IoU thresholds to average over; defaults to ``np.arange(0.5, 1.0, 0.05)``.
        ignore_label: Object label removed from both axes before matching. ``None`` keeps all objects.
        return_accuracies: Whether to also return the per-threshold accuracies.
        num_workers: Number of parallel workers used to build the contingency table.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        The mean segmentation accuracy, and (only if ``return_accuracies``) the per-threshold accuracies.
    """
    table = build_table(segmentation, groundtruth, num_workers=num_workers, block_shape=block_shape,
                        job_type=job_type, job_config=job_config, mask=mask)
    return mean_segmentation_accuracy_scores(table, thresholds=thresholds, ignore_label=ignore_label,
                                             return_accuracies=return_accuracies)
