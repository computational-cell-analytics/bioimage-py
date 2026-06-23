"""Object matching scores (precision / recall / segmentation accuracy / f1) by centroid distance.

This complements the overlap-based :func:`bioimage_py.evaluation.matching`: instead of densifying the
contingency table and scoring object pairs by IoU, it matches objects whose centroids lie within a
distance threshold of one another. This is the natural criterion when objects are point-like (spots,
nuclei, detections) or when voxel overlap is too strict. Like ``dice_score`` it is *not* built on the
contingency table — it needs object centroids, not voxel overlaps.

There are two layers. ``coordinate_matching`` is the low-level form: it matches two point sets directly.
``centroid_matching`` is the high-level form: it derives each segmentation's per-object centroids (the
center of mass, via :func:`bioimage_py.morphology.morphology`, which runs block-wise / distributed) and
then defers to ``coordinate_matching``. Both find the optimal one-to-one assignment with the Hungarian
algorithm (``scipy.optimize.linear_sum_assignment``) and reuse the precision / recall / f1 /
segmentation-accuracy formulas of the overlap-based matching.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..morphology import morphology
from ..runner.config import RunnerConfig
from ..sources import SourceLike
from .matching import _f1, _precision, _recall, _segmentation_accuracy

__all__ = ["coordinate_matching", "centroid_matching"]

_EPS = 1e-12


def _count_matches(distances: np.ndarray, distance_threshold: float) -> int:
    """The number of true positives: optimal one-to-one assignment of points within the threshold."""
    n_matched = min(distances.shape)
    valid = distances <= distance_threshold
    if n_matched == 0 or not valid.any():
        return 0
    # Prioritize within-threshold pairs; the distance is a tie-breaker normalized so each term is
    # <= 1 / (2 * n_matched) and can never outweigh the integer validity term (cf. matching._compute_tps).
    costs = -valid.astype("float64") + distances / (2 * n_matched * (float(distances.max()) + _EPS))
    row_ind, col_ind = linear_sum_assignment(costs)
    return int(np.count_nonzero(valid[row_ind, col_ind]))


def _as_coordinates(coordinates: object, name: str) -> np.ndarray:
    """Coerce ``coordinates`` to a ``(N, ndim)`` float64 array (an empty set becomes ``(0, 0)``)."""
    arr = np.asarray(coordinates, dtype="float64")
    if arr.size == 0:
        return arr.reshape(0, 0)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (N, ndim), got shape {arr.shape}.")
    return arr


def _centroids(
    segmentation: SourceLike,
    ignore_label: Optional[int],
    num_workers: int,
    block_shape: Optional[Tuple[int, ...]],
    job_type: str,
    job_config: Optional[RunnerConfig],
    mask: Optional[SourceLike],
) -> np.ndarray:
    """Per-object centers of mass of ``segmentation`` as a ``(N, ndim)`` voxel-coordinate array."""
    table = morphology(segmentation, num_workers=num_workers, block_shape=block_shape,
                       job_type=job_type, job_config=job_config, mask=mask)
    if ignore_label is not None:
        table = table[table["label"] != ignore_label]
    com_columns = [column for column in table.columns if column.startswith("com_")]
    return table[com_columns].to_numpy(dtype="float64")


def coordinate_matching(
    coordinates_a: object,
    coordinates_b: object,
    *,
    distance_threshold: float,
    resolution: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Match two point sets by centroid distance and compute the matching scores.

    Each point in ``coordinates_a`` is matched to at most one point in ``coordinates_b`` (and vice
    versa) via the optimal one-to-one assignment, counting a pair as a match only if their Euclidean
    distance does not exceed ``distance_threshold``. ``coordinates_a`` plays the role of the prediction
    and ``coordinates_b`` the reference, so precision is computed over ``coordinates_a`` and recall over
    ``coordinates_b`` (matching the orientation of :func:`matching`).

    Args:
        coordinates_a: The candidate points to evaluate, an array-like of shape ``(N, ndim)``.
        coordinates_b: The reference points, an array-like of shape ``(M, ndim)`` with the same
            ``ndim`` as ``coordinates_a``.
        distance_threshold: Maximum centroid distance (in the units of the coordinates, or of
            ``resolution`` if given) for two points to count as a match.
        resolution: Optional per-axis spacing; when given, coordinates are scaled by it before
            distances are computed, so ``distance_threshold`` is interpreted in physical units.

    Returns:
        A mapping with keys ``precision``, ``recall``, ``segmentation_accuracy`` and ``f1``.
    """
    coords_a = _as_coordinates(coordinates_a, "coordinates_a")
    coords_b = _as_coordinates(coordinates_b, "coordinates_b")
    n_a, n_b = coords_a.shape[0], coords_b.shape[0]

    if n_a and n_b:
        if coords_a.shape[1] != coords_b.shape[1]:
            raise ValueError(f"coordinates_a and coordinates_b must have the same ndim, got "
                             f"{coords_a.shape[1]} and {coords_b.shape[1]}.")
        if resolution is not None:
            spacing = np.asarray(resolution, dtype="float64")
            if spacing.shape != (coords_a.shape[1],):
                raise ValueError(f"resolution must have one entry per axis ({coords_a.shape[1]}), got "
                                 f"shape {spacing.shape}.")
            coords_a, coords_b = coords_a * spacing, coords_b * spacing
        tp = _count_matches(cdist(coords_a, coords_b), distance_threshold)
    else:
        tp = 0

    fp, fn = n_a - tp, n_b - tp
    return {"precision": _precision(tp, fp, fn), "recall": _recall(tp, fp, fn),
            "segmentation_accuracy": _segmentation_accuracy(tp, fp, fn), "f1": _f1(tp, fp, fn)}


def centroid_matching(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    distance_threshold: float,
    resolution: Optional[Sequence[float]] = None,
    ignore_label: Optional[int] = 0,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> Dict[str, float]:
    """Compute object-matching scores between two segmentations by centroid distance.

    Derives each object's center of mass with :func:`bioimage_py.morphology.morphology` (the centroid
    extraction runs block-wise / distributed via the runner arguments) and then matches the two centroid
    sets with :func:`coordinate_matching`.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`); must be
            integer-typed.
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        distance_threshold: Maximum centroid distance (in voxels, or physical units if ``resolution``
            is given) for two objects to count as a match.
        resolution: Optional per-axis voxel spacing; when given, centroids are scaled by it before
            distances are computed, so ``distance_threshold`` is interpreted in physical units.
        ignore_label: Object label removed from both segmentations before matching (e.g. background).
            ``None`` keeps all objects. Note ``morphology`` already excludes label ``0``.
        num_workers: Number of parallel workers used to extract the centroids.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        A mapping with keys ``precision``, ``recall``, ``segmentation_accuracy`` and ``f1``.
    """
    coords_a = _centroids(segmentation, ignore_label, num_workers, block_shape, job_type, job_config,
                          mask)
    coords_b = _centroids(groundtruth, ignore_label, num_workers, block_shape, job_type, job_config,
                          mask)
    return coordinate_matching(coords_a, coords_b, distance_threshold=distance_threshold,
                               resolution=resolution)
