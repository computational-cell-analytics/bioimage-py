"""Variation of information (split / merge) and its per-object decomposition.

Both are pure reductions of a :class:`ContingencyTable` built as
``contingency_table(segmentation, groundtruth)`` (axis A = segmentation, axis B = groundtruth). The
split score is the conditional entropy ``H(seg | gt)`` (over-segmentation) and the merge score is
``H(gt | seg)`` (under-segmentation); their sum is the variation of information.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..runner.config import RunnerConfig
from ..sources import SourceLike
from ._common import build_table
from .contingency_table import ContingencyTable

__all__ = ["vi_scores", "object_vi_scores", "variation_of_information", "object_vi"]


def _pair_sizes(table: ContingencyTable) -> Tuple[np.ndarray, np.ndarray]:
    """Return, per pair, the marginal sizes of its A-label and B-label (both float64)."""
    sa = table.sizes_a.astype("float64")[np.searchsorted(table.labels_a, table.pairs[:, 0])]
    sb = table.sizes_b.astype("float64")[np.searchsorted(table.labels_b, table.pairs[:, 1])]
    return sa, sb


def vi_scores(table: ContingencyTable, *, use_log2: bool = True) -> Tuple[float, float]:
    """Compute the split and merge variation of information from a contingency table.

    Args:
        table: A contingency table built as ``contingency_table(segmentation, groundtruth)``.
        use_log2: Whether to use ``log2`` (bits) or natural ``log`` (nats).

    Returns:
        The split variation of information (``H(seg | gt)``) and the merge variation of information
        (``H(gt | seg)``).
    """
    n = table.n_points
    if n == 0:
        return 0.0, 0.0
    log = np.log2 if use_log2 else np.log
    counts = table.counts.astype("float64")
    pa = table.sizes_a.astype("float64") / n
    pb = table.sizes_b.astype("float64") / n
    h_a = -np.sum(pa * log(pa))
    h_b = -np.sum(pb * log(pb))
    sa, sb = _pair_sizes(table)
    mutual = np.sum(counts / n * log(n * counts / (sa * sb)))
    return float(h_a - mutual), float(h_b - mutual)


def object_vi_scores(table: ContingencyTable, *, use_log2: bool = True) -> "pd.DataFrame":
    """Compute the per-groundtruth-object variation of information from a contingency table.

    Based on https://arxiv.org/pdf/1708.02599.pdf (page 16).

    Args:
        table: A contingency table built as ``contingency_table(segmentation, groundtruth)``.
        use_log2: Whether to use ``log2`` (bits) or natural ``log`` (nats).

    Returns:
        A pandas DataFrame with one row per groundtruth object, sorted by label, with columns
        ``label`` (groundtruth id), ``vi_split`` and ``vi_merge``.
    """
    if table.pairs.shape[0] == 0:
        return pd.DataFrame({"label": pd.Series(dtype="uint64"),
                             "vi_split": pd.Series(dtype="float64"),
                             "vi_merge": pd.Series(dtype="float64")})
    log = np.log2 if use_log2 else np.log
    counts = table.counts.astype("float64")
    sa, sb = _pair_sizes(table)

    # Group the pairs by their groundtruth (B) label.
    order = np.argsort(table.pairs[:, 1], kind="stable")
    b_sorted = table.pairs[:, 1][order]
    c, sa_o, sb_o = counts[order], sa[order], sb[order]
    starts = np.flatnonzero(np.concatenate(([True], b_sorted[1:] != b_sorted[:-1])))

    vi_merge = np.add.reduceat(-(c / sb_o) * log(c / sb_o), starts)
    vi_split = np.add.reduceat(-(c / sb_o) * log(c / sa_o), starts)
    return pd.DataFrame({"label": b_sorted[starts].astype("uint64"),
                         "vi_split": vi_split, "vi_merge": vi_merge}).reset_index(drop=True)


def variation_of_information(
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
) -> Tuple[float, float]:
    """Compute the split and merge variation of information between two segmentations.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        ignore_seg: Labels to ignore in the segmentation (their voxels are excluded).
        ignore_gt: Labels to ignore in the groundtruth (their voxels are excluded).
        use_log2: Whether to use ``log2`` (bits) or natural ``log`` (nats).
        num_workers: Number of parallel workers used to build the contingency table.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        The split variation of information and the merge variation of information.
    """
    table = build_table(segmentation, groundtruth, ignore_seg=ignore_seg, ignore_gt=ignore_gt,
                        num_workers=num_workers, block_shape=block_shape, job_type=job_type,
                        job_config=job_config, mask=mask)
    return vi_scores(table, use_log2=use_log2)


def object_vi(
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
) -> "pd.DataFrame":
    """Compute the per-groundtruth-object variation of information between two segmentations.

    Args:
        segmentation: Candidate segmentation to evaluate (a numpy/zarr/n5 array or a `Source`).
        groundtruth: The groundtruth segmentation; same shape as ``segmentation``.
        ignore_seg: Labels to ignore in the segmentation (their voxels are excluded).
        ignore_gt: Labels to ignore in the groundtruth (their voxels are excluded).
        use_log2: Whether to use ``log2`` (bits) or natural ``log`` (nats).
        num_workers: Number of parallel workers used to build the contingency table.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded.

    Returns:
        A pandas DataFrame with one row per groundtruth object (columns ``label``, ``vi_split``,
        ``vi_merge``).
    """
    table = build_table(segmentation, groundtruth, ignore_seg=ignore_seg, ignore_gt=ignore_gt,
                        num_workers=num_workers, block_shape=block_shape, job_type=job_type,
                        job_config=job_config, mask=mask)
    return object_vi_scores(table, use_log2=use_log2)
