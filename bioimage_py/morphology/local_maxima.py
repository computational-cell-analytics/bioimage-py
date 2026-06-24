"""Block-wise local maxima detection (halo-based) via ``skimage.feature.peak_local_max``.

A reduction operation that returns the coordinates of detected maxima rather than an array: each
(halo-padded) block detects peaks, keeps only those whose coordinate falls in its halo-free inner
block (so a peak is attributed to exactly one block), and shifts them to global coordinates; the main
process concatenates them. ``bioimage_cpp`` has no peak detector, so this falls back to scikit-image.
The halo is derived from ``min_distance`` so the non-maximum suppression near block boundaries matches
a whole-array run.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from skimage.feature import peak_local_max

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, ComputeFn, check_direct, check_rerun_args, full_roi, to_roi

__all__ = ["find_local_maxima"]


def _make_local_maxima(min_distance: int, threshold_abs: Optional[float],
                       threshold_rel: Optional[float]) -> ComputeFn:
    """Build the per-block peak-detection function (captures only picklable values)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        coords = peak_local_max(inputs[0][to_roi(block.outer_block)], min_distance=min_distance,
                                threshold_abs=threshold_abs, threshold_rel=threshold_rel)
        if coords.shape[0] == 0:
            return None
        # Keep only peaks inside the halo-free inner block, then shift to global coordinates.
        lo = np.array([int(b) for b in block.inner_block_local.begin])[None, :]
        hi = np.array([int(e) for e in block.inner_block_local.end])[None, :]
        keep = np.logical_and(coords >= lo, coords < hi).all(axis=1)
        coords = coords[keep]
        if coords.shape[0] == 0:
            return None
        return coords + np.array([int(b) for b in block.outer_block.begin])[None, :]

    return _compute


def find_local_maxima(
    input: SourceLike,
    *,
    min_distance: int = 1,
    threshold_abs: Optional[float] = None,
    threshold_rel: Optional[float] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> np.ndarray:
    """Find local maxima of the data, block-wise (based on ``skimage.feature.peak_local_max``).

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        min_distance: The minimum allowed distance between maxima (the non-maximum suppression
            radius); also drives the block halo.
        threshold_abs: Minimum intensity of a maximum. Defaults to the data minimum.
        threshold_rel: Minimum intensity of a maximum, as a fraction of the data maximum.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_ids: Restrict processing to these block ids; the maxima of just those blocks are
            returned. Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        An ``(n_maxima, ndim)`` array of the detected maxima coordinates.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    ndim = src.ndim
    if check_direct(job_type, num_workers, block_shape, None, block_ids):
        return peak_local_max(src[full_roi(ndim)], min_distance=min_distance,
                              threshold_abs=threshold_abs, threshold_rel=threshold_rel)

    halo = [min_distance + 8] * ndim
    runner = get_runner(job_type, job_config)
    results = runner.run(_make_local_maxima(min_distance, threshold_abs, threshold_rel),
                         [input], block_shape=block_shape, halo=halo, num_workers=num_workers,
                         block_ids=block_ids, resume_from=resume_from, has_return_val=True,
                         name="find_local_maxima")
    results = [r for r in results if r is not None]
    if not results:
        return np.zeros((0, ndim), dtype="int64")
    return np.concatenate(results, axis=0)
