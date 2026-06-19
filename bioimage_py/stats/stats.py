"""Block-wise reductions (max, min, mean, std) via the runner's return-value channel."""
from __future__ import annotations

from math import sqrt
from typing import Optional, Sequence, Tuple

import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, check_rerun_args, to_roi

__all__ = ["max", "min", "mean", "std", "mean_and_std", "min_and_max"]


def _is_direct(job_type: str, num_workers: int, block_shape: Optional[Tuple[int, ...]]) -> bool:
    """Return whether this call qualifies for direct (non-blocked) computation."""
    return job_type == "local" and num_workers == 1 and block_shape is None


def _check_direct(job_type: str, num_workers: int, block_shape: Optional[Tuple[int, ...]],
                  mask: Optional[SourceLike], block_ids: Optional[Sequence[int]]) -> bool:
    """Like :func:`_is_direct`, but reject masks/block_ids which the direct path can't honor."""
    if _is_direct(job_type, num_workers, block_shape):
        if mask is not None or block_ids is not None:
            raise ValueError("Direct computation does not support 'mask' or 'block_ids'.")
        return True
    return False


def _full_roi(source: Source) -> Tuple[slice, ...]:
    """Return a slicing that selects the whole source."""
    return tuple(slice(None) for _ in range(source.ndim))


def _masked_block_data(input_: Source, mask: Optional[Source],
                       roi: Tuple[slice, ...]) -> Optional[np.ndarray]:
    """Return the in-mask data for a block, or ``None`` if the block is fully masked out."""
    if mask is None:
        return input_[roi]
    block_mask = mask[roi].astype(bool)
    if not block_mask.any():
        return None
    return input_[roi][block_mask]


# --- max -------------------------------------------------------------------------------

def _max_block(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
               mask: Optional[Source]) -> Optional[float]:
    """Per-block maximum (``None`` if the block is fully masked out)."""
    input_ = inputs[0]
    roi = to_roi(block)
    if mask is None:
        return float(np.max(input_[roi]))
    block_mask = mask[roi].astype(bool)
    mask_sum = block_mask.sum()
    if mask_sum == 0:  # Nothing in the mask -> return early, without reading the input.
        return None
    if mask_sum == block_mask.size:  # Everything in the mask -> no need to apply it.
        return float(np.max(input_[roi]))
    return float(np.max(input_[roi][block_mask]))


def max(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> float:
    """Compute the maximum value of the data, optionally restricted to a mask.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; values outside the mask are excluded from the computation.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The maximum value.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    if _check_direct(job_type, num_workers, block_shape, mask, block_ids):
        src = as_source(input)
        return float(np.max(src[_full_roi(src)]))
    runner = get_runner(job_type, job_config)
    results = runner.run(_max_block, [input], num_workers=num_workers, block_shape=block_shape,
                         mask=mask, block_ids=block_ids, resume_from=resume_from,
                         has_return_val=True, name="max")
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No values within the mask; cannot compute a maximum.")
    return float(np.max(results))


# --- min / max -------------------------------------------------------------------------

def _min_and_max_block(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                       mask: Optional[Source]) -> Optional[Tuple[float, float]]:
    """Per-block (min, max) (``None`` if the block is fully masked out)."""
    d = _masked_block_data(inputs[0], mask, to_roi(block))
    if d is None or d.size == 0:
        return None
    return float(np.min(d)), float(np.max(d))


def min_and_max(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> Tuple[float, float]:
    """Compute the (minimum, maximum) of the data, optionally restricted to a mask.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; values outside the mask are excluded from the computation.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The minimum and maximum values, as a ``(min, max)`` tuple.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    if _check_direct(job_type, num_workers, block_shape, mask, block_ids):
        src = as_source(input)
        d = src[_full_roi(src)]
        return float(np.min(d)), float(np.max(d))
    runner = get_runner(job_type, job_config)
    results = runner.run(_min_and_max_block, [input], num_workers=num_workers, block_shape=block_shape,
                         mask=mask, block_ids=block_ids, resume_from=resume_from,
                         has_return_val=True, name="min_and_max")
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No values within the mask; cannot compute min/max.")
    mins = np.array([r[0] for r in results])
    maxs = np.array([r[1] for r in results])
    return float(mins.min()), float(maxs.max())


def min(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> float:
    """Compute the minimum value of the data, optionally restricted to a mask.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; values outside the mask are excluded from the computation.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The minimum value.
    """
    return min_and_max(input, num_workers, block_shape, job_type, job_config, mask, block_ids,
                       resume_from)[0]


# --- mean / std ------------------------------------------------------------------------

def _mean_and_std_block(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                        mask: Optional[Source]) -> Optional[Tuple[float, float, int]]:
    """Per-block (mean, variance, count) (``None`` if the block is fully masked out)."""
    d = _masked_block_data(inputs[0], mask, to_roi(block))
    if d is None or d.size == 0:
        return None
    return float(np.mean(d)), float(np.var(d)), int(d.size)


def mean_and_std(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> Tuple[float, float]:
    """Compute the (mean, standard deviation) of the data, optionally restricted to a mask.

    Per-block ``(mean, variance, count)`` triples are combined with the parallel-variance
    formula, so the result is exact (not an approximation).

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; values outside the mask are excluded from the computation.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The mean and standard deviation, as a ``(mean, std)`` tuple.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    if _check_direct(job_type, num_workers, block_shape, mask, block_ids):
        src = as_source(input)
        d = src[_full_roi(src)]
        return float(np.mean(d)), float(np.std(d))
    runner = get_runner(job_type, job_config)
    results = runner.run(_mean_and_std_block, [input], num_workers=num_workers, block_shape=block_shape,
                         mask=mask, block_ids=block_ids, resume_from=resume_from,
                         has_return_val=True, name="mean_and_std")
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No values within the mask; cannot compute mean/std.")
    means = np.array([r[0] for r in results])
    variances = np.array([r[1] for r in results])
    sizes = np.array([r[2] for r in results], dtype="float64")
    mean_val = float((sizes * means).sum() / sizes.sum())
    # Parallel variance combination (mirrors elf): account for the shift of each block mean.
    var_val = float((sizes * (variances + (means - mean_val) ** 2)).sum() / sizes.sum())
    return mean_val, sqrt(var_val)


def mean(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> float:
    """Compute the mean of the data, optionally restricted to a mask.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; values outside the mask are excluded from the computation.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The mean value.
    """
    return mean_and_std(input, num_workers, block_shape, job_type, job_config, mask, block_ids,
                        resume_from)[0]


def std(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> float:
    """Compute the standard deviation of the data, optionally restricted to a mask.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; values outside the mask are excluded from the computation.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The standard deviation.
    """
    return mean_and_std(input, num_workers, block_shape, job_type, job_config, mask, block_ids,
                        resume_from)[1]
