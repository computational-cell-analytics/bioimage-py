"""Block-wise statistics with bounded associative reducers."""
from __future__ import annotations

from math import sqrt
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..runner.reducer import DEFAULT_REDUCTION_BATCH_SIZE, _normalize_reduction_batch_size
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, check_direct, check_rerun_args, full_roi, to_roi

__all__ = ["max", "min", "mean", "std", "mean_and_std", "min_and_max"]


class _MaximumReducer:
    """Reduce optional block maxima."""

    def initial(self) -> Optional[float]:
        return None

    def update(self, accumulator: Optional[float], value: Optional[float]) -> Optional[float]:
        return self.merge(accumulator, value)

    def merge(self, left: Optional[float], right: Optional[float]) -> Optional[float]:
        if left is None:
            return right
        if right is None:
            return left
        return float(np.maximum(left, right))

    def finalize(self, accumulator: Optional[float]) -> Optional[float]:
        return accumulator


class _MinMaxReducer:
    """Reduce optional ``(minimum, maximum)`` block values."""

    def initial(self) -> Optional[Tuple[float, float]]:
        return None

    def update(
        self,
        accumulator: Optional[Tuple[float, float]],
        value: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        return self.merge(accumulator, value)

    def merge(
        self,
        left: Optional[Tuple[float, float]],
        right: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if left is None:
            return right
        if right is None:
            return left
        return float(np.minimum(left[0], right[0])), float(np.maximum(left[1], right[1]))

    def finalize(
        self, accumulator: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        return accumulator


class _Moments(NamedTuple):
    """Mergeable population moments."""

    count: int
    mean: float
    m2: float


def _merge_moments(left: _Moments, right: _Moments) -> _Moments:
    """Merge two population-moment accumulators."""
    if left.count == 0:
        return right
    if right.count == 0:
        return left
    count = left.count + right.count
    delta = right.mean - left.mean
    mean = left.mean + delta * right.count / count
    m2 = left.m2 + right.m2 + delta * delta * left.count * right.count / count
    return _Moments(count, mean, m2)


class _MomentsReducer:
    """Reduce optional block moments with the parallel-variance formula."""

    def initial(self) -> _Moments:
        return _Moments(0, 0.0, 0.0)

    def update(
        self,
        accumulator: _Moments,
        value: Optional[Tuple[float, float, int]],
    ) -> _Moments:
        if value is None:
            return accumulator
        mean, variance, count = value
        return _merge_moments(accumulator, _Moments(count, mean, variance * count))

    def merge(self, left: _Moments, right: _Moments) -> _Moments:
        return _merge_moments(left, right)

    def finalize(self, accumulator: _Moments) -> Optional[Tuple[float, float]]:
        if accumulator.count == 0:
            return None
        return accumulator.mean, sqrt(accumulator.m2 / accumulator.count)


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
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
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
        reduction_batch_size: Block results per durable reducer batch.

    Returns:
        The maximum value.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    reduction_batch_size = _normalize_reduction_batch_size(reduction_batch_size)
    if check_direct(job_type, num_workers, block_shape, mask, block_ids):
        src = as_source(input)
        return float(np.max(src[full_roi(src.ndim)]))
    runner = get_runner(job_type, job_config)
    result = runner.run(_max_block, [input], num_workers=num_workers, block_shape=block_shape,
                        mask=mask, block_ids=block_ids, resume_from=resume_from,
                        has_return_val=True, name="max", reducer=_MaximumReducer(),
                        reduction_batch_size=reduction_batch_size)
    if result is None:
        raise ValueError("No values within the mask; cannot compute a maximum.")
    return float(result)


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
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
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
        reduction_batch_size: Block results per durable reducer batch.

    Returns:
        The minimum and maximum values, as a ``(min, max)`` tuple.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    reduction_batch_size = _normalize_reduction_batch_size(reduction_batch_size)
    if check_direct(job_type, num_workers, block_shape, mask, block_ids):
        src = as_source(input)
        d = src[full_roi(src.ndim)]
        return float(np.min(d)), float(np.max(d))
    runner = get_runner(job_type, job_config)
    result = runner.run(
        _min_and_max_block, [input], num_workers=num_workers, block_shape=block_shape,
        mask=mask, block_ids=block_ids, resume_from=resume_from, has_return_val=True,
        name="min_and_max", reducer=_MinMaxReducer(),
        reduction_batch_size=reduction_batch_size,
    )
    if result is None:
        raise ValueError("No values within the mask; cannot compute min/max.")
    return result


def min(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
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
        reduction_batch_size: Block results per durable reducer batch.

    Returns:
        The minimum value.
    """
    return min_and_max(
        input, num_workers, block_shape, job_type, job_config, mask, block_ids,
        resume_from, reduction_batch_size,
    )[0]


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
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
) -> Tuple[float, float]:
    """Compute the (mean, standard deviation) of the data, optionally restricted to a mask.

    Per-block ``(mean, variance, count)`` triples use the parallel-variance formula. Floating-point
    grouping can cause small differences between direct and blocked results.

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
        reduction_batch_size: Block results per durable reducer batch.

    Returns:
        The mean and standard deviation, as a ``(mean, std)`` tuple.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    reduction_batch_size = _normalize_reduction_batch_size(reduction_batch_size)
    if check_direct(job_type, num_workers, block_shape, mask, block_ids):
        src = as_source(input)
        d = src[full_roi(src.ndim)]
        return float(np.mean(d)), float(np.std(d))
    runner = get_runner(job_type, job_config)
    result = runner.run(
        _mean_and_std_block, [input], num_workers=num_workers, block_shape=block_shape,
        mask=mask, block_ids=block_ids, resume_from=resume_from, has_return_val=True,
        name="mean_and_std", reducer=_MomentsReducer(),
        reduction_batch_size=reduction_batch_size,
    )
    if result is None:
        raise ValueError("No values within the mask; cannot compute mean/std.")
    return result


def mean(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
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
        reduction_batch_size: Block results per durable reducer batch.

    Returns:
        The mean value.
    """
    return mean_and_std(
        input, num_workers, block_shape, job_type, job_config, mask, block_ids,
        resume_from, reduction_batch_size,
    )[0]


def std(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
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
        reduction_batch_size: Block results per durable reducer batch.

    Returns:
        The standard deviation.
    """
    return mean_and_std(
        input, num_workers, block_shape, job_type, job_config, mask, block_ids,
        resume_from, reduction_batch_size,
    )[1]
