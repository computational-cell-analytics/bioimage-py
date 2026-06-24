"""Block-wise unique values (and optional counts) via the runner's return channel.

A reduction operation: each block computes the unique values it contains (and, optionally, their
counts) and the main process merges them. Without halo, the blocks partition the volume disjointly,
so the per-value counts are additive across blocks. The merge mirrors ``stats`` / ``contingency_table``
-- the per-block results flow through ``runner.run(..., has_return_val=True)`` and the merge is pure
numpy -- so it behaves identically across the ``local`` / ``subprocess`` / ``slurm`` backends.

The count merge groups the stacked ``(value, count)`` rows with a single ``argsort`` + ``reduceat``
(the 1-key variant of ``contingency_table``'s merge), rather than scattering into a dense
``counts[max_id + 1]`` array, so it stays memory-safe for sparse, large label ids.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, ComputeFn, check_direct, check_rerun_args, full_roi, to_roi

__all__ = ["unique"]


def _make_unique_block(return_counts: bool) -> ComputeFn:
    """Build the per-block unique function (captures only the picklable ``return_counts`` flag)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        roi = to_roi(block)
        d = inputs[0][roi]
        if mask is not None:
            block_mask = mask[roi].astype(bool)
            if not block_mask.any():
                return None
            d = d[block_mask]
        if return_counts:
            values, counts = np.unique(d, return_counts=True)
            return values, counts.astype("int64")
        return np.unique(d)

    return _compute


def _merge_unique(results: List, return_counts: bool,
                  dtype: np.dtype) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Merge per-block unique values (and counts) into the global result."""
    results = [r for r in results if r is not None]
    if not return_counts:
        if not results:
            return np.zeros((0,), dtype=dtype)
        return np.unique(np.concatenate(results))

    if not results:
        return np.zeros((0,), dtype=dtype), np.zeros((0,), dtype="int64")
    values = np.concatenate([r[0] for r in results])
    counts = np.concatenate([r[1] for r in results])
    order = np.argsort(values)
    values, counts = values[order], counts[order]
    starts = np.flatnonzero(np.concatenate(([True], values[1:] != values[:-1])))
    return values[starts], np.add.reduceat(counts, starts)


def unique(
    input: SourceLike,
    return_counts: bool = False,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute the unique values of the data, optionally with their counts.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        return_counts: Whether to also return the number of occurrences of each unique value.
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
        The sorted unique values. If ``return_counts`` is set, a ``(values, counts)`` tuple, where
        ``counts`` is an ``int64`` array aligned with ``values``.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    if check_direct(job_type, num_workers, block_shape, mask, block_ids):
        d = src[full_roi(src.ndim)]
        if return_counts:
            values, counts = np.unique(d, return_counts=True)
            return values, counts.astype("int64")
        return np.unique(d)
    runner = get_runner(job_type, job_config)
    results = runner.run(_make_unique_block(return_counts), [input], num_workers=num_workers,
                         block_shape=block_shape, mask=mask, block_ids=block_ids,
                         resume_from=resume_from, has_return_val=True, name="unique")
    return _merge_unique(results, return_counts, np.dtype(src.dtype))
