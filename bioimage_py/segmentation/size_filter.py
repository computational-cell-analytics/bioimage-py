"""Block-wise segmentation filtering: a generic per-block predicate and a size filter.

``segmentation_filter`` is the generic, single-pass form: it applies a user ``filter_function`` (and
optional ``relabel``) to each block, so it supports ``block_ids`` / ``resume_from``. Both callables
are cloudpickled to the workers, so they must be picklable (capture only picklable values).

``size_filter`` removes objects below ``min_size`` / above ``max_size``. It is multi-stage (a global
``unique`` count reduction, then a filter pass via ``segmentation_filter``), so it does **not** accept
``block_ids`` / ``resume_from``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..stats.unique import unique
from ..util import BlockDescriptor, ComputeFn, check_rerun_args, full_roi, is_direct, same_array, to_roi

__all__ = ["segmentation_filter", "size_filter"]

# A per-block predicate/relabel callable: ``f(block_seg, block_mask) -> block_seg``. ``block_mask`` is
# the boolean in-mask array for the block (or ``None``); when given, only its in-mask voxels are used.
BlockFn = Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]


def _make_filter_block(filter_function: BlockFn, relabel: Optional[BlockFn]) -> ComputeFn:
    """Build the per-block function applying ``filter_function`` (and optional ``relabel``)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        input_, output_ = inputs[0], outputs[0]
        roi = to_roi(block)
        if mask is None:
            block_mask = None
        else:
            block_mask = mask[roi].astype(bool)
            if not block_mask.any():
                return None
        filtered = filter_function(input_[roi], block_mask)
        if relabel is not None:
            filtered = relabel(filtered, block_mask)
        if block_mask is None:
            output_[roi] = filtered
        else:  # keep out-of-mask voxels of the output unchanged.
            output_[roi] = np.where(block_mask, filtered, output_[roi])
        return None

    return _compute


def segmentation_filter(
    input: SourceLike,
    filter_function: BlockFn,
    output: Optional[SourceLike] = None,
    *,
    relabel: Optional[BlockFn] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> SourceLike:
    """Filter a segmentation with a custom per-block criterion, block-wise.

    Args:
        input: The input segmentation (a numpy/zarr/n5 array or a `Source`).
        filter_function: A picklable callable ``filter_function(block_seg, block_mask)`` returning the
            filtered block. ``block_mask`` is the block's boolean in-mask array, or ``None`` when no
            mask is used; when a mask is used, restrict the criterion to the in-mask voxels.
        output: The output array to write into. Optional for local execution -- a numpy array
            matching the input shape and dtype is allocated and returned if omitted; **required** for
            distributed execution (a writable, file-backed zarr/n5 array).
        relabel: Optional picklable callable ``relabel(block_seg, block_mask)`` applied after
            ``filter_function`` (e.g. a consecutive relabeling); same masking contract.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; out-of-mask output voxels are left unchanged.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
            Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    ndim = src.ndim
    direct = (is_direct(job_type, num_workers, block_shape) and mask is None
              and block_ids is None and resume_from is None)

    if output is None:
        if job_type != "local":
            raise ValueError(
                f"'output' is required for distributed execution (job_type={job_type!r}); "
                "pass a file-backed (zarr/n5) output array."
            )
        out_array: SourceLike = np.zeros(tuple(src.shape), dtype=src.dtype)
    else:
        out_array = output
    out = as_source(out_array)
    if not direct and same_array(out, src):
        raise ValueError("Block-wise segmentation_filter needs 'output' to differ from 'input'.")

    if direct:
        filtered = filter_function(src[full_roi(ndim)], None)
        if relabel is not None:
            filtered = relabel(filtered, None)
        out[full_roi(ndim)] = filtered
        return out_array

    runner = get_runner(job_type, job_config)
    runner.run(_make_filter_block(filter_function, relabel), [input], outputs=[out_array],
               block_shape=block_shape, mask=mask, num_workers=num_workers,
               block_ids=block_ids, resume_from=resume_from, name="segmentation_filter")
    return out_array


def _make_size_filter(filter_ids: np.ndarray) -> BlockFn:
    """Build the filter callable that sets voxels of the discarded ids to ``0``."""

    def filter_function(block_seg: np.ndarray, block_mask: Optional[np.ndarray]) -> np.ndarray:
        discard = np.isin(block_seg, filter_ids)
        if block_mask is not None:
            discard &= block_mask
        out = block_seg.copy()
        out[discard] = 0
        return out

    return filter_function


def _make_size_relabel(mapping: Dict[int, int]) -> BlockFn:
    """Build the relabel callable mapping surviving ids to consecutive values."""

    def relabel(block_seg: np.ndarray, block_mask: Optional[np.ndarray]) -> np.ndarray:
        if block_mask is None:
            return bic.utils.take_dict(mapping, block_seg)
        out = block_seg.copy()
        out[block_mask] = bic.utils.take_dict(mapping, block_seg[block_mask])
        return out

    return relabel


def size_filter(
    input: SourceLike,
    output: Optional[SourceLike] = None,
    *,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    relabel: bool = True,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
) -> SourceLike:
    """Remove objects smaller than ``min_size`` and/or larger than ``max_size`` from a segmentation.

    Multi-stage (a global ``unique`` count reduction, then a filter pass), so it does **not** accept
    ``block_ids`` / ``resume_from``. By default it relabels the result consecutively; pass
    ``relabel=False`` to keep the original ids of the surviving objects.

    Args:
        input: The input segmentation (a numpy/zarr/n5 array or a `Source`); must be integer-typed.
        output: The output array to write into. Optional for local execution -- a numpy array
            matching the input shape and dtype is allocated and returned if omitted; **required** for
            distributed execution (a writable, file-backed zarr/n5 array).
        min_size: The minimum object size; smaller objects are removed. At least one of ``min_size`` /
            ``max_size`` is required.
        max_size: The maximum object size; larger objects are removed.
        relabel: Whether to relabel the surviving objects consecutively after filtering.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data. Required when a ``mask`` is given (the size reduction is block-wise).
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; out-of-mask output voxels are left unchanged.

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    if min_size is None and max_size is None:
        raise ValueError("size_filter requires at least one of 'min_size' or 'max_size'.")
    src = as_source(input)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"size_filter expects an integer label image, got dtype {src.dtype}.")

    # Pass 1: unique ids with their sizes.
    ids, counts = unique(input, return_counts=True, block_shape=block_shape, job_type=job_type,
                         job_config=job_config, num_workers=num_workers, mask=mask)

    # In-process: ids to discard and the consecutive relabeling of the survivors.
    discard = np.zeros(ids.shape, dtype=bool)
    if min_size is not None:
        discard |= counts < min_size
    if max_size is not None:
        discard |= counts > max_size
    filter_ids = ids[discard]

    relabel_fn: Optional[BlockFn] = None
    if relabel:
        # Reserve 0 for background and map the surviving foreground ids to 1..K consecutively, so a
        # surviving object can never collide with the (possibly newly introduced) background 0.
        remaining_fg = ids[(~discard) & (ids != 0)]
        mapping: Dict[int, int] = {int(v): i for i, v in enumerate(remaining_fg.tolist(), start=1)}
        mapping[0] = 0
        relabel_fn = _make_size_relabel(mapping)

    return segmentation_filter(input, _make_size_filter(filter_ids), output, relabel=relabel_fn,
                               block_shape=block_shape, job_type=job_type, job_config=job_config,
                               num_workers=num_workers, mask=mask)
