"""Block-wise consecutive relabeling (multi-stage: unique -> map -> write).

Relabels a segmentation so its ids are consecutive. Like ``label``, this is a multi-stage op (a
global ``unique`` reduction, an in-process mapping, then a block-wise write), so it re-runs whole and
does **not** accept ``block_ids`` / ``resume_from``. The label mapping is applied per block with
``bioimage_cpp.utils.take_dict``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..stats.unique import unique
from ..util import BlockDescriptor, ComputeFn, full_roi, is_direct, same_array, to_roi

__all__ = ["relabel_consecutive"]


def _make_relabel_block(mapping: Dict[int, int]) -> ComputeFn:
    """Build the per-block write function applying the global label mapping (picklable dict)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        input_, output_ = inputs[0], outputs[0]
        roi = to_roi(block)
        seg = input_[roi]
        if mask is None:
            output_[roi] = bic.utils.take_dict(mapping, seg)
            return None
        m = mask[roi].astype(bool)
        if not m.any():
            return None
        # Only in-mask voxels were seen by the unique pass, so only they are in the mapping;
        # out-of-mask output voxels are left unchanged.
        out_block = output_[roi].copy()
        out_block[m] = bic.utils.take_dict(mapping, seg[m])
        output_[roi] = out_block
        return None

    return _compute


def relabel_consecutive(
    input: SourceLike,
    output: Optional[SourceLike] = None,
    *,
    start_label: int = 0,
    keep_zeros: bool = True,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
) -> Tuple[SourceLike, int, Dict[int, int]]:
    """Relabel a segmentation to consecutive ids, block-wise.

    Like ``label``, this is multi-stage (a global ``unique`` reduction, an in-process mapping, then a
    block-wise write), so it does **not** accept ``block_ids`` or ``resume_from``: a failed run is
    re-run whole (it is idempotent given the same ``output``).

    Args:
        input: The input label image (a numpy/zarr/n5 array or a `Source`); must be integer-typed.
        output: The output array to write the relabeled segmentation into. Optional for local
            execution -- a numpy array matching the input shape and dtype is allocated and returned
            if omitted; **required** for distributed execution (a writable, file-backed zarr/n5
            array).
        start_label: The value the smallest unique id is mapped to (subsequent ids follow
            consecutively).
        keep_zeros: Whether to always keep ``0`` mapped to ``0`` (background), regardless of
            ``start_label``.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; values outside the mask are excluded from the computation and
            their output voxels are left unchanged.

    Returns:
        A ``(output, max_id, mapping)`` tuple: the relabeled output array, the maximum label id
        after relabeling, and the ``{old_id: new_id}`` mapping that was applied.
    """
    src = as_source(input)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"relabel_consecutive expects an integer label image, got dtype {src.dtype}.")
    ndim = src.ndim

    direct = is_direct(job_type, num_workers, block_shape) and mask is None

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
        raise ValueError("Block-wise relabel_consecutive needs 'output' to differ from 'input'.")

    # Pass 1: the global set of unique values.
    if direct:
        uniques = np.unique(src[full_roi(ndim)])
    else:
        uniques = unique(input, block_shape=block_shape, job_type=job_type, job_config=job_config,
                         num_workers=num_workers, mask=mask)

    # In-process: build the old -> new mapping (consecutive ids from start_label).
    mapping: Dict[int, int] = {int(v): i for i, v in enumerate(uniques.tolist(), start_label)}
    if keep_zeros and 0 in mapping:
        mapping[0] = 0
    max_id = max(mapping.values()) if mapping else 0

    # Pass 2: apply the mapping.
    if direct:
        out[full_roi(ndim)] = bic.utils.take_dict(mapping, src[full_roi(ndim)])
        return out_array, max_id, mapping

    runner = get_runner(job_type, job_config)
    runner.run(_make_relabel_block(mapping), [input], outputs=[out_array], block_shape=block_shape,
               mask=mask, num_workers=num_workers, name="relabel")
    return out_array, max_id, mapping
