"""Block-wise copy of one source into another.

Copies an input source into a writable output source block-wise, reusing the runner machinery.
This is the minimal array-output operation: there is no halo and no per-block computation, just a
read of each block from the input and a write to the output. Typical uses are converting between
storage formats (e.g. a tiff stack to zarr) and persisting an on-the-fly transformation (a
``wrapper`` source) to file so the result is stored rather than recomputed on every read.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .runner import get_runner
from .runner.config import RunnerConfig
from .sources import Source, SourceLike, as_source
from .util import BlockDescriptor, ComputeFn, check_rerun_args, full_roi, is_direct, to_roi

__all__ = ["copy"]


def _same_array(a: Source, b: Source) -> bool:
    """Return whether two sources wrap the same underlying array object."""
    return getattr(a, "array", None) is getattr(b, "array", object())


def _make_compute() -> ComputeFn:
    """Build the per-block copy function (no captured state, so trivially cloudpickle-safe)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        input_, output_ = inputs[0], outputs[0]
        roi = to_roi(block)
        if mask is not None:
            m = mask[roi].astype(bool)
            if not m.any():
                return None
            # Keep out-of-mask voxels of the output unchanged.
            output_[roi] = np.where(m, input_[roi], output_[roi])
            return None
        output_[roi] = input_[roi]
        return None

    return _compute


def _copy_source(
    input: SourceLike,
    output: Optional[SourceLike],
    *,
    block_shape: Optional[Tuple[int, ...]],
    job_type: str,
    job_config: Optional[RunnerConfig],
    num_workers: int,
    mask: Optional[SourceLike],
    name: str,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> SourceLike:
    """Materialize ``input`` into ``output`` block-wise (shared by :func:`copy` and downsample).

    Handles output allocation (a numpy array for local execution, a required file-backed array for
    distributed execution), the direct (whole-array) fast path, and the runner dispatch. The output
    shape and dtype are taken from ``input`` (so a shape-changing wrapper input is handled too).
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    ndim = src.ndim
    # A subset/resume rerun is block-wise, so it cannot use the direct (whole-array) path.
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
    if not direct and _same_array(out, src):
        raise ValueError(f"Block-wise {name} needs 'output' to differ from 'input'.")

    if direct:
        out[full_roi(out.ndim)] = src[full_roi(ndim)]
        return out_array

    compute = _make_compute()
    runner = get_runner(job_type, job_config)
    runner.run(compute, [src], outputs=[out], block_shape=block_shape,
               mask=mask, num_workers=num_workers, block_ids=block_ids, resume_from=resume_from,
               name=name)
    return out_array


def copy(
    input: SourceLike,
    output: Optional[SourceLike] = None,
    *,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> SourceLike:
    """Copy a source into an output, block-wise.

    The data is read from ``input`` and written into ``output`` one block at a time. The input may
    be any source, including a read-only ``FileSource`` (e.g. a tiff stack) or a ``wrapper`` source
    whose transformation is computed on read; copying it materializes the transformed data to the
    output. The data is written into the output as-is, so the output array's dtype governs and a
    cast is applied on assignment when it differs from the input dtype.

    Args:
        input: The input data to copy (a numpy/zarr/n5 array or a `Source`).
        output: The output array to write into. Optional for local execution — a numpy array
            matching the input shape and dtype is allocated and returned if omitted; **required**
            for distributed execution, where it must be a writable, file-backed (zarr/n5) array.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; only voxels within the mask are copied (out-of-mask output
            voxels are left unchanged).
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks
            into the existing ``output``). Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``); the missing blocks are written into ``output``. Mutually exclusive
            with ``block_ids``.

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    return _copy_source(input, output, block_shape=block_shape, job_type=job_type,
                        job_config=job_config, num_workers=num_workers, mask=mask, name="copy",
                        block_ids=block_ids, resume_from=resume_from)
