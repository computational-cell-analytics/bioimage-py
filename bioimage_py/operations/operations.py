"""Block-wise element-wise operations: arithmetic, comparison and membership.

A family of single-pass, halo-free array-output ops. ``apply_operation`` combines two operands with a
binary operation; the arithmetic/comparison helpers (``add``, ``multiply``, ``greater``, ...) are thin
wrappers that dispatch a numpy function by name, and ``isin`` tests membership. The second operand may
be a scalar, a same-shape array/source, or a broadcastable array (size-1 dims); a scalar / broadcast
operand is captured into the (cloudpickled) per-block closure, a same-shape operand is passed as a
second input source (so it must be file-backed for distributed execution).

Output follows the repo convention: ``output=None`` allocates a new array for local execution and is
required (file-backed) for distributed execution. Because the ops are element-wise and halo-free,
``output`` may also be the input itself (an explicit in-place computation).
"""
from __future__ import annotations

from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, ComputeFn, check_rerun_args, full_roi, is_direct, to_roi

__all__ = ["apply_operation", "isin", "add", "subtract", "multiply", "divide",
           "greater", "greater_equal", "less", "less_equal", "minimum", "maximum"]


def _broadcast_axes(shape_x: Sequence[int], shape_y: Sequence[int]) -> List[bool]:
    """Per-axis flags: ``True`` where the second operand has size 1 and must be broadcast."""
    axes = []
    for sx, sy in zip(shape_x, shape_y):
        if sx == sy:
            axes.append(False)
        elif sy == 1:
            axes.append(True)
        else:
            raise ValueError(f"Cannot broadcast operand shape {tuple(shape_y)} against {tuple(shape_x)}.")
    return axes


def _resolve_op(operation: Union[str, Callable]) -> Callable:
    """Resolve an operation given as a numpy-function name or returned as-is if callable."""
    if callable(operation):
        return operation
    if not hasattr(np, operation):
        raise ValueError(f"Unknown operation {operation!r}; expected a numpy function name or a callable.")
    return getattr(np, operation)


def _make_apply(operation: Union[str, Callable], mode: str, scalar_y: Optional[Number],
                broadcast_y: Optional[np.ndarray], broadcast_axes: Optional[List[bool]]) -> ComputeFn:
    """Build the per-block binary-operation function (captures only picklable values)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        op = _resolve_op(operation)
        roi = to_roi(block)
        x_block = inputs[0][roi]
        if mode == "scalar":
            res = op(x_block, scalar_y)
        elif mode == "array":
            res = op(x_block, inputs[1][roi])
        else:  # broadcast: index the captured operand, reading size-1 (broadcast) axes whole.
            roi_y = tuple(slice(None) if b else r for b, r in zip(broadcast_axes, roi))
            res = op(x_block, broadcast_y[roi_y])
        if mask is not None:
            m = mask[roi].astype(bool)
            if not m.any():
                return None
            res = np.where(m, res, outputs[0][roi])  # keep out-of-mask output voxels unchanged.
        outputs[0][roi] = res
        return None

    return _compute


def _result_dtype(op: Callable, x_src: Source, mode: str, scalar_y: Optional[Number],
                  broadcast_y: Optional[np.ndarray], y_src: Optional[Source]) -> np.dtype:
    """Infer the output dtype by applying ``op`` to a one-voxel sample (so bool/float results work)."""
    corner = tuple(slice(0, 1) for _ in range(x_src.ndim))
    sx = np.asarray(x_src[corner])
    if mode == "scalar":
        sample = op(sx, scalar_y)
    elif mode == "broadcast":
        sample = op(sx, broadcast_y[corner])
    else:
        sample = op(sx, np.asarray(y_src[corner]))
    return np.asarray(sample).dtype


def apply_operation(
    x: SourceLike,
    y: Union[SourceLike, Number],
    operation: Union[str, Callable],
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
    """Apply a binary operation to two operands block-wise.

    Args:
        x: The first operand (a numpy/zarr/n5 array or a `Source`).
        y: The second operand: a scalar, or an array/source. An array operand must have the same
            number of dimensions as ``x``; it may either match ``x``'s shape or be broadcastable to it
            (size-1 dimensions). A same-shape array operand is read block-wise (so it must be
            file-backed for distributed execution); a scalar or broadcast operand is captured into the
            worker closure.
        operation: The binary operation: a numpy function name (e.g. ``"add"``) or a picklable callable
            ``operation(x_block, y_block)``.
        output: The output array to write into. Optional for local execution -- a numpy array (dtype
            inferred from the operation) is allocated and returned if omitted; **required** for
            distributed execution. May be ``x`` itself for an in-place computation.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; out-of-mask output voxels are left unchanged. Not supported
            together with a broadcast operand.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
            Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    check_rerun_args(job_type, resume_from, block_ids)
    x_src = as_source(x)
    ndim = x_src.ndim
    op = _resolve_op(operation)

    scalar_y: Optional[Number] = None
    broadcast_y: Optional[np.ndarray] = None
    broadcast_axes: Optional[List[bool]] = None
    y_src: Optional[Source] = None
    if isinstance(y, Number):
        mode = "scalar"
        scalar_y = y
        inputs: List[SourceLike] = [x]
    else:
        y_src = as_source(y)
        if y_src.ndim != ndim:
            raise ValueError(f"operand dimensionalities differ: {ndim} vs {y_src.ndim}.")
        if tuple(y_src.shape) == tuple(x_src.shape):
            mode = "array"
            inputs = [x, y]
        else:
            mode = "broadcast"
            broadcast_axes = _broadcast_axes(x_src.shape, y_src.shape)
            broadcast_y = np.asarray(y_src[full_roi(ndim)])
            inputs = [x]
    if mode == "broadcast" and mask is not None:
        raise NotImplementedError("Combining a broadcast operand with a mask is not supported.")

    direct = (is_direct(job_type, num_workers, block_shape) and mask is None
              and block_ids is None and resume_from is None)

    if output is None:
        if job_type != "local":
            raise ValueError(
                f"'output' is required for distributed execution (job_type={job_type!r}); "
                "pass a file-backed (zarr/n5) output array."
            )
        out_dtype = _result_dtype(op, x_src, mode, scalar_y, broadcast_y, y_src)
        out_array: SourceLike = np.zeros(tuple(x_src.shape), dtype=out_dtype)
    else:
        out_array = output
    out = as_source(out_array)

    if direct:
        x_full = x_src[full_roi(ndim)]
        if mode == "scalar":
            res = op(x_full, scalar_y)
        elif mode == "array":
            res = op(x_full, y_src[full_roi(ndim)])
        else:
            res = op(x_full, broadcast_y)
        out[full_roi(out.ndim)] = res
        return out_array

    runner = get_runner(job_type, job_config)
    name = operation if isinstance(operation, str) else "apply_operation"
    runner.run(_make_apply(operation, mode, scalar_y, broadcast_y, broadcast_axes),
               inputs, outputs=[out_array], block_shape=block_shape, mask=mask,
               num_workers=num_workers, block_ids=block_ids, resume_from=resume_from, name=name)
    return out_array


def _make_isin(test_values: Union[np.ndarray, Number]) -> ComputeFn:
    """Build the per-block membership-test function (captures the picklable test values)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        roi = to_roi(block)
        res = np.isin(inputs[0][roi], test_values)
        if mask is not None:
            m = mask[roi].astype(bool)
            if not m.any():
                return None
            res = np.where(m, res, outputs[0][roi])
        outputs[0][roi] = res
        return None

    return _compute


def isin(
    x: SourceLike,
    test_values: Union[ArrayLike, Number],
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
    """Compute ``np.isin(x, test_values)`` block-wise (a boolean membership mask).

    Args:
        x: The input data (a numpy/zarr/n5 array or a `Source`).
        test_values: The values to test membership against (a scalar, list or array); captured into
            the worker closure.
        output: The boolean output array to write into. Optional for local execution -- a boolean
            numpy array is allocated and returned if omitted; **required** for distributed execution.
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
        The boolean output array (the provided ``output``, or a newly allocated numpy array).
    """
    check_rerun_args(job_type, resume_from, block_ids)
    x_src = as_source(x)
    ndim = x_src.ndim
    values = test_values if isinstance(test_values, Number) else np.asarray(test_values)

    direct = (is_direct(job_type, num_workers, block_shape) and mask is None
              and block_ids is None and resume_from is None)

    if output is None:
        if job_type != "local":
            raise ValueError(
                f"'output' is required for distributed execution (job_type={job_type!r}); "
                "pass a file-backed (zarr/n5) output array."
            )
        out_array: SourceLike = np.zeros(tuple(x_src.shape), dtype=bool)
    else:
        out_array = output
    out = as_source(out_array)

    if direct:
        out[full_roi(out.ndim)] = np.isin(x_src[full_roi(ndim)], values)
        return out_array

    runner = get_runner(job_type, job_config)
    runner.run(_make_isin(values), [x], outputs=[out_array], block_shape=block_shape, mask=mask,
               num_workers=num_workers, block_ids=block_ids, resume_from=resume_from, name="isin")
    return out_array


def _make_op(op_name: str) -> Callable:
    """Build a public wrapper applying ``np.<op_name>`` via :func:`apply_operation`."""

    def op(
        x: SourceLike,
        y: Union[SourceLike, Number],
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
        return apply_operation(x, y, op_name, output, block_shape=block_shape, job_type=job_type,
                               job_config=job_config, num_workers=num_workers, mask=mask,
                               block_ids=block_ids, resume_from=resume_from)

    op.__name__ = op_name
    op.__qualname__ = op_name
    op.__doc__ = (
        f"""Apply ``np.{op_name}`` to two operands block-wise.

    A thin wrapper over :func:`apply_operation` with ``operation={op_name!r}``; see it for the full
    parameter documentation (operands, ``output`` handling, backends, mask and re-run options).
    """
    )
    return op


add = _make_op("add")
subtract = _make_op("subtract")
multiply = _make_op("multiply")
divide = _make_op("divide")
greater = _make_op("greater")
greater_equal = _make_op("greater_equal")
less = _make_op("less")
less_equal = _make_op("less_equal")
minimum = _make_op("minimum")
maximum = _make_op("maximum")
