"""Block-wise image filters (gaussian family), generalized over a filter name.

Mirrors elf's parallel filters but built on the runner's halo + output-source channels.
The per-block function dispatches by filter name (a string) so it stays cloudpickle-friendly.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import (BlockDescriptor, ComputeFn, check_rerun_args, full_roi, is_direct, normalize_halo,
                    sigma_to_halo, to_roi)

__all__ = [
    "apply_filter",
    "gaussian_smoothing",
    "gaussian_gradient_magnitude",
    "laplacian_of_gaussian",
]

# A scalar (isotropic) or per-axis (anisotropic) filter standard deviation.
Sigma = Union[float, Sequence[float]]

_FILTER_FUNCTIONS = {
    "gaussian_smoothing": bic.filters.gaussian_smoothing,
    "gaussian_gradient_magnitude": bic.filters.gaussian_gradient_magnitude,
    "laplacian_of_gaussian": bic.filters.laplacian_of_gaussian,
    "hessian_of_gaussian_eigenvalues": bic.filters.hessian_of_gaussian_eigenvalues,
    "structure_tensor_eigenvalues": bic.filters.structure_tensor_eigenvalues,
}

# Derivative order per filter, used to size the halo.
_ORDERS = {
    "gaussian_smoothing": 0,
    "gaussian_gradient_magnitude": 1,
    "laplacian_of_gaussian": 2,
    "hessian_of_gaussian_eigenvalues": 2,
    "structure_tensor_eigenvalues": 1,
}

# Filters whose response has a trailing channel axis.
_MULTI_CHANNEL = {"hessian_of_gaussian_eigenvalues", "structure_tensor_eigenvalues"}


def _same_array(a: Source, b: Source) -> bool:
    """Return whether two sources wrap the same underlying array object."""
    return getattr(a, "array", None) is getattr(b, "array", object())


def _allocate_output(src: Source, ndim: int, multi_channel: bool) -> np.ndarray:
    """Allocate a numpy output array matching the filter response shape and dtype."""
    out_dtype = np.float64 if np.dtype(src.dtype) == np.dtype(np.float64) else np.float32
    shape: Tuple[int, ...] = ((ndim,) + tuple(src.shape)) if multi_channel else tuple(src.shape)
    return np.zeros(shape, dtype=out_dtype)


def _make_compute(filter_name: str, sigma: Sigma, extra_args: Tuple[float, ...], ndim: int,
                  return_channel: Optional[int]) -> ComputeFn:
    """Build the per-block filter function. Captures only picklable values."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        input_, output_ = inputs[0], outputs[0]
        outer = to_roi(block.outer_block)
        inner = to_roi(block.inner_block)
        inner_local = to_roi(block.inner_block_local)

        if mask is not None:
            m = mask[inner].astype(bool)
            if not m.any():
                return None

        block_data = input_[outer]
        response = _FILTER_FUNCTIONS[filter_name](block_data, sigma, *extra_args)
        # The spatial dims are leading; crop them to the inner block (channel axis stays).
        response = response[inner_local]
        if return_channel is not None:
            response = response[..., return_channel]

        multi = response.ndim > ndim
        if multi:  # channel-last -> channel-first to match a (C, *spatial) output.
            response = np.moveaxis(response, -1, 0)
            write_roi: Tuple[slice, ...] = (slice(None),) + inner
        else:
            write_roi = inner

        if mask is not None:  # keep out-of-mask voxels of the output unchanged.
            existing = output_[write_roi]
            response = np.where(m[None] if multi else m, response, existing)

        output_[write_roi] = response
        return None

    return _compute


def _apply_direct(src: Source, out: Source, filter_name: str, sigma: Sigma,
                  extra_args: Tuple[float, ...], ndim: int, return_channel: Optional[int]) -> None:
    """Apply the filter to the whole array at once (no blocking, no halo)."""
    data = src[full_roi(ndim)]
    response = _FILTER_FUNCTIONS[filter_name](data, sigma, *extra_args)
    if return_channel is not None:
        response = response[..., return_channel]
    if response.ndim > ndim:
        out[full_roi(out.ndim)] = np.moveaxis(response, -1, 0)
    else:
        out[full_roi(ndim)] = response


def apply_filter(
    input: SourceLike,
    filter_name: str,
    sigma: Sigma,
    output: Optional[SourceLike] = None,
    *,
    outer_scale: Optional[float] = None,
    return_channel: Optional[int] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> SourceLike:
    """Apply a (gaussian-family) filter block-wise.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        filter_name: The filter to apply: one of ``"gaussian_smoothing"``,
            ``"gaussian_gradient_magnitude"``, ``"laplacian_of_gaussian"``,
            ``"hessian_of_gaussian_eigenvalues"`` or ``"structure_tensor_eigenvalues"``.
        sigma: The filter standard deviation: a scalar (isotropic) or a per-axis sequence
            of floats (anisotropic).
        output: The output array to write into. Optional for local execution — a numpy array
            is allocated and returned if omitted; **required** for distributed execution. For a
            multi-channel response the output has a leading channel axis of shape ``(ndim, ...)``.
        outer_scale: Outer scale, required for ``structure_tensor_eigenvalues``.
        return_channel: For multi-channel filters, select a single channel (scalar output).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; values outside the mask are excluded from the computation
            (out-of-mask output voxels are left unchanged).
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks
            into the existing ``output``). Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``); the missing blocks are written into ``output``. Mutually exclusive
            with ``block_ids``.

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    if filter_name not in _FILTER_FUNCTIONS:
        raise ValueError(f"Unknown filter {filter_name!r}; valid filters: {sorted(_FILTER_FUNCTIONS)}.")
    if filter_name == "structure_tensor_eigenvalues" and outer_scale is None:
        raise ValueError("structure_tensor_eigenvalues requires 'outer_scale'.")
    check_rerun_args(job_type, resume_from, block_ids)

    src = as_source(input)
    ndim = src.ndim
    order = _ORDERS[filter_name]
    extra_args: Tuple[float, ...] = (
        (float(outer_scale),) if filter_name == "structure_tensor_eigenvalues" else ()
    )
    multi_channel = filter_name in _MULTI_CHANNEL and return_channel is None
    # A subset/resume rerun is inherently block-wise, so it cannot use the direct (whole-array) path.
    direct = (is_direct(job_type, num_workers, block_shape) and mask is None
              and block_ids is None and resume_from is None)

    if output is None:
        if job_type != "local":
            raise ValueError(
                f"'output' is required for distributed execution (job_type={job_type!r}); "
                "pass a file-backed (zarr/n5) output array."
            )
        out_array: SourceLike = _allocate_output(src, ndim, multi_channel)
    else:
        out_array = output

    out = as_source(out_array)
    if not direct and _same_array(out, src):
        raise ValueError("Block-wise filtering needs 'output' to differ from 'input'.")

    if direct:
        _apply_direct(src, out, filter_name, sigma, extra_args, ndim, return_channel)
        return out_array

    compute = _make_compute(filter_name, sigma, extra_args, ndim, return_channel)
    halo = normalize_halo(sigma_to_halo(sigma, order), ndim)
    runner = get_runner(job_type, job_config)
    runner.run(compute, [input], outputs=[out], halo=halo, block_shape=block_shape,
               mask=mask, num_workers=num_workers, block_ids=block_ids, resume_from=resume_from,
               name=filter_name)
    return out_array


def gaussian_smoothing(input: SourceLike, sigma: Sigma, output: Optional[SourceLike] = None,
                       **kwargs) -> SourceLike:
    """Gaussian smoothing, block-wise.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        sigma: The filter standard deviation: a scalar (isotropic) or a per-axis sequence
            of floats (anisotropic).
        output: The output array to write into. Optional for local execution — a numpy array
            is allocated and returned if omitted; **required** for distributed execution.
        **kwargs: Additional keyword arguments forwarded to :func:`apply_filter` (``block_shape``,
            ``job_type``, ``job_config``, ``num_workers``, ``mask``, ``block_ids``, ``resume_from``).

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    return apply_filter(input, "gaussian_smoothing", sigma, output, **kwargs)


def gaussian_gradient_magnitude(input: SourceLike, sigma: Sigma,
                                output: Optional[SourceLike] = None, **kwargs) -> SourceLike:
    """Gaussian gradient magnitude, block-wise.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        sigma: The filter standard deviation: a scalar (isotropic) or a per-axis sequence
            of floats (anisotropic).
        output: The output array to write into. Optional for local execution — a numpy array
            is allocated and returned if omitted; **required** for distributed execution.
        **kwargs: Additional keyword arguments forwarded to :func:`apply_filter` (``block_shape``,
            ``job_type``, ``job_config``, ``num_workers``, ``mask``, ``block_ids``, ``resume_from``).

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    return apply_filter(input, "gaussian_gradient_magnitude", sigma, output, **kwargs)


def laplacian_of_gaussian(input: SourceLike, sigma: Sigma, output: Optional[SourceLike] = None,
                          **kwargs) -> SourceLike:
    """Laplacian of Gaussian, block-wise.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        sigma: The filter standard deviation: a scalar (isotropic) or a per-axis sequence
            of floats (anisotropic).
        output: The output array to write into. Optional for local execution — a numpy array
            is allocated and returned if omitted; **required** for distributed execution.
        **kwargs: Additional keyword arguments forwarded to :func:`apply_filter` (``block_shape``,
            ``job_type``, ``job_config``, ``num_workers``, ``mask``, ``block_ids``, ``resume_from``).

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    return apply_filter(input, "laplacian_of_gaussian", sigma, output, **kwargs)
