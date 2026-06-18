"""Block-wise downsampling of a source.

Downsampling wraps the input in a :class:`~bioimage_py.wrapper.ResizedSource` at the downscaled
target shape and copies it block-wise into the output, reusing the copy machinery. Because the
wrapper resamples on read, this is exactly a block-wise copy of the resized source.
"""
from __future__ import annotations

from numbers import Integral
from typing import Optional, Sequence, Tuple, Union

from .copy import _copy_source
from .runner.config import RunnerConfig
from .sources import SourceLike, as_source
from .util import downscale_shape
from .wrapper import ResizedSource

__all__ = ["downsample"]

# A downscaling factor: a single int (isotropic) or a per-axis sequence of ints.
ScaleFactor = Union[int, Sequence[int]]


def downsample(
    input: SourceLike,
    scale_factor: ScaleFactor,
    output: Optional[SourceLike] = None,
    *,
    order: int = 0,
    anti_aliasing: bool = False,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
) -> SourceLike:
    """Downsample a source by an integer factor, block-wise.

    The input is wrapped in a :class:`~bioimage_py.wrapper.ResizedSource` at the downscaled shape
    (computed with :func:`~bioimage_py.util.downscale_shape`, ceil mode) and copied into the output.

    The defaults (``order=0`` nearest, ``anti_aliasing=False``) are label-safe — they preserve the
    input values and are appropriate for segmentations. For intensity / image data pass ``order=1``
    (or higher) and ``anti_aliasing=True`` for a smooth, alias-free downsample.

    Args:
        input: The input data to downsample (a numpy/zarr/n5 array or a `Source`). 2D or 3D.
        scale_factor: The downscaling factor: a single int (isotropic) or a per-axis sequence of
            ints. Each factor must be ``>= 1`` (1 leaves that axis unchanged).
        output: The output array to write into. Optional for local execution — a numpy array of the
            downscaled shape and the input dtype is allocated and returned if omitted; **required**
            for distributed execution, where it (and the input) must be file-backed (zarr/n5).
        order: The interpolation order (0 to 5). Use ``0`` (nearest) for label data.
        anti_aliasing: Whether to Gaussian pre-smooth before sampling to avoid aliasing.
            Recommended for image data; leave ``False`` for labels.
        block_shape: Shape of the processing blocks (in the downscaled output space). Defaults to
            the resized source's chunk shape; required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask in the downscaled output space; only voxels within the mask are
            written (out-of-mask output voxels are left unchanged).

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array).
    """
    src = as_source(input)
    ndim = src.ndim
    if isinstance(scale_factor, Integral):
        factors: Tuple[int, ...] = (int(scale_factor),) * ndim
    else:
        factors = tuple(int(f) for f in scale_factor)
        if len(factors) != ndim:
            raise ValueError(
                f"scale_factor {scale_factor} does not match the input dimensionality {ndim}."
            )
    if any(f < 1 for f in factors):
        raise ValueError(
            f"downsample requires scale factors >= 1, got {factors}; "
            "use a ResizedSource directly to upsample."
        )

    target_shape = downscale_shape(src.shape, factors)
    wrapped = ResizedSource(src, target_shape, order=order, anti_aliasing=anti_aliasing)
    return _copy_source(wrapped, output, block_shape=block_shape, job_type=job_type,
                        job_config=job_config, num_workers=num_workers, mask=mask, name="downsample")
