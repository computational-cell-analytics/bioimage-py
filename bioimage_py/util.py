"""Shared helpers: block-to-roi conversion, blocking construction and filter halos."""
from __future__ import annotations

import numbers
from math import ceil
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import bioimage_cpp as bic
from bioimage_cpp.utils import Block, BlockWithHalo, Blocking

from .sources.base import Source

# A per-block descriptor handed to compute functions: a plain ``Block`` (no halo) or a
# ``BlockWithHalo`` (halo operations).
BlockDescriptor = Union[Block, BlockWithHalo]

# Signature of a per-block compute function: ``function(block, inputs, outputs, mask)``.
ComputeFn = Callable[
    [BlockDescriptor, Sequence[Source], Sequence[Source], Optional[Source]], Any
]


def to_roi(block: BlockDescriptor) -> Tuple[slice, ...]:
    """Convert a ``bioimage_cpp.utils`` ``Block`` into a tuple of slices.

    Args:
        block: A ``Block`` (carrying ``begin``/``end`` coordinate lists). For halo
            operations pass one of ``block.outer_block`` / ``block.inner_block`` /
            ``block.inner_block_local``.

    Returns:
        A tuple of slices that indexes a source or array.
    """
    return tuple(slice(int(b), int(e)) for b, e in zip(block.begin, block.end))


def normalize_halo(halo: Union[int, Sequence[int]], ndim: int) -> List[int]:
    """Broadcast a halo to a per-axis list of length ``ndim``."""
    if isinstance(halo, numbers.Integral):
        return [int(halo)] * ndim
    halo = [int(h) for h in halo]
    if len(halo) != ndim:
        raise ValueError(f"Halo {halo} does not match ndim {ndim}.")
    return halo


def sigma_to_halo(sigma: Union[float, Sequence[float]], order: int) -> Union[int, List[int]]:
    """Compute the halo for applying an image filter block-wise.

    Mirrors elf's implementation, based on VIGRA's ``multi_blockwise.hxx``.

    Args:
        sigma: The sigma value(s) of the filter.
        order: The derivative order of the filter (0 for smoothing).

    Returns:
        The halo, as an int for scalar sigma or a per-axis list for sequence sigma.
    """
    multiplier = 2
    if isinstance(sigma, numbers.Number):
        return multiplier * int(ceil(3.0 * sigma + 0.5 * order + 0.5))
    return [multiplier * int(ceil(3.0 * sig + 0.5 * order + 0.5)) for sig in sigma]


def downscale_shape(shape: Sequence[int], scale_factor: Union[int, Sequence[int]],
                    ceil_mode: bool = True) -> Tuple[int, ...]:
    """Compute the shape resulting from downscaling by an integer factor.

    Mirrors elf's ``downscale_shape``.

    Args:
        shape: The input array shape.
        scale_factor: The downscaling factor: a single int (isotropic) or a per-axis sequence.
        ceil_mode: Whether to round the downscaled size up (so no input voxel is dropped) or
            down (strict integer division).

    Returns:
        The downscaled shape.

    Raises:
        ValueError: If a per-axis ``scale_factor`` does not match the dimensionality of ``shape``.
    """
    if isinstance(scale_factor, numbers.Integral):
        factors = [int(scale_factor)] * len(shape)
    else:
        factors = [int(f) for f in scale_factor]
        if len(factors) != len(shape):
            raise ValueError(
                f"scale_factor {scale_factor} does not match the dimensionality {len(shape)}."
            )
    if ceil_mode:
        return tuple(int(s) // f + int((int(s) % f) != 0) for s, f in zip(shape, factors))
    return tuple(int(s) // f for s, f in zip(shape, factors))


def derive_block_shape(source: Source, block_shape: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Resolve the block shape, falling back to the source's chunks.

    Args:
        source: A source exposing ``shape`` and ``chunks``.
        block_shape: The explicit block shape, or ``None`` to derive it from chunks.

    Returns:
        The resolved block shape.

    Raises:
        ValueError: If ``block_shape`` is ``None`` and the source is unchunked.
    """
    if block_shape is not None:
        return tuple(int(b) for b in block_shape)
    chunks = source.chunks
    if chunks is not None:
        return tuple(int(c) for c in chunks)
    raise ValueError(
        "block_shape is required for block-wise processing of an unchunked array "
        "(the source has no chunks to derive it from)."
    )


def get_blocking(shape: Sequence[int], block_shape: Sequence[int],
                 roi: Optional[Tuple[slice, ...]] = None) -> Blocking:
    """Build a ``bioimage_cpp.utils.Blocking`` over ``shape`` (or a sub-roi).

    Args:
        shape: The full array shape.
        block_shape: The block shape.
        roi: Optional region of interest to restrict the blocking to.

    Returns:
        A ``bioimage_cpp.utils.Blocking`` instance.
    """
    ndim = len(shape)
    if roi is None:
        roi_begin = [0] * ndim
        roi_end = [int(s) for s in shape]
    else:
        roi_begin = [int(sl.start) if sl.start is not None else 0 for sl in roi]
        roi_end = [int(sl.stop) if sl.stop is not None else int(s) for sl, s in zip(roi, shape)]
    return bic.utils.Blocking(roi_begin, roi_end, [int(b) for b in block_shape])
