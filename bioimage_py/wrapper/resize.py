"""A wrapper source that resizes (resamples) the wrapped source on read.

Reading a region of the resized (output) space lazily reads the corresponding region of the
wrapped (input) space — extended by an interpolation / anti-aliasing halo so that block-wise reads
are seam-free — and resamples it. The resize is an affine transformation with a diagonal scale
matrix mapping output coordinates to input coordinates; the interpolation is delegated to
``bioimage_cpp.transformation`` (2D and 3D only). Mirrors elf's ``ResizedVolume``.
"""
from __future__ import annotations

from math import ceil, floor
from typing import Optional, Tuple

import bioimage_cpp as bic
import numpy as np

from ..sources.dispatch import SourceLike
from ..util import sigma_to_halo
from .base import WrapperSource, register_wrapper


@register_wrapper
class ResizedSource(WrapperSource):
    """Resize the wrapped source to a target shape on read.

    Args:
        source: The wrapped source-like object (2D or 3D).
        shape: The target shape for the resized source.
        order: The interpolation order, supports orders 0 to 5 (see
            ``bioimage_cpp.transformation.affine_transform``). Use ``0`` (nearest) for label data.
        anti_aliasing: Whether to Gaussian pre-smooth the input before sampling to avoid aliasing
            when downsampling. Recommended for intensity image data; leave ``False`` for labels.
        fill_value: The value used for output coordinates that map outside the input.
    """

    def __init__(self, source: SourceLike, shape: Tuple[int, ...], *, order: int = 0,
                 anti_aliasing: bool = False, fill_value: float = 0) -> None:
        super().__init__(source)
        src_shape = self._source.shape
        if len(shape) != len(src_shape):
            raise ValueError(
                f"shape {tuple(shape)} must match the wrapped source dimensionality {len(src_shape)}."
            )
        if len(src_shape) not in (2, 3):
            raise ValueError(f"ResizedSource supports 2d or 3d data, got {len(src_shape)}d.")
        if not 0 <= int(order) <= 5:
            raise ValueError(f"order must be in [0, 5], got {order}.")
        self._shape = tuple(int(s) for s in shape)
        self._order = int(order)
        self._anti_aliasing = bool(anti_aliasing)
        self._fill_value = fill_value
        # Per-axis scale and the homogeneous affine matrix mapping output -> input coordinates.
        self._scale = [ish / float(osh) for ish, osh in zip(src_shape, self._shape)]
        self._matrix = np.diag(self._scale + [1.0])
        self._is_bool = np.dtype(self._source.dtype) == np.dtype(bool)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The target (resized) shape."""
        return self._shape

    @property
    def scale(self) -> Tuple[float, ...]:
        """The per-axis scale factors (input size / output size)."""
        return tuple(self._scale)

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The wrapped chunks scaled into the output space, or ``None`` if unchunked."""
        src_chunks = self._source.chunks
        if src_chunks is None:
            return None
        return tuple(
            max(1, int(ceil(c * osh / float(ish))))
            for c, ish, osh in zip(src_chunks, self._source.shape, self._shape)
        )

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        """Resized sources are unsharded (input-space shards do not map to the output)."""
        return None

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the resized data for the output region ``roi``."""
        ndim = len(self._shape)
        out_start = [int(sl.start) if sl.start is not None else 0 for sl in roi]
        out_stop = [int(sl.stop) if sl.stop is not None else self._shape[i]
                    for i, sl in enumerate(roi)]
        src_shape = self._source.shape

        # The input region this output region samples from (exact for a diagonal scale matrix).
        in_start_f = [self._scale[i] * out_start[i] for i in range(ndim)]
        in_stop_f = [self._scale[i] * out_stop[i] for i in range(ndim)]

        # Anti-aliasing sigma (per input axis), used to pre-smooth and to size the read halo.
        sigma = None
        if self._anti_aliasing:
            aa = np.asarray(bic.transformation.compute_anti_aliasing_sigma(self._matrix, ndim),
                            dtype="float64")
            if np.any(aa > 0):
                sigma = aa

        # Read halo: interpolation taps (order + 1) plus the smoothing extent.
        halo = [self._order + 1] * ndim
        if sigma is not None:
            halo = [h + sigma_to_halo(float(s), self._order) for h, s in zip(halo, sigma)]

        in_start = [max(0, int(floor(s)) - h) for s, h in zip(in_start_f, halo)]
        in_stop = [min(int(ish), int(ceil(s)) + h) for s, h, ish in zip(in_stop_f, halo, src_shape)]
        in_bb = tuple(slice(sta, sto) for sta, sto in zip(in_start, in_stop))
        in_region = np.asarray(self._source[in_bb])
        if self._is_bool:
            in_region = in_region.astype("uint8")

        # Shift the affine into the local frames: local output coords -> local input coords.
        local_matrix = self._matrix.copy()
        local_matrix[:ndim, ndim] = [self._scale[i] * out_start[i] - in_start[i]
                                     for i in range(ndim)]
        local_bb = tuple(slice(0, sto - sta) for sta, sto in zip(out_start, out_stop))

        if sigma is None:
            res = bic.transformation.affine_transform(
                in_region, local_matrix, bounding_box=local_bb, order=self._order,
                fill_value=self._fill_value,
            )
        else:
            res = bic.transformation.resample(
                in_region, local_matrix, bounding_box=local_bb, order=self._order,
                fill_value=self._fill_value, anti_aliasing_sigma=sigma,
            )

        if self._is_bool:
            res = res.astype(bool)
        return res

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {
            "shape": self._shape,
            "order": self._order,
            "anti_aliasing": self._anti_aliasing,
            "fill_value": self._fill_value,
        }
