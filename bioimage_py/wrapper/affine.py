"""A wrapper source that applies an affine transformation on read.

Reading an output region lazily reads the corresponding region of the wrapped (input) space -
extended by an interpolation / anti-aliasing halo so block-wise reads are seam-free - and resamples
it. The transformation is defined either by an explicit affine matrix (mapping output to input
coordinates) or by scale / rotation / shear / translation parameters. The interpolation is delegated
to :func:`bioimage_py.transformation.transform_subvolume_affine` (2D and 3D only). Mirrors elf's
``AffineVolume``.
"""
from __future__ import annotations

from numbers import Number
from typing import List, Optional, Tuple

import bioimage_cpp as bic
import numpy as np

from ..sources.dispatch import SourceLike
from ..transformation import compute_affine_matrix, transform_subvolume_affine
from .base import WrapperSource, register_wrapper


@register_wrapper
class AffineSource(WrapperSource):
    """Apply an affine transformation to the wrapped source on read.

    The transformation is given either by ``affine_matrix`` or by individual parameters for
    ``scale`` / ``rotation`` / ``shear`` / ``translation`` (exactly one of the two ways).

    Args:
        source: The wrapped source-like object (2D or 3D).
        shape: The output shape. Defaults to the wrapped source's shape.
        affine_matrix: The matrix defining the affine transformation, mapping output to input
            coordinates, of shape ``(ndim + 1, ndim + 1)``.
        scale: The scale factors (used when ``affine_matrix`` is not given).
        rotation: The rotation angles in degrees (used when ``affine_matrix`` is not given).
        shear: The shear angles in degrees (not implemented correctly yet; leave unset).
        translation: The translation vector (used when ``affine_matrix`` is not given).
        order: The interpolation order, supports orders 0 to 5. Use ``0`` (nearest) for label data.
        fill_value: The value used for output coordinates that map outside the input.
        anti_aliasing: Whether to Gaussian pre-smooth the input before sampling to avoid aliasing
            when downsampling. Recommended for intensity data; leave ``False`` for labels.
    """

    def __init__(
        self,
        source: SourceLike,
        shape: Optional[Tuple[int, ...]] = None,
        *,
        affine_matrix: Optional[np.ndarray] = None,
        scale: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        shear: Optional[List[float]] = None,
        translation: Optional[List[float]] = None,
        order: int = 0,
        fill_value: Number = 0,
        anti_aliasing: bool = False,
    ) -> None:
        super().__init__(source)
        ndim = self._source.ndim
        if ndim not in (2, 3):
            raise ValueError(f"AffineSource supports 2d or 3d data, got {ndim}d.")
        if not 0 <= int(order) <= 5:
            raise ValueError(f"order must be in [0, 5], got {order}.")

        have_matrix = affine_matrix is not None
        have_parameter = any(p is not None for p in (scale, rotation, shear, translation))
        if have_matrix == have_parameter:
            raise ValueError("Pass exactly one of affine_matrix or the scale/rotation/shear/translation parameters.")

        if have_matrix:
            matrix = np.asarray(affine_matrix, dtype="float64")
        else:
            matrix = compute_affine_matrix(scale, rotation, shear, translation)
        if matrix.shape != (ndim + 1, ndim + 1):
            raise ValueError(f"Invalid affine matrix shape {matrix.shape}, expected {(ndim + 1, ndim + 1)}.")

        self._matrix = matrix
        self._shape = self._source.shape if shape is None else tuple(int(s) for s in shape)
        self._order = int(order)
        self._fill_value = fill_value
        self._anti_aliasing = bool(anti_aliasing)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The output (transformed) shape."""
        return self._shape

    @property
    def matrix(self) -> np.ndarray:
        """The affine matrix mapping output to input coordinates."""
        return self._matrix

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        # An affine of the input does not map onto the input's shard grid.
        return None

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the affine-transformed data for the output region ``roi``."""
        sigma = None
        if self._anti_aliasing:
            aa = np.asarray(
                bic.transformation.compute_anti_aliasing_sigma(self._matrix, self._source.ndim),
                dtype="float64",
            )
            if np.any(aa > 0):
                sigma = aa
        return transform_subvolume_affine(
            self._source, self._matrix, roi, order=self._order,
            fill_value=self._fill_value, sigma=sigma,
        )

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {
            "shape": self._shape,
            "affine_matrix": self._matrix.tolist(),
            "order": self._order,
            "fill_value": self._fill_value,
            "anti_aliasing": self._anti_aliasing,
        }
