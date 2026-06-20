"""Affine transformation helpers: matrix composition and sub-volume resampling.

This mirrors ``elf.transformation.affine``, adapted to the :class:`~bioimage_py.sources.Source`
model (a source is indexed only with a tuple of slices). :func:`compute_affine_matrix` composes a
2D or 3D affine matrix from scale / rotation / shear / translation parameters; bioimage-cpp provides
only the matrix-based sampling primitives, not this composition. :func:`transform_subvolume_affine`
resamples the region of a (possibly chunked) source that an output region maps to, delegating the
interpolation to ``bioimage_cpp.transformation``.
"""
from __future__ import annotations

from functools import partial
from itertools import product
from numbers import Number
from typing import List, Optional, Sequence, Tuple

import bioimage_cpp as bic
import numpy as np

from .sources.dispatch import SourceLike, as_source
from .util import sigma_to_halo


def _update_parameters(scale, rotation, shear, translation, dim):
    """Fill in identity defaults for any unset affine parameters."""
    if scale is None:
        scale = [1.0] * dim
    if rotation is None:
        rotation = 0.0 if dim == 2 else [0.0] * 3
    if shear is None:
        shear = 0.0 if dim == 2 else [0.0] * 3
    if translation is None:
        translation = [0.0] * dim
    return scale, rotation, shear, translation


def _affine_matrix_2d(scale=None, rotation=None, shear=None, translation=None, angles_in_degree=True):
    """Compose a 3x3 homogeneous affine matrix from 2D parameters."""
    matrix = np.zeros((3, 3))
    scale, rotation, shear, translation = _update_parameters(scale, rotation, shear, translation, dim=2)

    # Wrapper for numpy behaviour that returns a 0-d array instead of a python scalar.
    def np_wrap(x, func):
        ret = func(x)
        if hasattr(ret, "item"):
            ret = ret.item()
        return ret

    cos, sin = partial(np_wrap, func=np.cos), partial(np_wrap, func=np.sin)
    sx, sy = scale

    if angles_in_degree:
        phi = np.deg2rad(rotation)
        shear_angle = np.deg2rad(shear)
    else:
        phi = rotation
        shear_angle = shear

    matrix[0, 0] = sx * cos(phi)
    matrix[0, 1] = - sy * sin(phi + shear_angle)
    matrix[1, 0] = sx * sin(phi)
    matrix[1, 1] = sy * cos(phi + shear_angle)
    matrix[:2, 2] = translation
    matrix[2, 2] = 1
    return matrix


def _affine_matrix_3d(scale=None, rotation=None, shear=None, translation=None, angles_in_degree=True):
    """Compose a 4x4 homogeneous affine matrix from 3D parameters (shear not yet supported)."""
    matrix = np.zeros((4, 4))
    scale, rotation, shear, translation = _update_parameters(scale, rotation, shear, translation, dim=3)

    cos, sin = np.cos, np.sin
    sx, sy, sz = scale
    if angles_in_degree:
        phi, theta, psi = np.deg2rad(rotation)
    else:
        phi, theta, psi = rotation

    matrix[0, 0] = sx * cos(theta) * cos(psi)
    matrix[0, 1] = sy * (-cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi))
    matrix[0, 2] = sz * (sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi))
    matrix[1, 0] = sx * cos(theta) * sin(psi)
    matrix[1, 1] = sy * (cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi))
    matrix[1, 2] = sz * (- sin(phi) * cos(theta) + cos(phi) * sin(theta) * sin(psi))
    matrix[2, 0] = -sx * sin(theta)
    matrix[2, 1] = sy * sin(phi) * sin(theta)
    matrix[2, 2] = sz * cos(phi) * cos(theta)
    matrix[:3, 3] = translation
    matrix[3, 3] = 1
    return matrix


def compute_affine_matrix(
    scale: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    shear: Optional[List[float]] = None,
    translation: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute a 2D or 3D affine matrix from individual parameters.

    Args:
        scale: Scaling factors for the dimensions, must have length 2 for 2D / 3 for 3D.
        rotation: Rotation, a single angle in 2D, three euler angles (phi, theta, psi) in 3D,
            given in degrees.
        shear: Shear angle. This is not implemented correctly yet and should be left unset.
        translation: Translation along the dimensions, must have length 2 for 2D / 3 for 3D.

    Returns:
        The homogeneous affine matrix, of shape ``(ndim + 1, ndim + 1)``.
    """
    parameters = [scale, rotation, shear, translation]
    if all(param is None for param in parameters):
        raise ValueError("At least one of scale, rotation, shear or translation must be given.")

    # scale and translation have length == dimension; rotation and shear have length 1 (2D) or 3 (3D).
    lens = [None if param is None else len(param) for param in parameters]
    dims = [ll if ii in (0, 3) else 2 if ll == 1 else 3 for ii, ll in enumerate(lens) if ll is not None]
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent dimensionality across affine parameters: {dims}.")
    dim = dims[0]

    return _affine_matrix_2d(scale, rotation, shear, translation) if dim == 2 else \
        _affine_matrix_3d(scale, rotation, shear, translation)


def transform_coordinate(coord: Sequence[float], matrix: np.ndarray) -> Tuple[float, ...]:
    """Apply an affine matrix to a single coordinate."""
    ndim = len(coord)
    return tuple(sum(coord[jj] * matrix[ii, jj] for jj in range(ndim)) + matrix[ii, -1] for ii in range(ndim))


def transform_roi_with_affine(
    roi_start: Sequence[float], roi_stop: Sequence[float], matrix: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Transform a region of interest with an affine matrix via its corners.

    Args:
        roi_start: The start (lower-left corner) of the region of interest.
        roi_stop: The stop (upper-right corner) of the region of interest.
        matrix: The affine matrix.

    Returns:
        The transformed start coordinates of the ROI.
        The transformed stop coordinates of the ROI.
    """
    dim = len(roi_start)
    corners = [corner for corner in product(*zip(roi_start, roi_stop))]
    transformed_corners = [transform_coordinate(corner, matrix) for corner in corners]
    transformed_start = [min(corner[d] for corner in transformed_corners) for d in range(dim)]
    transformed_stop = [max(corner[d] for corner in transformed_corners) for d in range(dim)]
    return transformed_start, transformed_stop


def transform_subvolume_affine(
    source: SourceLike,
    matrix: np.ndarray,
    roi: Tuple[slice, ...],
    order: int = 0,
    fill_value: Number = 0,
    sigma: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Apply an affine transformation to the output region ``roi`` of a source.

    The output region ``roi`` samples from a corresponding region of the input, found by mapping
    the output corners through ``matrix`` (which maps output to input coordinates). That input
    region is read once - extended by an interpolation halo (and a smoothing halo when ``sigma`` is
    given) so block-wise reads are seam-free - and resampled in memory. This generalises the
    diagonal-scale logic in :class:`~bioimage_py.wrapper.resize.ResizedSource` to an arbitrary affine.

    Args:
        source: The input source-like object (2D or 3D).
        matrix: The homogeneous affine matrix mapping output to input coordinates in numpy axis order.
        roi: The output region of interest, as a tuple of slices.
        order: The interpolation order, supports orders 0 to 5.
        fill_value: The value used for output coordinates that map outside the input.
        sigma: An optional per-input-axis Gaussian sigma used to pre-smooth the input (anti-aliasing).

    Returns:
        The transformed output region.
    """
    source = as_source(source)
    src_shape = source.shape
    ndim = len(src_shape)
    matrix = np.asarray(matrix, dtype="float64")
    linear = matrix[:ndim, :ndim]
    translation = matrix[:ndim, ndim]

    out_start = [int(sl.start) if sl.start is not None else 0 for sl in roi]
    out_stop = [int(sl.stop) if sl.stop is not None else src_shape[i] for i, sl in enumerate(roi)]

    # The input region this output region samples from (corner-mapped through the affine).
    in_start_f, in_stop_f = transform_roi_with_affine(out_start, out_stop, matrix)

    # Read halo: interpolation taps (order + 1) plus the smoothing extent when pre-smoothing.
    halo = [order + 1] * ndim
    if sigma is not None:
        halo = [h + sigma_to_halo(float(s), order) for h, s in zip(halo, sigma)]

    in_start = [max(0, int(np.floor(s)) - h) for s, h in zip(in_start_f, halo)]
    in_stop = [min(int(sh), int(np.ceil(s)) + h) for s, h, sh in zip(in_stop_f, halo, src_shape)]

    out_shape = tuple(sto - sta for sta, sto in zip(out_start, out_stop))
    if any(sto <= sta for sta, sto in zip(in_start, in_stop)):
        # The output region maps entirely outside the input; return the fill value.
        return np.full(out_shape, fill_value, dtype=source.dtype)

    in_bb = tuple(slice(sta, sto) for sta, sto in zip(in_start, in_stop))
    in_region = np.asarray(source[in_bb])
    is_bool = np.dtype(source.dtype) == np.dtype(bool)
    if is_bool:
        in_region = in_region.astype("uint8")

    # Shift the matrix into the local frames: local output coords -> local input coords.
    local_matrix = matrix.copy()
    local_matrix[:ndim, ndim] = linear @ np.array(out_start, dtype="float64") + translation - in_start
    local_bb = tuple(slice(0, sh) for sh in out_shape)

    if sigma is None:
        res = bic.transformation.affine_transform(
            in_region, local_matrix, bounding_box=local_bb, order=order, fill_value=fill_value,
        )
    else:
        res = bic.transformation.resample(
            in_region, local_matrix, bounding_box=local_bb, order=order,
            fill_value=fill_value, anti_aliasing_sigma=np.asarray(sigma, dtype="float64"),
        )

    if is_bool:
        res = res.astype(bool)
    return res
