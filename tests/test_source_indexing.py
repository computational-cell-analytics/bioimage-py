"""Tests for numpy-style basic indexing (scalars / ellipsis / partial tuples) on Sources."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources import as_source
from bioimage_py.wrapper import AffineSource, PadSource, ResizedSource, RoiSource, ThresholdSource


def _check_basic_indexing(src, ref):
    """Assert a Source reproduces numpy basic-indexing semantics against a reference array."""
    # Integer index squeezes the axis; equivalent to the explicit full-tuple form.
    np.testing.assert_array_equal(src[1], ref[1])
    np.testing.assert_array_equal(src[1], src[1, :, :])
    # Ellipsis and trailing integer.
    np.testing.assert_array_equal(src[..., 0], ref[..., 0])
    np.testing.assert_array_equal(src[0, ...], ref[0])
    # Short tuple pads the remaining axes.
    np.testing.assert_array_equal(src[0:1], ref[0:1])
    np.testing.assert_array_equal(src[1, 0:2], ref[1, 0:2])
    # Negative index.
    np.testing.assert_array_equal(src[-1], ref[-1])
    # Full tuple of slices is unchanged (the runner hot path).
    np.testing.assert_array_equal(src[(slice(None), slice(None), slice(None))], ref)


def test_array_source_numpy_basic_indexing(rng):
    a = rng.random((4, 5, 6)).astype("float32")
    _check_basic_indexing(as_source(a), a)


def test_array_source_zarr_basic_indexing(zarr_factory, rng):
    a = rng.random((4, 5, 6)).astype("float32")
    z = zarr_factory(a, chunks=(2, 5, 6))
    _check_basic_indexing(as_source(z), a)


def test_file_source_basic_indexing(n5_factory, rng):
    a = rng.random((4, 5, 6)).astype("float32")
    path, key = n5_factory(a, chunks=(2, 5, 6))
    _check_basic_indexing(bp.open_source(path, key), a)


def test_full_scalar_index_returns_scalar(rng):
    a = rng.random((4, 5, 6)).astype("float32")
    src = as_source(a)
    value = src[1, 2, 3]
    assert np.ndim(value) == 0
    assert float(value) == float(a[1, 2, 3])


def test_out_of_bounds_integer_raises(rng):
    src = as_source(rng.random((4, 5)).astype("float32"))
    with pytest.raises(ValueError):
        src[10]


def test_wrapper_scalar_indexing_threshold(rng):
    a = rng.random((4, 5, 6)).astype("float32")
    src = ThresholdSource(a, 0.5)
    np.testing.assert_array_equal(src[2], a[2] > 0.5)
    np.testing.assert_array_equal(src[..., 0], (a > 0.5)[..., 0])


@pytest.mark.parametrize("build,ref_shape", [
    (lambda a: ResizedSource(a, (4, 5, 6)), (5, 6)),
    (lambda a: AffineSource(a, affine_matrix=np.eye(4), order=1), (5, 6)),
    (lambda a: PadSource(a, (2, 0, 0)), (5, 6)),
])
def test_hand_written_wrappers_accept_scalar(rng, build, ref_shape):
    # These wrappers read sl.start/sl.stop and used to raise AttributeError on a bare int.
    a = rng.random((4, 5, 6)).astype("float32")
    src = build(a)
    plane = src[1]
    assert plane.shape == ref_shape


def test_roi_source_scalar_indexing(rng):
    a = rng.random((6, 5, 4)).astype("float32")
    src = RoiSource(a, (slice(1, 5), slice(None), slice(None)))  # shape (4, 5, 4)
    # A scalar request indexes within the roi and squeezes.
    np.testing.assert_array_equal(src[0], a[1])
    np.testing.assert_array_equal(src[-1], a[4])


def test_array_source_scalar_setitem(zarr_factory, rng):
    z = zarr_factory(np.zeros((4, 5, 6), dtype="float32"), chunks=(2, 5, 6))
    src = as_source(z)
    plane = rng.random((5, 6)).astype("float32")
    src[2] = plane  # scalar index on write -> axis re-inserted to match the singleton slice
    np.testing.assert_array_equal(z[:][2], plane)
    assert np.all(z[:][0] == 0.0)


def test_roi_source_scalar_setitem_passthrough(zarr_factory, rng):
    z = zarr_factory(np.zeros((6, 5, 4), dtype="float32"), chunks=(2, 5, 4))
    src = RoiSource(z, (slice(1, 5), slice(None), slice(None)))
    plane = rng.random((5, 4)).astype("float32")
    src[0] = plane
    np.testing.assert_array_equal(z[:][1], plane)
