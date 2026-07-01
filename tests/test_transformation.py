"""Tests for the affine matrix and sub-volume helpers in bioimage_py.transformation."""
import bioimage_cpp as bic
import numpy as np
import pytest

from bioimage_py.transformation import (
    compute_affine_matrix,
    transform_roi_with_affine,
    transform_subvolume_affine,
    transform_subvolume_coordinates,
)


def test_compute_affine_matrix_scale_2d():
    matrix = compute_affine_matrix(scale=[2.0, 3.0])
    expected = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_compute_affine_matrix_rotation_2d():
    # A 90 degree rotation maps (x, y) -> (-y, x) in matrix form.
    matrix = compute_affine_matrix(rotation=[90.0])
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_compute_affine_matrix_translation_3d():
    matrix = compute_affine_matrix(translation=[1.0, 2.0, 3.0])
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_compute_affine_matrix_scale_3d():
    matrix = compute_affine_matrix(scale=[2.0, 2.0, 2.0])
    np.testing.assert_allclose(matrix, np.diag([2.0, 2.0, 2.0, 1.0]), atol=1e-12)


def test_compute_affine_matrix_requires_parameter():
    with pytest.raises(ValueError, match="At least one"):
        compute_affine_matrix()


def test_transform_roi_with_affine_translation():
    start, stop = transform_roi_with_affine([0.0, 0.0], [10.0, 20.0],
                                            compute_affine_matrix(translation=[5.0, -3.0]))
    np.testing.assert_allclose(start, [5.0, -3.0])
    np.testing.assert_allclose(stop, [15.0, 17.0])


def test_transform_subvolume_affine_matches_reference(rng):
    a = rng.random((40, 48)).astype("float32")
    matrix = compute_affine_matrix(rotation=[15.0], translation=[3.0, 2.0])
    roi = (slice(0, 40), slice(0, 48))
    out = transform_subvolume_affine(a, matrix, roi, order=1, fill_value=0)
    ref = bic.transformation.affine_transform(a, matrix, order=1, fill_value=0)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_transform_subvolume_affine_subregion(rng):
    a = rng.random((40, 48)).astype("float32")
    matrix = compute_affine_matrix(translation=[2.0, -1.0])
    full = bic.transformation.affine_transform(a, matrix, order=1, fill_value=0)
    roi = (slice(8, 24), slice(16, 40))
    out = transform_subvolume_affine(a, matrix, roi, order=1, fill_value=0)
    np.testing.assert_allclose(out, full[8:24, 16:40], atol=1e-4)


def _shifted_coords(shape, shift):
    grid = np.indices(shape, dtype="float64")
    coords = np.empty_like(grid)
    for d in range(len(shape)):
        coords[d] = grid[d] + shift[d]
    return coords


def test_transform_subvolume_coordinates_matches_full(rng):
    # The sub-region read + local shift must be transparent: same result as map_coordinates on the
    # full array with the global coordinate field.
    a = rng.random((40, 48)).astype("float32")
    coords = _shifted_coords(a.shape, [2.5, -1.5])
    out = transform_subvolume_coordinates(a, coords, order=1, fill_value=0)
    ref = bic.transformation.map_coordinates(a, coords, order=1, fill_value=0)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_transform_subvolume_coordinates_identity(rng):
    a = rng.random((12, 15, 9)).astype("float32")
    coords = np.indices(a.shape).astype("float64")
    out = transform_subvolume_coordinates(a, coords, order=1)
    np.testing.assert_allclose(out, a, atol=1e-5)


def test_transform_subvolume_coordinates_subregion(rng):
    # A coordinate field that only samples a sub-block of a larger source (integer coords -> exact).
    a = rng.random((40, 48)).astype("float32")
    zz, xx = np.indices((16, 24), dtype="float64")
    coords = np.stack([zz + 8, xx + 16])
    out = transform_subvolume_coordinates(a, coords, order=1)
    np.testing.assert_allclose(out, a[8:24, 16:40], atol=1e-5)


def test_transform_subvolume_coordinates_out_of_bounds():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    coords = np.full((2, 3, 3), -50.0)
    out = transform_subvolume_coordinates(a, coords, order=1, fill_value=7.0)
    np.testing.assert_array_equal(out, np.full((3, 3), 7.0, dtype="float32"))


def test_transform_subvolume_coordinates_bad_shape():
    a = np.zeros((4, 4), dtype="float32")
    with pytest.raises(ValueError, match="ndim"):
        transform_subvolume_coordinates(a, np.zeros((3, 4, 4), dtype="float64"))
