"""Tests for the affine matrix and sub-volume helpers in bioimage_py.transformation."""
import bioimage_cpp as bic
import numpy as np
import pytest

from bioimage_py.transformation import (
    compute_affine_matrix,
    transform_roi_with_affine,
    transform_subvolume_affine,
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
