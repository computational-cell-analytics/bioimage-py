"""Tests for the AffineSource affine-transform wrapper."""
import bioimage_cpp as bic
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources import from_spec
from bioimage_py.transformation import compute_affine_matrix
from bioimage_py.wrapper import AffineSource


def test_affine_source_identity(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    src = AffineSource(z, affine_matrix=np.eye(3), order=1)
    assert src.shape == (40, 48)
    assert src.writable is False
    np.testing.assert_allclose(bp.copy(src), a, atol=1e-5)


def test_affine_source_matches_reference(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    src = AffineSource(z, affine_matrix=matrix, order=1)
    ref = bic.transformation.affine_transform(a, matrix, order=1, fill_value=0)
    np.testing.assert_allclose(bp.copy(src), ref, atol=1e-4)


@pytest.mark.parametrize("shape,rotation,order", [
    ((48, 56), [30.0], 1),
    ((40, 48), [0.0], 0),
    ((24, 32, 28), [10.0, 0.0, 0.0], 1),
])
def test_affine_source_seamless(zarr_factory, rng, shape, rotation, order):
    a = rng.random(shape).astype("float32")
    z = zarr_factory(a, chunks=tuple(min(16, s) for s in shape))
    matrix = compute_affine_matrix(rotation=rotation, translation=[2.0] * len(shape))
    src = AffineSource(z, affine_matrix=matrix, order=order)

    whole = bp.copy(src)
    block_shape = tuple(max(1, s // 3) for s in shape)
    blocked = bp.copy(src, block_shape=block_shape, num_workers=4)
    if order == 0:
        np.testing.assert_array_equal(whole, blocked)
    else:
        np.testing.assert_allclose(whole, blocked, atol=1e-4)


def test_affine_source_from_parameters(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    src = AffineSource(z, scale=[1.0, 1.0], translation=[4.0, 4.0], order=1)
    matrix = compute_affine_matrix(scale=[1.0, 1.0], translation=[4.0, 4.0])
    ref = bic.transformation.affine_transform(a, matrix, order=1, fill_value=0)
    np.testing.assert_allclose(bp.copy(src), ref, atol=1e-4)


def test_affine_source_labels_no_invented_ids(rng):
    seg = (rng.random((40, 48)) > 0.5).astype("uint16") * 7
    src = AffineSource(seg, scale=[1.0, 1.0], rotation=[15.0], order=0)
    out = bp.copy(src)
    assert set(np.unique(out)).issubset(set(np.unique(seg)) | {0})


def test_affine_source_spec_roundtrip(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    src = AffineSource(z, scale=[1.0, 1.0], rotation=[20.0], order=1, anti_aliasing=True)
    rebuilt = from_spec(src.to_spec())
    assert isinstance(rebuilt, AffineSource)
    assert rebuilt.shape == src.shape
    np.testing.assert_array_equal(bp.copy(rebuilt), bp.copy(src))


def test_affine_source_subprocess(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    matrix = compute_affine_matrix(rotation=[12.0], translation=[3.0, -2.0])
    src = AffineSource(z, affine_matrix=matrix, order=1)
    expected = bp.copy(src)

    out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
    bp.copy(src, output=out, block_shape=(16, 16), num_workers=3, job_type="subprocess")
    np.testing.assert_allclose(out[:], expected, atol=1e-4)


def test_affine_source_validation(rng):
    a = rng.random((40, 48)).astype("float32")
    with pytest.raises(ValueError, match="exactly one"):
        AffineSource(a)  # neither matrix nor parameters
    with pytest.raises(ValueError, match="exactly one"):
        AffineSource(a, affine_matrix=np.eye(3), scale=[1.0, 1.0])
    with pytest.raises(ValueError, match="2d or 3d"):
        AffineSource(rng.random((4, 4, 4, 4)), affine_matrix=np.eye(5))
    with pytest.raises(ValueError, match="order"):
        AffineSource(a, affine_matrix=np.eye(3), order=6)
