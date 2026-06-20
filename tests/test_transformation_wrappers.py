"""Tests for the generic transformation wrappers in bioimage_py.wrapper.base."""
import bioimage_cpp as bic
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources import from_spec
from bioimage_py.wrapper import (
    MultiTransformationSource,
    SimpleTransformationSource,
    SimpleTransformationWithHaloSource,
    TransformationSource,
)


def test_simple_transformation_parity(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    expected = a * 2.0 + 1.0

    src = SimpleTransformationSource(z, lambda block: block * 2.0 + 1.0)
    assert src.shape == (40, 48)
    assert src.dtype == np.dtype("float32")
    assert src.writable is False

    # Value-only transform: whole == block-wise == subprocess (no halo needed).
    np.testing.assert_allclose(bp.copy(src), expected, atol=1e-6)
    for nw, job in [(4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
        bp.copy(src, output=out, block_shape=(16, 16), num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], expected, atol=1e-6, err_msg=f"nw={nw} job={job}")


def test_simple_transformation_with_channels(zarr_factory, rng):
    a = rng.random((3, 40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(3, 16, 16))
    expected = a.sum(axis=0)

    src = SimpleTransformationSource(z, lambda block: block.sum(axis=0),
                                     with_channels=True, dtype="float32")
    assert src.shape == (40, 48)
    assert src.ndim == 2
    assert src.chunks == (16, 16)

    np.testing.assert_allclose(bp.copy(src), expected, atol=1e-5)
    blocked = bp.copy(src, block_shape=(16, 16), num_workers=4)
    np.testing.assert_allclose(blocked, expected, atol=1e-5)


def test_simple_transformation_with_halo_seamless(zarr_factory, rng):
    a = rng.random((48, 56)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    src = SimpleTransformationWithHaloSource(
        z, lambda block: bic.filters.gaussian_smoothing(block, 1.0), halo=(8, 8)
    )
    # The halo makes block-wise reads seam-free: whole == stitched blocks.
    whole = bp.copy(src)
    blocked = bp.copy(src, block_shape=(16, 16), num_workers=4)
    np.testing.assert_allclose(whole, blocked, atol=1e-4)
    # And it matches a single whole-array gaussian.
    np.testing.assert_allclose(whole, bic.filters.gaussian_smoothing(a, 1.0), atol=1e-4)


def test_simple_transformation_with_halo_validation(rng):
    a = rng.random((20, 24)).astype("float32")
    with pytest.raises(ValueError, match="halo"):
        SimpleTransformationWithHaloSource(a, lambda b: b, halo=(4, 4, 4))


def test_transformation_source_coordinate_aware(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    # Add the global row index to each row; this is coordinate-aware but seam-free.
    src = TransformationSource(
        z, lambda block, roi: block + np.arange(roi[0].start, roi[0].stop, dtype=block.dtype).reshape(-1, 1)
    )
    expected = a + np.arange(40, dtype="float32").reshape(-1, 1)
    np.testing.assert_allclose(bp.copy(src), expected, atol=1e-5)
    blocked = bp.copy(src, block_shape=(16, 16), num_workers=4)
    np.testing.assert_allclose(blocked, expected, atol=1e-5)


def test_multi_transformation_parity(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    b = rng.random((40, 48)).astype("float32")
    za = zarr_factory(a, chunks=(16, 16))
    zb = zarr_factory(b, chunks=(16, 16))
    expected = a + b

    src = MultiTransformationSource(lambda x, y: x + y, za, zb)
    assert src.shape == (40, 48)
    np.testing.assert_allclose(bp.copy(src), expected, atol=1e-6)
    for nw, job in [(4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
        bp.copy(src, output=out, block_shape=(16, 16), num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], expected, atol=1e-6, err_msg=f"nw={nw} job={job}")


def test_multi_transformation_apply_to_list(zarr_factory, rng):
    a = rng.random((20, 24)).astype("float32")
    b = rng.random((20, 24)).astype("float32")
    c = rng.random((20, 24)).astype("float32")
    src = MultiTransformationSource(
        lambda blocks: np.stack(blocks).max(axis=0), zarr_factory(a, chunks=(8, 8)),
        zarr_factory(b, chunks=(8, 8)), zarr_factory(c, chunks=(8, 8)), apply_to_list=True,
    )
    np.testing.assert_allclose(bp.copy(src), np.maximum(np.maximum(a, b), c), atol=1e-6)


def test_multi_transformation_shape_mismatch(rng):
    with pytest.raises(ValueError, match="same shape"):
        MultiTransformationSource(lambda x, y: x + y, rng.random((20, 24)), rng.random((20, 20)))


@pytest.mark.parametrize("build", [
    lambda z: SimpleTransformationSource(z, lambda block: block * 2.0 + 1.0),
    lambda z: SimpleTransformationWithHaloSource(
        z, lambda block: bic.filters.gaussian_smoothing(block, 1.0), halo=(6, 6)),
    lambda z: TransformationSource(z, lambda block, roi: block + 1.0),
])
def test_transformation_spec_roundtrip(zarr_factory, rng, build):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    src = build(z)
    rebuilt = from_spec(src.to_spec())
    assert type(rebuilt) is type(src)
    assert rebuilt.shape == src.shape
    np.testing.assert_array_equal(bp.copy(rebuilt), bp.copy(src))


def test_multi_transformation_spec_roundtrip(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    b = rng.random((40, 48)).astype("float32")
    src = MultiTransformationSource(lambda x, y: x + y, zarr_factory(a, chunks=(16, 16)),
                                    zarr_factory(b, chunks=(16, 16)))
    rebuilt = from_spec(src.to_spec())
    assert isinstance(rebuilt, MultiTransformationSource)
    np.testing.assert_array_equal(bp.copy(rebuilt), bp.copy(src))
