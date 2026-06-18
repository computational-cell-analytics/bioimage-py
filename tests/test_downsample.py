"""Tests for the block-wise downsample operation."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.util import downscale_shape


def test_downsample_label_parity(zarr_factory, rng):
    seg = ((rng.random((37, 41, 23)) > 0.5).astype("uint16")) * 5
    z = zarr_factory(seg, chunks=(16, 16, 16))
    target = downscale_shape(seg.shape, 2)

    expected = bp.downsample(seg, 2)  # direct (local, 1 worker), default order=0
    assert expected.shape == target

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=target, chunks=(8, 8, 8), dtype="uint16", fill=0)
        bp.downsample(z, 2, output=out, block_shape=(8, 8, 8), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(out[:], expected, err_msg=f"nw={nw} job={job}")


def test_downsample_image_parity(zarr_factory, rng):
    a = rng.random((40, 48, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16, 16))
    target = downscale_shape(a.shape, 2)

    expected = bp.downsample(a, 2, order=1, anti_aliasing=True)  # direct
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=target, chunks=(8, 8, 8), dtype="float32", fill=0.0)
        bp.downsample(z, 2, output=out, order=1, anti_aliasing=True,
                      block_shape=(8, 8, 8), num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], expected, atol=1e-4, err_msg=f"nw={nw} job={job}")


def test_downsample_optional_output_local(zarr_factory, rng):
    a = rng.random((40, 40)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    result = bp.downsample(z, 2, block_shape=(4, 4), num_workers=3)
    assert isinstance(result, np.ndarray)
    assert result.shape == downscale_shape(a.shape, 2)
    assert result.dtype == a.dtype


def test_downsample_anisotropic(zarr_factory, rng):
    a = rng.random((20, 40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(8, 16, 16))
    target = downscale_shape(a.shape, (1, 2, 2))
    assert target == (20, 20, 24)
    out = zarr_factory(shape=target, chunks=(8, 8, 8), dtype="float32", fill=0.0)
    bp.downsample(z, (1, 2, 2), output=out, block_shape=(8, 8, 8), num_workers=2)
    assert out.shape == target


def test_downsample_invalid_scale_factor(zarr_factory, rng):
    a = rng.random((20, 20)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    with pytest.raises(ValueError, match="scale factors"):
        bp.downsample(z, 0)
    with pytest.raises(ValueError, match="dimensionality"):
        bp.downsample(z, (2, 2, 2))


def test_downsample_output_required_for_distributed(zarr_factory, rng):
    a = rng.random((20, 20)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    with pytest.raises(ValueError, match="output"):
        bp.downsample(z, 2, block_shape=(4, 4), num_workers=2, job_type="subprocess")
