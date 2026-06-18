"""Tests for the ResizedSource resize wrapper."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources import from_spec
from bioimage_py.wrapper import ResizedSource


@pytest.mark.parametrize("shape,target", [
    ((40, 48), (20, 24)),           # 2d downscale
    ((40, 48), (80, 96)),           # 2d upscale
    ((32, 40, 24), (16, 20, 12)),   # 3d downscale
    ((32, 64, 48), (64, 32, 96)),   # 3d anisotropic (mixed up/down)
])
@pytest.mark.parametrize("order,anti_aliasing", [(0, False), (1, False), (1, True)])
def test_resized_source_seamless(zarr_factory, rng, shape, target, order, anti_aliasing):
    # The core halo correctness check: a single full read must equal stitched block-wise reads.
    a = rng.random(shape).astype("float32")
    z = zarr_factory(a, chunks=tuple(min(16, s) for s in shape))
    rs = ResizedSource(z, target, order=order, anti_aliasing=anti_aliasing)
    assert rs.shape == target

    whole = bp.copy(rs)  # direct (whole-array) read
    block_shape = tuple(max(1, t // 3) for t in target)
    blocked = bp.copy(rs, block_shape=block_shape, num_workers=4)
    if order == 0 and not anti_aliasing:
        np.testing.assert_array_equal(whole, blocked)
    else:
        np.testing.assert_allclose(whole, blocked, atol=1e-4)


def test_resized_source_metadata(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    rs = ResizedSource(z, (20, 24))
    assert rs.shape == (20, 24)
    assert rs.ndim == 2
    assert rs.dtype == np.dtype("float32")
    assert rs.writable is False
    assert rs.scale == (2.0, 2.0)
    # chunks scaled into the output space: ceil(16 * 20 / 40) == 8, ceil(16 * 24 / 48) == 8.
    assert rs.chunks == (8, 8)


def test_resized_source_labels_no_invented_ids(rng):
    seg = (rng.random((37, 41, 23)) > 0.5).astype("uint16") * 7
    rs = ResizedSource(seg, (19, 21, 12), order=0)
    out = bp.copy(rs)
    assert set(np.unique(out)).issubset(set(np.unique(seg)))


def test_resized_source_bool_roundtrip(rng):
    a = rng.random((40, 40)) > 0.5
    rs = ResizedSource(a, (20, 20), order=0)
    out = bp.copy(rs)
    assert out.dtype == np.dtype(bool)
    assert set(np.unique(out)).issubset({False, True})


def test_resized_source_spec_roundtrip(zarr_factory, rng):
    a = rng.random((40, 48, 24)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16, 16))
    rs = ResizedSource(z, (20, 24, 12), order=1, anti_aliasing=True)

    rebuilt = from_spec(rs.to_spec())
    assert isinstance(rebuilt, ResizedSource)
    assert rebuilt.shape == rs.shape
    np.testing.assert_array_equal(bp.copy(rebuilt), bp.copy(rs))


def test_resized_source_validation(rng):
    a = rng.random((40, 48)).astype("float32")
    with pytest.raises(ValueError, match="dimensionality"):
        ResizedSource(a, (20, 24, 12))
    with pytest.raises(ValueError, match="order"):
        ResizedSource(a, (20, 24), order=6)
    with pytest.raises(ValueError, match="2d or 3d"):
        ResizedSource(rng.random((4, 4, 4, 4)), (2, 2, 2, 2))
