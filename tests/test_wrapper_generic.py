"""Tests for the concrete generic wrappers: Threshold, Normalize, Roi, Pad."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources import from_spec
from bioimage_py.wrapper import NormalizeSource, PadSource, RoiSource, ThresholdSource


def test_threshold_operator(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))

    gt = ThresholdSource(z, 0.5)
    assert gt.dtype == np.dtype(bool)
    np.testing.assert_array_equal(bp.copy(gt), a > 0.5)

    lt = ThresholdSource(z, 0.5, operator=np.less)
    np.testing.assert_array_equal(bp.copy(lt), a < 0.5)


def test_threshold_spec_roundtrip(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    src = ThresholdSource(z, 0.3, operator=np.greater_equal)
    rebuilt = from_spec(src.to_spec())
    assert isinstance(rebuilt, ThresholdSource)
    np.testing.assert_array_equal(bp.copy(rebuilt), bp.copy(src))


def test_normalize_source(zarr_factory, rng):
    a = (rng.random((40, 48)) * 100).astype("float32")
    z = zarr_factory(a, chunks=(40, 48))  # single chunk -> whole-array normalization
    src = NormalizeSource(z)
    assert src.dtype == np.dtype("float32")
    out = bp.copy(src)
    # A whole read normalizes by the global min / max.
    expected = (a - a.min()) / (a.max() - a.min() + NormalizeSource.eps)
    np.testing.assert_allclose(out, expected, atol=1e-5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalize_spec_roundtrip(zarr_factory, rng):
    a = (rng.random((40, 48)) * 100).astype("float32")
    z = zarr_factory(a, chunks=(40, 48))
    src = NormalizeSource(z, dtype="float64")
    rebuilt = from_spec(src.to_spec())
    assert isinstance(rebuilt, NormalizeSource)
    assert rebuilt.dtype == np.dtype("float64")
    np.testing.assert_array_equal(bp.copy(rebuilt), bp.copy(src))


def test_roi_source_read(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    src = RoiSource(z, (slice(8, 32), slice(8, 40)))
    assert src.shape == (24, 32)
    assert src.ndim == 2
    np.testing.assert_array_equal(src[(slice(None), slice(None))], a[8:32, 8:40])
    # A sub-roi reads relative to the roi origin.
    np.testing.assert_array_equal(src[(slice(2, 10), slice(0, 5))], a[10:18, 8:13])


def test_roi_source_write_passthrough(zarr_factory, rng):
    z = zarr_factory(np.zeros((20, 20), dtype="float32"), chunks=(10, 10))
    src = RoiSource(z, (slice(4, 12), slice(4, 12)))
    assert src.writable is True
    block = rng.random((8, 8)).astype("float32")
    src[(slice(None), slice(None))] = block
    result = z[:]
    np.testing.assert_array_equal(result[4:12, 4:12], block)
    # Everything outside the roi is untouched.
    mask = np.ones((20, 20), dtype=bool)
    mask[4:12, 4:12] = False
    assert np.all(result[mask] == 0.0)


def test_roi_source_squeeze(zarr_factory, rng):
    a = rng.random((6, 40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(2, 16, 16))

    squeezed = RoiSource(z, (2, slice(None), slice(None)), squeeze=True)
    assert squeezed.shape == (40, 48)
    assert squeezed.ndim == 2
    np.testing.assert_array_equal(squeezed[(slice(None), slice(None))], a[2])

    # Without squeeze the integer entry keeps a singleton axis.
    kept = RoiSource(z, (2, slice(None), slice(None)))
    assert kept.shape == (1, 40, 48)


def test_roi_source_spec_roundtrip(zarr_factory, rng):
    a = rng.random((6, 40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(2, 16, 16))
    src = RoiSource(z, (3, slice(8, 32), slice(8, 40)), squeeze=True)
    rebuilt = from_spec(src.to_spec())
    assert isinstance(rebuilt, RoiSource)
    assert rebuilt.shape == src.shape
    np.testing.assert_array_equal(
        rebuilt[(slice(None), slice(None))], src[(slice(None), slice(None))]
    )


def test_pad_source(zarr_factory, rng):
    a = rng.random((16, 16)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    src = PadSource(z, (8, 8))
    assert src.shape == (24, 24)

    expected = np.pad(a, ((0, 8), (0, 8)), mode="constant")
    np.testing.assert_array_equal(bp.copy(src), expected)
    # Aligned padding tiles cleanly, so a block-wise read also matches.
    blocked = bp.copy(src, block_shape=(8, 8), num_workers=4)
    np.testing.assert_array_equal(blocked, expected)


def test_pad_source_left_padding_unsupported(rng):
    a = rng.random((16, 16)).astype("float32")
    src = PadSource(a, (8, 8))
    with pytest.raises(NotImplementedError):
        src[(slice(20, 24), slice(0, 8))]


def test_pad_source_spec_roundtrip(zarr_factory, rng):
    a = rng.random((16, 16)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    src = PadSource(z, (8, 4))
    rebuilt = from_spec(src.to_spec())
    assert isinstance(rebuilt, PadSource)
    assert rebuilt.shape == src.shape
    np.testing.assert_array_equal(bp.copy(rebuilt), bp.copy(src))
