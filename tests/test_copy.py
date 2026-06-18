"""Tests for the block-wise ``copy`` operation."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.wrapper import ThresholdSource


def test_copy_parity(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))

    # direct (local, 1 worker, no block_shape): returns a numpy array equal to the input.
    direct = bp.copy(z)
    assert isinstance(direct, np.ndarray)
    np.testing.assert_array_equal(direct, a)

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
        result = bp.copy(z, output=out, block_shape=(16, 16), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(result[:], a, err_msg=f"mismatch for nw={nw} job={job}")


def test_copy_optional_output_local(zarr_factory, rng):
    a = rng.random((40, 40)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))

    # Block-wise local path with no output -> a numpy array is allocated and returned.
    result = bp.copy(z, block_shape=(8, 8), num_workers=3)
    assert isinstance(result, np.ndarray)
    assert result.dtype == a.dtype
    np.testing.assert_array_equal(result, a)


def test_copy_wrapper_persist(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    expected = a > 0.5

    # Persist an on-the-fly threshold (bool output, casting from the wrapper dtype).
    for nw, job in [(2, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="bool", fill=False)
        bp.copy(ThresholdSource(z, 0.5), output=out, block_shape=(16, 16),
                num_workers=nw, job_type=job)
        np.testing.assert_array_equal(out[:], expected, err_msg=f"mismatch for nw={nw} job={job}")


def test_copy_mask(zarr_factory, rng):
    a = rng.random((40, 40)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    mask = np.zeros((40, 40), dtype="uint8")
    mask[8:32, 8:32] = 1
    sentinel = -999.0
    out = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="float32", fill=sentinel)

    bp.copy(z, output=out, block_shape=(8, 8), num_workers=3, mask=mask)

    result = out[:]
    m = mask.astype(bool)
    assert np.all(result[~m] == sentinel)
    np.testing.assert_array_equal(result[m], a[m])


def test_copy_output_required_for_distributed(zarr_factory, rng):
    a = rng.random((20, 20)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    with pytest.raises(ValueError, match="output"):
        bp.copy(z, block_shape=(8, 8), num_workers=2, job_type="subprocess")


def test_copy_in_place_rejected(zarr_factory, rng):
    a = rng.random((20, 20)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    with pytest.raises(ValueError, match="differ"):
        bp.copy(z, output=z, block_shape=(8, 8), num_workers=2)


def test_copy_hdf5_distributed_output_rejected(zarr_factory, hdf5_factory, rng):
    a = rng.random((20, 20)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    path, key = hdf5_factory(np.zeros_like(a), chunks=(8, 8))
    out = bp.open_source(path, key, mode="r+")
    with pytest.raises(ValueError, match="HDF5"):
        bp.copy(z, output=out, block_shape=(8, 8), num_workers=2, job_type="subprocess")
