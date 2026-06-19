"""Tests for block-wise statistics."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.util import get_blocking, to_roi


def test_reductions_match_numpy(zarr_factory, rng):
    a = rng.random((33, 28)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    kw = dict(block_shape=(8, 8), num_workers=3)
    assert np.isclose(bp.stats.max(z, **kw), a.max())
    assert np.isclose(bp.stats.min(z, **kw), a.min())
    assert np.isclose(bp.stats.mean(z, **kw), a.mean(), atol=1e-5)
    assert np.isclose(bp.stats.std(z, **kw), a.std(), atol=1e-5)
    mn, mx = bp.stats.min_and_max(z, **kw)
    assert np.isclose(mn, a.min()) and np.isclose(mx, a.max())


def test_mask(rng):
    a = rng.random((20, 20)).astype("float32")
    mask = np.zeros((20, 20), dtype="uint8")
    mask[2:8, 3:9] = 1  # a region that does not cover whole blocks
    expected = a[mask.astype(bool)].max()
    got = bp.stats.max(a, block_shape=(5, 5), num_workers=2, mask=mask)
    assert np.isclose(got, expected)


def test_all_masked_raises(rng):
    a = rng.random((10, 10)).astype("float32")
    mask = np.zeros((10, 10), dtype="uint8")
    with pytest.raises(ValueError, match="No values within the mask"):
        bp.stats.max(a, block_shape=(5, 5), num_workers=2, mask=mask)


def test_block_ids_subset(rng):
    a = rng.random((16, 16)).astype("float32")
    block_shape = (8, 8)
    blocking = get_blocking(a.shape, block_shape)
    # max restricted to block 0 only must equal the max over block 0's region.
    block0 = a[to_roi(blocking.get_block(0))]
    got = bp.stats.max(a, block_shape=block_shape, num_workers=2, block_ids=[0])
    assert np.isclose(got, block0.max())


def test_direct_rejects_mask(rng):
    a = rng.random((8, 8)).astype("float32")
    with pytest.raises(ValueError, match="Direct computation"):
        bp.stats.max(a, mask=np.ones((8, 8), dtype="uint8"))


def test_reductions_subprocess_parity(zarr_factory, rng):
    # Every reduction must agree across direct/local/subprocess, exercising the return-value
    # channel (and the parallel-variance merge for mean/std) through cloudpickle + partitioning.
    a = rng.random((33, 28)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    kw = dict(block_shape=(8, 8), num_workers=3, job_type="subprocess")
    assert np.isclose(bp.stats.max(z, **kw), a.max())
    assert np.isclose(bp.stats.min(z, **kw), a.min())
    assert np.isclose(bp.stats.mean(z, **kw), a.mean(), atol=1e-5)
    assert np.isclose(bp.stats.std(z, **kw), a.std(), atol=1e-5)
    mn, mx = bp.stats.min_and_max(z, **kw)
    assert np.isclose(mn, a.min()) and np.isclose(mx, a.max())
    mean_val, std_val = bp.stats.mean_and_std(z, **kw)
    assert np.isclose(mean_val, a.mean(), atol=1e-5) and np.isclose(std_val, a.std(), atol=1e-5)


def test_mask_shape_mismatch_raises(rng):
    a = rng.random((16, 16)).astype("float32")
    bad_mask = np.ones((8, 8), dtype="uint8")  # wrong shape
    with pytest.raises(ValueError, match="Shape mismatch"):
        bp.stats.max(a, block_shape=(8, 8), num_workers=2, mask=bad_mask)
