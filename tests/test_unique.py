"""Tests for block-wise unique."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.util import get_blocking, to_roi


def _labels(rng, shape):
    """Sparse, large label ids to exercise the additive (non-dense) count merge."""
    a = rng.integers(0, 50, size=shape).astype("uint64")
    a[a > 0] *= 1000  # spread ids out (0, 1000, 2000, ...) so a dense counts array would be huge.
    return a


def test_unique_matches_numpy(zarr_factory, rng):
    # direct / local(1) / local(4) / subprocess(3) must all agree with numpy, with and without counts.
    a = _labels(rng, (33, 28))
    z = zarr_factory(a, chunks=(8, 8))
    exp_vals, exp_counts = np.unique(a, return_counts=True)

    cases = [
        (a, dict()),  # direct
        (z, dict(block_shape=(8, 8), num_workers=1)),
        (z, dict(block_shape=(8, 8), num_workers=4)),
        (z, dict(block_shape=(8, 8), num_workers=3, job_type="subprocess")),
    ]
    for src, kw in cases:
        np.testing.assert_array_equal(bp.stats.unique(src, **kw), exp_vals)
        vals, counts = bp.stats.unique(src, return_counts=True, **kw)
        np.testing.assert_array_equal(vals, exp_vals)
        np.testing.assert_array_equal(counts, exp_counts)


def test_unique_mask(rng):
    a = _labels(rng, (20, 20))
    mask = np.zeros((20, 20), dtype="uint8")
    mask[2:14, 3:15] = 1  # a region that does not cover whole blocks
    exp_vals, exp_counts = np.unique(a[mask.astype(bool)], return_counts=True)
    vals, counts = bp.stats.unique(a, return_counts=True, block_shape=(5, 5), num_workers=2, mask=mask)
    np.testing.assert_array_equal(vals, exp_vals)
    np.testing.assert_array_equal(counts, exp_counts)


def test_unique_block_ids_subset(rng):
    a = _labels(rng, (16, 16))
    block_shape = (8, 8)
    blocking = get_blocking(a.shape, block_shape)
    block0 = a[to_roi(blocking.get_block(0))]
    got = bp.stats.unique(a, block_shape=block_shape, num_workers=2, block_ids=[0])
    np.testing.assert_array_equal(got, np.unique(block0))


def test_unique_direct_rejects_mask(rng):
    a = _labels(rng, (8, 8))
    with pytest.raises(ValueError, match="Direct computation"):
        bp.stats.unique(a, mask=np.ones((8, 8), dtype="uint8"))
