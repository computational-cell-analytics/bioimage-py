"""Headline parity tests: direct == local(1) == local(N) == subprocess(N)."""
import numpy as np
import pandas as pd
import pytest

import bioimage_cpp as bic
import bioimage_py as bp


def test_max_parity(zarr_factory, rng):
    a = rng.random((37, 41)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    expected = float(a.max())

    assert np.isclose(bp.stats.max(a), expected)  # direct
    assert np.isclose(bp.stats.max(z, num_workers=1, block_shape=(16, 16)), expected)
    assert np.isclose(bp.stats.max(z, num_workers=4, block_shape=(16, 16)), expected)
    assert np.isclose(
        bp.stats.max(z, num_workers=3, block_shape=(16, 16), job_type="subprocess"), expected
    )


def test_max_numpy_subprocess_raises(rng):
    a = rng.random((20, 20)).astype("float32")
    with pytest.raises(ValueError, match="numpy"):
        bp.stats.max(a, num_workers=2, block_shape=(8, 8), job_type="subprocess")


def test_gaussian_parity(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    ref = bic.filters.gaussian_smoothing(a, 2.0)

    # direct (in place on a copy via numpy source)
    direct = bp.filters.gaussian_smoothing(a.copy(), 2.0)
    np.testing.assert_allclose(direct[(slice(None), slice(None))], ref, atol=1e-4)

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
        bp.filters.gaussian_smoothing(z, 2.0, output=out, block_shape=(16, 16),
                                      num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], ref, atol=1e-4,
                                   err_msg=f"mismatch for nw={nw} job={job}")


def test_copy_sharded_parity(zarr_factory, rng):
    # Sharded zarr v3 output: the atomic write unit is the shard, not the inner chunk. The
    # block shape is deliberately NOT a shard multiple, so without shard-exclusive routing
    # concurrent workers would corrupt shards. Parity across backends proves routing works.
    a = rng.random((64, 64)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))

    # 4 blocks per 32x32 shard; block 16x16 is a chunk- but not a shard-multiple.
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(32, 32),
                           dtype="float32", fill=0.0)
        bp.copy(z, out, block_shape=(16, 16), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(out[:], a, err_msg=f"sharded copy nw={nw} job={job}")

    # Blocks straddling shard boundaries (block 32x16 vs shard 32x32): adjacent column blocks
    # share a shard, so they must be routed to the same worker.
    for nw, job in [(4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(32, 32),
                           dtype="float32", fill=0.0)
        bp.copy(z, out, block_shape=(32, 16), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(out[:], a, err_msg=f"straddling copy nw={nw} job={job}")


def test_watershed_parity(zarr_factory, rng):
    # Seeded watershed is single-stage and block-wise-approximate: a single block reproduces the
    # whole-array reference exactly, and for a fixed (block_shape, halo) every backend is bit-identical.
    shape, block_shape, halo = (48, 50), (16, 16), (8, 8)
    hmap = bic.filters.gaussian_smoothing(rng.random(shape).astype("float32"), 2.0)
    seeds = np.zeros(shape, dtype="uint32")
    idx = tuple(rng.integers(0, s, size=12) for s in shape)
    seeds[idx] = np.arange(1, 13, dtype="uint32")
    zh = zarr_factory(hmap, chunks=block_shape)
    zs = zarr_factory(seeds, chunks=block_shape)

    # Single block (full block_shape) reproduces the whole-array reference exactly.
    ref = bic.segmentation.watershed(hmap, seeds)
    one = np.zeros(shape, dtype="uint64")
    bp.segmentation.watershed(hmap, seeds, one, halo=(0, 0), block_shape=shape, num_workers=1)
    np.testing.assert_array_equal(one, ref)

    # Multi-block: bit-identical across backends / worker counts.
    results = []
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=shape, chunks=block_shape, dtype="uint64", fill=0)
        bp.segmentation.watershed(zh, zs, out, halo=halo, block_shape=block_shape,
                                  num_workers=nw, job_type=job)
        results.append(out[:])
    for r in results[1:]:
        np.testing.assert_array_equal(results[0], r, err_msg="watershed backend mismatch")


def test_morphology_parity(zarr_factory, rng):
    from skimage.measure import label as sklabel  # local import: test-only dependency.

    # A labeled volume whose objects straddle block boundaries, so the cross-block merge is exercised.
    seg = sklabel(rng.random((37, 41, 23)) > 0.55).astype("uint64")
    z = zarr_factory(seg, chunks=(16, 16, 16))
    expected = bp.morphology.morphology(seg)  # direct

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = bp.morphology.morphology(z, num_workers=nw, block_shape=(16, 16, 16), job_type=job)
        pd.testing.assert_frame_equal(expected, out, obj=f"morphology nw={nw} job={job}")


def test_regionprops_parity(zarr_factory):
    # A handful of objects (so the per-object map partitions into several tasks) straddling chunks.
    seg = np.zeros((24, 32, 28), dtype="uint64")
    seg[2:9, 3:14, 4:12] = 1
    seg[12:20, 18:30, 15:26] = 2
    seg[3:7, 20:28, 2:10] = 3
    seg[15:22, 4:12, 18:26] = 4
    z = zarr_factory(seg, chunks=(16, 16, 16))
    table = bp.morphology.morphology(seg)
    res = (2.0, 1.0, 1.5)

    expected = bp.morphology.regionprops(seg, table, resolution=res)  # direct (local, 1 worker)
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = bp.morphology.regionprops(z, table, resolution=res, num_workers=nw, job_type=job)
        pd.testing.assert_frame_equal(expected, out, obj=f"regionprops nw={nw} job={job}")
