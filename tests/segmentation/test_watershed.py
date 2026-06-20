"""Tests for block-wise seeded watershed."""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp


def _make_data(shape, rng, n_seeds=12):
    """Build a smooth float32 height map and a sparse, globally-unique uint32 seed map."""
    hmap = bic.filters.gaussian_smoothing(rng.random(shape).astype("float32"), 2.0)
    seeds = np.zeros(shape, dtype="uint32")
    idx = tuple(rng.integers(0, s, size=n_seeds) for s in shape)
    seeds[idx] = np.arange(1, n_seeds + 1, dtype="uint32")
    return hmap, seeds


@pytest.mark.parametrize("shape,block_shape,halo", [
    ((48, 50), (16, 16), (8, 8)),
    ((64, 64), (13, 17), (6, 6)),      # non-divisible block shape
    ((24, 28, 26), (8, 8, 8), (4, 4, 4)),
])
@pytest.mark.parametrize("nw,job_type", [(1, "local"), (4, "local"), (3, "subprocess")])
def test_watershed_backend_determinism(shape, block_shape, halo, nw, job_type, zarr_factory, rng):
    # For a fixed (block_shape, halo) every backend must produce bit-identical output, since each
    # block is an independent, deterministic computation. local(1) is the reference.
    hmap, seeds = _make_data(shape, rng)
    zh = zarr_factory(hmap, chunks=block_shape)
    zs = zarr_factory(seeds, chunks=block_shape)

    ref = zarr_factory(shape=shape, chunks=block_shape, dtype="uint64", fill=0)
    bp.segmentation.watershed(zh, zs, ref, halo=halo, block_shape=block_shape, num_workers=1)

    out = zarr_factory(shape=shape, chunks=block_shape, dtype="uint64", fill=0)
    bp.segmentation.watershed(zh, zs, out, halo=halo, block_shape=block_shape,
                              num_workers=nw, job_type=job_type)
    np.testing.assert_array_equal(out[:], ref[:])


def test_watershed_single_block_matches_reference(rng):
    shape = (40, 40)
    hmap, seeds = _make_data(shape, rng)
    ref = bic.segmentation.watershed(hmap, seeds)

    # Direct path (local, 1 worker, no block_shape) is a whole-array watershed.
    direct = bp.segmentation.watershed(hmap, seeds)
    assert isinstance(direct, np.ndarray) and direct.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(direct, ref)

    # A single block (block_shape == full shape) reproduces the reference regardless of halo,
    # because the outer block is clipped to the array bounds.
    out = np.zeros(shape, dtype="uint64")
    bp.segmentation.watershed(hmap, seeds, out, halo=(0, 0), block_shape=shape, num_workers=1)
    np.testing.assert_array_equal(out, ref)


def test_watershed_mask(zarr_factory, rng):
    shape = (40, 40)
    hmap, seeds = _make_data(shape, rng)
    seeds[16, 16] = 50  # guarantee labeled content inside the mask
    seeds[24, 24] = 51
    mask = np.zeros(shape, dtype="uint8")
    mask[8:32, 8:32] = 1

    zh = zarr_factory(hmap, chunks=(16, 16))
    zs = zarr_factory(seeds, chunks=(16, 16))
    out = zarr_factory(shape=shape, chunks=(16, 16), dtype="uint64", fill=0)
    bp.segmentation.watershed(zh, zs, out, halo=(8, 8), block_shape=(16, 16),
                              num_workers=4, mask=mask)
    result = out[:]
    mask_b = mask.astype(bool)
    assert np.all(result[~mask_b] == 0)
    assert result[mask_b].any()


def test_output_optional_local(zarr_factory, rng):
    shape = (40, 40)
    hmap, seeds = _make_data(shape, rng)
    zh = zarr_factory(hmap, chunks=(16, 16))
    zs = zarr_factory(seeds, chunks=(16, 16))
    # No output -> a uint64 numpy array is allocated and returned.
    result = bp.segmentation.watershed(zh, zs, halo=(8, 8), block_shape=(16, 16), num_workers=4)
    assert isinstance(result, np.ndarray) and result.dtype == np.dtype("uint64")


def test_output_required_distributed(zarr_factory, rng):
    shape = (32, 32)
    hmap, seeds = _make_data(shape, rng)
    zh = zarr_factory(hmap, chunks=(16, 16))
    zs = zarr_factory(seeds, chunks=(16, 16))
    with pytest.raises(ValueError, match="required for distributed execution"):
        bp.segmentation.watershed(zh, zs, halo=(8, 8), block_shape=(16, 16),
                                  num_workers=2, job_type="subprocess")


def test_output_dtype_validated(rng):
    shape = (16, 16)
    hmap, seeds = _make_data(shape, rng)
    out = np.zeros(shape, dtype="float32")
    with pytest.raises(ValueError, match="integer"):
        bp.segmentation.watershed(hmap, seeds, out, halo=(4, 4), block_shape=(8, 8), num_workers=2)


def test_halo_required_blockwise(zarr_factory, rng):
    shape = (32, 32)
    hmap, seeds = _make_data(shape, rng)
    zh = zarr_factory(hmap, chunks=(16, 16))
    zs = zarr_factory(seeds, chunks=(16, 16))
    out = zarr_factory(shape=shape, chunks=(16, 16), dtype="uint64", fill=0)
    with pytest.raises(ValueError, match="halo is required"):
        bp.segmentation.watershed(zh, zs, out, block_shape=(16, 16), num_workers=4)


def test_seeds_shape_mismatch(rng):
    hmap = rng.random((16, 16)).astype("float32")
    seeds = np.zeros((16, 8), dtype="uint32")
    with pytest.raises(ValueError, match="seeds shape"):
        bp.segmentation.watershed(hmap, seeds)
