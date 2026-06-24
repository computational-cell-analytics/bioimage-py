"""Tests for block-wise segmentation size filtering and the generic segmentation filter."""
import bioimage_cpp as bic
import numpy as np

import bioimage_py as bp


def test_size_filter_min_size(zarr_factory):
    a = np.zeros((20, 20), dtype="uint32")
    a[0:1, 0:1] = 1        # size 1
    a[2:5, 2:5] = 7        # size 9
    a[10:18, 10:18] = 3    # size 64
    z = zarr_factory(a, chunks=(8, 8))
    zout = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="uint32", fill=0)

    bp.segmentation.size_filter(z, zout, min_size=5, block_shape=(8, 8), num_workers=4)
    res = zout[:]

    assert set(np.unique(res).tolist()) == {0, 1, 2}  # one object removed, survivors relabeled 1..2
    assert np.all(res[0:1, 0:1] == 0)        # id 1 (size 1) removed
    assert np.all(res[10:18, 10:18] == 1)    # id 3 is the smallest surviving fg id -> 1
    assert np.all(res[2:5, 2:5] == 2)        # id 7 -> 2


def test_size_filter_no_relabel(zarr_factory):
    a = np.zeros((20, 20), dtype="uint32")
    a[0:1, 0:1] = 1
    a[2:5, 2:5] = 7
    a[10:18, 10:18] = 3
    z = zarr_factory(a, chunks=(8, 8))
    zout = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="uint32", fill=0)

    bp.segmentation.size_filter(z, zout, min_size=5, relabel=False, block_shape=(8, 8), num_workers=4)
    res = zout[:]
    assert set(np.unique(res).tolist()) == {0, 3, 7}  # original ids of survivors kept
    assert np.all(res[0:1, 0:1] == 0)


def test_size_filter_parity(zarr_factory, rng):
    binary = (rng.random((40, 36)) > 0.5).astype("uint8")
    a = bic.segmentation.label(binary.astype(bool), connectivity=1).astype("uint32")
    z = zarr_factory(a, chunks=(8, 8))

    ref = bp.segmentation.size_filter(a, min_size=3, max_size=50)  # direct
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        zout = zarr_factory(shape=a.shape, chunks=(8, 8), dtype=a.dtype, fill=0)
        bp.segmentation.size_filter(z, zout, min_size=3, max_size=50, block_shape=(8, 8),
                                    num_workers=nw, job_type=job)
        np.testing.assert_array_equal(zout[:], ref, err_msg=f"size_filter mismatch ({nw}, {job})")


def test_segmentation_filter_generic(zarr_factory, rng):
    # A nested function so cloudpickle serializes it by value (a module-level function in a pytest
    # test file would be pickled by reference and fail to import on the subprocess worker).
    def zero_odd_ids(seg, block_mask):
        out = seg.copy()
        out[(seg % 2) == 1] = 0
        return out

    a = rng.integers(0, 6, size=(24, 24)).astype("uint32")
    z = zarr_factory(a, chunks=(8, 8))
    ref = a.copy()
    ref[(a % 2) == 1] = 0

    out = bp.segmentation.segmentation_filter(a, zero_odd_ids)  # direct
    np.testing.assert_array_equal(out, ref)

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        zout = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="uint32", fill=0)
        bp.segmentation.segmentation_filter(z, zero_odd_ids, zout, block_shape=(8, 8),
                                            num_workers=nw, job_type=job)
        np.testing.assert_array_equal(zout[:], ref)
