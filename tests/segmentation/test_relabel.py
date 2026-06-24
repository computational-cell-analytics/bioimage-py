"""Tests for block-wise consecutive relabeling."""
import numpy as np

import bioimage_py as bp


def _expected_relabel(a, start_label=0, keep_zeros=True):
    """Reference relabeling via a plain python mapping over the global unique values."""
    uniques = np.unique(a)
    mapping = {int(v): i for i, v in enumerate(uniques.tolist(), start_label)}
    if keep_zeros and 0 in mapping:
        mapping[0] = 0
    out = np.zeros_like(a)
    for old, new in mapping.items():
        out[a == old] = new
    return out, mapping


def test_relabel_matches_reference(zarr_factory, rng):
    a = (rng.integers(0, 8, size=(33, 28)) * 7).astype("uint32")  # non-consecutive ids 0, 7, 14, ...
    exp, exp_map = _expected_relabel(a)
    z = zarr_factory(a, chunks=(8, 8))

    out, max_id, mapping = bp.segmentation.relabel_consecutive(a)  # direct
    np.testing.assert_array_equal(out, exp)
    assert mapping == exp_map
    assert max_id == max(exp_map.values())

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        zout = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="uint32", fill=0)
        out2, max_id2, map2 = bp.segmentation.relabel_consecutive(
            z, zout, block_shape=(8, 8), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(zout[:], exp)
        assert max_id2 == max_id and map2 == mapping


def test_relabel_start_label_keep_zeros(rng):
    a = (rng.integers(0, 6, size=(20, 20)) * 3).astype("uint32")  # 0, 3, 6, ...
    exp, exp_map = _expected_relabel(a, start_label=1, keep_zeros=True)
    out, _, mapping = bp.segmentation.relabel_consecutive(
        a, block_shape=(8, 8), num_workers=2, start_label=1, keep_zeros=True)
    np.testing.assert_array_equal(out, exp)
    assert mapping == exp_map
    assert mapping[0] == 0  # background preserved despite start_label=1


def test_relabel_mask(rng):
    a = (rng.integers(0, 6, size=(20, 20)) * 4).astype("uint32")
    mask = np.zeros((20, 20), dtype="uint8")
    mask[2:14, 3:15] = 1  # not block-aligned
    m = mask.astype(bool)

    out, _, mapping = bp.segmentation.relabel_consecutive(
        a, block_shape=(5, 5), num_workers=2, mask=mask)

    uniques = np.unique(a[m])
    exp_map = {int(v): i for i, v in enumerate(uniques.tolist())}
    if 0 in exp_map:
        exp_map[0] = 0
    assert mapping == exp_map
    exp = np.zeros_like(a)  # out-of-mask output voxels stay 0.
    for old, new in exp_map.items():
        exp[(a == old) & m] = new
    np.testing.assert_array_equal(out, exp)
