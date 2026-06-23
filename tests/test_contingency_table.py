"""Correctness tests for the contingency_table op, validated against bic.utils.segmentation_overlap."""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp


def _assert_ct_equal(a, b, msg=""):
    """Assert two ContingencyTable objects are element-wise identical."""
    np.testing.assert_array_equal(a.pairs, b.pairs, err_msg=f"pairs {msg}")
    np.testing.assert_array_equal(a.counts, b.counts, err_msg=f"counts {msg}")
    np.testing.assert_array_equal(a.labels_a, b.labels_a, err_msg=f"labels_a {msg}")
    np.testing.assert_array_equal(a.sizes_a, b.sizes_a, err_msg=f"sizes_a {msg}")
    np.testing.assert_array_equal(a.labels_b, b.labels_b, err_msg=f"labels_b {msg}")
    np.testing.assert_array_equal(a.sizes_b, b.sizes_b, err_msg=f"sizes_b {msg}")
    assert a.n_points == b.n_points, f"n_points {msg}"


def test_against_segmentation_overlap(rng):
    # The single-block result must match the C++ primitive on the whole array (the source of truth).
    a = rng.integers(0, 8, size=(40, 48)).astype("uint64")
    b = rng.integers(0, 8, size=(40, 48)).astype("uint64")
    ct = bp.evaluation.contingency_table(a, b)

    ov = bic.utils.segmentation_overlap(a, b)
    ot = ov.overlap_table()
    order = np.lexsort((ot["label_b"], ot["label_a"]))
    np.testing.assert_array_equal(ct.pairs, np.stack([ot["label_a"], ot["label_b"]], axis=1)[order])
    np.testing.assert_array_equal(ct.counts, ot["count"][order])

    counts_a, counts_b = ov.counts_a_table(), ov.counts_b_table()
    np.testing.assert_array_equal(ct.labels_a, counts_a["label"])
    np.testing.assert_array_equal(ct.sizes_a, counts_a["count"])
    np.testing.assert_array_equal(ct.labels_b, counts_b["label"])
    np.testing.assert_array_equal(ct.sizes_b, counts_b["count"])
    assert ct.n_points == ov.total_count == a.size


def test_elf_example():
    # The fixture from elf/test/evaluation/test_evaluation.py::TestContigencyTable.test_simple.
    a = np.array([0, 0, 1, 1, 2, 2], dtype="uint64")
    b = np.array([0, 1, 1, 2, 2, 2], dtype="uint64")
    ct = bp.evaluation.contingency_table(a, b)

    observed = {(int(ia), int(ib)): int(c) for (ia, ib), c in zip(ct.pairs, ct.counts)}
    assert observed == {(0, 0): 1, (0, 1): 1, (1, 1): 1, (1, 2): 1, (2, 2): 2}
    a_dict, b_dict = ct.as_dicts()
    assert a_dict == {0: 2, 1: 2, 2: 2}
    assert b_dict == {0: 1, 1: 2, 2: 3}
    assert ct.n_points == 6


def test_marginals_sum_to_total(rng):
    a = rng.integers(0, 6, size=(30, 30)).astype("uint64")
    b = rng.integers(0, 6, size=(30, 30)).astype("uint64")
    ct = bp.evaluation.contingency_table(a, b)
    assert int(ct.counts.sum()) == ct.n_points == a.size
    assert int(ct.sizes_a.sum()) == ct.n_points
    assert int(ct.sizes_b.sum()) == ct.n_points
    # Every present label shows up in the pairs and in the marginals.
    np.testing.assert_array_equal(ct.labels_a, np.unique(a))
    np.testing.assert_array_equal(ct.labels_b, np.unique(b))


def test_mask(zarr_factory, rng):
    # A masked, blocked computation equals the unmasked table of just the in-mask pixels.
    a = rng.integers(0, 8, size=(40, 48)).astype("uint64")
    b = rng.integers(0, 8, size=(40, 48)).astype("uint64")
    mask = rng.random((40, 48)) > 0.5
    za, zb = zarr_factory(a, chunks=(16, 16)), zarr_factory(b, chunks=(16, 16))

    masked = bp.evaluation.contingency_table(za, zb, num_workers=4, block_shape=(16, 16), mask=mask)
    reference = bp.evaluation.contingency_table(a[mask], b[mask])  # direct, on the 1-D subset
    _assert_ct_equal(masked, reference, msg="masked vs subset")


def test_empty_mask(zarr_factory, rng):
    a = rng.integers(0, 8, size=(32, 32)).astype("uint64")
    b = rng.integers(0, 8, size=(32, 32)).astype("uint64")
    za, zb = zarr_factory(a, chunks=(16, 16)), zarr_factory(b, chunks=(16, 16))
    mask = np.zeros((32, 32), dtype=bool)

    ct = bp.evaluation.contingency_table(za, zb, num_workers=2, block_shape=(16, 16), mask=mask)
    assert ct.pairs.shape == (0, 2)
    assert ct.counts.shape == (0,)
    assert ct.n_points == 0


def test_requires_integer(rng):
    a = np.zeros((8, 8), dtype="float32")
    b = np.zeros((8, 8), dtype="uint64")
    with pytest.raises(ValueError, match="integer"):
        bp.evaluation.contingency_table(a, b)
    with pytest.raises(ValueError, match="integer"):
        bp.evaluation.contingency_table(b, a)


def _keep_or(a, b, ignore_a, ignore_b):
    """The OR keep-mask: drop a voxel if it is an ignore label in either segmentation."""
    drop = np.zeros(a.shape, dtype=bool)
    if ignore_a is not None:
        drop |= np.isin(a, ignore_a)
    if ignore_b is not None:
        drop |= np.isin(b, ignore_b)
    return ~drop


def test_drop_ignore_equivalence(rng):
    # The core guarantee: filtering the merged table == excluding those voxels before counting.
    a = rng.integers(0, 8, size=(40, 48)).astype("uint64")
    b = rng.integers(0, 8, size=(40, 48)).astype("uint64")
    ct = bp.evaluation.contingency_table(a, b)
    for ia, ib in [([0], None), (None, [0]), ([0], [0]), ([1, 3], [2, 5]), ([0, 7], [0, 1, 6])]:
        keep = _keep_or(a, b, ia, ib)
        reference = bp.evaluation.contingency_table(a[keep], b[keep])
        _assert_ct_equal(ct.drop_ignore(ignore_a=ia, ignore_b=ib), reference, msg=f"ia={ia} ib={ib}")


def test_drop_ignore_or_semantics():
    # OR drops a pair if EITHER side is ignored; an AND impl would drop nothing here.
    a = np.array([0, 0, 1], dtype="uint64")
    b = np.array([1, 2, 0], dtype="uint64")  # pairs (0,1), (0,2), (1,0)
    ct = bp.evaluation.contingency_table(a, b).drop_ignore(ignore_a=[0], ignore_b=[0])
    assert ct.pairs.shape == (0, 2)
    assert ct.n_points == 0


def test_drop_ignore_example():
    a = np.array([0, 0, 1, 1, 2, 2], dtype="uint64")
    b = np.array([0, 1, 1, 2, 2, 2], dtype="uint64")
    ct = bp.evaluation.contingency_table(a, b).drop_ignore(ignore_a=[0])
    observed = {(int(ia), int(ib)): int(c) for (ia, ib), c in zip(ct.pairs, ct.counts)}
    assert observed == {(1, 1): 1, (1, 2): 1, (2, 2): 2}
    assert ct.as_dicts() == ({1: 2, 2: 2}, {1: 1, 2: 3})
    assert ct.n_points == 4


def test_drop_ignore_identity_and_marginals(rng):
    a = rng.integers(0, 6, size=(30, 30)).astype("uint64")
    b = rng.integers(0, 6, size=(30, 30)).astype("uint64")
    ct = bp.evaluation.contingency_table(a, b)

    _assert_ct_equal(ct.drop_ignore(), ct, msg="no-op")  # no ignore -> identity

    dropped = ct.drop_ignore(ignore_a=[2], ignore_b=[4])
    assert int(dropped.sizes_a.sum()) == int(dropped.sizes_b.sum()) == dropped.n_points
    assert int(dropped.counts.sum()) == dropped.n_points
    assert 2 not in dropped.labels_a.tolist()
    assert 4 not in dropped.labels_b.tolist()

    # Ignoring every present label on one side empties the table.
    empty = ct.drop_ignore(ignore_a=np.unique(a).tolist())
    assert empty.pairs.shape == (0, 2)
    assert empty.n_points == 0
