"""Tests for block-wise relabeling: ``relabel`` (explicit map) and ``relabel_consecutive``."""
import numpy as np
import pytest

import bioimage_py as bp


def _labeling(a):
    """Build a dense array and equivalent dict relabeling over the ids present in ``a``."""
    max_id = int(a.max())
    # A deterministic, non-identity relabeling covering the full id range 0..max_id.
    labels = ((np.arange(max_id + 1) * 5 + 1) % 97).astype("uint64")
    labels[0] = 0  # keep background at 0
    mapping = {i: int(labels[i]) for i in range(max_id + 1)}
    return labels, mapping


def _expected_relabel(a, start_label=0, keep_zeros=True):
    """Reference consecutive relabeling via a plain python mapping over the global unique values."""
    uniques = np.unique(a)
    mapping = {int(v): i for i, v in enumerate(uniques.tolist(), start_label)}
    if keep_zeros and 0 in mapping:
        mapping[0] = 0
    out = np.zeros_like(a)
    for old, new in mapping.items():
        out[a == old] = new
    return out, mapping


# ---------------------------------------------------------------------------
# relabel (explicit labeling)
# ---------------------------------------------------------------------------

def test_relabel_array_parity(zarr_factory, rng):
    a = rng.integers(0, 12, size=(33, 28)).astype("uint32")
    labels, _ = _labeling(a)
    exp = np.take(labels, a)
    z = zarr_factory(a, chunks=(8, 8))

    np.testing.assert_array_equal(bp.segmentation.relabel(a, labels), exp)  # direct

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        zlabels = zarr_factory(labels, chunks=(labels.shape[0],))
        zout = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="uint64", fill=0)
        bp.segmentation.relabel(z, zlabels, zout, block_shape=(8, 8), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(zout[:], exp, err_msg=f"array nw={nw} job={job}")


def test_relabel_dict_parity(zarr_factory, rng):
    a = rng.integers(0, 12, size=(30, 26)).astype("uint64")
    labels, mapping = _labeling(a)
    exp = np.take(labels, a)
    z = zarr_factory(a, chunks=(8, 8))

    np.testing.assert_array_equal(bp.segmentation.relabel(a, mapping), exp)  # direct

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        zout = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="uint64", fill=0)
        bp.segmentation.relabel(z, mapping, zout, block_shape=(8, 8), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(zout[:], exp, err_msg=f"dict nw={nw} job={job}")


def test_relabel_numpy_not_mutated(rng):
    # A plain numpy input (local-only) is never relabeled in place: a fresh array is returned.
    a = rng.integers(0, 10, size=(20, 20)).astype("uint32")
    labels, _ = _labeling(a)
    exp = np.take(labels, a)
    orig = a.copy()
    out = bp.segmentation.relabel(a, labels)  # output omitted, numpy input
    assert out is not a
    np.testing.assert_array_equal(a, orig)  # input left untouched
    np.testing.assert_array_equal(out, exp)


def test_relabel_in_place_blockwise(zarr_factory, rng):
    # A file-backed source with output omitted IS relabeled in place and returned.
    a = rng.integers(0, 10, size=(24, 24)).astype("uint64")
    labels, _ = _labeling(a)
    exp = np.take(labels, a)
    z = zarr_factory(a, chunks=(8, 8))

    out = bp.segmentation.relabel(z, labels, block_shape=(8, 8), num_workers=3,
                                  job_type="subprocess")  # in place, numpy labeling persisted
    assert out is z
    np.testing.assert_array_equal(z[:], exp)


def test_relabel_numpy_labeling_distributed_persist(zarr_factory, rng):
    # A numpy labeling array with a file-backed segmentation on a distributed backend: the labeling
    # is persisted to a temp zarr and reopened by the worker tasks (the subprocess backend is the CI
    # proxy for slurm).
    a = rng.integers(0, 15, size=(40, 32)).astype("uint32")
    labels, _ = _labeling(a)
    exp = np.take(labels, a)
    z = zarr_factory(a, chunks=(16, 16))
    zout = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="uint64", fill=0)

    bp.segmentation.relabel(z, labels, zout, block_shape=(16, 16), num_workers=3,
                            job_type="subprocess")
    np.testing.assert_array_equal(zout[:], exp)


def test_relabel_mask(rng):
    a = rng.integers(0, 8, size=(20, 20)).astype("uint32")
    labels, mapping = _labeling(a)
    mask = np.zeros((20, 20), dtype="uint8")
    mask[2:14, 3:15] = 1  # not block-aligned
    m = mask.astype(bool)

    exp = a.astype("uint64")
    exp[m] = np.take(labels, a[m])  # out-of-mask voxels keep their original ids

    for labeling in (labels, mapping):
        # Start the output from the input so out-of-mask voxels are the original ids.
        out = a.astype("uint64")
        bp.segmentation.relabel(a.astype("uint64"), labeling, out, block_shape=(5, 5),
                                num_workers=2, mask=mask)
        np.testing.assert_array_equal(out, exp)


def test_relabel_dict_subsample(monkeypatch, rng):
    # Force the gated subsampling path with a small threshold (instead of a 100k-entry dict), then
    # exercise both branches of the diversity gate. Correctness must match the dense reference.
    import importlib
    rl = importlib.import_module("bioimage_py.segmentation.relabel")  # the module (not the re-exported fn)
    monkeypatch.setattr(rl, "_RELABEL_SUBSAMPLE_MIN_DICT", 4)

    labels = ((np.arange(64, dtype="uint64") * 5 + 1) % 97).astype("uint64")
    labels[0] = 0
    mapping = {i: int(labels[i]) for i in range(labels.shape[0])}

    # (a) low-diversity block: few distinct ids -> the subsample branch (small dict) runs.
    few = rng.choice(np.arange(64), size=(24, 24)).astype("uint64")
    few = np.where(few < 4, few, few % 4)  # only ids {0,1,2,3} present -> len(unique)*8 < 64
    out_a = np.zeros_like(few)
    bp.segmentation.relabel(few, mapping, out_a, block_shape=(8, 8), num_workers=2)
    np.testing.assert_array_equal(out_a, np.take(labels, few))

    # (b) high-diversity block: many distinct ids -> the full-dict fallback runs.
    many = rng.integers(0, 64, size=(24, 24)).astype("uint64")
    out_b = np.zeros_like(many)
    bp.segmentation.relabel(many, mapping, out_b, block_shape=(8, 8), num_workers=2)
    np.testing.assert_array_equal(out_b, np.take(labels, many))


def test_relabel_errors(rng):
    a = rng.integers(0, 8, size=(16, 16)).astype("uint32")
    labels, _ = _labeling(a)

    # A numpy input cannot be relabeled in place on a distributed backend (not file-backed).
    with pytest.raises(ValueError, match="numpy"):
        bp.segmentation.relabel(a, labels, block_shape=(8, 8), num_workers=2, job_type="subprocess")

    # A dense labeling must be 1D.
    with pytest.raises(ValueError, match="1D"):
        bp.segmentation.relabel(a, np.stack([labels, labels]))

    # A non-integer input segmentation is rejected.
    with pytest.raises(ValueError, match="integer"):
        bp.segmentation.relabel(a.astype("float32"), labels)


# ---------------------------------------------------------------------------
# relabel_consecutive (derived map)
# ---------------------------------------------------------------------------

def test_relabel_matches_reference(zarr_factory, rng):
    a = (rng.integers(0, 8, size=(33, 28)) * 7).astype("uint32")  # non-consecutive ids 0, 7, 14, ...
    exp, exp_map = _expected_relabel(a)
    z = zarr_factory(a, chunks=(8, 8))

    out, max_id, mapping = bp.segmentation.relabel_consecutive(a)  # direct (numpy input not mutated)
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


def test_relabel_consecutive_in_place(zarr_factory, rng):
    a = (rng.integers(0, 8, size=(24, 24)) * 7).astype("uint32")  # non-consecutive ids
    exp, exp_map = _expected_relabel(a)

    # numpy input, output omitted -> NOT mutated; a fresh array is returned.
    orig = a.copy()
    out, max_id, mapping = bp.segmentation.relabel_consecutive(a)
    assert out is not a
    np.testing.assert_array_equal(a, orig)  # input left untouched
    np.testing.assert_array_equal(out, exp)
    assert mapping == exp_map

    # block-wise distributed, output omitted -> in place on the input zarr
    z = zarr_factory(a, chunks=(8, 8))
    out2, _, map2 = bp.segmentation.relabel_consecutive(
        z, block_shape=(8, 8), num_workers=3, job_type="subprocess")
    assert out2 is z
    np.testing.assert_array_equal(z[:], exp)
    assert map2 == exp_map


def test_relabel_consecutive_mask(rng):
    a = (rng.integers(0, 6, size=(20, 20)) * 4).astype("uint32")
    mask = np.zeros((20, 20), dtype="uint8")
    mask[2:14, 3:15] = 1  # not block-aligned
    m = mask.astype(bool)

    out = np.zeros_like(a)  # separate output; out-of-mask voxels stay 0.
    _, _, mapping = bp.segmentation.relabel_consecutive(
        a, out, block_shape=(5, 5), num_workers=2, mask=mask)

    uniques = np.unique(a[m])
    exp_map = {int(v): i for i, v in enumerate(uniques.tolist())}
    if 0 in exp_map:
        exp_map[0] = 0
    assert mapping == exp_map
    exp = np.zeros_like(a)
    for old, new in exp_map.items():
        exp[(a == old) & m] = new
    np.testing.assert_array_equal(out, exp)
