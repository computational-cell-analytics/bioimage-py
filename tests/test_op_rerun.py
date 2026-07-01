"""Op-level rerun surface: block_ids/item_ids subset runs, the resume_from/subset guards, and
which ops accept which rerun arguments.

The genuine failure -> resume -> correct-merged-result guarantee is proven at the runner level in
``test_runner_failures.py`` (a cloudpickled flaky closure crosses the subprocess boundary, which a
test-defined fault source cannot). End-to-end op resume then follows transitively: an op resume
calls ``runner.run(resume_from=...)`` which returns the full merged per-block result set (the runner
test), and the op's reduction over that full set is covered by the parity tests. Here we verify the
ops expose and correctly plumb the rerun arguments.
"""
import inspect

import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.util import check_rerun_args


@pytest.mark.parametrize("job_type", ["local", "subprocess"])
def test_copy_block_ids_subset(zarr_factory, rng, job_type):
    # block_ids restricts a fresh run to those blocks (written into the existing output).
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))  # 2x2 = 4 blocks
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="float32", fill=0.0)
    bp.copy(z, out, block_shape=(16, 16), num_workers=2, job_type=job_type, block_ids=[0])
    np.testing.assert_array_equal(out[0:16, 0:16], a[0:16, 0:16])  # block 0 written
    assert np.all(out[0:16, 16:32] == 0)  # the other blocks untouched
    assert np.all(out[16:32, :] == 0)


def test_filter_block_ids_subset(zarr_factory, rng):
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="float32", fill=0.0)
    bp.filters.gaussian_smoothing(z, 1.0, output=out, block_shape=(16, 16), num_workers=2,
                                  job_type="subprocess", block_ids=[0])
    assert np.any(out[0:16, 0:16] != 0)    # block 0 was processed
    assert np.all(out[16:32, 16:32] == 0)  # block 3 was not


@pytest.mark.parametrize("dict_mode", [False, True])
def test_relabel_block_ids_subset(zarr_factory, rng, dict_mode):
    # block_ids restricts a fresh run to those blocks (written into the existing output). The
    # array mode also exercises persisting the numpy node labels for the subprocess backend.
    a = rng.integers(1, 8, size=(32, 32)).astype("uint64")  # 2x2 = 4 blocks
    labels = (np.arange(int(a.max()) + 1) * 5 + 1).astype("uint64")
    labels[0] = 0
    node_labels = {i: int(v) for i, v in enumerate(labels)} if dict_mode else labels
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="uint64", fill=0)
    bp.segmentation.relabel(z, node_labels, out, block_shape=(16, 16), num_workers=2,
                            job_type="subprocess", block_ids=[0])
    np.testing.assert_array_equal(out[0:16, 0:16], np.take(labels, a[0:16, 0:16]))  # block 0
    assert np.all(out[0:16, 16:32] == 0)  # the other blocks untouched
    assert np.all(out[16:32, :] == 0)


def test_relabel_resume_from_and_subset_mutually_exclusive(zarr_factory, rng):
    a = rng.integers(1, 8, size=(32, 32)).astype("uint64")
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="uint64", fill=0)
    labels = np.arange(int(a.max()) + 1, dtype="uint64")
    with pytest.raises(ValueError, match="not both"):
        bp.segmentation.relabel(z, labels, out, block_shape=(16, 16), block_ids=[0],
                                resume_from="/x")


def test_regionprops_item_ids_subset(zarr_factory):
    seg = np.zeros((24, 32, 28), dtype="uint64")
    seg[2:9, 3:14, 4:12] = 1
    seg[12:20, 18:30, 15:26] = 2
    z = zarr_factory(seg, chunks=(16, 16, 16))
    table = bp.morphology.morphology(seg)
    out = bp.morphology.regionprops(z, table, num_workers=2, job_type="subprocess", item_ids=[0])
    assert len(out) == 1  # only the one requested object


def test_resume_from_and_subset_mutually_exclusive(zarr_factory, rng):
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="float32", fill=0.0)
    with pytest.raises(ValueError, match="not both"):
        bp.stats.mean(z, block_shape=(16, 16), block_ids=[0], resume_from="/x")
    with pytest.raises(ValueError, match="not both"):
        bp.copy(z, out, block_shape=(16, 16), block_ids=[0], resume_from="/x")
    with pytest.raises(ValueError, match="not both"):
        bp.filters.gaussian_smoothing(z, 1.0, output=out, block_shape=(16, 16),
                                      block_ids=[0], resume_from="/x")


def test_resume_from_rejected_on_local(zarr_factory, rng):
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    with pytest.raises(ValueError, match="distributed"):
        bp.stats.mean(z, block_shape=(16, 16), job_type="local", resume_from="/x")


def test_label_has_no_rerun_args():
    # label is multi-stage with a global merge -> re-run whole, no block_ids / resume_from.
    params = inspect.signature(bp.segmentation.label).parameters
    assert "resume_from" not in params
    assert "block_ids" not in params


def test_check_rerun_args():
    check_rerun_args("subprocess", None, None)          # nothing set
    check_rerun_args("subprocess", "/tmp/x", None)      # resume on a distributed backend
    check_rerun_args("local", None, [0])                # a subset on local is fine
    with pytest.raises(ValueError, match="not both"):
        check_rerun_args("subprocess", "/tmp/x", [0])
    with pytest.raises(ValueError, match="distributed"):
        check_rerun_args("local", "/tmp/x", None)
    with pytest.raises(ValueError, match="item_ids"):
        check_rerun_args("subprocess", "/tmp/x", [0], subset_name="item_ids")
