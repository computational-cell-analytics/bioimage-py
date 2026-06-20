"""Failure handling: RunnerError, preserved temp folder, block_ids re-run, and resume_from."""
import os

import numpy as np
import pytest

from bioimage_py.runner import RunnerError, get_runner
from bioimage_py.util import to_roi


def _make_flaky(marker_path):
    """Per-block function that fails the first-processed corner block exactly once.

    The failure is recorded via a marker file so that re-running the failed block
    succeeds (simulating a transient failure that re-run fixes).
    """
    def fn(block, inputs, outputs, mask):
        is_corner = all(int(b) == 0 for b in block.begin)
        if is_corner and not os.path.exists(marker_path):
            with open(marker_path, "w") as f:
                f.write("failed once")
            raise RuntimeError("transient boom")
        return int(block.begin[0])

    return fn


def _make_flaky_target(marker_path, fail_begin):
    """Per-block fn that fails the block whose ``begin == fail_begin`` exactly once.

    Writes input->output when an output is given (array-output op), else returns the block's
    integer sum (return-value op). ``fail_begin`` lets a test fail a *non-first* block of a
    task, so the earlier blocks complete and are preserved (exercising per-block recovery).
    """
    def fn(block, inputs, outputs, mask):
        roi = to_roi(block)
        if tuple(int(b) for b in block.begin) == tuple(fail_begin) and not os.path.exists(marker_path):
            with open(marker_path, "w") as f:
                f.write("failed once")
            raise RuntimeError("transient boom")
        if outputs:
            outputs[0][roi] = inputs[0][roi]
            return None
        return int(inputs[0][roi].sum())

    return fn


@pytest.mark.parametrize("job_type", ["local", "subprocess"])
def test_failure_then_rerun(job_type, zarr_factory, rng, tmp_path):
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    marker = str(tmp_path / "marker.txt")
    fn = _make_flaky(marker)
    runner = get_runner(job_type)

    with pytest.raises(RunnerError) as excinfo:
        runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True, name="flaky")
    err = excinfo.value
    assert err.failed_block_ids, "failed block ids should be reported"

    if job_type == "subprocess":
        assert err.tmp_folder is not None and os.path.isdir(err.tmp_folder)
        assert os.path.exists(os.path.join(err.tmp_folder, "source.py"))

    # Re-running the reported failed blocks now succeeds (marker exists).
    results = runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True,
                         block_ids=err.failed_block_ids, name="flaky-rerun")
    assert all(r is not None for r in results)


def test_failed_block_ids_are_precise(zarr_factory, rng, tmp_path):
    # 16 blocks over 4 contiguous tasks -> task 0 = [0, 1, 2, 3]. Fail the LAST block of task 0
    # (id 3, begin (0, 48)): blocks 0,1,2 complete and only block 3 is reported failed -- per
    # block, not the whole task. (Pre-change this reported the whole task [0,1,2,3].)
    a = rng.random((64, 64)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    marker = str(tmp_path / "marker.txt")
    fn = _make_flaky_target(marker, (0, 48))
    runner = get_runner("subprocess")
    with pytest.raises(RunnerError) as excinfo:
        runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True, name="flaky")
    assert excinfo.value.failed_block_ids == [3]


def test_resume_from_merges_return_values(zarr_factory, rng, tmp_path):
    # The #7 merge guarantee: a return-value run that loses one block, then resumes, must return
    # the SAME full per-block result set as a clean run (the survivors are persisted and merged).
    a = rng.random((64, 64)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    runner = get_runner("subprocess")
    clean = runner.run(_make_flaky_target(str(tmp_path / "never"), (-1, -1)), [z],
                       block_shape=(16, 16), num_workers=4, has_return_val=True, name="")

    marker = str(tmp_path / "marker.txt")
    fn = _make_flaky_target(marker, (0, 48))
    with pytest.raises(RunnerError) as excinfo:
        runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True, name="flaky")
    results = runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True,
                         resume_from=excinfo.value.tmp_folder, name="resume")
    assert len(results) == len(clean)
    assert sum(results) == sum(clean)


def test_resume_from_array_output(zarr_factory, rng, tmp_path):
    # Array-output resume: the failed block is re-run into the existing output; the survivors were
    # already written. The output must end up complete and correct.
    a = rng.random((64, 64)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(64, 64), chunks=(16, 16), dtype="float32", fill=0.0)
    marker = str(tmp_path / "marker.txt")
    fn = _make_flaky_target(marker, (0, 48))
    runner = get_runner("subprocess")
    with pytest.raises(RunnerError) as excinfo:
        runner.run(fn, [z], outputs=[out], block_shape=(16, 16), num_workers=4, name="flaky")
    runner.run(fn, [z], outputs=[out], block_shape=(16, 16), num_workers=4,
               resume_from=excinfo.value.tmp_folder, name="resume")
    np.testing.assert_array_equal(out[:], a)
