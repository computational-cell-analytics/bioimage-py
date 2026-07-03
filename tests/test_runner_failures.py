"""Failure handling: RunnerError, preserved temp folder, block_ids re-run, and resume_from."""
import glob
import os
import time

import numpy as np
import pytest

from bioimage_py.runner import RunnerConfig, RunnerError, SubprocessRunner, get_runner
from bioimage_py.runner import distributed
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


def _make_timeout(marker_path):
    """Per-block fn that hangs on the corner block once (until a marker exists).

    Sleeps far longer than the test's ``task_timeout`` so the worker is killed; writes the
    marker first so the re-run returns immediately (a transient hang that re-run clears).
    """
    def fn(block, inputs, outputs, mask):
        is_corner = all(int(b) == 0 for b in block.begin)
        if is_corner and not os.path.exists(marker_path):
            with open(marker_path, "w") as f:
                f.write("hung once")
            time.sleep(60)
        return int(block.begin[0])

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
    # 16 blocks over 4 contiguous tasks -> task 0 = [0, 1, 2, 3]. With the default
    # tasks_per_worker=1, n_tasks == num_workers, so the contiguous partition holds. Fail the LAST
    # block of task 0 (id 3, begin (0, 48)): blocks 0,1,2 complete and only block 3 is reported
    # failed -- per block, not the whole task. (Pre-change this reported the whole task [0,1,2,3].)
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


def test_task_timeout_reported_and_resumable(zarr_factory, rng, tmp_path):
    # A hung worker must be killed and reported as a normal (resumable) block failure rather
    # than hanging the run. 4 blocks over num_workers=4 (tasks_per_worker=1) -> one block per
    # task, so only the hung corner block's task (block 0) times out.
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    marker = str(tmp_path / "marker.txt")
    fn = _make_timeout(marker)
    # task_timeout is wall-clock over the whole subprocess, so it must comfortably exceed a cold
    # worker's startup (interpreter + heavy imports: numpy/bioimage_cpp/zarr, ~1s+ even warm) on a
    # slow/loaded runner -- otherwise the healthy workers are killed mid-startup too and every block
    # is (flakily) reported failed. Keep it well above that and well below the hung block's sleep.
    runner = SubprocessRunner(RunnerConfig(task_timeout=15, tmp_root=str(tmp_path)))

    with pytest.raises(RunnerError) as excinfo:
        runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True, name="timeout")
    err = excinfo.value
    assert err.failed_block_ids == [0]
    assert err.tmp_folder is not None and os.path.isdir(err.tmp_folder)
    with open(os.path.join(err.tmp_folder, "error", "0.txt")) as f:
        assert "TimeoutError" in f.read().strip().splitlines()[-1]

    # Re-running the timed-out block now succeeds (marker exists -> no hang).
    results = runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True,
                         block_ids=err.failed_block_ids, name="timeout-rerun")
    assert all(r is not None for r in results)


def test_launch_failure_raises_runner_error(zarr_factory, rng, tmp_path):
    # A worker that cannot be launched at all (bad python_executable -> OSError at fork/exec)
    # must surface as the standard RunnerError with a preserved temp folder, not as an escaping
    # OSError that bypasses _finalize (which would also leak the temp folder).
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    fn = _make_flaky(str(tmp_path / "never"))  # never fails; the launch itself does
    runner = SubprocessRunner(RunnerConfig(python_executable="/nonexistent/python",
                                           tmp_root=str(tmp_path)))

    with pytest.raises(RunnerError) as excinfo:
        runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True, name="badpy")
    err = excinfo.value
    assert len(err.failed_block_ids) == 4  # no worker launched -> every block failed
    assert err.tmp_folder is not None and os.path.isdir(err.tmp_folder)  # preserved, not leaked
    err_files = glob.glob(os.path.join(err.tmp_folder, "error", "*.txt"))
    assert err_files
    with open(err_files[0]) as f:
        assert "OSError" in f.read().strip().splitlines()[-1]


def test_check_versions_major_mismatch_raises():
    # A differing MAJOR version is a hard failure (naming the offending library).
    bogus = dict(distributed._env_versions())
    bogus["numpy"] = "999.0.0"
    with pytest.raises(RuntimeError, match="numpy version skew"):
        distributed._check_versions(bogus, role="worker")


@pytest.mark.parametrize("skewed", ["{major}.999.999", "not-a-version"])
def test_check_versions_minor_or_unparseable_only_warns(skewed):
    # A minor/patch difference under a matching major -- or a version that cannot be parsed --
    # warns rather than failing, so routine patch drift does not block a run.
    current = dict(distributed._env_versions())
    major = str(current["numpy"]).split(".", 1)[0]
    current["numpy"] = skewed.format(major=major)
    with pytest.warns(UserWarning, match="version skew"):
        distributed._check_versions(current, role="orchestrator")


def test_check_versions_matching_is_silent(recwarn):
    # An exact match neither raises nor warns; a missing key (older payload) is skipped.
    distributed._check_versions(distributed._env_versions(), role="worker")
    distributed._check_versions({"python": None}, role="worker")
    assert len(recwarn) == 0


def _make_copy_fn():
    """Per-block copy fn as a nested closure so cloudpickle captures it by value."""
    def fn(block, inputs, outputs, mask):
        roi = to_roi(block)
        outputs[0][roi] = inputs[0][roi]
    return fn


def test_version_skew_fails_the_distributed_run(zarr_factory, rng, monkeypatch):
    # Stamp a bogus MAJOR numpy version into the payload; each worker subprocess reads its real
    # numpy, the majors differ, and the raised RuntimeError surfaces as a RunnerError naming numpy.
    bogus = dict(distributed._env_versions())
    bogus["numpy"] = "999.0.0"
    monkeypatch.setattr(distributed, "_env_versions", lambda: bogus)
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="float32", fill=0.0)
    runner = get_runner("subprocess")
    with pytest.raises(RunnerError, match="numpy"):
        runner.run(_make_copy_fn(), [z], outputs=[out], block_shape=(16, 16), num_workers=2,
                   name="verskew")


def test_matching_version_stamp_runs_cleanly(zarr_factory, rng):
    # The happy path: with a real (matching) stamp the extended check is a no-op and the
    # distributed run completes correctly.
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="float32", fill=0.0)
    runner = get_runner("subprocess")
    runner.run(_make_copy_fn(), [z], outputs=[out], block_shape=(16, 16), num_workers=2, name="verok")
    np.testing.assert_array_equal(out[:], a)
