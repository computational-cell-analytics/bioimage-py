"""Over-partitioning via ``tasks_per_worker``: task-count resolution and the emitted partition.

These need no real ``sbatch`` -- the subprocess backend exercises the shared partition logic
(``_run_ids`` / ``_resolve_n_tasks``) that slurm inherits, and the slurm-specific cap is checked
against a ``max_array_size`` set in the config (so ``scontrol`` is never queried).
"""
import numpy as np

from bioimage_py.runner import RunnerConfig, SlurmConfig, SlurmRunner, SubprocessRunner, get_runner
from bioimage_py.runner._diagnostics import load_manifest
from bioimage_py.runner._work import iter_assignment, load_assignment, load_work_spec


def _make_return_one():
    """Build a trivial per-block function (nested so cloudpickle serializes it by value).

    A module-level function would be pickled by reference, which the worker subprocess cannot
    import (the test module is not on its path); a closure is pickled by value.
    """
    def _compute(block, inputs, outputs, mask):
        return 1
    return _compute


def test_resolve_n_tasks_over_partitions():
    # tasks_per_worker multiplies the worker count into more, smaller tasks.
    runner = SubprocessRunner(RunnerConfig(tasks_per_worker=3))
    assert runner._resolve_n_tasks(num_workers=2, n_items=16) == 6
    # Default (1) reproduces the historical one-task-per-worker partition.
    default = SubprocessRunner(RunnerConfig())
    assert default._resolve_n_tasks(num_workers=4, n_items=16) == 4


def test_resolve_n_tasks_clamped_to_items():
    # Never more tasks than schedulable units, however large the requested factor.
    runner = SubprocessRunner(RunnerConfig(tasks_per_worker=10))
    assert runner._resolve_n_tasks(num_workers=8, n_items=4) == 4
    # Empty work resolves to a single (empty) task.
    assert runner._resolve_n_tasks(num_workers=8, n_items=0) == 1


def test_resolve_n_tasks_defensive_on_bad_values():
    # A non-positive tasks_per_worker / num_workers cannot drop the task count below 1.
    runner = SubprocessRunner(RunnerConfig(tasks_per_worker=0))
    assert runner._resolve_n_tasks(num_workers=4, n_items=16) == 4
    assert runner._resolve_n_tasks(num_workers=0, n_items=16) == 1


def test_slurm_max_tasks_caps_partition():
    # The slurm cap comes from config.max_array_size here (no scontrol call).
    cfg = SlurmConfig(tmp_root="/scratch/shared", max_array_size=1000, tasks_per_worker=50)
    runner = SlurmRunner(cfg)
    assert runner._max_tasks() == 1000
    # 100 workers * 50 = 5000 desired, clamped to the array size.
    assert runner._resolve_n_tasks(num_workers=100, n_items=100_000) == 1000


def test_partition_written_reflects_tasks_per_worker(zarr_factory, rng, tmp_path):
    # A 64x64 array in 16x16 blocks is 16 blocks; num_workers=2 * tasks_per_worker=3 => 6 tasks.
    a = rng.random((64, 64)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    captured = {}

    def _capture(tmp_folder):
        manifest = load_manifest(tmp_folder)
        work = load_work_spec(tmp_folder, manifest["work_plan"])
        assignments = [load_assignment(tmp_folder, value)
                       for value in manifest["assignments"]]
        captured["n_tasks"] = len(assignments)
        captured["blocks"] = sorted(
            int(value)
            for assignment in assignments
            for _, value in iter_assignment(work, assignment)
        )

    runner = get_runner("subprocess", RunnerConfig(tmp_root=str(tmp_path), tasks_per_worker=3))
    runner.run(_make_return_one(), [z], block_shape=(16, 16), num_workers=2, has_return_val=True,
               pre_cleanup=_capture, name="over-partition")

    assert captured["n_tasks"] == 6
    # The over-partitioned tasks still cover every block exactly once (complete + disjoint).
    assert captured["blocks"] == list(range(16))


def test_over_partition_result_matches_single_task(zarr_factory, rng, tmp_path):
    # Over-partitioning changes scheduling, not results: a return-value run over 6 tasks returns
    # the same per-block set as the default 2-task partition.
    a = rng.random((64, 64)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    kw = dict(block_shape=(16, 16), num_workers=2, has_return_val=True, name="")

    ref = get_runner(
        "subprocess", RunnerConfig(tmp_root=str(tmp_path))
    ).run(_make_return_one(), [z], **kw)
    over = get_runner(
        "subprocess", RunnerConfig(tmp_root=str(tmp_path), tasks_per_worker=3)
    ).run(_make_return_one(), [z], **kw)
    assert len(ref) == len(over) == 16
    np.testing.assert_array_equal(ref, over)
