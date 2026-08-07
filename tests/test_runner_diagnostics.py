"""Structured runner failures and durable attempt diagnostics."""
import json
import os
import subprocess
from dataclasses import FrozenInstanceError

import cloudpickle
import pytest

from bioimage_py.runner import RunnerError, SlurmConfig, SlurmRunner, TaskFailure, get_runner
from bioimage_py.runner import distributed
from bioimage_py.runner._diagnostics import (append_attempt, create_manifest, load_manifest,
                                             read_task_outcome, update_attempt, write_task_outcome)


def _make_failing_item():
    def fail_item(item_id):
        if item_id == 1:
            raise RuntimeError("diagnostic boom")
        return item_id
    return fail_item


def test_task_failure_is_frozen_and_runner_error_defaults():
    failure = TaskFailure(task_id=3, backend="subprocess", attempt_number=2)
    with pytest.raises(FrozenInstanceError):
        failure.task_id = 4

    error = RunnerError("boom", [3], "/tmp/run")
    assert error.failed_block_ids == [3]
    assert error.tmp_folder == "/tmp/run"
    assert error.task_failures == []


def test_subprocess_failure_has_structured_diagnostics(tmp_path):
    runner = get_runner("subprocess")
    runner.config.tmp_root = str(tmp_path)

    with pytest.raises(RunnerError) as excinfo:
        runner.map(_make_failing_item(), n_items=3, num_workers=2, name="diagnostics")

    error = excinfo.value
    assert error.failed_block_ids == [1]
    assert len(error.task_failures) == 1
    failure = error.task_failures[0]
    assert failure.task_id == 0  # task 0 contains item ids 0 and 1
    assert failure.backend == "subprocess"
    assert failure.attempt_number == 1
    assert failure.exit_code == 1
    assert failure.signal is None
    assert failure.elapsed_seconds is not None and failure.elapsed_seconds > 0
    assert failure.stdout_path is not None and os.path.isfile(failure.stdout_path)
    assert failure.stderr_path is not None and os.path.isfile(failure.stderr_path)
    assert failure.traceback_path is not None and os.path.isfile(failure.traceback_path)
    with open(failure.traceback_path) as f:
        assert "diagnostic boom" in f.read()

    manifest = load_manifest(error.tmp_folder, expected_backend="subprocess")
    assert manifest["schema_version"] == 2
    assert manifest["work_plan"] == {
        "kind": "range", "start": 0, "stop": 3, "step": 1, "length": 3,
    }
    assert manifest["attempts"][0]["status"] == "failed"
    assert manifest["attempts"][0]["outcomes_path"] == "attempts/0001/tasks"


def test_subprocess_resume_appends_attempt_history(zarr_factory, rng, tmp_path):
    data = rng.random((32, 32)).astype("float32")
    source = zarr_factory(data, chunks=(16, 16))
    marker = str(tmp_path / "marker")

    def flaky(block, inputs, outputs, mask):
        if block.begin == [0, 0] and not os.path.exists(marker):
            open(marker, "w").close()
            raise RuntimeError("once")
        return int(block.begin[0])

    runner = get_runner("subprocess")
    runner.config.tmp_root = str(tmp_path)
    with pytest.raises(RunnerError) as excinfo:
        runner.run(flaky, [source], block_shape=(16, 16), num_workers=2,
                   has_return_val=True)

    captured = {}

    def inspect_run_folder(tmp):
        captured.update(load_manifest(tmp, expected_backend="subprocess"))

    results = runner.run(flaky, [source], block_shape=(16, 16), num_workers=2,
                         has_return_val=True, resume_from=excinfo.value.tmp_folder,
                         pre_cleanup=inspect_run_folder)
    assert len(results) == 4
    assert [attempt["attempt_number"] for attempt in captured["attempts"]] == [1, 2]
    assert [attempt["status"] for attempt in captured["attempts"]] == ["failed", "succeeded"]


def _prepare_slurm_attempt(tmp_path, task_ids=(2, 3)):
    tmp = str(tmp_path)
    for folder in ("error", "progress", "results", "success", "timings", "work"):
        os.makedirs(os.path.join(tmp, folder), exist_ok=True)
    create_manifest(tmp, backend="slurm", name="test", n_tasks=4,
                    python_executable="/usr/bin/python")
    attempt = append_attempt(tmp, task_ids=task_ids, concurrency=2)
    update_attempt(tmp, attempt["attempt_number"], status="running", job_id="900")
    return tmp, int(attempt["attempt_number"])


def test_slurm_accounting_merges_array_and_batch_rows(tmp_path, monkeypatch):
    tmp, attempt_number = _prepare_slurm_attempt(tmp_path)
    open(os.path.join(tmp, "success", "3.success"), "w").close()
    error_path = os.path.join(tmp, "attempts", "0001", "errors", "2.txt")
    with open(error_path, "w") as f:
        f.write("worker traceback\n")

    stdout = "\n".join([
        "900_2|FAILED|1:9|12|||node02|node02|out-pattern|err-pattern",
        "900_2.batch|FAILED|1:9|11|2048K|node02|node02|||",
        "900_3|COMPLETED|0:0|8|||node03||out-pattern|err-pattern",
        "900_3.batch|COMPLETED|0:0|8||||||",
    ])

    monkeypatch.setattr(distributed.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        distributed.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=""),
    )

    runner = SlurmRunner(SlurmConfig(tmp_root=tmp, max_array_size=1000))
    accounting_path = runner._persist_slurm_outcomes(
        tmp, attempt_number, "900", [2, 3], {2: "FAILED", 3: "COMPLETED"}
    )

    failed = read_task_outcome(tmp, attempt_number, 2)
    assert failed["scheduler_task_id"] == "900_2"
    assert failed["scheduler_state"] == "FAILED"
    assert failed["exit_code"] == 1
    assert failed["signal"] == 9
    assert failed["elapsed_seconds"] == 12.0
    assert failed["peak_rss_bytes"] == 2048 * 1024
    assert failed["failed_node"] == "node02"
    assert failed["traceback_path"] == "attempts/0001/errors/2.txt"
    assert failed["succeeded"] is False

    succeeded = read_task_outcome(tmp, attempt_number, 3)
    assert succeeded["scheduler_state"] == "COMPLETED"
    assert succeeded["exit_code"] == 0
    assert succeeded["signal"] is None
    assert succeeded["peak_rss_bytes"] is None
    assert succeeded["failed_node"] is None
    assert succeeded["succeeded"] is True

    with open(accounting_path) as f:
        accounting = json.load(f)
    assert len(accounting["rows"]) == 4
    assert accounting["last_poll_states"] == {"2": "FAILED", "3": "COMPLETED"}


def test_slurm_accounting_falls_back_to_core_fields(tmp_path, monkeypatch):
    tmp, attempt_number = _prepare_slurm_attempt(tmp_path, task_ids=(2,))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="field unavailable")
        return subprocess.CompletedProcess(cmd, 0, stdout="900_2|TIMEOUT|0:15|30|node02\n", stderr="")

    monkeypatch.setattr(distributed.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(distributed.subprocess, "run", fake_run)

    runner = SlurmRunner(SlurmConfig(tmp_root=tmp, max_array_size=1000))
    runner._persist_slurm_outcomes(tmp, attempt_number, "900", [2], {2: "TIMEOUT"})
    outcome = read_task_outcome(tmp, attempt_number, 2)
    assert len(calls) == 2
    assert outcome["scheduler_state"] == "TIMEOUT"
    assert outcome["exit_code"] == 0
    assert outcome["signal"] == 15
    assert outcome["elapsed_seconds"] == 30.0
    assert outcome["peak_rss_bytes"] is None


def test_reattach_uses_persisted_terminal_outcome(tmp_path, monkeypatch):
    tmp = str(tmp_path)
    for folder in ("error", "progress", "results", "success", "timings", "work"):
        os.makedirs(os.path.join(tmp, folder), exist_ok=True)
    with open(os.path.join(tmp, "payload.pkl"), "wb") as f:
        cloudpickle.dump({
            "has_return_val": False,
            "versions": distributed._env_versions(),
        }, f)

    create_manifest(
        tmp, backend="slurm", name="forgotten", n_tasks=1,
        python_executable="/usr/bin/python", mode="map",
        work_plan={"kind": "range", "start": 7, "stop": 8, "step": 1, "length": 1},
        assignments=[{"kind": "slice", "start": 0, "stop": 1}], logical_items=1,
    )
    append_attempt(tmp, task_ids=[0], concurrency=1)
    update_attempt(tmp, 1, status="failed", job_id="900")
    write_task_outcome(tmp, 1, {
        "task_id": 0,
        "backend": "slurm",
        "attempt_number": 1,
        "scheduler_task_id": "900_0",
        "scheduler_state": "OUT_OF_MEMORY",
        "exit_code": 0,
        "signal": 9,
        "elapsed_seconds": 10.0,
        "peak_rss_bytes": None,
        "failed_node": "node02",
        "stdout_path": None,
        "stderr_path": None,
        "traceback_path": None,
        "succeeded": False,
        "summary": None,
    })

    runner = SlurmRunner(SlurmConfig(tmp_root=tmp, max_array_size=1000))
    monkeypatch.setattr(runner, "_job_known",
                        lambda job_id: pytest.fail("reattach queried a forgotten job"))
    with pytest.raises(RunnerError) as excinfo:
        runner.reattach(tmp)
    assert excinfo.value.failed_block_ids == [7]
    assert excinfo.value.task_failures[0].scheduler_state == "OUT_OF_MEMORY"


@pytest.mark.parametrize("method", ["resume", "reattach"])
def test_old_manifest_schema_is_rejected(tmp_path, method):
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump({"job_id": "123", "n_tasks": 1}, f)
    runner = SlurmRunner(SlurmConfig(tmp_root=str(tmp_path), max_array_size=1000))
    with pytest.raises(ValueError, match="Unsupported run manifest schema"):
        getattr(runner, method)(str(tmp_path))
