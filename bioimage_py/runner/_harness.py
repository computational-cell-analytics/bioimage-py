"""Worker entry point for distributed tasks."""
from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import sys
import time
import traceback
from datetime import datetime

import cloudpickle

from ..sources.dispatch import from_spec
from ..util import get_blocking
from ._diagnostics import load_manifest
from ._reduction import reduce_batch
from ._work import (ReductionBatchPlan, assignment_length, iter_assignment,
                    load_assignment, load_work_spec, logical_size)
from .base import run_block
from .distributed import _COMPLETION_RECORD, _check_versions, _completion_counts


def _truncate_torn_journal(path: str) -> None:
    """Remove an incomplete final completion record before appending."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    valid_size = size - size % _COMPLETION_RECORD.size
    if valid_size != size:
        os.truncate(path, valid_size)


def _truncate_torn_results(path: str) -> None:
    """Remove an incomplete final framed result before appending."""
    try:
        with open(path, "rb") as file:
            valid_size = 0
            while True:
                header = file.read(8)
                if not header:
                    return
                if len(header) < 8:
                    break
                (length,) = struct.unpack("<Q", header)
                payload = file.read(length)
                if len(payload) < length:
                    break
                valid_size += 8 + length
    except FileNotFoundError:
        return
    os.truncate(path, valid_size)


def _run_task(tmp: str, task_id: int, attempt_number: int = 1) -> None:
    with open(os.path.join(tmp, "payload.pkl"), "rb") as f:
        payload = cloudpickle.load(f)

    _check_versions(payload.get("versions", {"python": payload.get("python")}), role="worker")

    manifest = load_manifest(tmp)
    work = load_work_spec(tmp, manifest["work_plan"])
    assignment = load_assignment(tmp, manifest["assignments"][task_id])

    function = payload["function"]
    has_return_val = payload["has_return_val"]
    mode = payload.get("mode", "block")
    result_mode = manifest.get(
        "result_mode", "ordered" if has_return_val else "none",
    )

    # In "map" mode the function carries its own data in its closure (a SourceSpec it
    # reopens, a file path it reads); the runner reopens no sources and builds no blocking.
    if mode == "map":
        def call_item(value):
            return function(int(value))
    elif mode == "batch":
        sink_descriptor = manifest.get("result_sink")
        if sink_descriptor is None:
            call_batch = function
        else:
            from ..tables import TableDataset

            dataset = TableDataset._from_descriptor(sink_descriptor)

            def call_batch(value):
                return dataset._run_batch(function, value)
    else:
        inputs = [from_spec(s) for s in payload["input_specs"]]
        outputs = [from_spec(s) for s in payload["output_specs"]]
        mask = from_spec(payload["mask_spec"]) if payload["mask_spec"] is not None else None
        blocking = get_blocking(payload["shape"], payload["block_shape"], payload["roi"])
        halo = payload["halo"]

        def call_item(value):
            return run_block(function, blocking, int(value), inputs, outputs, mask, halo)

    if result_mode == "reducer":
        if mode == "batch" or not isinstance(work, ReductionBatchPlan):
            raise ValueError("Reducer execution requires block or map reduction work.")
        reducer = payload.get("reducer")
        if reducer is None:
            raise ValueError("Reducer execution has no serialized reducer.")

        def call_reduction_batch(value):
            return reduce_batch(value, work, call_item, reducer, tmp=tmp)

        call_one = call_reduction_batch
    elif mode == "batch":
        call_one = call_batch
    elif mode != "batch":
        call_one = call_item

    journal_path = os.path.join(tmp, "progress", f"{task_id}.bin")
    completed_units, completed_logical_items = _completion_counts(tmp, task_id)
    completed_units = min(completed_units, assignment_length(assignment))
    _truncate_torn_journal(journal_path)

    # Measure task work after payload loading and source setup. This excludes scheduler wait.
    started = datetime.now().isoformat()
    t0 = time.time()
    n_processed = 0
    result_path = os.path.join(tmp, "results", f"{task_id}")
    writes_results = result_mode == "ordered"
    if writes_results:
        _truncate_torn_results(result_path)
    res_f = open(result_path, "ab") if writes_results else None
    journal = open(journal_path, "ab")
    try:
        for position, value in iter_assignment(work, assignment, completed_units):
            res = call_one(value)
            if writes_results:
                payload_bytes = cloudpickle.dumps((position, res))
                res_f.write(struct.pack("<Q", len(payload_bytes)) + payload_bytes)
                res_f.flush()
                os.fsync(res_f.fileno())
            completed_units += 1
            completed_logical_items += logical_size(value)
            journal.write(_COMPLETION_RECORD.pack(completed_units, completed_logical_items))
            journal.flush()
            os.fsync(journal.fileno())
            n_processed += 1
    finally:
        if res_f is not None:
            res_f.close()
        journal.close()
    compute_s = time.time() - t0
    ended = datetime.now().isoformat()

    # Write timing before the sentinel. Each task owns its timing file.
    timing = {
        "task_id": int(task_id),
        "n_units": assignment_length(assignment),
        "n_processed": n_processed,
        "completed_logical_items": completed_logical_items,
        "compute_s": compute_s,
        "started": started,
        "ended": ended,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_nodename": os.environ.get("SLURMD_NODENAME"),
    }
    attempt_timing_dir = os.path.join(tmp, "attempts", f"{attempt_number:04d}", "timings")
    os.makedirs(attempt_timing_dir, exist_ok=True)
    attempt_timing_path = os.path.join(attempt_timing_dir, f"{task_id}.json")
    with open(attempt_timing_path, "w") as f:
        json.dump(timing, f)
    shutil.copyfile(attempt_timing_path, os.path.join(tmp, "timings", f"{task_id}.json"))

    # Sentinel written last: its existence is the ground truth for a fully-completed task.
    open(os.path.join(tmp, "success", f"{task_id}.success"), "w").close()


def main() -> None:
    tmp = sys.argv[1]
    task_id = int(sys.argv[2])
    attempt_number = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    try:
        _run_task(tmp, task_id, attempt_number)
    except Exception:
        error_text = traceback.format_exc()
        attempt_error_dir = os.path.join(tmp, "attempts", f"{attempt_number:04d}", "errors")
        os.makedirs(attempt_error_dir, exist_ok=True)
        attempt_error_path = os.path.join(attempt_error_dir, f"{task_id}.txt")
        with open(attempt_error_path, "w") as f:
            f.write(error_text)
        with open(os.path.join(tmp, "error", f"{task_id}.txt"), "w") as f:
            f.write(error_text)
        sys.exit(1)


if __name__ == "__main__":
    main()
