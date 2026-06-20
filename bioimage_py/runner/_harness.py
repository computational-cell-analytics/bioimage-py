"""Worker entry point for distributed tasks.

Invoked as ``python -m bioimage_py.runner._harness <tempdir> <task_id>``. Loads the
cloudpickled payload, reopens the sources from their specs, and runs the assigned blocks via
the shared :func:`bioimage_py.runner.base.run_block`.

Progress is persisted **per block** so a partially-failed task preserves its completed work
and can be resumed. After each block succeeds the worker appends, in order: (1) for a
return-value task, a length-prefixed result record to ``results/<task_id>``; then (2) the
block id to the done-log ``progress/<task_id>.log``. The block's own output write to the
zarr already happened (durably) inside ``run_block`` before either append, so a crash leaves
at most a result record with no matching done-line -- harmless, since the done-log is the
authority for "which blocks are done" and :func:`_collect` dedups results by block id. The
``success/<task_id>.success`` sentinel is still written last and means "task fully done". On
a resume, blocks already in the done-log are skipped.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import time
import traceback
from datetime import datetime

import cloudpickle

from ..sources.dispatch import from_spec
from ..util import get_blocking
from .base import run_block


def _run_task(tmp: str, task_id: int) -> None:
    with open(os.path.join(tmp, "payload.pkl"), "rb") as f:
        payload = cloudpickle.load(f)

    expected = tuple(payload["python"])
    actual = tuple(sys.version_info[:2])
    if expected != actual:
        raise RuntimeError(
            f"Python version mismatch: payload built with {expected}, worker is {actual}. "
            "The worker environment must match the submitting environment."
        )

    with open(os.path.join(tmp, "blocks", f"{task_id}.json")) as f:
        block_ids = json.load(f)

    function = payload["function"]
    has_return_val = payload["has_return_val"]
    mode = payload.get("mode", "block")

    # In "map" mode the function carries its own data in its closure (a SourceSpec it
    # reopens, a file path it reads); the runner reopens no sources and builds no blocking.
    if mode == "map":
        def call_one(bid):
            return function(int(bid))
    else:
        inputs = [from_spec(s) for s in payload["input_specs"]]
        outputs = [from_spec(s) for s in payload["output_specs"]]
        mask = from_spec(payload["mask_spec"]) if payload["mask_spec"] is not None else None
        blocking = get_blocking(payload["shape"], payload["block_shape"], payload["roi"])
        halo = payload["halo"]

        def call_one(bid):
            return run_block(function, blocking, bid, inputs, outputs, mask, halo)

    # Resume support: skip blocks already recorded as done in this task's done-log.
    done_path = os.path.join(tmp, "progress", f"{task_id}.log")
    already_done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            for line in f:
                if line.endswith("\n"):  # only complete (flushed) lines are authoritative
                    s = line.strip()
                    if s:
                        already_done.add(int(s))

    # Time only the block-processing loop (the parallelizable read+compute+write work),
    # excluding the fixed per-task payload load / source reopen above and any scheduler
    # queue wait -- this is the "per-worker compute time" basis for scaling analysis.
    started = datetime.now().isoformat()
    t0 = time.time()
    n_processed = 0
    # Append mode so a resumed task adds to, rather than truncates, prior progress.
    res_f = open(os.path.join(tmp, "results", f"{task_id}"), "ab") if has_return_val else None
    done_f = open(done_path, "a")
    try:
        for bid in block_ids:
            bid = int(bid)
            if bid in already_done:
                continue
            res = call_one(bid)              # (1) durable output write happens in here
            if has_return_val:               # (2) append framed result, flush
                payload_bytes = cloudpickle.dumps((bid, res))
                res_f.write(struct.pack("<Q", len(payload_bytes)) + payload_bytes)
                res_f.flush()
            done_f.write(f"{bid}\n")         # (3) append done-mark, flush
            done_f.flush()
            n_processed += 1
    finally:
        if res_f is not None:
            res_f.close()
        done_f.close()
    compute_s = time.time() - t0
    ended = datetime.now().isoformat()

    # Per-task timing record, written before the sentinel so a finalized run always finds it.
    # Each task writes its own file (like results/ and progress/), so there is no contention.
    timing = {
        "task_id": int(task_id),
        "n_blocks": len(block_ids),
        "n_processed": n_processed,  # excludes blocks skipped because already done (resume)
        "compute_s": compute_s,
        "started": started,
        "ended": ended,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_nodename": os.environ.get("SLURMD_NODENAME"),
    }
    with open(os.path.join(tmp, "timings", f"{task_id}.json"), "w") as f:
        json.dump(timing, f)

    # Sentinel written last: its existence is the ground truth for a fully-completed task.
    open(os.path.join(tmp, "success", f"{task_id}.success"), "w").close()


def main() -> None:
    tmp = sys.argv[1]
    task_id = int(sys.argv[2])
    try:
        _run_task(tmp, task_id)
    except Exception:
        with open(os.path.join(tmp, "error", f"{task_id}.txt"), "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
