"""Worker entry point for distributed tasks.

Invoked as ``python -m bioimage_py.runner._harness <tempdir> <task_id>``. Loads the
cloudpickled payload, reopens the sources from their specs, runs the assigned blocks via
the shared :func:`bioimage_py.runner.base.run_block`, and writes one result file followed
by one success sentinel (writing the result first guarantees a sentinel never advertises a
missing or partial result).
"""
from __future__ import annotations

import json
import os
import socket
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

    inputs = [from_spec(s) for s in payload["input_specs"]]
    outputs = [from_spec(s) for s in payload["output_specs"]]
    mask = from_spec(payload["mask_spec"]) if payload["mask_spec"] is not None else None
    blocking = get_blocking(payload["shape"], payload["block_shape"], payload["roi"])
    function = payload["function"]
    halo = payload["halo"]
    has_return_val = payload["has_return_val"]

    # Time only the block-processing loop (the parallelizable read+compute+write work),
    # excluding the fixed per-task payload load / source reopen above and any scheduler
    # queue wait -- this is the "per-worker compute time" basis for scaling analysis.
    started = datetime.now().isoformat()
    t0 = time.time()
    results = []
    for bid in block_ids:
        res = run_block(function, blocking, bid, inputs, outputs, mask, halo)
        if has_return_val:
            results.append((int(bid), res))
    compute_s = time.time() - t0
    ended = datetime.now().isoformat()

    with open(os.path.join(tmp, "results", f"{task_id}.pkl"), "wb") as f:
        cloudpickle.dump(results, f)

    # Per-task timing record, written before the sentinel so a finalized run always finds it.
    # Each task writes its own file (like results/ and success/), so there is no contention.
    timing = {
        "task_id": int(task_id),
        "n_blocks": len(block_ids),
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

    # Sentinel written last: its existence is the ground truth for task success.
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
