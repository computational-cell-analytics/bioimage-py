"""Unit tests for the distributed-runner progress helpers.

``_DoneLogCounter`` drives the block-counting progress bar by reading each per-task done-log
incrementally (per-task byte offset + running count). These assert it matches a full recount,
defers a torn final line, and tolerates missing / late-appearing logs -- without a real runner.
"""
import os

from bioimage_py.runner.distributed import _DoneLogCounter


def _progress_dir(tmp_path):
    """Create and return the ``progress`` subfolder the counter reads from."""
    progress = os.path.join(str(tmp_path), "progress")
    os.makedirs(progress)
    return progress


def _append(progress, task_id, text):
    """Append raw text to a task's done-log (no implicit newline)."""
    with open(os.path.join(progress, f"{task_id}.log"), "a") as f:
        f.write(text)


def _full_recount(tmp, n_tasks):
    """Reference count: total complete (newline-terminated) lines across all logs."""
    total = 0
    for t in range(n_tasks):
        path = os.path.join(tmp, "progress", f"{t}.log")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            total += f.read().count(b"\n")
    return total


def test_counter_matches_full_recount(tmp_path):
    progress = _progress_dir(tmp_path)
    _append(progress, 0, "0\n1\n2\n")
    _append(progress, 1, "3\n4\n")
    counter = _DoneLogCounter(str(tmp_path), n_tasks=2)
    assert counter.count() == 5
    assert counter.count() == _full_recount(str(tmp_path), 2)


def test_counter_is_incremental_and_monotonic(tmp_path):
    progress = _progress_dir(tmp_path)
    counter = _DoneLogCounter(str(tmp_path), n_tasks=2)
    assert counter.count() == 0  # logs not created yet

    _append(progress, 0, "0\n1\n")
    assert counter.count() == 2
    # A second tick with no new bytes reads an empty tail and holds steady.
    assert counter.count() == 2

    _append(progress, 0, "2\n")
    _append(progress, 1, "3\n4\n5\n")
    assert counter.count() == 6
    assert counter.count() == _full_recount(str(tmp_path), 2)


def test_counter_defers_torn_final_line(tmp_path):
    progress = _progress_dir(tmp_path)
    counter = _DoneLogCounter(str(tmp_path), n_tasks=1)

    # A block id written but not yet newline-terminated must not be counted.
    _append(progress, 0, "0\n1\n42")
    assert counter.count() == 2
    # Once the line completes, the deferred block is picked up exactly once.
    _append(progress, 0, "\n")
    assert counter.count() == 3
    assert counter.count() == 3


def test_counter_handles_late_appearing_log(tmp_path):
    progress = _progress_dir(tmp_path)
    counter = _DoneLogCounter(str(tmp_path), n_tasks=3)

    _append(progress, 0, "0\n")
    assert counter.count() == 1  # task 1 and 2 have no log yet

    _append(progress, 2, "5\n6\n")
    assert counter.count() == 3
    assert counter.count() == _full_recount(str(tmp_path), 3)
