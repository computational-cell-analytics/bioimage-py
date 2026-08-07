"""Unit tests for distributed completion journal helpers."""
import os

from bioimage_py.runner.distributed import (_COMPLETION_RECORD, _CompletionJournalCounter,
                                            _completion_counts)


def _progress_dir(tmp_path):
    progress = os.path.join(str(tmp_path), "progress")
    os.makedirs(progress)
    return progress


def _append(progress, task_id, units, logical_items):
    with open(os.path.join(progress, f"{task_id}.bin"), "ab") as file:
        file.write(_COMPLETION_RECORD.pack(units, logical_items))


def test_counter_sums_latest_cumulative_records(tmp_path):
    progress = _progress_dir(tmp_path)
    _append(progress, 0, 1, 10)
    _append(progress, 0, 2, 17)
    _append(progress, 1, 1, 5)
    counter = _CompletionJournalCounter(str(tmp_path), n_tasks=2)
    assert counter.count() == 22


def test_counter_handles_missing_and_late_journals(tmp_path):
    progress = _progress_dir(tmp_path)
    counter = _CompletionJournalCounter(str(tmp_path), n_tasks=3)
    assert counter.count() == 0
    _append(progress, 2, 1, 8)
    assert counter.count() == 8
    _append(progress, 0, 1, 3)
    assert counter.count() == 11


def test_reader_ignores_torn_final_record(tmp_path):
    progress = _progress_dir(tmp_path)
    _append(progress, 0, 1, 10)
    with open(os.path.join(progress, "0.bin"), "ab") as file:
        file.write(_COMPLETION_RECORD.pack(2, 20)[:7])
    assert _completion_counts(str(tmp_path), 0) == (1, 10)


def test_reader_returns_zero_for_short_first_record(tmp_path):
    progress = _progress_dir(tmp_path)
    with open(os.path.join(progress, "0.bin"), "wb") as file:
        file.write(b"short")
    assert _completion_counts(str(tmp_path), 0) == (0, 0)
