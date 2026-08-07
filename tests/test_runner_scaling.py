"""Compact runner work plans, batch execution, and completion journals."""
import os
import threading
from concurrent import futures
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bioimage_py.runner import Batch, RunnerConfig, RunnerError, get_runner
from bioimage_py.runner import base as runner_base
from bioimage_py.runner import distributed
from bioimage_py.runner import _work
from bioimage_py.runner._diagnostics import load_manifest
from bioimage_py.runner._harness import _truncate_torn_journal
from bioimage_py.runner._work import (BoundaryBatchPlan, ExplicitIdsSpec, RangeSpec,
                                      RegularBatchPlan, iter_assignment, load_assignment,
                                      load_work_spec, make_shard_routing, partition_slices,
                                      persist_routed_positions, persist_work_spec)
from bioimage_py.util import get_blocking, group_blocks_by_shard


def _make_batch_result():
    def batch_result(batch):
        return batch.batch_id, batch.start, batch.stop, sum(batch)
    return batch_result


def test_batch_contract_is_frozen_and_half_open():
    batch = Batch(2, 6, 10)
    assert batch.size == 4
    assert list(batch) == [6, 7, 8, 9]
    with pytest.raises(FrozenInstanceError):
        batch.stop = 11


def test_map_batches_validates_sizes_and_accepts_empty_work():
    runner = get_runner("local")
    with pytest.raises(ValueError, match="requires n_items"):
        runner.map_batches(_make_batch_result(), batch_size=3)
    with pytest.raises(ValueError, match="requires batch_size"):
        runner.map_batches(_make_batch_result(), n_items=3)
    with pytest.raises(ValueError, match="non-negative"):
        runner.map_batches(_make_batch_result(), n_items=-1, batch_size=3)
    with pytest.raises(ValueError, match="positive"):
        runner.map_batches(_make_batch_result(), n_items=3, batch_size=0)
    assert runner.map_batches(_make_batch_result(), 0, batch_size=3,
                              has_return_val=True) == []


@pytest.mark.parametrize("backend", ["local", "subprocess"])
def test_map_batches_returns_batch_order(backend):
    result = get_runner(backend).map_batches(
        _make_batch_result(), n_items=10, batch_size=3, num_workers=2, has_return_val=True,
    )
    assert result == [(0, 0, 3, 3), (1, 3, 6, 12), (2, 6, 9, 21), (3, 9, 10, 9)]


def test_map_batches_no_return_writes_no_result_records(tmp_path):
    output = tmp_path / "parts"
    output.mkdir()
    captured = {}

    def write_batch(batch):
        (output / f"{batch.batch_id}.txt").write_text(f"{batch.start}:{batch.stop}")

    def inspect_run(tmp):
        captured["result_files"] = os.listdir(os.path.join(tmp, "results"))

    runner = get_runner("subprocess", RunnerConfig(tmp_root=str(tmp_path)))
    assert runner.map_batches(write_batch, 11, batch_size=4, num_workers=2,
                              pre_cleanup=inspect_run) is None
    assert captured["result_files"] == []
    assert sorted(path.name for path in output.iterdir()) == ["0.txt", "1.txt", "2.txt"]


def test_batch_resume_reuses_exact_assignments(tmp_path):
    marker_folder = tmp_path / "markers"
    marker_folder.mkdir()
    fail_once = tmp_path / "fail-once"

    def process(batch):
        done = marker_folder / str(batch.batch_id)
        if done.exists():
            raise RuntimeError(f"batch {batch.batch_id} ran twice")
        if batch.batch_id == 1 and not fail_once.exists():
            fail_once.write_text("failed")
            raise RuntimeError("fail once")
        done.write_text("done")
        return batch.batch_id

    runner = get_runner(
        "subprocess", RunnerConfig(tmp_root=str(tmp_path), tasks_per_worker=2)
    )
    with pytest.raises(RunnerError) as exc_info:
        runner.map_batches(process, 17, batch_size=4, num_workers=2,
                           has_return_val=True)
    error = exc_info.value
    assert [batch.batch_id for batch in error.failed_batches] == [1]
    original_assignments = load_manifest(error.tmp_folder)["assignments"]
    resumed = {}

    def inspect_run(tmp):
        resumed["assignments"] = load_manifest(tmp)["assignments"]

    result = runner.map_batches(process, resume_from=error.tmp_folder,
                                has_return_val=True, pre_cleanup=inspect_run)
    assert result == [0, 1, 2, 3, 4]
    assert resumed["assignments"] == original_assignments


def test_range_planning_and_partitioning_stay_compact():
    work = RangeSpec(0, 1_000_000_000)
    assignments = partition_slices(len(work), 64)
    assert len(work) == 1_000_000_000
    assert len(assignments) == 64
    assert assignments[0] == {"kind": "slice", "start": 0, "stop": 15_625_000}
    assert assignments[-1]["stop"] == len(work)


def test_explicit_ids_use_json_then_memory_mapped_npy(tmp_path, monkeypatch):
    monkeypatch.setattr(_work, "EXPLICIT_IDS_JSON_LIMIT", 3)
    small = persist_work_spec(str(tmp_path), ExplicitIdsSpec([9, 2, 9]))
    assert small["storage"] == "json"
    assert list(load_work_spec(str(tmp_path), small)) == [9, 2, 9]

    large_root = tmp_path / "large"
    large_root.mkdir()
    large = persist_work_spec(str(large_root), ExplicitIdsSpec([8, 3, 8, 1]))
    loaded = load_work_spec(str(large_root), large)
    assert large == {
        "kind": "explicit_ids", "storage": "npy", "length": 4,
        "dtype": "int64", "path": "work/ids.npy",
    }
    assert isinstance(loaded.values, np.memmap)
    assert list(loaded) == [8, 3, 8, 1]


def test_large_sparse_run_manifest_records_binary_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(_work, "EXPLICIT_IDS_JSON_LIMIT", 3)

    def fail(item_id):
        if item_id == 8:
            raise RuntimeError("stop")
        return item_id

    runner = get_runner("subprocess", RunnerConfig(tmp_root=str(tmp_path)))
    with pytest.raises(RunnerError) as exc_info:
        runner.map(fail, item_ids=[8, 3, 8, 1], num_workers=1)
    manifest = load_manifest(exc_info.value.tmp_folder)
    assert manifest["work_plan"]["storage"] == "npy"
    assert manifest["work_plan"]["dtype"] == "int64"
    assert manifest["work_plan"]["length"] == 4


def test_local_pending_future_count_is_bounded(monkeypatch):
    gate = threading.Event()
    reached_window = threading.Event()
    state = {"pending": 0, "peak": 0}
    lock = threading.Lock()
    real_executor = futures.ThreadPoolExecutor

    class TrackingExecutor(real_executor):
        def submit(self, function, /, *args, **kwargs):
            with lock:
                state["pending"] += 1
                state["peak"] = max(state["peak"], state["pending"])
                if state["pending"] == 4:
                    reached_window.set()

            def wrapped():
                try:
                    return function(*args, **kwargs)
                finally:
                    with lock:
                        state["pending"] -= 1

            return super().submit(wrapped)

    monkeypatch.setattr(runner_base.futures, "ThreadPoolExecutor", TrackingExecutor)

    result = {}

    def wait_for_release(item_id):
        gate.wait()
        return item_id

    def run():
        result["value"] = get_runner("local").map(
            wait_for_release, 100, num_workers=2,
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert reached_window.wait(5)
    gate.set()
    thread.join(10)
    assert not thread.is_alive()
    assert state["peak"] == 4
    assert result["value"] == list(range(100))


def test_completion_journal_ignores_and_truncates_torn_tail(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    path = progress / "0.bin"
    path.write_bytes(
        distributed._COMPLETION_RECORD.pack(1, 10)
        + distributed._COMPLETION_RECORD.pack(2, 17)
        + b"torn"
    )
    assert distributed._completion_counts(str(tmp_path), 0) == (2, 17)
    counter = distributed._CompletionJournalCounter(str(tmp_path), 1)
    assert counter.count() == 17
    _truncate_torn_journal(str(path))
    assert path.stat().st_size == 2 * distributed._COMPLETION_RECORD.size


def test_runner_error_keeps_large_failed_range_lazy():
    error = RunnerError("failed", failed_id_specs=[RangeSpec(5, 1_000_000_005)])
    assert error.failed_block_count == 1_000_000_000
    assert error._failed_block_ids is None
    iterator = error.iter_failed_block_ids()
    assert [next(iterator), next(iterator), next(iterator)] == [5, 6, 7]
    assert error._failed_block_ids is None


def test_arithmetic_shard_router_matches_reference_union_find():
    rng = np.random.default_rng(42)

    class Output:
        def __init__(self, shards):
            self.shards = shards

    for _ in range(100):
        ndim = int(rng.integers(1, 4))
        begin = rng.integers(0, 8, size=ndim).tolist()
        end = (np.asarray(begin) + rng.integers(1, 25, size=ndim)).tolist()
        block_shape = rng.integers(1, 9, size=ndim).tolist()
        blocking = get_blocking(end, block_shape, tuple(
            slice(int(start), int(stop)) for start, stop in zip(begin, end)
        ))
        shard_shapes = [tuple(rng.integers(1, 13, size=ndim).tolist())
                        for _ in range(int(rng.integers(1, 4)))]
        outputs = [Output(shape) for shape in shard_shapes]
        reference = group_blocks_by_shard(
            blocking, outputs, range(int(blocking.number_of_blocks)),
        )
        routing = make_shard_routing(blocking, shard_shapes)
        actual = [sorted(routing.iter_component_positions(component))
                  for component in range(routing.n_components)]
        assert sorted(actual) == sorted(reference)
        for stop in range(routing.n_components + 1):
            assert routing.prefix_component_size(stop) == sum(
                len(actual[component]) for component in range(stop)
            )
        for component, positions in enumerate(actual):
            assert all(routing.component_for_position(position) == component
                       for position in positions)
        assignment = routing.assignments(1)[0]
        expected_positions = [position for positions in actual for position in positions]
        work = RangeSpec(0, int(blocking.number_of_blocks))
        for skip in (0, len(work) // 2, len(work)):
            resumed = [position for position, _ in
                       iter_assignment(work, assignment, skip)]
            assert resumed == expected_positions[skip:]


def test_sparse_sharded_positions_stream_to_one_task_per_component(tmp_path):
    blocking = get_blocking((64, 64), (16, 16))
    routing = make_shard_routing(blocking, [(32, 32)])
    work = ExplicitIdsSpec([15, 0, 5, 10, 1, 4, 11, 14])
    descriptors = persist_routed_positions(str(tmp_path), work, routing, n_tasks=3)
    owners = {}
    recovered_positions = []
    for task_id, descriptor in enumerate(descriptors):
        assignment = load_assignment(str(tmp_path), descriptor)
        for position, block_id in iter_assignment(work, assignment):
            recovered_positions.append(position)
            component = routing.component_for_position(int(block_id))
            assert owners.setdefault(component, task_id) == task_id
    assert sorted(recovered_positions) == list(range(len(work)))


def test_regular_batch_plan_does_not_depend_on_worker_count():
    plan = RegularBatchPlan(23, 5)
    assert [(batch.batch_id, batch.start, batch.stop) for batch in plan] == [
        (0, 0, 5), (1, 5, 10), (2, 10, 15), (3, 15, 20), (4, 20, 23),
    ]
    for task_count in (1, 2, 7):
        covered = [position for assignment in partition_slices(len(plan), task_count)
                   for position in range(assignment["start"], assignment["stop"])]
        assert covered == list(range(len(plan)))


def test_irregular_batch_boundaries_persist_per_batch(tmp_path):
    plan = BoundaryBatchPlan((0, 3, 11, 12))
    descriptor = persist_work_spec(str(tmp_path), plan)
    assert descriptor == {
        "kind": "boundary_batches", "boundaries": [0, 3, 11, 12], "length": 3,
    }
    assert list(load_work_spec(str(tmp_path), descriptor)) == [
        Batch(0, 0, 3), Batch(1, 3, 11), Batch(2, 11, 12),
    ]
