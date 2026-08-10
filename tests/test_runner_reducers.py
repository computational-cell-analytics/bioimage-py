"""Associative runner reductions and durable accumulator recovery."""
import os

import numpy as np
import pytest

from bioimage_py.runner import Reducer, RunnerConfig, RunnerError, get_runner
from bioimage_py.runner._diagnostics import load_manifest
from bioimage_py.runner._reduction import accumulator_path, write_accumulator
from bioimage_py.runner._work import (Batch, ExplicitIdsSpec, RangeSpec,
                                      ReductionBatchPlan, load_work_spec,
                                      persist_work_spec)
from bioimage_py.util import to_roi


class _SumReducer:
    def initial(self):
        return 0

    def update(self, accumulator, value):
        return accumulator + value

    def merge(self, left, right):
        return left + right

    def finalize(self, accumulator):
        return accumulator


class _ConcatReducer:
    def initial(self):
        return ""

    def update(self, accumulator, value):
        return accumulator + str(value)

    def merge(self, left, right):
        return left + right

    def finalize(self, accumulator):
        return accumulator


def _make_sum_reducer():
    class SumReducer:
        def initial(self):
            return 0

        def update(self, accumulator, value):
            return accumulator + value

        def merge(self, left, right):
            return left + right

        def finalize(self, accumulator):
            return accumulator
    return SumReducer()


def _make_concat_reducer():
    class ConcatReducer:
        def initial(self):
            return ""

        def update(self, accumulator, value):
            return accumulator + str(value)

        def merge(self, left, right):
            return left + right

        def finalize(self, accumulator):
            return accumulator
    return ConcatReducer()


def _identity(value):
    return value


def _make_identity():
    def identity(value):
        return value
    return identity


def _make_sum_block():
    def sum_block(block, inputs, outputs, mask):
        return int(np.asarray(inputs[0][to_roi(block)]).sum())
    return sum_block


def _make_resumable(forbid, fail_once):
    def process(value):
        if value >= 6 and forbid.exists():
            raise RuntimeError("completed batch ran again")
        if value == 4 and not fail_once.exists():
            fail_once.write_text("failed")
            raise RuntimeError("fail once")
        return value
    return process


def test_reducer_is_public_protocol():
    reducer: Reducer[int, int, int] = _SumReducer()
    assert reducer.finalize(reducer.update(reducer.initial(), 3)) == 3


@pytest.mark.parametrize("backend", ["local", "subprocess"])
def test_map_reducer_preserves_logical_order(backend, tmp_path):
    config = RunnerConfig(tmp_root=str(tmp_path)) if backend == "subprocess" else None
    runner = get_runner(backend, config)
    result = runner.map(
        _make_identity(),
        item_ids=[8, 3, 8, 1],
        num_workers=2,
        reducer=_make_concat_reducer(),
        reduction_batch_size=2,
    )
    assert result == "8381"


@pytest.mark.parametrize("backend", ["local", "subprocess"])
def test_block_reducer_matches_whole_array(backend, zarr_factory, tmp_path):
    data = np.arange(35, dtype="int64").reshape(5, 7)
    source = zarr_factory(data, chunks=(2, 3))
    config = RunnerConfig(tmp_root=str(tmp_path)) if backend == "subprocess" else None
    result = get_runner(backend, config).run(
        _make_sum_block(),
        [source],
        block_shape=(2, 3),
        num_workers=3,
        has_return_val=True,
        reducer=_make_sum_reducer(),
        reduction_batch_size=2,
    )
    assert result == int(data.sum())


@pytest.mark.parametrize("backend", ["local", "subprocess"])
def test_reducer_empty_work(backend, tmp_path):
    config = RunnerConfig(tmp_root=str(tmp_path)) if backend == "subprocess" else None
    runner = get_runner(backend, config)
    assert runner.map(_make_identity(), 0, reducer=_make_sum_reducer()) == 0


def test_reducer_validation():
    runner = get_runner("local")
    with pytest.raises(ValueError, match="has_return_val"):
        runner.map(_identity, 3, has_return_val=False, reducer=_SumReducer())
    with pytest.raises(ValueError, match="requires reducer"):
        runner.map(_identity, 3, reduction_batch_size=2)
    with pytest.raises(TypeError, match="must be an integer"):
        runner.map(_identity, 3, reducer=_SumReducer(), reduction_batch_size=1.5)
    with pytest.raises(ValueError, match="must be positive"):
        runner.map(_identity, 3, reducer=_SumReducer(), reduction_batch_size=0)


def test_block_reducer_rejects_outputs(zarr_factory):
    source = zarr_factory(np.ones((4, 4), dtype="uint8"), chunks=(2, 2))
    output = zarr_factory(shape=(4, 4), chunks=(2, 2), dtype="uint8", fill=0)
    with pytest.raises(ValueError, match="does not support output arrays"):
        get_runner("local").run(
            _make_sum_block(), [source], [output], block_shape=(2, 2),
            has_return_val=True, reducer=_SumReducer(),
        )


def test_reduction_plan_stays_compact_and_roundtrips(tmp_path):
    plan = ReductionBatchPlan(RangeSpec(0, 1_000_000_000), 100_000)
    assert plan.n_items == 1_000_000_000
    assert len(plan) == 10_000
    assert plan[9_999] == Batch(9_999, 999_900_000, 1_000_000_000)

    explicit = ReductionBatchPlan(ExplicitIdsSpec([8, 3, 8, 1]), 2)
    descriptor = persist_work_spec(str(tmp_path), explicit)
    restored = load_work_spec(str(tmp_path), descriptor)
    assert isinstance(restored, ReductionBatchPlan)
    assert list(restored.items) == [8, 3, 8, 1]
    assert list(restored) == [Batch(0, 0, 2), Batch(1, 2, 4)]


def test_subprocess_reducer_resume_recovers_unjournaled_accumulator(tmp_path):
    forbid = tmp_path / "forbid-late-items"
    fail_once = tmp_path / "fail-once"
    runner = get_runner("subprocess", RunnerConfig(tmp_root=str(tmp_path)))
    with pytest.raises(RunnerError) as exc_info:
        runner.map(
            _make_resumable(forbid, fail_once),
            8,
            num_workers=1,
            reducer=_make_concat_reducer(),
            reduction_batch_size=2,
        )
    error = exc_info.value
    assert [batch.batch_id for batch in error.failed_batches] == [2, 3]
    assert error.failed_block_ids == [4, 5, 6, 7]
    assert sorted(os.listdir(os.path.join(error.tmp_folder, "results"))) == []

    manifest = load_manifest(error.tmp_folder)
    work = load_work_spec(error.tmp_folder, manifest["work_plan"])
    assert isinstance(work, ReductionBatchPlan)
    write_accumulator(error.tmp_folder, work[3], "67")
    with open(accumulator_path(error.tmp_folder, 0), "r+b") as file:
        file.truncate(4)
    forbid.write_text("do not call")
    captured = {}

    def inspect(tmp):
        captured["accumulators"] = sorted(os.listdir(os.path.join(tmp, "accumulators")))
        captured["results"] = sorted(os.listdir(os.path.join(tmp, "results")))

    result = runner.map(
        _identity,
        resume_from=error.tmp_folder,
        pre_cleanup=inspect,
    )
    assert result == "01234567"
    assert captured["accumulators"] == [
        f"part-{batch_id:012d}.pkl" for batch_id in range(4)
    ]
    assert captured["results"] == []


def test_corrupt_accumulator_is_not_valid(tmp_path):
    batch = Batch(0, 0, 2)
    root = str(tmp_path)
    write_accumulator(root, batch, 1)
    with open(accumulator_path(root, batch.batch_id), "r+b") as file:
        file.truncate(4)
    from bioimage_py.runner._reduction import read_accumulator

    assert read_accumulator(root, batch) == (False, None)
