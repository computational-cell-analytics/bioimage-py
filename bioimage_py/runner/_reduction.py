"""Batch execution and atomic storage for associative reductions."""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Tuple

import cloudpickle

from ._work import Batch, ReductionBatchPlan
from .reducer import Reducer


ACCUMULATOR_SCHEMA_VERSION = 1


def accumulator_path(tmp: str, batch_id: int) -> str:
    """Return the final path for one batch accumulator."""
    return os.path.join(tmp, "accumulators", f"part-{int(batch_id):012d}.pkl")


def write_accumulator(tmp: str, batch: Batch, accumulator: Any) -> None:
    """Write one versioned accumulator through an atomic rename."""
    folder = os.path.join(tmp, "accumulators")
    os.makedirs(folder, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".part-{int(batch.batch_id):012d}.", suffix=".tmp", dir=folder,
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            cloudpickle.dump({
                "schema_version": ACCUMULATOR_SCHEMA_VERSION,
                "batch_id": int(batch.batch_id),
                "start": int(batch.start),
                "stop": int(batch.stop),
                "step": int(batch.step),
                "accumulator": accumulator,
            }, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, accumulator_path(tmp, batch.batch_id))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_accumulator(tmp: str, batch: Batch) -> Tuple[bool, Any]:
    """Read and validate one batch accumulator."""
    try:
        with open(accumulator_path(tmp, batch.batch_id), "rb") as file:
            record = cloudpickle.load(file)
            trailing = file.read(1)
    except Exception:  # noqa: BLE001 - any unreadable record is invalid
        return False, None
    if trailing or not isinstance(record, dict):
        return False, None
    expected = {
        "schema_version": ACCUMULATOR_SCHEMA_VERSION,
        "batch_id": int(batch.batch_id),
        "start": int(batch.start),
        "stop": int(batch.stop),
        "step": int(batch.step),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return False, None
    if "accumulator" not in record:
        return False, None
    return True, record["accumulator"]


def reduce_batch(
    batch: Batch,
    plan: ReductionBatchPlan,
    call_one: Callable[[int], Any],
    reducer: Reducer[Any, Any, Any],
    *,
    tmp: str | None = None,
) -> Any:
    """Reduce one logical batch and optionally persist its accumulator."""
    if tmp is not None:
        valid, accumulator = read_accumulator(tmp, batch)
        if valid:
            return accumulator
    accumulator = reducer.initial()
    for item_id in plan.batch_items(batch):
        accumulator = reducer.update(accumulator, call_one(int(item_id)))
    if tmp is not None:
        write_accumulator(tmp, batch, accumulator)
    return accumulator
