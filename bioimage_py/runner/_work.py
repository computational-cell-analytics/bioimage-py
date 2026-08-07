"""Compact work specifications for runner planning and persistence."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import ceil, gcd
from typing import (Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple,
                    TypeGuard, Union)

import numpy as np


EXPLICIT_IDS_JSON_LIMIT = 10_000


@dataclass(frozen=True)
class Batch:
    """A deterministic half-open batch of logical item indices."""

    batch_id: int
    start: int
    stop: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.batch_id < 0:
            raise ValueError("batch_id must be non-negative.")
        if self.step <= 0:
            raise ValueError("Batch.step must be positive.")
        if self.start < 0 or self.stop < self.start:
            raise ValueError("A batch requires 0 <= start <= stop.")

    @property
    def size(self) -> int:
        """Return the number of logical items in the batch."""
        return len(range(self.start, self.stop, self.step))

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.stop, self.step))


@dataclass(frozen=True)
class RangeSpec(Sequence[int]):
    """A tagged integer range used by runner internals."""

    start: int
    stop: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("RangeSpec.step must be positive.")

    def __len__(self) -> int:
        return len(range(self.start, self.stop, self.step))

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.stop, self.step))

    def __getitem__(self, index: Union[int, slice]) -> Union[int, "RangeSpec"]:
        values = range(self.start, self.stop, self.step)
        if isinstance(index, slice):
            selected = values[index]
            return RangeSpec(selected.start, selected.stop, selected.step)
        return values[index]


@dataclass(frozen=True)
class ExplicitIdsSpec(Sequence[int]):
    """An explicit integer sequence that preserves order and duplicates."""

    values: Sequence[int]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return (int(value) for value in self.values)

    def __getitem__(self, index: Union[int, slice]) -> Union[int, "ExplicitIdsSpec"]:
        if isinstance(index, slice):
            return ExplicitIdsSpec(self.values[index])
        return int(self.values[index])


@dataclass(frozen=True)
class RegularBatchPlan(Sequence[Batch]):
    """A regular batch plan represented by two integers."""

    n_items: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.n_items < 0:
            raise ValueError("n_items must be non-negative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def __len__(self) -> int:
        return int(ceil(self.n_items / self.batch_size)) if self.n_items else 0

    def __iter__(self) -> Iterator[Batch]:
        for batch_id in range(len(self)):
            yield self[batch_id]

    def __getitem__(self, index: Union[int, slice]) -> Union[Batch, Tuple[Batch, ...]]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.batch_size
        return Batch(index, start, min(start + self.batch_size, self.n_items))


@dataclass(frozen=True)
class BoundaryBatchPlan(Sequence[Batch]):
    """A compact list of irregular contiguous batch boundaries."""

    boundaries: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.boundaries or self.boundaries[0] != 0:
            raise ValueError("Batch boundaries must start at 0.")
        if any(stop <= start for start, stop in zip(self.boundaries, self.boundaries[1:])):
            raise ValueError("Batch boundaries must increase.")

    @property
    def n_items(self) -> int:
        return self.boundaries[-1]

    def __len__(self) -> int:
        return len(self.boundaries) - 1

    def __iter__(self) -> Iterator[Batch]:
        for batch_id in range(len(self)):
            yield self[batch_id]

    def __getitem__(self, index: Union[int, slice]) -> Union[Batch, Tuple[Batch, ...]]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return Batch(index, self.boundaries[index], self.boundaries[index + 1])


BatchPlan = Union[RegularBatchPlan, BoundaryBatchPlan]
WorkSpec = Union[RangeSpec, ExplicitIdsSpec, BatchPlan]


def is_batch_plan(value: object) -> TypeGuard[BatchPlan]:
    """Return whether ``value`` is a regular or boundary-based batch plan."""
    return isinstance(value, (RegularBatchPlan, BoundaryBatchPlan))


def logical_size(value: Union[int, Batch]) -> int:
    """Return the logical item count for one work unit."""
    return value.size if isinstance(value, Batch) else 1


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file)
        file.write("\n")


def persist_work_spec(tmp: str, work: WorkSpec) -> Dict[str, Any]:
    """Persist a work specification and return its manifest descriptor."""
    if isinstance(work, RangeSpec):
        return {
            "kind": "range",
            "start": int(work.start),
            "stop": int(work.stop),
            "step": int(work.step),
            "length": len(work),
        }
    if isinstance(work, RegularBatchPlan):
        return {
            "kind": "regular_batches",
            "n_items": int(work.n_items),
            "batch_size": int(work.batch_size),
            "length": len(work),
        }
    if isinstance(work, BoundaryBatchPlan):
        return {
            "kind": "boundary_batches",
            "boundaries": [int(value) for value in work.boundaries],
            "length": len(work),
        }

    folder = os.path.join(tmp, "work")
    os.makedirs(folder, exist_ok=True)
    length = len(work)
    if length <= EXPLICIT_IDS_JSON_LIMIT:
        relative = os.path.join("work", "ids.json")
        _write_json(os.path.join(tmp, relative), [int(value) for value in work])
        storage = "json"
    else:
        relative = os.path.join("work", "ids.npy")
        values = np.fromiter((int(value) for value in work), dtype=np.int64, count=length)
        np.save(os.path.join(tmp, relative), values, allow_pickle=False)
        storage = "npy"
    return {
        "kind": "explicit_ids",
        "storage": storage,
        "length": length,
        "dtype": "int64",
        "path": relative,
    }


def load_work_spec(tmp: str, descriptor: Mapping[str, Any]) -> WorkSpec:
    """Load a compact work specification from a manifest descriptor."""
    kind = descriptor.get("kind")
    if kind == "range":
        return RangeSpec(int(descriptor["start"]), int(descriptor["stop"]),
                         int(descriptor["step"]))
    if kind == "regular_batches":
        return RegularBatchPlan(int(descriptor["n_items"]), int(descriptor["batch_size"]))
    if kind == "boundary_batches":
        return BoundaryBatchPlan(tuple(int(value) for value in descriptor["boundaries"]))
    if kind != "explicit_ids":
        raise ValueError(f"Unsupported work specification kind {kind!r}.")
    if descriptor.get("dtype") != "int64":
        raise ValueError(f"Unsupported explicit-ID dtype {descriptor.get('dtype')!r}.")
    path = os.path.join(tmp, str(descriptor["path"]))
    storage = descriptor.get("storage")
    if storage == "json":
        with open(path) as file:
            values: Sequence[int] = json.load(file)
    elif storage == "npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError(f"Unsupported explicit-ID storage {storage!r}.")
    if len(values) != int(descriptor["length"]):
        raise ValueError(f"Explicit-ID file {path!r} has an unexpected length.")
    return ExplicitIdsSpec(values)


def partition_slices(length: int, n_tasks: int) -> Tuple[Dict[str, int], ...]:
    """Partition positions into contiguous near-equal task slices."""
    base, extra = divmod(int(length), int(n_tasks))
    assignments = []
    start = 0
    for task_id in range(int(n_tasks)):
        size = base + int(task_id < extra)
        assignments.append({"kind": "slice", "start": start, "stop": start + size})
        start += size
    return tuple(assignments)


def assignment_length(assignment: Mapping[str, Any]) -> int:
    """Return the number of work units in one task assignment."""
    kind = assignment.get("kind")
    if kind == "slice":
        return max(0, int(assignment["stop"]) - int(assignment["start"]))
    if kind == "positions":
        return len(assignment["positions"])
    if kind == "positions_file":
        return int(assignment["length"])
    if kind == "shard_components":
        if "length" in assignment:
            return int(assignment["length"])
        plan = ShardRoutingPlan.from_descriptor(assignment["routing"])
        start, stop = int(assignment["start"]), int(assignment["stop"])
        return plan.component_range_size(start, stop)
    raise ValueError(f"Unsupported task assignment kind {kind!r}.")


def iter_assignment(work: WorkSpec, assignment: Mapping[str, Any],
                    skip: int = 0) -> Iterator[Tuple[int, Union[int, Batch]]]:
    """Yield ``(global_position, value)`` pairs for a task assignment."""
    kind = assignment.get("kind")
    if kind == "slice":
        start = int(assignment["start"]) + int(skip)
        stop = int(assignment["stop"])
        for position in range(start, stop):
            yield position, work[position]
        return
    if kind == "positions":
        for position in assignment["positions"][int(skip):]:
            position = int(position)
            yield position, work[position]
        return
    if kind == "shard_components":
        plan = ShardRoutingPlan.from_descriptor(assignment["routing"])
        start = int(assignment["start"])
        stop = int(assignment["stop"])
        remaining = int(skip)
        if remaining >= plan.component_range_size(start, stop):
            return
        low, high = start, stop
        while low < high:
            middle = (low + high) // 2
            if plan.component_range_size(start, middle + 1) <= remaining:
                low = middle + 1
            else:
                high = middle
        first_component = low
        component_skip = remaining - plan.component_range_size(start, first_component)
        for component_id in range(first_component, stop):
            for position in plan.iter_component_positions(
                    component_id, component_skip if component_id == first_component else 0):
                yield position, work[position]
        return
    raise ValueError(f"Unsupported task assignment kind {kind!r}.")


def load_assignment(tmp: str, assignment: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve an assignment's external position storage when necessary."""
    if assignment.get("kind") != "positions_file":
        return assignment
    if assignment.get("dtype") != "int64":
        raise ValueError(f"Unsupported assignment dtype {assignment.get('dtype')!r}.")
    path = os.path.join(tmp, str(assignment["path"]))
    storage = assignment.get("storage")
    if storage == "json":
        with open(path) as file:
            positions: Sequence[int] = json.load(file)
    elif storage == "npy":
        positions = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError(f"Unsupported assignment storage {storage!r}.")
    offset = int(assignment.get("offset", 0))
    length = int(assignment["length"])
    if offset < 0 or offset + length > len(positions):
        raise ValueError(f"Assignment slice for {path!r} is out of bounds.")
    positions = positions[offset:offset + length]
    return {"kind": "positions", "positions": positions}


def persist_routed_positions(tmp: str, work: ExplicitIdsSpec, routing: "ShardRoutingPlan",
                             n_tasks: int) -> Tuple[Dict[str, Any], ...]:
    """Stream explicit block positions into shard-safe task slices."""
    counts = [0] * int(n_tasks)
    for block_id in work:
        counts[routing.component_for_position(int(block_id)) % n_tasks] += 1
    offsets = [0] * int(n_tasks)
    for task_id in range(1, int(n_tasks)):
        offsets[task_id] = offsets[task_id - 1] + counts[task_id - 1]

    relative = os.path.join("work", "assignments", "positions.npy")
    path = os.path.join(tmp, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not len(work):
        np.save(path, np.empty(0, dtype=np.int64), allow_pickle=False)
        return ({"kind": "positions_file", "storage": "npy", "length": 0,
                 "offset": 0, "dtype": "int64", "path": relative},)
    positions = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64,
                                          shape=(len(work),))
    cursors = list(offsets)
    for position, block_id in enumerate(work):
        task_id = routing.component_for_position(int(block_id)) % n_tasks
        positions[cursors[task_id]] = position
        cursors[task_id] += 1
    positions.flush()
    del positions
    return tuple({
        "kind": "positions_file", "storage": "npy", "length": counts[task_id],
        "offset": offsets[task_id], "dtype": "int64", "path": relative,
    } for task_id in range(int(n_tasks)))


class AssignmentValues(Iterable[Union[int, Batch]]):
    """A lazy value view over a task assignment or an assignment suffix."""

    def __init__(self, work: WorkSpec, assignment: Mapping[str, Any], skip: int = 0,
                 tmp: Optional[str] = None):
        self._work = work
        self._assignment = assignment
        self._skip = int(skip)
        self._tmp = tmp

    def __len__(self) -> int:
        return assignment_length(self._assignment) - self._skip

    def __iter__(self) -> Iterator[Union[int, Batch]]:
        assignment = (load_assignment(self._tmp, self._assignment)
                      if self._tmp is not None else self._assignment)
        for _, value in iter_assignment(self._work, assignment, self._skip):
            yield value


@dataclass(frozen=True)
class _AxisRouting:
    n_blocks: int
    first_cut: Optional[int]
    period: int
    n_components: int

    @classmethod
    def build(cls, n_blocks: int, roi_begin: int, block_size: int,
              shard_period: int) -> "_AxisRouting":
        common = gcd(block_size, shard_period)
        if roi_begin % common:
            return cls(n_blocks, None, 0, 1)
        period = shard_period // common
        if period == 1:
            residue = 0
        else:
            residue = ((-roi_begin // common) *
                       pow(block_size // common, -1, period)) % period
        first_cut = period if residue == 0 else residue
        if first_cut >= n_blocks:
            return cls(n_blocks, None, period, 1)
        n_cuts = 1 + (n_blocks - 1 - first_cut) // period
        return cls(n_blocks, first_cut, period, n_cuts + 1)

    def bounds(self, component: int) -> Tuple[int, int]:
        if component < 0 or component >= self.n_components:
            raise IndexError(component)
        if self.first_cut is None:
            return 0, self.n_blocks
        start = 0 if component == 0 else self.first_cut + (component - 1) * self.period
        stop = (self.n_blocks if component + 1 == self.n_components
                else self.first_cut + component * self.period)
        return start, stop

    def component_for_block(self, coordinate: int) -> int:
        if coordinate < 0 or coordinate >= self.n_blocks:
            raise IndexError(coordinate)
        if self.first_cut is None or coordinate < self.first_cut:
            return 0
        return 1 + (coordinate - self.first_cut) // self.period

    def prefix_blocks(self, n_components: int) -> int:
        """Return the block count in the first axis components."""
        if n_components < 0 or n_components > self.n_components:
            raise IndexError(n_components)
        if n_components == self.n_components:
            return self.n_blocks
        return self.bounds(n_components)[0]

    def descriptor(self) -> Dict[str, Any]:
        return {
            "n_blocks": self.n_blocks,
            "first_cut": self.first_cut,
            "period": self.period,
            "n_components": self.n_components,
        }

    @classmethod
    def from_descriptor(cls, value: Mapping[str, Any]) -> "_AxisRouting":
        first = value.get("first_cut")
        return cls(int(value["n_blocks"]), None if first is None else int(first),
                   int(value["period"]), int(value["n_components"]))


@dataclass(frozen=True)
class ShardRoutingPlan:
    """An arithmetic partition of a regular block grid into shard-safe components."""

    axes: Tuple[_AxisRouting, ...]

    @property
    def n_components(self) -> int:
        result = 1
        for axis in self.axes:
            result *= axis.n_components
        return result

    @property
    def n_blocks(self) -> int:
        result = 1
        for axis in self.axes:
            result *= axis.n_blocks
        return result

    def _component_coordinates(self, component_id: int) -> Tuple[int, ...]:
        if component_id < 0 or component_id >= self.n_components:
            raise IndexError(component_id)
        coordinates = [0] * len(self.axes)
        value = int(component_id)
        for axis_index in range(len(self.axes) - 1, -1, -1):
            count = self.axes[axis_index].n_components
            coordinates[axis_index] = value % count
            value //= count
        return tuple(coordinates)

    def component_size(self, component_id: int) -> int:
        result = 1
        for axis, coordinate in zip(self.axes, self._component_coordinates(component_id)):
            start, stop = axis.bounds(coordinate)
            result *= stop - start
        return result

    def prefix_component_size(self, n_components: int) -> int:
        """Return the block count in a C-order component prefix."""
        if n_components < 0 or n_components > self.n_components:
            raise IndexError(n_components)

        def prefix(axis_index: int, count: int) -> int:
            axis = self.axes[axis_index]
            if axis_index + 1 == len(self.axes):
                return axis.prefix_blocks(count)
            tail_components = 1
            tail_blocks = 1
            for tail_axis in self.axes[axis_index + 1:]:
                tail_components *= tail_axis.n_components
                tail_blocks *= tail_axis.n_blocks
            full, remainder = divmod(count, tail_components)
            total = axis.prefix_blocks(full) * tail_blocks
            if remainder:
                start, stop = axis.bounds(full)
                total += (stop - start) * prefix(axis_index + 1, remainder)
            return total

        return prefix(0, int(n_components))

    def component_range_size(self, start: int, stop: int) -> int:
        """Return the block count in a half-open component range."""
        if start < 0 or stop < start or stop > self.n_components:
            raise IndexError((start, stop))
        return self.prefix_component_size(stop) - self.prefix_component_size(start)

    def component_for_position(self, position: int) -> int:
        """Return the component ID for one C-order block position."""
        if position < 0 or position >= self.n_blocks:
            raise IndexError(position)
        block_coordinates = [0] * len(self.axes)
        value = int(position)
        for axis_index in range(len(self.axes) - 1, -1, -1):
            count = self.axes[axis_index].n_blocks
            block_coordinates[axis_index] = value % count
            value //= count
        component_id = 0
        for axis, coordinate in zip(self.axes, block_coordinates):
            component_id *= axis.n_components
            component_id += axis.component_for_block(coordinate)
        return component_id

    def iter_component_positions(self, component_id: int, skip: int = 0) -> Iterator[int]:
        coordinates = self._component_coordinates(component_id)
        bounds = [axis.bounds(coordinate)
                  for axis, coordinate in zip(self.axes, coordinates)]
        lengths = [stop - start for start, stop in bounds]
        size = 1
        for length in lengths:
            size *= length
        if skip < 0 or skip > size:
            raise IndexError(skip)
        strides = []
        stride = 1
        for axis in reversed(self.axes):
            strides.append(stride)
            stride *= axis.n_blocks
        strides.reverse()

        if skip:
            for local_position in range(int(skip), size):
                remainder = local_position
                position = 0
                for axis_index in range(len(self.axes) - 1, -1, -1):
                    coordinate = remainder % lengths[axis_index]
                    remainder //= lengths[axis_index]
                    position += (bounds[axis_index][0] + coordinate) * strides[axis_index]
                yield position
            return

        ranges = [range(start, stop) for start, stop in bounds]

        def visit(axis_index: int, position: int) -> Iterator[int]:
            if axis_index == len(ranges):
                yield position
                return
            for coordinate in ranges[axis_index]:
                yield from visit(axis_index + 1,
                                 position + coordinate * strides[axis_index])

        yield from visit(0, 0)

    def assignments(self, n_tasks: int) -> Tuple[Dict[str, Any], ...]:
        routing = self.descriptor()
        slices = partition_slices(self.n_components, n_tasks)
        return tuple({"kind": "shard_components", "start": value["start"],
                      "stop": value["stop"],
                      "length": self.component_range_size(value["start"], value["stop"]),
                      "routing": routing} for value in slices)

    def descriptor(self) -> Dict[str, Any]:
        return {"axes": [axis.descriptor() for axis in self.axes]}

    @classmethod
    def from_descriptor(cls, value: Mapping[str, Any]) -> "ShardRoutingPlan":
        return cls(tuple(_AxisRouting.from_descriptor(axis) for axis in value["axes"]))


def make_shard_routing(blocking: Any, shard_shapes: Iterable[Sequence[int]]) -> ShardRoutingPlan:
    """Build an arithmetic shard routing plan for a regular blocking."""
    shapes = [tuple(int(value) for value in shape) for shape in shard_shapes]
    if not shapes:
        raise ValueError("At least one shard shape is required.")
    ndim = int(blocking.ndim)
    spatial = [shape[-ndim:] for shape in shapes]
    periods = []
    for axis in range(ndim):
        period = 1
        for shape in spatial:
            period = period * shape[axis] // gcd(period, shape[axis])
        periods.append(period)
    axes = tuple(
        _AxisRouting.build(int(blocking.blocks_per_axis[axis]),
                           int(blocking.roi_begin[axis]), int(blocking.block_shape[axis]),
                           periods[axis])
        for axis in range(ndim)
    )
    return ShardRoutingPlan(axes)
