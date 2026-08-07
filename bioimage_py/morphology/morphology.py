"""Per-label morphology in memory or as a partitioned table dataset.

Each block computes per-label sufficient statistics. The DataFrame path returns block tables to the
main process. The file-backed path writes bounded partial tables and reduces populated label ranges
independently. Both paths return one row per nonzero label.
"""
from __future__ import annotations

import functools
import hashlib
import heapq
import json
import numbers
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..runner import get_runner
from ..runner._diagnostics import load_manifest
from ..runner._work import RegularBatchPlan
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..tables import TableDataset
from ..util import (BlockDescriptor, check_direct, check_rerun_args, derive_block_shape,
                    full_roi, get_blocking, to_roi)
from ._table_utils import resolve_source, source_identity

__all__ = ["morphology"]

_MAX_UINT64 = int(np.iinfo(np.uint64).max)
_MAX_INT64 = int(np.iinfo(np.int64).max)
_PARTIAL_OPERATION = "morphology-partials"
_FINAL_OPERATION = "morphology"


def _axis_names(ndim: int) -> List[str]:
    """Return per-axis names: ``z``/``y``/``x`` for 2D/3D, else ``axis0`` ... ``axis{ndim-1}``."""
    if ndim == 2:
        return ["y", "x"]
    if ndim == 3:
        return ["z", "y", "x"]
    return [f"axis{a}" for a in range(ndim)]


def _block_table(seg: np.ndarray, offset: Sequence[int]) -> Optional[np.ndarray]:
    """Compute the per-label morphology table for one block.

    A single sort of the flattened labels feeds every statistic (count, weighted coordinate sum and
    bounding box), which is markedly faster than ``np.unique`` + ``bincount`` (those pay for a second
    sort to group the bounding-box coordinates).

    Args:
        seg: The integer label block. ``0`` is treated as background and dropped.
        offset: The block's begin coordinate, added to make coordinates global.

    Returns:
        A ``(K, 2 + 3 * ndim)`` float64 array with columns
        ``[label, size, wsum_axis..., bb_min_axis..., bb_max_axis...]`` where ``wsum`` is the weighted
        coordinate sum (``size * com``) and ``bb_max`` is the inclusive max coordinate. ``None`` if the
        block contains no foreground.
    """
    ndim = seg.ndim
    flat = seg.ravel()
    order = np.argsort(flat)  # default quicksort: intra-group order is irrelevant to sum/min/max/count.
    sl = flat[order]
    starts = np.flatnonzero(np.concatenate(([True], sl[1:] != sl[:-1])))
    uniq = sl[starts]
    size = np.diff(np.append(starts, sl.size)).astype("float64")

    table = np.empty((uniq.shape[0], 2 + 3 * ndim), dtype="float64")
    table[:, 0] = uniq
    table[:, 1] = size
    for a in range(ndim):
        reshape = [1] * ndim
        reshape[a] = seg.shape[a]
        coord = np.broadcast_to(
            (np.arange(seg.shape[a], dtype="float64") + offset[a]).reshape(reshape), seg.shape
        ).ravel()[order]
        table[:, 2 + a] = np.add.reduceat(coord, starts)
        table[:, 2 + ndim + a] = np.minimum.reduceat(coord, starts)
        table[:, 2 + 2 * ndim + a] = np.maximum.reduceat(coord, starts)

    table = table[table[:, 0] != 0]  # drop background (label 0); uniq is sorted, so it is row 0 if present.
    return table if table.shape[0] else None


def _merge_tables(tables: List[np.ndarray], ndim: int) -> Tuple[np.ndarray, ...]:
    """Merge per-block tables into per-label statistics.

    Uses the same single-sort + ``reduceat`` scheme as :func:`_block_table`: group the stacked rows by
    label, sum the size and weighted-coordinate columns, and take the min/max of the bounding-box
    columns. The center of mass is then ``weighted_sum / size`` per axis.

    Args:
        tables: The non-``None`` per-block tables (``(_, 2 + 3 * ndim)`` arrays).
        ndim: The number of spatial dimensions.

    Returns:
        A tuple ``(labels, size, com, bb_min, bb_max)`` with shapes ``(K,)``, ``(K,)``, ``(K, ndim)``,
        ``(K, ndim)``, ``(K, ndim)``. ``bb_max`` is the inclusive max coordinate.
    """
    if not tables:
        return (np.zeros((0,), "float64"), np.zeros((0,), "float64"),
                np.zeros((0, ndim), "float64"), np.zeros((0, ndim), "float64"),
                np.zeros((0, ndim), "float64"))
    stacked = np.vstack(tables)
    order = np.argsort(stacked[:, 0])
    stacked = stacked[order]
    labels = stacked[:, 0]
    starts = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1])))

    uniq = labels[starts]
    summed = np.add.reduceat(stacked[:, 1:2 + ndim], starts, axis=0)  # [size, wsum_axis...]
    size = summed[:, 0]
    com = summed[:, 1:] / size[:, None]
    bb_min = np.minimum.reduceat(stacked[:, 2 + ndim:2 + 2 * ndim], starts, axis=0)
    bb_max = np.maximum.reduceat(stacked[:, 2 + 2 * ndim:2 + 3 * ndim], starts, axis=0)
    return uniq, size, com, bb_min, bb_max


def _to_dataframe(labels: np.ndarray, size: np.ndarray, com: np.ndarray,
                  bb_min: np.ndarray, bb_max: np.ndarray, ndim: int) -> "pd.DataFrame":
    """Assemble the per-label statistics into a sorted pandas DataFrame.

    ``bb_max`` is converted to an exclusive slice stop (``max_coordinate + 1``) so the bounding box of a
    row is ``tuple(slice(row.bb_min_<ax>, row.bb_max_<ax>) for ax in axes)``.
    """
    axes = _axis_names(ndim)
    columns = {"label": labels.astype("uint64"), "size": size.astype("int64")}
    for a, ax in enumerate(axes):
        columns[f"com_{ax}"] = com[:, a].astype("float64")
    for a, ax in enumerate(axes):
        columns[f"bb_min_{ax}"] = bb_min[:, a].astype("int64")
    for a, ax in enumerate(axes):
        columns[f"bb_max_{ax}"] = (bb_max[:, a] + 1).astype("int64")
    return pd.DataFrame(columns).sort_values("label").reset_index(drop=True)


def _morphology_block(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                      mask: Optional[Source]) -> Optional[np.ndarray]:
    """Per-block morphology table (``None`` if the block has no foreground)."""
    roi = to_roi(block)
    seg = inputs[0][roi]
    if mask is not None:
        block_mask = mask[roi].astype(bool)
        if not block_mask.any():
            return None
        seg = np.where(block_mask, seg, 0)  # Masked-out voxels fall into the dropped background.
    return _block_table(seg, list(block.begin))


def _partial_schema(axes: Sequence[str]) -> pa.Schema:
    fields = [
        pa.field("label", pa.uint64(), nullable=False),
        pa.field("size", pa.uint64(), nullable=False),
    ]
    fields.extend(pa.field(f"coord_sum_{axis}", pa.uint64(), nullable=False)
                  for axis in axes)
    fields.extend(pa.field(f"bb_min_{axis}", pa.int64(), nullable=False)
                  for axis in axes)
    fields.extend(pa.field(f"bb_max_{axis}", pa.int64(), nullable=False)
                  for axis in axes)
    return pa.schema(fields)


def _output_schema(axes: Sequence[str]) -> pa.Schema:
    fields = [
        pa.field("label", pa.uint64(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
    ]
    fields.extend(pa.field(f"com_{axis}", pa.float64(), nullable=False)
                  for axis in axes)
    fields.extend(pa.field(f"bb_min_{axis}", pa.int64(), nullable=False)
                  for axis in axes)
    fields.extend(pa.field(f"bb_max_{axis}", pa.int64(), nullable=False)
                  for axis in axes)
    return pa.schema(fields)


def _checked_group_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Group-sum uint64 values and reject overflow."""
    if values.size == 0:
        return np.zeros((0,), dtype="uint64")
    counts = np.diff(np.append(starts, values.size)).astype("uint64")
    maxima = np.maximum.reduceat(values, starts)
    safe = maxima == 0
    nonzero = ~safe
    safe[nonzero] = counts[nonzero] <= np.uint64(_MAX_UINT64) // maxima[nonzero]
    result = np.add.reduceat(values, starts, dtype="uint64")
    if not safe.all():
        for group in np.flatnonzero(~safe):
            start = int(starts[group])
            stop = start + int(counts[group])
            exact = sum(int(value) for value in values[start:stop])
            if exact > _MAX_UINT64:
                raise OverflowError("Morphology sufficient statistics exceed uint64.")
            result[group] = np.uint64(exact)
    return result


def _typed_block_table(seg: np.ndarray, offset: Sequence[int]) -> Optional[Tuple[np.ndarray, ...]]:
    """Compute exact integer sufficient statistics for one block."""
    if np.issubdtype(seg.dtype, np.signedinteger) and np.any(seg < 0):
        raise ValueError("File-backed morphology does not support negative labels.")
    flat = seg.ravel().astype("uint64", copy=False)
    order = np.argsort(flat)
    sorted_labels = flat[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1])))
    labels = sorted_labels[starts]
    counts = np.diff(np.append(starts, sorted_labels.size)).astype("uint64")

    sums = []
    minima = []
    maxima = []
    for axis in range(seg.ndim):
        high = int(offset[axis]) + int(seg.shape[axis]) - 1
        if high < 0 or high > _MAX_INT64:
            raise OverflowError("Morphology coordinates do not fit in int64.")
        reshape = [1] * seg.ndim
        reshape[axis] = seg.shape[axis]
        coordinates = np.broadcast_to(
            (np.arange(seg.shape[axis], dtype="int64") + int(offset[axis])).reshape(reshape),
            seg.shape,
        ).ravel()[order]
        unsigned = coordinates.astype("uint64", copy=False)
        sums.append(_checked_group_sum(unsigned, starts))
        minima.append(np.minimum.reduceat(coordinates, starts))
        exclusive_max = np.maximum.reduceat(coordinates, starts)
        if np.any(exclusive_max == _MAX_INT64):
            raise OverflowError("Exclusive morphology bounding boxes do not fit in int64.")
        maxima.append(exclusive_max + 1)

    keep = labels != 0
    if not keep.any():
        return None
    return tuple([labels[keep], counts[keep]]
                 + [values[keep] for values in sums]
                 + [values[keep] for values in minima]
                 + [values[keep] for values in maxima])


def _merge_partial_tables(
    tables: Sequence[Tuple[np.ndarray, ...]], ndim: int,
) -> Tuple[np.ndarray, ...]:
    """Combine one bounded block batch into sorted sufficient statistics."""
    if not tables:
        empty_u = np.zeros((0,), dtype="uint64")
        empty_i = np.zeros((0,), dtype="int64")
        return tuple([empty_u, empty_u.copy()]
                     + [empty_u.copy() for _ in range(ndim)]
                     + [empty_i.copy() for _ in range(2 * ndim)])
    columns = [np.concatenate([table[index] for table in tables])
               for index in range(2 + 3 * ndim)]
    order = np.argsort(columns[0])
    columns = [column[order] for column in columns]
    starts = np.flatnonzero(np.concatenate((
        [True], columns[0][1:] != columns[0][:-1],
    )))
    labels = columns[0][starts]
    summed = [_checked_group_sum(columns[index], starts)
              for index in range(1, 2 + ndim)]
    minima = [np.minimum.reduceat(columns[index], starts)
              for index in range(2 + ndim, 2 + 2 * ndim)]
    maxima = [np.maximum.reduceat(columns[index], starts)
              for index in range(2 + 2 * ndim, 2 + 3 * ndim)]
    return tuple([labels] + summed + minima + maxima)


def _partial_arrow(table: Tuple[np.ndarray, ...], axes: Sequence[str]) -> pa.Table:
    columns: Dict[str, Any] = {"label": table[0], "size": table[1]}
    position = 2
    for axis in axes:
        columns[f"coord_sum_{axis}"] = table[position]
        position += 1
    for axis in axes:
        columns[f"bb_min_{axis}"] = table[position]
        position += 1
    for axis in axes:
        columns[f"bb_max_{axis}"] = table[position]
        position += 1
    return pa.table(columns, schema=_partial_schema(axes))


_BLOCK_ID_CACHE: Dict[str, np.ndarray] = {}


def _block_id_at(descriptor: Mapping[str, Any], position: int) -> int:
    if descriptor["kind"] == "range":
        return int(position)
    path = str(descriptor["path"])
    values = _BLOCK_ID_CACHE.get(path)
    if values is None:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        _BLOCK_ID_CACHE[path] = values
    return int(values[int(position)])


def _partial_batch(batch: Any, writer: Any, *, context: Mapping[str, Any]) -> None:
    source = resolve_source(context["source"])
    mask = (resolve_source(context["mask"])
            if context.get("mask") is not None else None)
    blocking = get_blocking(context["shape"], context["block_shape"])
    tables = []
    for position in range(int(batch.start), int(batch.stop)):
        block_id = _block_id_at(context["block_ids"], position)
        block = blocking.get_block(block_id)
        roi = to_roi(block)
        segmentation = np.asarray(source[roi])
        if mask is not None:
            block_mask = np.asarray(mask[roi]).astype(bool, copy=False)
            if not block_mask.any():
                continue
            segmentation = np.where(block_mask, segmentation, 0)
        table = _typed_block_table(segmentation, block.begin)
        if table is not None:
            tables.append(table)
    merged = _merge_partial_tables(tables, int(context["ndim"]))
    writer.write(_partial_arrow(merged, context["axes"]))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str, value: Mapping[str, Any]) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    descriptor, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=folder,
    )
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _prepare_block_ids(
    work_dir: str,
    block_ids: Optional[Sequence[int]],
    n_blocks: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Persist explicit block IDs and return worker and identity descriptors."""
    if block_ids is None:
        descriptor = {"kind": "range", "length": int(n_blocks)}
        return descriptor, descriptor.copy()

    length = len(block_ids)
    os.makedirs(work_dir, exist_ok=True)
    final_path = os.path.join(work_dir, "block_ids.npy")
    temporary_path = os.path.join(work_dir, f".block_ids.{os.getpid()}.npy")
    values: Optional[np.ndarray] = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype="int64", shape=(length,),
    )
    try:
        chunk_size = 1_000_000
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            chunk = np.asarray(block_ids[start:stop])
            if not np.issubdtype(chunk.dtype, np.integer):
                raise TypeError("block_ids must contain integers.")
            chunk = chunk.astype("int64", copy=False)
            if np.any(chunk < 0) or np.any(chunk >= int(n_blocks)):
                raise ValueError("block_ids contains an ID outside the block grid.")
            values[start:stop] = chunk
        values.flush()
        del values
        values = None
        checksum = _sha256_file(temporary_path)
        identity = {"kind": "explicit", "length": int(length), "sha256": checksum}
        state_path = os.path.join(work_dir, "block_ids.json")
        if os.path.exists(state_path):
            with open(state_path) as file:
                existing = json.load(file)
            if existing != identity:
                raise ValueError(
                    f"Existing morphology work directory {work_dir!r} uses different block_ids."
                )
            os.unlink(temporary_path)
        else:
            os.replace(temporary_path, final_path)
            _atomic_json(state_path, identity)
    finally:
        if values is not None:
            del values
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
    return {**identity, "path": final_path}, identity


def _index_record(index_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(index_dir, "index.json")
    try:
        with open(path) as file:
            record = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    for name in ("range_ids.npy", "offsets.npy", "part_ids.npy"):
        if not os.path.isfile(os.path.join(index_dir, name)):
            return None
    return record


def _raw_uint64_to_npy(raw_path: str, output_path: str, count: int) -> None:
    """Copy one raw uint64 stream into a memory-mappable NumPy file."""
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype="uint64", shape=(int(count),),
    )
    if count:
        source = np.memmap(raw_path, mode="r", dtype="uint64", shape=(int(count),))
        chunk_size = 1_000_000
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            output[start:stop] = source[start:stop]
        del source
    output.flush()
    del output


def _build_partition_index(
    partial: TableDataset,
    index_dir: str,
    label_partition_size: int,
) -> Dict[str, Any]:
    """Build a disk-backed label-range to partial-part index."""
    manifest = partial._manifest_record()
    manifest_parts = json.dumps(
        manifest["parts"], sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    expected = {
        "version": 3,
        "partial_identity": partial._descriptor()["identity"],
        "partial_content_fingerprint": hashlib.sha256(manifest_parts).hexdigest(),
        "label_partition_size": int(label_partition_size),
    }
    existing = _index_record(index_dir)
    if existing is not None and all(existing.get(key) == value
                                    for key, value in expected.items()):
        return existing

    parent = os.path.dirname(index_dir)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=".partition-index.", dir=parent)
    try:
        run_dir = os.path.join(temporary, "runs")
        os.makedirs(run_dir)
        runs = []
        for part in partial.iter_parts():
            run_path = os.path.join(run_dir, f"part-{int(part.batch_id):012d}.bin")
            count = 0
            last_range = None
            with open(run_path, "wb") as output:
                parquet = pq.ParquetFile(part.path)
                for batch in parquet.iter_batches(
                    batch_size=1_000_000, columns=["label"],
                ):
                    labels = batch.column(0).to_numpy(
                        zero_copy_only=False,
                    ).astype("uint64", copy=False)
                    if not labels.size:
                        continue
                    range_ids = np.unique(
                        labels // np.uint64(label_partition_size),
                    )
                    if last_range is not None and int(range_ids[0]) == last_range:
                        range_ids = range_ids[1:]
                    if range_ids.size:
                        output.write(range_ids.astype("uint64", copy=False).tobytes())
                        count += int(range_ids.size)
                        last_range = int(range_ids[-1])
            if count:
                runs.append((int(part.batch_id), run_path, count))

        mapped_runs = {
            part_id: np.memmap(path, mode="r", dtype="uint64", shape=(count,))
            for part_id, path, count in runs
        }
        heap = [
            (int(values[0]), part_id, 0)
            for part_id, values in mapped_runs.items()
        ]
        heapq.heapify(heap)
        raw_ranges = os.path.join(temporary, "range_ids.bin")
        raw_offsets = os.path.join(temporary, "offsets.bin")
        raw_parts = os.path.join(temporary, "part_ids.bin")
        n_ranges = 0
        n_part_references = 0
        current_range = None
        with (open(raw_ranges, "wb") as ranges_output,
              open(raw_offsets, "wb") as offsets_output,
              open(raw_parts, "wb") as parts_output):
            while heap:
                range_id, part_id, position = heapq.heappop(heap)
                if range_id != current_range:
                    ranges_output.write(np.uint64(range_id).tobytes())
                    offsets_output.write(np.uint64(n_part_references).tobytes())
                    n_ranges += 1
                    current_range = range_id
                parts_output.write(np.uint64(part_id).tobytes())
                n_part_references += 1
                position += 1
                values = mapped_runs[part_id]
                if position < len(values):
                    heapq.heappush(
                        heap, (int(values[position]), part_id, position),
                    )
            offsets_output.write(np.uint64(n_part_references).tobytes())
        mapped_runs.clear()
        shutil.rmtree(run_dir)

        for raw_name, output_name, count in (
            (raw_ranges, "range_ids.npy", n_ranges),
            (raw_offsets, "offsets.npy", n_ranges + 1),
            (raw_parts, "part_ids.npy", n_part_references),
        ):
            _raw_uint64_to_npy(
                raw_name, os.path.join(temporary, output_name), count,
            )
            os.unlink(raw_name)
        record = {
            **expected,
            "n_ranges": n_ranges,
            "n_part_references": n_part_references,
        }
        _atomic_json(os.path.join(temporary, "index.json"), record)
        if os.path.isdir(index_dir):
            shutil.rmtree(index_dir)
        os.replace(temporary, index_dir)
    finally:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)
    return record


def _arrow_uint64(table: pa.Table, name: str) -> np.ndarray:
    column = table[name]
    if column.null_count:
        raise ValueError(f"Partial morphology column {name!r} contains null values.")
    return column.cast(pa.uint64(), safe=True).to_numpy(zero_copy_only=False)


def _arrow_int64(table: pa.Table, name: str) -> np.ndarray:
    column = table[name]
    if column.null_count:
        raise ValueError(f"Partial morphology column {name!r} contains null values.")
    return column.cast(pa.int64(), safe=True).to_numpy(zero_copy_only=False)


def _checked_index_add(target: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    current = target[indices]
    if np.any(values > np.uint64(_MAX_UINT64) - current):
        raise OverflowError("Morphology sufficient statistics exceed uint64.")
    target[indices] = current + values


def _reduce_partition(batch: Any, writer: Any, *, context: Mapping[str, Any]) -> None:
    index_dir = str(context["index_dir"])
    range_ids = np.load(os.path.join(index_dir, "range_ids.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(index_dir, "offsets.npy"), mmap_mode="r")
    part_ids = np.load(os.path.join(index_dir, "part_ids.npy"), mmap_mode="r")
    position = int(batch.start)
    range_id = int(range_ids[position])
    partition_size = int(context["label_partition_size"])
    label_start = range_id * partition_size
    label_stop = min((range_id + 1) * partition_size, _MAX_UINT64 + 1)
    axes = tuple(context["axes"])
    ndim = len(axes)

    present = np.zeros(partition_size, dtype=bool)
    size = np.zeros(partition_size, dtype="uint64")
    coordinate_sums = np.zeros((ndim, partition_size), dtype="uint64")
    bb_min = np.full((ndim, partition_size), _MAX_INT64, dtype="int64")
    bb_max = np.full((ndim, partition_size), np.iinfo(np.int64).min, dtype="int64")

    columns = (["label", "size"] + [f"coord_sum_{axis}" for axis in axes]
               + [f"bb_min_{axis}" for axis in axes]
               + [f"bb_max_{axis}" for axis in axes])
    start = int(offsets[position])
    stop = int(offsets[position + 1])
    for part_id in part_ids[start:stop]:
        part_path = os.path.join(
            context["partial_path"], "parts", f"part-{int(part_id):012d}.parquet",
        )
        filters = [("label", ">=", np.uint64(label_start))]
        if label_stop <= _MAX_UINT64:
            filters.append(("label", "<", np.uint64(label_stop)))
        table = pq.read_table(part_path, columns=columns, filters=filters)
        if not table.num_rows:
            continue
        labels = _arrow_uint64(table, "label")
        indices = (labels - np.uint64(label_start)).astype("int64")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("A partial morphology part contains duplicate labels.")
        if np.any(indices < 0) or np.any(indices >= partition_size):
            raise ValueError("A partial morphology label is outside its reduction range.")
        _checked_index_add(size, indices, _arrow_uint64(table, "size"))
        for axis_index, axis in enumerate(axes):
            _checked_index_add(
                coordinate_sums[axis_index], indices,
                _arrow_uint64(table, f"coord_sum_{axis}"),
            )
            bb_min[axis_index, indices] = np.minimum(
                bb_min[axis_index, indices], _arrow_int64(table, f"bb_min_{axis}"),
            )
            bb_max[axis_index, indices] = np.maximum(
                bb_max[axis_index, indices], _arrow_int64(table, f"bb_max_{axis}"),
            )
        present[indices] = True

    indices = np.flatnonzero(present)
    labels = np.asarray(label_start + indices.astype(object), dtype="uint64")
    kept_size = size[indices]
    if np.any(kept_size > np.uint64(_MAX_INT64)):
        raise OverflowError("Morphology size does not fit in the public int64 schema.")
    columns_out: Dict[str, Any] = {
        "label": labels,
        "size": kept_size.astype("int64"),
    }
    for axis_index, axis in enumerate(axes):
        sums = coordinate_sums[axis_index, indices].astype("float64")
        columns_out[f"com_{axis}"] = sums / kept_size.astype("float64")
    for axis_index, axis in enumerate(axes):
        columns_out[f"bb_min_{axis}"] = bb_min[axis_index, indices]
    for axis_index, axis in enumerate(axes):
        columns_out[f"bb_max_{axis}"] = bb_max[axis_index, indices]
    writer.write(pa.table(columns_out, schema=_output_schema(axes)))


def _resume_sink_path(resume_from: Optional[str], job_type: str) -> Optional[str]:
    if resume_from is None:
        return None
    manifest = load_manifest(resume_from, expected_backend=job_type.lower())
    sink = manifest.get("result_sink")
    if not isinstance(sink, dict):
        raise ValueError("The resumed morphology run has no table result sink.")
    return os.path.abspath(str(sink.get("path", "")))


def _file_backed_morphology(
    src: Source,
    *,
    output_table: Union[str, os.PathLike[str]],
    blocks_per_batch: int,
    label_partition_size: int,
    provenance: Optional[Mapping[str, Any]],
    num_workers: int,
    block_shape: Optional[Tuple[int, ...]],
    job_type: str,
    job_config: Optional[RunnerConfig],
    mask: Optional[SourceLike],
    block_ids: Optional[Sequence[int]],
    resume_from: Optional[str],
    pre_cleanup: Optional[Callable[[str], None]],
) -> TableDataset:
    for name, value in (("blocks_per_batch", blocks_per_batch),
                        ("label_partition_size", label_partition_size)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer.")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    blocks_per_batch = int(blocks_per_batch)
    label_partition_size = int(label_partition_size)
    if label_partition_size > _MAX_UINT64:
        raise ValueError("label_partition_size must fit in uint64.")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None.")

    try:
        source_spec = src.to_spec()
    except ValueError as error:
        raise ValueError(
            f"File-backed morphology requires a reopenable segmentation. {error}"
        ) from error
    mask_source = as_source(mask) if mask is not None else None
    if mask_source is not None:
        if tuple(mask_source.shape) != tuple(src.shape):
            raise ValueError("Morphology mask shape does not match the segmentation.")
        try:
            mask_spec = mask_source.to_spec()
        except ValueError as error:
            raise ValueError(
                f"File-backed morphology requires a reopenable mask. {error}"
            ) from error
    else:
        mask_spec = None

    resolved_block_shape = derive_block_shape(src, block_shape)
    blocking = get_blocking(src.shape, resolved_block_shape)
    n_blocks = int(blocking.number_of_blocks)
    output_path = os.path.abspath(os.fspath(output_table))
    work_dir = output_path + ".morphology-work"
    os.makedirs(work_dir, exist_ok=True)
    block_descriptor, block_identity = _prepare_block_ids(
        work_dir, block_ids, n_blocks,
    )
    n_selected = int(block_descriptor["length"])
    axes = tuple(_axis_names(src.ndim))
    input_identities: Dict[str, Any] = {
        "segmentation": source_identity(source_spec, src),
        "block_ids": block_identity,
    }
    if mask_spec is not None:
        input_identities["mask"] = source_identity(mask_spec, mask_source)
    parameters = {
        "axes": list(axes),
        "block_shape": [int(value) for value in resolved_block_shape],
        "blocks_per_batch": blocks_per_batch,
        "label_partition_size": label_partition_size,
        "provenance": dict(provenance or {}),
    }

    final = TableDataset.create(
        output_path,
        schema=_output_schema(axes),
        schema_version=1,
        operation=_FINAL_OPERATION,
        operation_version="1",
        input_identities=input_identities,
        parameters=parameters,
    )
    if final.complete:
        try:
            final.validate()
        except ValueError:
            pass
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            return final

    partial_path = os.path.join(work_dir, "partials")
    partial = TableDataset.create(
        partial_path,
        schema=_partial_schema(axes),
        schema_version=1,
        operation=_PARTIAL_OPERATION,
        operation_version="1",
        input_identities=input_identities,
        parameters=parameters,
    )
    runner = get_runner(job_type, job_config)
    sink_path = _resume_sink_path(resume_from, job_type)
    valid_sinks = {os.path.abspath(partial_path), os.path.abspath(output_path)}
    if sink_path is not None and sink_path not in valid_sinks:
        raise ValueError("resume_from belongs to a different morphology output.")

    reuse_partial = False
    if partial.complete:
        try:
            partial.validate()
        except ValueError:
            pass
        else:
            reuse_partial = True

    if not reuse_partial:
        context = {
            "source": source_spec,
            "mask": mask_spec,
            "shape": tuple(int(value) for value in src.shape),
            "block_shape": tuple(int(value) for value in resolved_block_shape),
            "block_ids": block_descriptor,
            "axes": axes,
            "ndim": src.ndim,
        }
        partial = runner.map_batches(
            functools.partial(_partial_batch, context=context),
            n_items=n_selected,
            batch_size=blocks_per_batch,
            num_workers=num_workers,
            name=_PARTIAL_OPERATION,
            pre_cleanup=pre_cleanup,
            resume_from=(resume_from if sink_path == os.path.abspath(partial_path) else None),
            result_sink=partial,
        )
    index_dir = os.path.join(work_dir, "partition-index")
    index = _build_partition_index(partial, index_dir, label_partition_size)
    range_ids = np.load(os.path.join(index_dir, "range_ids.npy"), mmap_mode="r")
    plan = RegularBatchPlan(int(index["n_ranges"]), 1)
    partition_metadata = []
    for range_id in range_ids:
        start = int(range_id) * label_partition_size
        partition_metadata.append({
            "label_start": start,
            "label_stop": min(start + label_partition_size, _MAX_UINT64 + 1),
        })
    final._bind_batches(plan, partition_metadata=partition_metadata)

    context = {
        "partial_path": partial_path,
        "index_dir": index_dir,
        "label_partition_size": label_partition_size,
        "axes": axes,
    }
    final = runner.map_batches(
        functools.partial(_reduce_partition, context=context),
        n_items=len(plan),
        batch_size=1,
        num_workers=num_workers,
        name="morphology-reduce",
        pre_cleanup=pre_cleanup,
        resume_from=(resume_from if sink_path == os.path.abspath(output_path) else None),
        result_sink=final,
    )
    final.validate()
    shutil.rmtree(work_dir)
    return final


def morphology(
    input: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
    pre_cleanup: Optional[Callable[[str], None]] = None,
    output_table: Optional[Union[str, os.PathLike[str]]] = None,
    blocks_per_batch: int = 1_000,
    label_partition_size: int = 1_000_000,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Union["pd.DataFrame", TableDataset]:
    """Compute per-label morphology (size, center of mass, bounding box) of a labeled volume.

    Statistics are computed block-wise and merged so the result is exact regardless of how labels
    straddle block boundaries. The background label ``0`` is excluded. Set ``output_table`` to
    write bounded Parquet parts and return a `TableDataset`.

    Args:
        input: The input label image (a numpy/zarr/n5 array or a `Source`); must be integer-typed.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; voxels outside the mask are excluded from the computation.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks);
            the table then reflects only those blocks.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). The returned table then covers the full volume (the
            already-completed blocks merged with the re-run ones). Mutually exclusive with
            ``block_ids``.
        pre_cleanup: Optional ``pre_cleanup(tmp_folder)`` callback invoked on the orchestrating
            process with the job temp folder right before it is deleted (distributed backends only).
            Use it to read out the per-task timing files under ``tmp_folder/timings/`` before cleanup.
            Ignored for the ``local`` backend and for the direct (single-worker, unchunked) path, which
            have no temp folder.
        output_table: Optional partitioned Parquet dataset directory. This path selects the
            scalable two-stage implementation and requires reopenable inputs.
        blocks_per_batch: Image blocks per durable partial-table batch.
        label_partition_size: Maximum label span reduced by one final-table batch.
        provenance: Optional canonical-JSON metadata included in the dataset identity.

    Returns:
        A pandas DataFrame when ``output_table`` is absent. Otherwise, a `TableDataset` with the
        same ordered columns and row semantics. The bounding box is slice-ready: ``bb_min`` is
        inclusive and ``bb_max`` is the exclusive stop.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"morphology expects an integer label image, got dtype {src.dtype}.")
    ndim = src.ndim

    if output_table is not None:
        return _file_backed_morphology(
            src,
            output_table=output_table,
            blocks_per_batch=blocks_per_batch,
            label_partition_size=label_partition_size,
            provenance=provenance,
            num_workers=num_workers,
            block_shape=block_shape,
            job_type=job_type,
            job_config=job_config,
            mask=mask,
            block_ids=block_ids,
            resume_from=resume_from,
            pre_cleanup=pre_cleanup,
        )
    if provenance is not None:
        raise ValueError("provenance requires output_table.")

    if check_direct(job_type, num_workers, block_shape, mask, block_ids):
        table = _block_table(src[full_roi(src.ndim)], [0] * ndim)
        tables = [table] if table is not None else []
    else:
        runner = get_runner(job_type, job_config)
        results = runner.run(_morphology_block, [input], num_workers=num_workers,
                             block_shape=block_shape, mask=mask, block_ids=block_ids,
                             resume_from=resume_from, has_return_val=True, name="morphology",
                             pre_cleanup=pre_cleanup)
        tables = [r for r in results if r is not None]

    return _to_dataframe(*_merge_tables(tables, ndim), ndim)
