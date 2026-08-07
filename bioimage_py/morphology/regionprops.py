"""Compute per-object morphology features from a base morphology table."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import numbers
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import bioimage_cpp as bic
import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa

from ..runner import get_runner
from ..runner._diagnostics import load_manifest
from ..runner._work import RegularBatchPlan
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, SourceSpec, as_source
from ..tables import TableDataset, _describe_parquet_input, _read_parquet_rows
from ..util import check_rerun_args
from .morphology import _axis_names

__all__ = ["regionprops"]

_SEG_CACHE: Dict[Any, Source] = {}


def _read_table(path: str) -> "pd.DataFrame":
    """Read a serialized table from a ``.csv`` / ``.xlsx`` path."""
    lower = str(path).lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path!r} (use .csv or .xlsx/.xls).")


def _load_table(table: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
    """Return ``table`` as a DataFrame (pass-through, or read from a path)."""
    if isinstance(table, pd.DataFrame):
        return table
    return _read_table(str(table))


def _required_columns(axes: Sequence[str]) -> List[str]:
    """The base-morphology columns this op consumes."""
    return (["label"] + [f"com_{a}" for a in axes]
            + [f"bb_min_{a}" for a in axes] + [f"bb_max_{a}" for a in axes])


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {key: item for key, item in sorted(
            (str(key), _json_value(item)) for key, item in value.items()
        )}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, slice):
        return {"slice": [_json_value(value.start), _json_value(value.stop),
                          _json_value(value.step)]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    serialized = cloudpickle.dumps(value)
    value_type = type(value)
    return {
        "python_type": f"{value_type.__module__}.{value_type.__qualname__}",
        "cloudpickle_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _source_identity(spec: SourceSpec, source: Source) -> Dict[str, Any]:
    return {
        "spec": _json_value(spec),
        "shape": [int(size) for size in source.shape],
        "dtype": np.dtype(source.dtype).str,
    }


def _resolve_seg(seg: Union[Source, Any]) -> Source:
    """Return an opened segmentation source, reopening (and caching) a `SourceSpec` if needed."""
    if isinstance(seg, Source):
        return seg
    key = json.dumps(_json_value(seg), sort_keys=True, separators=(",", ":"))
    src = _SEG_CACHE.get(key)
    if src is None:
        from ..sources.dispatch import from_spec
        src = from_spec(seg)
        _SEG_CACHE[key] = src
    return src


def _column_arrays(df: "pd.DataFrame", axes: Sequence[str]) -> Dict[str, np.ndarray]:
    """Extract the consumed columns as numpy arrays (built once, indexed by row position).

    Indexing these arrays by position avoids per-object pandas scalar access, which dominates the
    cost when there are many objects. Returns ``label`` (``(N,)``) and ``com`` / ``bb_min`` /
    ``bb_max`` (each ``(N, ndim)``).
    """
    return {
        "label": df["label"].to_numpy(dtype="int64"),
        "com": np.stack([df[f"com_{a}"].to_numpy(dtype="float64") for a in axes], axis=1),
        "bb_min": np.stack([df[f"bb_min_{a}"].to_numpy(dtype="int64") for a in axes], axis=1),
        "bb_max": np.stack([df[f"bb_max_{a}"].to_numpy(dtype="int64") for a in axes], axis=1),
    }


def _surface_area(mask: np.ndarray, resolution: Sequence[float]) -> float:
    """Physical surface area (nm² etc.) of a 3D mask via marching cubes; 0.0 if degenerate."""
    from skimage.measure import marching_cubes, mesh_surface_area
    if not mask.any():
        return 0.0
    # Pad by 1 so border voxels produce closed faces; spacing makes the area physical.
    padded = np.pad(mask, 1).astype("float32")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(resolution))
        return float(mesh_surface_area(verts, faces))
    except (RuntimeError, ValueError):
        # Objects too thin/small for a closed surface.
        return 0.0


def _axis_lengths(coords: np.ndarray, spacing: np.ndarray, ndim: int) -> "tuple[float, float]":
    """Major/minor axis lengths (physical) of the inertia-equivalent ellipse/ellipsoid.

    Mirrors scikit-image: the inertia tensor is ``trace(C) * I - C`` for the population covariance
    ``C`` of the (physical) voxel coordinates; the axis lengths derive from its eigenvalues (sorted
    descending). Returns ``(0.0, 0.0)`` for an empty object.
    """
    n = coords.shape[0]
    if n == 0:
        return 0.0, 0.0
    x = coords.astype("float64") * spacing
    xc = x - x.mean(axis=0)
    cov = (xc.T @ xc) / n  # population covariance == central second moments / area.
    inertia = np.trace(cov) * np.eye(ndim) - cov
    ev = np.sort(np.clip(np.linalg.eigvalsh(inertia), 0.0, None))[::-1]  # descending.
    if ndim == 3:
        major = float(np.sqrt(max(0.0, 10.0 * (ev[0] + ev[1] - ev[2]))))
        minor = float(np.sqrt(max(0.0, 10.0 * (-ev[0] + ev[1] + ev[2]))))
        return major, minor
    # 2D (and the nD fallback): the ellipse axes are 4 * sqrt(extreme eigenvalues).
    return float(4.0 * np.sqrt(ev[0])), float(4.0 * np.sqrt(ev[-1]))


def _corrected_centroid(mask: np.ndarray, com_global_vox: np.ndarray, origin: np.ndarray,
                        resolution: Sequence[float]) -> np.ndarray:
    """Return the corrected centroid in physical units.

    Keeps the center-of-mass when it lands inside the object; otherwise uses the deepest-interior
    voxel (argmax of the Euclidean distance transform), then scales to physical units.
    """
    spacing = np.asarray(resolution, dtype="float64")
    if not mask.any():
        return com_global_vox * spacing
    com_local = com_global_vox - origin
    rounded = np.round(com_local).astype(int)
    inside = (np.all(rounded >= 0) and np.all(rounded < np.asarray(mask.shape))
              and bool(mask[tuple(rounded)]))
    if inside:
        centroid_vox = com_global_vox
    else:
        # Deepest-interior point: argmax of the (anisotropic) exact EDT from bioimage_cpp.
        dt = bic.distance.distance_transform(mask.astype("uint8"), sampling=tuple(resolution))
        idx = np.unravel_index(int(np.argmax(dt)), mask.shape)
        centroid_vox = np.asarray(idx, dtype="float64") + origin
    return centroid_vox * spacing


def _compute_object_features(
    seg: Source,
    label: int,
    com: np.ndarray,
    bb_min: np.ndarray,
    bb_max: np.ndarray,
    resolution: Sequence[float],
    ndim: int,
    compute_surface: bool,
) -> tuple:
    res = np.asarray(resolution, dtype="float64")
    crop = np.asarray(seg[tuple(slice(int(lo), int(hi)) for lo, hi in zip(bb_min, bb_max))])
    mask = crop == label
    coords = np.argwhere(mask)
    n_voxels = int(coords.shape[0])
    area = n_voxels * float(np.prod(res))

    bbox_voxels = int(np.prod(bb_max - bb_min))
    major, minor = _axis_lengths(coords, res, ndim)
    extent = (n_voxels / bbox_voxels) if bbox_voxels > 0 else float("nan")
    equivalent_diameter = (
        (2 * ndim * area / np.pi) ** (1.0 / ndim) if area > 0 else 0.0
    )
    surface = _surface_area(mask, resolution) if compute_surface and ndim == 3 else None
    centroid = _corrected_centroid(mask, com, bb_min.astype("float64"), resolution)
    return (n_voxels, area, extent, equivalent_diameter, major, minor, centroid, surface)


def _object_features(index: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compute one feature row for the DataFrame execution path."""
    seg = _resolve_seg(ctx["seg"])
    axes, ndim = ctx["axes"], ctx["ndim"]
    label = int(ctx["label"][index])
    com = ctx["com"][index]
    bb_min = ctx["bb_min"][index]
    bb_max = ctx["bb_max"][index]
    values = _compute_object_features(
        seg, label, com, bb_min, bb_max, ctx["resolution"], ndim,
        ctx["compute_surface"],
    )
    n_voxels, area, extent, equivalent_diameter, major, minor, centroid, surface = values
    result: Dict[str, Any] = {
        "label": label,
        "n_voxels": n_voxels,
        "area": area,
        "extent": extent,
        "equivalent_diameter_area": equivalent_diameter,
        "axis_major_length": major,
        "axis_minor_length": minor,
    }
    if surface is not None:
        result["surface_area"] = surface
    for a, ax in enumerate(axes):
        result[f"centroid_{ax}"] = float(centroid[a])
    for a, ax in enumerate(axes):
        result[f"bb_min_{ax}"] = int(bb_min[a])
        result[f"bb_max_{ax}"] = int(bb_max[a])
    return result


def _order_columns(axes: Sequence[str], compute_surface: bool, ndim: int) -> List[str]:
    """The fixed output column order: identity, shape features, geometry, then surface."""
    cols = ["label", "n_voxels", "area", "extent", "equivalent_diameter_area",
            "axis_major_length", "axis_minor_length"]
    cols += [f"centroid_{a}" for a in axes]
    cols += [f"bb_min_{a}" for a in axes] + [f"bb_max_{a}" for a in axes]
    if compute_surface and ndim == 3:
        cols.append("surface_area")
    return cols


def _output_schema(axes: Sequence[str], compute_surface: bool, ndim: int) -> pa.Schema:
    fields = [
        pa.field("label", pa.uint64(), nullable=False),
        pa.field("n_voxels", pa.uint64(), nullable=False),
        pa.field("area", pa.float64(), nullable=False),
        pa.field("extent", pa.float64(), nullable=False),
        pa.field("equivalent_diameter_area", pa.float64(), nullable=False),
        pa.field("axis_major_length", pa.float64(), nullable=False),
        pa.field("axis_minor_length", pa.float64(), nullable=False),
    ]
    fields.extend(pa.field(f"centroid_{axis}", pa.float64(), nullable=False)
                  for axis in axes)
    fields.extend(pa.field(f"bb_min_{axis}", pa.int64(), nullable=False)
                  for axis in axes)
    fields.extend(pa.field(f"bb_max_{axis}", pa.int64(), nullable=False)
                  for axis in axes)
    if compute_surface and ndim == 3:
        fields.append(pa.field("surface_area", pa.float64(), nullable=False))
    return pa.schema(fields)


def _validate_file_table_schema(schema: pa.Schema, axes: Sequence[str]) -> None:
    missing = [name for name in _required_columns(axes) if name not in schema.names]
    if missing:
        raise ValueError(
            f"table is missing required columns {missing}; pass the output of "
            "bioimage_py.morphology.morphology."
        )
    integer_columns = ["label"]
    integer_columns.extend(f"bb_min_{axis}" for axis in axes)
    integer_columns.extend(f"bb_max_{axis}" for axis in axes)
    for name in integer_columns:
        if not pa.types.is_integer(schema.field(name).type):
            raise ValueError(f"table column {name!r} must have an integer type.")
    for axis in axes:
        name = f"com_{axis}"
        column_type = schema.field(name).type
        if not (pa.types.is_integer(column_type) or pa.types.is_floating(column_type)):
            raise ValueError(f"table column {name!r} must have a numeric type.")


def _batch_column(table: pa.Table, name: str, dtype: pa.DataType) -> np.ndarray:
    column = table[name]
    if column.null_count:
        raise ValueError(f"table column {name!r} contains null values.")
    try:
        cast = column.cast(dtype, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise ValueError(f"table column {name!r} cannot be converted to {dtype}.") from error
    return cast.to_numpy(zero_copy_only=False)


def _regionprops_batch(batch: Any, writer: Any, *, ctx: Dict[str, Any]) -> None:
    axes = ctx["axes"]
    ndim = ctx["ndim"]
    required = _required_columns(axes)
    table = _read_parquet_rows(ctx["table"], batch.start, batch.stop, required)
    if table.num_rows != batch.size:
        raise ValueError(
            f"Batch {batch.batch_id} read {table.num_rows} rows, expected {batch.size}."
        )

    labels = _batch_column(table, "label", pa.uint64()).astype("uint64", copy=False)
    com = np.stack([
        _batch_column(table, f"com_{axis}", pa.float64()) for axis in axes
    ], axis=1)
    bb_min = np.stack([
        _batch_column(table, f"bb_min_{axis}", pa.int64()) for axis in axes
    ], axis=1)
    bb_max = np.stack([
        _batch_column(table, f"bb_max_{axis}", pa.int64()) for axis in axes
    ], axis=1)
    if not np.isfinite(com).all():
        raise ValueError(f"Batch {batch.batch_id} contains a non-finite center of mass.")

    seg = _resolve_seg(ctx["seg"])
    shape = np.asarray(seg.shape, dtype="int64")
    invalid_boxes = ((bb_min < 0).any(axis=1) | (bb_max < bb_min).any(axis=1)
                     | (bb_max > shape).any(axis=1))
    if invalid_boxes.any():
        index = int(np.flatnonzero(invalid_boxes)[0]) + int(batch.start)
        raise ValueError(f"table row {index} has a bounding box outside the segmentation.")

    row_count = int(table.num_rows)
    n_voxels = np.empty(row_count, dtype="uint64")
    area = np.empty(row_count, dtype="float64")
    extent = np.empty(row_count, dtype="float64")
    equivalent_diameter = np.empty(row_count, dtype="float64")
    major = np.empty(row_count, dtype="float64")
    minor = np.empty(row_count, dtype="float64")
    centroid = np.empty((row_count, ndim), dtype="float64")
    surface = (np.empty(row_count, dtype="float64")
               if ctx["compute_surface"] and ndim == 3 else None)

    for index in range(row_count):
        values = _compute_object_features(
            seg, int(labels[index]), com[index], bb_min[index], bb_max[index],
            ctx["resolution"], ndim, ctx["compute_surface"],
        )
        (n_voxels[index], area[index], extent[index], equivalent_diameter[index],
         major[index], minor[index], centroid[index], surface_value) = values
        if surface is not None:
            surface[index] = surface_value

    columns: Dict[str, Any] = {
        "label": labels,
        "n_voxels": n_voxels,
        "area": area,
        "extent": extent,
        "equivalent_diameter_area": equivalent_diameter,
        "axis_major_length": major,
        "axis_minor_length": minor,
    }
    columns.update({f"centroid_{axis}": centroid[:, index]
                    for index, axis in enumerate(axes)})
    columns.update({f"bb_min_{axis}": bb_min[:, index]
                    for index, axis in enumerate(axes)})
    columns.update({f"bb_max_{axis}": bb_max[:, index]
                    for index, axis in enumerate(axes)})
    if surface is not None:
        columns["surface_area"] = surface
    writer.write(pa.table(columns, schema=_output_schema(
        axes, ctx["compute_surface"], ndim,
    )))


def _write_table(out: "pd.DataFrame", output_path: str) -> None:
    """Write the result table to a ``.csv`` / ``.xlsx`` path (creating parent dirs)."""
    parent = os.path.dirname(os.path.abspath(str(output_path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    lower = str(output_path).lower()
    if lower.endswith((".xlsx", ".xls")):
        out.to_excel(output_path, index=False)
    elif lower.endswith(".csv"):
        out.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path!r} (use .csv or .xlsx/.xls).")


def _dataframe_regionprops(
    src: Source,
    table: Union[str, "pd.DataFrame"],
    *,
    resolution: Sequence[float],
    compute_surface: bool,
    output_path: Optional[str],
    num_workers: int,
    job_type: str,
    job_config: Optional[RunnerConfig],
    item_ids: Optional[Sequence[int]],
    resume_from: Optional[str],
    pre_cleanup: Optional[Callable[[str], None]],
) -> "pd.DataFrame":
    ndim = src.ndim
    axes = _axis_names(ndim)
    df = _load_table(table)
    missing = [column for column in _required_columns(axes) if column not in df.columns]
    if missing:
        raise ValueError(
            f"table is missing required columns {missing}; pass the output of "
            "bioimage_py.morphology.morphology (label / com_* / bb_min_* / bb_max_*)."
        )

    n_items = len(df)
    if n_items == 0:
        columns = _order_columns(axes, compute_surface, ndim)
        return pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})

    seg_arg: Any = src
    if job_type != "local":
        try:
            seg_arg = src.to_spec()
        except ValueError as error:
            raise ValueError(
                f"Distributed regionprops requires a file-backed segmentation. {error}"
            ) from error

    ctx = {
        "seg": seg_arg,
        "resolution": resolution,
        "axes": tuple(axes),
        "ndim": ndim,
        "compute_surface": bool(compute_surface),
        **_column_arrays(df, axes),
    }
    runner = get_runner(job_type, job_config)
    results = runner.map(
        functools.partial(_object_features, ctx=ctx),
        n_items,
        item_ids=item_ids,
        resume_from=resume_from,
        num_workers=num_workers,
        has_return_val=True,
        name="regionprops",
        pre_cleanup=pre_cleanup,
    )
    out = pd.DataFrame(results)
    out = out[_order_columns(axes, compute_surface, ndim)].sort_values("label")
    out = out.reset_index(drop=True)
    if output_path is not None:
        _write_table(out, output_path)
    return out


def _overlapping_paths(first: str, second: str) -> bool:
    first = os.path.realpath(first)
    second = os.path.realpath(second)
    try:
        common = os.path.commonpath([first, second])
    except ValueError:
        return False
    return common == first or common == second


def _file_backed_regionprops(
    src: Source,
    table: Any,
    *,
    resolution: Sequence[float],
    compute_surface: bool,
    output_table: Union[str, os.PathLike[str]],
    rows_per_batch: int,
    num_workers: int,
    job_type: str,
    job_config: Optional[RunnerConfig],
    resume_from: Optional[str],
    pre_cleanup: Optional[Callable[[str], None]],
) -> TableDataset:
    if isinstance(rows_per_batch, bool) or not isinstance(rows_per_batch, numbers.Integral):
        raise TypeError("rows_per_batch must be an integer.")
    rows_per_batch = int(rows_per_batch)
    if rows_per_batch <= 0:
        raise ValueError("rows_per_batch must be positive.")

    runner = get_runner(job_type, job_config)
    try:
        seg_spec = src.to_spec()
    except ValueError as error:
        raise ValueError(
            f"File-backed regionprops requires a reopenable segmentation. {error}"
        ) from error

    axes = tuple(_axis_names(src.ndim))
    table_descriptor, table_schema, table_identity = _describe_parquet_input(
        table, _required_columns(axes),
    )
    _validate_file_table_schema(table_schema, axes)
    output_path = os.path.abspath(os.fspath(output_table))
    if _overlapping_paths(output_path, table_descriptor.path):
        raise ValueError("output_table must not contain or overwrite the input table.")

    stored_sink = None
    if resume_from is not None:
        manifest = load_manifest(resume_from, expected_backend=job_type.lower())
        stored_sink = manifest.get("result_sink")
        if not isinstance(stored_sink, dict):
            raise ValueError("The resumed run has no table result sink.")
        if os.path.abspath(str(stored_sink.get("path", ""))) != output_path:
            raise ValueError("output_table does not match the resumed run result sink.")

    output_schema = _output_schema(axes, compute_surface, src.ndim)
    dataset = TableDataset.create(
        output_path,
        schema=output_schema,
        schema_version=1,
        operation="regionprops",
        operation_version="1",
        input_identities={
            "segmentation": _source_identity(seg_spec, src),
            "base_table": table_identity,
        },
        parameters={
            "resolution": [float(value) for value in resolution],
            "compute_surface": bool(compute_surface),
            "axes": list(axes),
            "rows_per_batch": rows_per_batch,
        },
    )
    batches = RegularBatchPlan(table_descriptor.row_count, rows_per_batch)
    dataset._bind_batches(batches)
    if stored_sink is not None and stored_sink != dataset._descriptor():
        raise ValueError("The resumed run uses an incompatible table result sink.")

    ctx = {
        "seg": seg_spec,
        "table": table_descriptor,
        "resolution": tuple(float(value) for value in resolution),
        "axes": axes,
        "ndim": src.ndim,
        "compute_surface": bool(compute_surface),
    }
    return runner.map_batches(
        functools.partial(_regionprops_batch, ctx=ctx),
        table_descriptor.row_count,
        batch_size=rows_per_batch,
        num_workers=num_workers,
        name="regionprops",
        pre_cleanup=pre_cleanup,
        resume_from=resume_from,
        result_sink=dataset,
    )


def regionprops(
    input: SourceLike,
    table: Union[str, os.PathLike[str], "pd.DataFrame", TableDataset],
    *,
    resolution: Optional[Sequence[float]] = None,
    compute_surface: bool = False,
    output_path: Optional[str] = None,
    output_table: Optional[Union[str, os.PathLike[str]]] = None,
    rows_per_batch: int = 100_000,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    item_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
    pre_cleanup: Optional[Callable[[str], None]] = None,
) -> Union["pd.DataFrame", TableDataset]:
    """Compute per-object features for a labeled image.

    Set ``output_table`` to use bounded Parquet batches. This path requires a file-backed
    segmentation and a `TableDataset` or Parquet input. It returns a `TableDataset` in input-row
    order. Without ``output_table``, the function returns the existing label-sorted DataFrame.

    Args:
        input: Integer label image. File-backed output requires a reopenable source.
        table: Base morphology table. The DataFrame path accepts DataFrames, CSV, and XLSX.
            The file-backed path accepts `TableDataset` handles and Parquet paths.
        resolution: Physical voxel size in array-axis order. The default is one for each axis.
        compute_surface: Add 3D marching-cubes surface area.
        output_path: Write the DataFrame result to CSV or XLSX.
        output_table: Write a partitioned Parquet dataset to this directory.
        rows_per_batch: Maximum input rows in each file-backed batch.
        num_workers: Local threads or distributed worker concurrency.
        job_type: Execution backend.
        job_config: Runner configuration.
        item_ids: DataFrame-path row indices to process. Do not use with ``output_table``.
        resume_from: Preserved distributed run folder.
        pre_cleanup: Callback that runs before distributed temporary-folder cleanup.

    Returns:
        A DataFrame, or a `TableDataset` when ``output_table`` is set.
    """
    check_rerun_args(job_type, resume_from, item_ids, subset_name="item_ids")
    src = as_source(input)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"regionprops expects an integer label image, got dtype {src.dtype}.")
    ndim = src.ndim

    if resolution is None:
        resolution = tuple(1.0 for _ in range(ndim))
    else:
        resolution = tuple(float(r) for r in resolution)
        if len(resolution) != ndim:
            raise ValueError(f"resolution {resolution} does not match the input ndim {ndim}.")
    if output_table is not None:
        if output_path is not None:
            raise ValueError("output_path and output_table are mutually exclusive.")
        if item_ids is not None:
            raise ValueError("item_ids is not supported with output_table.")
        return _file_backed_regionprops(
            src,
            table,
            resolution=resolution,
            compute_surface=compute_surface,
            output_table=output_table,
            rows_per_batch=rows_per_batch,
            num_workers=num_workers,
            job_type=job_type,
            job_config=job_config,
            resume_from=resume_from,
            pre_cleanup=pre_cleanup,
        )
    if isinstance(table, TableDataset):
        raise ValueError("A TableDataset input requires output_table.")
    return _dataframe_regionprops(
        src,
        table,
        resolution=resolution,
        compute_surface=compute_surface,
        output_path=output_path,
        num_workers=num_workers,
        job_type=job_type,
        job_config=job_config,
        item_ids=item_ids,
        resume_from=resume_from,
        pre_cleanup=pre_cleanup,
    )
