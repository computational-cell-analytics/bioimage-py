"""Advanced per-object morphology: numpy moment features + surface area and centroid correction.

A *second pass* on top of :func:`bioimage_py.morphology.morphology`: for each labeled object it crops
the sub-volume by the precomputed bounding box, masks ``== label``, and computes shape features
directly with numpy (in physical units via ``resolution``) — the physical volume (``area``), the
``extent``, the ``equivalent_diameter_area`` and the major/minor ``axis_*_length`` from the object's
second moments — plus an optional marching-cubes ``surface_area`` (3D) and a corrected centroid (the
center-of-mass if it lies inside the object, else the deepest-interior voxel via the Euclidean
distance transform).

The moment features reproduce the corresponding scikit-image ``regionprops`` definitions exactly, but
without constructing a ``RegionProperties`` object per label (and without the expensive convex-hull
``solidity`` / topological ``euler_number``), so the pass scales linearly in the number of objects.
Work is mapped one task per object with the generic
:meth:`bioimage_py.runner.base.Runner.map`, so it runs identically across ``local`` / ``subprocess``
/ ``slurm``; for distributed backends the base table is serialized to a temp file the workers read.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import bioimage_cpp as bic
import numpy as np
import pandas as pd

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import check_rerun_args
from .morphology import _axis_names

__all__ = ["regionprops"]

# Per-worker cache so each task reopens the segmentation `SourceSpec` once, not per object (keyed by
# the spec's stable fields). The consumed table columns travel in the closure as numpy arrays, so
# they need no worker-side caching.
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


def _resolve_seg(seg: Union[Source, Any]) -> Source:
    """Return an opened segmentation source, reopening (and caching) a `SourceSpec` if needed."""
    if isinstance(seg, Source):
        return seg
    key = (seg.kind, seg.path, seg.internal_path)
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


def _object_features(index: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the feature row for a single object (the per-item function handed to ``runner.map``)."""
    seg = _resolve_seg(ctx["seg"])
    axes, ndim = ctx["axes"], ctx["ndim"]
    res = np.asarray(ctx["resolution"], dtype="float64")

    label = int(ctx["label"][index])
    com = ctx["com"][index]
    bb_min = ctx["bb_min"][index]
    bb_max = ctx["bb_max"][index]

    crop = np.asarray(seg[tuple(slice(int(lo), int(hi)) for lo, hi in zip(bb_min, bb_max))])
    mask = crop == label
    coords = np.argwhere(mask)  # crop-local voxel coordinates of the object.
    n_voxels = int(coords.shape[0])
    area = n_voxels * float(np.prod(res))  # physical volume under spacing (== skimage's 'area').

    bbox_voxels = int(np.prod(bb_max - bb_min))
    major, minor = _axis_lengths(coords, res, ndim)
    result: Dict[str, Any] = {
        "label": label,
        "n_voxels": n_voxels,
        "area": area,
        "extent": (n_voxels / bbox_voxels) if bbox_voxels > 0 else float("nan"),
        "equivalent_diameter_area": ((2 * ndim * area / np.pi) ** (1.0 / ndim)) if area > 0 else 0.0,
        "axis_major_length": major,
        "axis_minor_length": minor,
    }

    if ctx["compute_surface"] and ndim == 3:
        result["surface_area"] = _surface_area(mask, ctx["resolution"])

    centroid = _corrected_centroid(mask, com, bb_min.astype("float64"), ctx["resolution"])
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


def regionprops(
    input: SourceLike,
    table: Union[str, "pd.DataFrame"],
    *,
    resolution: Optional[Sequence[float]] = None,
    compute_surface: bool = False,
    output_path: Optional[str] = None,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    item_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
    pre_cleanup: Optional[Callable[[str], None]] = None,
) -> "pd.DataFrame":
    """Compute per-object morphology features for a labeled volume, one task per object.

    For each object listed in ``table`` (the output of :func:`morphology`), the sub-volume is cropped by
    its bounding box, masked to the label, and described with numpy in physical units (via
    ``resolution``): the physical volume ``area``, the ``extent`` (filled fraction of the bounding box),
    the ``equivalent_diameter_area`` (diameter of the ball with the same volume) and the major/minor
    ``axis_*_length`` (from the object's second moments). These reproduce the corresponding
    scikit-image ``regionprops`` definitions exactly. Optionally a marching-cubes ``surface_area`` (3D
    only) and a corrected centroid (the center-of-mass when it lies inside the object, otherwise the
    deepest-interior voxel — the argmax of the Euclidean distance transform) are added.

    Args:
        input: The labeled segmentation (a numpy/zarr/n5 array or a `Source`); integer-typed. For the
            ``subprocess``/``slurm`` backends it must be file-backed (zarr/n5).
        table: The base morphology table — a pandas DataFrame or a path to a ``.csv`` / ``.xlsx`` file.
            Must contain ``label``, ``com_<axis>``, ``bb_min_<axis>`` and ``bb_max_<axis>`` (``bb_max``
            is the exclusive slice stop, as produced by :func:`morphology`).
        resolution: Per-axis physical voxel size in array (e.g. z, y, x) order. Defaults to ones (voxel
            units).
        compute_surface: Whether to add a marching-cubes ``surface_area`` (3D inputs only). This is the
            most expensive per-object step, so it is off by default.
        output_path: Optional ``.csv`` / ``.xlsx`` path to also write the result to.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed backends).
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        item_ids: Restrict processing to these item indices (rows of ``table``), e.g. to re-run
            previously failed objects; the result then covers only those rows. An item id indexes a
            *row* of ``table``, so the same table (same row order) must be passed as in the original
            run. Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and merge
            (see ``runner.run``). The result then covers all objects (the already-completed merged
            with the re-run ones). The recommended rerun path for regionprops. Mutually exclusive
            with ``item_ids``.
        pre_cleanup: Optional ``pre_cleanup(tmp_folder)`` callback invoked on the orchestrating
            process with the job temp folder right before it is deleted (distributed backends only).
            Use it to read out the per-task timing files under ``tmp_folder/timings/`` before cleanup.
            Ignored for the ``local`` backend (no temp folder).

    Returns:
        A pandas DataFrame with one row per object, sorted by ``label``: ``label``, ``n_voxels`` (raw
        voxel count), ``area`` (physical volume), ``extent``, ``equivalent_diameter_area``,
        ``axis_major_length``, ``axis_minor_length``, ``centroid_<axis>`` (corrected, physical),
        ``bb_min_<axis>``/``bb_max_<axis>`` (global voxels), and ``surface_area`` (only when
        ``compute_surface`` and the input is 3D).
    """
    check_rerun_args(job_type, resume_from, item_ids, subset_name="item_ids")
    src = as_source(input)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"regionprops expects an integer label image, got dtype {src.dtype}.")
    ndim = src.ndim
    axes = _axis_names(ndim)

    if resolution is None:
        resolution = tuple(1.0 for _ in range(ndim))
    else:
        resolution = tuple(float(r) for r in resolution)
        if len(resolution) != ndim:
            raise ValueError(f"resolution {resolution} does not match the input ndim {ndim}.")

    df = _load_table(table)
    missing = [c for c in _required_columns(axes) if c not in df.columns]
    if missing:
        raise ValueError(
            f"table is missing required columns {missing}; pass the output of "
            "bioimage_py.morphology.morphology (label / com_* / bb_min_* / bb_max_*)."
        )

    n = len(df)
    if n == 0:
        cols = _order_columns(axes, compute_surface, ndim)
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})

    seg_arg: Any = src
    if job_type != "local":
        try:
            seg_arg = src.to_spec()
        except ValueError as err:
            raise ValueError(
                f"Distributed regionprops requires a file-backed (zarr/n5) segmentation. {err}"
            ) from err

    # The consumed columns travel with the closure as numpy arrays (built once): for distributed
    # backends they are cloudpickled into the single shared payload the workers read.
    ctx = {
        "seg": seg_arg, "resolution": resolution, "axes": tuple(axes), "ndim": ndim,
        "compute_surface": bool(compute_surface), **_column_arrays(df, axes),
    }
    runner = get_runner(job_type, job_config)
    results = runner.map(functools.partial(_object_features, ctx=ctx), n,
                         item_ids=item_ids, resume_from=resume_from,
                         num_workers=num_workers, has_return_val=True, name="regionprops",
                         pre_cleanup=pre_cleanup)

    out = pd.DataFrame(results)
    out = out[_order_columns(axes, compute_surface, ndim)].sort_values("label")
    out = out.reset_index(drop=True)
    if output_path is not None:
        _write_table(out, output_path)
    return out
