"""Block-wise (halo-based) distance transform and point-to-object mapping.

``distance_transform`` computes the euclidean distance transform (and/or the feature/index transform)
of 2D/3D data via ``bioimage_cpp.distance.distance_transform``. Like the seeded watershed it is a
single-stage, halo-based operation (so it supports ``block_ids`` / ``resume_from``): each block is
computed over a halo-padded region and the halo-free inner block is written back. The result is exact
only up to the block boundary plus halo, so choose a halo covering the maximum distance of interest;
for a fixed ``(block_shape, halo)`` the output is bit-identical across backends.

``map_points_to_objects`` assigns each point to the nearest object in a segmentation and reports the
distance, by computing the index transform of the background within each (halo-padded) block.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import (BlockDescriptor, ComputeFn, check_rerun_args, full_roi, is_direct,
                    normalize_halo, to_roi)

__all__ = ["distance_transform", "map_points_to_objects"]

Sampling = Optional[Union[float, Sequence[float]]]


def _make_distance_block(sampling: Sampling, return_distances: bool, return_indices: bool,
                         ndim: int) -> ComputeFn:
    """Build the per-block distance-transform function (captures only picklable values)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        outer = to_roi(block.outer_block)
        inner = to_roi(block.inner_block)
        inner_local = to_roi(block.inner_block_local)
        ret = bic.distance.distance_transform(
            inputs[0][outer], sampling=sampling, return_distances=return_distances,
            return_indices=return_indices, number_of_threads=1,
        )
        if return_distances and return_indices:
            dist, ind = ret
        elif return_distances:
            dist, ind = ret, None
        else:
            dist, ind = None, ret

        out_i = 0
        if return_distances:
            outputs[out_i][inner] = dist[inner_local]
            out_i += 1
        if return_indices:
            # The index transform is local to the (outer) block; shift it to global coordinates.
            offset = np.array([int(b) for b in block.outer_block.begin], dtype="int32")
            offset = offset[(slice(None),) + (np.newaxis,) * ndim]
            ind_inner = ind[(slice(None),) + inner_local] + offset
            outputs[out_i][(slice(None),) + inner] = ind_inner
        return None

    return _compute


def _allocate_output(array: Optional[SourceLike], shape: Tuple[int, ...], dtype: str, job_type: str,
                     name: str) -> SourceLike:
    """Return ``array`` if given, else allocate a numpy output for local execution (else raise)."""
    if array is not None:
        return array
    if job_type != "local":
        raise ValueError(
            f"'{name}' is required for distributed execution (job_type={job_type!r}); "
            "pass a file-backed (zarr/n5) output array."
        )
    return np.zeros(shape, dtype=dtype)


def distance_transform(
    input: SourceLike,
    halo: Sequence[int],
    *,
    sampling: Sampling = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Optional[SourceLike] = None,
    indices: Optional[SourceLike] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> Union[SourceLike, Tuple[SourceLike, SourceLike]]:
    """Compute the (halo-based) distance transform of 2D/3D data, block-wise.

    Each block is computed over a halo-padded region via ``bioimage_cpp.distance.distance_transform``
    and the halo-free inner block is written back, so the result is exact only up to the block
    boundary plus ``halo``. Single-stage, so it supports ``block_ids`` / ``resume_from``.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`); 2D or 3D.
        halo: Per-axis halo enlarging each block; **required** for the block-wise path (choose it to
            cover the maximum distance of interest). Ignored by the direct (single-block) path.
        sampling: The per-axis voxel spacing passed to the distance transform (isotropic ``1`` by
            default).
        return_distances: Whether to compute the distance map.
        return_indices: Whether to compute the index (feature) transform, an ``(ndim, *shape)`` array
            holding, per voxel, the global coordinates of the nearest background voxel.
        distances: Output array for the distances (``float32``, shape ``input.shape``). Optional for
            local execution -- allocated and returned if omitted; **required** for distributed
            execution.
        indices: Output array for the indices (``int32``, shape ``(ndim, *input.shape)``). Optional
            for local execution; **required** for distributed execution if ``return_indices``.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
            Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        The distances array if only ``return_distances``; the indices array if only
        ``return_indices``; a ``(distances, indices)`` tuple if both.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    ndim = src.ndim
    if ndim not in (2, 3):
        raise ValueError(f"distance_transform supports 2D or 3D data, got {ndim}D.")
    if not (return_distances or return_indices):
        raise ValueError("At least one of 'return_distances' or 'return_indices' must be True.")

    direct = (is_direct(job_type, num_workers, block_shape)
              and block_ids is None and resume_from is None)

    dist_out = (_allocate_output(distances, tuple(src.shape), "float32", job_type, "distances")
                if return_distances else None)
    idx_out = (_allocate_output(indices, (ndim,) + tuple(src.shape), "int32", job_type, "indices")
               if return_indices else None)
    outputs: List[SourceLike] = []
    if return_distances:
        outputs.append(dist_out)
    if return_indices:
        outputs.append(idx_out)

    if direct:
        ret = bic.distance.distance_transform(
            src[full_roi(ndim)], sampling=sampling, return_distances=return_distances,
            return_indices=return_indices, number_of_threads=1,
        )
        if return_distances and return_indices:
            dist, ind = ret
        elif return_distances:
            dist, ind = ret, None
        else:
            dist, ind = None, ret
        if return_distances:
            as_source(dist_out)[full_roi(ndim)] = dist
        if return_indices:
            as_source(idx_out)[full_roi(ndim + 1)] = ind
    else:
        runner = get_runner(job_type, job_config)
        runner.run(_make_distance_block(sampling, return_distances, return_indices, ndim),
                   [input], outputs=outputs, halo=normalize_halo(halo, ndim), block_shape=block_shape,
                   num_workers=num_workers, block_ids=block_ids, resume_from=resume_from,
                   name="distance_transform")

    if return_distances and return_indices:
        return dist_out, idx_out
    return dist_out if return_distances else idx_out


def _make_map_points(points: np.ndarray, sampling: Sampling, has_halo: bool,
                     seg_dtype: np.dtype) -> ComputeFn:
    """Build the per-block point-to-object mapping function (captures the picklable points)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        seg_src = inputs[0]
        blk = block.outer_block if has_halo else block
        bb_min = np.array([int(b) for b in blk.begin])
        bb_max = np.array([int(e) for e in blk.end])
        ndim = bb_min.shape[0]

        in_block = np.logical_and.reduce(
            [points[:, i] >= bb_min[i] for i in range(ndim)]
            + [points[:, i] < bb_max[i] for i in range(ndim)]
        )
        if not in_block.any():
            return None
        point_ids = np.where(in_block)[0]
        block_points = (points[in_block] - bb_min[None, :]).astype("int64")

        block_seg = seg_src[tuple(slice(int(b), int(e)) for b, e in zip(bb_min, bb_max))]
        if not block_seg.any():  # no objects in the block -> background id 0 at infinite distance.
            return (point_ids, np.zeros(len(point_ids), dtype=seg_dtype),
                    np.full(len(point_ids), np.inf, dtype="float32"))

        distances, indices = bic.distance.distance_transform(
            block_seg == 0, sampling=sampling, return_distances=True, return_indices=True,
            number_of_threads=1,
        )
        coords = tuple(block_points[:, i] for i in range(ndim))
        object_distances = distances[coords].astype("float32")
        nearest = tuple(indices[i][coords] for i in range(ndim))
        object_ids = block_seg[nearest]
        return point_ids, object_ids, object_distances

    return _compute


def map_points_to_objects(
    segmentation: SourceLike,
    points: np.ndarray,
    block_shape: Tuple[int, ...],
    *,
    halo: Optional[Sequence[int]] = None,
    sampling: Sampling = None,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map point coordinates to the nearest object in a segmentation and measure the distance.

    Each (halo-padded) block computes the index transform of its background and maps the points it
    contains to the nearest object. Choose ``halo`` to cover the maximum distance of interest; points
    near a block boundary are resolved to the assignment with the smallest distance across overlapping
    blocks.

    Args:
        segmentation: The label image (a numpy/zarr/n5 array or a `Source`).
        points: The integer point coordinates, an ``(n_points, ndim)`` array.
        block_shape: Shape of the processing blocks.
        halo: Per-axis halo enlarging each block; choose it large enough to cover the maximum
            distance of interest. ``None`` uses non-overlapping blocks (distances are then only
            correct within each block).
        sampling: The per-axis voxel spacing passed to the distance transform.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks).
            Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``). Mutually exclusive with ``block_ids``.

    Returns:
        A ``(object_ids, object_distances)`` tuple, each of length ``n_points``: the id of the nearest
        object per point (``0`` if none was found) and the corresponding distance (``inf`` if none).
    """
    check_rerun_args(job_type, resume_from, block_ids)
    seg_src = as_source(segmentation)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != seg_src.ndim:
        raise ValueError(
            f"points must be an (n_points, {seg_src.ndim}) array, got shape {points.shape}."
        )
    n_points = len(points)

    has_halo = halo is not None
    runner = get_runner(job_type, job_config)
    results = runner.run(_make_map_points(points, sampling, has_halo, np.dtype(seg_src.dtype)),
                         [segmentation], block_shape=block_shape,
                         halo=normalize_halo(halo, seg_src.ndim) if has_halo else None,
                         num_workers=num_workers, block_ids=block_ids, resume_from=resume_from,
                         has_return_val=True, name="map_points_to_objects")

    object_ids = np.zeros(n_points, dtype=seg_src.dtype)
    object_distances = np.full(n_points, np.inf, dtype="float32")
    for res in results:
        if res is None:
            continue
        pids, oids, dists = res
        take = dists < object_distances[pids]  # overlapping blocks: keep the closest assignment.
        object_ids[pids[take]] = oids[take]
        object_distances[pids[take]] = dists[take]
    return object_ids, object_distances
