"""Per-label morphology (size, center of mass, bounding box) via the runner's return channel.

This is a reduction operation: each block computes per-label sufficient statistics, and the main
process merges them into one table (one row per label). It mirrors the ``stats`` ops
(:func:`bioimage_py.stats.mean_and_std`) — the per-block statistics flow through
``runner.run(..., has_return_val=True)`` and the merge is pure numpy — so it behaves identically across
the ``local`` / ``subprocess`` / ``slurm`` backends. No custom C++ (nifty) code is required.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, check_rerun_args, to_roi

__all__ = ["morphology"]


def _is_direct(job_type: str, num_workers: int, block_shape: Optional[Tuple[int, ...]]) -> bool:
    """Return whether this call qualifies for direct (non-blocked) computation."""
    return job_type == "local" and num_workers == 1 and block_shape is None


def _check_direct(job_type: str, num_workers: int, block_shape: Optional[Tuple[int, ...]],
                  mask: Optional[SourceLike], block_ids: Optional[Sequence[int]]) -> bool:
    """Like :func:`_is_direct`, but reject masks/block_ids which the direct path can't honor."""
    if _is_direct(job_type, num_workers, block_shape):
        if mask is not None or block_ids is not None:
            raise ValueError("Direct computation does not support 'mask' or 'block_ids'.")
        return True
    return False


def _full_roi(source: Source) -> Tuple[slice, ...]:
    """Return a slicing that selects the whole source."""
    return tuple(slice(None) for _ in range(source.ndim))


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
) -> "pd.DataFrame":
    """Compute per-label morphology (size, center of mass, bounding box) of a labeled volume.

    Statistics are computed block-wise and merged so the result is exact regardless of how labels
    straddle block boundaries. The background label ``0`` is excluded.

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

    Returns:
        A pandas DataFrame with one row per label, sorted by label, with columns ``label``, ``size``,
        ``com_<axis>`` (center of mass), ``bb_min_<axis>`` and ``bb_max_<axis>``. The bounding box is
        slice-ready: ``bb_min`` is inclusive and ``bb_max`` is the exclusive stop, so the object's box
        is ``tuple(slice(bb_min_<axis>, bb_max_<axis>) for axis in axes)``. Axis names are ``z``/``y``/
        ``x`` for 2D/3D data and ``axis0`` ... otherwise.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"morphology expects an integer label image, got dtype {src.dtype}.")
    ndim = src.ndim

    if _check_direct(job_type, num_workers, block_shape, mask, block_ids):
        table = _block_table(src[_full_roi(src)], [0] * ndim)
        tables = [table] if table is not None else []
    else:
        runner = get_runner(job_type, job_config)
        results = runner.run(_morphology_block, [input], num_workers=num_workers,
                             block_shape=block_shape, mask=mask, block_ids=block_ids,
                             resume_from=resume_from, has_return_val=True, name="morphology",
                             pre_cleanup=pre_cleanup)
        tables = [r for r in results if r is not None]

    return _to_dataframe(*_merge_tables(tables, ndim), ndim)
