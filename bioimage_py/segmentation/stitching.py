"""Stitch a tile-wise over-segmentation into a globally consistent one (multicut over tile overlaps).

When a large image is segmented tile by tile, the same physical object gets a different id in each
tile it spans. The functions here reconcile the tiles by measuring how strongly the tile-local objects
overlap where two tiles describe the same (or abutting) voxels, and by solving a multicut (see
:mod:`bioimage_py.segmentation.multicut`) over the resulting *instance-correspondence graph* -- one
node per tile-local object, one edge per measured correspondence -- to decide which objects are one
and the same.

- `stitch_segmentation` runs a user segmentation function tile-wise (each tile read with a halo) and
  stitches the results. The halo region -- segmented by a tile but not written into the result -- is
  where the two neighbouring predictions of the *same* voxels are compared.
- `stitch_block_segmentations` stitches haloed tile segmentations that were produced externally and
  written into a *block store* (one slot per tile, see `block_store_shape`), e.g. by a pool of
  persistent GPU workers. It is the second half of `stitch_segmentation`.
- `stitch_tiled_segmentation` stitches an already-tiled segmentation (unique ids per tile, tiles
  non-overlapping) by measuring the overlap of objects *touching* across tile interfaces.

Compared to elf's stitcher, correspondences are used directly as graph edges (two objects that agree
in the halo are merged even if their written cores do not touch at the tile boundary), the overlap
confidence is symmetric (``overlap_metric``) and support-aware (``min_overlap``), and *competing*
correspondences -- two objects of one tile that both match the same object of the neighbouring tile
-- are represented by soft repulsive edges (``competition_disaffinity``).

The per-voxel phases (tile segmentation, core assembly, overlap counting, the final relabeling) run
through the `bioimage_py` runner and scale across the ``local`` / ``subprocess`` / ``slurm`` backends.
Only the compact correspondence graph -- O(number of tile-local objects), not O(voxels) -- is built and
solved in the orchestrating process; the solved node labeling is written back with the canonical
node-label writer :func:`bioimage_py.segmentation.relabel`.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import bioimage_cpp as bic
import numpy as np

from ..runner import RunnerError, get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source, from_spec
from ..stats.unique import unique
from ..util import BlockDescriptor, ComputeFn, full_roi, get_blocking, is_inmemory_numpy, to_roi
from .multicut import compute_edge_costs, multicut_kernighan_lin
from .relabel import relabel

__all__ = ["block_store_shape", "stitch_block_segmentations", "stitch_segmentation",
           "stitch_tiled_segmentation"]

#: The supported overlap metrics (see the ``overlap_metric`` argument of `stitch_segmentation`).
OVERLAP_METRICS = ("geometric", "iou", "dice", "min", "max")

# Column layout of the per-face correspondence rows returned by the overlap stages (all uint64):
# the two (offset) ids, their overlap count, their sizes within the face, the face axis and the two
# tile ids the labels belong to.
_LABEL_A, _LABEL_B, _COUNT, _SIZE_A, _SIZE_B, _AXIS, _BLOCK_A, _BLOCK_B = range(8)
_N_COLUMNS = 8


# ---------------------------------------------------------------------------------------------
# Id offsets and validation of tile-local label images.
# ---------------------------------------------------------------------------------------------

def _block_offset(block_id: int, offset_factor: int, with_background: bool) -> int:
    """The id offset of a tile: ``block_id * offset_factor`` (plus 1 without background, so no id is 0)."""
    return int(block_id) * int(offset_factor) + (0 if with_background else 1)


def _apply_offset(labels: np.ndarray, offset: int, with_background: bool) -> np.ndarray:
    """Return ``labels`` as uint64 shifted by ``offset``; with background only non-zero ids are shifted."""
    labels = labels.astype("uint64", copy=True)
    if offset == 0:
        return labels
    if with_background:
        labels[labels != 0] += np.uint64(offset)
    else:
        labels += np.uint64(offset)
    return labels


def _validate_tile_labels(labels: np.ndarray, expected_shape: Tuple[int, ...], offset_factor: int,
                          what: str) -> None:
    """Validate a tile-local label image: shape, integer dtype and the id range ``[0, offset_factor)``."""
    if tuple(labels.shape) != tuple(expected_shape):
        raise ValueError(f"{what} has shape {tuple(labels.shape)}, expected the haloed tile shape "
                         f"{tuple(expected_shape)}.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{what} must be an integer label image, got dtype {labels.dtype}.")
    if labels.size:
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= offset_factor:
            raise ValueError(f"{what} has ids in [{lo}, {hi}]; tile-local ids must be non-negative and "
                             f"smaller than prod(haloed tile shape) = {offset_factor}.")


# ---------------------------------------------------------------------------------------------
# Overlap counting (shared) + per-block compute functions.
# ---------------------------------------------------------------------------------------------

def _group_sums(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per-element sum of ``values`` over all elements sharing the element's key."""
    _, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros(int(inverse.max()) + 1, dtype="uint64")
    np.add.at(sums, inverse, values)
    return sums[inverse]


def _face_overlap_rows(face_a: np.ndarray, face_b: np.ndarray, axis: int, block_a: int, block_b: int,
                       with_background: bool) -> Optional[np.ndarray]:
    """Return the ``(K, 8)`` uint64 correspondence rows of one face (see the column layout above).

    The sizes are the label sizes within the face: both faces cover the same voxels, so a label's size
    is the sum of its overlap counts over *all* partners (background included, hence computed before
    the background rows are dropped). Returns ``None`` when there is no (foreground) overlap.
    """
    face_a = np.ascontiguousarray(face_a, dtype="uint64")
    face_b = np.ascontiguousarray(face_b, dtype="uint64")
    table = bic.utils.segmentation_overlap(face_a, face_b).overlap_table()
    if table.shape[0] == 0:
        return None
    label_a = np.asarray(table["label_a"], dtype="uint64")
    label_b = np.asarray(table["label_b"], dtype="uint64")
    count = np.asarray(table["count"], dtype="uint64")
    size_a = _group_sums(label_a, count)
    size_b = _group_sums(label_b, count)
    if with_background:
        keep = (label_a != 0) & (label_b != 0)
        if not keep.any():
            return None
        label_a, label_b, count, size_a, size_b = (x[keep] for x in (label_a, label_b, count, size_a, size_b))
    rows = np.empty((len(label_a), _N_COLUMNS), dtype="uint64")
    rows[:, _LABEL_A] = label_a
    rows[:, _LABEL_B] = label_b
    rows[:, _COUNT] = count
    rows[:, _SIZE_A] = size_a
    rows[:, _SIZE_B] = size_b
    rows[:, _AXIS] = axis
    rows[:, _BLOCK_A] = block_a
    rows[:, _BLOCK_B] = block_b
    return rows


def _make_tiled_overlap(shape: Tuple[int, ...], tile_shape: Tuple[int, ...], overlap: int, ndim: int,
                        with_background: bool) -> ComputeFn:
    """Build the overlap compute fn for `stitch_tiled_segmentation` (abutting slabs of the input)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        seg = inputs[0]
        blocking = get_blocking(shape, tile_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.begin])
        rows: List[np.ndarray] = []
        for axis in range(ndim):
            ngb_id = blocking.get_neighbor_id(block_id, axis, True)
            if ngb_id == -1:
                continue
            beg = [int(b) for b in block.begin]
            end = [int(e) for e in block.end]
            # This tile's first `overlap` slab along the axis (clipped to the tile) and the lower
            # neighbour's abutting slab of the same width just below it; the full tile on the other axes.
            width = min(int(overlap), end[axis] - beg[axis])
            this_roi = tuple(slice(beg[d], beg[d] + width) if d == axis else slice(beg[d], end[d])
                             for d in range(ndim))
            ngb_roi = tuple(slice(beg[d] - width, beg[d]) if d == axis else slice(beg[d], end[d])
                            for d in range(ndim))
            r = _face_overlap_rows(seg[this_roi], seg[ngb_roi], axis, block_id, ngb_id, with_background)
            if r is not None:
                rows.append(r)
        return np.concatenate(rows, axis=0) if rows else None

    return _compute


def _make_store_overlap(shape: Tuple[int, ...], tile_shape: Tuple[int, ...], tile_overlap: Tuple[int, ...],
                        offset_factor: int, with_background: bool, store_handle, ndim: int) -> ComputeFn:
    """Build the overlap compute fn over haloed tiles (reads the shared halo faces from the block store)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        store = _resolve_source(store_handle)
        blocking = get_blocking(shape, tile_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.begin])
        halo = [int(h) for h in tile_overlap]
        rows: List[np.ndarray] = []
        for axis in range(ndim):
            ngb_id = blocking.get_neighbor_id(block_id, axis, True)
            if ngb_id == -1:
                continue
            # The region both haloed tiles segmented, in each tile's local (store slot) coordinates.
            overlaps = blocking.get_local_overlaps(block_id, ngb_id, halo)
            if overlaps is None:  # no shared region: zero overlap along this axis
                continue
            begin_a, end_a, begin_b, end_b = overlaps
            face_a = store[(block_id,) + tuple(slice(int(b), int(e)) for b, e in zip(begin_a, end_a))]
            face_b = store[(ngb_id,) + tuple(slice(int(b), int(e)) for b, e in zip(begin_b, end_b))]
            if face_a.shape != face_b.shape:
                raise RuntimeError(f"Internal error: mismatching face shapes {face_a.shape} / {face_b.shape} "
                                   f"for tiles {block_id} / {ngb_id}.")
            r = _face_overlap_rows(face_a, face_b, axis, block_id, ngb_id, with_background)
            if r is None:
                continue
            r[:, _LABEL_A] = _apply_offset(r[:, _LABEL_A], _block_offset(block_id, offset_factor, with_background),
                                           with_background)
            r[:, _LABEL_B] = _apply_offset(r[:, _LABEL_B], _block_offset(ngb_id, offset_factor, with_background),
                                           with_background)
            rows.append(r)
        return np.concatenate(rows, axis=0) if rows else None

    return _compute


def _make_segment(shape: Tuple[int, ...], tile_shape: Tuple[int, ...], seg_func: Callable, offset_factor: int,
                  store_handle, input_handle, ndim: int, in_ndim: int) -> ComputeFn:
    """Build the tile-segmentation stage: segment each haloed tile and write it into its store slot."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        input_ = _resolve_source(input_handle)
        store = _resolve_source(store_handle)
        blocking = get_blocking(shape, tile_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.inner_block.begin])

        outer_roi = to_roi(block.outer_block)
        if in_ndim > ndim:  # the input carries trailing channel axes beyond the spatial shape
            outer_roi = outer_roi + tuple(slice(None) for _ in range(in_ndim - ndim))
        tile_seg = np.asarray(seg_func(input_[outer_roi], block_id))
        _validate_tile_labels(tile_seg, tuple(int(s) for s in block.outer_block.shape), offset_factor,
                              f"The segmentation of tile {block_id}")
        # Persist the haloed tile (tile-local ids) at the leading corner of its store slot.
        store[(block_id,) + tuple(slice(0, s) for s in tile_seg.shape)] = tile_seg.astype(store.dtype, copy=False)
        return None

    return _compute


def _make_assemble(shape: Tuple[int, ...], tile_shape: Tuple[int, ...], offset_factor: int,
                   with_background: bool, store_handle) -> ComputeFn:
    """Build the core-assembly stage: offset a tile's ids, write its core, return the core's ids."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        store = _resolve_source(store_handle)
        output_ = outputs[0]
        blocking = get_blocking(shape, tile_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.inner_block.begin])

        outer_shape = tuple(int(s) for s in block.outer_block.shape)
        tile_seg = np.asarray(store[(block_id,) + tuple(slice(0, s) for s in outer_shape)])
        _validate_tile_labels(tile_seg, outer_shape, offset_factor, f"block_store slot {block_id}")
        tile_seg = _apply_offset(tile_seg, _block_offset(block_id, offset_factor, with_background), with_background)

        core = tile_seg[to_roi(block.inner_block_local)]
        output_[to_roi(block.inner_block)] = core
        # The ids that contribute to the output (i.e. the graph nodes); halo-only ids are not nodes.
        ids = np.unique(core[core != 0]) if with_background else np.unique(core)
        return ids if ids.size else None

    return _compute


# ---------------------------------------------------------------------------------------------
# Orchestration helpers.
# ---------------------------------------------------------------------------------------------

def _resolve_source(handle) -> Source:
    """Resolve a closure-captured handle to a live `Source` (already-live for local, spec otherwise)."""
    return handle if isinstance(handle, Source) else from_spec(handle)


def _capture(source: Source, job_type: str):
    """Capture a source for a per-block closure: the live object for local, its spec otherwise.

    For distributed backends ``to_spec()`` also validates the source is reopenable (raising a clear
    error for in-memory numpy arrays).
    """
    return source if job_type == "local" else source.to_spec()


def _prepare_output(output: Optional[SourceLike], shape: Tuple[int, ...], job_type: str) -> SourceLike:
    """Resolve the output array: allocate a numpy array for local, require a file-backed one otherwise."""
    if output is not None:
        return output
    if job_type != "local":
        raise ValueError(
            f"'output' is required for distributed execution (job_type={job_type!r}); "
            "pass a file-backed (zarr/n5) output array."
        )
    return np.zeros(tuple(shape), dtype="uint64")


def _check_output(out_array: SourceLike, shape: Tuple[int, ...]) -> Source:
    """Validate the (uint64) output array of the haloed stitchers against the spatial shape."""
    out = as_source(out_array)
    if tuple(int(s) for s in out.shape) != tuple(shape):
        raise ValueError(f"output has shape {tuple(out.shape)}, expected {tuple(shape)}.")
    if out.dtype != np.dtype("uint64"):
        raise ValueError(f"output must have dtype uint64, got {out.dtype}.")
    return out


def _check_tiling(shape: Tuple[int, ...], tile_shape: Sequence[int],
                  tile_overlap: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Normalize and validate ``tile_shape`` / ``tile_overlap`` against the spatial shape."""
    ndim = len(shape)
    tile_shape = tuple(int(t) for t in tile_shape)
    tile_overlap = tuple(int(t) for t in tile_overlap)
    if len(tile_shape) != ndim or len(tile_overlap) != ndim:
        raise ValueError(f"tile_shape {tile_shape} and tile_overlap {tile_overlap} must have one entry per "
                         f"spatial axis ({ndim}).")
    if any(t <= 0 for t in tile_shape):
        raise ValueError(f"tile_shape must be positive, got {tile_shape}.")
    if any(o < 0 for o in tile_overlap):
        raise ValueError(f"tile_overlap must be non-negative, got {tile_overlap}.")
    if any(o == 0 for o in tile_overlap):
        warnings.warn(
            f"tile_overlap {tile_overlap} is 0 along some axis: no overlap evidence is measured across the "
            "tile seams of that axis, so objects are never merged across them.", stacklevel=3,
        )
    return tile_shape, tile_overlap


def _check_stitching_args(beta: float, overlap_metric: str, min_overlap: int,
                          competition_disaffinity: Optional[float]) -> None:
    """Validate the stitching hyper-parameters shared by the public functions."""
    if not 0.0 < float(beta) < 1.0:
        raise ValueError(f"beta must be in the exclusive range (0, 1), got {beta}.")
    if overlap_metric not in OVERLAP_METRICS:
        raise ValueError(f"overlap_metric must be one of {OVERLAP_METRICS}, got {overlap_metric!r}.")
    if int(min_overlap) < 1:
        raise ValueError(f"min_overlap must be at least 1, got {min_overlap}.")
    if competition_disaffinity is not None and not 0.0 < float(competition_disaffinity) < 1.0:
        raise ValueError("competition_disaffinity must be None or in the exclusive range (0, 1), got "
                         f"{competition_disaffinity}.")


def _offset_factor(tile_shape: Tuple[int, ...], tile_overlap: Tuple[int, ...], n_blocks: int) -> int:
    """The per-tile id offset factor, ``prod(haloed tile shape)``, checked against uint64 overflow."""
    factor = int(np.prod([int(ts) + 2 * int(ov) for ts, ov in zip(tile_shape, tile_overlap)]))
    if n_blocks * factor >= int(np.iinfo(np.uint64).max):
        raise ValueError("Label id overflow: number_of_blocks * prod(haloed tile shape) exceeds uint64. "
                         "Reduce the tile shape or the volume size.")
    return factor


def _make_block_store(store_shape: Tuple[int, ...], dtype: str, in_memory: bool, job_type: str,
                      job_config: Optional[RunnerConfig]) -> Tuple[Source, Callable[[], None], object, Optional[str]]:
    """Create the block store of `stitch_segmentation`: ``(store, cleanup, capture handle, path)``.

    Slot ``block_id`` is one chunk, so concurrent per-tile writes never touch the same chunk. A numpy
    array backs the in-memory store (local execution into an in-memory output); a temporary zarr array
    (under ``job_config.tmp_root`` when set) backs it otherwise, so distributed workers can reopen it.
    """
    if in_memory:
        store = as_source(np.zeros(store_shape, dtype=dtype))
        return store, (lambda: None), store, None

    import zarr

    tmp_root = job_config.tmp_root if job_config is not None else None
    if job_type == "slurm" and not tmp_root:
        raise ValueError("Distributed stitching on slurm requires job_config.tmp_root on a shared "
                         "filesystem (it holds the temporary per-tile block store).")
    tmp_dir = tempfile.mkdtemp(prefix="bioimage_py_stitch_", dir=tmp_root)
    path = os.path.join(tmp_dir, "blocks.zarr")
    z = zarr.open_array(path, mode="w", shape=store_shape, chunks=(1,) + tuple(store_shape[1:]), dtype=dtype)
    store = as_source(z)

    def _cleanup() -> None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return store, _cleanup, _capture(store, job_type), path


def _merge_ids(results: Optional[list]) -> np.ndarray:
    """Merge the per-block core-id arrays into the sorted, unique node id array (uint64)."""
    arrays = [a for a in (results or []) if a is not None and len(a)]
    if not arrays:
        return np.zeros((0,), dtype="uint64")
    return np.unique(np.concatenate(arrays)).astype("uint64", copy=False)


def _collect_rows(results: Optional[list]) -> np.ndarray:
    """Concatenate the per-block correspondence rows into one ``(K, 8)`` uint64 array."""
    rows = [r for r in (results or []) if r is not None and len(r)]
    if not rows:
        return np.zeros((0, _N_COLUMNS), dtype="uint64")
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------------------------
# The instance-correspondence graph: construction, solve, projection.
# ---------------------------------------------------------------------------------------------

def _overlap_probability(count: np.ndarray, size_a: np.ndarray, size_b: np.ndarray, metric: str) -> np.ndarray:
    """The merge probability of a correspondence from its overlap count and the two label sizes (float64)."""
    if metric == "geometric":
        p = count / np.sqrt(size_a * size_b)
    elif metric == "iou":
        p = count / (size_a + size_b - count)
    elif metric == "dice":
        p = 2.0 * count / (size_a + size_b)
    elif metric == "min":
        p = np.minimum(count / size_a, count / size_b)
    else:  # "max": the overlap coefficient
        p = np.maximum(count / size_a, count / size_b)
    return np.clip(p, 0.0, 1.0)


def _map_to_nodes(node_ids: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map label ids to node indices via the sorted ``node_ids``; returns ``(indices, valid mask)``."""
    if len(node_ids) == 0:
        return np.zeros(len(ids), dtype="int64"), np.zeros(len(ids), dtype=bool)
    idx = np.minimum(np.searchsorted(node_ids, ids), len(node_ids) - 1)
    return idx, node_ids[idx] == ids


def _competition_pairs(uv: np.ndarray, block_a: np.ndarray, block_b: np.ndarray,
                       attractive: np.ndarray) -> np.ndarray:
    """Pairs of same-tile nodes that compete for one node of the neighbouring tile via attractive edges.

    For every node ``u`` with attractive edges to two or more nodes of the *same* neighbouring tile, all
    pairs among those competitors are returned (canonical ``u < v``, lexicographically sorted, unique).
    Competitors are same-tile objects, so these pairs are disjoint from the (cross-tile)
    correspondence edges.
    """
    empty = np.zeros((0, 2), dtype="uint64")
    uv, block_a, block_b = uv[attractive], block_a[attractive], block_b[attractive]
    if len(uv) < 2:
        return empty
    # Both directions: for the anchor u the competitors are the v's in tile block_b, and vice versa.
    anchor = np.concatenate([uv[:, 0], uv[:, 1]])
    tile = np.concatenate([block_b, block_a])
    competitor = np.concatenate([uv[:, 1], uv[:, 0]])
    order = np.lexsort((competitor, tile, anchor))
    anchor, tile, competitor = anchor[order], tile[order], competitor[order]
    new_group = np.concatenate(([True], (anchor[1:] != anchor[:-1]) | (tile[1:] != tile[:-1])))
    group = np.cumsum(new_group) - 1
    sizes = np.bincount(group)
    if sizes.max() < 2:
        return empty
    # Pair every element with the k-th next element of its group, for all lags k (vectorized per lag).
    pairs: List[np.ndarray] = []
    for k in range(1, int(sizes.max())):
        same = group[k:] == group[:-k]
        if same.any():
            pairs.append(np.stack([competitor[:-k][same], competitor[k:][same]], axis=1))
    pairs = np.concatenate(pairs, axis=0)
    pairs = np.stack([np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])], axis=1)
    return np.unique(pairs, axis=0).astype("uint64", copy=False)


def _build_correspondence_graph(rows: np.ndarray, node_ids: np.ndarray, overlap_metric: str, min_overlap: int,
                                beta: float, competition_disaffinity: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Build the edges and costs of the instance-correspondence graph from the correspondence rows.

    Returns ``(uv, costs)``: the unique edges as sorted ``(E, 2)`` uint64 node-index pairs with ``u < v``
    (valid input for ``UndirectedGraph.from_unique_edges``) and their float64 multicut costs (positive =
    attractive). Rows whose ids are not nodes (halo-only objects) or whose overlap is below
    ``min_overlap`` are dropped; the reduction is independent of the block / row order.
    """
    empty = (np.zeros((0, 2), dtype="uint64"), np.zeros((0,), dtype="float64"))
    if len(rows) == 0 or len(node_ids) == 0:
        return empty
    u, ok_u = _map_to_nodes(node_ids, rows[:, _LABEL_A])
    v, ok_v = _map_to_nodes(node_ids, rows[:, _LABEL_B])
    keep = ok_u & ok_v & (u != v) & (rows[:, _COUNT] >= np.uint64(int(min_overlap)))
    if not keep.any():
        return empty
    rows, u, v = rows[keep], u[keep], v[keep]

    # Canonicalise u < v (swapping the per-side statistics along), sort, and merge duplicate pairs. The
    # statistics are float64 from here on: uint64 products would overflow silently.
    swap = u > v
    uu = np.where(swap, v, u).astype("uint64")
    vv = np.where(swap, u, v).astype("uint64")
    count = rows[:, _COUNT].astype("float64")
    size_a = np.where(swap, rows[:, _SIZE_B], rows[:, _SIZE_A]).astype("float64")
    size_b = np.where(swap, rows[:, _SIZE_A], rows[:, _SIZE_B]).astype("float64")
    block_a = np.where(swap, rows[:, _BLOCK_B], rows[:, _BLOCK_A])
    block_b = np.where(swap, rows[:, _BLOCK_A], rows[:, _BLOCK_B])
    order = np.lexsort((vv, uu))
    uu, vv, count, size_a, size_b, block_a, block_b = (
        x[order] for x in (uu, vv, count, size_a, size_b, block_a, block_b)
    )
    starts = np.flatnonzero(np.concatenate(([True], (uu[1:] != uu[:-1]) | (vv[1:] != vv[:-1]))))
    uv = np.stack([uu[starts], vv[starts]], axis=1)
    count, size_a, size_b = (np.add.reduceat(x, starts) for x in (count, size_a, size_b))
    block_a, block_b = block_a[starts], block_b[starts]

    p_merge = _overlap_probability(count, size_a, size_b, overlap_metric)
    costs = np.asarray(compute_edge_costs(1.0 - p_merge, beta=beta), dtype="float64")

    if competition_disaffinity is not None:
        competition = _competition_pairs(uv, block_a, block_b, costs > 0)
        if len(competition):
            competition_costs = np.asarray(
                compute_edge_costs(np.full(len(competition), float(competition_disaffinity), dtype="float64"),
                                   beta=beta), dtype="float64")
            uv = np.concatenate([uv, competition], axis=0)
            costs = np.concatenate([costs, competition_costs])
            order = np.lexsort((uv[:, 1], uv[:, 0]))
            uv, costs = uv[order], costs[order]
    return uv, costs


def _solve_correspondence_graph(n_nodes: int, uv: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Solve the multicut on the correspondence graph; returns dense ``uint64`` node labels.

    This is the seam for a distributed solve: everything before it produces ``(uv, costs)``, everything
    after it only consumes the node labeling.
    """
    if n_nodes == 0:
        return np.zeros((0,), dtype="uint64")
    if len(uv) == 0:
        return np.arange(n_nodes, dtype="uint64")
    graph = bic.graph.UndirectedGraph.from_unique_edges(int(n_nodes), np.ascontiguousarray(uv, dtype="uint64"))
    graph.freeze()

    # Components held together by attractive edges only are merged as a whole: cutting any of their
    # edges only adds cost. (A zero-cost edge inside such a component is a tie; it is merged.) Only the
    # components containing a non-attractive edge between two of their own nodes need a solver; a
    # repulsive edge between two different components is cut for free and needs no decision.
    attractive = costs > 0
    components = np.asarray(bic.graph.connected_components(graph, edge_mask=attractive), dtype="uint64")
    internal = components[uv[:, 0]] == components[uv[:, 1]]
    conflict = ~attractive & internal
    if not conflict.any():
        return components

    conflict_components = np.unique(components[uv[conflict, 0]])
    node_mask = np.isin(components, conflict_components)
    sub_nodes = np.flatnonzero(node_mask)  # sorted
    edge_mask = node_mask[uv[:, 0]] & node_mask[uv[:, 1]]
    # The node compaction is monotone, so the edges stay canonical (u < v) and lexicographically sorted.
    sub_uv = np.searchsorted(sub_nodes, uv[edge_mask]).astype("uint64")
    sub_graph = bic.graph.UndirectedGraph.from_unique_edges(len(sub_nodes), np.ascontiguousarray(sub_uv))
    sub_labels = np.asarray(multicut_kernighan_lin(sub_graph, costs[edge_mask]), dtype="uint64")

    labels = components.copy()
    labels[sub_nodes] = np.uint64(int(components.max()) + 1) + sub_labels
    _, dense = np.unique(labels, return_inverse=True)
    return dense.reshape(-1).astype("uint64")


def _final_mapping(node_ids: np.ndarray, node_labels: np.ndarray, with_background: bool) -> Dict[int, int]:
    """The ``{id: stitched id}`` relabeling: consecutive ids from 1 (0 stays background when there is one)."""
    _, dense = np.unique(node_labels, return_inverse=True)
    mapping = {int(nid): int(lab) + 1 for nid, lab in zip(node_ids.tolist(), dense.reshape(-1).tolist())}
    if with_background:
        mapping[0] = 0
    return mapping


def _stitch_nodes(input_: SourceLike, out_array: SourceLike, tile_shape: Tuple[int, ...], node_ids: np.ndarray,
                  rows: np.ndarray, *, beta: float, with_background: bool, overlap_metric: str, min_overlap: int,
                  competition_disaffinity: Optional[float], num_workers: int, job_type: str,
                  job_config: Optional[RunnerConfig]) -> None:
    """Graph + solve in-process, then write the node labeling from ``input_`` into ``out_array`` via `relabel`."""
    uv, costs = _build_correspondence_graph(rows, node_ids, overlap_metric, int(min_overlap), float(beta),
                                            competition_disaffinity)
    node_labels = _solve_correspondence_graph(len(node_ids), uv, costs)
    mapping = _final_mapping(node_ids, node_labels, with_background)
    relabel(input_, mapping, output=out_array, block_shape=tile_shape, job_type=job_type, job_config=job_config,
            num_workers=num_workers)


def _stitch_from_store(store_handle, shape: Tuple[int, ...], tile_shape: Tuple[int, ...],
                       tile_overlap: Tuple[int, ...], out_array: SourceLike, *, beta: float, with_background: bool,
                       overlap_metric: str, min_overlap: int, competition_disaffinity: Optional[float],
                       num_workers: int, job_type: str, job_config: Optional[RunnerConfig],
                       return_before_stitching: bool) -> Union[SourceLike, Tuple[SourceLike, np.ndarray]]:
    """The shared stitching pipeline over a filled block store (see `stitch_block_segmentations`)."""
    ndim = len(shape)
    blocking = get_blocking(shape, tile_shape)
    offset_factor = _offset_factor(tile_shape, tile_overlap, int(blocking.number_of_blocks))
    runner = get_runner(job_type, job_config)

    # Stage 1: write the (offset) tile cores into the output; the core ids are the graph nodes.
    assemble = _make_assemble(shape, tile_shape, offset_factor, with_background, store_handle)
    id_results = runner.run(assemble, [], outputs=[out_array], block_shape=tile_shape, halo=tile_overlap,
                            num_workers=num_workers, has_return_val=True, name="stitch-assemble")
    node_ids = _merge_ids(id_results)

    # Stage 2: count the object overlaps in the shared halo faces of face-neighbouring tiles.
    overlap_fn = _make_store_overlap(shape, tile_shape, tile_overlap, offset_factor, with_background,
                                     store_handle, ndim)
    ov_results = runner.run(overlap_fn, [out_array], block_shape=tile_shape, num_workers=num_workers,
                            has_return_val=True, name="stitch-overlaps")
    rows = _collect_rows(ov_results)

    pre_stitch = np.asarray(as_source(out_array)[full_roi(ndim)]).copy() if return_before_stitching else None

    # Stage 3: the correspondence graph (in-process) and the block-wise relabeling of the output.
    _stitch_nodes(out_array, out_array, tile_shape, node_ids, rows, beta=beta, with_background=with_background,
                  overlap_metric=overlap_metric, min_overlap=min_overlap,
                  competition_disaffinity=competition_disaffinity, num_workers=num_workers, job_type=job_type,
                  job_config=job_config)
    if return_before_stitching:
        return out_array, pre_stitch
    return out_array


# ---------------------------------------------------------------------------------------------
# Public functions.
# ---------------------------------------------------------------------------------------------

def block_store_shape(shape: Sequence[int], tile_shape: Sequence[int], tile_overlap: Sequence[int]) -> Tuple[int, ...]:
    """The shape of the block store expected by `stitch_block_segmentations`.

    The store has one slot per tile along its first axis, followed by the haloed tile shape
    ``tile_shape + 2 * tile_overlap``. Slot ``block_id`` holds the segmentation of the haloed tile
    ``block_id`` -- in the tile ordering of ``bioimage_cpp.utils.Blocking`` over ``shape`` with
    ``tile_shape`` (see :func:`bioimage_py.util.get_blocking`) -- written at the slot's leading corner
    (tiles clipped at the volume border are smaller than the slot; the rest stays 0). Chunk a
    file-backed store as ``(1, *haloed tile shape)`` so that every tile writes its own chunk.

    Args:
        shape: The spatial shape of the volume.
        tile_shape: The shape of the individual tiles.
        tile_overlap: The halo added on each side of a tile.

    Returns:
        The store shape ``(number_of_tiles, *haloed tile shape)``.
    """
    shape = tuple(int(s) for s in shape)
    tile_shape, tile_overlap = _check_tiling(shape, tile_shape, tile_overlap)
    blocking = get_blocking(shape, tile_shape)
    return (int(blocking.number_of_blocks),) + tuple(ts + 2 * ov for ts, ov in zip(tile_shape, tile_overlap))


def stitch_block_segmentations(
    block_store: SourceLike,
    shape: Sequence[int],
    tile_shape: Sequence[int],
    tile_overlap: Sequence[int],
    output: Optional[SourceLike] = None,
    *,
    beta: float = 0.5,
    with_background: bool = True,
    overlap_metric: str = "geometric",
    min_overlap: int = 1,
    competition_disaffinity: Optional[float] = 0.8,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    return_before_stitching: bool = False,
) -> Union[SourceLike, Tuple[SourceLike, np.ndarray]]:
    """Stitch precomputed segmentations of haloed tiles into one consistent segmentation.

    This is the second half of `stitch_segmentation` for tiles that were segmented externally -- e.g.
    by a pool of persistent GPU workers that cannot be shipped as a per-tile callback -- and written
    into a block store laid out as described in `block_store_shape`: slot ``block_id`` holds the
    label image of the haloed tile ``block_id``, with ids unique *within* the tile (0 is background
    when ``with_background``; any integer dtype). The tile cores (the non-overlapping inner blocks)
    are copied into the output, the two predictions of every shared halo region are compared, and
    objects that correspond are merged via a multicut over the instance-correspondence graph (see the
    module docstring and `stitch_segmentation` for the parameters).

    Args:
        block_store: The block store (a numpy/zarr/n5 array or a `Source`) of shape
            ``block_store_shape(shape, tile_shape, tile_overlap)``; must be integer-typed and, for
            distributed execution, file-backed (reopenable). It is only read.
        shape: The spatial shape of the segmentation / volume.
        tile_shape: The shape of the individual tiles (the inner, non-overlapping blocks).
        tile_overlap: The halo added on each side of a tile; the haloed tiles have shape
            ``tile_shape + 2 * tile_overlap`` and the overlap is measured where two of them share voxels.
        output: The ``uint64`` output array. Optional for local execution -- a numpy array is
            allocated and returned if omitted; **required** (file-backed) for distributed execution.
            Its chunks must divide ``tile_shape``.
        beta: The boundary bias of the multicut in the exclusive range ``(0, 1)``: a correspondence with
            merge probability ``p`` (see ``overlap_metric``) is attractive iff ``p > beta``, so ``> 0.5``
            biases towards over-segmentation and ``< 0.5`` towards under-segmentation.
        with_background: Whether the problem has a background label (hard-coded ``0``) that is never
            stitched. Without background every id is an object, including ``0``.
        overlap_metric: How the merge probability of two objects is computed from their overlap in the
            shared region, with ``c`` the overlap count and ``|A|``, ``|B|`` the two object sizes there:
            ``"geometric"`` (``c / sqrt(|A| |B|)``, the balanced default), ``"iou"``
            (``c / (|A| + |B| - c)``), ``"dice"`` (``2c / (|A| + |B|)``), ``"min"`` (``min(c/|A|, c/|B|)``)
            or ``"max"`` (``c / min(|A|, |B|)``, the overlap coefficient: lenient towards fragments
            contained in a larger prediction).
        min_overlap: Correspondences with fewer overlapping voxels are ignored (absolute support).
        competition_disaffinity: The soft repulsion between two objects of one tile that both match the
            same object of the neighbouring tile (a one-to-many correspondence), as a disaffinity in
            the exclusive range ``(0, 1)``, or ``None`` to disable it. Merging both competitors with their
            common match beats keeping only the stronger one iff the *weaker* match's merge probability
            exceeds this value (several agreeing neighbours add up); with the default ``0.8`` one
            neighbouring tile cannot override a tile-local split of a cleanly halved object (each half
            has ``p ~ 0.71`` under the geometric mean), two agreeing neighbours can. ``None`` merges
            transitively whenever the correspondences themselves are attractive.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed backends)
            for the block-wise phases.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        return_before_stitching: Also return the pre-stitch segmentation (the assembled tile cores with
            tile-unique offset ids), for debugging. It is an in-memory numpy copy of the whole output.

    Returns:
        The output array with the stitched segmentation (consecutive ids from 1), or
        ``(output, pre_stitch)`` if ``return_before_stitching`` is set.
    """
    shape = tuple(int(s) for s in shape)
    tile_shape, tile_overlap = _check_tiling(shape, tile_shape, tile_overlap)
    _check_stitching_args(beta, overlap_metric, min_overlap, competition_disaffinity)

    store = as_source(block_store)
    expected_shape = block_store_shape(shape, tile_shape, tile_overlap)
    if tuple(int(s) for s in store.shape) != expected_shape:
        raise ValueError(f"block_store has shape {tuple(store.shape)}, expected block_store_shape(...) = "
                         f"{expected_shape}.")
    if not np.issubdtype(np.dtype(store.dtype), np.integer):
        raise ValueError(f"block_store must hold integer labels, got dtype {store.dtype}.")

    out_array = _prepare_output(output, shape, job_type)
    _check_output(out_array, shape)
    store_handle = _capture(store, job_type)
    return _stitch_from_store(
        store_handle, shape, tile_shape, tile_overlap, out_array, beta=beta, with_background=with_background,
        overlap_metric=overlap_metric, min_overlap=min_overlap, competition_disaffinity=competition_disaffinity,
        num_workers=num_workers, job_type=job_type, job_config=job_config,
        return_before_stitching=return_before_stitching,
    )


def stitch_segmentation(
    input: SourceLike,
    segmentation_function: Callable,
    tile_shape: Sequence[int],
    tile_overlap: Sequence[int],
    output: Optional[SourceLike] = None,
    *,
    beta: float = 0.5,
    shape: Optional[Sequence[int]] = None,
    with_background: bool = True,
    overlap_metric: str = "geometric",
    min_overlap: int = 1,
    competition_disaffinity: Optional[float] = 0.8,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    return_before_stitching: bool = False,
) -> Union[SourceLike, Tuple[SourceLike, np.ndarray]]:
    """Run a segmentation function tile-wise and stitch the results based on overlap.

    Each tile is read with a halo (``tile_shape + 2 * tile_overlap``) and segmented independently. The
    non-overlapping tile cores are written into the output; where two haloed tiles share voxels, their
    two predictions are compared object by object, and objects that correspond are merged via a
    multicut over the instance-correspondence graph -- whether or not their written cores touch at the
    tile boundary. The haloed tile segmentations are kept in a temporary block store (an in-memory
    array when the output is an in-memory numpy array, else a temporary zarr under
    ``job_config.tmp_root``) that is removed on success and preserved on failure, so the already
    segmented tiles can be stitched with `stitch_block_segmentations`.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`). If it has channels they must be
            the last (trailing) axes, and `shape` must give the spatial shape.
        segmentation_function: The per-tile segmentation function with signature
            ``f(tile_input, tile_id) -> labels``. It receives the haloed tile input and the tile id
            (passed in case the segmentation depends on the tile; ignore it otherwise) and returns an
            integer label image of the tile's (haloed) spatial shape with ids unique within the tile
            (0 is background when ``with_background``; ids must be smaller than the haloed tile's voxel
            count). It is cloudpickled for distributed execution, so it must be picklable.
        tile_shape: The shape of the individual tiles.
        tile_overlap: The halo added on each side of a tile; the input to the segmentation function
            has size ``tile_shape + 2 * tile_overlap``, and the overlap is measured in the halo. A zero
            overlap along an axis means objects are never merged across the seams of that axis.
        output: The ``uint64`` output array. Optional for local execution -- a numpy array is
            allocated and returned if omitted; **required** (file-backed) for distributed execution.
            Its chunks must divide ``tile_shape``.
        beta: The boundary bias of the multicut in the exclusive range ``(0, 1)``: a correspondence with
            merge probability ``p`` (see ``overlap_metric``) is attractive iff ``p > beta``, so ``> 0.5``
            biases towards over-segmentation and ``< 0.5`` towards under-segmentation.
        shape: The spatial shape of the segmentation. Defaults to the input shape; must be passed if
            the input has trailing channel axes.
        with_background: Whether the problem has a background label (hard-coded ``0``) that is never
            stitched. Without background every id is an object, including ``0``.
        overlap_metric: How the merge probability of two objects is computed from their overlap in the
            shared halo region; one of ``"geometric"`` (default), ``"iou"``, ``"dice"``, ``"min"`` or
            ``"max"``. See `stitch_block_segmentations` for the definitions.
        min_overlap: Correspondences with fewer overlapping voxels are ignored (absolute support).
        competition_disaffinity: The soft repulsion between two objects of one tile that both match the
            same object of the neighbouring tile, as a disaffinity in ``(0, 1)``, or ``None`` to disable
            it. See `stitch_block_segmentations` for the semantics of the default ``0.8``.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends) for the tile-segmentation, core-assembly, overlap-counting and relabeling phases.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`). For distributed
            backends the temporary block store is created under ``job_config.tmp_root`` (required on
            slurm, where it must be on a shared filesystem).
        return_before_stitching: Also return the pre-stitch segmentation (the assembled tile cores with
            tile-unique offset ids), for debugging. It is an in-memory numpy copy of the whole output.

    Returns:
        The output array with the stitched segmentation (consecutive ids from 1), or
        ``(output, pre_stitch)`` if ``return_before_stitching`` is set.
    """
    src = as_source(input)
    in_ndim = src.ndim
    shape = tuple(int(s) for s in (src.shape if shape is None else shape))
    ndim = len(shape)
    if in_ndim < ndim or tuple(int(s) for s in src.shape[:ndim]) != shape:
        raise ValueError(f"The input shape {tuple(src.shape)} does not start with the spatial shape {shape}.")
    tile_shape, tile_overlap = _check_tiling(shape, tile_shape, tile_overlap)
    _check_stitching_args(beta, overlap_metric, min_overlap, competition_disaffinity)

    out_array = _prepare_output(output, shape, job_type)
    out = _check_output(out_array, shape)

    blocking = get_blocking(shape, tile_shape)
    offset_factor = _offset_factor(tile_shape, tile_overlap, int(blocking.number_of_blocks))
    store_shape = block_store_shape(shape, tile_shape, tile_overlap)
    # Tile-local ids are < offset_factor, so they fit uint32 unless the haloed tile is enormous.
    store_dtype = "uint32" if offset_factor - 1 <= int(np.iinfo(np.uint32).max) else "uint64"
    in_memory = job_type == "local" and is_inmemory_numpy(out)

    runner = get_runner(job_type, job_config)
    input_handle = _capture(src, job_type)
    store, store_cleanup, store_handle, store_path = _make_block_store(store_shape, store_dtype, in_memory,
                                                                       job_type, job_config)
    try:
        # Stage 0: segment each haloed tile into its block store slot (tile-local ids).
        segment = _make_segment(shape, tile_shape, segmentation_function, offset_factor, store_handle,
                                input_handle, ndim, in_ndim)
        runner.run(segment, [out_array], block_shape=tile_shape, halo=tile_overlap, num_workers=num_workers,
                   name="stitch-segment")
        # Stages 1-3: assemble the cores, count the halo overlaps, solve and relabel.
        result = _stitch_from_store(
            store_handle, shape, tile_shape, tile_overlap, out_array, beta=beta, with_background=with_background,
            overlap_metric=overlap_metric, min_overlap=min_overlap, competition_disaffinity=competition_disaffinity,
            num_workers=num_workers, job_type=job_type, job_config=job_config,
            return_before_stitching=return_before_stitching,
        )
    except RunnerError as err:
        if store_path is not None:
            err.add_note(f"The per-tile block store of this run is preserved at {store_path}; after fixing the "
                         "failure, the already segmented tiles can be stitched with stitch_block_segmentations(...).")
        raise
    store_cleanup()
    return result


def stitch_tiled_segmentation(
    segmentation: SourceLike,
    tile_shape: Sequence[int],
    output: Optional[SourceLike] = None,
    *,
    overlap: int = 1,
    with_background: bool = True,
    beta: float = 0.5,
    overlap_metric: str = "geometric",
    min_overlap: int = 1,
    competition_disaffinity: Optional[float] = 0.8,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
) -> SourceLike:
    """Stitch a segmentation that is already split into tiles with unique ids per tile.

    The ids in the tiles of the input have to be unique (the segmentations are separate across
    tiles). Objects that touch across a tile interface are merged based on how strongly they overlap
    there -- measured between this tile's first ``overlap`` slab and the neighbouring tile's abutting
    last ``overlap`` slab -- via a multicut over the instance-correspondence graph (see the module
    docstring and `stitch_segmentation` for the parameters).

    Args:
        segmentation: The input tiled segmentation (a numpy/zarr/n5 array or a `Source`); must be
            integer-typed with ids unique across tiles.
        tile_shape: The shape of the tiles (the block shape of the tiling).
        output: The output array. Optional for local execution -- a ``uint64`` numpy array is allocated
            and returned if omitted; **required** (file-backed) for distributed execution. Its chunks
            must divide ``tile_shape``.
        overlap: The thickness (in pixels) of the tile-interface slabs used to measure object overlap.
        with_background: Whether the problem has a background label (hard-coded ``0``) that is never
            stitched. Without background every id is an object, including ``0``.
        beta: The boundary bias of the multicut; ``> 0.5`` biases towards over-segmentation,
            ``< 0.5`` towards under-segmentation. Must be in the exclusive range ``(0, 1)``.
        overlap_metric: How the merge probability of two touching objects is computed from their overlap
            in the interface slabs; one of ``"geometric"`` (default), ``"iou"``, ``"dice"``, ``"min"``
            or ``"max"``. See `stitch_block_segmentations` for the definitions.
        min_overlap: Correspondences with fewer overlapping voxels are ignored (absolute support).
        competition_disaffinity: The soft repulsion between two objects of one tile that both match the
            same object of the neighbouring tile, as a disaffinity in ``(0, 1)``, or ``None`` to disable
            it. See `stitch_block_segmentations` for the semantics of the default ``0.8``.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends) for the block-wise phases.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).

    Returns:
        The output array with merged labels (consecutive ids from 1; 0 stays background when there is one).
    """
    src = as_source(segmentation)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"stitch_tiled_segmentation expects an integer label image, got {src.dtype}.")
    ndim = src.ndim
    shape = tuple(int(s) for s in src.shape)
    tile_shape = tuple(int(t) for t in tile_shape)
    if len(tile_shape) != ndim:
        raise ValueError(f"tile_shape {tile_shape} must have one entry per axis ({ndim}).")
    if int(overlap) < 1:
        raise ValueError(f"overlap must be at least 1, got {overlap}.")
    _check_stitching_args(beta, overlap_metric, min_overlap, competition_disaffinity)
    out_array = _prepare_output(output, shape, job_type)

    # The graph nodes are all ids of the segmentation (a block-wise reduction, no whole-volume pass).
    node_ids = np.asarray(unique(segmentation, num_workers=num_workers, block_shape=tile_shape, job_type=job_type,
                                 job_config=job_config)).astype("uint64")
    if with_background:
        node_ids = node_ids[node_ids != 0]

    runner = get_runner(job_type, job_config)
    overlap_fn = _make_tiled_overlap(shape, tile_shape, int(overlap), ndim, with_background)
    results = runner.run(overlap_fn, [segmentation], block_shape=tile_shape, num_workers=num_workers,
                         has_return_val=True, name="stitch-overlaps")
    rows = _collect_rows(results)

    _stitch_nodes(segmentation, out_array, tile_shape, node_ids, rows, beta=beta, with_background=with_background,
                  overlap_metric=overlap_metric, min_overlap=min_overlap,
                  competition_disaffinity=competition_disaffinity, num_workers=num_workers, job_type=job_type,
                  job_config=job_config)
    return out_array
