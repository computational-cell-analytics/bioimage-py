"""Stitch a tile-wise over-segmentation into a globally consistent one (multicut over tile overlaps).

When a large image is segmented tile by tile, the same physical object gets a different id in each
tile it spans. These two functions reconcile the tiles by measuring object overlap where adjacent
tiles describe the same (or touching) pixels, building a region adjacency graph (RAG) over the
tiled labeling, turning each overlap into an attractive edge cost, and solving a multicut (see
:mod:`bioimage_py.segmentation.multicut`) to decide which cross-tile objects are one and the same.

- `stitch_segmentation` runs a user segmentation function tile-wise (each tile read with a halo) and
  stitches the results. The halo region — segmented by a tile but not written into the result — is
  where the overlap with the neighbouring tile's segmentation of the *same* pixels is measured.
- `stitch_tiled_segmentation` stitches an already-tiled segmentation (unique ids per tile, tiles
  non-overlapping) by measuring the overlap of objects *touching* across tile interfaces.

The per-voxel phases (tile segmentation, overlap counting) run through the `bioimage_py` runner, so
they scale across the ``local`` / ``subprocess`` / ``slurm`` backends. The RAG construction and the
multicut solve are still done in one process (see ``_compute_rag`` below).
"""
from __future__ import annotations

import multiprocessing
import os
import shutil
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple, Union

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source, from_spec
from ..util import BlockDescriptor, ComputeFn, full_roi, get_blocking, to_roi
from .multicut import compute_edge_costs, multicut_decomposition

__all__ = ["stitch_segmentation", "stitch_tiled_segmentation"]

# Edge disaffinity initial value (low overlap) and the value forced onto background-touching
# edges, mirroring elf: a high disaffinity makes merging very unlikely but not impossible.
_DEFAULT_DISAFFINITY = 0.9
_BACKGROUND_DISAFFINITY = 0.99

_INT_DTYPES = (np.uint32, np.uint64, np.int32, np.int64)


# ---------------------------------------------------------------------------------------------
# Region adjacency graph helpers.
#
# TODO: these wrap ``bioimage_cpp.graph`` directly and pull the whole segmentation into memory in
# the orchestrating process. Once ``bioimage_py`` grows dedicated (distributed, block-wise) graph
# functionality, the RAG construction and the node-label projection should move there and these
# private helpers should be replaced by it.
# ---------------------------------------------------------------------------------------------

def _compute_rag(segmentation: np.ndarray, n_threads: Optional[int] = None):
    """Compute the region adjacency graph of ``segmentation`` (``bic.graph.RegionAdjacencyGraph``)."""
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if segmentation.dtype not in _INT_DTYPES:
        segmentation = segmentation.astype("uint32")
    return bic.graph.region_adjacency_graph(segmentation, number_of_threads=n_threads)


def _project_node_labels_to_pixels(rag, segmentation: np.ndarray, node_labels: np.ndarray,
                                   n_threads: Optional[int] = None) -> np.ndarray:
    """Project per-node labels back to pixels via the RAG, yielding the merged segmentation."""
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if segmentation.dtype not in _INT_DTYPES:
        segmentation = segmentation.astype("uint64")
    node_labels = np.asarray(node_labels)
    if node_labels.dtype not in _INT_DTYPES:
        node_labels = node_labels.astype("uint64")
    return bic.graph.project_node_labels_to_pixels(rag, segmentation, node_labels,
                                                   number_of_threads=n_threads)


# ---------------------------------------------------------------------------------------------
# Overlap counting (shared) + per-block compute functions.
# ---------------------------------------------------------------------------------------------

def _face_overlap_rows(this_face: np.ndarray, ngb_face: np.ndarray) -> Optional[np.ndarray]:
    """Return ``(K, 3)`` float64 rows ``[label_a, label_b, fraction]`` for one tile interface.

    The fraction is the overlap count normalized by the size of ``label_a`` within the face (i.e.
    ``segmentation_overlap.overlaps_for_label_a(..., normalize=True)``), computed vectorially: the
    two faces have equal shape and cover every position once, so the size of ``label_a`` equals the
    sum of its overlap counts. Returns ``None`` when there is no overlap.
    """
    this_face = np.ascontiguousarray(this_face, dtype="uint64")
    ngb_face = np.ascontiguousarray(ngb_face, dtype="uint64")
    table = bic.utils.segmentation_overlap(this_face, ngb_face).overlap_table()
    if table.shape[0] == 0:
        return None
    la = table["label_a"].astype("float64")
    lb = table["label_b"].astype("float64")
    counts = table["count"].astype("float64")
    _, inv = np.unique(la, return_inverse=True)
    size_a = np.zeros(int(inv.max()) + 1, dtype="float64")
    np.add.at(size_a, inv, counts)
    frac = counts / size_a[inv]
    return np.stack([la, lb, frac], axis=1)


def _make_tiled_overlap(shape: Tuple[int, ...], tile_shape: Tuple[int, ...], overlap: int,
                        ndim: int) -> ComputeFn:
    """Build the overlap compute fn for `stitch_tiled_segmentation` (reads faces from the input)."""

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
            # This tile's first `overlap` slab along the axis; the lower neighbour's last `overlap`
            # slab is the abutting region just below it. Other axes use this tile's full extent.
            this_roi = tuple(slice(beg[d], beg[d] + overlap) if d == axis else slice(beg[d], end[d])
                             for d in range(ndim))
            ngb_roi = tuple(slice(beg[d] - overlap, beg[d]) if d == axis else slice(beg[d], end[d])
                            for d in range(ndim))
            this_face, ngb_face = seg[this_roi], seg[ngb_roi]
            if this_face.shape != ngb_face.shape:
                continue
            r = _face_overlap_rows(this_face, ngb_face)
            if r is not None:
                rows.append(r)
        return np.concatenate(rows, axis=0) if rows else None

    return _compute


def _make_seg_overlap(shape: Tuple[int, ...], tile_shape: Tuple[int, ...],
                      tile_overlap: Tuple[int, ...], store_handle, ndim: int) -> ComputeFn:
    """Build the overlap compute fn for `stitch_segmentation` (reads haloed tiles from the store)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        store = _resolve_source(store_handle)
        blocking = get_blocking(shape, tile_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.begin])
        rows: List[np.ndarray] = []
        for axis in range(ndim):
            ngb_id = blocking.get_neighbor_id(block_id, axis, True)
            if ngb_id == -1:
                continue
            this_b = blocking.get_block_with_halo(block_id, list(tile_overlap))
            ngb_b = blocking.get_block_with_halo(ngb_id, list(tile_overlap))
            this_ob = [int(c) for c in this_b.outer_block.begin]
            this_oe = [int(c) for c in this_b.outer_block.end]
            this_ib = [int(c) for c in this_b.inner_block.begin]
            ngb_ob = [int(c) for c in ngb_b.outer_block.begin]
            # Global face: along the axis from this tile's outer begin to its inner begin + overlap,
            # the full outer extent on the other axes (elf's `face`). Both tiles segmented it.
            face = [slice(this_ob[d], this_oe[d]) if d != axis
                    else slice(this_ob[d], this_ib[d] + tile_overlap[d]) for d in range(ndim)]
            this_local = tuple(slice(f.start - this_ob[d], f.stop - this_ob[d])
                               for d, f in enumerate(face))
            ngb_local = tuple(slice(f.start - ngb_ob[d], f.stop - ngb_ob[d])
                              for d, f in enumerate(face))
            this_face = store[(block_id,) + this_local]
            ngb_face = store[(ngb_id,) + ngb_local]
            if this_face.shape != ngb_face.shape:
                continue
            r = _face_overlap_rows(this_face, ngb_face)
            if r is not None:
                rows.append(r)
        return np.concatenate(rows, axis=0) if rows else None

    return _compute


def _make_segment(shape: Tuple[int, ...], tile_shape: Tuple[int, ...], tile_overlap: Tuple[int, ...],
                  seg_func: Callable, with_background: bool, offset_factor: int, store_handle,
                  input_handle, ndim: int, in_ndim: int) -> ComputeFn:
    """Build stage 1: segment each (haloed) tile, offset its ids, write the inner block + the store slot."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        input_ = _resolve_source(input_handle)
        store = _resolve_source(store_handle)
        output_ = outputs[0]
        blocking = get_blocking(shape, tile_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.inner_block.begin])

        outer_roi = to_roi(block.outer_block)
        if in_ndim > ndim:  # input carries trailing channel axes beyond the spatial shape
            outer_roi = outer_roi + tuple(slice(None) for _ in range(in_ndim - ndim))
        block_seg = np.asarray(seg_func(input_[outer_roi], block_id)).astype("uint64", copy=False)

        offset = np.uint64(int(block_id) * int(offset_factor))
        if with_background:
            fg = block_seg != 0
            if fg.any():
                block_seg[fg] += offset
        else:
            block_seg = block_seg + offset

        # Write the non-overlapping inner block into the pre-stitch output.
        output_[to_roi(block.inner_block)] = block_seg[to_roi(block.inner_block_local)]
        # Persist the full haloed tile into this block's store slot (leading corner; rest stays 0).
        store[(block_id,) + tuple(slice(0, s) for s in block_seg.shape)] = block_seg

        nz = block_seg[block_seg != 0]
        return np.unique(nz) if nz.size else None

    return _compute


def _make_relabel(mapping: dict) -> ComputeFn:
    """Build the in-place relabel stage applying ``mapping`` (offset ids -> dense ids)."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        output_ = outputs[0]
        roi = to_roi(block)
        output_[roi] = bic.utils.take_dict(mapping, output_[roi])
        return None

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


def _make_tile_store(n_blocks: int, max_halo: Tuple[int, ...], job_type: str,
                     job_config: Optional[RunnerConfig]) -> Tuple[Source, Callable[[], None], object]:
    """Create the temp store holding one haloed tile per block; return (store, cleanup, capture handle).

    Slot ``block_id`` is one chunk, so concurrent per-block writes never touch the same chunk. A
    numpy array backs the local path; a temporary zarr array (under ``job_config.tmp_root`` when set)
    backs the distributed path so workers can reopen it.
    """
    store_shape = (int(n_blocks),) + tuple(int(s) for s in max_halo)
    if job_type == "local":
        store = as_source(np.zeros(store_shape, dtype="uint64"))
        return store, (lambda: None), store

    import zarr

    tmp_root = job_config.tmp_root if job_config is not None else None
    if job_type == "slurm" and not tmp_root:
        raise ValueError("Distributed stitching on slurm requires job_config.tmp_root on a shared "
                         "filesystem (it holds the temporary per-tile store).")
    tmp_dir = tempfile.mkdtemp(prefix="bp-stitch-", dir=tmp_root)
    path = os.path.join(tmp_dir, "tiles.zarr")
    z = zarr.open_array(path, mode="w", shape=store_shape, chunks=(1,) + tuple(max_halo),
                        dtype="uint64")
    store = as_source(z)

    def _cleanup() -> None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return store, _cleanup, store.to_spec()


def _collect_edges(results: Optional[list]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the per-block ``[label_a, label_b, fraction]`` rows into ``(uv, fractions)``."""
    rows = [r for r in (results or []) if r is not None and len(r)]
    if not rows:
        return np.zeros((0, 2), dtype="uint64"), np.zeros((0,), dtype="float64")
    stacked = np.concatenate(rows, axis=0)
    return stacked[:, :2].astype("uint64"), stacked[:, 2].astype("float64")


def _map_edges(uv: np.ndarray, frac: np.ndarray, mapping: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Map overlap edge ids through ``mapping`` (offset -> dense), dropping ids absent from it."""
    if len(uv) == 0:
        return uv, frac
    keys = np.fromiter(mapping.keys(), dtype="uint64", count=len(mapping))
    valid = np.isin(uv[:, 0], keys) & np.isin(uv[:, 1], keys)
    uv, frac = uv[valid], frac[valid]
    if len(uv) == 0:
        return uv, frac
    u = bic.utils.take_dict(mapping, np.ascontiguousarray(uv[:, 0]))
    v = bic.utils.take_dict(mapping, np.ascontiguousarray(uv[:, 1]))
    return np.stack([u, v], axis=1).astype("uint64"), frac


def _stitch_via_multicut(seg: np.ndarray, uv: np.ndarray, frac: np.ndarray, with_background: bool,
                         beta: float, n_threads: Optional[int]) -> np.ndarray:
    """Build the RAG over ``seg``, set edge costs from the overlaps, solve multicut, project back."""
    rag = _compute_rag(seg, n_threads=n_threads)
    if rag.number_of_edges == 0 or rag.number_of_nodes <= 1:
        return np.asarray(seg).copy()

    disaffinities = np.full(int(rag.number_of_edges), _DEFAULT_DISAFFINITY, dtype="float32")

    if len(uv):
        # Keep only pairs whose ids exist in the segmentation (halo-only ids are not RAG nodes).
        seg_ids = np.unique(seg)
        valid = np.isin(uv, seg_ids).all(axis=1)
        uv_v, frac_v = uv[valid], frac[valid]
        if len(uv_v):
            edge_ids = np.asarray(rag.find_edges(uv_v.astype("uint64")))
            ok = edge_ids != -1
            edge_ids, frac_v = edge_ids[ok], frac_v[ok]
            if len(edge_ids):
                # Deterministically reduce duplicate edges by the maximum overlap (block order is
                # not guaranteed, so this is order-independent, unlike elf's last-writer).
                order = np.argsort(edge_ids, kind="stable")
                edge_ids, frac_v = edge_ids[order], frac_v[order]
                starts = np.flatnonzero(np.concatenate(([True], edge_ids[1:] != edge_ids[:-1])))
                disaffinities[edge_ids[starts]] = 1.0 - np.maximum.reduceat(frac_v, starts)

    if with_background:
        uv_ids = np.asarray(rag.uv_ids())
        bg_edges = np.asarray(rag.find_edges(uv_ids[(uv_ids == 0).any(axis=1)].astype("uint64")))
        bg_edges = bg_edges[bg_edges != -1]
        disaffinities[bg_edges] = _BACKGROUND_DISAFFINITY

    costs = compute_edge_costs(disaffinities, beta=beta)
    node_labels = multicut_decomposition(rag, costs, n_threads=n_threads or 1)
    return _project_node_labels_to_pixels(rag, seg, np.asarray(node_labels), n_threads=n_threads)


# ---------------------------------------------------------------------------------------------
# Public functions.
# ---------------------------------------------------------------------------------------------

def stitch_tiled_segmentation(
    segmentation: SourceLike,
    tile_shape: Tuple[int, ...],
    output: Optional[SourceLike] = None,
    *,
    overlap: int = 1,
    with_background: bool = True,
    beta: float = 0.5,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
) -> SourceLike:
    """Stitch a segmentation that is already split into tiles with unique ids per tile.

    The ids in the tiles of the input have to be unique (the segmentations are separate across
    tiles). Objects that touch across a tile interface are merged based on how strongly they overlap
    there, via a multicut over the region adjacency graph of the tiled segmentation.

    Args:
        segmentation: The input tiled segmentation (a numpy/zarr/n5 array or a `Source`); must be
            integer-typed with ids unique across tiles.
        tile_shape: The shape of the tiles (the block shape of the tiling).
        output: The ``uint64`` output array. Optional for local execution — a numpy array is
            allocated and returned if omitted; **required** (file-backed) for distributed execution.
        overlap: The thickness (in pixels) of the tile-interface slab used to measure object overlap.
        with_background: Whether the problem has a background label (hard-coded ``0``) that must not
            be merged with foreground objects.
        beta: The boundary bias of the multicut; ``> 0.5`` biases towards over-segmentation,
            ``< 0.5`` towards under-segmentation.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends) for the overlap-counting phase.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).

    Returns:
        The output array with merged labels.
    """
    src = as_source(segmentation)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"stitch_tiled_segmentation expects an integer label image, got {src.dtype}.")
    ndim = src.ndim
    shape = tuple(int(s) for s in src.shape)
    tile_shape = tuple(int(t) for t in tile_shape)
    out_array = _prepare_output(output, shape, job_type)

    runner = get_runner(job_type, job_config)
    overlap_fn = _make_tiled_overlap(shape, tile_shape, int(overlap), ndim)
    results = runner.run(overlap_fn, [segmentation], block_shape=tile_shape, num_workers=num_workers,
                         has_return_val=True, name="stitch-overlaps")
    uv, frac = _collect_edges(results)

    seg = src[full_roi(ndim)]
    stitched = _stitch_via_multicut(seg, uv, frac, with_background, beta, n_threads=None)

    out = as_source(out_array)
    out[full_roi(ndim)] = stitched.astype(out.dtype, copy=False)
    return out_array


def stitch_segmentation(
    input: SourceLike,
    segmentation_function: Callable,
    tile_shape: Tuple[int, ...],
    tile_overlap: Tuple[int, ...],
    output: Optional[SourceLike] = None,
    *,
    beta: float = 0.5,
    shape: Optional[Tuple[int, ...]] = None,
    with_background: bool = True,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    return_before_stitching: bool = False,
) -> Union[SourceLike, Tuple[SourceLike, np.ndarray]]:
    """Run a segmentation function tile-wise and stitch the results based on overlap.

    Each tile is read with a halo (``tile_shape + 2 * tile_overlap``) and segmented independently;
    the halo region is where the overlap with the neighbouring tile's segmentation of the same
    pixels is measured. Objects that overlap strongly there are merged via a multicut over the
    region adjacency graph of the per-tile labeling.

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`). If it has channels they must be
            the last (trailing) axes, and `shape` must give the spatial shape.
        segmentation_function: The per-tile segmentation function with signature
            ``f(tile_input, tile_id) -> labels``. It receives the haloed tile input and the tile id
            (passed in case the segmentation depends on the tile; ignore it otherwise) and returns a
            label image of the tile's (haloed) spatial shape. It is cloudpickled for distributed
            execution, so it must be picklable.
        tile_shape: The shape of the individual tiles.
        tile_overlap: The halo added on each side of a tile; the input to the segmentation function
            has size ``tile_shape + 2 * tile_overlap``, and the overlap is measured in the halo.
        output: The ``uint64`` output array. Optional for local execution — a numpy array is
            allocated and returned if omitted; **required** (file-backed) for distributed execution.
        beta: The boundary bias of the multicut; ``> 0.5`` biases towards over-segmentation,
            ``< 0.5`` towards under-segmentation. Must be in the exclusive range ``(0, 1)``.
        shape: The spatial shape of the segmentation. Defaults to the input shape; must be passed if
            the input has trailing channel axes.
        with_background: Whether the problem has a background label (hard-coded ``0``) that must not
            be merged with foreground objects.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends) for the tile-segmentation and overlap-counting phases.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`). For distributed
            backends a temporary per-tile store is created under ``job_config.tmp_root``.
        return_before_stitching: Also return the (relabeled) pre-stitch segmentation, for debugging.

    Returns:
        The output array with the stitched segmentation, or ``(output, pre_stitch)`` if
        ``return_before_stitching`` is set (``pre_stitch`` is an in-memory numpy array).
    """
    src = as_source(input)
    in_ndim = src.ndim
    shape = tuple(int(s) for s in (src.shape if shape is None else shape))
    ndim = len(shape)
    tile_shape = tuple(int(t) for t in tile_shape)
    tile_overlap = tuple(int(t) for t in tile_overlap)

    out_array = _prepare_output(output, shape, job_type)
    out = as_source(out_array)
    if out.dtype != np.dtype("uint64"):
        raise ValueError(f"output must have dtype uint64, got {out.dtype}.")

    blocking = get_blocking(shape, tile_shape)
    n_blocks = int(blocking.number_of_blocks)
    max_halo = tuple(ts + 2 * ov for ts, ov in zip(tile_shape, tile_overlap))
    offset_factor = int(np.prod(max_halo))
    if n_blocks * offset_factor >= int(np.iinfo(np.uint64).max):
        raise ValueError("Label id overflow: number_of_blocks * prod(haloed tile shape) exceeds "
                         "uint64. Reduce the tile shape or the volume size.")

    runner = get_runner(job_type, job_config)
    input_handle = _capture(src, job_type)
    store, store_cleanup, store_handle = _make_tile_store(n_blocks, max_halo, job_type, job_config)
    try:
        # Stage 1: segment each haloed tile, offset its ids, write the inner block + the store slot.
        stage1 = _make_segment(shape, tile_shape, tile_overlap, segmentation_function,
                               with_background, offset_factor, store_handle, input_handle,
                               ndim, in_ndim)
        id_results = runner.run(stage1, [], outputs=[out_array], block_shape=tile_shape,
                                halo=tile_overlap, num_workers=num_workers, has_return_val=True,
                                name="stitch-segment")
        id_arrays = [a for a in id_results if a is not None and len(a)]
        real = (np.unique(np.concatenate(id_arrays)) if id_arrays
                else np.zeros((0,), dtype="uint64"))

        # Build the dense relabeling (offset ids are sparse; this keeps the RAG node space compact).
        mapping = {0: 0}
        for i, lab in enumerate(real.tolist()):
            mapping[int(lab)] = i + 1

        # Stage 2: apply the dense relabeling to the pre-stitch output in place.
        if real.size:
            runner.run(_make_relabel(mapping), [], outputs=[out_array], block_shape=tile_shape,
                       num_workers=num_workers, has_return_val=False, name="stitch-densify")

        # Stage 3: count object overlaps in the halo bands, reading haloed tiles from the store.
        overlap_fn = _make_seg_overlap(shape, tile_shape, tile_overlap, store_handle, ndim)
        ov_results = runner.run(overlap_fn, [out_array], block_shape=tile_shape,
                                num_workers=num_workers, has_return_val=True, name="stitch-overlaps")
    finally:
        store_cleanup()

    uv, frac = _collect_edges(ov_results)
    uv, frac = _map_edges(uv, frac, mapping)  # overlap ids are offset ids -> map to dense.

    seg = out[full_roi(ndim)]
    stitched = _stitch_via_multicut(seg, uv, frac, with_background, beta, n_threads=None)

    # Relabel to consecutive ids (elf semantics): keep 0 as background, or renumber 0 too when there
    # is no background.
    if with_background:
        stitched, _, _ = bic.segmentation.relabel_sequential(stitched.astype("uint64"), offset=1)
    else:
        stitched, _, _ = bic.segmentation.relabel_sequential(stitched.astype("uint64") + 1, offset=1)

    pre_stitch = seg.copy() if return_before_stitching else None
    out[full_roi(ndim)] = stitched.astype(out.dtype, copy=False)
    if return_before_stitching:
        return out_array, pre_stitch
    return out_array
