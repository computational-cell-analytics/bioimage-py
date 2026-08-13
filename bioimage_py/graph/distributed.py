"""Distributed region-adjacency graphs and edge features (block-wise, across backends).

These operations build the region adjacency graph (RAG) of a labeled volume, and optionally its edge
features, block by block and merge the per-block results into one whole-volume graph. They wrap the
low-level primitives in :mod:`bioimage_cpp.graph.distributed` with the ``bioimage_py`` orchestration
(block iteration, halo sizing, I/O, and the in-process merge), so they scale identically across the
``local`` / ``subprocess`` / ``slurm`` backends -- mirroring the reduction ops in
:mod:`bioimage_py.evaluation` (per-block returns through ``runner.run(..., has_return_val=True)``,
merged in the orchestrating process).

Two contracts the caller must honor:

- **Globally consistent labels.** A segment must have the *same id in every block*. Run a
  stitching / consistent-id step first (e.g. :func:`bioimage_py.segmentation.stitch_segmentation`,
  or a distributed watershed with global ids) -- making per-block-local ids consistent is not part
  of these operations.
- **Moment-only complex features.** Median / percentiles cannot be reconstructed from block
  partials, so the distributed complex feature output is the moment subset ``[mean, std, min, max,
  size]`` -- it equals the corresponding columns of the in-core complex features.

The graph node ids are the label ids (node ``i`` is segment ``i``), so a multicut / clustering node
labeling is written back to pixels with the canonical node-label writer
:func:`bioimage_py.segmentation.relabel` using a dense array labeling.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source, capture_source, resolve_source
from ..util import BlockDescriptor, ComputeFn, check_direct, full_roi, to_roi

__all__ = ["distributed_rag", "distributed_edge_features"]

# Label dtypes the bioimage_cpp graph primitives accept directly; others are cast to uint64 per block.
_SUPPORTED_LABEL_DTYPES = (
    np.dtype("uint32"), np.dtype("uint64"), np.dtype("int32"), np.dtype("int64"),
)
_FEATURE_TYPES = ("edge_map", "affinity")


def _offset_halo(offsets: Sequence[Sequence[int]], ndim: int) -> List[int]:
    """Per-axis halo for the affinity pass: ``max(1, max_offset)`` on every axis.

    The affinity pass extracts both the nearest-neighbor RAG (needs halo >= 1 on every axis) and the
    affinity statistics (need halo >= ``max |offset component|`` per axis), so the halo is the
    elementwise maximum of the two -- the ``1`` floor is what keeps NN edges on axes that only carry
    long-range offsets.
    """
    halo = [1] * ndim
    for offset in offsets:
        for axis in range(ndim):
            halo[axis] = max(halo[axis], abs(int(offset[axis])))
    return halo


def _block_extract(labels: np.ndarray, data: Optional[np.ndarray], feature_type: str,
                   offsets: Optional[Sequence[Sequence[int]]], own_begin: Sequence[int],
                   own_shape: Sequence[int]) -> Tuple[np.ndarray, Optional[np.ndarray],
                                                      Optional[np.ndarray], int]:
    """Extract one (haloed) block's owned edges, partial stats and label max.

    Shared by the direct fast path and the per-block compute function, so ``direct == blocked`` holds
    by construction. Returns ``(graph_edges, stats_edges, stats, local_max)``: ``graph_edges`` are
    the owned nearest-neighbor edges (feed the global graph); ``stats_edges`` / ``stats`` are the
    edges / partial statistics for the feature fold (``None`` for the graph-only pass); ``local_max``
    is the maximum label id in the block (folds the node-count reduction into this pass).
    """
    if labels.dtype not in _SUPPORTED_LABEL_DTYPES:
        labels = labels.astype("uint64")
    local_max = int(labels.max()) if labels.size else 0
    dist = bic.graph.distributed
    if feature_type == "edge_map":
        # The stats primitive's edges equal block_region_adjacency_edges for the block, so one call
        # yields both the graph edges and the feature stats.
        edges, stats = dist.block_edge_map_stats(labels, data, own_begin, own_shape)
        return edges, edges, stats, local_max
    if feature_type == "affinity":
        # The affinity stats edges may include long-range-only pairs, so the graph must come from the
        # nearest-neighbor extraction; long-range pairs are dropped by merge_block_edge_stats.
        graph_edges = dist.block_region_adjacency_edges(labels, own_begin, own_shape)
        stats_edges, stats = dist.block_affinity_stats(labels, data, offsets, own_begin, own_shape)
        return graph_edges, stats_edges, stats, local_max
    graph_edges = dist.block_region_adjacency_edges(labels, own_begin, own_shape)
    return graph_edges, None, None, local_max


def _make_compute(feature_type: str, offsets: Optional[Sequence[Sequence[int]]],
                  data_handle: Optional[object]) -> ComputeFn:
    """Build the per-block compute function (a cloudpicklable closure over the captured data handle).

    ``labels`` is the runner input (it defines the domain / blocking); ``data`` (the edge map or the
    affinities) is captured and reopened per block, never passed as a runner input -- the affinities'
    ``(channels, *spatial)`` shape would fail the runner's input shape check, and capturing is
    uniform for the edge map too.
    """

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Tuple[np.ndarray, Optional[np.ndarray],
                                                  Optional[np.ndarray], int]:
        labels_src = inputs[0]
        outer = to_roi(block.outer_block)  # halo included; symmetric + clamped at borders.
        labels = np.ascontiguousarray(labels_src[outer])
        own_begin = [int(b) for b in block.inner_block_local.begin]
        own_shape = [int(e) - int(b)
                     for b, e in zip(block.inner_block_local.begin, block.inner_block_local.end)]
        data = None
        if data_handle is not None:
            data_src = resolve_source(data_handle)
            if feature_type == "affinity":
                data = np.ascontiguousarray(data_src[(slice(None),) + outer])
            else:
                data = np.ascontiguousarray(data_src[outer])
        return _block_extract(labels, data, feature_type, offsets, own_begin, own_shape)

    return _compute


def _resolve_n_nodes(results: Sequence[tuple], n_nodes: Optional[int]) -> int:
    """Return the node count: ``max(label) + 1`` (from the per-block maxes) unless given explicitly.

    Must be ``max(label) + 1`` (never ``edges.max() + 1``): isolated segments have no edges but still
    need a node id for the labeling / writeback and for the stats fold's edge-endpoint validity. A
    caller-supplied ``n_nodes`` must be at least this large.
    """
    local_max = max((int(r[3]) for r in results), default=0)
    required = local_max + 1
    if n_nodes is None:
        return required
    n_nodes = int(n_nodes)
    if n_nodes < required:
        raise ValueError(
            f"n_nodes={n_nodes} is too small: the labels contain id {local_max}, so n_nodes must be "
            f">= {required}."
        )
    return n_nodes


def _build_graph(results: Sequence[tuple], n_nodes: int) -> "bic.graph.UndirectedGraph":
    """Merge the per-block edges into the global :class:`bioimage_cpp.graph.UndirectedGraph`."""
    merged = bic.graph.distributed.merge_edges([r[0] for r in results])
    if len(merged) == 0:
        # from_unique_edges must not be called with an empty edge array; build the node-only graph.
        return bic.graph.UndirectedGraph(int(n_nodes))
    return bic.graph.UndirectedGraph.from_unique_edges(int(n_nodes), merged)


def _reduce_features(results: Sequence[tuple], graph: "bic.graph.UndirectedGraph",
                     complex_features: bool) -> np.ndarray:
    """Fold the per-block partial stats onto the global edges and finalize the features."""
    dist = bic.graph.distributed
    n_edges = int(graph.number_of_edges)
    acc = dist.empty_edge_stats(n_edges)
    if n_edges:
        for _, stats_edges, stats, _ in results:
            acc = dist.merge_block_edge_stats(graph, acc, stats_edges, stats)
    return dist.finalize_edge_features(acc, compute_complex_features=complex_features)


def _validate_labels(labels_src: Source) -> None:
    """Validate the label source is an integer 2D/3D image (the primitives' domain)."""
    if not np.issubdtype(np.dtype(labels_src.dtype), np.integer):
        raise ValueError(f"labels must be an integer label image, got dtype {labels_src.dtype}.")
    if labels_src.ndim not in (2, 3):
        raise ValueError(f"labels must be a 2D or 3D image, got ndim={labels_src.ndim}.")


def _validate_data(data_src: Source, labels_src: Source, feature_type: str,
                   offsets: Optional[Sequence[Sequence[int]]]) -> None:
    """Validate the feature data shape against the labels (edge map same-shape / affinity channels)."""
    label_shape = tuple(int(s) for s in labels_src.shape)
    if feature_type == "edge_map":
        if tuple(int(s) for s in data_src.shape) != label_shape:
            raise ValueError(
                f"edge_map data shape {tuple(data_src.shape)} must match the labels shape "
                f"{label_shape}."
            )
    else:  # affinity
        if data_src.ndim != labels_src.ndim + 1:
            raise ValueError(
                "affinity data must have shape (len(offsets), *labels.shape), got ndim="
                f"{data_src.ndim} for {labels_src.ndim}D labels."
            )
        if tuple(int(s) for s in data_src.shape[1:]) != label_shape:
            raise ValueError(
                f"affinity data spatial shape {tuple(data_src.shape[1:])} must match the labels "
                f"shape {label_shape}."
            )
        if int(data_src.shape[0]) != len(offsets):
            raise ValueError(
                f"offsets length {len(offsets)} must match the affinity channel count "
                f"{int(data_src.shape[0])}."
            )


def _run(labels: SourceLike, data: Optional[SourceLike], feature_type: str,
         offsets: Optional[Sequence[Sequence[int]]], complex_features: bool,
         block_shape: Optional[Tuple[int, ...]], n_nodes: Optional[int], num_workers: int,
         job_type: str, job_config: Optional[RunnerConfig]):
    """Shared driver for the graph-only ("rag") and feature passes (see the public functions)."""
    labels_src = as_source(labels)
    _validate_labels(labels_src)
    ndim = labels_src.ndim
    data_src = as_source(data) if data is not None else None
    if data_src is not None:
        _validate_data(data_src, labels_src, feature_type, offsets)

    halo = _offset_halo(offsets, ndim) if feature_type == "affinity" else [1] * ndim

    if check_direct(job_type, num_workers, block_shape, None, None):
        # Direct fast path: one whole-array extraction feeding the same merge/fold helpers.
        labels_arr = np.ascontiguousarray(labels_src[full_roi(ndim)])
        data_arr = None
        if data_src is not None:
            data_arr = np.ascontiguousarray(
                data_src[(slice(None),) + full_roi(ndim)] if feature_type == "affinity"
                else data_src[full_roi(ndim)])
        shape = [int(s) for s in labels_src.shape]
        results = [_block_extract(labels_arr, data_arr, feature_type, offsets, [0] * ndim, shape)]
    else:
        runner = get_runner(job_type, job_config)
        data_handle = capture_source(data_src, job_type) if data_src is not None else None
        compute = _make_compute(feature_type, offsets, data_handle)
        results = runner.run(compute, [labels], block_shape=block_shape, halo=halo,
                             num_workers=num_workers, has_return_val=True,
                             name=f"distributed-{feature_type}")

    n_nodes_ = _resolve_n_nodes(results, n_nodes)
    graph = _build_graph(results, n_nodes_)
    if feature_type == "rag":
        return graph
    features = _reduce_features(results, graph, complex_features)
    return graph, features


def distributed_rag(
    labels: SourceLike,
    *,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_nodes: Optional[int] = None,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
) -> "bic.graph.UndirectedGraph":
    """Build the region adjacency graph of a labeled volume, block-wise.

    Each block is read with a one-pixel halo, its owned nearest-neighbor edges are extracted, and the
    per-block edge sets are merged into one whole-volume graph. The result is exact regardless of how
    objects straddle block boundaries: it equals the whole-volume
    ``bioimage_cpp.graph.region_adjacency_graph(labels)``. Graph node ``i`` is segment ``i``.

    Args:
        labels: The input segmentation (a numpy/zarr/n5 array or a `Source`); must be an integer 2D
            or 3D label image with **globally consistent** ids (the same id in every block).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required for
            unchunked data.
        n_nodes: The number of graph nodes. Defaults to ``max(label) + 1`` (derived in the block
            pass). If given, must be at least ``max(label) + 1``.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).

    Returns:
        The whole-volume region adjacency graph as a `bioimage_cpp.graph.UndirectedGraph`.
    """
    return _run(labels, None, "rag", None, False, block_shape, n_nodes, num_workers, job_type,
                job_config)


def distributed_edge_features(
    labels: SourceLike,
    data: SourceLike,
    *,
    feature_type: str = "edge_map",
    offsets: Optional[Sequence[Sequence[int]]] = None,
    complex_features: bool = False,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_nodes: Optional[int] = None,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
) -> Tuple["bic.graph.UndirectedGraph", np.ndarray]:
    """Build the region adjacency graph and its edge features in one block pass.

    Each block is read with a halo, its owned edges and partial edge statistics are extracted, and
    both are merged into the whole-volume graph and finalized edge features. The features match the
    moment columns of the in-core ``bioimage_cpp.graph.features`` on the whole volume: ``size``,
    ``min`` and ``max`` are exact; ``mean`` and ``std`` match to floating-point tolerance (their
    block-merge order is not fixed).

    Args:
        labels: The input segmentation (a numpy/zarr/n5 array or a `Source`); must be an integer 2D
            or 3D label image with **globally consistent** ids (the same id in every block).
        data: The data the features are computed from. For ``feature_type="edge_map"`` an array the
            same shape as ``labels`` (the edge value is the average of the two endpoint pixels). For
            ``feature_type="affinity"`` an array of shape ``(len(offsets), *labels.shape)``.
        feature_type: The feature source; one of ``"edge_map"`` or ``"affinity"``.
        offsets: The affinity offsets (one per channel); required for ``feature_type="affinity"``,
            ignored otherwise. The halo is sized from them (``max(1, max |offset|)`` per axis).
        complex_features: If ``False`` the features are ``(E, 2)`` columns ``[mean, size]``; if
            ``True`` they are ``(E, 5)`` columns ``[mean, std, min, max, size]``.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required for
            unchunked data.
        n_nodes: The number of graph nodes. Defaults to ``max(label) + 1`` (derived in the block
            pass). If given, must be at least ``max(label) + 1``.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).

    Returns:
        A tuple ``(graph, features)`` of the whole-volume `bioimage_cpp.graph.UndirectedGraph` and
        the ``(number_of_edges, k)`` feature array (rows aligned to the graph edge ids).

    Raises:
        ValueError: If ``feature_type`` is unknown, or ``offsets`` is missing for the affinity
            feature type, or the data shape is incompatible with the labels.
    """
    if feature_type not in _FEATURE_TYPES:
        types = ", ".join(_FEATURE_TYPES)
        raise ValueError(f"feature_type must be one of ({types}), got {feature_type!r}.")
    if feature_type == "affinity":
        if offsets is None:
            raise ValueError("offsets is required for feature_type='affinity'.")
        offsets = [[int(v) for v in offset] for offset in offsets]
        if not offsets:
            raise ValueError("offsets must not be empty.")
        if any(len(offset) != as_source(labels).ndim for offset in offsets):
            raise ValueError("each offset must have length matching the labels ndim.")
    return _run(labels, data, feature_type, offsets, complex_features, block_shape, n_nodes,
                num_workers, job_type, job_config)
