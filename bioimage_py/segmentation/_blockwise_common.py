"""Shared machinery for the block-wise (lifted) multicut solvers.

The block-wise multicut (Pape et al. 2017) solves a global multicut by hierarchical domain
decomposition: at each level the graph is partitioned into blocks, each block's induced sub-problem is
solved independently (the *map* step, distributed across the runner backends), and the per-block cut
decisions are merged into a coarser graph (the *reduce* step, in the orchestrating process). Repeating
this over successively larger blocks contracts the problem until a final global solve is cheap.

This module holds the pieces shared by :mod:`bioimage_py.segmentation.blockwise_multicut` and
:mod:`bioimage_py.segmentation.blockwise_lifted_multicut`:

- the numpy reduce primitives ported from ``elf`` (union-find merge + edge contraction with summed
  costs; ``bioimage_cpp.utils.UnionFind`` replaces nifty's ufd, and :func:`_remap_edges_sum_costs`
  replaces nifty's ``EdgeMapping``),
- :func:`reduce_problem`, the in-process reduction of one level,
- :func:`solve_subproblems_distributed`, the distributed map step (per-block sub-problem solve via
  ``runner.run``), and
- the internal-solver dispatch-by-name (so the cloudpickled per-block closure captures a string, not a
  solver object).

The block's node membership is derived from the segmentation volume, mapped through the running
composed node labeling ``labels`` (original node id -> current supernode id). This keeps the *original*
segmentation as the only inter-level volume and, unlike the ``elf`` reference, makes ``n_levels > 1``
correct (``elf`` indexes the contracted graph with original ids).
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import bioimage_cpp as bic
import numpy as np

from ..sources import Source, SourceSpec, as_source, from_spec
from ..util import BlockDescriptor, ComputeFn, full_roi, to_roi


def _undirected_graph_types() -> tuple:
    """The concrete ``UndirectedGraph`` classes (the public wrapper and the raw ``_core`` class).

    ``bioimage_cpp`` exposes a Python wrapper ``bic.graph.UndirectedGraph`` whose instances (from
    ``from_edges``) are distinct from the raw ``bic._core.UndirectedGraph`` returned by some builders
    (e.g. ``from_unique_edges``, used by ``distributed_rag``), so a single ``isinstance`` check is not
    enough. Both are accepted as an already-built graph.
    """
    types = [bic.graph.UndirectedGraph]
    core = getattr(bic, "_core", None)
    core_type = getattr(core, "UndirectedGraph", None)
    if core_type is not None and core_type not in types:
        types.append(core_type)
    return tuple(types)


_UNDIRECTED_GRAPH_TYPES = _undirected_graph_types()


def _relabel_from_zero(node_ids: np.ndarray) -> Tuple[np.ndarray, int, Dict[int, int]]:
    """Map ``node_ids`` to consecutive integers from 0. Returns ``(relabeled, max_id, {old: new})``."""
    uniq, inverse = np.unique(node_ids, return_inverse=True)
    relabeled = np.reshape(inverse, -1).astype(node_ids.dtype, copy=False)
    mapping = {int(old): int(new) for new, old in enumerate(uniq)}
    return relabeled, int(uniq.size - 1), mapping


def _relabel_keep_zero(labels: np.ndarray) -> np.ndarray:
    """Map labels to consecutive integers from 1 while keeping 0 fixed at 0."""
    out = np.zeros_like(labels)
    nonzero_mask = labels != 0
    if nonzero_mask.any():
        _, inverse = np.unique(labels[nonzero_mask], return_inverse=True)
        out[nonzero_mask] = (np.reshape(inverse, -1) + 1).astype(labels.dtype, copy=False)
    return out


def _remap_edges_sum_costs(uv_ids: np.ndarray, new_labels: np.ndarray,
                           costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Contract ``uv_ids`` onto ``new_labels`` and sum the costs collapsing onto each supernode pair.

    Builds one row per unique pair of distinct supernodes (self-loops are dropped) and sums the costs
    of all original edges mapping onto that pair. Replaces nifty's ``EdgeMapping``.
    """
    uv_ids = np.asarray(uv_ids)
    costs = np.asarray(costs)
    if uv_ids.size == 0:
        return uv_ids.reshape(0, 2).astype("uint64"), np.zeros(0, dtype=costs.dtype)

    remapped = new_labels[uv_ids]
    keep = remapped[:, 0] != remapped[:, 1]
    remapped = remapped[keep]
    sub_costs = costs[keep]
    if remapped.shape[0] == 0:
        return np.zeros((0, 2), dtype="uint64"), np.zeros(0, dtype=sub_costs.dtype)

    # canonicalize each row so the same undirected edge always maps to the same key
    u = np.minimum(remapped[:, 0], remapped[:, 1]).astype(np.int64)
    v = np.maximum(remapped[:, 0], remapped[:, 1]).astype(np.int64)
    n_new = int(new_labels.max()) + 1
    key = u * n_new + v

    unique_key, inverse = np.unique(key, return_inverse=True)
    inverse = np.reshape(inverse, -1)
    new_costs = np.zeros(unique_key.size, dtype=sub_costs.dtype)
    np.add.at(new_costs, inverse, sub_costs)

    new_u = (unique_key // n_new).astype("uint64")
    new_v = (unique_key % n_new).astype("uint64")
    new_uv_ids = np.stack([new_u, new_v], axis=1)
    return new_uv_ids, new_costs


def find_inner_lifted_edges(lifted_uv_ids: np.ndarray, node_list: np.ndarray) -> np.ndarray:
    """Return the indices of the lifted edges with **both** endpoints in ``node_list``."""
    lifted_uv_ids = np.asarray(lifted_uv_ids)
    if lifted_uv_ids.size == 0:
        return np.zeros(0, dtype="int64")
    lifted_indices = np.arange(len(lifted_uv_ids))
    inner_us = np.isin(lifted_uv_ids[:, 0], node_list)
    inner_indices = lifted_indices[inner_us]
    inner_uvs = lifted_uv_ids[inner_us]
    inner_vs = np.isin(inner_uvs[:, 1], node_list)
    return inner_indices[inner_vs]


def build_graph(n_nodes: int, uv_ids: np.ndarray) -> "bic.graph.UndirectedGraph":
    """Build a ``bic.graph.UndirectedGraph`` from ``uv_ids`` (node-only when there are no edges)."""
    uv_ids = np.ascontiguousarray(np.asarray(uv_ids), dtype="uint64")
    if uv_ids.size == 0:
        return bic.graph.UndirectedGraph(int(n_nodes))
    return bic.graph.UndirectedGraph.from_edges(int(n_nodes), uv_ids)


def as_undirected_graph(graph) -> "bic.graph.UndirectedGraph":
    """Return ``graph`` as an ``UndirectedGraph`` (rebuild from edges if it is a RAG)."""
    if isinstance(graph, _UNDIRECTED_GRAPH_TYPES):
        return graph
    return bic.graph.UndirectedGraph.from_edges(
        graph.number_of_nodes, np.asarray(graph.uv_ids(), dtype="uint64"),
    )


def multicut_solver_by_name(name: str):
    """Return a whole-graph multicut solver ``solver(graph, costs) -> node_labels`` for ``name``."""
    from .multicut import multicut_decomposition, multicut_gaec, multicut_kernighan_lin

    if name == "kernighan-lin":
        return multicut_kernighan_lin
    if name == "greedy-additive":
        return multicut_gaec
    if name == "decomposition":
        return multicut_decomposition
    raise ValueError(
        f"Unknown internal multicut solver {name!r}; expected one of 'kernighan-lin', "
        "'greedy-additive', 'decomposition'."
    )


def lifted_solver_by_name(name: str):
    """Return a lifted solver ``solver(graph, costs, lifted_uv, lifted_costs) -> node_labels``."""
    from .lifted_multicut import lifted_multicut_gaec, lifted_multicut_kernighan_lin

    if name == "kernighan-lin":
        return lifted_multicut_kernighan_lin
    if name == "greedy-additive":
        return lifted_multicut_gaec
    raise ValueError(
        f"Unknown internal lifted multicut solver {name!r}; expected one of 'kernighan-lin', "
        "'greedy-additive'."
    )


def reduce_problem(graph, costs: np.ndarray, merge_mask: np.ndarray,
                   lifted_uv_ids: Optional[np.ndarray] = None,
                   lifted_costs: Optional[np.ndarray] = None):
    """Reduce one level: union-find merge over the kept edges, then contract the graph.

    Args:
        graph: The current level's graph.
        costs: The current level's base edge costs (aligned to ``graph`` edge ids).
        merge_mask: Boolean mask over the base edges; ``True`` edges are merged (kept), ``False`` cut.
        lifted_uv_ids: Optional current-level lifted edges (contracted through the same labeling).
        lifted_costs: Optional current-level lifted costs.

    Returns:
        ``(new_graph, new_costs, new_labels)`` for the base problem, or
        ``(new_graph, new_costs, new_labels, new_lifted_uv_ids, new_lifted_costs)`` when lifted arrays
        are given. ``new_labels`` maps every current node id to its (consecutive) supernode id.
    """
    n_nodes = int(graph.number_of_nodes)
    nodes = np.arange(n_nodes, dtype="uint64")
    uv_ids = np.asarray(graph.uv_ids(), dtype="uint64")

    ufd = bic.utils.UnionFind(n_nodes)
    merge_uv = np.ascontiguousarray(uv_ids[merge_mask], dtype="uint64")
    if merge_uv.size:
        ufd.merge(merge_uv)
    new_labels = _relabel_keep_zero(np.asarray(ufd.find(nodes)))

    new_uv_ids, new_costs = _remap_edges_sum_costs(uv_ids, new_labels, np.asarray(costs))
    n_new_nodes = int(new_labels.max()) + 1 if new_labels.size else 0
    new_graph = build_graph(n_new_nodes, new_uv_ids)

    if lifted_uv_ids is None:
        return new_graph, new_costs, new_labels

    lifted_uv_ids = np.asarray(lifted_uv_ids, dtype="uint64")
    if lifted_uv_ids.size:
        new_lifted_uvs, new_lifted_costs = _remap_edges_sum_costs(
            lifted_uv_ids, new_labels, np.asarray(lifted_costs),
        )
    else:
        new_lifted_uvs = lifted_uv_ids.reshape(0, 2)
        new_lifted_costs = np.zeros(0, dtype="float64")
    return new_graph, new_costs, new_labels, new_lifted_uvs, new_lifted_costs


# --- distributed map step ---------------------------------------------------------------------------


_CANONICAL_DTYPES = {
    ("u", 1): np.uint8, ("u", 2): np.uint16, ("u", 4): np.uint32, ("u", 8): np.uint64,
    ("i", 1): np.int8, ("i", 2): np.int16, ("i", 4): np.int32, ("i", 8): np.int64,
    ("f", 2): np.float16, ("f", 4): np.float32, ("f", 8): np.float64, ("b", 1): np.bool_,
}


def _canonicalize(arr: np.ndarray) -> np.ndarray:
    """Cast to the canonical numpy dtype for its (kind, itemsize).

    ``bioimage_cpp`` returns e.g. a ``ULongLong``-backed ``uint64`` (C ``unsigned long long``), which
    is ``==``-equal to but a distinct dtype class from numpy's canonical ``UInt64`` -- and zarr v3's
    dtype registry matches only the canonical class. Re-casting normalizes it so the array persists.
    """
    arr = np.asarray(arr)
    canon = _CANONICAL_DTYPES.get((arr.dtype.kind, arr.dtype.itemsize))
    if canon is None:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr.astype(canon))


def _persist_array(arr: np.ndarray, tmp_dir: str, name: str):
    """Persist an array to a temp zarr and return its reopen spec (for distributed workers)."""
    import zarr

    arr = _canonicalize(arr)
    path = os.path.join(tmp_dir, f"{name}.zarr")
    chunks = tuple(max(1, int(s)) for s in arr.shape) if arr.ndim else None
    z = zarr.open_array(path, mode="w", shape=arr.shape, dtype=arr.dtype, chunks=chunks)
    if arr.size:
        z[:] = arr
    return as_source(z).to_spec()


def _capture_graph(graph, job_type: str, tmp_dir: Optional[str]):
    """Capture the graph for the per-block closure: the live object for local, its edges' spec else."""
    if job_type == "local":
        return graph
    return _persist_array(np.asarray(graph.uv_ids(), dtype="uint64"), tmp_dir, "uv")


def _capture_array(arr: np.ndarray, job_type: str, tmp_dir: Optional[str], name: str):
    """Capture an array for the per-block closure: the array for local, a reopen spec otherwise."""
    if job_type == "local":
        return np.ascontiguousarray(arr)
    return _persist_array(arr, tmp_dir, name)


def _read_array(handle) -> np.ndarray:
    """Materialize a captured array handle (a numpy array for local, a spec to reopen otherwise)."""
    if isinstance(handle, np.ndarray):
        return handle
    src = from_spec(handle)
    return np.asarray(src[full_roi(src.ndim)])


def _make_solve_block(graph_handle, n_nodes: int, costs_handle, labels_handle,
                      solver_name: str, has_halo: bool, lifted_handles) -> ComputeFn:
    """Build the per-block sub-problem solver (a cloudpicklable closure over the captured handles).

    ``inputs[0]`` is the segmentation (it defines the blocking domain). The graph, costs, composed
    labels and (optional) lifted arrays are captured as handles and materialized + cached once per
    worker process. The block returns its cut edge ids (inner edges cut by the local solve, plus all
    block-boundary edges) through the return-value channel.
    """
    cache: Dict[str, object] = {}

    def _graph() -> "bic.graph.UndirectedGraph":
        g = cache.get("graph")
        if g is None:
            # Distributed: graph_handle is a SourceSpec of the persisted edges -> rebuild.
            # Local: graph_handle is the live graph object -> use directly.
            g = build_graph(int(n_nodes), _read_array(graph_handle)) \
                if isinstance(graph_handle, SourceSpec) else graph_handle
            cache["graph"] = g
            cache["uv_ids"] = np.asarray(g.uv_ids(), dtype="uint64")
        return g  # type: ignore[return-value]

    def _cached(key: str, handle) -> np.ndarray:
        val = cache.get(key)
        if val is None:
            val = _read_array(handle)
            cache[key] = val
        return val  # type: ignore[return-value]

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> np.ndarray:
        seg_src = inputs[0]
        roi = to_roi(block.outer_block) if has_halo else to_roi(block)
        node_ids = np.unique(seg_src[roi]).astype("uint64")
        if labels_handle is not None:
            labels = _cached("labels", labels_handle)
            node_ids = np.unique(labels[node_ids]).astype("uint64")

        graph = _graph()
        uv_ids = cache["uv_ids"]  # type: ignore[assignment]
        inner_edges, outer_edges = graph.extract_subgraph_from_nodes(node_ids)
        inner_edges = np.asarray(inner_edges)
        outer_edges = np.asarray(outer_edges).astype("uint64")
        if inner_edges.size == 0:
            return outer_edges

        sub_uvs = np.asarray(uv_ids)[inner_edges]
        _, max_id, mapping = _relabel_from_zero(node_ids)
        sub_uvs = bic.utils.take_dict(mapping, np.ascontiguousarray(sub_uvs, dtype="uint64"))
        sub_graph = build_graph(max_id + 1, sub_uvs)
        sub_costs = _cached("costs", costs_handle)[inner_edges]

        if lifted_handles is None:
            sub_result = multicut_solver_by_name(solver_name)(sub_graph, sub_costs)
        else:
            luv_handle, lcosts_handle = lifted_handles
            lifted_uv = _cached("luv", luv_handle)
            inner_lifted = find_inner_lifted_edges(lifted_uv, node_ids)
            if inner_lifted.size:
                sub_lifted_uvs = bic.utils.take_dict(
                    mapping, np.ascontiguousarray(lifted_uv[inner_lifted], dtype="uint64"),
                )
                sub_lifted_costs = _cached("lcosts", lcosts_handle)[inner_lifted]
                sub_result = lifted_solver_by_name(solver_name)(
                    sub_graph, sub_costs, sub_lifted_uvs, sub_lifted_costs,
                )
            else:
                sub_result = multicut_solver_by_name(solver_name)(sub_graph, sub_costs)

        sub_result = np.asarray(sub_result)
        cut = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]
        return np.concatenate([inner_edges[cut].astype("uint64"), outer_edges])

    return _compute


def solve_subproblems_distributed(runner, graph, costs: np.ndarray, segmentation_src: Source,
                                  labels: Optional[np.ndarray], *, block_shape: Tuple[int, ...],
                                  halo: Optional[Sequence[int]], internal_solver: str,
                                  num_workers: int, job_type: str, tmp_dir: Optional[str],
                                  name: str, lifted_uv_ids: Optional[np.ndarray] = None,
                                  lifted_costs: Optional[np.ndarray] = None) -> np.ndarray:
    """Run the per-block sub-problem solves (map) and reduce them to a base-edge merge mask.

    Returns a boolean mask over ``graph`` edges: ``True`` where the edge is kept (merged), ``False``
    where it is cut by at least one block (block-boundary edges are always cut at the current level).
    """
    n_edges = int(graph.number_of_edges)
    has_halo = halo is not None
    level_dir: Optional[str] = None
    if job_type != "local":
        level_dir = tempfile.mkdtemp(dir=tmp_dir)

    graph_handle = _capture_graph(graph, job_type, level_dir)
    costs_handle = _capture_array(np.asarray(costs), job_type, level_dir, "costs")
    labels_handle = None if labels is None \
        else _capture_array(np.asarray(labels), job_type, level_dir, "labels")
    lifted_handles = None
    if lifted_uv_ids is not None:
        luv = np.asarray(lifted_uv_ids, dtype="uint64").reshape(-1, 2)
        lifted_handles = (
            _capture_array(luv, job_type, level_dir, "luv"),
            _capture_array(np.asarray(lifted_costs, dtype="float64"), job_type, level_dir, "lcosts"),
        )

    compute = _make_solve_block(graph_handle, int(graph.number_of_nodes), costs_handle,
                                labels_handle, internal_solver, has_halo, lifted_handles)
    results: List[Optional[np.ndarray]] = runner.run(
        compute, [segmentation_src], outputs=[], block_shape=block_shape, halo=halo,
        num_workers=num_workers, has_return_val=True, name=name,
    )

    cut = np.zeros(n_edges, dtype=bool)
    for res in results:
        if res is None:
            continue
        res = np.asarray(res)
        if res.size:
            cut[res.astype("int64")] = True
    return ~cut
