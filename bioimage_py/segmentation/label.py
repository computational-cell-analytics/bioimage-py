"""Block-wise connected-component labeling (multi-stage: label -> merge -> relabel).

The block-wise path runs three :meth:`~bioimage_py.runner.base.Runner.run` calls plus one
in-process reduction, with the labeled volume persisted in the ``output`` source between
stages. This mirrors cluster_tools' connected-components workflow.

Inter-stage data visibility (distributed/slurm): stage N+1 is only launched after every stage-N
task's ``.success`` sentinel is visible, i.e. after all stage-N workers have exited and flushed
their writes. Stage N+1 then reads the output's chunks by direct path (zarr/n5 address a chunk by
key, with no directory listing), so NFS close-to-open consistency guarantees it sees the fresh
data -- the attribute-cache lag that affects sentinel *discovery* does not apply to reads of an
already-named chunk file.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, ComputeFn, get_blocking, to_roi

__all__ = ["label"]


def _full_roi(ndim: int) -> Tuple[slice, ...]:
    """Return a slicing that selects the whole array."""
    return tuple(slice(None) for _ in range(ndim))


def _binarize(data: np.ndarray, threshold: Optional[float]) -> np.ndarray:
    """Binarize ``data`` by threshold, or interpret it as a boolean foreground mask."""
    if threshold is not None:
        return data > threshold
    return data if data.dtype == np.dtype(bool) else data.astype(bool)


def _resolve_block_shape(src: Source, out: Source,
                         block_shape: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Resolve the block shape from the explicit value or the input/output chunks."""
    if block_shape is not None:
        return tuple(int(b) for b in block_shape)
    chunks = src.chunks or out.chunks
    if chunks is None:
        raise ValueError("block_shape is required for block-wise labeling of an unchunked array.")
    return tuple(int(c) for c in chunks)


# --- per-block stage functions (built as closures, capturing only picklable values) ----

def _make_stage1(shape: Tuple[int, ...], block_shape: Tuple[int, ...], connectivity: int,
                 threshold: Optional[float], offset_factor: int) -> ComputeFn:
    """Build stage 1: label each block independently and apply a globally-unique offset."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        input_, output_ = inputs[0], outputs[0]
        roi = to_roi(block)
        blocking = get_blocking(shape, block_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.begin])

        binary = _binarize(input_[roi], threshold)
        if mask is not None:
            binary = binary & mask[roi].astype(bool)

        if not binary.any():
            output_[roi] = np.zeros(binary.shape, dtype="uint64")
            return None

        comp = bic.segmentation.label(binary, connectivity=connectivity).astype("uint64", copy=False)
        offset = np.uint64(int(block_id) * int(offset_factor))
        comp[comp != 0] += offset
        output_[roi] = comp
        # Return the block's actual (globally-unique) labels so stage 3 can relabel only
        # over labels that exist, not over the sparse offset space.
        return np.unique(comp[comp != 0])

    return _compute


def _make_stage2(shape: Tuple[int, ...], block_shape: Tuple[int, ...]) -> ComputeFn:
    """Build stage 2: collect label equivalences across lower block faces."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> Optional[np.ndarray]:
        output_ = inputs[0]  # the labeled volume, passed as a read-only input
        ndim = len(shape)
        blocking = get_blocking(shape, block_shape)
        block_id = blocking.coordinates_to_block_id([int(c) for c in block.begin])

        pairs = []
        block_roi = to_roi(block)
        for axis in range(ndim):
            if blocking.get_neighbor_id(block_id, axis, True) == -1:  # no lower neighbor
                continue
            b0 = int(block.begin[axis])
            slab_roi = list(block_roi)
            slab_roi[axis] = slice(b0 - 1, b0 + 1)  # 2-thick slab straddling the boundary
            slab = output_[tuple(slab_roi)]
            lo = tuple(slice(0, 1) if d == axis else slice(None) for d in range(ndim))
            hi = tuple(slice(1, 2) if d == axis else slice(None) for d in range(ndim))
            labels_b = np.squeeze(slab[lo], axis=axis)  # neighbor side
            labels_a = np.squeeze(slab[hi], axis=axis)  # this block side
            keep = (labels_a != 0) & (labels_b != 0)
            if keep.any():
                pairs.append(np.stack([labels_a[keep], labels_b[keep]], axis=1).astype("uint64"))

        if not pairs:
            return None
        return np.unique(np.concatenate(pairs, axis=0), axis=0)

    return _compute


def _make_stage4(mapping: Dict[int, int]) -> ComputeFn:
    """Build stage 4: apply the final label mapping in place."""

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        output_ = outputs[0]
        roi = to_roi(block)
        output_[roi] = bic.utils.take_dict(mapping, output_[roi])
        return None

    return _compute


def label(
    input: SourceLike,
    output: Optional[SourceLike] = None,
    *,
    threshold: Optional[float] = None,
    connectivity: Optional[int] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
) -> SourceLike:
    """Label connected components of (optionally thresholded) data, block-wise.

    Unlike the single-pass operations, ``label`` is multi-stage with a global cross-block merge
    (per-block labeling, then a union-find over touching components across block faces), so it
    does **not** accept ``block_ids`` or ``resume_from``: a failed run must be re-run whole (it is
    idempotent given the same ``output``).

    Args:
        input: The input data (a numpy/zarr/n5 array or a `Source`).
        output: The ``uint64`` output array to write the labels into. Optional for local
            execution — a numpy array is allocated and returned if omitted; **required** for
            distributed execution.
        threshold: If given, the input is binarized as ``input > threshold``; otherwise the
            input is treated as a binary foreground mask.
        connectivity: Neighbour connectivity in ``[1, ndim]`` (``1`` = orthogonal). Defaults
            to ``1``; values ``> 1`` are only supported for the direct (single-block) path.
        block_shape: Shape of the processing blocks. Defaults to the input/output chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; values outside the mask are excluded from the foreground.

    Returns:
        The output array (the provided ``output``, or a newly allocated numpy array), labeled
        with consecutive ids (background stays ``0``).
    """
    src = as_source(input)
    ndim = src.ndim
    conn = 1 if connectivity is None else int(connectivity)
    if not 1 <= conn <= ndim:
        raise ValueError(f"connectivity must be in [1, {ndim}], got {conn}.")

    direct = job_type == "local" and num_workers == 1 and block_shape is None and mask is None
    if conn > 1 and not direct:
        raise NotImplementedError(
            "Block-wise labeling only supports connectivity=1 (orthogonal). Use the direct "
            "path (local, single worker, no block_shape, no mask) for higher connectivity."
        )

    if output is None:
        if job_type != "local":
            raise ValueError(
                f"'output' is required for distributed execution (job_type={job_type!r}); "
                "pass a file-backed (zarr/n5) output array."
            )
        out_array: SourceLike = np.zeros(tuple(src.shape), dtype="uint64")
    else:
        out_array = output

    out = as_source(out_array)
    if out.dtype != np.dtype("uint64"):
        raise ValueError(f"output must have dtype uint64, got {out.dtype}.")

    if direct:
        binary = _binarize(src[_full_roi(ndim)], threshold)
        comp = bic.segmentation.label(binary, connectivity=conn).astype("uint64", copy=False)
        out[_full_roi(ndim)] = comp
        return out_array

    block_shape = _resolve_block_shape(src, out, block_shape)
    offset_factor = int(np.prod(block_shape))
    blocking = get_blocking(src.shape, block_shape)
    n_blocks = int(blocking.number_of_blocks)
    if (n_blocks * offset_factor) >= int(np.iinfo(np.uint64).max):
        raise ValueError(
            "Label id overflow: number_of_blocks * prod(block_shape) exceeds uint64. "
            "Reduce the block shape or the volume size."
        )

    runner = get_runner(job_type, job_config)

    # Stage 1: label each block independently with a globally-unique offset.
    stage1 = _make_stage1(tuple(src.shape), block_shape, conn, threshold, offset_factor)
    id_results = runner.run(stage1, [input], outputs=[out_array], block_shape=block_shape,
                            mask=mask, num_workers=num_workers, has_return_val=True,
                            name="label-blocks")
    id_arrays = [a for a in id_results if a is not None and len(a)]
    real_labels = np.unique(np.concatenate(id_arrays)) if id_arrays else np.zeros((0,), dtype="uint64")
    max_label = int(real_labels.max()) if real_labels.size else 0

    # Stage 2: collect label equivalences across lower block faces.
    stage2 = _make_stage2(tuple(src.shape), block_shape)
    pair_results = runner.run(stage2, [out_array], block_shape=block_shape,
                              num_workers=num_workers, has_return_val=True, name="merge-faces")
    pairs = [p for p in pair_results if p is not None]
    assignments = (np.unique(np.concatenate(pairs, axis=0), axis=0)
                   if pairs else np.zeros((0, 2), dtype="uint64"))

    # Stage 3 (in process): union-find merge, then relabel only the labels that exist to
    # consecutive ids (the offset space is sparse, so relabeling over it would not be compact).
    uf = bic.utils.UnionFind(max_label + 1)
    if len(assignments):
        uf.merge(assignments.astype("uint64"))
    mapping: Dict[int, int] = {0: 0}
    if real_labels.size:
        roots = np.asarray(uf.find(real_labels.astype("uint64")))
        root_to_new = {int(r): i + 1 for i, r in enumerate(np.unique(roots).tolist())}
        for lab, root in zip(real_labels.tolist(), roots.tolist()):
            mapping[int(lab)] = root_to_new[int(root)]

    # Stage 4: apply the mapping in place.
    stage4 = _make_stage4(mapping)
    runner.run(stage4, [], outputs=[out_array], block_shape=block_shape,
               num_workers=num_workers, has_return_val=False, name="relabel")
    return out_array
