"""Block-wise relabeling of a segmentation.

Two operations live here:

- :func:`relabel` applies an explicit, caller-provided relabeling (a *labeling*) to a segmentation:
  a mapping from each old segment id to a new id (e.g. the node-to-label result of a graph-based
  segmentation such as multicut, or a merge/split assignment). It is a single-stage, disjoint
  per-block point op with no halo, so it accepts ``block_ids`` / ``resume_from`` and may be applied
  **in place** (the default when ``output`` is omitted). The labeling is applied per block with
  ``bioimage_cpp.utils.take_dict`` (when given as a ``dict``) or with ``numpy.take`` (when given as a
  dense 1D array/source where ``labeling[old_id]`` is the new id); a dense array passed as an
  in-memory numpy array for a distributed job is persisted to a temporary zarr under the runner's
  temp root so worker tasks can reopen it.

- :func:`relabel_consecutive` relabels a segmentation so its ids are consecutive. It *derives* the
  mapping from the data (a global ``unique`` reduction), then delegates the block-wise write to
  :func:`relabel`. Being multi-stage, it re-runs whole (it does not accept ``block_ids`` /
  ``resume_from``); like :func:`relabel` it relabels in place when ``output`` is omitted.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, SourceSpec, as_source, from_spec
from ..stats.unique import unique
from ..util import BlockDescriptor, ComputeFn, check_rerun_args, full_roi, is_direct, to_roi

__all__ = ["relabel", "relabel_consecutive"]

# Gated per-block subsampling for the dict kernel. ``bioimage_cpp.utils.take_dict`` rebuilds a hash
# map from the *full* mapping on every call (O(dict size) per block, regardless of how few ids the
# block holds), so for a large dict we first restrict it to the ids actually present in the block.
# Benchmarked crossovers: the full rebuild is <10 ms below ~1e5 entries (not worth subsampling), and
# subsampling only pays off while the block's distinct-id count is well below the dict size -- for a
# nearly-as-diverse-as-the-dict block, building the per-block dict costs more than the rebuild.
_RELABEL_SUBSAMPLE_MIN_DICT = 100_000
_RELABEL_SUBSAMPLE_MAX_DIVERSITY = 8


def _require_integer(source: Source, message: str) -> None:
    """Raise ``ValueError`` unless ``source`` has an integer dtype."""
    if not np.issubdtype(np.dtype(source.dtype), np.integer):
        raise ValueError(f"{message}, got dtype {source.dtype}.")


def _is_inmemory_numpy(source: Source) -> bool:
    """Return whether ``source`` wraps a plain in-memory numpy array (local-only, not reopenable)."""
    return isinstance(getattr(source, "array", None), np.ndarray)


def _take_mapping(mapping: Dict[int, int], seg: np.ndarray, subsample: bool) -> np.ndarray:
    """Map ``seg`` through ``mapping``, restricting the dict to the block's ids when that is cheaper.

    ``take_dict`` rebuilds a hash map from the whole ``mapping`` per call, so for a large dict we
    subsample it to the ids present in ``seg`` -- but only while they are far fewer than the dict
    (otherwise the per-block dict costs more than the rebuild it saves).
    """
    if subsample:
        present = np.unique(seg)
        if len(present) * _RELABEL_SUBSAMPLE_MAX_DIVERSITY < len(mapping):
            mapping = {int(x): mapping[int(x)] for x in present.tolist()}
    return bic.utils.take_dict(mapping, seg)


def _make_relabel_block(mapping: Dict[int, int]) -> ComputeFn:
    """Build the per-block write function applying the label mapping (picklable dict)."""
    subsample = len(mapping) > _RELABEL_SUBSAMPLE_MIN_DICT

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        input_, output_ = inputs[0], outputs[0]
        roi = to_roi(block)
        seg = input_[roi]
        if mask is None:
            output_[roi] = _take_mapping(mapping, seg, subsample)
            return None
        m = mask[roi].astype(bool)
        if not m.any():
            return None
        # Only in-mask voxels are in the mapping; out-of-mask output voxels are left unchanged.
        out_block = output_[roi].copy()
        out_block[m] = _take_mapping(mapping, seg[m], subsample)
        output_[roi] = out_block
        return None

    return _compute


def _resume_placeholder(block: BlockDescriptor, inputs: Sequence[Source],
                        outputs: Sequence[Source], mask: Optional[Source]) -> None:
    """No-op compute fn for the resume path: ``run(resume_from=...)`` never calls it."""
    return None


def _make_cleanup(tmp_dir: str) -> Callable[[str], None]:
    """Build a success-path callback that removes the persisted labeling temp dir."""

    def _cleanup(_tmp: str) -> None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return _cleanup


def _persist_labeling(labeling: np.ndarray, tmp_root: Optional[str]) -> Tuple[SourceSpec, str]:
    """Persist a 1D labeling array to a temp zarr so distributed workers can reopen it.

    Args:
        labeling: The dense 1D labeling array to persist.
        tmp_root: The runner temp root to create the temp dir under (shared filesystem for slurm);
            ``None`` uses the system default.

    Returns:
        A ``(spec, tmp_dir)`` tuple: the reopen spec of the persisted array and the temp directory
        (removed on the run's success path, preserved on failure so ``resume_from`` can reopen it).
    """
    import zarr

    tmp_dir = tempfile.mkdtemp(prefix="bioimage_py_labeling_", dir=tmp_root)
    path = os.path.join(tmp_dir, "labeling.zarr")
    array = zarr.open_array(path, mode="w", shape=labeling.shape, dtype=labeling.dtype,
                            chunks=labeling.shape)
    array[:] = labeling
    return as_source(array).to_spec(), tmp_dir


def _make_take_array_block(labeling: Optional[np.ndarray],
                           spec: Optional[SourceSpec]) -> ComputeFn:
    """Build the per-block write function mapping ids through a dense 1D labeling array.

    Exactly one of ``labeling`` (captured directly, for local execution) and ``spec`` (reopened on
    the worker, for distributed execution) is set; the labeling is read once and cached per worker.
    """
    cache: Dict[str, np.ndarray] = {}

    def _labeling() -> np.ndarray:
        labels = cache.get("labeling")
        if labels is None:
            labels = labeling if labeling is not None else np.asarray(from_spec(spec)[full_roi(1)])
            cache["labeling"] = labels
        return labels

    def _compute(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                 mask: Optional[Source]) -> None:
        input_, output_ = inputs[0], outputs[0]
        roi = to_roi(block)
        seg = input_[roi]
        labels = _labeling()
        if mask is None:
            output_[roi] = np.take(labels, seg)
            return None
        m = mask[roi].astype(bool)
        if not m.any():
            return None
        # Only in-mask voxels are remapped; out-of-mask output voxels are left unchanged.
        out_block = output_[roi].copy()
        out_block[m] = np.take(labels, seg[m])
        output_[roi] = out_block
        return None

    return _compute


def relabel(
    input: SourceLike,
    labeling: Union[SourceLike, Mapping[int, int]],
    output: Optional[SourceLike] = None,
    *,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> SourceLike:
    """Apply a labeling (relabeling map) to a segmentation, block-wise.

    Each block of ``input`` is read, its ids are mapped through ``labeling``, and the result is
    written to ``output`` (or back to ``input`` when ``output`` is omitted). This is a single-stage,
    disjoint per-block point operation, so it may be applied in place and supports ``block_ids`` /
    ``resume_from`` re-runs.

    Args:
        input: The input segmentation (a numpy/zarr/n5 array or a `Source`); must be integer-typed.
        labeling: The relabeling to apply. Either a ``dict`` ``{old_id: new_id}`` (applied with
            ``bioimage_cpp.utils.take_dict``; every id present in ``input`` must be a key) or a dense
            1D array/source where ``labeling[old_id]`` is the new id (applied with ``numpy.take``; it
            must be long enough to index every id present in ``input``). A dict must map every id of
            the (masked) input; a dense array must cover the id range ``0 .. max_id``.
        output: The output array to write the relabeled segmentation into. Optional -- when omitted
            the relabeling is applied **in place** to ``input`` (which must then be writable, and
            file-backed for distributed execution). As an exception, a plain in-memory numpy input
            (which is local-only) is never mutated: a fresh array is allocated and returned. When
            given, ``output`` must match the input shape; ids are cast to its dtype on write.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required for
            unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; only voxels within the mask are relabeled (out-of-mask output
            voxels are left unchanged).
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks
            into the existing ``output``). Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``); the missing blocks are relabeled using the original run's labeling.
            Mutually exclusive with ``block_ids``. (A numpy labeling array persisted by the failed
            run is preserved with that temp folder; after a successful resume it is best-effort left
            behind since the labeling is small.)

    Returns:
        The output array: the provided ``output`` if given, else ``input`` itself when relabeling a
        file-backed source in place, or a freshly allocated array for an in-memory numpy input.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src = as_source(input)
    _require_integer(src, "relabel expects an integer label image")
    ndim = src.ndim

    # Resolve the output. By default relabel in place (a disjoint per-block point op with no halo,
    # so this is safe). Exception: a plain in-memory numpy input is local-only (distributed rejects
    # it), and silently mutating a passed-in array is surprising -- so allocate a fresh copy instead.
    if output is not None:
        out_array: SourceLike = output
    elif job_type == "local" and _is_inmemory_numpy(src):
        out_array = np.array(src.array)
    else:
        out_array = input
    out = as_source(out_array)

    runner = get_runner(job_type, job_config)

    # Resume short-circuits inside run() to the preserved run's payload (its own sources and
    # labeling), so the function/inputs passed here are placeholders that run() does not use.
    if resume_from is not None:
        runner.run(_resume_placeholder, [src], outputs=[out], resume_from=resume_from,
                   name="relabel")
        return out_array

    direct = is_direct(job_type, num_workers, block_shape) and mask is None and block_ids is None

    # Dict mode: apply the mapping with take_dict.
    if isinstance(labeling, Mapping):
        mapping = dict(labeling)
        if direct:
            out[full_roi(ndim)] = bic.utils.take_dict(mapping, src[full_roi(ndim)])
            return out_array
        runner.run(_make_relabel_block(mapping), [src], outputs=[out], block_shape=block_shape,
                   mask=mask, num_workers=num_workers, block_ids=block_ids, name="relabel")
        return out_array

    # Dense 1D labeling array/source: applied with numpy.take.
    labeling_src = as_source(labeling)
    if labeling_src.ndim != 1:
        raise ValueError(f"Dense labeling must be a 1D array; got shape {labeling_src.shape}.")
    _require_integer(labeling_src, "labeling must be integer-typed")

    if direct:
        labels = np.asarray(labeling_src[full_roi(1)])
        out[full_roi(ndim)] = np.take(labels, src[full_roi(ndim)])
        return out_array

    # Carry the labeling in the (cloudpickled) closure: the array directly for local execution, or
    # a reopen spec for distributed workers -- persisting an in-memory numpy array to a temp zarr.
    labeling_array: Optional[np.ndarray] = None
    spec: Optional[SourceSpec] = None
    pre_cleanup: Optional[Callable[[str], None]] = None
    if job_type == "local":
        labeling_array = np.asarray(labeling_src[full_roi(1)])
    else:
        try:
            spec = labeling_src.to_spec()
        except ValueError:  # an in-memory numpy array: persist it so worker tasks can reopen it.
            spec, tmp_dir = _persist_labeling(np.asarray(labeling_src[full_roi(1)]),
                                              runner.config.tmp_root)
            pre_cleanup = _make_cleanup(tmp_dir)

    runner.run(_make_take_array_block(labeling_array, spec), [src], outputs=[out],
               block_shape=block_shape, mask=mask, num_workers=num_workers, block_ids=block_ids,
               pre_cleanup=pre_cleanup, name="relabel")
    return out_array


def relabel_consecutive(
    input: SourceLike,
    output: Optional[SourceLike] = None,
    *,
    start_label: int = 0,
    keep_zeros: bool = True,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
) -> Tuple[SourceLike, int, Dict[int, int]]:
    """Relabel a segmentation to consecutive ids, block-wise.

    This is multi-stage: a global ``unique`` reduction derives the ``{old_id: new_id}`` mapping, then
    the block-wise write is delegated to :func:`relabel`. Because of the reduction it does **not**
    accept ``block_ids`` or ``resume_from``: a failed run is re-run whole (it is idempotent given the
    same ``output``).

    Args:
        input: The input label image (a numpy/zarr/n5 array or a `Source`); must be integer-typed.
        output: The output array to write the relabeled segmentation into. Optional -- when omitted
            the relabeling is applied **in place** to ``input`` (which must then be writable, and
            file-backed for distributed execution); a plain in-memory numpy input (local-only) is
            never mutated -- a fresh array is allocated and returned. When given, it must match the
            input shape.
        start_label: The value the smallest unique id is mapped to (subsequent ids follow
            consecutively).
        keep_zeros: Whether to always keep ``0`` mapped to ``0`` (background), regardless of
            ``start_label``.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; values outside the mask are excluded from the computation and
            their output voxels are left unchanged.

    Returns:
        A ``(output, max_id, mapping)`` tuple: the relabeled output array (``input`` itself when
        relabeling a file-backed source in place, or a freshly allocated array for a numpy input),
        the maximum label id after relabeling, and the ``{old_id: new_id}`` mapping that was applied.

    """
    src = as_source(input)
    _require_integer(src, "relabel_consecutive expects an integer label image")
    ndim = src.ndim

    # Pass 1: the global set of unique values.
    direct = is_direct(job_type, num_workers, block_shape) and mask is None
    if direct:
        uniques = np.unique(src[full_roi(ndim)])
    else:
        uniques = unique(input, block_shape=block_shape, job_type=job_type, job_config=job_config,
                         num_workers=num_workers, mask=mask)

    # In-process: build the old -> new mapping (consecutive ids from start_label).
    mapping: Dict[int, int] = {int(v): i for i, v in enumerate(uniques.tolist(), start_label)}
    if keep_zeros and 0 in mapping:
        mapping[0] = 0
    max_id = max(mapping.values()) if mapping else 0

    # Pass 2: apply the mapping (in place when output is omitted), reusing relabel's dict path.
    out = relabel(input, mapping, output, block_shape=block_shape, job_type=job_type,
                  job_config=job_config, num_workers=num_workers, mask=mask)
    return out, max_id, mapping
