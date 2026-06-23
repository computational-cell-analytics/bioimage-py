"""Block-wise contingency table (sparse segmentation overlap) via the runner's return channel.

The contingency table counts, for every pair of labels ``(a, b)``, the number of pixels assigned to
``a`` in the first segmentation and ``b`` in the second. It is the shared primitive underlying nearly
every segmentation-comparison metric (variation of information, rand index, cremi score, object VI,
object matching, symmetric best dice).

It is a reduction operation: each block computes its sparse overlap counts and the main process sums
them into one table. The overlap counts are additive across blocks with no halo, because the blocks
partition the volume disjointly. This mirrors the ``morphology`` / ``stats`` ops — the per-block tables
flow through ``runner.run(..., has_return_val=True)`` and the merge is pure numpy — so it behaves
identically across the ``local`` / ``subprocess`` / ``slurm`` backends. The per-block counting itself is
delegated to :func:`bioimage_cpp.utils.segmentation_overlap`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import bioimage_cpp as bic

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, check_direct, check_rerun_args, full_roi, to_roi

__all__ = ["ContingencyTable", "contingency_table"]


@dataclass(frozen=True)
class ContingencyTable:
    """The sparse contingency table between two segmentations.

    All arrays use ``uint64`` (lossless for label ids and counts). Background (label ``0``) is kept;
    ignoring it is a metric-level concern handled by the callers of this primitive.

    Attributes:
        pairs: The ``(N, 2)`` array of co-occurring label pairs ``[label_a, label_b]``, sorted
            lexicographically by ``(label_a, label_b)``.
        counts: The ``(N,)`` overlap count for each pair in `pairs`.
        labels_a: The ``(Ka,)`` sorted unique labels present in the first segmentation.
        sizes_a: The ``(Ka,)`` size (pixel count) of each label in `labels_a`.
        labels_b: The ``(Kb,)`` sorted unique labels present in the second segmentation.
        sizes_b: The ``(Kb,)`` size (pixel count) of each label in `labels_b`.
        n_points: The total number of pixels counted (after any masking).
    """

    pairs: np.ndarray
    counts: np.ndarray
    labels_a: np.ndarray
    sizes_a: np.ndarray
    labels_b: np.ndarray
    sizes_b: np.ndarray
    n_points: int

    def as_dicts(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Return the per-label sizes as ``({label_a: size}, {label_b: size})`` dictionaries.

        Returns:
            A dictionary mapping each label in the first segmentation to its size, and the analogous
            dictionary for the second segmentation.
        """
        a_dict = {int(lab): int(cnt) for lab, cnt in zip(self.labels_a, self.sizes_a)}
        b_dict = {int(lab): int(cnt) for lab, cnt in zip(self.labels_b, self.sizes_b)}
        return a_dict, b_dict

    def drop_ignore(self, ignore_a: Optional[Sequence[int]] = None,
                    ignore_b: Optional[Sequence[int]] = None) -> "ContingencyTable":
        """Return a copy with the voxels of the given ignore labels removed.

        A pair is dropped if its A-label is in ``ignore_a`` **or** its B-label is in ``ignore_b`` (the
        marginal sizes and ``n_points`` are recomputed over what remains). This is equivalent to
        excluding those voxels before counting. Passing ``None`` (or an empty sequence) for a side is a
        no-op for that side; dropping every present label yields the empty table.

        Args:
            ignore_a: Labels to ignore in the first segmentation.
            ignore_b: Labels to ignore in the second segmentation.

        Returns:
            A new `ContingencyTable` without the ignored voxels.
        """
        if ignore_a is None and ignore_b is None:
            return self
        drop = np.zeros(self.pairs.shape[0], dtype=bool)
        if ignore_a is not None:
            drop |= np.isin(self.pairs[:, 0], np.asarray(list(ignore_a), dtype=self.pairs.dtype))
        if ignore_b is not None:
            drop |= np.isin(self.pairs[:, 1], np.asarray(list(ignore_b), dtype=self.pairs.dtype))
        keep = ~drop
        return _table_from_pairs(self.pairs[keep], self.counts[keep])


def _overlap_rows(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Compute the sparse overlap counts of two equally-shaped label arrays.

    Returns a ``(K, 3)`` uint64 array with columns ``[label_a, label_b, count]`` (plain rather than the
    structured ``overlap_table`` so the merge can ``vstack`` / ``lexsort`` / ``reduceat`` directly), or
    ``None`` if there is no overlap (e.g. empty inputs).
    """
    table = bic.utils.segmentation_overlap(a, b).overlap_table()
    if table.shape[0] == 0:
        return None
    return np.stack([table["label_a"], table["label_b"], table["count"]], axis=1).astype("uint64")


def _merge_tables(tables: List[np.ndarray]) -> ContingencyTable:
    """Merge per-block overlap rows into one :class:`ContingencyTable`.

    Groups the stacked ``[label_a, label_b, count]`` rows by their ``(label_a, label_b)`` pair and sums
    the counts (single ``lexsort`` + ``reduceat``, mirroring ``morphology._merge_tables``). Marginals are
    derived from the merged pairs, since every pixel maps to exactly one pair.

    Args:
        tables: The non-``None`` per-block ``(_, 3)`` uint64 arrays.

    Returns:
        The merged contingency table.
    """
    if not tables:
        return _table_from_pairs(np.zeros((0, 2), "uint64"), np.zeros((0,), "uint64"))

    stacked = np.vstack(tables)
    order = np.lexsort((stacked[:, 1], stacked[:, 0]))  # sort by label_a, then label_b.
    stacked = stacked[order]
    a, b = stacked[:, 0], stacked[:, 1]
    starts = np.flatnonzero(np.concatenate(([True], (a[1:] != a[:-1]) | (b[1:] != b[:-1]))))
    pairs = stacked[starts][:, :2].copy()
    counts = np.add.reduceat(stacked[:, 2], starts)
    return _table_from_pairs(pairs, counts)


def _table_from_pairs(pairs: np.ndarray, counts: np.ndarray) -> ContingencyTable:
    """Build a :class:`ContingencyTable` from merged, ``(a, b)``-sorted unique pairs and their counts.

    Derives the per-label marginals (sum over the other label) and ``n_points`` from the pairs. The
    caller must pass unique pairs whose first column is sorted ascending — true of
    :func:`_merge_tables`'s output, and preserved by the boolean row-selection in
    :meth:`ContingencyTable.drop_ignore`.

    Args:
        pairs: The ``(N, 2)`` uint64 ``[label_a, label_b]`` pairs (unique, sorted by ``(a, b)``).
        counts: The ``(N,)`` uint64 overlap count for each pair.

    Returns:
        The contingency table (empty when ``pairs`` has no rows).
    """
    if pairs.shape[0] == 0:
        empty = np.zeros((0,), "uint64")
        return ContingencyTable(np.zeros((0, 2), "uint64"), empty, empty, empty, empty, empty, 0)

    # Marginal for a: pairs[:, 0] is already sorted ascending, so group it directly.
    a_col = pairs[:, 0]
    a_starts = np.flatnonzero(np.concatenate(([True], a_col[1:] != a_col[:-1])))
    labels_a = a_col[a_starts].copy()
    sizes_a = np.add.reduceat(counts, a_starts)

    # Marginal for b: re-sort by label_b (stable, to keep grouping deterministic), then group.
    b_order = np.argsort(pairs[:, 1], kind="stable")
    b_sorted = pairs[:, 1][b_order]
    b_counts = counts[b_order]
    b_starts = np.flatnonzero(np.concatenate(([True], b_sorted[1:] != b_sorted[:-1])))
    labels_b = b_sorted[b_starts].copy()
    sizes_b = np.add.reduceat(b_counts, b_starts)

    return ContingencyTable(pairs, counts, labels_a, sizes_a, labels_b, sizes_b, int(counts.sum()))


def _contingency_block(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                       mask: Optional[Source]) -> Optional[np.ndarray]:
    """Per-block overlap rows (``None`` if the block is fully masked out or has no overlap)."""
    roi = to_roi(block)
    a = inputs[0][roi]
    b = inputs[1][roi]
    if mask is not None:
        block_mask = mask[roi].astype(bool)
        if not block_mask.any():
            return None
        a, b = a[block_mask], b[block_mask]  # 1D, same length; segmentation_overlap accepts this.
    return _overlap_rows(a, b)


def contingency_table(
    seg_a: SourceLike,
    seg_b: SourceLike,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
    pre_cleanup: Optional[Callable[[str], None]] = None,
) -> ContingencyTable:
    """Compute the contingency table (sparse overlap counts) between two segmentations.

    The two segmentations are compared pixel-by-pixel and the overlap counts are accumulated
    block-wise, so the result is exact regardless of how labels straddle block boundaries. The pairing
    is symmetric: which input is the candidate and which is the ground truth is up to the caller.
    Background (label ``0``) is included; ignoring labels is left to the metrics built on top of this.

    Args:
        seg_a: The first segmentation (a numpy/zarr/n5 array or a `Source`); must be integer-typed.
        seg_b: The second segmentation; must be integer-typed and the same shape as `seg_a`.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape;
            required for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        mask: Optional binary mask; pixels outside the mask are excluded from the counts.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks);
            the table then reflects only those blocks.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume and
            merge (see ``runner.run``). The returned table then covers the full volume (the
            already-completed blocks merged with the re-run ones). Mutually exclusive with
            ``block_ids``.
        pre_cleanup: Optional ``pre_cleanup(tmp_folder)`` callback invoked on the orchestrating
            process with the job temp folder right before it is deleted (distributed backends only).
            Ignored for the ``local`` backend and for the direct (single-worker, unchunked) path.

    Returns:
        The merged `ContingencyTable`.
    """
    check_rerun_args(job_type, resume_from, block_ids)
    src_a = as_source(seg_a)
    src_b = as_source(seg_b)
    for name, src in (("seg_a", src_a), ("seg_b", src_b)):
        if not np.issubdtype(np.dtype(src.dtype), np.integer):
            raise ValueError(f"contingency_table expects integer label images, got dtype {src.dtype} "
                             f"for {name}.")

    if check_direct(job_type, num_workers, block_shape, mask, block_ids):
        table = _overlap_rows(src_a[full_roi(src_a.ndim)], src_b[full_roi(src_b.ndim)])
        tables = [table] if table is not None else []
    else:
        runner = get_runner(job_type, job_config)
        results = runner.run(_contingency_block, [seg_a, seg_b], num_workers=num_workers,
                             block_shape=block_shape, mask=mask, block_ids=block_ids,
                             resume_from=resume_from, has_return_val=True, name="contingency_table",
                             pre_cleanup=pre_cleanup)
        tables = [r for r in results if r is not None]

    return _merge_tables(tables)
