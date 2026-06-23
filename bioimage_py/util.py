"""Shared helpers: block-to-roi conversion, blocking construction and filter halos."""
from __future__ import annotations

import itertools
import numbers
import warnings
from math import ceil
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import bioimage_cpp as bic
from bioimage_cpp.utils import Block, BlockWithHalo, Blocking

from .sources.base import Source

if TYPE_CHECKING:
    from .sources import SourceLike

# A per-block descriptor handed to compute functions: a plain ``Block`` (no halo) or a
# ``BlockWithHalo`` (halo operations).
BlockDescriptor = Union[Block, BlockWithHalo]

# Signature of a per-block compute function: ``function(block, inputs, outputs, mask)``.
ComputeFn = Callable[
    [BlockDescriptor, Sequence[Source], Sequence[Source], Optional[Source]], Any
]


def to_roi(block: BlockDescriptor) -> Tuple[slice, ...]:
    """Convert a ``bioimage_cpp.utils`` ``Block`` into a tuple of slices.

    Args:
        block: A ``Block`` (carrying ``begin``/``end`` coordinate lists). For halo
            operations pass one of ``block.outer_block`` / ``block.inner_block`` /
            ``block.inner_block_local``.

    Returns:
        A tuple of slices that indexes a source or array.
    """
    return tuple(slice(int(b), int(e)) for b, e in zip(block.begin, block.end))


def full_roi(ndim: int) -> Tuple[slice, ...]:
    """Return a slicing tuple that selects an entire ``ndim``-dimensional array."""
    return tuple(slice(None) for _ in range(ndim))


def is_direct(job_type: str, num_workers: int, block_shape: Optional[Tuple[int, ...]]) -> bool:
    """Return whether a call qualifies for the direct (whole-array, non-blocked) fast path."""
    return job_type == "local" and num_workers == 1 and block_shape is None


def check_direct(job_type: str, num_workers: int, block_shape: Optional[Tuple[int, ...]],
                 mask: "Optional[SourceLike]", block_ids: Optional[Sequence[int]]) -> bool:
    """Like :func:`is_direct`, but reject mask/block_ids the direct reduction path cannot honor."""
    if is_direct(job_type, num_workers, block_shape):
        if mask is not None or block_ids is not None:
            raise ValueError("Direct computation does not support 'mask' or 'block_ids'.")
        return True
    return False


def normalize_halo(halo: Union[int, Sequence[int]], ndim: int) -> List[int]:
    """Broadcast a halo to a per-axis list of length ``ndim``."""
    if isinstance(halo, numbers.Integral):
        return [int(halo)] * ndim
    halo = [int(h) for h in halo]
    if len(halo) != ndim:
        raise ValueError(f"Halo {halo} does not match ndim {ndim}.")
    return halo


def sigma_to_halo(sigma: Union[float, Sequence[float]], order: int) -> Union[int, List[int]]:
    """Compute the halo for applying an image filter block-wise.

    Mirrors elf's implementation, based on VIGRA's ``multi_blockwise.hxx``.

    Args:
        sigma: The sigma value(s) of the filter.
        order: The derivative order of the filter (0 for smoothing).

    Returns:
        The halo, as an int for scalar sigma or a per-axis list for sequence sigma.
    """
    multiplier = 2
    if isinstance(sigma, numbers.Number):
        return multiplier * int(ceil(3.0 * sigma + 0.5 * order + 0.5))
    return [multiplier * int(ceil(3.0 * sig + 0.5 * order + 0.5)) for sig in sigma]


def downscale_shape(shape: Sequence[int], scale_factor: Union[int, Sequence[int]],
                    ceil_mode: bool = True) -> Tuple[int, ...]:
    """Compute the shape resulting from downscaling by an integer factor.

    Mirrors elf's ``downscale_shape``.

    Args:
        shape: The input array shape.
        scale_factor: The downscaling factor: a single int (isotropic) or a per-axis sequence.
        ceil_mode: Whether to round the downscaled size up (so no input voxel is dropped) or
            down (strict integer division).

    Returns:
        The downscaled shape.

    Raises:
        ValueError: If a per-axis ``scale_factor`` does not match the dimensionality of ``shape``.
    """
    if isinstance(scale_factor, numbers.Integral):
        factors = [int(scale_factor)] * len(shape)
    else:
        factors = [int(f) for f in scale_factor]
        if len(factors) != len(shape):
            raise ValueError(
                f"scale_factor {scale_factor} does not match the dimensionality {len(shape)}."
            )
    if ceil_mode:
        return tuple(int(s) // f + int((int(s) % f) != 0) for s, f in zip(shape, factors))
    return tuple(int(s) // f for s, f in zip(shape, factors))


def derive_block_shape(source: Source, block_shape: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Resolve the block shape, falling back to the source's chunks.

    Args:
        source: A source exposing ``shape`` and ``chunks``.
        block_shape: The explicit block shape, or ``None`` to derive it from chunks.

    Returns:
        The resolved block shape.

    Raises:
        ValueError: If ``block_shape`` is ``None`` and the source is unchunked.
    """
    if block_shape is not None:
        return tuple(int(b) for b in block_shape)
    chunks = source.chunks
    if chunks is not None:
        return tuple(int(c) for c in chunks)
    raise ValueError(
        "block_shape is required for block-wise processing of an unchunked array "
        "(the source has no chunks to derive it from)."
    )


def get_blocking(shape: Sequence[int], block_shape: Sequence[int],
                 roi: Optional[Tuple[slice, ...]] = None) -> Blocking:
    """Build a ``bioimage_cpp.utils.Blocking`` over ``shape`` (or a sub-roi).

    Args:
        shape: The full array shape.
        block_shape: The block shape.
        roi: Optional region of interest to restrict the blocking to.

    Returns:
        A ``bioimage_cpp.utils.Blocking`` instance.
    """
    ndim = len(shape)
    if roi is None:
        roi_begin = [0] * ndim
        roi_end = [int(s) for s in shape]
    else:
        roi_begin = [int(sl.start) if sl.start is not None else 0 for sl in roi]
        roi_end = [int(sl.stop) if sl.stop is not None else int(s) for sl, s in zip(roi, shape)]
    return bic.utils.Blocking(roi_begin, roi_end, [int(b) for b in block_shape])


def check_rerun_args(job_type: str, resume_from: Optional[str],
                     subset: Optional[Sequence[int]], *, subset_name: str = "block_ids") -> None:
    """Validate an operation's rerun arguments (``resume_from`` vs a subset).

    Args:
        job_type: The execution backend (``"local"``/``"subprocess"``/``"slurm"``).
        resume_from: The preserved temp folder to resume from, or ``None``.
        subset: The explicit subset (``block_ids``/``item_ids``) to process, or ``None``.
        subset_name: The subset argument's name, for error messages.

    Raises:
        ValueError: If both ``resume_from`` and ``subset`` are given, or if ``resume_from`` is
            used with the local backend (which keeps no temp folder to resume from).
    """
    if resume_from is not None:
        if subset is not None:
            raise ValueError(f"Pass either 'resume_from' or '{subset_name}', not both.")
        if job_type == "local":
            raise ValueError(
                "resume_from is only valid for distributed backends (subprocess/slurm); the "
                "local runner keeps no temp folder. Re-run the operation in-process instead "
                f"(optionally with {subset_name}=err.failed_block_ids for a subset)."
            )


def group_blocks_by_shard(
    blocking: Blocking,
    outputs: Sequence[Source],
    block_ids: Sequence[int],
) -> Optional[List[List[int]]]:
    """Group blocks so that every shard is written by a single worker.

    For a sharded zarr v3 array the atomic write unit is the *shard*, not the inner chunk:
    two blocks writing different inner chunks of the same shard concurrently corrupt it. To
    keep the block shape flexible (rather than forcing it to a shard multiple) the runners
    route each group to one worker, which processes its blocks sequentially — so same-shard
    writes never race. This computes those groups: blocks that share any shard (for any
    sharded output) are placed in the same group via a union-find over the block ids.

    The shard grid is anchored at coordinate 0 and considered along the trailing (spatial)
    shard axes only; a leading channel axis on an output is fully written by every block and
    is not a routing axis (mirrors the chunk handling in
    :meth:`Runner._validate_write_safety`).

    Args:
        blocking: The blocking used to map a block id to its (non-halo) write region.
        outputs: The output sources; only those with a ``shards`` shape drive the grouping.
        block_ids: The block ids to group.

    Returns:
        A list of groups (each a sorted list of block ids), ordered by each group's smallest
        id; ``None`` if no output is sharded (the caller should then use the default
        one-block-per-unit path); an empty list if ``block_ids`` is empty.
    """
    sharded = [(idx, out) for idx, out in enumerate(outputs) if out.shards is not None]
    if not sharded:
        return None
    block_ids = [int(b) for b in block_ids]
    if not block_ids:
        return []

    ndim = len(blocking.get_block(block_ids[0]).begin)
    # Per sharded output, the spatial (trailing) shard extent that defines its cell grid.
    shard_spatial = [(idx, tuple(int(s) for s in out.shards[-ndim:])) for idx, out in sharded]

    # Union-find over the positions in block_ids (dense 0..n-1); bic's UnionFind, as used in
    # segmentation/label.py, instead of a hand-rolled one.
    uf = bic.utils.UnionFind(len(block_ids))
    cell_owner: Dict[Tuple[int, ...], int] = {}
    for pos, bid in enumerate(block_ids):
        block = blocking.get_block(bid)
        begin = [int(b) for b in block.begin]
        end = [int(e) for e in block.end]
        for out_idx, shard in shard_spatial:
            ranges = [range(begin[d] // shard[d], (end[d] + shard[d] - 1) // shard[d])
                      for d in range(ndim)]
            for cell in itertools.product(*ranges):
                owner = cell_owner.setdefault((out_idx,) + cell, pos)
                if owner != pos:
                    uf.merge(pos, owner)

    groups: Dict[int, List[int]] = {}
    for pos, bid in enumerate(block_ids):
        groups.setdefault(int(uf.find(pos)), []).append(bid)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def maybe_warn_imbalance(loads: Sequence[int], num_workers: int, n_groups: int,
                         name: str) -> None:
    """Warn when shard-exclusive routing leaves workers idle or badly load-imbalanced.

    Args:
        loads: The per-worker (or per-task) block counts of the assignment.
        num_workers: The requested number of workers.
        n_groups: The number of shard groups (schedulable units) the blocks formed.
        name: A short run name used in the warning message.
    """
    if not loads:
        return
    if n_groups < int(num_workers):
        warnings.warn(
            f"Shard routing for '{name or 'run'}' produced only {n_groups} shard-group(s) for "
            f"{num_workers} workers, so {int(num_workers) - n_groups} worker(s) will be idle. "
            "A few shards span the data; use a smaller shard shape or fewer workers to balance. "
            "Results are still correct.",
            stacklevel=2,
        )
        return
    mx, mn = max(loads), min(loads)
    mean = sum(loads) / len(loads)
    if mx > mn and mx > 1.5 * mean:
        warnings.warn(
            f"Uneven worker load for '{name or 'run'}': block counts per worker range {mn}..{mx} "
            f"(mean {mean:.1f}). Some shards span disproportionately many blocks; results are "
            "still correct but parallelism is reduced.",
            stacklevel=2,
        )
