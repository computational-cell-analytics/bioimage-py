"""Chunk/roi overlap helpers for the file-format wrappers.

The index-normalization helpers (``normalize_index``, ``squeeze_singletons``,
``slice_to_start_stop``, ``int_to_start_stop``) now live in :mod:`bioimage_py._indexing`; they are
re-exported here for backwards compatibility with the io wrappers' imports.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence, Tuple

from .._indexing import (
    int_to_start_stop,
    normalize_index,
    slice_to_start_stop,
    squeeze_singletons,
)

# Re-exported for the io wrappers that import the index helpers from here.
__all__ = [
    "slice_to_start_stop",
    "int_to_start_stop",
    "normalize_index",
    "squeeze_singletons",
    "map_chunk_to_roi",
    "chunks_overlapping_roi",
]


def map_chunk_to_roi(
    chunk_id: Sequence[int], roi: Tuple[slice, ...], chunks: Tuple[int, ...]
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """Compute the overlap of a chunk and a roi, in chunk-local and roi-local coordinates."""
    block_begin = [cid * ch for cid, ch in zip(chunk_id, chunks)]
    block_end = [beg + ch for beg, ch in zip(block_begin, chunks)]

    roi_begin = [rr.start for rr in roi]
    roi_end = [rr.stop for rr in roi]

    chunk_bb, roi_bb = [], []
    ndim = len(chunk_id)
    for dim in range(ndim):
        off_diff = block_begin[dim] - roi_begin[dim]
        end_diff = roi_end[dim] - block_end[dim]

        if off_diff < 0:
            begin_in_roi = 0
            begin_in_block = -off_diff
            shape_in_roi = (
                block_end[dim] - roi_begin[dim]
                if block_end[dim] <= roi_end[dim]
                else roi_end[dim] - roi_begin[dim]
            )
        elif end_diff < 0:
            begin_in_roi = block_begin[dim] - roi_begin[dim]
            begin_in_block = 0
            shape_in_roi = roi_end[dim] - block_begin[dim]
        else:
            begin_in_roi = block_begin[dim] - roi_begin[dim]
            begin_in_block = 0
            shape_in_roi = chunks[dim]

        chunk_bb.append(slice(begin_in_block, begin_in_block + shape_in_roi))
        roi_bb.append(slice(begin_in_roi, begin_in_roi + shape_in_roi))

    return tuple(chunk_bb), tuple(roi_bb)


def chunks_overlapping_roi(roi: Tuple[slice, ...], chunks: Tuple[int, ...]) -> Sequence[Tuple[int, ...]]:
    """Return the grid ids of all chunks overlapping a region of interest."""
    ranges = [
        range(rr.start // ch, rr.stop // ch if rr.stop % ch == 0 else rr.stop // ch + 1)
        for rr, ch in zip(roi, chunks)
    ]
    return product(*ranges)
