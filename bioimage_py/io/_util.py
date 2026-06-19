"""Indexing helpers shared by the file-format wrappers (ported from ``elf.util``)."""
from __future__ import annotations

import numbers
from itertools import product
from typing import Any, Sequence, Tuple

import numpy as np


def slice_to_start_stop(s: slice, size: int) -> slice:
    """Normalize a slice so its start/stop are positive, in-bounds coordinates."""
    if s.step not in (None, 1):
        raise ValueError("Nontrivial steps are not supported")

    if s.start is None:
        start = 0
    elif -size <= s.start < 0:
        start = size + s.start
    elif s.start < -size or s.start >= size:
        return slice(None, 0)
    else:
        start = s.start

    if s.stop is None or s.stop > size:
        stop = size
    elif s.stop < 0:
        stop = size + s.stop
    else:
        stop = s.stop

    if stop < 1:
        return slice(None, 0)

    # Clamp so a reversed slice (start > stop) yields an empty region rather than a negative
    # extent (which crashes the assemble-from-pieces wrappers downstream).
    return slice(start, max(start, stop))


def int_to_start_stop(i: int, size: int) -> slice:
    """Return the unit slice corresponding to an integer coordinate."""
    if -size < i < 0:
        start = i + size
    elif i >= size or i < -size:
        raise ValueError("Index ({}) out of range (0-{})".format(i, size - 1))
    else:
        start = i
    return slice(start, start + 1)


def normalize_index(index: Any, shape: Tuple[int, ...]) -> Tuple[Tuple[slice, ...], Tuple[int, ...]]:
    """Normalize an index into a full tuple of in-bounds slices plus the axes to squeeze.

    The index may be an integer, a slice, an ellipsis, or a tuple thereof. It is expanded to one
    entry per axis with positive start/stop coordinates; integer entries record the axis to squeeze.

    Args:
        index: The index to normalize.
        shape: The shape of the array-like object being indexed.

    Returns:
        A tuple of the normalized slices and the axes that should be squeezed after indexing.
    """
    type_msg = (
        "Advanced selection inappropriate. "
        "Only numbers, slices (`:`), and ellipsis (`...`) are valid indices (or tuples thereof)"
    )

    if isinstance(index, tuple):
        slices_lst = list(index)
    elif isinstance(index, (numbers.Number, slice, type(Ellipsis))):
        slices_lst = [index]
    else:
        raise TypeError(type_msg)

    ndim = len(shape)
    if len([item for item in slices_lst if item != Ellipsis]) > ndim:
        raise TypeError("Argument sequence too long")
    elif len(slices_lst) < ndim and Ellipsis not in slices_lst:
        slices_lst.append(Ellipsis)

    normalized = []
    found_ellipsis = False
    squeeze = []
    for item in slices_lst:
        d = len(normalized)
        if isinstance(item, slice):
            normalized.append(slice_to_start_stop(item, shape[d]))
        elif isinstance(item, numbers.Number):
            squeeze.append(d)
            normalized.append(int_to_start_stop(int(item), shape[d]))
        elif isinstance(item, type(Ellipsis)):
            if found_ellipsis:
                raise ValueError("Only one ellipsis may be used")
            found_ellipsis = True
            while len(normalized) + (len(slices_lst) - d - 1) < ndim:
                normalized.append(slice(0, shape[len(normalized)]))
        else:
            raise TypeError(type_msg)
    return tuple(normalized), tuple(squeeze)


def squeeze_singletons(item: np.ndarray, to_squeeze: Tuple[int, ...]) -> np.ndarray:
    """Squeeze the axes recorded by :func:`normalize_index` from a read result."""
    if len(to_squeeze) == len(item.shape):
        return item.flatten()[0]
    elif to_squeeze:
        return item.squeeze(to_squeeze)
    else:
        return item


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
