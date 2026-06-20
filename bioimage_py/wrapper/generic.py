"""Generic, composable wrapper sources."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..sources.dispatch import SourceLike
from .base import SimpleTransformationSource, WrapperSource, register_wrapper


@register_wrapper
class ThresholdSource(SimpleTransformationSource):
    """Threshold the wrapped source on read.

    ``operator(source[roi], threshold)`` is returned as a boolean array (``operator`` defaults to
    ``numpy.greater``, i.e. ``source[roi] > threshold``).

    Args:
        source: The wrapped source-like object.
        threshold: The threshold value.
        operator: The comparison operator applied as ``operator(block, threshold)``.
    """

    def __init__(self, source: SourceLike, threshold: float, operator: callable = np.greater) -> None:
        self._threshold = float(threshold)
        self._operator = operator
        super().__init__(source, lambda block: operator(block, self._threshold), dtype=np.dtype(bool))

    @property
    def threshold(self) -> float:
        """The threshold value."""
        return self._threshold

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"threshold": self._threshold, "operator": self._operator}


@register_wrapper
class NormalizeSource(SimpleTransformationSource):
    """Normalize the wrapped source to ``[0, 1]`` on read.

    Each read block is independently normalized by its own min and max. Note that this means a
    block-wise read is *not* equivalent to a single whole-array read (the normalization is
    block-local), matching elf's ``NormalizeWrapper``.

    Args:
        source: The wrapped source-like object.
        dtype: The output (floating point) dtype.
        with_channels: Whether the wrapped source has a leading channel axis.
    """

    eps = 1.0e-6

    def __init__(self, source: SourceLike, dtype: str = "float32", with_channels: bool = False) -> None:
        self._norm_dtype = np.dtype(dtype)
        super().__init__(source, self._normalize, dtype=self._norm_dtype, with_channels=with_channels)

    def _normalize(self, block: np.ndarray) -> np.ndarray:
        """Normalize a block to ``[0, 1]`` using its own min and max."""
        block = block.astype(self._norm_dtype)
        block -= block.min()
        block /= (block.max() + self.eps)
        return block

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"dtype": str(self._norm_dtype), "with_channels": self._with_channels}


def _normalize_construction_roi(
    roi: Sequence[Union[int, slice]], shape: Tuple[int, ...]
) -> Tuple[Tuple[slice, ...], Tuple[int, ...]]:
    """Normalize a construction roi to a full tuple of slices, tracking integer (singleton) axes.

    Integer entries become singleton slices and their axes are recorded; missing trailing axes are
    filled with full-extent slices. Negative / open-ended slice bounds are resolved against ``shape``.
    """
    if len(roi) > len(shape):
        raise ValueError(f"roi has more entries ({len(roi)}) than source dimensions ({len(shape)}).")
    normalized: List[slice] = []
    squeeze: List[int] = []
    for ax, sh in enumerate(shape):
        if ax >= len(roi):
            normalized.append(slice(0, sh))
            continue
        entry = roi[ax]
        if isinstance(entry, (int, np.integer)):
            idx = int(entry)
            if idx < 0:
                idx += sh
            if not 0 <= idx < sh:
                raise IndexError(f"index {entry} is out of bounds for axis {ax} with size {sh}.")
            normalized.append(slice(idx, idx + 1))
            squeeze.append(ax)
        elif isinstance(entry, slice):
            start, stop, step = entry.indices(sh)
            if step != 1:
                raise ValueError("RoiSource does not support strided slices.")
            normalized.append(slice(start, stop))
        else:
            raise TypeError(f"roi entries must be ints or slices, got {type(entry)!r}.")
    return tuple(normalized), tuple(squeeze)


@register_wrapper
class RoiSource(WrapperSource):
    """Restrict the wrapped source to a region of interest.

    This is a read-and-write view: reads and writes are offset into the parent source, so writing
    through a :class:`RoiSource` updates the parent (when the parent is writable). For block-wise
    *distributed* output, the ROI offset must be chunk-aligned in the parent, otherwise concurrent
    writes to a shared parent chunk can corrupt it (the runner aligns blocks to this wrapper's
    shape, not the parent's). For an unaligned or read-only ROI, use it purely as an input view.

    Args:
        source: The wrapped source-like object.
        roi: The region of interest as a tuple of ints / slices over the source. Integer entries
            select a single index along that axis; missing trailing axes default to the full extent.
        squeeze: Whether to drop axes selected by an integer entry from the wrapper's shape and
            output (and require matching input on write). Defaults to False.
    """

    def __init__(self, source: SourceLike, roi: Sequence[Union[int, slice]], squeeze: bool = False) -> None:
        super().__init__(source)
        self._orig_roi = tuple(roi)
        self._squeeze = bool(squeeze)
        self._roi, roi_squeeze = _normalize_construction_roi(self._orig_roi, self._source.shape)
        # Axes introduced as singletons by an integer roi entry; only dropped when squeeze is set.
        self._squeeze_axes = roi_squeeze if self._squeeze else ()
        self._kept_axes = tuple(ax for ax in range(len(self._roi)) if ax not in self._squeeze_axes)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._roi[ax].stop - self._roi[ax].start for ax in self._kept_axes)

    @property
    def ndim(self) -> int:
        return len(self._kept_axes)

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        src_chunks = self._source.chunks
        if src_chunks is None:
            return None
        return tuple(min(src_chunks[ax], self._roi[ax].stop - self._roi[ax].start) for ax in self._kept_axes)

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        # A sub-region does not in general share the parent's shard grid.
        return None

    @property
    def writable(self) -> bool:
        """A roi view is writable when its parent is."""
        return self._source.writable

    def _map_roi_to_source(self, roi: Tuple[slice, ...]) -> Tuple[slice, ...]:
        """Map a roi over the kept (reduced) axes into a full index of the parent source."""
        full_index = list(self._roi)  # Default to the construction roi (singleton on squeezed axes).
        for sl, ax in zip(roi, self._kept_axes):
            offset = self._roi[ax].start
            start = (int(sl.start) if sl.start is not None else 0) + offset
            stop = (int(sl.stop) if sl.stop is not None else (self._roi[ax].stop - offset)) + offset
            full_index[ax] = slice(start, stop)
        return tuple(full_index)

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the parent data for the (reduced) region ``roi``."""
        out = self._source[self._map_roi_to_source(roi)]
        if self._squeeze_axes:
            out = np.squeeze(out, axis=self._squeeze_axes)
        return out

    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        """Write ``value`` into the corresponding region of the parent source."""
        value = np.asarray(value)
        if self._squeeze_axes:
            value = np.expand_dims(value, axis=self._squeeze_axes)
        self._source[self._map_roi_to_source(roi)] = value

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"roi": self._orig_roi, "squeeze": self._squeeze}


@register_wrapper
class PadSource(WrapperSource):
    """Right-pad the wrapped source on read.

    The wrapper's shape is the source shape grown by ``pad_width`` per axis. Reads that extend past
    the source are filled by :func:`numpy.pad`. Only right-padding is supported (the source occupies
    the lower corner of the padded space). This is a read-only view.

    Args:
        source: The wrapped source-like object.
        pad_width: The number of elements to append along each axis.
        mode: The padding mode passed to :func:`numpy.pad`.
    """

    def __init__(self, source: SourceLike, pad_width: Sequence[int], mode: str = "constant") -> None:
        super().__init__(source)
        if len(pad_width) != self._source.ndim:
            raise ValueError(f"Expect pad_width of length {self._source.ndim}, got {len(pad_width)}.")
        self._pad_width = tuple(int(p) for p in pad_width)
        self._src_shape = self._source.shape
        self._mode = mode

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(sh + pw for sh, pw in zip(self._src_shape, self._pad_width))

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the (possibly right-padded) data at ``roi``."""
        local_pad, local_index = [], []
        for sl, sh, psh in zip(roi, self._src_shape, self.shape):
            start = int(sl.start) if sl.start is not None else 0
            stop = int(sl.stop) if sl.stop is not None else psh
            overhang_start = max(0, start - sh)
            overhang_stop = max(0, stop - sh)
            if overhang_start > 0:
                raise NotImplementedError("PadSource only supports right-padding.")
            elif overhang_stop > 0:
                local_pad.append(overhang_stop)
                local_index.append(slice(start, sh))
            else:
                local_pad.append(0)
                local_index.append(slice(start, stop))

        out = self._source[tuple(local_index)]
        if any(lpad > 0 for lpad in local_pad):
            pad_width = tuple((0, lpad) for lpad in local_pad)
            out = np.pad(out, pad_width, mode=self._mode)
        return out

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"pad_width": self._pad_width, "mode": self._mode}
