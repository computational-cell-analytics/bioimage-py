"""Source abstractions: the unit of data the runner reads from and writes to."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import EllipsisType
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .._indexing import normalize_index, squeeze_singletons

# A numpy-style basic index: an int, slice or ellipsis, or a tuple thereof.
_IndexItem = Union[int, slice, EllipsisType]
Index = Union[_IndexItem, Tuple[_IndexItem, ...]]


@dataclass
class SourceSpec:
    """Serializable description of how to (re)open a :class:`Source` on another process.

    This is intentionally a plain dataclass so that it cloudpickles trivially and is
    human-readable in the debug dump of a distributed job.

    Attributes:
        kind: The source kind, e.g. ``"zarr"``, ``"z5py"`` or ``"wrapper"``.
        path: Filesystem path of the container (for array sources).
        internal_path: Path of the array inside the container (for array sources).
        params: Extra keyword arguments needed to reconstruct the source.
        wrapped: The spec of the wrapped source, for wrapper sources.
    """

    kind: str
    path: Optional[str] = None
    internal_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    wrapped: Optional["SourceSpec"] = None


class Source(ABC):
    """Array-like data handle with a serializable open-spec.

    A source supports numpy-style basic indexing: an integer, slice or ellipsis, or a tuple thereof.
    The index is normalized to a full tuple of in-bounds slices (axes selected by an integer are
    squeezed out of the result), so ``src[5]``, ``src[5, :]``, ``src[..., 0]`` and
    ``src[(slice(0, 8), slice(0, 8))]`` all work. Subclasses implement the normalized read/write via
    :meth:`_getitem` / :meth:`_setitem`, which always receive a full tuple of slices.

    In the runner hot path, per-block compute functions build that full tuple explicitly with
    :func:`bioimage_py.util.to_roi` (rather than relying on this normalization) so it is clear which
    region -- outer / inner / inner-local -- is being indexed.
    """

    def __getitem__(self, index: Index) -> np.ndarray:
        """Read a region, normalizing ``index`` and squeezing integer-indexed axes."""
        roi, to_squeeze = normalize_index(index, self.shape)
        return squeeze_singletons(self._getitem(roi), to_squeeze)

    def __setitem__(self, index: Index, value: np.ndarray) -> None:
        """Write a region, normalizing ``index`` and re-inserting integer-indexed axes."""
        roi, to_squeeze = normalize_index(index, self.shape)
        value = np.asarray(value)
        if to_squeeze:
            value = np.expand_dims(value, to_squeeze)
        self._setitem(roi, value)

    @abstractmethod
    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Read the region given by a full tuple of in-bounds slices."""
        ...

    @abstractmethod
    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        """Write the region given by a full tuple of in-bounds slices."""
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        ...

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The chunk shape of the underlying array, or ``None`` if unchunked."""
        return None

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        """The shard shape of the underlying array, or ``None`` if unsharded."""
        return None

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def writable(self) -> bool:
        """Whether this source supports writing via :meth:`__setitem__`.

        Distributed runs reject non-writable sources passed as outputs.
        """
        return True

    @abstractmethod
    def to_spec(self) -> SourceSpec:
        """Return a serializable spec to reopen this source on another process.

        Raises:
            ValueError: If the source cannot be reopened elsewhere (e.g. in-memory data).
        """
        ...
