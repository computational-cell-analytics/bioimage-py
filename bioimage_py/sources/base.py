"""Source abstractions: the unit of data the runner reads from and writes to."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


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

    A source is indexed only with a tuple of slices (a region of interest). Block
    descriptors from ``bioimage_cpp.utils`` are converted to such tuples by
    :func:`bioimage_py.util.to_roi` in the compute function, never by the source.
    """

    @abstractmethod
    def __getitem__(self, roi: Tuple[slice, ...]) -> np.ndarray:
        ...

    @abstractmethod
    def __setitem__(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
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
