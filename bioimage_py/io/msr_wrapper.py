"""Read-only array-like wrapper for ``.msr`` (Abberior/Zeiss) microscopy files."""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ._util import normalize_index, squeeze_singletons

try:
    from msr_reader import OBFFile
except ImportError:
    OBFFile = None


_MSR_READER_INSTALL_ERROR = (
    "msr_reader is required for MSR images, but is not installed. Install it with `pip install msr-reader`."
)

StackSelection = Union[int, str, Sequence[Union[int, str]]]
PathLike = Union[os.PathLike, str]


def _require_msr_reader() -> None:
    """Raise a clear error if msr_reader is not installed."""
    if OBFFile is None:
        raise AttributeError(_MSR_READER_INSTALL_ERROR)


def _normalize_stack_selection(stack_selection: StackSelection) -> Tuple[Union[int, str], ...]:
    """Normalize a stack selection into a non-empty tuple of indices / names."""
    if isinstance(stack_selection, (str, int)):
        return (stack_selection,)
    stack_selection = tuple(stack_selection)
    if not stack_selection:
        raise ValueError("At least one MSR stack index is required")
    return stack_selection


def _resolve_stack_indices(msr: Any, stack_selection: StackSelection) -> Tuple[int, ...]:
    """Resolve a stack selection (indices or names) to integer stack indices."""
    resolved = []
    for stack in _normalize_stack_selection(stack_selection):
        if isinstance(stack, int):
            resolved.append(stack)
        else:
            resolved.append(msr.stack_names.index(stack))
    return tuple(resolved)


class MSRFile(Mapping):
    """Root handle for an MSR file, keyed by stack index, stack name, or ``"data"`` (all stacks).

    Args:
        path: The filepath of the msr file.
        mode: The mode for opening the file; only read mode (``"r"``) is supported.
    """

    default_key = "0"
    writable = False

    def __init__(self, path: PathLike, mode: str = "r") -> None:
        _require_msr_reader()
        if mode != "r":
            raise ValueError("MSR files only support read mode.")
        self.path = os.fspath(path)
        self.mode = mode
        self.msr = OBFFile(self.path)

    def __enter__(self) -> "MSRFile":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying OBF file handle."""
        self.msr.close()

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}

    def _normalize_key(self, key: str) -> Union[int, str]:
        """Map a string key to ``"data"`` or an integer stack index."""
        if key == "data":
            return key
        try:
            return int(key)
        except (TypeError, ValueError):
            if key in self.msr.stack_names:
                return self.msr.stack_names.index(key)
            raise KeyError(f"Could not find key {key}")

    def __getitem__(self, key: str) -> "MSRDataset":
        key = self._normalize_key(key)
        if key == "data":
            return MSRDataset(self.path, tuple(range(self.msr.num_stacks)))
        return MSRDataset(self.path, key)

    def __iter__(self):
        for index in range(self.msr.num_stacks):
            yield str(index)
        yield "data"

    def __len__(self) -> int:
        return self.msr.num_stacks + 1

    def __contains__(self, name: object) -> bool:
        if name == "data":
            return True
        try:
            index = int(name)
            return 0 <= index < self.msr.num_stacks
        except (TypeError, ValueError):
            return name in self.msr.stack_names


class MSRDataset:
    """Array-like view of one stack (``(H, W)``) or several stacks (``(C, H, W)``) in an MSR file."""

    def __init__(self, path: PathLike, stack_selection: StackSelection) -> None:
        self.path = os.fspath(path)
        self.stack_selection = _normalize_stack_selection(stack_selection)

        with OBFFile(self.path) as msr:
            self.stack_indices = _resolve_stack_indices(msr, self.stack_selection)
            sample = msr.read_stack(self.stack_indices[0])

        if sample.ndim != 2:
            raise ValueError(f"Expected a 2D MSR stack from {self.path}, got shape {sample.shape}")

        self._dtype = sample.dtype
        self._shape = sample.shape if len(self.stack_indices) == 1 else (len(self.stack_indices),) + sample.shape
        self._size = int(np.prod(self._shape))
        # Decoded stacks are cached so each is read+decoded once per dataset, not per block read.
        # The same dataset instance is shared across worker threads (local backend), so the
        # cache is guarded by a lock; decoded stacks are immutable once stored.
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_lock = threading.Lock()

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the data."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The chunk shape; msr stacks are unchunked, so this is ``None``."""
        return None

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the data."""
        return self._shape

    @property
    def size(self) -> int:
        """The number of elements in the data."""
        return self._size

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}

    def _stack_data(self, stack_index: int) -> np.ndarray:
        """Return the decoded stack, reading+caching it on first access."""
        with self._cache_lock:
            data = self._cache.get(stack_index)
            if data is None:
                with OBFFile(self.path) as msr:
                    data = msr.read_stack(stack_index)
                self._cache[stack_index] = data
        return data

    def _read_stack(self, stack_index: int, spatial_index: Any) -> np.ndarray:
        """Read (from cache) and crop a single stack."""
        return self._stack_data(stack_index)[spatial_index]

    def __getitem__(self, key: Any) -> np.ndarray:
        key, to_squeeze = normalize_index(key, self.shape)
        if len(self.stack_indices) == 1:
            data = self._read_stack(self.stack_indices[0], key)
            return squeeze_singletons(data, to_squeeze).copy()

        # Multi-stack: key[0] selects channels. normalize_index converts an integer channel to a
        # unit slice (+ squeeze) and rejects non-trivial steps, so the channel slice has step 1.
        channel_index, spatial_index = key[0], key[1:]
        selected = self.stack_indices[channel_index.start:channel_index.stop]
        if selected:
            stacked = np.stack([self._read_stack(idx, spatial_index) for idx in selected], axis=0)
        else:  # empty channel selection -> empty array with the right spatial shape.
            spatial_shape = tuple(len(range(*sl.indices(n)))
                                  for sl, n in zip(spatial_index, self.shape[1:]))
            stacked = np.empty((0,) + spatial_shape, dtype=self._dtype)
        return squeeze_singletons(stacked, to_squeeze).copy()
