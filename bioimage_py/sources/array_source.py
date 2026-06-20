"""Concrete :class:`Source` wrapping numpy / zarr / z5py arrays."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .base import Source, SourceSpec


def _is_numpy(array: Any) -> bool:
    """Return whether ``array`` is a numpy ndarray."""
    return isinstance(array, np.ndarray)


def _zarr_spec(array: Any) -> SourceSpec:
    """Build a spec for a zarr array (container root + internal path)."""
    # zarr v3: the container lives at ``array.store.root`` (LocalStore), the internal
    # path is ``array.path``. Stores without a filesystem root cannot be reopened.
    store = getattr(array, "store", None)
    root = getattr(store, "root", None)
    if root is None:
        raise ValueError(
            "Cannot serialize this zarr array: its store has no filesystem root "
            "(e.g. an in-memory store). Use a file-backed array for distributed execution."
        )
    return SourceSpec(kind="zarr", path=str(root), internal_path=str(array.path))


def _z5py_spec(array: Any) -> SourceSpec:
    """Build a spec for a z5py dataset (file path + internal path)."""
    # z5py: container path is ``array.file.filename``, internal path is ``array.name``.
    return SourceSpec(
        kind="z5py",
        path=str(array.file.filename),
        internal_path=str(array.name).lstrip("/"),
    )


class ArraySource(Source):
    """Wrap a numpy, zarr or z5py array as a :class:`Source`.

    Args:
        array: The wrapped array. numpy arrays are usable for local execution only;
            their :meth:`to_spec` raises because they cannot be reopened on another node.
    """

    def __init__(self, array: Any) -> None:
        self._array = array

    @property
    def array(self) -> Any:
        """The wrapped array object."""
        return self._array

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        return np.asarray(self._array[roi])

    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        self._array[roi] = value

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array.dtype)

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        chunks = getattr(self._array, "chunks", None)
        return None if chunks is None else tuple(int(c) for c in chunks)

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        shards = getattr(self._array, "shards", None)
        return None if shards is None else tuple(int(s) for s in shards)

    def to_spec(self) -> SourceSpec:
        """Return a reopen spec, raising for in-memory (numpy) arrays."""
        array = self._array
        if _is_numpy(array):
            raise ValueError(
                "Cannot serialize an in-memory numpy array for distributed execution. "
                "numpy arrays are supported for local execution only; pass a zarr or "
                "n5 (z5py) array for the 'subprocess' or 'slurm' backends."
            )
        module = type(array).__module__.split(".")[0]
        if module == "zarr":
            return _zarr_spec(array)
        if module == "z5py":
            return _z5py_spec(array)
        raise ValueError(f"Cannot serialize array of type {type(array)!r}: unsupported source kind.")

    @staticmethod
    def reopen(spec: SourceSpec) -> "ArraySource":
        """Reopen an array source from its spec (read-write)."""
        if spec.kind == "zarr":
            import zarr

            array = zarr.open_array(spec.path, path=spec.internal_path, mode="r+")
            return ArraySource(array)
        if spec.kind == "z5py":
            import z5py

            f = z5py.File(spec.path, mode="r+")
            return ArraySource(f[spec.internal_path])
        raise ValueError(f"ArraySource cannot reopen spec of kind {spec.kind!r}.")
