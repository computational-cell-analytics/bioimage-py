"""File-backed :class:`Source` opened by path, with a reopenable ``kind="file"`` spec."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..io import constructor_for_format, infer_format, is_writable_format, open_file
from .array_source import ArraySource
from .base import SourceSpec

PathLike = Union[os.PathLike, str]


class FileSource(ArraySource):
    """A :class:`Source` wrapping a file-backed dataset, reopenable from its path + format.

    Unlike a plain :class:`ArraySource` (whose spec is introspected from a live zarr/z5py handle),
    a :class:`FileSource` stores the recipe to reopen the dataset via :func:`open_source`, so it
    round-trips for any registered format (hdf5, mrc, nifti, tif, ...).

    Args:
        array: The opened array-like dataset.
        path: Path of the container/file the dataset was opened from.
        internal_path: Key of the dataset inside the container (``""`` for single-array files).
        format: The registered format name (e.g. ``"hdf5"``, ``"mrc"``).
        mode: The mode the file was opened in.
        open_kwargs: Extra keyword arguments forwarded to the backend constructor.
        writable: Whether writes are permitted (format is writable and ``mode`` is not read-only).
    """

    def __init__(
        self,
        array: Any,
        *,
        path: PathLike,
        internal_path: str,
        format: str,
        mode: str = "r",
        open_kwargs: Optional[Dict[str, Any]] = None,
        writable: bool = False,
    ) -> None:
        super().__init__(array)
        self._path = str(path)
        self._internal_path = internal_path
        self._format = format
        self._mode = mode
        self._open_kwargs = dict(open_kwargs or {})
        self._writable = bool(writable)

    @property
    def format(self) -> str:
        """The registered format name of this source."""
        return self._format

    @property
    def writable(self) -> bool:
        """Whether this source supports writing via :meth:`__setitem__`."""
        return self._writable

    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        if not self._writable:
            raise TypeError(
                f"FileSource for format {self._format!r} opened in mode {self._mode!r} is read-only."
            )
        self._array[roi] = value

    def to_spec(self) -> SourceSpec:
        """Return a ``kind="file"`` spec recording the path, key, format and open options."""
        params: Dict[str, Any] = {"format": self._format, "mode": self._mode}
        params.update(self._open_kwargs)
        return SourceSpec(
            kind="file",
            path=self._path,
            internal_path=self._internal_path,
            params=params,
        )

    @staticmethod
    def reopen(spec: SourceSpec) -> "FileSource":
        """Reopen a file source from its spec (mirrors the original :func:`open_source` call)."""
        params = dict(spec.params)
        fmt = params.pop("format")
        mode = params.pop("mode", "r")
        return open_source(spec.path, spec.internal_path, format=fmt, mode=mode, **params)


def _resolve_dataset(handle: Any, key: Optional[str]) -> Tuple[Any, str]:
    """Resolve the dataset inside an opened file handle, returning ``(dataset, recorded_key)``."""
    # Some backends (e.g. zarr.open on an array path) return an array-like directly.
    if key in (None, "") and hasattr(handle, "shape") and hasattr(handle, "dtype"):
        return handle, ""
    if key is None:
        raise ValueError(
            "This format holds multiple arrays; pass 'internal_path' to select one "
            "(the key/path of the array inside the container)."
        )
    return handle[key], key


def open_source(
    path: PathLike,
    internal_path: Optional[str] = None,
    format: Optional[str] = None,
    mode: str = "r",
    **kwargs: Any,
) -> FileSource:
    """Open a file-backed array as a :class:`Source`.

    The format is inferred from the path extension (overridable via ``format``). ``internal_path``
    selects the array inside a container; when omitted it defaults to the format's natural key
    (e.g. ``"data"`` for mrc/nifti, ``"mag1"`` for knossos, ``""`` for a single image stack), and is
    required for multi-array containers (hdf5/zarr/n5).

    Args:
        path: Path to the file or folder to open.
        internal_path: Key of the array inside the container; format-dependent default if omitted.
        format: Force a registered format name, overriding extension inference.
        mode: Open mode. ``"r"`` (default) is read-only; write modes (``"a"``/``"r+"``/``"w"``)
            are only honored for writable formats (zarr/n5/hdf5).
        kwargs: Extra keyword arguments forwarded to the backend constructor.

    Returns:
        A :class:`FileSource` with a reopenable ``kind="file"`` spec.
    """
    fmt = format if format is not None else infer_format(path)
    # Validate the format is installed up front (raises a clear error otherwise).
    constructor_for_format(fmt)

    handle = open_file(path, mode=mode, format=fmt, **kwargs)
    key = internal_path if internal_path is not None else getattr(handle, "default_key", None)
    dataset, recorded_key = _resolve_dataset(handle, key)

    writable = is_writable_format(fmt) and mode != "r"
    return FileSource(
        dataset,
        path=path,
        internal_path=recorded_key,
        format=fmt,
        mode=mode,
        open_kwargs=kwargs,
        writable=writable,
    )
