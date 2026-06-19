"""Conversion of source-like objects to :class:`Source`, and reconstruction from specs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import numpy as np

from .array_source import ArraySource
from .base import Source, SourceSpec

if TYPE_CHECKING:  # imported only for the type alias below; optional at runtime.
    import z5py
    import zarr

# Registry mapping a predicate to a converter function.
_CONVERTERS: list = []


def register_source(predicate: Callable[[object], bool], converter: Callable[[object], Source]) -> None:
    """Register a converter used by :func:`as_source`.

    Args:
        predicate: Returns ``True`` if ``converter`` can handle the object.
        converter: Builds a :class:`Source` from the object.
    """
    _CONVERTERS.append((predicate, converter))


def as_source(obj: "SourceLike") -> Source:
    """Convert a supported object into a :class:`Source`.

    Idempotent on :class:`Source` inputs. numpy / zarr / z5py arrays are wrapped in an
    :class:`ArraySource`. Bare paths are intentionally not supported (see the design doc).

    Args:
        obj: The object to convert.

    Returns:
        A :class:`Source`.

    Raises:
        TypeError: If the object cannot be converted (e.g. a string path).
    """
    if isinstance(obj, Source):
        return obj
    if isinstance(obj, (str, bytes)):
        raise TypeError(
            "Passing strings / file paths as a source is not supported. Open the array "
            "yourself (e.g. with zarr or z5py) and pass the handle."
        )
    for predicate, converter in _CONVERTERS:
        if predicate(obj):
            return converter(obj)
    # numpy and any array-like with shape/dtype fall back to ArraySource.
    if isinstance(obj, np.ndarray) or (hasattr(obj, "shape") and hasattr(obj, "dtype")):
        return ArraySource(obj)
    raise TypeError(f"Cannot convert object of type {type(obj)!r} to a Source.")


def from_spec(spec: SourceSpec) -> Source:
    """Reconstruct a :class:`Source` from its :class:`SourceSpec`."""
    if spec.kind in ("zarr", "z5py"):
        return ArraySource.reopen(spec)
    if spec.kind == "file":
        from .file_source import FileSource

        return FileSource.reopen(spec)
    if spec.kind == "cloudvolume":
        from .cloudvolume_source import CloudVolumeSource

        return CloudVolumeSource.reopen(spec)
    if spec.kind == "webknossos":
        from .webknossos_source import WebKnossosSource

        return WebKnossosSource.reopen(spec)
    if spec.kind == "wrapper":
        from ..wrapper.base import wrapper_from_spec

        return wrapper_from_spec(spec)
    raise ValueError(f"Cannot reconstruct source from spec of kind {spec.kind!r}.")


# SourceLike is the public input/output type for operations. At runtime ``as_source`` also
# accepts any duck-typed array (an object exposing ``shape``, ``dtype`` and ``__getitem__``);
# the listed members are the statically-known supported types.
SourceLike = Union[Source, np.ndarray, "zarr.Array", "z5py.Dataset"]


def _is_module(obj: object, module: str) -> bool:
    """Return whether ``obj``'s top-level module is ``module`` (without importing it)."""
    return type(obj).__module__.split(".")[0] == module


def _convert_h5py(dataset: object) -> Source:
    """Wrap a live h5py dataset as a reopenable :class:`FileSource`."""
    from .file_source import FileSource

    file = dataset.file  # type: ignore[attr-defined]
    writable = file.mode != "r"
    return FileSource(
        dataset,
        path=file.filename,
        internal_path=dataset.name.lstrip("/"),  # type: ignore[attr-defined]
        format="hdf5",
        mode="r+" if writable else "r",
        writable=writable,
    )


def _convert_cloudvolume(volume: object) -> Source:
    """Wrap a live CloudVolume handle as a :class:`CloudVolumeSource`.

    Captures the write-affecting open params too (``non_aligned_writes``, ``cache``) so the spec
    round-trips the volume's write semantics, not just its read settings.
    """
    from .cloudvolume_source import CloudVolumeSource

    cache = getattr(volume, "cache", None)
    open_params = {
        "mip": int(volume.mip),  # type: ignore[attr-defined]
        "fill_missing": bool(volume.fill_missing),  # type: ignore[attr-defined]
        "bounded": bool(volume.bounded),  # type: ignore[attr-defined]
        "non_aligned_writes": bool(getattr(volume, "non_aligned_writes", False)),
        "cache": bool(getattr(cache, "enabled", False)),
    }
    return CloudVolumeSource(volume, open_params=open_params)


# Live-object converters for the optional backends. The predicates only inspect the type's module
# name, so importing this module never imports h5py / cloudvolume; the converters import lazily.
register_source(lambda o: _is_module(o, "h5py") and hasattr(o, "file") and hasattr(o, "shape"), _convert_h5py)
register_source(lambda o: _is_module(o, "cloudvolume") and hasattr(o, "cloudpath"), _convert_cloudvolume)
