"""On-the-fly transformation sources (wrappers).

A wrapper is a :class:`~bioimage_py.sources.Source` that wraps another source and applies
a transformation in :meth:`__getitem__`. Because wrappers are sources with a serializable
``to_spec``/``from_spec`` round-trip, they are reopened on workers like any other source.
"""
from __future__ import annotations

from typing import Optional, Tuple, Type

import numpy as np

from ..sources.base import Source, SourceSpec
from ..sources.dispatch import SourceLike, as_source, from_spec

# Registry of wrapper classes by name, for reconstruction from a spec.
_WRAPPER_REGISTRY: dict = {}


def register_wrapper(cls: Type["WrapperSource"]) -> Type["WrapperSource"]:
    """Class decorator registering a wrapper for spec-based reconstruction."""
    _WRAPPER_REGISTRY[cls.__name__] = cls
    return cls


class WrapperSource(Source):
    """Base class for wrappers that delegate metadata to the wrapped source.

    Subclasses implement :meth:`__getitem__` (the transform) and :meth:`_params` (the
    keyword arguments needed to rebuild the wrapper). The output dtype may differ from the
    wrapped source, so subclasses can override :attr:`dtype`.
    """

    def __init__(self, source: SourceLike) -> None:
        self._source = as_source(source)

    @property
    def source(self) -> Source:
        """The wrapped source."""
        return self._source

    def __setitem__(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        raise TypeError("Wrapper sources are read-only.")

    @property
    def writable(self) -> bool:
        """Wrapper sources are read-only."""
        return False

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._source.shape

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        return self._source.chunks

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        return self._source.shards

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {}

    def to_spec(self) -> SourceSpec:
        """Return a spec recording the wrapper class, its params, and the wrapped spec."""
        params = dict(self._params())
        params["cls"] = type(self).__name__
        return SourceSpec(kind="wrapper", params=params, wrapped=self._source.to_spec())


def wrapper_from_spec(spec: SourceSpec) -> WrapperSource:
    """Reconstruct a wrapper source from its spec."""
    params = dict(spec.params)
    cls_name = params.pop("cls")
    cls = _WRAPPER_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(f"Unknown wrapper class {cls_name!r}; it must be registered with @register_wrapper.")
    inner = from_spec(spec.wrapped)
    return cls(inner, **params)
