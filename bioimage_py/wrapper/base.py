"""On-the-fly transformation sources (wrappers).

A wrapper is a :class:`~bioimage_py.sources.Source` that wraps another source and applies
a transformation in :meth:`__getitem__`. Because wrappers are sources with a serializable
``to_spec``/``from_spec`` round-trip, they are reopened on workers like any other source.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Type

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

    @property
    def _wrapped_sources(self) -> Tuple[Source, ...]:
        """The wrapped source(s). Single-source wrappers wrap exactly one; override for many."""
        return (self._source,)

    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
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

    @classmethod
    def _from_wrapped(cls, sources: Sequence[Source], params: dict) -> "WrapperSource":
        """Reconstruct the wrapper from its wrapped source(s) and params.

        The default rebuilds a single-source wrapper as ``cls(source, **params)``. Multi-source
        wrappers override this to consume all reconstructed sources.
        """
        return cls(sources[0], **params)

    def to_spec(self) -> SourceSpec:
        """Return a spec recording the wrapper class, its params, and the wrapped spec(s)."""
        params = dict(self._params())
        params["cls"] = type(self).__name__
        wrapped_specs = [s.to_spec() for s in self._wrapped_sources]
        # Single-source wrappers store a single spec; multi-source wrappers store a list.
        wrapped = wrapped_specs[0] if len(wrapped_specs) == 1 else wrapped_specs
        return SourceSpec(kind="wrapper", params=params, wrapped=wrapped)


def wrapper_from_spec(spec: SourceSpec) -> WrapperSource:
    """Reconstruct a wrapper source from its spec."""
    params = dict(spec.params)
    cls_name = params.pop("cls")
    cls = _WRAPPER_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(f"Unknown wrapper class {cls_name!r}; it must be registered with @register_wrapper.")
    wrapped = spec.wrapped
    specs = wrapped if isinstance(wrapped, (list, tuple)) else [wrapped]
    sources = [from_spec(s) for s in specs]
    return cls._from_wrapped(sources, params)


@register_wrapper
class SimpleTransformationSource(WrapperSource):
    """Apply a value-only transformation to the wrapped source on read.

    The transformation depends only on the data values, not on coordinates, so its signature is
    ``transformation(block)``. The callable is captured in the spec (it round-trips via cloudpickle),
    so prefer picklable callables; concrete wrappers (e.g. :class:`ThresholdSource`) instead store
    simple parameters and rebuild the callable in their constructor.

    Args:
        source: The wrapped source-like object.
        transformation: The value transformation to apply to each read block.
        with_channels: Whether the wrapped source has a leading channel axis. If set, that axis is
            hidden from this wrapper's shape and passed through to the transformation on read.
        dtype: The output dtype. Defaults to the wrapped source's dtype.
    """

    def __init__(self, source: SourceLike, transformation: Callable, *,
                 with_channels: bool = False, dtype: Optional[np.dtype] = None) -> None:
        super().__init__(source)
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self._transformation = transformation
        self._with_channels = bool(with_channels)
        self._dtype = None if dtype is None else np.dtype(dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._source.shape[1:] if self._with_channels else self._source.shape

    @property
    def ndim(self) -> int:
        return self._source.ndim - 1 if self._with_channels else self._source.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype if self._dtype is not None else self._source.dtype

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        src_chunks = self._source.chunks
        if src_chunks is None:
            return None
        return src_chunks[1:] if self._with_channels else src_chunks

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the transformed data at ``roi``."""
        index = (slice(None),) + tuple(roi) if self._with_channels else roi
        return self._transformation(self._source[index])

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"transformation": self._transformation, "with_channels": self._with_channels,
                "dtype": self._dtype}


@register_wrapper
class SimpleTransformationWithHaloSource(SimpleTransformationSource):
    """Apply a value-only transformation that needs a halo, reading through it on each block.

    Like :class:`SimpleTransformationSource`, but the read region is extended by ``halo`` on each
    spatial axis (clamped to the data bounds) before the transformation, and the result is cropped
    back to the requested region. Use this for transformations whose output near a block boundary
    depends on neighbouring values (e.g. a local filter).

    Args:
        source: The wrapped source-like object.
        transformation: The value transformation to apply to each (haloed) read block.
        halo: The per-axis halo, with one entry per spatial axis.
        with_channels: Whether the wrapped source has a leading channel axis (see
            :class:`SimpleTransformationSource`).
        dtype: The output dtype. Defaults to the wrapped source's dtype.
    """

    def __init__(self, source: SourceLike, transformation: Callable, halo: Sequence[int], *,
                 with_channels: bool = False, dtype: Optional[np.dtype] = None) -> None:
        super().__init__(source, transformation, with_channels=with_channels, dtype=dtype)
        halo = tuple(int(h) for h in halo)
        if len(halo) != self.ndim:
            raise ValueError(f"Expect halo of length {self.ndim}, got {len(halo)}.")
        self._halo = halo

    def _extend_halo(self, roi: Tuple[slice, ...]) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
        """Extend ``roi`` by the halo (clamped to bounds) and return it with the local crop."""
        shape = self.shape
        extended_index, local_crop = [], []
        for sl, ha, sh in zip(roi, self._halo, shape):
            idx_start = int(sl.start) if sl.start is not None else 0
            idx_stop = int(sl.stop) if sl.stop is not None else sh
            start = max(idx_start - ha, 0)
            stop = min(idx_stop + ha, sh)
            extended_index.append(slice(start, stop))
            crop_len = stop - start
            left_halo = idx_start - start
            right_halo = stop - idx_stop
            local_crop.append(slice(left_halo, crop_len - right_halo))
        return tuple(extended_index), tuple(local_crop)

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the transformed data at ``roi``, reading through the halo."""
        index, local_crop = self._extend_halo(roi)
        if self._with_channels:
            index = (slice(None),) + index
            local_crop = (slice(None),) + local_crop
        out = self._transformation(self._source[index])
        return out[local_crop]

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        params = super()._params()
        params["halo"] = self._halo
        return params


@register_wrapper
class TransformationSource(WrapperSource):
    """Apply a coordinate-aware transformation to the wrapped source on read.

    The transformation may depend on both the data values and the read coordinates, so its
    signature is ``transformation(block, roi)`` where ``roi`` is the tuple of slices being read.

    Args:
        source: The wrapped source-like object.
        transformation: The transformation to apply, called as ``transformation(block, roi)``.
        dtype: The output dtype. Defaults to the wrapped source's dtype.
    """

    def __init__(self, source: SourceLike, transformation: Callable, *,
                 dtype: Optional[np.dtype] = None) -> None:
        super().__init__(source)
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self._transformation = transformation
        self._dtype = None if dtype is None else np.dtype(dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype if self._dtype is not None else self._source.dtype

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the transformed data at ``roi`` (a full tuple of concrete slices)."""
        return self._transformation(self._source[roi], roi)

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"transformation": self._transformation, "dtype": self._dtype}


@register_wrapper
class MultiTransformationSource(WrapperSource):
    """Apply a transformation jointly to blocks read from several equally-shaped sources.

    The transformation depends only on the data values, so its signature is
    ``transformation(*blocks)`` (or ``transformation(blocks)`` when ``apply_to_list`` is set). All
    wrapped sources must have the same shape.

    Args:
        transformation: The transformation to apply to the per-source blocks.
        sources: The wrapped source-like objects (at least one).
        apply_to_list: Whether the blocks are passed as a single list argument instead of being
            splatted as individual arguments.
        dtype: The output dtype. Defaults to the first source's dtype.
    """

    def __init__(self, transformation: Callable, *sources: SourceLike,
                 apply_to_list: bool = False, dtype: Optional[np.dtype] = None) -> None:
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        if len(sources) == 0:
            raise ValueError("Expect at least one source.")
        self._sources: List[Source] = [as_source(s) for s in sources]
        if any(s.shape != self._sources[0].shape for s in self._sources[1:]):
            raise ValueError("All sources must have the same shape.")
        # The first source backs the inherited metadata (shape / chunks / shards).
        super().__init__(self._sources[0])
        self._transformation = transformation
        self._apply_to_list = bool(apply_to_list)
        self._dtype = None if dtype is None else np.dtype(dtype)

    @property
    def _wrapped_sources(self) -> Tuple[Source, ...]:
        return tuple(self._sources)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype if self._dtype is not None else self._sources[0].dtype

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the joint transformation of all sources at ``roi``."""
        inputs = [s[roi] for s in self._sources]
        return self._transformation(inputs) if self._apply_to_list else self._transformation(*inputs)

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"transformation": self._transformation, "apply_to_list": self._apply_to_list,
                "dtype": self._dtype}

    @classmethod
    def _from_wrapped(cls, sources: Sequence[Source], params: dict) -> "MultiTransformationSource":
        """Rebuild from all wrapped sources (transformation comes first, positionally)."""
        params = dict(params)
        transformation = params.pop("transformation")
        return cls(transformation, *sources, **params)
