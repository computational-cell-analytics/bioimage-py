"""TensorStore-backed neuroglancer-precomputed source, presented in local ZYX order."""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .base import Source, SourceSpec

PathLike = Union[os.PathLike, str]


def _as_tuple(values: Sequence[int], *, name: str) -> Tuple[int, int, int]:
    """Normalize and validate a three-dimensional metadata value."""
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must have three entries, got {result!r}.")
    return result


def _validate_expected(name: str, expected: Optional[Sequence[int]], actual: Tuple[int, ...]) -> None:
    """Reject caller-supplied metadata that does not match the opened layer."""
    if expected is None:
        return
    normalized = _as_tuple(expected, name=name)
    if normalized != actual:
        raise ValueError(
            f"TensorStore precomputed {name} does not match the layer: "
            f"expected {normalized}, found {actual}."
        )


class TensorStorePrecomputedSource(Source):
    """A local-ZYX :class:`Source` over a TensorStore neuroglancer-precomputed layer.

    TensorStore exposes precomputed data in absolute ``(x, y, z, channel)`` coordinates. This
    adapter requires exactly those dimension labels and a single channel, and presents the full
    spatial domain as a local, zero-based ``(z, y, x)`` array. Reads and writes are transposed and
    translated to the absolute TensorStore domain internally.

    The TensorStore read-chunk shape is exposed as :attr:`chunks`; the write-chunk shape (a shard
    for sharded precomputed data) is exposed as :attr:`shards`. This lets the runner keep all
    blocks that touch the same TensorStore write chunk on one worker and avoid concurrent shard
    updates.

    Optional geometry and dtype arguments are expectations, not an arbitrary ROI definition. If
    supplied, they must match the layer metadata. This keeps the local origin aligned with the
    TensorStore chunk grids, which is required for safe shard routing.

    Args:
        path: Filesystem path of the neuroglancer-precomputed layer.
        mode: ``"r"`` (default) for read-only access or ``"r+"`` for synchronous writes.
        offset_xyz: Optional expected absolute XYZ origin.
        size_zyx: Optional expected spatial shape in ZYX order.
        chunks_zyx: Optional expected read-chunk shape in ZYX order.
        dtype: Optional expected numpy dtype.
    """

    def __init__(
        self,
        path: PathLike,
        mode: str = "r",
        *,
        offset_xyz: Optional[Sequence[int]] = None,
        size_zyx: Optional[Sequence[int]] = None,
        chunks_zyx: Optional[Sequence[int]] = None,
        dtype: Optional[Union[str, np.dtype, type]] = None,
    ) -> None:
        if mode not in ("r", "r+"):
            raise ValueError(
                f"TensorStorePrecomputedSource mode must be 'r' or 'r+', got {mode!r}."
            )

        # TensorStore is an optional dependency; importing bioimage_py must not require it.
        import tensorstore as ts

        self._path = os.path.abspath(os.fspath(path))
        self._mode = mode
        self._store = ts.open(
            {
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": self._path},
            },
            open=True,
            read=True,
            write=mode == "r+",
        ).result()

        domain = self._store.domain
        if domain.rank != 4:
            raise ValueError(
                "TensorStorePrecomputedSource requires a rank-4 (x, y, z, channel) layer, "
                f"got rank {domain.rank}."
            )
        labels = tuple(domain.labels)
        if labels != ("x", "y", "z", "channel"):
            raise ValueError(
                "TensorStorePrecomputedSource requires dimension labels "
                f"('x', 'y', 'z', 'channel'), got {labels!r}."
            )

        origin_xyzc = tuple(int(value) for value in domain.inclusive_min)
        shape_xyzc = tuple(int(value) for value in domain.shape)
        if shape_xyzc[3] != 1:
            raise ValueError(
                "TensorStorePrecomputedSource supports single-channel layers only, "
                f"got {shape_xyzc[3]} channels."
            )

        read_chunk_xyzc = tuple(int(value) for value in self._store.chunk_layout.read_chunk.shape)
        write_chunk_xyzc = tuple(int(value) for value in self._store.chunk_layout.write_chunk.shape)
        grid_origin_xyzc = tuple(int(value) for value in self._store.chunk_layout.grid_origin)
        if read_chunk_xyzc[3] != 1 or write_chunk_xyzc[3] != 1:
            raise ValueError(
                "TensorStorePrecomputedSource requires read and write chunks with one channel, "
                f"got {read_chunk_xyzc!r} and {write_chunk_xyzc!r}."
            )
        if grid_origin_xyzc != origin_xyzc:
            raise ValueError(
                "TensorStorePrecomputedSource requires the chunk grid to start at the layer "
                f"domain origin for safe shard routing, got grid origin {grid_origin_xyzc!r} "
                f"and domain origin {origin_xyzc!r}."
            )

        self._offset_xyz = origin_xyzc[:3]
        self._channel_origin = origin_xyzc[3]
        self._shape = shape_xyzc[:3][::-1]
        self._chunks = read_chunk_xyzc[:3][::-1]
        self._shards = write_chunk_xyzc[:3][::-1]
        self._dtype = np.dtype(self._store.dtype.numpy_dtype)

        _validate_expected("offset_xyz", offset_xyz, self._offset_xyz)
        _validate_expected("size_zyx", size_zyx, self._shape)
        _validate_expected("chunks_zyx", chunks_zyx, self._chunks)
        if dtype is not None and np.dtype(dtype) != self._dtype:
            raise ValueError(
                "TensorStore precomputed dtype does not match the layer: "
                f"expected {np.dtype(dtype)}, found {self._dtype}."
            )

    @property
    def store(self):
        """The wrapped TensorStore handle."""
        return self._store

    @property
    def shape(self) -> Tuple[int, ...]:
        """The spatial shape in local ZYX order."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the layer."""
        return self._dtype

    @property
    def chunks(self) -> Tuple[int, ...]:
        """The TensorStore read-chunk shape in ZYX order."""
        return self._chunks

    @property
    def shards(self) -> Tuple[int, ...]:
        """The TensorStore write-chunk (shard) shape in ZYX order."""
        return self._shards

    @property
    def writable(self) -> bool:
        """Whether the layer was opened in ``"r+"`` mode."""
        return self._mode == "r+"

    def _concrete_roi(self, roi: Tuple[slice, ...]) -> Tuple[slice, ...]:
        """Resolve normalized local slices to concrete, in-bounds ZYX coordinates."""
        concrete = []
        for axis_slice, size in zip(roi, self._shape):
            start, stop, step = axis_slice.indices(size)
            if step != 1:
                raise ValueError("TensorStorePrecomputedSource only supports a step of 1.")
            concrete.append(slice(start, stop))
        return tuple(concrete)

    def _absolute_index(self, roi: Tuple[slice, ...]) -> Tuple[slice, slice, slice, int]:
        """Translate a concrete local ZYX roi to an absolute TensorStore XYZC index."""
        z, y, x = roi
        ox, oy, oz = self._offset_xyz
        return (
            slice(ox + x.start, ox + x.stop),
            slice(oy + y.start, oy + y.stop),
            slice(oz + z.start, oz + z.stop),
            self._channel_origin,
        )

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        roi = self._concrete_roi(roi)
        value_xyz = self._store[self._absolute_index(roi)].read().result()
        return np.asarray(value_xyz).transpose(2, 1, 0)

    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        if not self.writable:
            raise TypeError(
                f"TensorStorePrecomputedSource opened in mode {self._mode!r} is read-only."
            )
        roi = self._concrete_roi(roi)
        roi_shape = tuple(axis_slice.stop - axis_slice.start for axis_slice in roi)
        value_zyx = np.broadcast_to(np.asarray(value, dtype=self._dtype), roi_shape)
        value_xyz = value_zyx.transpose(2, 1, 0)
        # Waiting on the future is deliberate: a completed block means its output is durable, and
        # write failures are attributed to the task that issued them.
        self._store[self._absolute_index(roi)].write(value_xyz).result()

    def to_spec(self) -> SourceSpec:
        """Return a serializable spec containing the verified layer metadata."""
        return SourceSpec(
            kind="tensorstore_precomputed",
            path=self._path,
            params={
                "mode": self._mode,
                "offset_xyz": list(self._offset_xyz),
                "size_zyx": list(self._shape),
                "chunks_zyx": list(self._chunks),
                "dtype": self._dtype.str,
            },
        )

    @staticmethod
    def reopen(spec: SourceSpec) -> "TensorStorePrecomputedSource":
        """Reopen a source from current or legacy adapter specs.

        Specs written by the earlier local shim did not record a mode; those reopen read-only.
        """
        params = dict(spec.params)
        params.setdefault("mode", "r")
        return TensorStorePrecomputedSource(spec.path, **params)


def open_tensorstore_precomputed(
    path: PathLike,
    mode: str = "r",
    *,
    offset_xyz: Optional[Sequence[int]] = None,
    size_zyx: Optional[Sequence[int]] = None,
    chunks_zyx: Optional[Sequence[int]] = None,
    dtype: Optional[Union[str, np.dtype, type]] = None,
) -> TensorStorePrecomputedSource:
    """Open a neuroglancer-precomputed layer as a local-ZYX :class:`Source`.

    Metadata arguments are optional consistency checks. The layer metadata is always authoritative;
    passing a mismatching expectation raises instead of silently producing incorrect coordinates.
    """
    return TensorStorePrecomputedSource(
        path,
        mode=mode,
        offset_xyz=offset_xyz,
        size_zyx=size_zyx,
        chunks_zyx=chunks_zyx,
        dtype=dtype,
    )
