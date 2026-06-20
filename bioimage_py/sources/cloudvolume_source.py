"""Writable :class:`Source` over a CloudVolume (precomputed) layer, presented in ZYX order."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Source, SourceSpec


def _start_stop(index: Any, size: int) -> Tuple[int, int]:
    """Normalize a single ZYX index entry into an in-bounds ``(start, stop)`` pair."""
    if isinstance(index, slice):
        start, stop, step = index.indices(size)
        if step != 1:
            raise ValueError("CloudVolumeSource only supports a step of 1.")
        return start, stop
    index = int(index)
    if index < 0:
        index += size
    return index, index + 1


class CloudVolumeSource(Source):
    """A ZYX-ordered, writable :class:`Source` view of a CloudVolume layer.

    CloudVolume stores data in ``(x, y, z, channel)`` order; this source exposes a 3D ``(z, y, x)``
    numpy-order view (single channel only), transposing on read and write. Indices are local to the
    source origin (``offset``) and translated to absolute CloudVolume coordinates internally.

    Not thread-safe: the CloudVolume handle is not safe to share across threads, so do not run the
    ``local`` backend with ``num_workers > 1`` over this source. For parallelism use the
    ``subprocess``/``slurm`` backends, where each worker reopens the source from its spec; concurrent
    block writes must still be chunk-aligned (the runner's write-safety guard enforces this).

    Args:
        volume: An opened CloudVolume (precomputed) handle.
        offset: Absolute XYZ origin of the view; defaults to the volume's ``voxel_offset``.
        size: XYZ size of the view; defaults to the volume's ``volume_size``.
        open_params: The constructor parameters used to (re)open the volume, recorded in the spec.
    """

    def __init__(
        self,
        volume: Any,
        offset: Optional[Tuple[int, int, int]] = None,
        size: Optional[Tuple[int, int, int]] = None,
        open_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._vol = volume
        if int(volume.shape[3]) != 1:
            raise ValueError(
                f"CloudVolumeSource supports single-channel volumes only, got {int(volume.shape[3])} channels."
            )
        self._offset = tuple(int(v) for v in (offset if offset is not None else volume.voxel_offset))
        size_xyz = tuple(int(v) for v in (size if size is not None else volume.volume_size))
        self._size = size_xyz  # XYZ
        self._open_params = dict(open_params or {})

    @property
    def volume(self) -> Any:
        """The wrapped CloudVolume handle."""
        return self._vol

    @property
    def shape(self) -> Tuple[int, ...]:
        """The ZYX shape of the view."""
        return (self._size[2], self._size[1], self._size[0])

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the volume."""
        return np.dtype(self._vol.dtype)

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The ZYX chunk shape of the volume."""
        cs = [int(c) for c in self._vol.chunk_size]
        return (cs[2], cs[1], cs[0])

    @property
    def writable(self) -> bool:
        """CloudVolume sources support writing."""
        return True

    def _abs_bounds(self, roi: Tuple[slice, ...]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Return absolute XYZ ``(start, stop)`` bounds for a ZYX roi."""
        if not isinstance(roi, tuple):
            roi = (roi,)
        roi = roi + (slice(None),) * (3 - len(roi))
        z0, z1 = _start_stop(roi[0], self.shape[0])
        y0, y1 = _start_stop(roi[1], self.shape[1])
        x0, x1 = _start_stop(roi[2], self.shape[2])
        ox, oy, oz = self._offset
        return (ox + x0, ox + x1), (oy + y0, oy + y1), (oz + z0, oz + z1)

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        (x0, x1), (y0, y1), (z0, z1) = self._abs_bounds(roi)
        block = np.asarray(self._vol[x0:x1, y0:y1, z0:z1])  # (x, y, z, c)
        block = block[..., 0]  # drop the single channel
        return block.transpose(2, 1, 0)  # -> (z, y, x)

    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        (x0, x1), (y0, y1), (z0, z1) = self._abs_bounds(roi)
        arr = np.asarray(value).transpose(2, 1, 0)[..., None]  # (z,y,x) -> (x,y,z,1)
        self._vol[x0:x1, y0:y1, z0:z1] = arr

    def to_spec(self) -> SourceSpec:
        """Return a ``kind="cloudvolume"`` spec recording the cloudpath, open params and ROI."""
        params = dict(self._open_params)
        params["offset"] = list(self._offset)
        params["size"] = list(self._size)
        return SourceSpec(kind="cloudvolume", path=str(self._vol.cloudpath), params=params)

    @staticmethod
    def reopen(spec: SourceSpec) -> "CloudVolumeSource":
        """Reopen a CloudVolume source from its spec."""
        params = dict(spec.params)
        offset = params.pop("offset", None)
        size = params.pop("size", None)
        return open_cloudvolume(
            spec.path,
            offset=None if offset is None else tuple(offset),
            size=None if size is None else tuple(size),
            **params,
        )


def open_cloudvolume(
    cloudpath: str,
    mip: int = 0,
    fill_missing: bool = False,
    bounded: bool = True,
    cache: bool = False,
    non_aligned_writes: bool = True,
    offset: Optional[Tuple[int, int, int]] = None,
    size: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any,
) -> CloudVolumeSource:
    """Open a CloudVolume (precomputed) layer as a writable ZYX :class:`Source`.

    Args:
        cloudpath: The CloudVolume cloudpath (e.g. ``"precomputed://..."`` or ``"file://..."``).
        mip: The resolution (mip) level to open.
        fill_missing: Whether to zero-fill missing chunks instead of raising. For a *writable*
            output whose volume size is not a multiple of the chunk size, set this to ``True`` so
            the partial boundary chunks can be read-modify-written into a fresh layer.
        bounded: Whether reads/writes are restricted to the volume bounds.
        cache: Whether to enable CloudVolume's local cache.
        non_aligned_writes: Whether to allow writes that are not chunk-aligned (needed for the
            partial blocks at the volume boundary in block-wise writes).
        offset: Optional absolute XYZ origin of the view; defaults to the layer's ``voxel_offset``.
        size: Optional XYZ size of the view; defaults to the layer's ``volume_size``.
        kwargs: Extra keyword arguments forwarded to ``CloudVolume``.

    Returns:
        A :class:`CloudVolumeSource`.
    """
    from cloudvolume import CloudVolume

    open_params: Dict[str, Any] = dict(
        mip=mip,
        fill_missing=fill_missing,
        bounded=bounded,
        cache=cache,
        non_aligned_writes=non_aligned_writes,
    )
    open_params.update(kwargs)
    volume = CloudVolume(cloudpath, progress=False, **open_params)
    return CloudVolumeSource(volume, offset=offset, size=size, open_params=open_params)
