"""Read-only :class:`Source` over a (remote or local) WebKnossos layer, presented in ZYX order."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Source, SourceSpec


def _start_stop(index: Any, size: int) -> Tuple[int, int]:
    """Normalize a single ZYX index entry into an in-bounds ``(start, stop)`` pair."""
    if isinstance(index, slice):
        start, stop, step = index.indices(size)
        if step != 1:
            raise ValueError("WebKnossosSource only supports a step of 1.")
        return start, stop
    index = int(index)
    if index < 0:
        index += size
    return index, index + 1


def _open_layer(dataset_name_or_url: str, organization_id: Optional[str], layer_name: str, mag: int) -> Any:
    """Open a WebKnossos dataset (remote, annotation, or local folder) and return the layer's mag view."""
    import webknossos as wk

    if os.path.isdir(dataset_name_or_url):  # a local WebKnossos dataset folder
        dataset = wk.Dataset.open(dataset_name_or_url)
    else:
        try:
            dataset = wk.Dataset.open_remote(
                dataset_name_or_url=dataset_name_or_url,
                organization_id=organization_id,
            )
        except ValueError:
            dataset = wk.Annotation.download(dataset_name_or_url).get_remote_annotation_dataset()

    try:
        return dataset.get_layer(layer_name).get_mag(mag)
    except IndexError:
        raise IndexError(f"The layer {layer_name!r} is not available. Choose one of {dataset.layers}.")


class WebKnossosSource(Source):
    """A ZYX-ordered, read-only :class:`Source` view of a WebKnossos layer.

    WebKnossos stores data in ``(x, y, z)`` order; this source exposes a 3D ``(z, y, x)`` numpy-order
    view (single channel only), transposing on read. ``offset`` and ``size`` are given in **Mag(1)
    (full-resolution) XYZ** coordinates (the convention of WebKnossos bounding boxes), defaulting to
    the layer's bounding box. The presented ``shape`` is at the opened magnification, i.e. the
    Mag(1) size divided by ``mag``; local indices are at that magnification and are scaled by ``mag``
    to address the Mag(1) coordinates that ``MagView.read`` expects.

    Not thread-safe: the remote layer handle is not safe to share across threads, so do not run the
    ``local`` backend with ``num_workers > 1`` over this source. For parallelism use the
    ``subprocess``/``slurm`` backends, where each worker reopens the source from its spec.

    Args:
        dataset_name_or_url: The WebKnossos dataset name or URL (or an annotation URL).
        organization_id: The organization id (required for opening by dataset name).
        layer_name: The name of the layer to open.
        mag: The magnification (resolution) level.
        offset: Optional Mag(1) XYZ origin of the view; defaults to the layer bbox ``topleft``.
        size: Optional Mag(1) XYZ size of the view; defaults to the layer bbox ``size``.
    """

    def __init__(
        self,
        dataset_name_or_url: str,
        organization_id: Optional[str] = None,
        layer_name: str = "",
        mag: int = 1,
        offset: Optional[Tuple[int, int, int]] = None,
        size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        import webknossos as wk

        self._dataset_name_or_url = dataset_name_or_url
        self._organization_id = organization_id
        self._layer_name = layer_name
        self._mag = int(mag)

        self._layer = _open_layer(dataset_name_or_url, organization_id, layer_name, mag)
        num_channels = int(self._layer.layer.num_channels)
        if num_channels != 1:
            raise ValueError(
                f"WebKnossosSource supports single-channel layers only, got {num_channels} channels."
            )
        # MagView.bounding_box is the data extent in Mag(1) coordinates (x, y, z) -- not
        # info.bounding_box, which is the channel-prefixed, shard-padded storage extent.
        bbox = self._layer.bounding_box
        topleft = bbox.topleft if offset is None else offset
        size_xyz = bbox.size if size is None else size
        self._offset = (int(topleft[0]), int(topleft[1]), int(topleft[2]))  # Mag(1) XYZ
        self._size = (int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2]))  # Mag(1) XYZ
        # The actually-readable extent at this magnification (Mag(1) size aligned + divided by mag).
        ml = wk.BoundingBox(topleft=self._offset, size=self._size).in_mag(wk.Mag(self._mag))
        self._size_ml = (int(ml.size[0]), int(ml.size[1]), int(ml.size[2]))  # mag-level XYZ
        info = self._layer.info
        chunk = info.chunk_shape  # mag-level XYZ
        self._chunks = (int(chunk[2]), int(chunk[1]), int(chunk[0]))  # ZYX
        self._dtype = np.dtype(info.voxel_type)

    @property
    def layer(self) -> Any:
        """The wrapped WebKnossos mag view."""
        return self._layer

    @property
    def shape(self) -> Tuple[int, ...]:
        """The ZYX shape of the view at the opened magnification."""
        return (self._size_ml[2], self._size_ml[1], self._size_ml[0])

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the layer."""
        return self._dtype

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The ZYX chunk shape of the layer."""
        return self._chunks

    @property
    def writable(self) -> bool:
        """WebKnossos sources are read-only."""
        return False

    def _getitem(self, roi: Tuple[slice, ...]) -> np.ndarray:
        import webknossos as wk

        z0, z1 = _start_stop(roi[0], self.shape[0])  # mag-level ZYX
        y0, y1 = _start_stop(roi[1], self.shape[1])
        x0, x1 = _start_stop(roi[2], self.shape[2])

        # Local indices are at the opened magnification; MagView.read expects a Mag(1) absolute
        # bounding box, so scale by mag (topleft = offset + local*mag, size = extent*mag).
        mag, (ox, oy, oz) = self._mag, self._offset
        wk_bbox = wk.BoundingBox(
            topleft=(ox + x0 * mag, oy + y0 * mag, oz + z0 * mag),  # Mag(1) XYZ
            size=((x1 - x0) * mag, (y1 - y0) * mag, (z1 - z0) * mag),  # Mag(1) XYZ
        )
        data = self._layer.read(absolute_bounding_box=wk_bbox)
        data = data[0]  # drop the single channel -> (x, y, z) at the opened magnification
        expected = (x1 - x0, y1 - y0, z1 - z0)
        if data.shape != expected:
            raise RuntimeError(
                f"WebKnossos read returned XYZ shape {data.shape}, expected {expected} (mag={mag}); "
                "this indicates a coordinate-system mismatch."
            )
        return np.transpose(data, (2, 1, 0))  # -> (z, y, x)

    def _setitem(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
        raise TypeError("WebKnossosSource is read-only.")

    def to_spec(self) -> SourceSpec:
        """Return a ``kind="webknossos"`` spec recording the dataset, layer, mag and ROI."""
        params: Dict[str, Any] = {
            "dataset_name_or_url": self._dataset_name_or_url,
            "organization_id": self._organization_id,
            "layer_name": self._layer_name,
            "mag": self._mag,
            "offset": list(self._offset),
            "size": list(self._size),
        }
        return SourceSpec(kind="webknossos", params=params)

    @staticmethod
    def reopen(spec: SourceSpec) -> "WebKnossosSource":
        """Reopen a WebKnossos source from its spec."""
        params = dict(spec.params)
        offset = params.pop("offset", None)
        size = params.pop("size", None)
        return WebKnossosSource(
            dataset_name_or_url=params["dataset_name_or_url"],
            organization_id=params.get("organization_id"),
            layer_name=params["layer_name"],
            mag=params.get("mag", 1),
            offset=None if offset is None else tuple(offset),
            size=None if size is None else tuple(size),
        )


def open_webknossos(
    dataset_name_or_url: str,
    organization_id: Optional[str] = None,
    layer_name: str = "",
    mag: int = 1,
    offset: Optional[Tuple[int, int, int]] = None,
    size: Optional[Tuple[int, int, int]] = None,
) -> WebKnossosSource:
    """Open a (remote) WebKnossos layer as a read-only ZYX :class:`Source`.

    Args:
        dataset_name_or_url: The WebKnossos dataset name or URL (or an annotation URL).
        organization_id: The organization id (required when opening by dataset name).
        layer_name: The name of the layer to open.
        mag: The magnification (resolution) level.
        offset: Optional absolute XYZ origin of the view; defaults to the layer bbox ``topleft``.
        size: Optional XYZ size of the view; defaults to the layer bbox ``size``.

    Returns:
        A :class:`WebKnossosSource`.
    """
    return WebKnossosSource(
        dataset_name_or_url=dataset_name_or_url,
        organization_id=organization_id,
        layer_name=layer_name,
        mag=mag,
        offset=offset,
        size=size,
    )
