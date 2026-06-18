"""Read-only :class:`Source` over a (remote) WebKnossos layer, presented in ZYX order."""
from __future__ import annotations

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
    """Open a WebKnossos dataset (or annotation) and return the requested layer's mag view."""
    import webknossos as wk

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
    view, transposing on read. Indices are local to the view origin (the layer's bounding-box
    ``topleft``, or an explicit ``offset``) and translated to absolute WebKnossos coordinates.

    Args:
        dataset_name_or_url: The WebKnossos dataset name or URL (or an annotation URL).
        organization_id: The organization id (required for opening by dataset name).
        layer_name: The name of the layer to open.
        mag: The magnification (resolution) level.
        offset: Optional absolute XYZ origin of the view; defaults to the layer bbox ``topleft``.
        size: Optional XYZ size of the view; defaults to the layer bbox ``size``.
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
        self._dataset_name_or_url = dataset_name_or_url
        self._organization_id = organization_id
        self._layer_name = layer_name
        self._mag = mag

        self._layer = _open_layer(dataset_name_or_url, organization_id, layer_name, mag)
        info = self._layer.info
        bbox = info.bounding_box

        topleft = bbox.topleft if offset is None else offset
        size_xyz = bbox.size if size is None else size
        self._offset = (int(topleft[0]), int(topleft[1]), int(topleft[2]))  # XYZ
        self._size = (int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2]))  # XYZ
        chunk = info.chunk_shape
        self._chunks = (int(chunk[2]), int(chunk[1]), int(chunk[0]))  # ZYX
        self._dtype = np.dtype(info.voxel_type)

    @property
    def layer(self) -> Any:
        """The wrapped WebKnossos mag view."""
        return self._layer

    @property
    def shape(self) -> Tuple[int, ...]:
        """The ZYX shape of the view."""
        return (self._size[2], self._size[1], self._size[0])

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

    def __getitem__(self, roi: Tuple[slice, ...]) -> np.ndarray:
        import webknossos as wk

        if not isinstance(roi, tuple):
            roi = (roi,)
        roi = roi + (slice(None),) * (3 - len(roi))
        z0, z1 = _start_stop(roi[0], self.shape[0])
        y0, y1 = _start_stop(roi[1], self.shape[1])
        x0, x1 = _start_stop(roi[2], self.shape[2])

        ox, oy, oz = self._offset
        wk_bbox = wk.BoundingBox(
            topleft=(ox + x0, oy + y0, oz + z0),  # XYZ absolute
            size=(x1 - x0, y1 - y0, z1 - z0),  # XYZ
        )
        data = self._layer.read(absolute_bounding_box=wk_bbox)
        data = data[0]  # drop the single channel -> (x, y, z)
        return np.transpose(data, (2, 1, 0))  # -> (z, y, x)

    def __setitem__(self, roi: Tuple[slice, ...], value: np.ndarray) -> None:
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
