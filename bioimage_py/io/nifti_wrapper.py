"""Read-only array-like wrapper for NIfTI (``.nii`` / ``.nii.gz``) files."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ._util import normalize_index, squeeze_singletons

try:
    import nibabel
except ImportError:
    nibabel = None


class NiftiFile(Mapping):
    """Root handle for a nifti file, exposing its data under the ``"data"`` key.

    Args:
        path: The filepath of the nifti file.
        mode: The mode for opening the file; only read mode (``"r"``) is supported.
    """

    default_key = "data"
    writable = False

    def __init__(self, path: Union[os.PathLike, str], mode: str = "r") -> None:
        if nibabel is None:
            raise AttributeError("nibabel is required for nifti images, but is not installed.")
        self.path = path
        self.mode = mode
        self.nifti = nibabel.load(self.path)

    def __enter__(self) -> "NiftiFile":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}

    def __getitem__(self, key: str) -> "NiftiDataset":
        if key != "data":
            raise KeyError(f"Could not find key {key}")
        return NiftiDataset(self.nifti)

    def __iter__(self):
        yield "data"

    def __len__(self) -> int:
        return 1

    def __contains__(self, name: object) -> bool:
        return name == "data"


class NiftiDataset:
    """Array-like view of the data in a nifti file (presented in reversed/C axis order)."""

    def __init__(self, data: Any) -> None:
        self._data = data

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the data."""
        return self._data.get_data_dtype()

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._data.ndim

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The chunk shape; nifti files are unchunked, so this is ``None``."""
        return None

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the data (reversed relative to the nibabel axis order)."""
        return self._data.shape[::-1]

    @property
    def size(self) -> int:
        """The number of elements in the data."""
        return int(np.prod(self._data.shape))

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}

    def __getitem__(self, key: Any) -> np.ndarray:
        key, to_squeeze = normalize_index(key, self.shape)
        transposed_key = key[::-1]
        data = self._data.dataobj[transposed_key].T
        return squeeze_singletons(data, to_squeeze).copy()
