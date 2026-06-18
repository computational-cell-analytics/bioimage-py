"""Read-only array-like wrapper for ``.mrc`` / ``.rec`` electron-microscopy files."""
from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import mrcfile
except ImportError:
    mrcfile = None


class MRCDataset:
    """Array-like view of the single dataset in an mrc file."""

    def __init__(self, data_object: Any) -> None:
        # Flip the data's axis to meet the (z, y, x) axis convention.
        self._data = np.flip(data_object, axis=1) if data_object.ndim == 3 else np.flip(data_object, axis=0)

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the data."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._data.ndim

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The chunk shape; mrc files are unchunked, so this is ``None``."""
        return None

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the data."""
        return self._data.shape

    @property
    def size(self) -> int:
        """The number of elements in the data."""
        return self._data.size

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}

    def __getitem__(self, key: Any) -> np.ndarray:
        return self._data[key].copy()


class MRCFile(Mapping):
    """Root handle for an mrc / rec file, exposing its data under the ``"data"`` key.

    Args:
        path: The filepath of the mrc file.
        mode: The mode for opening the file; only read mode (``"r"``) is supported.
    """

    default_key = "data"
    writable = False

    def __init__(self, path: Union[os.PathLike, str], mode: str = "r") -> None:
        self.path = path
        self.mode = mode
        if mrcfile is None:
            raise AttributeError("mrcfile is required to read mrc or rec files, but is not installed.")
        try:
            self._f = mrcfile.mmap(self.path, self.mode)
        except ValueError as e:
            # An old SerialEM acquisition can produce an unrecognised machine stamp; retry permissively.
            if (
                "Unrecognised machine stamp: 0x44 0x00 0x00 0x00" in str(e)
                or "Unrecognised machine stamp: 0x00 0x00 0x00 0x00" in str(e)
            ):
                try:
                    self._f = mrcfile.mmap(self.path, self.mode, permissive="True")
                except ValueError:
                    self._f = mrcfile.open(self.path, self.mode, permissive="True")
            else:  # Other kind of error -> try to open without mmap.
                try:
                    self._f = mrcfile.open(self.path, self.mode)
                except ValueError as e:
                    self._f = mrcfile.open(self.path, self.mode, permissive="True")
                    warnings.warn(
                        f"Opening mrcfile {self.path} failed with unknown error {e} without permissive opening."
                        "The file will still be opened but the contents may be incorrect."
                    )

    def __getitem__(self, key: str) -> MRCDataset:
        if key != "data":
            raise KeyError(f"Could not find key {key}")
        return MRCDataset(self._f.data)

    def __iter__(self):
        yield "data"

    def __len__(self) -> int:
        return 1

    def __contains__(self, name: object) -> bool:
        return name == "data"

    def __enter__(self) -> "MRCFile":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._f.close()

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}
