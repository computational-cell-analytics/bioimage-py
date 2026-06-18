"""Read-only array-like wrapper for image stacks (a single multi-page file or a folder of slices)."""
from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent import futures
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import imageio.v3 as imageio
except ImportError:
    imageio = None

try:
    import tifffile
except ImportError:
    tifffile = None

from ._util import normalize_index, squeeze_singletons


def _require_imageio() -> None:
    """Raise a clear error if imageio is not installed (needed for non-tif image stacks)."""
    if imageio is None:
        raise AttributeError("imageio is required to read image stacks, but is not installed.")


class ImageStackFile(Mapping):
    """Root handle for an image stack: a single multi-page file (key ``""``) or a folder of slices.

    For a single multi-page file the only key is the empty string. For a folder, the key is a glob
    pattern (e.g. ``"*.tif"``) selecting the per-slice files.

    Args:
        path: The filepath to the file or folder.
        mode: The mode for opening; only read mode (``"r"``) is supported.
    """

    default_key = ""
    writable = False

    def __init__(self, path: Union[os.PathLike, str], mode: str = "r") -> None:
        self.path = path
        self.mode = mode
        self.file_name = os.path.split(self.path)[1]

    def __getitem__(self, key: str) -> "ImageStackDataset":
        # An empty key denotes a single multi-page file holding the whole stack.
        if key == "":
            if not os.path.isfile(self.path):
                raise ValueError(f"{self.path} needs to be a file to be loaded as image stack")
            if TifStackDataset.is_tif_stack(self.path):
                return TifStackDataset.from_stack(self.path)
            return ImageStackDataset.from_stack(self.path)

        # Otherwise the key is a glob pattern selecting the per-slice files in the folder.
        pattern = os.path.join(self.path, key)
        files = glob(pattern)
        if len(files) == 0:
            raise ValueError(f"Invalid file pattern {pattern}")
        if TifStackDataset.is_tif_slices(files):
            return TifStackDataset(files, sort_files=True)
        return ImageStackDataset(files, sort_files=True)

    def get_all_patterns(self) -> List[str]:
        """Return one glob pattern per file extension present in the folder."""
        all_files = glob(os.path.join(self.path, "*"))
        extensions = list(set(os.path.splitext(ff)[1] for ff in all_files))
        return ["*" + ext for ext in extensions]

    def __iter__(self):
        for pattern in self.get_all_patterns():
            yield pattern

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, name: object) -> bool:
        if name == "":
            return os.path.isfile(self.path)
        if not isinstance(name, str):
            return False
        return len(glob(os.path.join(self.path, name))) > 0

    def __enter__(self) -> "ImageStackFile":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}


class ImageStackDataset:
    """Array-like view of an image stack read slice-by-slice (or as a single volume) via imageio."""

    def get_im_shape_and_dtype(self, files: Sequence[str]) -> Tuple[Tuple[int, ...], np.dtype]:
        """Return the per-slice shape and dtype by reading the first file."""
        _require_imageio()
        im0 = imageio.imread(files[0])
        return im0.shape, im0.dtype

    def initialize_from_slices(self, files: List[str], sort_files: bool = True) -> None:
        """Initialize the dataset from a list of per-slice files."""
        if sort_files:
            files.sort()
        self.files = files

        n_slices = len(files)
        self.im_shape, dtype = self.get_im_shape_and_dtype(files)

        self._shape = (n_slices,) + self.im_shape
        self._chunks = (1,) + self.im_shape
        self._dtype = dtype
        self._size = int(np.prod(list(self._shape)))

    def initialize_from_stack(self, files: Any) -> None:
        """Initialize the dataset from a single multi-page stack file."""
        self.files = files
        self._volume = self._read_volume()

        self._shape = self._volume.shape
        self._chunks = None
        self._dtype = self._volume.dtype
        self._size = int(np.prod(list(self._shape)))

    @classmethod
    def from_pattern(cls, folder: str, pattern: str, n_threads: int = 1) -> "ImageStackDataset":
        """Build a dataset from a folder and a glob pattern."""
        files = glob(os.path.join(folder, pattern))
        return cls(files, n_threads=n_threads, sort_files=True)

    @classmethod
    def from_stack(cls, stack_path: str, n_threads: int = 1) -> "ImageStackDataset":
        """Build a dataset from a single multi-page stack file."""
        return cls(stack_path, n_threads=n_threads, is_stack=True)

    def __init__(self, files: Any, n_threads: int = 1, sort_files: bool = True, is_stack: bool = False) -> None:
        if is_stack:
            self.initialize_from_stack(files)
        else:
            self.initialize_from_slices(files, sort_files=sort_files)
        self.is_stack = is_stack
        self.n_threads = n_threads

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the data."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """The chunk shape (one slice per chunk for slice folders, else ``None``)."""
        return self._chunks

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the stack."""
        return self._shape

    @property
    def size(self) -> int:
        """The number of elements in the stack."""
        return self._size

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}

    def _read_image(self, index: int) -> np.ndarray:
        """Read a single slice file."""
        _require_imageio()
        return imageio.imread(self.files[index])

    def _read_volume(self) -> np.ndarray:
        """Read a single multi-page stack file."""
        _require_imageio()
        return imageio.imread(self.files)

    def _load_roi_from_stack(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Read a region of interest from the in-memory volume."""
        return self._volume[roi]

    def _load_roi_from_slices(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Read a region of interest by loading the overlapping slices."""
        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype=self.dtype)

        z0 = roi[0].start
        im_roi = roi[1:]

        def _load_and_write_image(z: int) -> None:
            z_abs = z + z0
            im = self._read_image(z_abs)
            assert im.shape == self.im_shape, f"{im.shape}, {self.im_shape}"
            data[z] = im[im_roi]

        with futures.ThreadPoolExecutor(self.n_threads) as tp:
            tasks = [tp.submit(_load_and_write_image, z) for z in range(roi_shape[0])]
            [t.result() for t in tasks]

        return data

    def __getitem__(self, key: Any) -> np.ndarray:
        roi, to_squeeze = normalize_index(key, self.shape)
        if self.is_stack:
            data = self._load_roi_from_stack(roi)
        else:
            data = self._load_roi_from_slices(roi)
        return squeeze_singletons(data, to_squeeze)


class TifStackDataset(ImageStackDataset):
    """Image-stack dataset that uses ``tifffile.memmap`` for (memory-mapped) tif access."""

    tif_exts = (".tif", ".tiff")

    @staticmethod
    def is_tif_slices(files: Sequence[str]) -> bool:
        """Return whether all files are memory-mappable tif slices."""
        if tifffile is None:
            return False
        f0 = files[0]
        ext = os.path.splitext(f0)[1]
        if ext.lower() not in TifStackDataset.tif_exts:
            return False
        try:
            for ff in files:
                tifffile.memmap(ff, mode="r")
        except ValueError:
            return False
        return True

    @staticmethod
    def is_tif_stack(path: str) -> bool:
        """Return whether ``path`` is a memory-mappable tif stack."""
        if tifffile is None:
            return False
        ext = os.path.splitext(path)[1]
        if ext.lower() not in TifStackDataset.tif_exts:
            return False
        try:
            tifffile.memmap(path, mode="r")
        except ValueError:
            return False
        return True

    def _read_image(self, index: int) -> np.ndarray:
        return tifffile.memmap(self.files[index], mode="r")

    def _read_volume(self) -> np.ndarray:
        return tifffile.memmap(self.files, mode="r")

    def get_im_shape_and_dtype(self, files: Sequence[str]) -> Tuple[Tuple[int, ...], np.dtype]:
        """Return the per-slice shape and dtype, validating that all slices agree."""
        im0 = tifffile.memmap(files[0], mode="r")
        im_shape = im0.shape
        im_shapes = [tifffile.memmap(ff, mode="r").shape for ff in files[1:]]
        if any(sh != im_shape for sh in im_shapes):
            raise ValueError("Incompatible shapes for Image Stack")
        return im_shape, im0.dtype
