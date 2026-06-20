"""Read-only array-like wrapper for KNOSSOS folder-based datasets."""
from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent import futures
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

from ._util import chunks_overlapping_roi, map_chunk_to_roi, normalize_index, squeeze_singletons

try:
    import imageio.v3 as imageio
except ImportError:
    imageio = None


class KnossosDataset:
    """Array-like view of one magnification level of a KNOSSOS dataset."""

    block_size = 128

    @staticmethod
    def _chunks_dim(dim_root: str) -> int:
        """Count the sub-folders (the grid extent) below ``dim_root``."""
        files = os.listdir(dim_root)
        files = [f for f in files if os.path.isdir(os.path.join(dim_root, f))]
        return len(files)

    def get_shape_and_grid(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Derive the (z, y, x) shape and block grid by walking the folder structure."""
        cx = self._chunks_dim(self.path)
        y_root = os.path.join(self.path, "x0000")
        cy = self._chunks_dim(y_root)
        z_root = os.path.join(y_root, "y0000")
        cz = self._chunks_dim(z_root)

        grid = (cz, cy, cx)
        shape = tuple(sh * self.block_size for sh in grid)
        return shape, grid

    def __init__(self, path: str, file_prefix: str, load_png: bool) -> None:
        self.path = path
        self.ext = "png" if load_png else "jpg"
        self.file_prefix = file_prefix

        self._ndim = 3
        self._chunks = self._ndim * (self.block_size,)
        self._shape, self._grid = self.get_shape_and_grid()
        self.n_threads = 1

    @property
    def dtype(self) -> np.dtype:
        """The (uint8) dtype of a KNOSSOS dataset."""
        return np.dtype("uint8")

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 3)."""
        return self._ndim

    @property
    def chunks(self) -> Tuple[int, ...]:
        """The chunk (block) shape, ``(128, 128, 128)``."""
        return self._chunks

    @property
    def shape(self) -> Tuple[int, ...]:
        """The (z, y, x) shape of this magnification level."""
        return self._shape

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}

    def load_block(self, grid_id: Sequence[int]) -> np.ndarray:
        """Load a single 128^3 block given its (z, y, x) grid id."""
        # KNOSSOS folders are stored in x, y, z order, so the grid id is reversed.
        block_path = ["%s%04i" % (dim, gid) for dim, gid in zip(("x", "y", "z"), grid_id[::-1])]
        dim_str = "_".join(block_path)
        fname = "%s_%s.%s" % (self.file_prefix, dim_str, self.ext)
        block_path.append(fname)
        path = os.path.join(self.path, *block_path)
        data = np.array(imageio.imread(path)).reshape(self._chunks)
        return data

    def _load_roi(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Assemble a region of interest from the overlapping blocks."""
        grid_points = chunks_overlapping_roi(roi, self.chunks)

        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype="uint8")

        def load_tile(grid_id: Sequence[int]) -> None:
            tile_data = self.load_block(grid_id)
            tile_bb, out_bb = map_chunk_to_roi(grid_id, roi, self.chunks)
            data[out_bb] = tile_data[tile_bb]

        if self.n_threads > 1:
            with futures.ThreadPoolExecutor(self.n_threads) as tp:
                tasks = [tp.submit(load_tile, sp) for sp in grid_points]
                [t.result() for t in tasks]
        else:
            [load_tile(sp) for sp in grid_points]
        return data

    def __getitem__(self, key: Any) -> np.ndarray:
        roi, to_squeeze = normalize_index(key, self.shape)
        return squeeze_singletons(self._load_roi(roi), to_squeeze)


class KnossosFile(Mapping):
    """Root handle for a KNOSSOS dataset, keyed by magnification level (e.g. ``"mag1"``).

    Args:
        path: Filepath to the KNOSSOS dataset folder.
        mode: The mode for opening the folder; only read mode (``"r"``) is supported.
        load_png: Whether the blocks are stored as png (else jpg).
    """

    default_key = "mag1"
    writable = False

    def __init__(self, path: Union[os.PathLike, str], mode: str = "r", load_png: bool = True) -> None:
        if not os.path.exists(os.path.join(path, "mag1")):
            raise RuntimeError("Not a knossos file structure")
        if imageio is None:
            raise AttributeError("imageio is required to read knossos datasets, but is not installed.")
        self.path = path
        self.mode = mode
        self.load_png = load_png
        self.file_name = os.path.split(self.path)[1]

    def __getitem__(self, key: str) -> KnossosDataset:
        sub_path = os.path.join(self.path, key)
        if not os.path.exists(sub_path):
            raise ValueError("Key %s does not exist" % key)
        if not os.path.isdir(sub_path) and key.startswith("mag"):
            raise ValueError("Key %s is not a valid knossos dataset" % key)
        file_prefix = "%s_%s" % (self.file_name, key)
        return KnossosDataset(sub_path, file_prefix, self.load_png)

    def __iter__(self):
        for name in os.listdir(self.path):
            if os.path.isdir(os.path.join(self.path, name)) and name.startswith("mag"):
                yield name

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, name: object) -> bool:
        # Match __iter__/__getitem__: only magnification folders are valid keys, so a stray
        # non-mag subfolder must not test as present.
        return (isinstance(name, str) and name.startswith("mag")
                and os.path.isdir(os.path.join(self.path, name.lstrip("/"))))

    def __enter__(self) -> "KnossosFile":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass

    @property
    def attrs(self) -> Dict[str, Any]:
        """Dummy attributes for compatibility with the hdf5/zarr API."""
        return {}
