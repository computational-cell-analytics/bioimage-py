"""Shared pytest fixtures."""
import os
import shutil
import uuid

import numpy as np
import pytest
import zarr

# Root for shared-filesystem test data, used by the slurm tests. Must be visible to compute
# nodes (node-local /tmp / pytest's tmp_path are not), so it is opted into via an env var.
_SHARED_ROOT = os.environ.get("BIOIMAGE_PY_SHARED_TMP")


def _write_zarr(path, array=None, chunks=None, *, shape=None, dtype=None, fill=None):
    """Create a fresh on-disk zarr array, optionally filled from ``array`` or ``fill``."""
    if array is not None:
        shape, dtype = array.shape, array.dtype
    z = zarr.open_array(path, mode="w", shape=shape, chunks=tuple(chunks), dtype=dtype)
    if array is not None:
        z[:] = array
    elif fill is not None:
        z[:] = fill
    return z


@pytest.fixture
def rng():
    """A seeded random generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def shared_tmp_path():
    """A per-test directory on the shared filesystem (the shared-FS analogue of tmp_path)."""
    base = os.path.join(_SHARED_ROOT, f"bp_test_{uuid.uuid4().hex[:12]}")
    os.makedirs(base, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def shared_zarr_factory(shared_tmp_path):
    """Like :func:`zarr_factory` but writes the arrays under the shared filesystem."""
    counter = {"i": 0}

    def _make(array=None, chunks=None, *, shape=None, dtype=None, fill=None):
        counter["i"] += 1
        path = os.path.join(shared_tmp_path, f"arr_{counter['i']}.zarr")
        return _write_zarr(path, array, chunks, shape=shape, dtype=dtype, fill=fill)

    return _make


@pytest.fixture
def hdf5_factory(tmp_path):
    """Return a factory writing arrays to fresh on-disk hdf5 files, returning ``(path, key)``."""
    h5py = pytest.importorskip("h5py")

    counter = {"i": 0}

    def _make(array, chunks=None, key="vol"):
        counter["i"] += 1
        path = str(tmp_path / f"arr_{counter['i']}.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset(key, data=array, chunks=None if chunks is None else tuple(chunks))
        return path, key

    return _make


@pytest.fixture
def n5_factory(tmp_path):
    """Return a factory writing arrays to fresh on-disk n5 files via z5py, returning ``(path, key)``."""
    z5py = pytest.importorskip("z5py")

    counter = {"i": 0}

    def _make(array, chunks=None, key="vol"):
        counter["i"] += 1
        path = str(tmp_path / f"arr_{counter['i']}.n5")
        with z5py.File(path, mode="w") as f:
            f.create_dataset(key, data=array, chunks=None if chunks is None else tuple(chunks))
        return path, key

    return _make


@pytest.fixture
def tif_slices_factory(tmp_path):
    """Return a factory writing a ZYX array as per-slice ``.tif`` files in a folder, returning the folder."""
    tifffile = pytest.importorskip("tifffile")

    counter = {"i": 0}

    def _make(array_zyx):
        counter["i"] += 1
        folder = tmp_path / f"slices_{counter['i']}"
        folder.mkdir()
        for z in range(array_zyx.shape[0]):
            tifffile.imwrite(str(folder / f"slice_{z:04d}.tif"), np.asarray(array_zyx[z]))
        return str(folder)

    return _make


@pytest.fixture
def cloudvolume_factory(tmp_path):
    """Return a factory creating a local ``file://`` precomputed layer from a ZYX array.

    Returns the cloudpath; the data is stored transposed to CloudVolume's XYZ order internally.
    """
    CloudVolume = pytest.importorskip("cloudvolume").CloudVolume

    counter = {"i": 0}

    def _make(array_zyx=None, *, shape=None, dtype="uint8", chunk_size=(16, 16, 16), fill=None):
        counter["i"] += 1
        path = "file://" + str(tmp_path / f"layer_{counter['i']}")
        if array_zyx is not None:
            shape, dtype = array_zyx.shape, array_zyx.dtype
        size_xyz = [int(shape[2]), int(shape[1]), int(shape[0])]
        chunk_xyz = [int(chunk_size[2]), int(chunk_size[1]), int(chunk_size[0])]
        info = CloudVolume.create_new_info(
            num_channels=1, layer_type="image", data_type=np.dtype(dtype).name, encoding="raw",
            resolution=[1, 1, 1], voxel_offset=[0, 0, 0], chunk_size=chunk_xyz, volume_size=size_xyz,
        )
        vol = CloudVolume(path, info=info, mip=0, fill_missing=True, non_aligned_writes=True, progress=False)
        vol.commit_info()
        if array_zyx is not None:
            vol[:, :, :] = np.asarray(array_zyx).transpose(2, 1, 0)[..., None]
        elif fill is not None:
            vol[:, :, :] = np.full(size_xyz + [1], fill, dtype=dtype)
        return path

    return _make


@pytest.fixture
def knossos_factory(tmp_path):
    """Return a factory writing a KNOSSOS ``mag1`` dataset, returning its path.

    The ZYX array's shape must be a multiple of 128 in every dimension; one 128^3 png block is
    written per grid cell (folders are nested ``mag1/x####/y####/z####/`` in KNOSSOS' x,y,z order).
    """
    imageio = pytest.importorskip("imageio.v3")

    block = 128
    counter = {"i": 0}

    def _make(array_zyx):
        assert array_zyx.ndim == 3 and array_zyx.dtype == np.uint8
        assert all(s % block == 0 for s in array_zyx.shape), "knossos shape must be a multiple of 128"
        counter["i"] += 1
        name = f"knossos_{counter['i']}"
        root = tmp_path / name
        cz, cy, cx = (s // block for s in array_zyx.shape)
        for gz in range(cz):
            for gy in range(cy):
                for gx in range(cx):
                    sub = array_zyx[
                        gz * block:(gz + 1) * block,
                        gy * block:(gy + 1) * block,
                        gx * block:(gx + 1) * block,
                    ]
                    block_dir = root / "mag1" / f"x{gx:04d}" / f"y{gy:04d}" / f"z{gz:04d}"
                    block_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{name}_mag1_x{gx:04d}_y{gy:04d}_z{gz:04d}.png"
                    # KnossosDataset reads each block via imread(...).reshape((128, 128, 128)).
                    imageio.imwrite(str(block_dir / fname), sub.reshape(block, block * block))
        return str(root)

    return _make


@pytest.fixture
def zarr_factory(tmp_path):
    """Return a factory that writes arrays to fresh on-disk zarr arrays.

    Usage: ``z = zarr_factory(array, chunks)`` or ``z = zarr_factory(shape=..., chunks=...,
    dtype=..., fill=...)`` for an empty (optionally filled) output array.
    """
    counter = {"i": 0}

    def _make(array=None, chunks=None, *, shape=None, dtype=None, fill=None):
        counter["i"] += 1
        path = str(tmp_path / f"arr_{counter['i']}.zarr")
        return _write_zarr(path, array, chunks, shape=shape, dtype=dtype, fill=fill)

    return _make
