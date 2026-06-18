"""Tests for the expanded data-format spec support (file formats, CloudVolume, WebKnossos)."""
import os

import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources import FileSource, as_source, open_source
from bioimage_py.sources.dispatch import from_spec
from bioimage_py.runner.distributed import _DistributedRunner


# --- file-backed formats ----------------------------------------------------------------------

def test_hdf5_spec_roundtrip(hdf5_factory, rng):
    pytest.importorskip("h5py")
    data = (rng.random((8, 16, 16)) * 255).astype("uint8")
    path, key = hdf5_factory(data, chunks=(4, 8, 8))

    src = open_source(path, key)
    assert src.shape == (8, 16, 16)
    assert src.chunks == (4, 8, 8)
    assert src.writable is False  # opened read-only
    spec = src.to_spec()
    assert spec.kind == "file" and spec.params["format"] == "hdf5"
    np.testing.assert_array_equal(from_spec(spec)[:], data)


def test_hdf5_requires_internal_path(hdf5_factory, rng):
    pytest.importorskip("h5py")
    data = (rng.random((4, 4)) * 255).astype("uint8")
    path, _ = hdf5_factory(data, key="vol")
    with pytest.raises(ValueError, match="internal_path"):
        open_source(path)


def test_tif_spec_roundtrip(tmp_path, rng):
    tifffile = pytest.importorskip("tifffile")
    data = (rng.random((8, 16, 16)) * 255).astype("uint8")
    path = str(tmp_path / "stack.tif")
    tifffile.imwrite(path, data)

    src = open_source(path)
    assert src.shape == (8, 16, 16)
    assert src.to_spec().internal_path == ""  # single multi-page stack
    np.testing.assert_array_equal(src[:], data)
    np.testing.assert_array_equal(from_spec(src.to_spec())[:], data)


def test_mrc_spec_roundtrip(tmp_path, rng):
    mrcfile = pytest.importorskip("mrcfile")
    data = (rng.random((8, 16, 16)) * 255).astype("uint8")
    path = str(tmp_path / "vol.mrc")
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)

    src = open_source(path)
    assert src.shape == (8, 16, 16)
    assert src.writable is False
    # mrc is read-only; round-trip is verified against the wrapper read (axis-flip is internal).
    np.testing.assert_array_equal(from_spec(src.to_spec())[:], src[:])


def test_mrc_readonly_setitem_raises(tmp_path, rng):
    mrcfile = pytest.importorskip("mrcfile")
    data = (rng.random((4, 8, 8)) * 255).astype("uint8")
    path = str(tmp_path / "vol.mrc")
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
    src = open_source(path)
    with pytest.raises(TypeError, match="read-only"):
        src[0:1, 0:1, 0:1] = 0


def test_knossos_read(knossos_factory, rng):
    pytest.importorskip("imageio")
    data = (rng.random((128, 128, 128)) * 255).astype("uint8")
    path = knossos_factory(data)

    src = open_source(path)  # extension-less folder -> knossos via folder_based
    assert src.shape == (128, 128, 128)
    assert src.dtype == np.dtype("uint8")
    assert src.to_spec().internal_path == "mag1"
    roi = (slice(0, 32), slice(16, 48), slice(8, 40))
    np.testing.assert_array_equal(src[roi], data[roi])


def test_nifti_spec_roundtrip(tmp_path, rng):
    nibabel = pytest.importorskip("nibabel")
    data = (rng.random((8, 16, 16)) * 255).astype("uint8")
    path = str(tmp_path / "vol.nii.gz")
    nibabel.save(nibabel.Nifti1Image(data, affine=np.eye(4)), path)

    src = open_source(path)
    assert src.writable is False
    np.testing.assert_array_equal(from_spec(src.to_spec())[:], src[:])


def test_msr_spec_roundtrip(tmp_path):
    pytest.importorskip("msr_reader")
    pytest.skip("Writing a synthetic .msr file is not supported by msr_reader; needs a sample file.")


# --- distributed parity: hdf5 input -> zarr output --------------------------------------------

def test_hdf5_input_zarr_output_gaussian_parity(hdf5_factory, zarr_factory, rng):
    import bioimage_cpp as bic

    pytest.importorskip("h5py")
    a = rng.random((40, 48)).astype("float32")
    path, key = hdf5_factory(a, chunks=(16, 16))
    ref = bic.filters.gaussian_smoothing(a, 2.0)

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        src = open_source(path, key)
        out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
        bp.filters.gaussian_smoothing(src, 2.0, output=out, block_shape=(16, 16),
                                      num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], ref, atol=1e-4, err_msg=f"nw={nw} job={job}")


# --- CloudVolume (offline file:// layer) ------------------------------------------------------

def test_cloudvolume_read_and_roundtrip(cloudvolume_factory, rng):
    pytest.importorskip("cloudvolume")
    from bioimage_py.sources import open_cloudvolume

    data = (rng.random((32, 48, 64)) * 255).astype("uint8")  # ZYX
    path = cloudvolume_factory(data, chunk_size=(16, 16, 16))

    src = open_cloudvolume(path, mip=0)
    assert src.shape == (32, 48, 64)  # ZYX
    assert src.chunks == (16, 16, 16)
    assert src.writable is True
    np.testing.assert_array_equal(src[:, :, :], data)
    roi = (slice(4, 20), slice(8, 40), slice(0, 32))
    np.testing.assert_array_equal(src[roi], data[roi])

    spec = src.to_spec()
    assert spec.kind == "cloudvolume"
    np.testing.assert_array_equal(from_spec(spec)[:, :, :], data)


def test_cloudvolume_write_roundtrip(cloudvolume_factory, rng):
    pytest.importorskip("cloudvolume")
    from bioimage_py.sources import open_cloudvolume

    data = (rng.random((32, 48, 64)) * 255).astype("uint8")
    path = cloudvolume_factory(shape=(32, 48, 64), dtype="uint8", chunk_size=(16, 16, 16), fill=0)

    src = open_cloudvolume(path, mip=0, fill_missing=True)
    src[0:16, 0:16, 0:16] = data[0:16, 0:16, 0:16]
    np.testing.assert_array_equal(src[0:16, 0:16, 0:16], data[0:16, 0:16, 0:16])


def test_cloudvolume_distributed_input_max_parity(cloudvolume_factory, rng):
    pytest.importorskip("cloudvolume")
    from bioimage_py.sources import open_cloudvolume

    data = (rng.random((24, 32, 32)) * 255).astype("uint8")
    path = cloudvolume_factory(data, chunk_size=(16, 16, 16))
    expected = float(data.max())

    assert np.isclose(bp.stats.max(open_cloudvolume(path), num_workers=1, block_shape=(16, 16, 16)), expected)
    assert np.isclose(
        bp.stats.max(open_cloudvolume(path), num_workers=3, block_shape=(16, 16, 16), job_type="subprocess"),
        expected,
    )


# --- distributed write-safety gate ------------------------------------------------------------

def test_readonly_output_rejected(tmp_path, rng):
    mrcfile = pytest.importorskip("mrcfile")
    data = (rng.random((4, 8, 8)) * 255).astype("uint8")
    path = str(tmp_path / "vol.mrc")
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
    out = open_source(path)
    with pytest.raises(ValueError, match="writable"):
        _DistributedRunner._require_reopenable([], [out], None)


def test_hdf5_output_rejected(hdf5_factory, rng):
    pytest.importorskip("h5py")
    data = (rng.random((8, 8)) * 255).astype("uint8")
    path, key = hdf5_factory(data)
    out = open_source(path, key, mode="r+")  # writable, but HDF5 is unsafe for distributed writes
    assert out.writable is True
    with pytest.raises(ValueError, match="HDF5"):
        _DistributedRunner._require_reopenable([], [out], None)


# --- WebKnossos (gated: needs the package and a live dataset) ----------------------------------

def test_webknossos_source_gated():
    pytest.importorskip("webknossos")
    url = os.environ.get("BIOIMAGE_PY_WK_URL")
    if not url:
        pytest.skip("Set BIOIMAGE_PY_WK_URL (and BIOIMAGE_PY_WK_ORG/LAYER) to test WebKnossos.")
    from bioimage_py.sources import open_webknossos

    src = open_webknossos(
        url,
        organization_id=os.environ.get("BIOIMAGE_PY_WK_ORG"),
        layer_name=os.environ.get("BIOIMAGE_PY_WK_LAYER", ""),
        mag=int(os.environ.get("BIOIMAGE_PY_WK_MAG", "1")),
    )
    assert src.ndim == 3
    assert src.writable is False
    spec = src.to_spec()
    assert spec.kind == "webknossos"
    block = src[0:8, 0:8, 0:8]
    assert block.shape == (8, 8, 8)


# --- zarr / n5 via open_source (FileSource) ---------------------------------------------------

def test_zarr_open_source_roundtrip(tmp_path, rng):
    import zarr

    data = rng.random((8, 16, 16)).astype("float32")

    # Direct-array path: open_source on a top-level array, internal_path defaults to "".
    apath = str(tmp_path / "arr.zarr")
    z = zarr.open_array(apath, mode="w", shape=data.shape, chunks=(4, 8, 8), dtype=data.dtype)
    z[:] = data
    src = open_source(apath)
    assert isinstance(src, FileSource)
    assert src.to_spec().params["format"] == "zarr"
    assert src.writable is False  # opened read-only by default
    np.testing.assert_array_equal(from_spec(src.to_spec())[:], data)

    # Container + internal_path: an array inside a group.
    gpath = str(tmp_path / "grp.zarr")
    group = zarr.open_group(gpath, mode="w")
    arr = group.create_array("vol", shape=data.shape, chunks=(4, 8, 8), dtype=data.dtype)
    arr[:] = data
    gsrc = open_source(gpath, "vol")
    assert gsrc.to_spec().internal_path == "vol"
    np.testing.assert_array_equal(from_spec(gsrc.to_spec())[:], data)


def test_n5_open_source_roundtrip(n5_factory, rng):
    pytest.importorskip("z5py")
    data = rng.random((8, 16, 16)).astype("float32")
    path, key = n5_factory(data, chunks=(4, 8, 8))

    src = open_source(path, key)
    assert isinstance(src, FileSource)
    assert src.to_spec().params["format"] == "n5"
    assert src.chunks == (4, 8, 8)
    np.testing.assert_array_equal(from_spec(src.to_spec())[:], data)


# --- live-object dispatch (register_source) ----------------------------------------------------

def test_live_h5py_dataset_dispatch(hdf5_factory, rng):
    h5py = pytest.importorskip("h5py")
    data = (rng.random((8, 16, 16)) * 255).astype("uint8")
    path, key = hdf5_factory(data, chunks=(4, 8, 8))

    with h5py.File(path, "r") as f:
        src = as_source(f[key])
        assert isinstance(src, FileSource)
        spec = src.to_spec()
        assert spec.kind == "file" and spec.params["format"] == "hdf5"
        assert spec.internal_path == key
        np.testing.assert_array_equal(src[:], data)
    # The spec reopens independently of the (now-closed) handle.
    np.testing.assert_array_equal(from_spec(spec)[:], data)


def test_live_cloudvolume_dispatch(cloudvolume_factory, rng):
    pytest.importorskip("cloudvolume")
    from cloudvolume import CloudVolume

    data = (rng.random((24, 32, 32)) * 255).astype("uint8")  # ZYX
    path = cloudvolume_factory(data, chunk_size=(16, 16, 16))

    vol = CloudVolume(path, mip=0, progress=False)
    src = as_source(vol)
    assert type(src).__name__ == "CloudVolumeSource"
    assert src.to_spec().kind == "cloudvolume"
    np.testing.assert_array_equal(src[:, :, :], data)
    np.testing.assert_array_equal(from_spec(src.to_spec())[:, :, :], data)


# --- image-stack folder-of-slices (glob) ------------------------------------------------------

def test_image_stack_folder_of_slices(tif_slices_factory, rng):
    pytest.importorskip("tifffile")
    data = (rng.random((5, 12, 10)) * 255).astype("uint8")
    folder = tif_slices_factory(data)

    src = open_source(folder, "*.tif")
    assert src.shape == (5, 12, 10)
    assert src.chunks == (1, 12, 10)  # one slice per chunk
    np.testing.assert_array_equal(src[:], data)
    np.testing.assert_array_equal(from_spec(src.to_spec())[:], data)


# --- CloudVolume as a distributed output ------------------------------------------------------

def test_cloudvolume_distributed_output_gaussian_parity(cloudvolume_factory, zarr_factory, rng):
    import bioimage_cpp as bic

    pytest.importorskip("cloudvolume")
    from bioimage_py.sources import open_cloudvolume

    # Volume size is an exact multiple of the chunk shape, so every block write is chunk-aligned.
    a = rng.random((32, 48, 64)).astype("float32")
    z_in = zarr_factory(a, chunks=(16, 16, 16))
    ref = bic.filters.gaussian_smoothing(a, 2.0)

    for nw, job in [(1, "local"), (3, "subprocess")]:
        out_path = cloudvolume_factory(shape=(32, 48, 64), dtype="float32", chunk_size=(16, 16, 16), fill=0)
        out = open_cloudvolume(out_path, mip=0, fill_missing=True)
        bp.filters.gaussian_smoothing(z_in, 2.0, output=out, block_shape=(16, 16, 16),
                                      num_workers=nw, job_type=job)
        result = open_cloudvolume(out_path, mip=0)[:, :, :]
        np.testing.assert_allclose(result, ref, atol=1e-4, err_msg=f"nw={nw} job={job}")


# --- multi-block knossos ----------------------------------------------------------------------

def test_knossos_multiblock_read(knossos_factory, rng):
    pytest.importorskip("imageio")
    data = (rng.random((256, 128, 128)) * 255).astype("uint8")  # two blocks along z
    path = knossos_factory(data)

    src = open_source(path)
    assert src.shape == (256, 128, 128)
    # A roi straddling the z block boundary exercises multi-block assembly.
    roi = (slice(120, 140), slice(0, 64), slice(0, 96))
    np.testing.assert_array_equal(src[roi], data[roi])


# --- read-only source metadata ----------------------------------------------------------------

def test_readonly_source_metadata(tmp_path, rng):
    mrcfile = pytest.importorskip("mrcfile")
    data = (rng.random((6, 12, 10)) * 255).astype("uint8")
    path = str(tmp_path / "vol.mrc")
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
    src = open_source(path)
    assert src.ndim == 3
    assert src.chunks is None
    assert src.shards is None
