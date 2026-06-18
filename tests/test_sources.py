"""Tests for the Source abstraction and dispatch."""
import numpy as np
import pytest

from bioimage_py.sources import ArraySource, Source, as_source, from_spec


def test_as_source_idempotent(rng):
    src = as_source(rng.random((8, 8)))
    assert isinstance(src, Source)
    assert as_source(src) is src


def test_as_source_numpy_metadata(rng):
    a = rng.random((6, 7)).astype("float32")
    src = as_source(a)
    assert isinstance(src, ArraySource)
    assert src.shape == (6, 7)
    assert src.dtype == np.dtype("float32")
    assert src.ndim == 2
    assert src.chunks is None
    np.testing.assert_array_equal(src[(slice(None), slice(None))], a)


def test_numpy_to_spec_raises(rng):
    src = as_source(rng.random((4, 4)))
    with pytest.raises(ValueError, match="numpy"):
        src.to_spec()


def test_string_input_rejected():
    with pytest.raises(TypeError, match="strings / file paths"):
        as_source("/some/path.zarr")


def test_zarr_spec_roundtrip(zarr_factory, rng):
    a = rng.random((12, 10)).astype("float32")
    z = zarr_factory(a, chunks=(4, 5))
    src = as_source(z)
    assert src.shape == (12, 10)
    assert src.chunks == (4, 5)
    spec = src.to_spec()
    assert spec.kind == "zarr"
    reopened = from_spec(spec)
    np.testing.assert_array_equal(reopened[(slice(None), slice(None))], a)


def test_setitem(zarr_factory):
    z = zarr_factory(shape=(8, 8), chunks=(4, 4), dtype="float32", fill=0.0)
    src = as_source(z)
    src[(slice(0, 4), slice(0, 4))] = np.ones((4, 4), dtype="float32")
    assert src[(slice(0, 4), slice(0, 4))].sum() == 16


def test_z5py_spec_roundtrip(n5_factory, rng):
    pytest.importorskip("z5py")
    import z5py

    a = rng.random((12, 10)).astype("float32")
    path, key = n5_factory(a, chunks=(4, 5))

    # A live z5py dataset dispatches to ArraySource with a kind="z5py" spec.
    with z5py.File(path, mode="r") as f:
        src = as_source(f[key])
        assert isinstance(src, ArraySource)
        assert src.chunks == (4, 5)
        spec = src.to_spec()
        assert spec.kind == "z5py"
    np.testing.assert_array_equal(from_spec(spec)[(slice(None), slice(None))], a)
