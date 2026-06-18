"""Tests for the io format registry and extension/format inference."""
import pytest

from bioimage_py.io import (
    constructor_for_format,
    format_for_extension,
    is_writable_format,
    supported_extensions,
    supported_formats,
)
from bioimage_py.io.files import infer_extension, infer_format


def test_readonly_formats_always_registered():
    # The read-only wrappers register unconditionally (their heavy import is guarded inside).
    assert {"mrc", "nifti", "msr", "folder"} <= set(supported_formats())
    assert {".mrc", ".rec", ".nii", ".nii.gz", ".msr", ".tif", ".tiff", ""} <= set(supported_extensions())


def test_is_writable_format_readonly():
    for fmt in ("mrc", "nifti", "msr", "folder"):
        assert is_writable_format(fmt) is False


def test_is_writable_format_writable():
    pytest.importorskip("h5py")
    assert is_writable_format("hdf5") is True


def test_format_for_extension():
    assert format_for_extension(".mrc") == "mrc"
    assert format_for_extension(".nii.gz") == "nifti"
    assert format_for_extension(".tif") == "folder"
    assert format_for_extension("") == "folder"
    assert format_for_extension(".unknownext") is None


def test_constructor_for_format_unknown_raises():
    with pytest.raises(ValueError, match="Unknown or unavailable format"):
        constructor_for_format("definitely-not-a-format")


def test_constructor_for_format_known():
    assert callable(constructor_for_format("mrc"))


def test_infer_extension():
    assert infer_extension("foo.nii.gz") == ".nii.gz"
    assert infer_extension("foo.h5") == ".h5"
    assert infer_extension("/some/folder") == ""
    assert infer_extension("/some/folder/") == ""  # trailing slash stripped
    assert infer_extension("FOO.H5") == ".h5"  # lower-cased
    assert infer_extension("foo.NII.GZ") == ".nii.gz"


def test_infer_format():
    assert infer_format("foo.mrc") == "mrc"
    assert infer_format("foo.nii.gz") == "nifti"
    with pytest.raises(ValueError, match="Could not infer"):
        infer_format("foo.unknownext")
