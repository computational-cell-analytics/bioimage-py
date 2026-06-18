"""Native file-format IO layer: array-like handles for hdf5, zarr, n5, mrc, nifti, msr, tif, knossos."""
from .files import infer_format, open_file
from .registry import (
    constructor_for_format,
    format_for_extension,
    is_writable_format,
    register_format,
    supported_extensions,
    supported_formats,
)

__all__ = [
    "open_file",
    "infer_format",
    "register_format",
    "supported_extensions",
    "supported_formats",
    "format_for_extension",
    "constructor_for_format",
    "is_writable_format",
]
