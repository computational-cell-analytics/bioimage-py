"""Native file-format IO layer: array-like handles for hdf5, zarr, n5, mrc, nifti, msr, tif, knossos.

Indexing contract: the assemble-from-pieces wrappers (nifti, knossos, msr, image-stack) accept only
integer, slice (step 1), and ellipsis indices -- non-trivial steps (e.g. ``src[::2]``) are rejected,
unlike numpy / zarr / mrc which support them. Index these formats with contiguous slices.
"""
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
