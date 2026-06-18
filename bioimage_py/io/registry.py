"""Registry mapping file extensions to format backends, with optional-dependency guards."""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Union

# Maps populated by ``register_format`` below. Each backend is registered behind a guarded import,
# so a missing optional dependency simply omits that format rather than failing at import time.
_EXT_TO_FORMAT: Dict[str, str] = {}
_FORMAT_TO_CONSTRUCTOR: Dict[str, Callable] = {}
_FORMAT_WRITABLE: Dict[str, bool] = {}

PathLike = Union[os.PathLike, str]


def register_format(
    name: str,
    extensions: Sequence[str],
    constructor: Callable,
    *,
    writable: bool = False,
) -> None:
    """Register a file-format backend.

    Args:
        name: The format name, recorded in the source spec (e.g. ``"hdf5"``, ``"mrc"``).
        extensions: The file extensions this format claims (lower-case; ``""`` for folders).
        constructor: A callable ``(path, mode="r", **kwargs)`` returning an array-like file handle.
        writable: Whether the format supports writing.
    """
    _FORMAT_TO_CONSTRUCTOR[name] = constructor
    _FORMAT_WRITABLE[name] = writable
    for ext in extensions:
        _EXT_TO_FORMAT.setdefault(ext.lower(), name)


def supported_formats() -> List[str]:
    """Return the names of all registered (i.e. installed) formats."""
    return list(_FORMAT_TO_CONSTRUCTOR.keys())


def supported_extensions() -> List[str]:
    """Return all file extensions for which a backend is installed."""
    return list(_EXT_TO_FORMAT.keys())


def format_for_extension(ext: str) -> Optional[str]:
    """Return the format name registered for ``ext``, or ``None``."""
    return _EXT_TO_FORMAT.get(ext.lower())


def constructor_for_format(name: str) -> Callable:
    """Return the file constructor for a registered format, raising a clear error otherwise."""
    try:
        return _FORMAT_TO_CONSTRUCTOR[name]
    except KeyError:
        raise ValueError(
            f"Unknown or unavailable format {name!r}. Supported formats: {sorted(supported_formats())}. "
            "You may need to install the corresponding optional dependency."
        )


def is_writable_format(name: str) -> bool:
    """Return whether a registered format supports writing."""
    return _FORMAT_WRITABLE.get(name, False)


# --- backend registrations (each guarded by its optional dependency) ---------------------------

try:
    import h5py

    def _open_hdf5(path: PathLike, mode: str = "r", **kwargs):
        """Open an hdf5 file."""
        return h5py.File(path, mode=mode, **kwargs)

    register_format("hdf5", [".h5", ".hdf", ".hdf5"], _open_hdf5, writable=True)
except ImportError:
    h5py = None

try:
    import z5py

    def _open_n5(path: PathLike, mode: str = "r", **kwargs):
        """Open an n5 file via z5py."""
        return z5py.File(path, mode=mode, **kwargs)

    register_format("n5", [".n5"], _open_n5, writable=True)
except ImportError:
    z5py = None

try:
    import zarr

    def _open_zarr(path: PathLike, mode: str = "r", **kwargs):
        """Open a zarr container (group or array)."""
        return zarr.open(path, mode=mode, **kwargs)

    register_format("zarr", [".zarr", ".zr"], _open_zarr, writable=True)
except ImportError:
    zarr = None

# The read-only scientific formats always register their wrappers; the heavy import is guarded
# inside the wrapper's constructor, so registration here never fails.
from .mrc_wrapper import MRCFile  # noqa: E402
from .nifti_wrapper import NiftiFile  # noqa: E402
from .msr_wrapper import MSRFile  # noqa: E402
from .knossos_wrapper import KnossosFile  # noqa: E402
from .image_stack_wrapper import ImageStackFile  # noqa: E402

register_format("mrc", [".mrc", ".rec"], MRCFile, writable=False)
register_format("nifti", [".nii", ".nii.gz"], NiftiFile, writable=False)
register_format("msr", [".msr"], MSRFile, writable=False)


def folder_based(path: PathLike, mode: str = "r", **kwargs):
    """Open an extension-less folder (or single image file), trying KNOSSOS then an image stack."""
    try:
        return KnossosFile(path, mode, **kwargs)
    except RuntimeError:
        return ImageStackFile(path, mode, **kwargs)


register_format("folder", ["", ".tif", ".tiff"], folder_based, writable=False)
