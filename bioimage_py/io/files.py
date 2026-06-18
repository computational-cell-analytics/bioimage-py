"""``open_file``: infer a file format from a path and open it as an array-like handle."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from .registry import constructor_for_format, format_for_extension, supported_extensions

PathLike = Union[os.PathLike, str]


def infer_extension(path: PathLike) -> str:
    """Infer the (lower-case) file extension used for format lookup.

    Handles the two-suffix ``.nii.gz`` case and treats an extension-less path as a folder (``""``).

    Args:
        path: The path to inspect.

    Returns:
        The inferred extension, e.g. ``".h5"``, ``".nii.gz"`` or ``""``.
    """
    path_ = Path(str(path).rstrip("/"))
    suffixes = path_.suffixes
    if len(suffixes) >= 2 and "".join(suffixes[-2:]).lower() == ".nii.gz":
        return ".nii.gz"
    if len(suffixes) == 0:
        return ""
    return suffixes[-1].lower()


def infer_format(path: PathLike, ext: Optional[str] = None) -> str:
    """Infer the format name for a path (optionally with an explicit extension).

    Args:
        path: The path to open.
        ext: An explicit extension to force, overriding inference from ``path``.

    Returns:
        The registered format name.

    Raises:
        ValueError: If no installed backend handles the (inferred) extension.
    """
    ext = infer_extension(path) if ext is None else ext.lower()
    fmt = format_for_extension(ext)
    if fmt is None:
        raise ValueError(
            f"Could not infer a file format from extension {ext!r}; "
            f"it is not among the supported extensions: {sorted(supported_extensions())}. "
            "You may need to install an additional dependency (h5py, z5py, zarr, mrcfile, nibabel, ...)."
        )
    return fmt


def open_file(
    path: PathLike,
    mode: str = "r",
    ext: Optional[str] = None,
    format: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Open a file as an array-like handle, dispatching on the (inferred) format.

    Args:
        path: Path to the file or folder to open.
        mode: Mode in which to open the file. ``"r"`` (read) is supported by all formats; some
            formats also support write modes (``"a"``, ``"w"``).
        ext: Force a specific extension when it cannot be inferred from ``path``.
        format: Force a specific (registered) format name, overriding extension inference.
        kwargs: Extra keyword arguments forwarded to the backend constructor.

    Returns:
        A file handle (a mapping of datasets, or an array-like dataset).
    """
    fmt = format if format is not None else infer_format(path, ext)
    constructor = constructor_for_format(fmt)
    return constructor(path, mode=mode, **kwargs)
