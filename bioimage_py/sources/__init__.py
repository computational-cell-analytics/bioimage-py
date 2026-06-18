"""Data sources: serializable, array-like handles for the runner."""
from .array_source import ArraySource
from .base import Source, SourceSpec
from .cloudvolume_source import CloudVolumeSource, open_cloudvolume
from .dispatch import SourceLike, as_source, from_spec, register_source
from .file_source import FileSource, open_source
from .webknossos_source import WebKnossosSource, open_webknossos

__all__ = [
    "ArraySource",
    "Source",
    "SourceSpec",
    "SourceLike",
    "as_source",
    "from_spec",
    "register_source",
    "FileSource",
    "open_source",
    "CloudVolumeSource",
    "open_cloudvolume",
    "WebKnossosSource",
    "open_webknossos",
]
