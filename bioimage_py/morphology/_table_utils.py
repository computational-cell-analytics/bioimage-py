"""Shared helpers for file-backed morphology operations."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from typing import Any, Dict, Mapping

import cloudpickle
import numpy as np

from ..sources import Source, SourceSpec


_SOURCE_CACHE: Dict[str, Source] = {}


def json_value(value: Any) -> Any:
    """Convert a source description to canonical JSON-compatible values."""
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {key: item for key, item in sorted(
            (str(key), json_value(item)) for key, item in value.items()
        )}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, slice):
        return {"slice": [json_value(value.start), json_value(value.stop),
                          json_value(value.step)]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    serialized = cloudpickle.dumps(value)
    value_type = type(value)
    return {
        "python_type": f"{value_type.__module__}.{value_type.__qualname__}",
        "cloudpickle_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def source_identity(spec: SourceSpec, source: Source) -> Dict[str, Any]:
    """Return stable source metadata for a table dataset definition."""
    return {
        "spec": json_value(spec),
        "shape": [int(size) for size in source.shape],
        "dtype": np.dtype(source.dtype).str,
    }


def resolve_source(value: Any) -> Source:
    """Open and cache a source specification in one worker process."""
    if isinstance(value, Source):
        return value
    key = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"))
    source = _SOURCE_CACHE.get(key)
    if source is None:
        from ..sources.dispatch import from_spec
        source = from_spec(value)
        _SOURCE_CACHE[key] = source
    return source


def overlapping_paths(first: str, second: str) -> bool:
    """Return whether either normalized path contains the other."""
    first = os.path.realpath(first)
    second = os.path.realpath(second)
    try:
        common = os.path.commonpath([first, second])
    except ValueError:
        return False
    return common == first or common == second
