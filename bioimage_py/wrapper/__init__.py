"""On-the-fly transformation sources (wrappers)."""
from .base import WrapperSource, register_wrapper, wrapper_from_spec
from .generic import ThresholdSource
from .resize import ResizedSource

__all__ = ["WrapperSource", "ThresholdSource", "ResizedSource", "register_wrapper", "wrapper_from_spec"]
