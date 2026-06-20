"""On-the-fly transformation sources (wrappers)."""
from .affine import AffineSource
from .base import (
    MultiTransformationSource,
    SimpleTransformationSource,
    SimpleTransformationWithHaloSource,
    TransformationSource,
    WrapperSource,
    register_wrapper,
    wrapper_from_spec,
)
from .generic import NormalizeSource, PadSource, RoiSource, ThresholdSource
from .resize import ResizedSource

__all__ = [
    "WrapperSource",
    "SimpleTransformationSource",
    "SimpleTransformationWithHaloSource",
    "TransformationSource",
    "MultiTransformationSource",
    "ThresholdSource",
    "NormalizeSource",
    "RoiSource",
    "PadSource",
    "AffineSource",
    "ResizedSource",
    "register_wrapper",
    "wrapper_from_spec",
]
