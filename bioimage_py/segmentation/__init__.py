"""Segmentation: connected-component labeling and related operations."""
from .label import label
from .watershed import watershed

__all__ = ["label", "watershed"]
