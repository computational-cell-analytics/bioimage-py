"""Segmentation comparison metrics built on the block-wise contingency table primitive.

Each metric comes in two layers: a high-level wrapper that takes two segmentations and builds the
contingency table in parallel, and a low-level ``*_scores`` function that takes a pre-built
`ContingencyTable` (for reuse across metrics, or for tables built distributedly / with resume).
"""
from .contingency_table import ContingencyTable, contingency_table
from .variation_of_information import (object_vi, object_vi_scores, variation_of_information,
                                       vi_scores)
from .rand_index import rand_index, rand_scores
from .cremi_score import cremi_score, cremi_scores
from .matching import (matching, matching_scores, mean_segmentation_accuracy,
                       mean_segmentation_accuracy_scores)
from .centroid_matching import centroid_matching, coordinate_matching
from .dice import best_dice_scores, dice_score, symmetric_best_dice_score

__all__ = [
    "ContingencyTable", "contingency_table",
    "variation_of_information", "vi_scores", "object_vi", "object_vi_scores",
    "rand_index", "rand_scores",
    "cremi_score", "cremi_scores",
    "matching", "matching_scores", "mean_segmentation_accuracy", "mean_segmentation_accuracy_scores",
    "centroid_matching", "coordinate_matching",
    "dice_score", "symmetric_best_dice_score", "best_dice_scores",
]
