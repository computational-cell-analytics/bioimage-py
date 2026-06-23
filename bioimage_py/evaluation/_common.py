"""Shared helper for the metric wrappers: build the (segmentation, groundtruth) contingency table."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..runner.config import RunnerConfig
from ..sources import SourceLike
from .contingency_table import ContingencyTable, contingency_table


def build_table(
    segmentation: SourceLike,
    groundtruth: SourceLike,
    *,
    ignore_seg: Optional[Sequence[int]] = None,
    ignore_gt: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    mask: Optional[SourceLike] = None,
) -> ContingencyTable:
    """Build the contingency table in ``(segmentation, groundtruth)`` orientation, then drop ignores.

    Centralizes the runner plumbing shared by the metric wrappers. The ignore labels (if any) are
    applied with :meth:`ContingencyTable.drop_ignore` (the OR pixel-drop semantic).
    """
    table = contingency_table(segmentation, groundtruth, num_workers=num_workers,
                              block_shape=block_shape, job_type=job_type, job_config=job_config,
                              mask=mask)
    return table.drop_ignore(ignore_seg, ignore_gt)
