"""Efficient, parallel, and distributed implementation of image analysis and segmentation functionality for biomedical imaging.

Reimplements functionality from [elf](https://github.com/constantinpape/elf) and [cluster_tools](https://github.com/constantinpape/cluster_tools) in a more efficient and scalable manner.

**Note:** this package is in an early state and mainly provides support for data conversion, downsampling, and some initial segmentation functionality (connected components and watershed).
The functionality will be extended soon; the implementation of seamlessly switching between local and distributed execution (via slurm) is already in place. Any feedback on issues you find or on how to improve usability is welcome!

.. include:: ../docs/installation.md
.. include:: ../docs/usage.md
"""  # noqa
from . import evaluation, filters, io, morphology, segmentation, stats  # noqa: F401
from .copy import copy
from .downsample import downsample
from .runner import get_runner
from .sources import as_source, open_cloudvolume, open_source, open_webknossos
from .util import to_roi
from .__version__ import __version__

__all__ = [
    "__version__",
    "stats",
    "filters",
    "segmentation",
    "morphology",
    "evaluation",
    "io",
    "copy",
    "downsample",
    "get_runner",
    "as_source",
    "open_source",
    "open_cloudvolume",
    "open_webknossos",
    "to_roi",
]
