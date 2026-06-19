"""Tests for run()-level concerns: roi restriction and output shape validation."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.runner import get_runner
from bioimage_py.util import to_roi


def _max_block(block, inputs, outputs, mask):
    """A tiny per-block reduction (module-level so it is cloudpickle-safe)."""
    return float(np.max(inputs[0][to_roi(block)]))


def test_run_roi_restricts_processing(zarr_factory, rng):
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    roi = (slice(8, 24), slice(8, 24))  # a 16x16 sub-region -> 2x2 = 4 blocks of (8, 8)
    runner = get_runner("local")
    results = runner.run(_max_block, [z], block_shape=(8, 8), num_workers=2,
                         has_return_val=True, roi=roi)
    assert len(results) == 4  # only the blocks inside the roi are processed
    assert np.isclose(max(results), a[8:24, 8:24].max())


def test_output_shape_mismatch_raises(zarr_factory, rng):
    z = zarr_factory(rng.random((16, 16)).astype("float32"), chunks=(8, 8))
    bad_out = zarr_factory(shape=(16, 8), chunks=(8, 8), dtype="float32", fill=0.0)
    with pytest.raises(ValueError, match="incompatible with the domain"):
        bp.copy(z, bad_out, block_shape=(8, 8), num_workers=2)
