"""Tests for block-wise local maxima detection."""
import bioimage_cpp as bic
import numpy as np

import bioimage_py as bp


def _sorted(coords):
    """Sort coordinates so two maxima sets can be compared order-independently."""
    return coords[np.lexsort(coords.T[::-1])] if len(coords) else coords


def test_find_local_maxima_matches_whole_array(zarr_factory, rng):
    shape, block_shape = (50, 48), (16, 16)
    img = bic.filters.gaussian_smoothing(rng.random(shape).astype("float32"), 1.5)
    za = zarr_factory(img, chunks=block_shape)

    ref = _sorted(bp.morphology.find_local_maxima(img, min_distance=3, threshold_abs=0.5))  # direct
    assert len(ref) > 0
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        got = bp.morphology.find_local_maxima(za, min_distance=3, threshold_abs=0.5,
                                              block_shape=block_shape, num_workers=nw, job_type=job)
        np.testing.assert_array_equal(_sorted(got), ref, err_msg=f"maxima mismatch ({nw}, {job})")
