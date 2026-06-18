"""Unit tests for the indexing helpers in bioimage_py/io/_util.py."""
import numpy as np
import pytest

from bioimage_py.io._util import (
    chunks_overlapping_roi,
    int_to_start_stop,
    map_chunk_to_roi,
    normalize_index,
    slice_to_start_stop,
    squeeze_singletons,
)


def test_slice_to_start_stop_basic():
    assert slice_to_start_stop(slice(2, 5), 10) == slice(2, 5)
    assert slice_to_start_stop(slice(None, None), 10) == slice(0, 10)
    assert slice_to_start_stop(slice(None, 20), 10) == slice(0, 10)  # stop clamped


def test_slice_to_start_stop_negative():
    assert slice_to_start_stop(slice(-3, None), 10) == slice(7, 10)
    assert slice_to_start_stop(slice(None, -2), 10) == slice(0, 8)


def test_slice_to_start_stop_out_of_bounds():
    assert slice_to_start_stop(slice(20, 30), 10) == slice(None, 0)  # start past end
    assert slice_to_start_stop(slice(0, -20), 10) == slice(None, 0)  # empty range


def test_slice_to_start_stop_step_raises():
    with pytest.raises(ValueError, match="steps"):
        slice_to_start_stop(slice(0, 10, 2), 10)


def test_int_to_start_stop():
    assert int_to_start_stop(3, 10) == slice(3, 4)
    assert int_to_start_stop(-1, 10) == slice(9, 10)
    with pytest.raises(ValueError, match="out of range"):
        int_to_start_stop(10, 10)
    with pytest.raises(ValueError, match="out of range"):
        int_to_start_stop(-11, 10)


def test_normalize_index_slices():
    norm, squeeze = normalize_index((slice(2, 5), slice(1, 3)), (10, 10))
    assert norm == (slice(2, 5), slice(1, 3))
    assert squeeze == ()


def test_normalize_index_integer_records_squeeze():
    norm, squeeze = normalize_index(3, (10,))
    assert norm == (slice(3, 4),)
    assert squeeze == (0,)


def test_normalize_index_mixed():
    norm, squeeze = normalize_index((2, slice(1, 3)), (5, 5))
    assert norm == (slice(2, 3), slice(1, 3))
    assert squeeze == (0,)


def test_normalize_index_ellipsis_and_padding():
    # Ellipsis expands to fill missing axes.
    norm, _ = normalize_index((Ellipsis,), (4, 5))
    assert norm == (slice(0, 4), slice(0, 5))
    # A short index is padded with a trailing full slice.
    norm, _ = normalize_index((slice(1, 2),), (4, 5))
    assert norm == (slice(1, 2), slice(0, 5))


def test_normalize_index_too_long_raises():
    with pytest.raises(TypeError, match="too long"):
        normalize_index((slice(0, 1), slice(0, 1), slice(0, 1)), (4, 5))


def test_squeeze_singletons():
    arr = np.arange(12).reshape(1, 12)
    np.testing.assert_array_equal(squeeze_singletons(arr, (0,)), np.arange(12))

    scalar = squeeze_singletons(np.array([[7]]), (0, 1))
    assert np.ndim(scalar) == 0 and int(scalar) == 7

    arr2 = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(squeeze_singletons(arr2, ()), arr2)


def test_chunks_overlapping_roi():
    ids = list(chunks_overlapping_roi((slice(0, 20), slice(0, 20)), (16, 16)))
    assert set(ids) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    ids = list(chunks_overlapping_roi((slice(0, 16),), (16,)))
    assert ids == [(0,)]


def test_map_chunk_to_roi_partial_and_full():
    roi = (slice(8, 40),)
    chunks = (16,)
    # Start block (partial): chunk 0 covers global 0-16, roi starts at 8.
    chunk_bb, roi_bb = map_chunk_to_roi((0,), roi, chunks)
    assert chunk_bb == (slice(8, 16),) and roi_bb == (slice(0, 8),)
    # Middle block (full): chunk 1 covers 16-32.
    chunk_bb, roi_bb = map_chunk_to_roi((1,), roi, chunks)
    assert chunk_bb == (slice(0, 16),) and roi_bb == (slice(8, 24),)
    # End block (partial): chunk 2 covers 32-48, roi ends at 40.
    chunk_bb, roi_bb = map_chunk_to_roi((2,), roi, chunks)
    assert chunk_bb == (slice(0, 8),) and roi_bb == (slice(24, 32),)
