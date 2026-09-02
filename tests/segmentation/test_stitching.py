"""Tests for tile-wise segmentation stitching (multicut over an instance-correspondence graph).

Mirrors elf's ``test/segmentation/test_stitching.py`` (binary-blobs data, compared with the adapted
Rand error after dropping tiny segments that may stitch ambiguously), adds the local/subprocess
parity check that is the core correctness guarantee of bioimage_py, and a suite of synthetic tile
scenarios with prescribed per-tile labels (``_LookupSegmenter``) that isolate the stitching behaviour:
halo correspondences whose cores do not touch, one-to-many splits, tiny overlaps, multi-axis crossings,
the two-phase block-store API, ...
"""
import os
import re
import sys

import cloudpickle
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp
from bioimage_py.runner import RunnerConfig, RunnerError
from bioimage_py.util import get_blocking

# The blob generator lives in scikit-image; skip the whole module if it is not installed.
binary_blobs = pytest.importorskip("skimage.data").binary_blobs

# The subprocess backend cloudpickles the segmentation function. Functions defined in a normal user
# script (``__main__``) or an installed package are shipped by value / re-imported automatically;
# functions defined in a pytest-imported test module are not re-importable on the worker, so ship
# this module's functions by value.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


# --- module-level segmentation functions (picklable, so the subprocess backend can ship them) ---

def _segment(tile, tile_id=None):
    """Connected-component label the foreground of a tile."""
    return bic.segmentation.label(tile > 0, connectivity=1).astype("uint32")


def _segment_no_background(tile, tile_id=None):
    """Label a tile and shift by 1 so there is no background (0) label."""
    return bic.segmentation.label(tile > 0, connectivity=1).astype("uint32") + 1


def _failing_segment(tile, tile_id=None):
    """Label a tile, but fail for tile 1 (to test the block store preservation)."""
    if tile_id == 1:
        raise RuntimeError("tile 1 failed on purpose")
    return _segment(tile, tile_id)


class _LookupSegmenter:
    """Picklable segmentation function returning prescribed tile-local labels per tile.

    ``labels_by_block`` maps a block id to a full-volume label array (tile-local ids); the haloed tile's
    region is cropped out of it, so border-clipped tiles get the right shape. Blocks without an entry
    return zeros.
    """

    def __init__(self, shape, tile_shape, tile_overlap, labels_by_block):
        self.shape = tuple(shape)
        self.tile_shape = tuple(tile_shape)
        self.tile_overlap = tuple(tile_overlap)
        self.labels_by_block = {int(k): np.asarray(v) for k, v in labels_by_block.items()}

    def __call__(self, tile, block_id):
        blocking = get_blocking(self.shape, self.tile_shape)
        outer = blocking.get_block_with_halo(int(block_id), list(self.tile_overlap)).outer_block
        roi = tuple(slice(int(b), int(e)) for b, e in zip(outer.begin, outer.end))
        labels = self.labels_by_block.get(int(block_id))
        if labels is None:
            return np.zeros(tuple(int(s) for s in outer.shape), dtype="uint32")
        return labels[roi]


# --- data helpers (after elf's TestStitching) ---

def _get_data(size=256, ndim=2, seed=0):
    """A binary blob image, labeled into instances (the blobs are otherwise semantic)."""
    data = binary_blobs(size, blob_size_fraction=0.1, volume_fraction=0.25, n_dim=ndim, rng=seed)
    return bic.segmentation.label(data, connectivity=1).astype("uint64") > 0


def _make_tiled(data, tile_shape):
    """Build a tiled labeling with ids unique across tiles, plus the un-tiled reference."""
    reference = bic.segmentation.label(data, connectivity=1).astype("uint64")
    blocking = bic.utils.Blocking([0] * data.ndim, list(data.shape), list(tile_shape))
    tiled = np.zeros(data.shape, dtype="uint64")
    offset = 0
    for block_id in range(blocking.number_of_blocks):
        block = blocking.get_block(block_id)
        roi = tuple(slice(int(b), int(e)) for b, e in zip(block.begin, block.end))
        tile = bic.segmentation.label(data[roi], connectivity=1).astype("uint64")
        mask = tile != 0
        if mask.sum() > 0:
            tile[mask] += offset
            offset = int(tile.max())
        tiled[roi] = tile
    return tiled, reference


def _check_result(segmentation, expected, rtol=1e-2, atol=1e-2):
    """Assert the stitched result matches the reference up to the adapted Rand error (small-segment tolerant)."""
    segmentation = np.asarray(segmentation).copy()
    expected = np.asarray(expected).copy()
    assert segmentation.shape == expected.shape
    # Drop small segments before evaluation: they can stitch ambiguously.
    ids, sizes = np.unique(segmentation, return_counts=True)
    drop = np.isin(segmentation, ids[sizes < 250])
    segmentation[drop] = 0
    expected[drop] = 0
    are, _ = bp.evaluation.rand_index(segmentation, expected)
    assert np.isclose(are, 0.0, rtol=rtol, atol=atol), f"adapted Rand error too high: {are}"


# --- synthetic scenario helpers ---

# Two tiles along axis 0: tile 0 = rows [0, 32) (haloed: [0, 40)), tile 1 = rows [32, 64) (haloed: [24, 64)).
# The shared halo face is rows [24, 40).
_SHAPE2, _TILE2, _HALO2 = (64, 64), (32, 64), (8, 8)


def _paint(shape, boxes, dtype="uint32"):
    """A full-volume label array with ``label`` painted into each ``(label, roi)`` box."""
    labels = np.zeros(shape, dtype=dtype)
    for label, roi in boxes:
        labels[roi] = label
    return labels


def _stitch(segmenter, **kwargs):
    """Run `stitch_segmentation` with a `_LookupSegmenter` (the input data is irrelevant)."""
    data = np.zeros(segmenter.shape, dtype="uint8")
    return bp.segmentation.stitch_segmentation(data, segmenter, segmenter.tile_shape, segmenter.tile_overlap,
                                               **kwargs)


def _n_objects(segmentation):
    segmentation = np.asarray(segmentation)
    return len(np.unique(segmentation[segmentation != 0]))


def _fill_store(store, segmenter, shape, tile_shape, tile_overlap):
    """Fill a block store by calling ``segmenter`` on every haloed tile (of a zero input)."""
    data = np.zeros(shape, dtype="uint8")
    blocking = get_blocking(shape, tile_shape)
    for block_id in range(blocking.number_of_blocks):
        outer = blocking.get_block_with_halo(block_id, list(tile_overlap)).outer_block
        roi = tuple(slice(int(b), int(e)) for b, e in zip(outer.begin, outer.end))
        labels = np.asarray(segmenter(data[roi], block_id))
        store[(block_id,) + tuple(slice(0, s) for s in labels.shape)] = labels


# --- stitch_segmentation on blob data ---

@pytest.mark.parametrize("tile_shape", [(128, 128), (256, 256), (128, 256), (224, 224)])
def test_stitch_segmentation(tile_shape):
    for seed in range(3):
        data = _get_data(seed=seed)
        expected = _segment(data).astype("uint64")
        stitched = bp.segmentation.stitch_segmentation(data, _segment, tile_shape, (32, 32))
        _check_result(stitched, expected)


@pytest.mark.parametrize("tile_shape", [(32, 32, 32), (64, 64, 64), (32, 64, 24)])
def test_stitch_segmentation_3d(tile_shape):
    data = _get_data(size=128, ndim=3)
    expected = _segment(data).astype("uint64")
    stitched = bp.segmentation.stitch_segmentation(data, _segment, tile_shape, (8, 8, 8))
    _check_result(stitched, expected, rtol=0.1, atol=0.1)


def test_stitch_segmentation_return_before():
    data = _get_data()
    stitched, pre = bp.segmentation.stitch_segmentation(
        data, _segment, (128, 128), (16, 16), return_before_stitching=True,
    )
    assert stitched.shape == data.shape
    assert pre.shape == data.shape
    # Stitching can only merge ids, so it never increases the object count; its ids are consecutive.
    assert int(stitched.max()) == _n_objects(stitched) <= _n_objects(pre)


def test_stitch_segmentation_no_background():
    data = _get_data()
    stitched = bp.segmentation.stitch_segmentation(
        data, _segment_no_background, (128, 128), (16, 16), with_background=False,
    )
    assert stitched.shape == data.shape
    assert not (stitched == 0).any()


def test_stitch_segmentation_channels():
    data = _get_data()
    data_xyc = np.stack([data, data, data], axis=-1).astype("uint8")

    def _segment_c0(tile, tile_id=None):
        return bic.segmentation.label(tile[..., 0] > 0, connectivity=1).astype("uint32")

    expected = _segment(data).astype("uint64")
    stitched = bp.segmentation.stitch_segmentation(
        data_xyc, _segment_c0, (128, 128), (32, 32), shape=data.shape,
    )
    assert stitched.shape == data.shape
    _check_result(stitched, expected)


def test_stitch_segmentation_single_tile_equals_unblocked():
    data = _get_data()
    stitched = bp.segmentation.stitch_segmentation(data, _segment, data.shape, (8, 8))
    expected, _, _ = bic.segmentation.relabel_sequential(_segment(data).astype("uint64"), offset=1)
    assert np.array_equal(stitched, expected)


# --- synthetic scenarios: the stitching decisions in isolation ---

def test_halo_correspondence_without_core_contact():
    # Tile 1's prediction of the object is eroded at the top so that the two written cores do not touch
    # at the seam (rows 32/33 are empty), while both predictions overlap strongly in the shared halo.
    tile0 = _paint(_SHAPE2, [(1, (slice(20, 40), slice(20, 40)))])
    tile1 = _paint(_SHAPE2, [(1, (slice(34, 50), slice(20, 40)))])
    stitched = _stitch(_LookupSegmenter(_SHAPE2, _TILE2, _HALO2, {0: tile0, 1: tile1}))
    assert _n_objects(stitched) == 1
    assert stitched[25, 25] == stitched[40, 25] != 0
    assert (stitched[32:34, 20:40] == 0).all()  # cores are copied as they are (no seam composition)


def test_one_to_many_is_side_independent():
    whole = _paint(_SHAPE2, [(1, (slice(20, 44), slice(20, 40)))])
    split = _paint(_SHAPE2, [(1, (slice(20, 44), slice(20, 30))), (2, (slice(20, 44), slice(30, 40)))])
    cases = [
        (_LookupSegmenter(_SHAPE2, _TILE2, _HALO2, {0: whole, 1: split}), (25, 25), [(40, 25), (40, 35)]),
        (_LookupSegmenter(_SHAPE2, _TILE2, _HALO2, {0: split, 1: whole}), (40, 25), [(25, 25), (25, 35)]),
    ]
    for segmenter, whole_point, half_points in cases:
        # Default competition: one neighbouring tile cannot override the tile-local split, the whole
        # object joins exactly one of the halves (the other one stays a separate object).
        stitched = _stitch(segmenter)
        assert _n_objects(stitched) == 2
        assert sum(stitched[whole_point] == stitched[p] for p in half_points) == 1
        # Without competition the correspondences merge transitively.
        merged = _stitch(segmenter, competition_disaffinity=None)
        assert _n_objects(merged) == 1


def test_two_agreeing_neighbours_override_split():
    # Three tiles along z; the middle tile splits the object that both its neighbours predict whole.
    shape, tile_shape, halo = (96, 32, 32), (32, 32, 32), (8, 8, 8)
    whole = _paint(shape, [(1, (slice(24, 72), slice(8, 24), slice(8, 24)))])
    split = _paint(shape, [(1, (slice(24, 72), slice(8, 24), slice(8, 16))),
                           (2, (slice(24, 72), slice(8, 24), slice(16, 24)))])
    stitched = _stitch(_LookupSegmenter(shape, tile_shape, halo, {0: whole, 1: split, 2: whole}))
    assert _n_objects(stitched) == 1
    # With a single agreeing neighbour the split is kept (the whole object joins one half).
    stitched = _stitch(_LookupSegmenter((64, 32, 32), tile_shape, halo, {0: whole[:64], 1: split[:64]}))
    assert _n_objects(stitched) == 2


def test_tiny_fragment_needs_support():
    # A one-voxel fragment (id 2) in tile 0's core is fully contained in tile 1's prediction of the
    # object: the containment-based "max" metric merges it, unless it lacks absolute support.
    tile0 = _paint(_SHAPE2, [(1, (slice(20, 40), slice(20, 40))), (2, (slice(31, 32), slice(45, 46)))])
    tile1 = _paint(_SHAPE2, [(1, (slice(24, 50), slice(20, 50)))])
    segmenter = _LookupSegmenter(_SHAPE2, _TILE2, _HALO2, {0: tile0, 1: tile1})
    assert _n_objects(_stitch(segmenter)) == 2  # geometric mean: the fragment stays separate
    assert _n_objects(_stitch(segmenter, overlap_metric="max")) == 1
    assert _n_objects(_stitch(segmenter, overlap_metric="max", min_overlap=2)) == 2


def test_accidental_seam_overlap_stays_cut():
    # Two different objects that overlap by a single row at the seam.
    tile0 = _paint(_SHAPE2, [(1, (slice(10, 33), slice(20, 40)))])
    tile1 = _paint(_SHAPE2, [(1, (slice(32, 54), slice(20, 40)))])
    stitched = _stitch(_LookupSegmenter(_SHAPE2, _TILE2, _HALO2, {0: tile0, 1: tile1}))
    assert _n_objects(stitched) == 2


def test_touching_objects_on_seam_stay_separate():
    truth = _paint(_SHAPE2, [(1, (slice(10, 32), slice(20, 40))), (2, (slice(32, 54), slice(20, 40)))])
    stitched = _stitch(_LookupSegmenter(_SHAPE2, _TILE2, _HALO2, {0: truth, 1: truth}))
    assert _n_objects(stitched) == 2
    assert stitched[20, 30] != stitched[45, 30]


@pytest.mark.parametrize("box", [
    (slice(20, 44), slice(4, 20), slice(4, 20)),    # crosses the z seam only
    (slice(4, 20), slice(20, 44), slice(4, 20)),    # crosses the y seam only
    (slice(4, 20), slice(4, 20), slice(20, 44)),    # crosses the x seam only
    (slice(20, 44), slice(20, 44), slice(20, 44)),  # crosses all seams, spans all 8 tiles
])
def test_object_crossing_seams_3d(box):
    shape, tile_shape, halo = (64, 64, 64), (32, 32, 32), (8, 8, 8)
    truth = _paint(shape, [(1, box)])
    stitched = _stitch(_LookupSegmenter(shape, tile_shape, halo, {b: truth for b in range(8)}))
    assert _n_objects(stitched) == 1
    assert np.array_equal(stitched != 0, truth != 0)


def test_object_crossing_corner_2d():
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    truth = _paint(shape, [(1, (slice(24, 40), slice(24, 40)))])
    stitched = _stitch(_LookupSegmenter(shape, tile_shape, halo, {b: truth for b in range(4)}))
    assert _n_objects(stitched) == 1
    assert np.array_equal(stitched != 0, truth != 0)


def test_non_divisible_tiles_with_halo():
    shape, tile_shape, halo = (70, 50), (32, 24), (8, 8)
    truth = _paint(shape, [(1, (slice(20, 50), slice(10, 40)))])
    n_blocks = get_blocking(shape, tile_shape).number_of_blocks
    stitched = _stitch(_LookupSegmenter(shape, tile_shape, halo, {b: truth for b in range(n_blocks)}))
    assert _n_objects(stitched) == 1
    assert np.array_equal(stitched != 0, truth != 0)


def test_zero_overlap_axis_warns_and_does_not_merge():
    shape, tile_shape, halo = (64, 64), (32, 32), (0, 8)
    # Object 1 crosses the axis-0 seam only (no overlap evidence there), object 2 the axis-1 seam only.
    truth = _paint(shape, [(1, (slice(24, 40), slice(4, 20))), (2, (slice(4, 20), slice(24, 40)))])
    segmenter = _LookupSegmenter(shape, tile_shape, halo, {b: truth for b in range(4)})
    with pytest.warns(UserWarning, match="tile_overlap"):
        stitched = _stitch(segmenter)
    assert _n_objects(stitched) == 3
    assert stitched[10, 30] == stitched[10, 35]  # object 2 merged across the axis-1 seam
    assert stitched[28, 10] != stitched[36, 10]  # object 1 split at the axis-0 seam


def test_no_background_label_zero_is_an_object():
    # Without background the tile-local id 0 of tile 0 is a genuine object and must be stitched.
    tile0 = _paint(_SHAPE2, [(0, (slice(0, 28), slice(None))), (1, (slice(28, 64), slice(None)))])
    tile1 = _paint(_SHAPE2, [(5, (slice(0, 28), slice(None))), (7, (slice(28, 64), slice(None)))])
    stitched = _stitch(_LookupSegmenter(_SHAPE2, _TILE2, _HALO2, {0: tile0, 1: tile1}), with_background=False)
    assert not (stitched == 0).any()
    assert _n_objects(stitched) == 2
    assert stitched[10, 0] == stitched[26, 0]
    assert stitched[30, 0] == stitched[50, 0]
    assert stitched[10, 0] != stitched[50, 0]


# --- validation errors ---

def test_segmentation_function_result_is_validated():
    data = np.zeros(_SHAPE2, dtype="uint8")

    def _wrong_shape(tile, tile_id=None):
        return np.zeros(tuple(s + 1 for s in tile.shape), dtype="uint32")

    def _negative(tile, tile_id=None):
        return -np.ones(tile.shape, dtype="int32")

    def _too_large(tile, tile_id=None):
        labels = np.zeros(tile.shape, dtype="uint64")
        labels[0, 0] = np.iinfo("uint32").max
        return labels

    for func, match in [(_wrong_shape, "haloed tile shape"), (_negative, "non-negative"),
                        (_too_large, "smaller than")]:
        with pytest.raises(RunnerError, match=match):
            bp.segmentation.stitch_segmentation(data, func, _TILE2, _HALO2)


def test_stitching_arguments_are_validated():
    data = np.zeros(_SHAPE2, dtype="uint8")
    with pytest.raises(ValueError, match="beta"):
        bp.segmentation.stitch_segmentation(data, _segment, _TILE2, _HALO2, beta=1.0)
    with pytest.raises(ValueError, match="overlap_metric"):
        bp.segmentation.stitch_segmentation(data, _segment, _TILE2, _HALO2, overlap_metric="jaccard")
    with pytest.raises(ValueError, match="min_overlap"):
        bp.segmentation.stitch_segmentation(data, _segment, _TILE2, _HALO2, min_overlap=0)
    with pytest.raises(ValueError, match="competition_disaffinity"):
        bp.segmentation.stitch_segmentation(data, _segment, _TILE2, _HALO2, competition_disaffinity=1.5)
    with pytest.raises(ValueError, match="one entry per spatial axis"):
        bp.segmentation.stitch_segmentation(data, _segment, (32,), _HALO2)
    with pytest.raises(ValueError, match="dtype uint64"):
        bp.segmentation.stitch_segmentation(data, _segment, _TILE2, _HALO2, output=np.zeros(_SHAPE2, "uint32"))


def test_stitch_segmentation_output_required_for_distributed():
    data = _get_data()
    with pytest.raises(ValueError, match="output.*required"):
        bp.segmentation.stitch_segmentation(data, _segment, (128, 128), (32, 32),
                                            job_type="subprocess")


# --- stitch_block_segmentations (the two-phase API) ---

def test_block_store_shape():
    assert bp.segmentation.block_store_shape((256, 256), (128, 128), (32, 32)) == (4, 192, 192)
    assert bp.segmentation.block_store_shape((70, 50), (32, 24), (8, 8)) == (9, 48, 40)


def test_stitch_block_segmentations_matches_stitch_segmentation(zarr_factory):
    shape, tile_shape, tile_overlap = (64, 64, 64), (32, 32, 32), (8, 8, 8)
    truth = _paint(shape, [(1, (slice(20, 44), slice(20, 44), slice(20, 44))),
                           (2, (slice(4, 16), slice(24, 40), slice(4, 20)))])
    # Tile 3 splits object 1 along x; the others predict the truth.
    split = truth.copy()
    split[truth == 1] = 0
    split[20:44, 20:44, 20:32][truth[20:44, 20:44, 20:32] == 1] = 1
    split[20:44, 20:44, 32:44][truth[20:44, 20:44, 32:44] == 1] = 3
    segmenter = _LookupSegmenter(shape, tile_shape, tile_overlap, {b: (split if b == 3 else truth) for b in range(8)})
    expected = _stitch(segmenter)

    store_shape = bp.segmentation.block_store_shape(shape, tile_shape, tile_overlap)
    store = np.zeros(store_shape, dtype="uint32")
    _fill_store(store, segmenter, shape, tile_shape, tile_overlap)
    stitched = bp.segmentation.stitch_block_segmentations(store, shape, tile_shape, tile_overlap)
    assert np.array_equal(stitched, expected)

    zstore = zarr_factory(store, chunks=(1,) + store_shape[1:])
    zout = zarr_factory(shape=shape, chunks=tile_shape, dtype="uint64", fill=0)
    bp.segmentation.stitch_block_segmentations(zstore, shape, tile_shape, tile_overlap, output=zout,
                                               job_type="subprocess", num_workers=3)
    assert np.array_equal(zout[:], expected)


def test_stitch_block_segmentations_validation():
    shape, tile_shape, tile_overlap = _SHAPE2, _TILE2, _HALO2
    store_shape = bp.segmentation.block_store_shape(shape, tile_shape, tile_overlap)
    with pytest.raises(ValueError, match="block_store has shape"):
        bp.segmentation.stitch_block_segmentations(np.zeros((3,) + store_shape[1:], "uint32"), shape,
                                                   tile_shape, tile_overlap)
    with pytest.raises(ValueError, match="integer labels"):
        bp.segmentation.stitch_block_segmentations(np.zeros(store_shape, "float32"), shape, tile_shape,
                                                   tile_overlap)
    store = np.zeros(store_shape, dtype="uint64")
    store[0, 0, 0] = np.prod(store_shape[1:])  # a tile-local id that collides with the next tile's range
    with pytest.raises(RunnerError, match="smaller than"):
        bp.segmentation.stitch_block_segmentations(store, shape, tile_shape, tile_overlap)


def test_block_store_preserved_on_failure(zarr_factory, tmp_path):
    data = _get_data()
    zin = zarr_factory(data.astype("uint8"), chunks=(128, 128))
    zout = zarr_factory(shape=data.shape, chunks=(128, 128), dtype="uint64", fill=0)
    with pytest.raises(RunnerError) as excinfo:
        bp.segmentation.stitch_segmentation(zin, _failing_segment, (128, 128), (32, 32), output=zout,
                                            job_type="subprocess", job_config=RunnerConfig(tmp_root=str(tmp_path)))
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("preserved at" in note for note in notes)
    path = re.search(r"preserved at (\S+);", notes[0]).group(1)
    assert os.path.exists(path)
    assert path.startswith(str(tmp_path))


# --- stitch_tiled_segmentation ---

@pytest.mark.parametrize("tile_shape", [(224, 224), (256, 256), (512, 512)])
def test_stitch_tiled_segmentation(tile_shape):
    data = _get_data(size=512)
    tiled, reference = _make_tiled(data, tile_shape)
    stitched = bp.segmentation.stitch_tiled_segmentation(tiled, tile_shape)
    _check_result(stitched, reference)


def test_stitch_tiled_segmentation_no_background():
    tiled = _paint(_SHAPE2, [(0, (slice(0, 20), slice(None))), (1, (slice(20, 32), slice(None))),
                             (2, (slice(32, 64), slice(None)))], dtype="uint64")
    stitched = bp.segmentation.stitch_tiled_segmentation(tiled, _TILE2, with_background=False)
    assert not (stitched == 0).any()
    assert _n_objects(stitched) == 2
    assert stitched[25, 0] == stitched[40, 0] != stitched[10, 0]
    # The input is not mutated.
    assert tiled[0, 0] == 0


# --- local / subprocess parity (the headline correctness guarantee) ---

@pytest.mark.parametrize("job_type,num_workers,tasks_per_worker", [
    ("local", 1, 1), ("local", 4, 1), ("subprocess", 3, 1), ("subprocess", 2, 3),
])
def test_stitch_segmentation_parity(job_type, num_workers, tasks_per_worker, zarr_factory):
    data = _get_data()
    local = bp.segmentation.stitch_segmentation(data, _segment, (128, 128), (32, 32))

    zin = zarr_factory(data.astype("uint8"), chunks=(128, 128))
    zout = zarr_factory(shape=data.shape, chunks=(128, 128), dtype="uint64", fill=0)
    bp.segmentation.stitch_segmentation(
        zin, _segment, (128, 128), (32, 32), output=zout, job_type=job_type, num_workers=num_workers,
        job_config=RunnerConfig(tasks_per_worker=tasks_per_worker),
    )
    assert np.array_equal(np.asarray(local), zout[:])


@pytest.mark.parametrize("job_type,num_workers", [("local", 4), ("subprocess", 3)])
def test_stitch_tiled_segmentation_parity(job_type, num_workers, zarr_factory):
    data = _get_data(size=512)
    tile_shape = (256, 256)
    tiled, _ = _make_tiled(data, tile_shape)
    local = bp.segmentation.stitch_tiled_segmentation(tiled, tile_shape)

    zin = zarr_factory(tiled, chunks=tile_shape)
    zout = zarr_factory(shape=data.shape, chunks=tile_shape, dtype="uint64", fill=0)
    bp.segmentation.stitch_tiled_segmentation(
        zin, tile_shape, output=zout, job_type=job_type, num_workers=num_workers,
    )
    assert np.array_equal(np.asarray(local), zout[:])


# --- multicut solver unit test ---

def test_multicut_decomposition_merges_attractive_edge():
    # Four superpixels in a 2x2 grid; force the (1, 2) edge to be strongly attractive.
    seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype="uint32")
    rag = bic.graph.region_adjacency_graph(seg)
    disaffinities = np.full(rag.number_of_edges, 0.9, dtype="float32")
    disaffinities[rag.find_edges(np.array([[1, 2]], dtype="uint64"))] = 0.0
    costs = bp.segmentation.compute_edge_costs(disaffinities, beta=0.5)
    node_labels = np.asarray(bp.segmentation.multicut_decomposition(rag, costs))
    assert node_labels[1] == node_labels[2]
    assert node_labels[1] != node_labels[4]
