"""Tests for tile-wise segmentation stitching (multicut over tile overlaps).

Mirrors elf's ``test/segmentation/test_stitching.py`` (binary-blobs data, compared with the adapted
Rand error after dropping tiny segments that may stitch ambiguously), and adds the local/subprocess
parity check that is the core correctness guarantee of bioimage_py.
"""
import sys

import cloudpickle
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp

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


# --- stitch_segmentation ---

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
    # Stitching can only merge ids, so it never increases the object count.
    assert int(stitched.max()) <= len(np.unique(pre[pre != 0]))


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


# --- stitch_tiled_segmentation ---

@pytest.mark.parametrize("tile_shape", [(224, 224), (256, 256), (512, 512)])
def test_stitch_tiled_segmentation(tile_shape):
    data = _get_data(size=512)
    tiled, reference = _make_tiled(data, tile_shape)
    stitched = bp.segmentation.stitch_tiled_segmentation(tiled, tile_shape)
    _check_result(stitched, reference)


# --- local / subprocess parity (the headline correctness guarantee) ---

@pytest.mark.parametrize("job_type,num_workers", [("local", 4), ("subprocess", 3)])
def test_stitch_segmentation_parity(job_type, num_workers, zarr_factory):
    data = _get_data()
    local = bp.segmentation.stitch_segmentation(data, _segment, (128, 128), (32, 32))

    zin = zarr_factory(data.astype("uint8"), chunks=(128, 128))
    zout = zarr_factory(shape=data.shape, chunks=(128, 128), dtype="uint64", fill=0)
    bp.segmentation.stitch_segmentation(
        zin, _segment, (128, 128), (32, 32), output=zout, job_type=job_type, num_workers=num_workers,
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


def test_stitch_segmentation_output_required_for_distributed():
    data = _get_data()
    with pytest.raises(ValueError, match="output.*required"):
        bp.segmentation.stitch_segmentation(data, _segment, (128, 128), (32, 32),
                                            job_type="subprocess")


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
