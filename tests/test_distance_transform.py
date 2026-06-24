"""Tests for block-wise distance transform and point-to-object mapping."""
import bioimage_cpp as bic
import numpy as np

import bioimage_py as bp


def _binary(rng, shape):
    a = np.zeros(shape, dtype="uint8")
    a[rng.random(shape) > 0.85] = 1  # sparse foreground blobs
    return a


def test_distance_transform_single_block_matches_reference(zarr_factory, rng):
    shape = (40, 44)
    a = _binary(rng, shape)
    ref = bic.distance.distance_transform(a, return_distances=True, number_of_threads=1)

    np.testing.assert_allclose(bp.morphology.distance_transform(a, halo=(0, 0)), ref)  # direct

    za = zarr_factory(a, chunks=shape)
    zd = zarr_factory(shape=shape, chunks=shape, dtype="float32", fill=0)
    bp.morphology.distance_transform(za, halo=(0, 0), distances=zd, block_shape=shape, num_workers=1)
    np.testing.assert_allclose(zd[:], ref)  # one block reproduces the whole-array reference


def test_distance_transform_indices_and_both(rng):
    shape = (32, 30)
    a = _binary(rng, shape)
    ref_dist, ref_idx = bic.distance.distance_transform(
        a, return_distances=True, return_indices=True, number_of_threads=1)

    idx = bp.morphology.distance_transform(a, halo=(0, 0), return_distances=False, return_indices=True)
    np.testing.assert_array_equal(idx, ref_idx)

    d2, i2 = bp.morphology.distance_transform(a, halo=(0, 0), return_distances=True, return_indices=True)
    np.testing.assert_allclose(d2, ref_dist)
    np.testing.assert_array_equal(i2, ref_idx)


def test_distance_transform_backend_determinism(zarr_factory, rng):
    shape, block_shape, halo = (48, 50), (16, 16), (12, 12)
    a = _binary(rng, shape)
    za = zarr_factory(a, chunks=block_shape)

    results = []
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        zd = zarr_factory(shape=shape, chunks=block_shape, dtype="float32", fill=0)
        bp.morphology.distance_transform(za, halo=halo, distances=zd, block_shape=block_shape,
                                         num_workers=nw, job_type=job)
        results.append(zd[:])
    for r in results[1:]:
        np.testing.assert_array_equal(results[0], r, err_msg="distance_transform backend mismatch")


def test_map_points_to_objects(zarr_factory):
    seg = np.zeros((40, 40), dtype="uint32")
    seg[5:10, 5:10] = 1
    seg[30:35, 30:35] = 2
    points = np.array([[7, 7], [32, 32], [12, 12]])  # inside obj1, inside obj2, nearest obj1
    zseg = zarr_factory(seg, chunks=(20, 20))

    res = []
    for nw, job, src in [(1, "local", seg), (4, "local", zseg), (3, "subprocess", zseg)]:
        ids, dists = bp.morphology.map_points_to_objects(
            src, points, block_shape=(20, 20), halo=(18, 18), num_workers=nw, job_type=job)
        res.append((ids, dists))

    ids0, dists0 = res[0]
    assert ids0[0] == 1 and dists0[0] == 0.0
    assert ids0[1] == 2 and dists0[1] == 0.0
    assert ids0[2] == 1  # (12, 12) is closest to object 1
    for ids, dists in res[1:]:
        np.testing.assert_array_equal(ids, ids0)
        np.testing.assert_allclose(dists, dists0)
