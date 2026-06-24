"""Tests for block-wise element-wise operations."""
import numpy as np

import bioimage_py as bp


def test_scalar_op_parity(zarr_factory, rng):
    a = rng.random((33, 28)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    exp = np.add(a, 2.5)

    np.testing.assert_allclose(bp.operations.add(a, 2.5), exp, rtol=1e-6)  # direct
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(8, 8), dtype=exp.dtype, fill=0)
        bp.operations.add(z, 2.5, out, block_shape=(8, 8), num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], exp, rtol=1e-6, err_msg=f"add mismatch ({nw}, {job})")


def test_array_op_parity(zarr_factory, rng):
    a = rng.random((24, 20)).astype("float32")
    b = rng.random((24, 20)).astype("float32")
    za, zb = zarr_factory(a, chunks=(8, 8)), zarr_factory(b, chunks=(8, 8))
    exp = np.multiply(a, b)

    np.testing.assert_allclose(bp.operations.multiply(a, b), exp, rtol=1e-6)  # direct
    for nw, job in [(4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="float32", fill=0)
        bp.operations.multiply(za, zb, out, block_shape=(8, 8), num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], exp, rtol=1e-6)


def test_broadcast_operand(zarr_factory, rng):
    a = rng.random((4, 12)).astype("float32")
    b = rng.random((1, 12)).astype("float32")  # broadcast along axis 0
    za = zarr_factory(a, chunks=(2, 4))
    exp = a + b

    np.testing.assert_allclose(bp.operations.add(a, b), exp, rtol=1e-6)  # direct
    for nw, job in [(4, "local"), (3, "subprocess")]:  # b is captured into the worker closure
        out = zarr_factory(shape=a.shape, chunks=(2, 4), dtype="float32", fill=0)
        bp.operations.add(za, b, out, block_shape=(2, 4), num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], exp, rtol=1e-6)


def test_comparison_allocates_bool(rng):
    a = rng.random((16, 14)).astype("float32")
    out = bp.operations.greater(a, 0.5)  # direct, output=None -> allocate (inferred dtype)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, a > 0.5)


def test_isin_parity(zarr_factory, rng):
    a = rng.integers(0, 10, size=(20, 18)).astype("uint8")
    za = zarr_factory(a, chunks=(8, 8))
    test_vals = [1, 3, 5, 7]
    exp = np.isin(a, test_vals)

    np.testing.assert_array_equal(bp.operations.isin(a, test_vals), exp)  # direct
    for nw, job in [(4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(8, 8), dtype=bool, fill=False)
        bp.operations.isin(za, test_vals, out, block_shape=(8, 8), num_workers=nw, job_type=job)
        np.testing.assert_array_equal(out[:], exp)


def test_in_place(zarr_factory, rng):
    a = rng.random((20, 16)).astype("float32")
    za = zarr_factory(a, chunks=(8, 8))
    bp.operations.multiply(za, 3.0, za, block_shape=(8, 8), num_workers=4)  # output is the input
    np.testing.assert_allclose(za[:], a * 3.0, rtol=1e-6)


def test_mask_leaves_outside_unchanged(rng):
    a = rng.random((20, 18)).astype("float32")
    mask = np.zeros((20, 18), dtype="uint8")
    mask[2:14, 3:15] = 1  # not block-aligned
    out = bp.operations.add(a, 100.0, block_shape=(5, 5), num_workers=2, mask=mask)
    m = mask.astype(bool)
    exp = np.zeros_like(out)  # out-of-mask voxels stay at the freshly allocated 0.
    exp[m] = a[m] + 100.0
    np.testing.assert_allclose(out, exp, rtol=1e-6)
