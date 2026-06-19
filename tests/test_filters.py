"""Tests for block-wise filters."""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp


@pytest.mark.parametrize("filter_name,bic_fn", [
    ("gaussian_smoothing", bic.filters.gaussian_smoothing),
    ("gaussian_gradient_magnitude", bic.filters.gaussian_gradient_magnitude),
    ("laplacian_of_gaussian", bic.filters.laplacian_of_gaussian),
])
@pytest.mark.parametrize("block_shape", [(16, 16), (13, 17)])
def test_filter_matches_reference(filter_name, bic_fn, block_shape, zarr_factory, rng):
    a = rng.random((48, 50)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    ref = bic_fn(a, 2.0)
    # Output chunks == block shape so each block writes exactly one chunk (write-safe). The
    # array shape is not divisible by the block shape, so edge blocks are still exercised.
    out = zarr_factory(shape=a.shape, chunks=block_shape, dtype="float32", fill=0.0)
    bp.filters.apply_filter(z, filter_name, 2.0, output=out, block_shape=block_shape, num_workers=4)
    # The halo must be large enough that there are no block-boundary seams.
    np.testing.assert_allclose(out[:], ref, atol=1e-4)


def test_filter_3d(zarr_factory, rng):
    a = rng.random((24, 22, 26)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8, 8))
    ref = bic.filters.gaussian_smoothing(a, 1.5)
    out = zarr_factory(shape=a.shape, chunks=(12, 12, 12), dtype="float32", fill=0.0)
    bp.filters.gaussian_smoothing(z, 1.5, output=out, block_shape=(12, 12, 12), num_workers=4)
    np.testing.assert_allclose(out[:], ref, atol=1e-4)


def test_output_optional_local(zarr_factory, rng):
    a = rng.random((40, 40)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    ref = bic.filters.gaussian_smoothing(a, 2.0)
    # No output -> a numpy array is allocated and returned (block-wise, local).
    result = bp.filters.gaussian_smoothing(z, 2.0, block_shape=(8, 8), num_workers=3)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, ref, atol=1e-4)


def test_output_required_distributed(zarr_factory, rng):
    z = zarr_factory(rng.random((16, 16)).astype("float32"), chunks=(8, 8))
    with pytest.raises(ValueError, match="required for distributed execution"):
        bp.filters.gaussian_smoothing(z, 1.0, block_shape=(8, 8), num_workers=2, job_type="subprocess")


def test_numpy_output_distributed_rejected_by_runner(zarr_factory, rng):
    # Explicitly passing an in-memory numpy output to a distributed run is rejected by the runner.
    z = zarr_factory(rng.random((16, 16)).astype("float32"), chunks=(8, 8))
    out = np.zeros((16, 16), dtype="float32")
    with pytest.raises(ValueError, match="file-backed"):
        bp.filters.gaussian_smoothing(z, 1.0, output=out, block_shape=(8, 8),
                                      num_workers=2, job_type="subprocess")


@pytest.mark.parametrize("filter_name,kwargs", [
    ("hessian_of_gaussian_eigenvalues", {}),
    ("structure_tensor_eigenvalues", {"outer_scale": 2.0}),
])
def test_multichannel_filter_parity(filter_name, kwargs, zarr_factory, rng):
    # Multi-channel filters write a leading channel axis (ndim, *spatial). Block-wise must match
    # the direct path, and the write-safety guard must align the block to the trailing chunks.
    a = rng.random((48, 50)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    ndim = a.ndim
    direct = bp.filters.apply_filter(a.copy(), filter_name, 2.0, **kwargs)  # (ndim, *spatial)
    assert direct.shape == (ndim,) + a.shape

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=(ndim,) + a.shape, chunks=(ndim, 16, 16), dtype="float32", fill=0.0)
        bp.filters.apply_filter(z, filter_name, 2.0, output=out, block_shape=(16, 16),
                                num_workers=nw, job_type=job, **kwargs)
        np.testing.assert_allclose(out[:], direct, atol=1e-3, err_msg=f"nw={nw} job={job}")


def test_multichannel_return_channel_scalar(zarr_factory, rng):
    # return_channel selects a single channel -> a scalar (spatial-only) output.
    a = rng.random((48, 50)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    direct = bp.filters.apply_filter(a.copy(), "hessian_of_gaussian_eigenvalues", 2.0, return_channel=0)
    assert direct.shape == a.shape
    out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
    bp.filters.apply_filter(z, "hessian_of_gaussian_eigenvalues", 2.0, return_channel=0,
                            output=out, block_shape=(16, 16), num_workers=4)
    np.testing.assert_allclose(out[:], direct, atol=1e-3)


def test_structure_tensor_requires_outer_scale(zarr_factory, rng):
    z = zarr_factory(rng.random((16, 16)).astype("float32"), chunks=(8, 8))
    with pytest.raises(ValueError, match="outer_scale"):
        bp.filters.apply_filter(z, "structure_tensor_eigenvalues", 1.0, block_shape=(8, 8))


def test_mask_keeps_out_of_mask_unchanged(zarr_factory, rng):
    a = rng.random((24, 24)).astype("float32")
    z = zarr_factory(a, chunks=(8, 8))
    mask = np.zeros((24, 24), dtype="uint8")
    mask[4:20, 4:20] = 1
    sentinel = -999.0
    out = zarr_factory(shape=a.shape, chunks=(8, 8), dtype="float32", fill=sentinel)
    bp.filters.gaussian_smoothing(z, 1.5, output=out, block_shape=(8, 8), num_workers=3, mask=mask)
    result = out[:]
    m = mask.astype(bool)
    assert np.all(result[~m] == sentinel)
    ref = bic.filters.gaussian_smoothing(a, 1.5)
    np.testing.assert_allclose(result[m], ref[m], atol=1e-4)
