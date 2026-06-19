"""Parity tests using n5 (z5py) as an op input/output (the other op tests cover only zarr)."""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp


def test_n5_reduction_input_subprocess(n5_factory, rng):
    z5py = pytest.importorskip("z5py")
    a = rng.random((40, 48)).astype("float32")
    in_path, in_key = n5_factory(a, chunks=(16, 16))
    src = z5py.File(in_path, "r")[in_key]
    # A reduction (return-value channel) over an n5 input, distributed.
    got = bp.stats.max(src, block_shape=(16, 16), num_workers=3, job_type="subprocess")
    assert np.isclose(got, a.max())


def test_n5_filter_input_and_output_subprocess(n5_factory, tmp_path, rng):
    z5py = pytest.importorskip("z5py")
    a = rng.random((40, 48)).astype("float32")
    in_path, in_key = n5_factory(a, chunks=(16, 16))
    src = z5py.File(in_path, "r")[in_key]

    ref = bic.filters.gaussian_smoothing(a, 2.0)
    out = z5py.File(str(tmp_path / "out.n5"), "a").create_dataset(
        "out", shape=a.shape, chunks=(16, 16), dtype="float32"
    )
    # An array-output op reading an n5 input and writing an n5 output, distributed.
    bp.filters.gaussian_smoothing(src, 2.0, output=out, block_shape=(16, 16),
                                  num_workers=3, job_type="subprocess")
    np.testing.assert_allclose(out[:], ref, atol=1e-4)
