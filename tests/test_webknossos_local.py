"""Local-dataset WebKnossos source tests: mag handling + ZYX transpose, no live server needed."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources.dispatch import from_spec


@pytest.fixture
def wk_local(tmp_path):
    """Build a tiny local WebKnossos dataset (mag1 + a manually written mag2)."""
    wk = pytest.importorskip("webknossos")
    data = np.arange(16 * 16 * 16, dtype="uint8").reshape(16, 16, 16)  # x, y, z
    ds = wk.Dataset(str(tmp_path / "ds"), voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", wk.COLOR_CATEGORY, dtype="uint8", num_channels=1)
    layer.add_mag(1).write(data, absolute_offset=(0, 0, 0), allow_resize=True)
    # Write mag2 directly (strided subsample) to avoid the multiprocessing downsample executor.
    layer.add_mag(2).write(data[::2, ::2, ::2], absolute_offset=(0, 0, 0), allow_resize=True)
    return str(tmp_path / "ds"), data


@pytest.mark.parametrize("mag", [1, 2])
def test_webknossos_local_mag(wk_local, mag):
    path, data = wk_local
    src = bp.open_webknossos(path, layer_name="color", mag=mag)
    gt = data if mag == 1 else data[::2, ::2, ::2]  # x, y, z ground truth at this magnification
    gt_zyx = np.transpose(gt, (2, 1, 0))            # the source presents z, y, x

    # The shape is the mag-level extent (8,8,8 at mag2), not the mag1 / shard-padded storage extent.
    assert src.shape == gt_zyx.shape
    assert src.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(src[:], gt_zyx)
    np.testing.assert_array_equal(src[1:5, 2:6, 0:7], gt_zyx[1:5, 2:6, 0:7])  # block-wise stitch

    # The spec round-trips: a reopened source indexes identically.
    src2 = from_spec(src.to_spec())
    assert src2.shape == gt_zyx.shape
    np.testing.assert_array_equal(src2[:], gt_zyx)


def test_webknossos_multichannel_rejected(tmp_path):
    wk = pytest.importorskip("webknossos")
    ds = wk.Dataset(str(tmp_path / "ds"), voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", wk.COLOR_CATEGORY, dtype="uint8", num_channels=3)
    layer.add_mag(1).write(np.zeros((3, 8, 8, 8), dtype="uint8"),
                           absolute_offset=(0, 0, 0), allow_resize=True)
    with pytest.raises(ValueError, match="single-channel"):
        bp.open_webknossos(str(tmp_path / "ds"), layer_name="color", mag=1)
