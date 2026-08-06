"""Tests for the TensorStore neuroglancer-precomputed source."""
from __future__ import annotations

import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.sources import SourceSpec, from_spec, open_tensorstore_precomputed
from bioimage_py.util import get_blocking, group_blocks_by_shard


@pytest.fixture
def precomputed_factory(tmp_path):
    """Create local precomputed layers with independently controlled read/write chunks."""
    ts = pytest.importorskip("tensorstore")
    counter = {"i": 0}

    def _make(
        array_zyx=None,
        *,
        shape_zyx=None,
        dtype="uint64",
        offset_xyz=(11, 22, 33),
        chunks_zyx=(4, 8, 8),
        shards_zyx=(8, 16, 16),
        num_channels=1,
    ):
        counter["i"] += 1
        path = tmp_path / f"precomputed_{counter['i']}"
        if array_zyx is not None:
            shape_zyx = array_zyx.shape
            dtype = array_zyx.dtype
        if shape_zyx is None:
            raise ValueError("shape_zyx or array_zyx is required")

        shape_xyzc = tuple(int(value) for value in shape_zyx[::-1]) + (num_channels,)
        chunks_xyzc = tuple(int(value) for value in chunks_zyx[::-1]) + (num_channels,)
        shards_xyzc = tuple(int(value) for value in shards_zyx[::-1]) + (num_channels,)
        store = ts.open(
            {
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": str(path)},
            },
            create=True,
            dtype=ts.dtype(np.dtype(dtype)),
            domain=ts.IndexDomain(
                inclusive_min=tuple(offset_xyz) + (0,),
                shape=shape_xyzc,
                labels=("x", "y", "z", "channel"),
            ),
            chunk_layout=ts.ChunkLayout(
                read_chunk_shape=chunks_xyzc,
                write_chunk_shape=shards_xyzc,
            ),
            codec=ts.CodecSpec(
                {
                    "driver": "neuroglancer_precomputed",
                    "encoding": "compressed_segmentation",
                    "shard_data_encoding": "gzip",
                }
            ),
            dimension_units=[None, None, None, None],
        ).result()
        if array_zyx is not None:
            store[..., 0].write(np.asarray(array_zyx).transpose(2, 1, 0)).result()
        return str(path)

    return _make


def test_metadata_indexing_and_read(precomputed_factory):
    data = np.arange(16 * 32 * 32, dtype="uint64").reshape(16, 32, 32)
    path = precomputed_factory(data)

    src = open_tensorstore_precomputed(path)
    assert src.shape == data.shape
    assert src.chunks == (4, 8, 8)
    assert src.shards == (8, 16, 16)
    assert src.dtype == data.dtype
    assert src.writable is False
    np.testing.assert_array_equal(src[:], data)
    np.testing.assert_array_equal(src[2:11, 3:19, 5:27], data[2:11, 3:19, 5:27])
    assert src[-1, -2, -3] == data[-1, -2, -3]


def test_empty_slices_match_numpy(precomputed_factory):
    data = np.arange(8 * 16 * 16, dtype="uint64").reshape(8, 16, 16)
    src = open_tensorstore_precomputed(precomputed_factory(data))
    indices = [
        (slice(None, 0), slice(None), slice(None)),
        (slice(3, 2), slice(None), slice(None)),
        (slice(None), slice(99, None), slice(None)),
        (slice(None), slice(None), slice(None, -99)),
    ]

    for index in indices:
        np.testing.assert_array_equal(src[index], data[index])


def test_readonly_and_readwrite_modes(precomputed_factory):
    data = np.zeros((8, 16, 16), dtype="uint64")
    path = precomputed_factory(data)

    with pytest.raises(TypeError, match="read-only"):
        open_tensorstore_precomputed(path)[1:3, 2:5, 4:8] = 7

    writable = open_tensorstore_precomputed(path, mode="r+")
    assert writable.writable is True
    expected = np.arange(2 * 3 * 4, dtype="uint64").reshape(2, 3, 4)
    writable[1:3, 2:5, 4:8] = expected
    np.testing.assert_array_equal(
        open_tensorstore_precomputed(path)[1:3, 2:5, 4:8], expected
    )

    with pytest.raises(ValueError, match="mode"):
        open_tensorstore_precomputed(path, mode="w")


def test_broadcast_writes_and_empty_write(precomputed_factory):
    data = np.zeros((8, 16, 16), dtype="uint64")
    path = precomputed_factory(data)
    writable = open_tensorstore_precomputed(path, mode="r+")

    writable[0:2, 0:3, 0:4] = np.uint64(7)
    np.testing.assert_array_equal(writable[0:2, 0:3, 0:4], np.full((2, 3, 4), 7))

    plane = np.arange(4 * 4, dtype="uint64").reshape(4, 4)
    writable[2:4, 3:7, 4:8] = plane
    np.testing.assert_array_equal(
        writable[2:4, 3:7, 4:8], np.broadcast_to(plane, (2, 4, 4))
    )

    before = writable[:]
    writable[:0] = np.uint64(99)
    np.testing.assert_array_equal(writable[:], before)

    with pytest.raises(ValueError, match="broadcast"):
        writable[0:2, 0:4, 0:4] = np.ones((3, 3), dtype="uint64")


def test_copy_casts_to_precomputed_dtype(precomputed_factory):
    data = np.arange(8 * 16 * 16, dtype="int64").reshape(8, 16, 16)
    path = precomputed_factory(shape_zyx=data.shape, dtype="uint64")
    output = open_tensorstore_precomputed(path, mode="r+")

    bp.copy(data, output)

    expected = data.astype("uint64")
    np.testing.assert_array_equal(open_tensorstore_precomputed(path)[:], expected)


def test_current_and_legacy_spec_roundtrip(precomputed_factory):
    data = np.arange(8 * 16 * 16, dtype="uint64").reshape(8, 16, 16)
    path = precomputed_factory(data)

    writable = open_tensorstore_precomputed(path, mode="r+")
    spec = writable.to_spec()
    assert spec.kind == "tensorstore_precomputed"
    assert spec.params["mode"] == "r+"
    reopened = from_spec(spec)
    assert reopened.writable is True
    np.testing.assert_array_equal(reopened[:], data)

    legacy_params = dict(spec.params)
    legacy_params.pop("mode")
    legacy = SourceSpec(kind="tensorstore_precomputed", path=path, params=legacy_params)
    legacy_reopened = from_spec(legacy)
    assert legacy_reopened.writable is False
    np.testing.assert_array_equal(legacy_reopened[:], data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"offset_xyz": (12, 22, 33)}, "offset_xyz"),
        ({"size_zyx": (9, 16, 16)}, "size_zyx"),
        ({"chunks_zyx": (2, 8, 8)}, "chunks_zyx"),
        ({"dtype": "uint32"}, "dtype"),
    ],
)
def test_metadata_expectations_are_validated(precomputed_factory, kwargs, match):
    path = precomputed_factory(shape_zyx=(8, 16, 16))
    with pytest.raises(ValueError, match=match):
        open_tensorstore_precomputed(path, **kwargs)


def test_multichannel_layer_is_rejected(precomputed_factory):
    path = precomputed_factory(shape_zyx=(8, 16, 16), num_channels=2)
    with pytest.raises(ValueError, match="single-channel"):
        open_tensorstore_precomputed(path)


def test_subprocess_reduction_reopens_source(precomputed_factory):
    data = np.arange(16 * 32 * 32, dtype="uint64").reshape(16, 32, 32)
    path = precomputed_factory(data)
    result = bp.stats.max(
        open_tensorstore_precomputed(path),
        block_shape=(4, 8, 8),
        num_workers=3,
        job_type="subprocess",
    )
    assert result == float(data.max())


@pytest.mark.parametrize("job_type", ["local", "subprocess"])
def test_copy_to_sharded_precomputed_is_safe(
    precomputed_factory, zarr_factory, job_type
):
    data = np.arange(16 * 32 * 32, dtype="uint64").reshape(16, 32, 32)
    input_ = zarr_factory(data, chunks=(4, 8, 8))
    path = precomputed_factory(shape_zyx=data.shape)
    output = open_tensorstore_precomputed(path, mode="r+")

    blocking = get_blocking(data.shape, (4, 8, 8))
    block_ids = list(range(int(blocking.number_of_blocks)))
    groups = group_blocks_by_shard(blocking, [output], block_ids)
    assert groups is not None
    assert len(groups) == 8
    assert all(len(group) == 8 for group in groups)

    bp.copy(
        input_,
        output,
        block_shape=(4, 8, 8),
        num_workers=4,
        job_type=job_type,
    )
    np.testing.assert_array_equal(open_tensorstore_precomputed(path)[:], data)
