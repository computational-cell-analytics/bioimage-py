"""Tests for sharded-output write-safety: shard-exclusive block grouping, the relaxed
``_validate_write_safety`` guard, and the load-imbalance warning (audit issue #1)."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.runner.base import Runner
from bioimage_py.sources.dispatch import as_source
from bioimage_py.util import get_blocking, group_blocks_by_shard


def _shard_cell(block, shard):
    """The shard-grid cell a block's origin falls into."""
    return tuple(int(b) // s for b, s in zip(block.begin, shard))


def test_group_blocks_by_shard_basic(zarr_factory):
    # 64x64 in 16x16 blocks (16 blocks); 32x32 shards -> 2x2 blocks per shard -> 4 groups of 4.
    out = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(32, 32),
                                 dtype="float32", fill=0.0))
    blocking = get_blocking((64, 64), (16, 16))
    ids = list(range(int(blocking.number_of_blocks)))

    groups = group_blocks_by_shard(blocking, [out], ids)
    assert groups is not None
    assert sorted(len(g) for g in groups) == [4, 4, 4, 4]
    # Every block in a group writes into the same shard cell.
    for g in groups:
        cells = {_shard_cell(blocking.get_block(b), (32, 32)) for b in g}
        assert len(cells) == 1, (g, cells)
    # Partition is complete and disjoint.
    assert sorted(b for g in groups for b in g) == ids


def test_group_blocks_by_shard_none_when_unsharded(zarr_factory):
    out = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), dtype="float32", fill=0.0))
    blocking = get_blocking((64, 64), (16, 16))
    ids = list(range(int(blocking.number_of_blocks)))
    assert group_blocks_by_shard(blocking, [out], ids) is None


def test_group_blocks_by_shard_straddling(zarr_factory):
    # block (32, 16) vs shard (32, 32): the two adjacent column-blocks straddle into one shard
    # column and must be merged -> 4 groups of 2.
    out = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(32, 32),
                                 dtype="float32", fill=0.0))
    blocking = get_blocking((64, 64), (32, 16))
    ids = list(range(int(blocking.number_of_blocks)))
    groups = group_blocks_by_shard(blocking, [out], ids)
    assert sorted(len(g) for g in groups) == [2, 2, 2, 2]
    for g in groups:
        cells = {_shard_cell(blocking.get_block(b), (32, 32)) for b in g}
        assert len(cells) == 1, (g, cells)


def test_group_blocks_by_shard_multiple_outputs(zarr_factory):
    # Two sharded outputs with different shard shapes must be respected jointly: output A
    # (32x32) groups quadrants; output B (16x64) groups whole block-rows. Their union is
    # coarser than either alone -> top and bottom halves merge into 2 groups of 8.
    out_a = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(32, 32),
                                   dtype="float32", fill=0.0))
    out_b = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(16, 64),
                                   dtype="float32", fill=0.0))
    blocking = get_blocking((64, 64), (16, 16))
    ids = list(range(int(blocking.number_of_blocks)))

    assert len(group_blocks_by_shard(blocking, [out_a], ids)) == 4  # A alone
    joint = group_blocks_by_shard(blocking, [out_a, out_b], ids)
    assert sorted(len(g) for g in joint) == [8, 8]


def test_validate_write_safety_sharded_allows_unaligned(zarr_factory):
    # A sharded output with a block shape that is not a shard multiple must NOT raise: routing
    # makes it safe.
    out = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(64, 64),
                                 dtype="float32", fill=0.0))
    Runner._validate_write_safety([out], (16, 16))  # no exception


def test_validate_write_safety_unsharded_still_raises(zarr_factory):
    out = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), dtype="float32", fill=0.0))
    with pytest.raises(ValueError, match="not a multiple"):
        Runner._validate_write_safety([out], (24, 24))


def test_validate_write_safety_mixed_guards_unsharded(zarr_factory):
    # A sharded output is exempt, but a non-sharded sibling with a misaligned block still trips.
    sharded = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(64, 64),
                                     dtype="float32", fill=0.0))
    plain = as_source(zarr_factory(shape=(64, 64), chunks=(16, 16), dtype="float32", fill=0.0))
    with pytest.raises(ValueError, match="not a multiple"):
        Runner._validate_write_safety([sharded, plain], (24, 24))


@pytest.mark.parametrize("job", ["local", "subprocess"])
def test_imbalance_warning(zarr_factory, rng, job):
    # One shard covering the whole array -> a single shard-group -> fewer groups than workers,
    # so some workers are idle and a warning fires.
    a = rng.random((64, 64)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=(64, 64), chunks=(16, 16), shards=(64, 64), dtype="float32", fill=0.0)
    with pytest.warns(UserWarning, match="idle"):
        bp.copy(z, out, block_shape=(16, 16), num_workers=4, job_type=job)
    np.testing.assert_array_equal(out[:], a)
