"""File-backed base morphology tests."""
import importlib
import os

import numpy as np
import pandas as pd
import pytest

import bioimage_py as bp
from bioimage_py.runner import RunnerError
from bioimage_py.tables import TableDataset


def _segmentation():
    segmentation = np.zeros((12, 16), dtype="uint64")
    segmentation[1:8, 2:7] = 1
    segmentation[5:11, 9:15] = 11
    segmentation[0:2, 13:16] = 21
    return segmentation


def test_file_backed_morphology_matches_dataframe_and_partitions(
    tmp_path, zarr_factory,
):
    segmentation = _segmentation()
    source = zarr_factory(segmentation, chunks=(4, 4))
    output = tmp_path / "base.parquet"
    result = bp.morphology.morphology(
        source,
        block_shape=(4, 4),
        num_workers=2,
        output_table=output,
        blocks_per_batch=3,
        label_partition_size=10,
        provenance={"mask": "abc"},
    )

    assert isinstance(result, TableDataset)
    pd.testing.assert_frame_equal(
        result.to_pandas(), bp.morphology.morphology(segmentation),
    )
    assert [part.partition_metadata for part in result.iter_parts()] == [
        {"label_start": 0, "label_stop": 10},
        {"label_start": 10, "label_stop": 20},
        {"label_start": 20, "label_stop": 30},
    ]
    assert not os.path.exists(str(output) + ".morphology-work")
    definition = result._dataset_record()
    assert definition["parameters"]["provenance"] == {"mask": "abc"}


def test_file_backed_morphology_preserves_high_labels_and_empty_input(
    tmp_path, zarr_factory,
):
    high_label = 2**53 + 7
    segmentation = np.zeros((4, 6), dtype="uint64")
    segmentation[1:3, 2:5] = high_label
    source = zarr_factory(segmentation, chunks=(2, 3))
    result = bp.morphology.morphology(
        source,
        block_shape=(2, 3),
        output_table=tmp_path / "high.parquet",
        blocks_per_batch=2,
        label_partition_size=10,
    )
    frame = result.to_pandas()
    row = frame.iloc[0]
    assert int(frame["label"].iloc[0]) == high_label
    assert int(row["size"]) == 6
    assert row[["bb_min_y", "bb_min_x", "bb_max_y", "bb_max_x"]].tolist() == [1, 2, 3, 5]

    maximum = np.zeros((2, 2), dtype="uint64")
    maximum[1, 1] = np.iinfo(np.uint64).max
    maximum_source = zarr_factory(maximum, chunks=(1, 1))
    maximum_result = bp.morphology.morphology(
        maximum_source,
        block_shape=(1, 1),
        output_table=tmp_path / "maximum.parquet",
        label_partition_size=10,
    ).to_pandas()
    assert maximum_result["label"].tolist() == [np.iinfo(np.uint64).max]
    assert maximum_result["size"].tolist() == [1]

    empty_source = zarr_factory(
        np.zeros((4, 6), dtype="uint64"), chunks=(2, 3),
    )
    empty = bp.morphology.morphology(
        empty_source,
        output_table=tmp_path / "empty.parquet",
        block_shape=(2, 3),
    )
    assert empty.row_count == 0
    assert empty.part_count == 0
    assert list(empty.to_pandas().columns) == [
        "label", "size", "com_y", "com_x",
        "bb_min_y", "bb_min_x", "bb_max_y", "bb_max_x",
    ]


def test_file_backed_morphology_honors_mask_and_explicit_blocks(
    tmp_path, zarr_factory,
):
    segmentation = _segmentation()
    source = zarr_factory(segmentation, chunks=(4, 4))
    mask_data = np.zeros_like(segmentation, dtype="uint8")
    mask_data[:8, :8] = 1
    mask = zarr_factory(mask_data, chunks=(4, 4))
    result = bp.morphology.morphology(
        source,
        block_shape=(4, 4),
        mask=mask,
        block_ids=[0, 1, 4, 5],
        output_table=tmp_path / "masked.parquet",
        blocks_per_batch=2,
        label_partition_size=10,
    )
    expected = bp.morphology.morphology(
        segmentation,
        block_shape=(4, 4),
        mask=mask_data,
        block_ids=[0, 1, 4, 5],
    )
    pd.testing.assert_frame_equal(result.to_pandas(), expected)


def test_file_backed_morphology_reuses_partial_work_after_failure(
    tmp_path, zarr_factory, monkeypatch,
):
    module = importlib.import_module("bioimage_py.morphology.morphology")
    source = zarr_factory(_segmentation(), chunks=(4, 4))
    output = tmp_path / "resumed.parquet"
    real_partial = module._partial_batch
    failed = {"once": False}

    def fail_second(batch, writer, *, context):
        if batch.batch_id == 1 and not failed["once"]:
            failed["once"] = True
            raise RuntimeError("stop partial stage")
        return real_partial(batch, writer, context=context)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_partial_batch", fail_second)
        with pytest.raises(RunnerError):
            bp.morphology.morphology(
                source,
                block_shape=(4, 4),
                num_workers=1,
                output_table=output,
                blocks_per_batch=2,
                label_partition_size=10,
            )
    assert os.path.isdir(str(output) + ".morphology-work")

    result = bp.morphology.morphology(
        source,
        block_shape=(4, 4),
        num_workers=1,
        output_table=output,
        blocks_per_batch=2,
        label_partition_size=10,
    )
    pd.testing.assert_frame_equal(
        result.to_pandas(), bp.morphology.morphology(_segmentation()),
    )
    assert not os.path.exists(str(output) + ".morphology-work")


def test_file_backed_morphology_resumes_label_reduction(
    tmp_path, zarr_factory, monkeypatch,
):
    module = importlib.import_module("bioimage_py.morphology.morphology")
    source = zarr_factory(_segmentation(), chunks=(4, 4))
    output = tmp_path / "reduce-resumed.parquet"
    real_reduce = module._reduce_partition
    failed = {"once": False}

    def fail_second(batch, writer, *, context):
        if batch.batch_id == 1 and not failed["once"]:
            failed["once"] = True
            raise RuntimeError("stop reduction stage")
        return real_reduce(batch, writer, context=context)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_reduce_partition", fail_second)
        with pytest.raises(RunnerError):
            bp.morphology.morphology(
                source,
                block_shape=(4, 4),
                output_table=output,
                blocks_per_batch=2,
                label_partition_size=10,
            )

    def unexpected_partial(*args, **kwargs):
        raise AssertionError("completed partial batches must be reused")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_partial_batch", unexpected_partial)
        result = bp.morphology.morphology(
            source,
            block_shape=(4, 4),
            output_table=output,
            blocks_per_batch=2,
            label_partition_size=10,
        )
    pd.testing.assert_frame_equal(
        result.to_pandas(), bp.morphology.morphology(_segmentation()),
    )
    assert not os.path.exists(str(output) + ".morphology-work")


def test_file_backed_morphology_rejects_negative_labels(tmp_path, zarr_factory):
    segmentation = np.zeros((4, 4), dtype="int64")
    segmentation[0, 0] = -1
    source = zarr_factory(segmentation, chunks=(2, 2))
    with pytest.raises(RunnerError, match="batch"):
        bp.morphology.morphology(
            source,
            output_table=tmp_path / "negative.parquet",
            block_shape=(2, 2),
        )
