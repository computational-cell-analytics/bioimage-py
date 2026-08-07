"""File-backed regionprops behavior and scaling contracts."""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bioimage_py as bp
from bioimage_py.morphology import regionprops as regionprops_function
from bioimage_py.runner import RunnerConfig, RunnerError, get_runner
from bioimage_py.tables import TableDataset, _describe_parquet_input, _read_parquet_rows


def _segmentation():
    seg = np.zeros((24, 28, 32), dtype="uint64")
    seg[2:8, 3:12, 4:14] = 7
    seg[10:19, 15:25, 18:29] = 2
    seg[4:9, 18:24, 2:8] = 11
    return seg


def _expected_in_input_order(seg, base, *, resolution, compute_surface):
    expected = bp.morphology.regionprops(
        seg, base, resolution=resolution, compute_surface=compute_surface,
    ).set_index("label")
    expected = expected.loc[base["label"].tolist()].reset_index()
    expected["label"] = expected["label"].astype("uint64")
    expected["n_voxels"] = expected["n_voxels"].astype("uint64")
    return expected


def _base_dataset(path, base, *, batch_size=2):
    schema = pa.Table.from_pandas(base, preserve_index=False).schema
    dataset = TableDataset.create(
        path,
        schema=schema,
        schema_version=1,
        operation="test-base-morphology",
        operation_version="1",
        input_identities={"segmentation": "test"},
        parameters={},
    )

    def write(batch, writer):
        frame = base.iloc[batch.start:batch.stop]
        writer.write(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))

    return get_runner("local").map_batches(
        write, len(base), batch_size=batch_size, result_sink=dataset,
    )


@pytest.mark.parametrize("compute_surface", [False, True])
@pytest.mark.parametrize("table_kind", ["raw", "dataset"])
def test_file_backed_regionprops_matches_dataframe_and_preserves_order(
    tmp_path, zarr_factory, compute_surface, table_kind,
):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg).iloc[[2, 0, 1]].reset_index(drop=True)
    resolution = (2.0, 1.0, 1.5)
    if table_kind == "raw":
        table = tmp_path / "base.parquet"
        base.to_parquet(table, index=False, row_group_size=1)
    else:
        table = _base_dataset(tmp_path / "base-dataset", base, batch_size=2)

    result = bp.morphology.regionprops(
        source,
        table,
        resolution=resolution,
        compute_surface=compute_surface,
        output_table=tmp_path / f"result-{table_kind}-{compute_surface}.parquet",
        rows_per_batch=2,
        num_workers=2,
    )

    assert isinstance(result, TableDataset)
    assert [part.batch_id for part in result.iter_parts()] == [0, 1]
    actual = result.to_pandas()
    expected = _expected_in_input_order(
        seg, base, resolution=resolution, compute_surface=compute_surface,
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert actual["label"].tolist() == base["label"].tolist()
    assert result.schema.field("label").type == pa.uint64()
    assert result.schema.field("n_voxels").type == pa.uint64()
    assert all(not field.nullable for field in result.schema)
    assert ("surface_area" in result.schema.names) is compute_surface


def test_raw_parquet_directory_uses_lexical_file_order_and_bounded_ranges(tmp_path):
    root = tmp_path / "input"
    nested = root / "a"
    nested.mkdir(parents=True)
    pd.DataFrame({"value": [2, 3], "extra": [8, 9]}).to_parquet(
        root / "b.parquet", index=False, row_group_size=1,
    )
    pd.DataFrame({"value": [0, 1]}).to_parquet(
        nested / "part.parquet", index=False, row_group_size=1,
    )

    descriptor, _, _ = _describe_parquet_input(root, ["value"])
    table = _read_parquet_rows(descriptor, 1, 3, ["value"])
    assert table["value"].to_pylist() == [1, 2]


def test_file_backed_regionprops_subprocess_writes_no_result_records(tmp_path, zarr_factory):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg)
    base_path = tmp_path / "base.parquet"
    base.to_parquet(base_path, index=False)
    captured = {}

    def inspect(tmp_folder):
        captured["results"] = os.listdir(os.path.join(tmp_folder, "results"))
        captured["payload_size"] = os.path.getsize(os.path.join(tmp_folder, "payload.pkl"))

    result = bp.morphology.regionprops(
        source,
        base_path,
        output_table=tmp_path / "result.parquet",
        rows_per_batch=1,
        num_workers=2,
        job_type="subprocess",
        job_config=RunnerConfig(tmp_root=str(tmp_path)),
        pre_cleanup=inspect,
    )

    assert result.row_count == len(base)
    assert captured["results"] == []
    assert captured["payload_size"] < 4_096


def test_file_backed_payload_does_not_scale_with_rows(tmp_path, zarr_factory):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    one = bp.morphology.morphology(seg).iloc[[0]].reset_index(drop=True)
    payload_sizes = []
    for name, repeats in (("small", 5), ("large", 500)):
        base = pd.concat([one] * repeats, ignore_index=True)
        input_path = tmp_path / f"input-{name}.parquet"
        output_path = tmp_path / f"output-{name}.parquet"
        base.to_parquet(input_path, index=False, row_group_size=50)

        def inspect(tmp_folder):
            payload_sizes.append(os.path.getsize(os.path.join(tmp_folder, "payload.pkl")))

        bp.morphology.regionprops(
            source,
            input_path,
            output_table=output_path,
            rows_per_batch=100,
            num_workers=2,
            job_type="subprocess",
            job_config=RunnerConfig(tmp_root=str(tmp_path)),
            pre_cleanup=inspect,
        )

    assert len(payload_sizes) == 2
    assert abs(payload_sizes[1] - payload_sizes[0]) < 128


def test_file_backed_regionprops_empty_input(tmp_path, zarr_factory):
    seg = np.zeros((6, 8, 10), dtype="uint64")
    source = zarr_factory(seg, chunks=(3, 4, 5))
    base = bp.morphology.morphology(seg)
    base_path = tmp_path / "empty.parquet"
    base.to_parquet(base_path, index=False)

    result = bp.morphology.regionprops(
        source,
        base_path,
        compute_surface=True,
        output_table=tmp_path / "result.parquet",
        rows_per_batch=3,
    )

    assert result.row_count == 0
    assert result.part_count == 0
    assert result.to_pandas().empty
    assert "surface_area" in result.schema.names


def test_file_backed_regionprops_2d_omits_surface_column(tmp_path, zarr_factory):
    seg = np.zeros((18, 22), dtype="uint64")
    seg[2:9, 3:12] = 3
    source = zarr_factory(seg, chunks=(9, 11))
    base = bp.morphology.morphology(seg)
    base_path = tmp_path / "base-2d.parquet"
    base.to_parquet(base_path, index=False)

    result = bp.morphology.regionprops(
        source,
        base_path,
        compute_surface=True,
        output_table=tmp_path / "result-2d.parquet",
        rows_per_batch=1,
    )

    expected = _expected_in_input_order(
        seg, base, resolution=(1.0, 1.0), compute_surface=True,
    )
    pd.testing.assert_frame_equal(result.to_pandas(), expected)
    assert "surface_area" not in result.schema.names


def test_file_backed_regionprops_handles_absent_and_degenerate_objects(
    tmp_path, zarr_factory,
):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    row = bp.morphology.morphology(seg).iloc[[0]].reset_index(drop=True)
    absent = row.copy()
    absent["label"] = np.uint64(100)
    degenerate = row.copy()
    degenerate["label"] = np.uint64(101)
    for axis in ("z", "y", "x"):
        degenerate[f"bb_max_{axis}"] = degenerate[f"bb_min_{axis}"]
    base = pd.concat([absent, degenerate], ignore_index=True)
    base_path = tmp_path / "base.parquet"
    base.to_parquet(base_path, index=False)

    result = bp.morphology.regionprops(
        source,
        base_path,
        output_table=tmp_path / "result.parquet",
        rows_per_batch=2,
    ).to_pandas()

    assert result["label"].tolist() == [100, 101]
    assert result["n_voxels"].tolist() == [0, 0]
    assert result.loc[0, "extent"] == 0.0
    assert np.isnan(result.loc[1, "extent"])


def test_file_backed_rerun_skips_valid_parts_and_repairs_invalid_part(
    tmp_path, zarr_factory, monkeypatch,
):
    import importlib

    regionprops_module = importlib.import_module("bioimage_py.morphology.regionprops")

    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg)
    base_path = tmp_path / "base.parquet"
    output_path = tmp_path / "result.parquet"
    base.to_parquet(base_path, index=False)
    first = bp.morphology.regionprops(
        source, base_path, output_table=output_path, rows_per_batch=2,
    )

    real_compute = regionprops_module._compute_object_features
    calls = {"count": 0}

    def track(*args, **kwargs):
        calls["count"] += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(regionprops_module, "_compute_object_features", track)
    unchanged = bp.morphology.regionprops(
        source, base_path, output_table=output_path, rows_per_batch=2,
    )
    assert calls["count"] == 0
    pd.testing.assert_frame_equal(unchanged.to_pandas(), first.to_pandas())

    first_part = next(unchanged.iter_parts()).path
    with open(first_part, "r+b") as file:
        file.truncate(16)
    repaired = bp.morphology.regionprops(
        source, base_path, output_table=output_path, rows_per_batch=2,
    )
    assert calls["count"] == 2
    pd.testing.assert_frame_equal(repaired.to_pandas(), first.to_pandas())


def test_file_backed_regionprops_resumes_distributed_run(tmp_path, zarr_factory):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg)
    base_path = tmp_path / "base.parquet"
    output_path = tmp_path / "result.parquet"
    base.to_parquet(base_path, index=False)

    with pytest.raises(RunnerError) as error_info:
        bp.morphology.regionprops(
            source,
            base_path,
            output_table=output_path,
            rows_per_batch=1,
            num_workers=2,
            job_type="subprocess",
            job_config=RunnerConfig(tmp_root=str(tmp_path), task_timeout=0.01),
        )

    result = bp.morphology.regionprops(
        source,
        base_path,
        output_table=output_path,
        rows_per_batch=1,
        num_workers=2,
        job_type="subprocess",
        job_config=RunnerConfig(tmp_root=str(tmp_path)),
        resume_from=error_info.value.tmp_folder,
    )
    expected = _expected_in_input_order(
        seg, base, resolution=(1.0, 1.0, 1.0), compute_surface=False,
    )
    pd.testing.assert_frame_equal(result.to_pandas(), expected)


def test_file_backed_regionprops_rejects_incompatible_arguments(tmp_path, zarr_factory):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg)
    base_path = tmp_path / "base.parquet"
    base.to_parquet(base_path, index=False)

    with pytest.raises(ValueError, match="reopenable segmentation"):
        bp.morphology.regionprops(
            seg, base_path, output_table=tmp_path / "numpy-output",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        bp.morphology.regionprops(
            source,
            base_path,
            output_path=str(tmp_path / "out.csv"),
            output_table=tmp_path / "table-output",
        )
    with pytest.raises(ValueError, match="item_ids"):
        bp.morphology.regionprops(
            source, base_path, output_table=tmp_path / "subset-output", item_ids=[0],
        )
    with pytest.raises(ValueError, match="must be positive"):
        bp.morphology.regionprops(
            source, base_path, output_table=tmp_path / "bad-size", rows_per_batch=0,
        )
    with pytest.raises(ValueError, match="must not contain"):
        bp.morphology.regionprops(
            source, base_path, output_table=tmp_path,
        )


def test_file_backed_regionprops_rejects_schema_and_data_errors(tmp_path, zarr_factory):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg)
    missing_path = tmp_path / "missing.parquet"
    base.drop(columns=["com_z"]).to_parquet(missing_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        bp.morphology.regionprops(
            source, missing_path, output_table=tmp_path / "missing-output",
        )

    negative = base.copy()
    negative["label"] = negative["label"].astype("int64")
    negative.loc[0, "label"] = -1
    negative_path = tmp_path / "negative.parquet"
    negative.to_parquet(negative_path, index=False)
    with pytest.raises(RunnerError):
        bp.morphology.regionprops(
            source, negative_path, output_table=tmp_path / "negative-output",
            rows_per_batch=2,
        )

    null_com = base.copy()
    null_com.loc[0, "com_z"] = np.nan
    null_path = tmp_path / "null.parquet"
    null_com.to_parquet(null_path, index=False)
    with pytest.raises(RunnerError):
        bp.morphology.regionprops(
            source, null_path, output_table=tmp_path / "null-output", rows_per_batch=2,
        )

    invalid_box = base.copy()
    invalid_box.loc[0, "bb_min_z"] = -1
    invalid_box_path = tmp_path / "invalid-box.parquet"
    invalid_box.to_parquet(invalid_box_path, index=False)
    with pytest.raises(RunnerError):
        bp.morphology.regionprops(
            source,
            invalid_box_path,
            output_table=tmp_path / "invalid-box-output",
            rows_per_batch=2,
        )

    fragments = tmp_path / "fragments"
    fragments.mkdir()
    base.iloc[:1].to_parquet(fragments / "a.parquet", index=False)
    base.iloc[1:].assign(com_z="invalid").to_parquet(
        fragments / "b.parquet", index=False,
    )
    with pytest.raises(ValueError, match="different schema"):
        bp.morphology.regionprops(
            source, fragments, output_table=tmp_path / "fragment-output",
        )


def test_file_backed_regionprops_rejects_changed_output_identity(tmp_path, zarr_factory):
    seg = _segmentation()
    source = zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg)
    base_path = tmp_path / "base.parquet"
    output_path = tmp_path / "result.parquet"
    base.to_parquet(base_path, index=False)
    bp.morphology.regionprops(
        source,
        base_path,
        resolution=(1.0, 1.0, 1.0),
        output_table=output_path,
        rows_per_batch=2,
    )

    with pytest.raises(ValueError, match="incompatible definition"):
        bp.morphology.regionprops(
            source,
            base_path,
            resolution=(2.0, 1.0, 1.0),
            output_table=output_path,
            rows_per_batch=2,
        )
    with pytest.raises(ValueError, match="incompatible definition"):
        bp.morphology.regionprops(
            source,
            base_path,
            resolution=(1.0, 1.0, 1.0),
            output_table=output_path,
            rows_per_batch=1,
        )

    other_source = zarr_factory(seg, chunks=(12, 14, 16))
    with pytest.raises(ValueError, match="incompatible definition"):
        bp.morphology.regionprops(
            other_source,
            base_path,
            resolution=(1.0, 1.0, 1.0),
            output_table=output_path,
            rows_per_batch=2,
        )

    pd.concat([base, base.iloc[[0]]], ignore_index=True).to_parquet(base_path, index=False)
    with pytest.raises(ValueError, match="incompatible definition"):
        bp.morphology.regionprops(
            source,
            base_path,
            resolution=(1.0, 1.0, 1.0),
            output_table=output_path,
            rows_per_batch=2,
        )


def test_table_dataset_input_requires_file_backed_output(tmp_path):
    base = bp.morphology.morphology(_segmentation())
    dataset = _base_dataset(tmp_path / "base", base)
    with pytest.raises(ValueError, match="requires output_table"):
        regionprops_function(_segmentation(), dataset)
