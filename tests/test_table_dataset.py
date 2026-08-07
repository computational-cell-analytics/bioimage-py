"""Parquet table datasets and runner result sinks."""
import json
import os

import pyarrow as pa
import pytest

from bioimage_py.runner import RunnerConfig, RunnerError, get_runner
from bioimage_py.tables import TableDataset


SCHEMA = pa.schema([("value", pa.int64()), ("batch", pa.int32())])


def _dataset(path, **changes):
    arguments = {
        "schema": SCHEMA,
        "schema_version": 1,
        "operation": "test-table",
        "operation_version": "1",
        "input_identities": {"source": {"path": "/input", "version": 3}},
        "parameters": {"enabled": True},
    }
    arguments.update(changes)
    return TableDataset.create(path, **arguments)


def _write_values(batch, writer):
    writer.write(pa.table({
        "value": pa.array(list(batch), type=pa.int64()),
        "batch": pa.array([batch.batch_id] * batch.size, type=pa.int32()),
    }, schema=SCHEMA))


def test_local_table_sink_roundtrip_and_order(tmp_path):
    dataset = _dataset(tmp_path / "result.parquet")
    result = get_runner("local").map_batches(
        _write_values, 7, batch_size=3, num_workers=2, result_sink=dataset,
    )

    assert isinstance(result, TableDataset)
    assert result.path == str(tmp_path / "result.parquet")
    assert result.row_count == 7
    assert result.part_count == 3
    assert result.schema.equals(SCHEMA, check_metadata=True)
    assert [part.batch_id for part in result.iter_parts()] == [0, 1, 2]
    assert [os.path.basename(part.path) for part in result.iter_parts()] == [
        "part-000000000000.parquet",
        "part-000000000001.parquet",
        "part-000000000002.parquet",
    ]
    assert result.to_pandas().to_dict(orient="list") == {
        "value": list(range(7)), "batch": [0, 0, 0, 1, 1, 1, 2],
    }
    assert result.validate() is result


def test_table_sink_accepts_and_persists_explicit_boundaries(tmp_path):
    path = tmp_path / "irregular.parquet"
    result = get_runner("local").map_batches(
        _write_values,
        batch_boundaries=[0, 1, 5, 7],
        num_workers=2,
        result_sink=_dataset(path),
    )

    assert result.row_count == 7
    assert [(part.start, part.stop) for part in result.iter_parts()] == [
        (0, 1), (1, 5), (5, 7),
    ]
    with open(path / "dataset.json") as file:
        record = json.load(file)
    assert record["work_plan"] == {
        "kind": "boundary_batches", "boundaries": [0, 1, 5, 7], "length": 3,
    }
    assert TableDataset.open(path).validate().row_count == 7

    with pytest.raises(ValueError, match="different batch work plan"):
        get_runner("local").map_batches(
            _write_values,
            batch_boundaries=[0, 2, 5, 7],
            result_sink=_dataset(path),
        )


def test_empty_table_dataset_and_column_selection(tmp_path):
    result = get_runner("local").map_batches(
        _write_values, 0, batch_size=3, result_sink=_dataset(tmp_path / "empty"),
    )
    assert result.row_count == 0
    assert result.part_count == 0
    assert list(result.to_pandas().columns) == ["value", "batch"]
    assert list(result.to_pandas(columns=["batch"]).columns) == ["batch"]


def test_zero_row_batch_writes_a_typed_part(tmp_path):
    def write_empty(batch, writer):
        writer.write(pa.Table.from_batches([], schema=SCHEMA))

    result = get_runner("local").map_batches(
        write_empty, 1, batch_size=1, result_sink=_dataset(tmp_path / "empty-part"),
    )
    assert result.row_count == 0
    assert result.part_count == 1
    assert list(result.to_pandas().columns) == ["value", "batch"]


def test_table_dataset_rejects_identity_and_work_plan_mismatches(tmp_path):
    path = tmp_path / "result"
    dataset = _dataset(path)
    get_runner("local").map_batches(
        _write_values, 4, batch_size=2, result_sink=dataset,
    )

    assert _dataset(path).operation == "test-table"
    with pytest.raises(ValueError, match="incompatible definition"):
        _dataset(path, operation_version="2")
    with pytest.raises(ValueError, match="different batch work plan"):
        get_runner("local").map_batches(
            _write_values, 5, batch_size=2, result_sink=_dataset(path),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        get_runner("local").map_batches(
            _write_values, 4, batch_size=2, result_sink=_dataset(path),
            has_return_val=True,
        )


@pytest.mark.parametrize("changes", [
    {"schema": pa.schema([("value", pa.float64()), ("batch", pa.int32())])},
    {"schema_version": 2},
    {"operation": "other-operation"},
    {"operation_version": "2"},
    {"input_identities": {"source": {"path": "/other"}}},
    {"parameters": {"enabled": False}},
])
def test_table_dataset_rejects_each_identity_mismatch(tmp_path, changes):
    path = tmp_path / "result"
    _dataset(path)
    with pytest.raises(ValueError, match="incompatible definition"):
        _dataset(path, **changes)


def test_table_dataset_requires_canonical_json_metadata(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        _dataset(tmp_path / "nan", parameters={"threshold": float("nan")})
    with pytest.raises(TypeError, match="unsupported value"):
        _dataset(tmp_path / "set", input_identities={"labels": {1, 2}})


def test_compatible_fresh_call_recovers_sidecar_without_computing(tmp_path):
    path = tmp_path / "result"
    result = get_runner("local").map_batches(
        _write_values, 2, batch_size=2, result_sink=_dataset(path),
    )
    sidecar = path / "completions" / "part-000000000000.json"
    sidecar.unlink()
    assert result.complete is False
    with pytest.raises(ValueError, match="metadata is missing"):
        TableDataset.open(path)

    def must_not_run(batch, writer):
        raise AssertionError(f"batch {batch.batch_id} was recomputed")

    recovered = get_runner("local").map_batches(
        must_not_run, 2, batch_size=2, result_sink=_dataset(path),
    )
    assert recovered.to_pandas().equals(result.to_pandas())
    assert sidecar.exists()


def test_failure_after_rename_is_recovered_before_local_finalization(tmp_path, monkeypatch):
    path = tmp_path / "result"
    dataset = _dataset(path)
    real_promote = dataset._promote_completion
    failed = {"once": False}

    def fail_sidecar(batch_id):
        if not failed["once"]:
            failed["once"] = True
            raise RuntimeError("stop before sidecar")
        real_promote(batch_id)

    with monkeypatch.context() as patch:
        patch.setattr(dataset, "_promote_completion", fail_sidecar)
        result = get_runner("local").map_batches(
            _write_values, 2, batch_size=2, result_sink=dataset,
        )

    assert (path / "parts" / "part-000000000000.parquet").exists()
    assert (path / "completions" / "part-000000000000.json").exists()
    assert result.to_pandas()["value"].tolist() == [0, 1]


@pytest.mark.parametrize("failure", ["no_write", "return_value", "wrong_schema"])
def test_table_sink_enforces_batch_contract(tmp_path, failure):
    dataset = _dataset(tmp_path / failure)

    def invalid(batch, writer):
        if failure == "no_write":
            return None
        if failure == "wrong_schema":
            writer.write(pa.table({"other": [1]}))
            return None
        _write_values(batch, writer)
        return "unexpected"

    with pytest.raises(RunnerError):
        get_runner("local").map_batches(
            invalid, 1, batch_size=1, result_sink=dataset,
        )


def test_validation_rejects_truncated_part(tmp_path):
    path = tmp_path / "result"
    result = get_runner("local").map_batches(
        _write_values, 2, batch_size=2, result_sink=_dataset(path),
    )
    part = next(result.iter_parts()).path
    with open(part, "r+b") as file:
        file.truncate(16)
    with pytest.raises(ValueError, match="batch 0 is invalid"):
        result.validate()


def test_validation_rejects_checksum_mismatch(tmp_path):
    path = tmp_path / "result"
    result = get_runner("local").map_batches(
        _write_values, 2, batch_size=2, result_sink=_dataset(path),
    )
    part = next(result.iter_parts()).path
    with open(part, "r+b") as file:
        file.seek(16)
        original = file.read(1)
        file.seek(16)
        file.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="batch 0 is invalid"):
        result.validate()


def test_manifest_contains_ordered_typed_part_metadata(tmp_path):
    path = tmp_path / "result"
    get_runner("local").map_batches(
        _write_values, 5, batch_size=2, result_sink=_dataset(path),
    )
    with open(path / "manifest.json") as file:
        manifest = json.load(file)
    assert manifest["complete"] is True
    assert manifest["row_count"] == 5
    assert [part["batch_id"] for part in manifest["parts"]] == [0, 1, 2]
    assert all(part["checksum"]["algorithm"] == "sha256" for part in manifest["parts"])
    assert all(part["schema_fingerprint"] for part in manifest["parts"])


def test_manifest_consolidation_does_not_read_table_rows(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    monkeypatch.setattr(
        pq, "read_table", lambda *args, **kwargs: pytest.fail("consolidation read table rows"),
    )
    result = get_runner("local").map_batches(
        _write_values, 4, batch_size=2, result_sink=_dataset(tmp_path / "result"),
    )
    assert result.row_count == 4


def test_subprocess_accepts_valid_parts_after_worker_crash(tmp_path):
    path = tmp_path / "result"
    schema = SCHEMA

    def write_then_crash(batch, writer):
        writer.write(pa.table({
            "value": pa.array(list(batch), type=pa.int64()),
            "batch": pa.array([batch.batch_id] * batch.size, type=pa.int32()),
        }, schema=schema))
        if batch.batch_id == 1:
            os._exit(1)

    result = get_runner(
        "subprocess", RunnerConfig(tmp_root=str(tmp_path)),
    ).map_batches(
        write_then_crash, 2, batch_size=1, num_workers=2, result_sink=_dataset(path),
    )
    assert result.to_pandas()["value"].tolist() == [0, 1]


def test_subprocess_rejects_non_none_sink_result(tmp_path):
    schema = SCHEMA

    def invalid(batch, writer):
        writer.write(pa.table({
            "value": pa.array(list(batch), type=pa.int64()),
            "batch": pa.array([batch.batch_id] * batch.size, type=pa.int32()),
        }, schema=schema))
        return "unexpected"

    with pytest.raises(RunnerError):
        get_runner(
            "subprocess", RunnerConfig(tmp_root=str(tmp_path)),
        ).map_batches(
            invalid, 1, batch_size=1, result_sink=_dataset(tmp_path / "invalid"),
        )
    assert not (tmp_path / "invalid" / "parts" / "part-000000000000.parquet").exists()


def test_subprocess_resume_rewinds_invalid_output_and_skips_later_valid_part(tmp_path):
    path = tmp_path / "result"
    fail_once = tmp_path / "fail-once"
    batch_one_called = tmp_path / "batch-one-called"
    schema = SCHEMA

    def process(batch, writer):
        if batch.batch_id == 1:
            if batch_one_called.exists():
                raise RuntimeError("valid batch one was recomputed")
            batch_one_called.write_text("called")
        if batch.batch_id == 2 and not fail_once.exists():
            fail_once.write_text("failed")
            raise RuntimeError("fail once")
        writer.write(pa.table({
            "value": pa.array(list(batch), type=pa.int64()),
            "batch": pa.array([batch.batch_id] * batch.size, type=pa.int32()),
        }, schema=schema))

    runner = get_runner(
        "subprocess", RunnerConfig(tmp_root=str(tmp_path)),
    )
    with pytest.raises(RunnerError) as exc_info:
        runner.map_batches(
            process, 4, batch_size=1, num_workers=2, result_sink=_dataset(path),
        )
    error = exc_info.value
    assert {batch.batch_id for batch in error.failed_batches} == {2, 3}
    assert os.listdir(os.path.join(error.tmp_folder, "results")) == []

    part_zero = path / "parts" / "part-000000000000.parquet"
    part_zero.write_bytes(b"corrupt")
    captured = {}

    def inspect(tmp):
        captured["results"] = os.listdir(os.path.join(tmp, "results"))

    result = runner.map_batches(
        process, resume_from=error.tmp_folder, pre_cleanup=inspect,
    )
    assert result.to_pandas()["value"].tolist() == [0, 1, 2, 3]
    assert captured["results"] == []
