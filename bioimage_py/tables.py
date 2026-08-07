"""Typed, partitioned table datasets for batch runner outputs."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Union

from .runner._work import Batch, RegularBatchPlan


DATASET_FORMAT_VERSION = 1
_DATASET_FILE = "dataset.json"
_MANIFEST_FILE = "manifest.json"
_PARTS_FOLDER = "parts"
_COMPLETIONS_FOLDER = "completions"


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Table datasets require pyarrow. Install bioimage-py with the 'table' extra."
        ) from error
    return pa, pq


def _atomic_write_json(path: str, value: Any) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                                    dir=folder)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
        _sync_directory(folder)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _sync_directory(path: str) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path) as file:
            value = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Table dataset metadata is missing: {path!r}.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read table dataset metadata {path!r}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Table dataset metadata {path!r} must contain a JSON object.")
    return value


def _normalize_json(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, list):
        return [_normalize_json(item, f"{path}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key {key!r}.")
            normalized[key] = _normalize_json(item, f"{path}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    raise TypeError(f"{path} contains unsupported value {value!r}.")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_checksum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _schema_record(schema: Any) -> Dict[str, str]:
    serialized = schema.serialize().to_pybytes()
    return {
        "encoding": "arrow-ipc-base64",
        "data": base64.b64encode(serialized).decode("ascii"),
        "fingerprint": hashlib.sha256(serialized).hexdigest(),
    }


def _schema_from_record(record: Mapping[str, Any]):
    pa, _ = _pyarrow()
    if record.get("encoding") != "arrow-ipc-base64":
        raise ValueError(f"Unsupported table schema encoding {record.get('encoding')!r}.")
    try:
        serialized = base64.b64decode(str(record["data"]), validate=True)
    except (KeyError, ValueError) as error:
        raise ValueError("The table dataset contains an invalid serialized schema.") from error
    expected = hashlib.sha256(serialized).hexdigest()
    if record.get("fingerprint") != expected:
        raise ValueError("The serialized table schema does not match its fingerprint.")
    return pa.ipc.read_schema(pa.BufferReader(serialized))


def _part_stem(batch_id: int) -> str:
    return f"part-{int(batch_id):012d}"


@dataclass(frozen=True)
class TablePartMetadata:
    """Metadata for one completed table part."""

    batch_id: int
    start: int
    stop: int
    path: str
    row_count: int
    file_size: int
    schema_fingerprint: str
    checksum: str


class TablePartsError(ValueError):
    """Raised when finalization finds missing or invalid table parts."""

    def __init__(self, failed_batches: Sequence[Batch]):
        self.failed_batches = tuple(failed_batches)
        super().__init__(
            f"{len(self.failed_batches)} table part(s) are missing or invalid."
        )


class TablePartWriter:
    """Write the table part for one runner batch."""

    def __init__(self, dataset: "TableDataset", batch: Batch):
        self._dataset = dataset
        self._batch = batch
        self._written = False

    def write(self, table: Any) -> None:
        """Write one Arrow table for this batch."""
        pa, pq = _pyarrow()
        if self._written:
            raise ValueError(f"Batch {self._batch.batch_id} already wrote its table part.")
        if not isinstance(table, pa.Table):
            raise TypeError("TablePartWriter.write() requires a pyarrow.Table.")
        expected_schema = self._dataset.schema
        if not table.schema.equals(expected_schema, check_metadata=True):
            raise ValueError(
                f"Batch {self._batch.batch_id} returned schema {table.schema}, expected "
                f"{expected_schema}."
            )

        parts_folder = self._dataset._parts_folder
        os.makedirs(parts_folder, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{_part_stem(self._batch.batch_id)}.", suffix=".tmp", dir=parts_folder,
        )
        os.close(fd)
        try:
            pq.write_table(table, tmp_path)
            with open(tmp_path, "rb") as file:
                os.fsync(file.fileno())
            file_size = os.path.getsize(tmp_path)
            checksum = _file_checksum(tmp_path)
            parquet = pq.ParquetFile(tmp_path)
            actual_schema = parquet.schema_arrow
            if not actual_schema.equals(expected_schema, check_metadata=True):
                raise ValueError(
                    f"The Parquet schema for batch {self._batch.batch_id} changed during write."
                )
            if int(parquet.metadata.num_rows) != int(table.num_rows):
                raise ValueError(
                    f"The Parquet row count for batch {self._batch.batch_id} changed during write."
                )

            record = self._dataset._make_part_record(
                self._batch,
                row_count=int(table.num_rows),
                file_size=file_size,
                checksum=checksum,
            )
            _atomic_write_json(
                self._dataset._pending_completion_path(self._batch.batch_id), record,
            )
            final_path = self._dataset._part_path(self._batch.batch_id)
            os.replace(tmp_path, final_path)
            _sync_directory(parts_folder)
            self._dataset._promote_completion(self._batch.batch_id)
            self._written = True
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _finish(self, result: Any) -> None:
        if result is not None:
            raise ValueError("A table sink batch function must return None.")
        if not self._written:
            raise ValueError(
                f"Batch {self._batch.batch_id} did not call TablePartWriter.write()."
            )

    def _discard(self) -> None:
        if not self._written:
            return
        for path in (
            self._dataset._pending_completion_path(self._batch.batch_id),
            self._dataset._completion_path(self._batch.batch_id),
            self._dataset._part_path(self._batch.batch_id),
        ):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        _sync_directory(self._dataset._parts_folder)
        _sync_directory(os.path.join(self._dataset.path, _COMPLETIONS_FOLDER))
        self._written = False


class TableDataset:
    """A lightweight handle for an ordered Parquet table dataset."""

    def __init__(self, path: Union[str, os.PathLike[str]], *, expected_identity: Optional[str] = None):
        self._path = os.path.abspath(os.fspath(path))
        self._expected_identity = expected_identity

    @classmethod
    def create(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        schema: Any,
        schema_version: int,
        operation: str,
        operation_version: str,
        input_identities: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> "TableDataset":
        """Create a dataset definition or open an identical existing definition."""
        pa, _ = _pyarrow()
        if not isinstance(schema, pa.Schema):
            raise TypeError("TableDataset.create() requires a pyarrow.Schema.")
        if int(schema_version) < 1:
            raise ValueError("schema_version must be positive.")
        if not operation:
            raise ValueError("operation must not be empty.")
        if not operation_version:
            raise ValueError("operation_version must not be empty.")

        expected = {
            "format_version": DATASET_FORMAT_VERSION,
            "schema": _schema_record(schema),
            "schema_version": int(schema_version),
            "operation": {"name": str(operation), "version": str(operation_version)},
            "input_identities": _normalize_json(input_identities, "input_identities"),
            "parameters": _normalize_json(parameters, "parameters"),
            "work_plan": None,
            "identity": None,
        }
        expected["definition_fingerprint"] = _fingerprint({
            key: value for key, value in expected.items()
            if key not in ("identity", "work_plan", "definition_fingerprint")
        })

        root = os.path.abspath(os.fspath(path))
        if os.path.exists(root) and not os.path.isdir(root):
            raise ValueError(f"Table dataset path {root!r} is not a directory.")
        os.makedirs(root, exist_ok=True)
        metadata_path = os.path.join(root, _DATASET_FILE)
        if os.path.exists(metadata_path):
            actual = _read_json(metadata_path)
            if actual.get("definition_fingerprint") != expected["definition_fingerprint"]:
                raise ValueError(
                    f"Existing table dataset {root!r} has an incompatible definition."
                )
            _validate_dataset_record(actual, metadata_path)
        else:
            entries = os.listdir(root)
            if entries:
                raise ValueError(
                    f"Table dataset directory {root!r} is not empty and has no {_DATASET_FILE}."
                )
            os.makedirs(os.path.join(root, _PARTS_FOLDER), exist_ok=True)
            os.makedirs(os.path.join(root, _COMPLETIONS_FOLDER), exist_ok=True)
            _atomic_write_json(metadata_path, expected)
        return cls(root)

    @classmethod
    def open(cls, path: Union[str, os.PathLike[str]]) -> "TableDataset":
        """Open a completed table dataset without reading table rows."""
        dataset = cls(path)
        dataset._dataset_record()
        dataset._manifest_record()
        return dataset

    @classmethod
    def _from_descriptor(cls, descriptor: Mapping[str, Any]) -> "TableDataset":
        if descriptor.get("kind") != "parquet_table":
            raise ValueError(f"Unsupported table result sink {descriptor.get('kind')!r}.")
        dataset = cls(str(descriptor["path"]), expected_identity=str(descriptor["identity"]))
        dataset._dataset_record()
        return dataset

    @property
    def path(self) -> str:
        return self._path

    @property
    def schema(self):
        return _schema_from_record(self._dataset_record()["schema"])

    @property
    def schema_version(self) -> int:
        return int(self._dataset_record()["schema_version"])

    @property
    def operation(self) -> str:
        return str(self._dataset_record()["operation"]["name"])

    @property
    def operation_version(self) -> str:
        return str(self._dataset_record()["operation"]["version"])

    @property
    def row_count(self) -> int:
        return int(self._manifest_record()["row_count"])

    @property
    def part_count(self) -> int:
        return int(self._manifest_record()["part_count"])

    @property
    def complete(self) -> bool:
        try:
            self._manifest_record()
        except ValueError:
            return False
        return True

    @property
    def _parts_folder(self) -> str:
        return os.path.join(self._path, _PARTS_FOLDER)

    def _dataset_record(self) -> Dict[str, Any]:
        path = os.path.join(self._path, _DATASET_FILE)
        record = _read_json(path)
        _validate_dataset_record(record, path)
        identity = record.get("identity")
        if self._expected_identity is not None and identity != self._expected_identity:
            raise ValueError(
                f"Table dataset {self._path!r} no longer matches the runner sink identity."
            )
        return record

    def _manifest_record(self) -> Dict[str, Any]:
        record = _read_json(os.path.join(self._path, _MANIFEST_FILE))
        dataset = self._dataset_record()
        if record.get("format_version") != DATASET_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported table manifest version {record.get('format_version')!r}."
            )
        if not record.get("complete"):
            raise ValueError(f"Table dataset {self._path!r} is incomplete.")
        if record.get("dataset_identity") != dataset.get("identity"):
            raise ValueError(f"Table manifest for {self._path!r} has the wrong identity.")
        if not isinstance(record.get("parts"), list):
            raise ValueError(f"Table manifest for {self._path!r} has no valid parts list.")
        plan = self._batch_plan()
        parts = record["parts"]
        if len(parts) != len(plan) or int(record.get("part_count", -1)) != len(plan):
            raise ValueError(f"Table manifest for {self._path!r} has the wrong part count.")
        row_count = 0
        for batch, part in zip(plan, parts):
            if not isinstance(part, dict) or int(part.get("batch_id", -1)) != batch.batch_id:
                raise ValueError(f"Table manifest for {self._path!r} is not in batch order.")
            sidecar = _read_json(self._completion_path(batch.batch_id))
            if part != sidecar:
                raise ValueError(
                    f"Table manifest metadata for batch {batch.batch_id} is invalid."
                )
            row_count += int(part["row_count"])
        if int(record.get("row_count", -1)) != row_count:
            raise ValueError(f"Table manifest for {self._path!r} has the wrong row count.")
        return record

    def _bind_batches(self, batches: RegularBatchPlan) -> None:
        work_plan = {
            "kind": "regular_batches",
            "n_items": int(batches.n_items),
            "batch_size": int(batches.batch_size),
            "length": len(batches),
        }
        path = os.path.join(self._path, _DATASET_FILE)
        record = self._dataset_record()
        if record.get("work_plan") is None:
            record["work_plan"] = work_plan
            record["identity"] = _fingerprint({
                key: value for key, value in record.items()
                if key not in ("identity", "definition_fingerprint")
            })
            _atomic_write_json(path, record)
            record = self._dataset_record()
        elif record.get("work_plan") != work_plan:
            raise ValueError(
                f"Existing table dataset {self._path!r} uses a different batch work plan."
            )
        self._expected_identity = str(record["identity"])

    def _descriptor(self) -> Dict[str, str]:
        record = self._dataset_record()
        if record.get("identity") is None:
            raise ValueError("The table dataset is not bound to a batch work plan.")
        return {"kind": "parquet_table", "path": self._path,
                "identity": str(record["identity"])}

    def _batch_plan(self) -> RegularBatchPlan:
        record = self._dataset_record()
        work = record.get("work_plan")
        if not isinstance(work, dict) or work.get("kind") != "regular_batches":
            raise ValueError("The table dataset has no supported batch work plan.")
        plan = RegularBatchPlan(int(work["n_items"]), int(work["batch_size"]))
        if int(work.get("length", -1)) != len(plan):
            raise ValueError("The table dataset work plan has an invalid length.")
        return plan

    def _part_path(self, batch_id: int) -> str:
        return os.path.join(self._parts_folder, f"{_part_stem(batch_id)}.parquet")

    def _completion_path(self, batch_id: int) -> str:
        return os.path.join(
            self._path, _COMPLETIONS_FOLDER, f"{_part_stem(batch_id)}.json",
        )

    def _pending_completion_path(self, batch_id: int) -> str:
        return os.path.join(
            self._path, _COMPLETIONS_FOLDER,
            f".{_part_stem(batch_id)}.pending.json",
        )

    def _promote_completion(self, batch_id: int) -> None:
        os.replace(
            self._pending_completion_path(batch_id), self._completion_path(batch_id),
        )
        _sync_directory(os.path.join(self._path, _COMPLETIONS_FOLDER))

    def _make_part_record(self, batch: Batch, *, row_count: int, file_size: int,
                          checksum: str) -> Dict[str, Any]:
        dataset = self._dataset_record()
        return {
            "format_version": DATASET_FORMAT_VERSION,
            "dataset_identity": dataset["identity"],
            "batch_id": int(batch.batch_id),
            "partition": {"start": int(batch.start), "stop": int(batch.stop)},
            "path": os.path.join(_PARTS_FOLDER, f"{_part_stem(batch.batch_id)}.parquet"),
            "row_count": int(row_count),
            "file_size": int(file_size),
            "schema_fingerprint": dataset["schema"]["fingerprint"],
            "checksum": {"algorithm": "sha256", "value": str(checksum)},
        }

    def _validate_batch(self, batch: Batch) -> None:
        plan = self._batch_plan()
        try:
            expected = plan[int(batch.batch_id)]
        except IndexError as error:
            raise ValueError(f"Batch ID {batch.batch_id} is outside the dataset work plan.") from error
        if expected != batch:
            raise ValueError(f"Batch {batch!r} does not match dataset batch {expected!r}.")

    def _validate_part(self, batch: Batch, *, recover: bool) -> Optional[TablePartMetadata]:
        self._validate_batch(batch)
        part_path = self._part_path(batch.batch_id)
        try:
            file_size = os.path.getsize(part_path)
        except OSError:
            return None

        _, pq = _pyarrow()
        try:
            parquet = pq.ParquetFile(part_path)
            row_count = int(parquet.metadata.num_rows)
            actual_schema = parquet.schema_arrow
            checksum = _file_checksum(part_path)
        except (OSError, ValueError, TypeError):
            return None
        expected_schema = self.schema
        if not actual_schema.equals(expected_schema, check_metadata=True):
            return None

        expected = self._make_part_record(
            batch, row_count=row_count, file_size=file_size, checksum=checksum,
        )
        sidecar_path = self._completion_path(batch.batch_id)
        pending_path = self._pending_completion_path(batch.batch_id)
        sidecar_exists = os.path.exists(sidecar_path)
        try:
            actual = _read_json(sidecar_path)
        except ValueError:
            actual = None
        if actual == expected:
            if recover:
                try:
                    os.unlink(pending_path)
                except FileNotFoundError:
                    pass
            return TablePartMetadata(
                batch_id=int(batch.batch_id), start=int(batch.start), stop=int(batch.stop),
                path=part_path, row_count=row_count, file_size=file_size,
                schema_fingerprint=str(expected["schema_fingerprint"]), checksum=checksum,
            )

        pending_exists = os.path.exists(pending_path)
        try:
            pending = _read_json(pending_path)
        except ValueError:
            pending = None
        if pending == expected:
            if not recover:
                return None
            self._promote_completion(batch.batch_id)
        elif not sidecar_exists and not pending_exists:
            if not recover:
                return None
            _atomic_write_json(sidecar_path, expected)
        else:
            return None
        return TablePartMetadata(
            batch_id=int(batch.batch_id), start=int(batch.start), stop=int(batch.stop),
            path=part_path, row_count=row_count, file_size=file_size,
            schema_fingerprint=str(expected["schema_fingerprint"]), checksum=checksum,
        )

    def _run_batch(self, function: Any, batch: Batch) -> None:
        if self._validate_part(batch, recover=True) is not None:
            return
        writer = TablePartWriter(self, batch)
        try:
            result = function(batch, writer)
            writer._finish(result)
        except BaseException:
            writer._discard()
            raise

    def _finalize(self, batches: Sequence[Batch]) -> "TableDataset":
        os.makedirs(self._path, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{_MANIFEST_FILE}.", suffix=".tmp",
                                        dir=self._path)
        row_count = 0
        part_count = 0
        failed_batches = []
        try:
            with os.fdopen(fd, "w") as file:
                dataset_identity = self._dataset_record()["identity"]
                file.write("{\n")
                file.write(f'  "format_version": {DATASET_FORMAT_VERSION},\n')
                file.write(f'  "dataset_identity": {json.dumps(dataset_identity)},\n')
                file.write('  "complete": true,\n')
                file.write('  "parts": [\n')
                first = True
                for batch in batches:
                    metadata = self._validate_part(batch, recover=True)
                    if metadata is None:
                        failed_batches.append(batch)
                        continue
                    record = _read_json(self._completion_path(batch.batch_id))
                    if not first:
                        file.write(",\n")
                    file.write("    ")
                    file.write(json.dumps(record, sort_keys=True, allow_nan=False))
                    first = False
                    row_count += metadata.row_count
                    part_count += 1
                file.write("\n  ],\n")
                file.write(f'  "part_count": {part_count},\n')
                file.write(f'  "row_count": {row_count}\n')
                file.write("}\n")
                file.flush()
                os.fsync(file.fileno())
            if failed_batches:
                raise TablePartsError(failed_batches)
            os.replace(tmp_path, os.path.join(self._path, _MANIFEST_FILE))
            _sync_directory(self._path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return TableDataset.open(self._path)

    def iter_parts(self) -> Iterator[TablePartMetadata]:
        """Iterate over completed parts in dataset order."""
        manifest = self._manifest_record()
        for record in manifest["parts"]:
            partition = record["partition"]
            yield TablePartMetadata(
                batch_id=int(record["batch_id"]),
                start=int(partition["start"]),
                stop=int(partition["stop"]),
                path=self._part_path(int(record["batch_id"])),
                row_count=int(record["row_count"]),
                file_size=int(record["file_size"]),
                schema_fingerprint=str(record["schema_fingerprint"]),
                checksum=str(record["checksum"]["value"]),
            )

    def validate(self) -> "TableDataset":
        """Validate all completed parts without reading table rows."""
        plan = self._batch_plan()
        manifest = self._manifest_record()
        records = manifest["parts"]
        if len(records) != len(plan):
            raise ValueError(
                f"Table manifest contains {len(records)} parts, expected {len(plan)}."
            )
        row_count = 0
        for index, batch in enumerate(plan):
            metadata = self._validate_part(batch, recover=False)
            if metadata is None:
                raise ValueError(f"Table part for batch {batch.batch_id} is invalid.")
            if int(records[index].get("batch_id", -1)) != batch.batch_id:
                raise ValueError("Table manifest parts are not in batch order.")
            sidecar = _read_json(self._completion_path(batch.batch_id))
            if records[index] != sidecar:
                raise ValueError(
                    f"Table manifest metadata for batch {batch.batch_id} is invalid."
                )
            row_count += metadata.row_count
        if int(manifest["part_count"]) != len(plan):
            raise ValueError("Table manifest part_count is invalid.")
        if int(manifest["row_count"]) != row_count:
            raise ValueError("Table manifest row_count is invalid.")
        return self

    def to_pandas(self, columns: Optional[Sequence[str]] = None):
        """Materialize selected columns as a pandas DataFrame."""
        pa, pq = _pyarrow()
        tables = [pq.read_table(part.path, columns=columns) for part in self.iter_parts()]
        if tables:
            table = pa.concat_tables(tables)
        else:
            schema = self.schema
            if columns is not None:
                schema = pa.schema([schema.field(name) for name in columns])
            table = pa.Table.from_batches([], schema=schema)
        return table.to_pandas()


def _validate_dataset_record(record: Mapping[str, Any], path: str) -> None:
    if record.get("format_version") != DATASET_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported table dataset version {record.get('format_version')!r} in {path!r}."
        )
    schema = record.get("schema")
    if not isinstance(schema, dict):
        raise ValueError(f"Table dataset {path!r} has no valid schema.")
    _schema_from_record(schema)
    if not isinstance(record.get("operation"), dict):
        raise ValueError(f"Table dataset {path!r} has no valid operation metadata.")
    expected_definition = _fingerprint({
        key: value for key, value in record.items()
        if key not in ("identity", "work_plan", "definition_fingerprint")
    })
    if record.get("definition_fingerprint") != expected_definition:
        raise ValueError(f"Table dataset definition fingerprint is invalid in {path!r}.")
    work = record.get("work_plan")
    if work is None:
        if record.get("identity") is not None:
            raise ValueError(f"Unbound table dataset {path!r} has an identity.")
        return
    expected_identity = _fingerprint({
        key: value for key, value in record.items()
        if key not in ("identity", "definition_fingerprint")
    })
    if record.get("identity") != expected_identity:
        raise ValueError(f"Table dataset identity is invalid in {path!r}.")


__all__ = ["TableDataset", "TablePartMetadata", "TablePartWriter", "TablePartsError"]
