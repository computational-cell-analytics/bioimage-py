"""Typed, partitioned table datasets for batch runner outputs."""
from __future__ import annotations

import base64
import bisect
import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

from .runner._work import Batch, BatchPlan, BoundaryBatchPlan, RegularBatchPlan


DATASET_FORMAT_VERSION = 2
DEFAULT_ROW_GROUP_ROWS = 65_536
_DATASET_FILE = "dataset.json"
_MANIFEST_FILE = "manifest.json"
_PARTS_FOLDER = "parts"
_COMPLETIONS_FOLDER = "completions"
_INPUT_LAYOUT_VERSION = 1


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
    partition_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ParquetInputDescriptor:
    """Compact worker description of an ordered Parquet table."""

    kind: str
    path: str
    identity: str
    row_count: int
    schema_fingerprint: str
    columns: Tuple[str, ...]
    layout_path: Optional[str] = None
    layout_fingerprint: Optional[str] = None


@dataclass(frozen=True)
class _ParquetInputPart:
    path: str
    row_count: int
    row_group_counts: Tuple[int, ...]


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
        self._buffer: List[pa.Table] = []
        self._buffer_rows = 0
        self._row_count = 0
        self._parquet: Optional[pq.ParquetWriter] = None
        self._tmp_path: Optional[str] = None
        self._appended = False
        self._sealed = False
        self._committed = False
        self._part_promoted = False

    def write(self, table: Any) -> None:
        """Write one Arrow table for this batch."""
        if self._appended:
            raise ValueError(f"Batch {self._batch.batch_id} already wrote its table part.")
        self.append(table)
        self._sealed = True
        self._commit()

    def append(self, table: Any) -> None:
        """Append one Arrow table to this batch's durable part."""
        if self._sealed:
            raise ValueError(f"Batch {self._batch.batch_id} already sealed its table part.")
        if not isinstance(table, pa.Table):
            raise TypeError("TablePartWriter.append() requires a pyarrow.Table.")
        expected_schema = self._dataset.schema
        if not table.schema.equals(expected_schema, check_metadata=True):
            raise ValueError(
                f"Batch {self._batch.batch_id} returned schema {table.schema}, expected "
                f"{expected_schema}."
            )
        self._open()
        self._appended = True
        self._row_count += int(table.num_rows)
        if table.num_rows:
            self._buffer.append(table)
            self._buffer_rows += int(table.num_rows)
            self._flush_complete_groups()

    def _open(self) -> None:
        if self._parquet is not None:
            return
        parts_folder = self._dataset._parts_folder
        os.makedirs(parts_folder, exist_ok=True)
        temporary_prefix = f".{_part_stem(self._batch.batch_id)}."
        for name in os.listdir(parts_folder):
            if name.startswith(temporary_prefix) and name.endswith(".tmp"):
                try:
                    os.unlink(os.path.join(parts_folder, name))
                except FileNotFoundError:
                    pass
        fd, self._tmp_path = tempfile.mkstemp(
            prefix=temporary_prefix, suffix=".tmp",
            dir=parts_folder,
        )
        os.close(fd)
        self._parquet = pq.ParquetWriter(self._tmp_path, self._dataset.schema)

    def _flush_complete_groups(self) -> None:
        row_group_rows = self._dataset.row_group_rows
        if self._buffer_rows < row_group_rows:
            return
        assert self._parquet is not None
        buffered = (
            self._buffer[0] if len(self._buffer) == 1 else pa.concat_tables(self._buffer)
        )
        offset = 0
        while buffered.num_rows - offset >= row_group_rows:
            self._parquet.write_table(buffered.slice(offset, row_group_rows))
            offset += row_group_rows
        remainder = buffered.slice(offset)
        self._buffer = [remainder] if remainder.num_rows else []
        self._buffer_rows = int(remainder.num_rows)

    def _flush_remainder(self) -> None:
        if not self._buffer_rows:
            return
        assert self._parquet is not None
        buffered = (
            self._buffer[0] if len(self._buffer) == 1 else pa.concat_tables(self._buffer)
        )
        self._parquet.write_table(buffered)
        self._buffer = []
        self._buffer_rows = 0

    def _commit(self) -> None:
        if self._committed:
            return
        assert self._parquet is not None and self._tmp_path is not None
        expected_schema = self._dataset.schema
        tmp_path = self._tmp_path
        self._flush_remainder()
        self._parquet.close()
        self._parquet = None

        try:
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
            if int(parquet.metadata.num_rows) != self._row_count:
                raise ValueError(
                    f"The Parquet row count for batch {self._batch.batch_id} changed during write."
                )

            record = self._dataset._make_part_record(
                self._batch,
                row_count=self._row_count,
                file_size=file_size,
                checksum=checksum,
            )
            _atomic_write_json(
                self._dataset._pending_completion_path(self._batch.batch_id), record,
            )
            final_path = self._dataset._part_path(self._batch.batch_id)
            os.replace(tmp_path, final_path)
            self._part_promoted = True
            self._tmp_path = None
            _sync_directory(self._dataset._parts_folder)
            self._dataset._promote_completion(self._batch.batch_id)
            self._committed = True
        except BaseException:
            if not self._part_promoted:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
            raise

    def _finish(self, result: Any) -> None:
        if result is not None:
            raise ValueError("A table sink batch function must return None.")
        if not self._appended:
            raise ValueError(
                f"Batch {self._batch.batch_id} did not write or append a table part."
            )
        self._commit()

    def _discard(self) -> None:
        if self._parquet is not None:
            self._parquet.close()
            self._parquet = None
        if self._tmp_path is not None:
            try:
                os.unlink(self._tmp_path)
            except FileNotFoundError:
                pass
            self._tmp_path = None
        if self._part_promoted and not self._committed:
            return
        if not self._committed:
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
        self._committed = False


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
        row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
    ) -> "TableDataset":
        """Create a dataset definition or open an identical existing definition."""
        if not isinstance(schema, pa.Schema):
            raise TypeError("TableDataset.create() requires a pyarrow.Schema.")
        if int(schema_version) < 1:
            raise ValueError("schema_version must be positive.")
        if not operation:
            raise ValueError("operation must not be empty.")
        if not operation_version:
            raise ValueError("operation_version must not be empty.")
        if isinstance(row_group_rows, bool) or not isinstance(row_group_rows, int):
            raise TypeError("row_group_rows must be an integer.")
        if row_group_rows <= 0:
            raise ValueError("row_group_rows must be positive.")

        expected = {
            "format_version": DATASET_FORMAT_VERSION,
            "schema": _schema_record(schema),
            "schema_version": int(schema_version),
            "operation": {"name": str(operation), "version": str(operation_version)},
            "input_identities": _normalize_json(input_identities, "input_identities"),
            "parameters": _normalize_json(parameters, "parameters"),
            "row_group_rows": int(row_group_rows),
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
    def row_group_rows(self) -> int:
        return int(self._dataset_record()["row_group_rows"])

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

    def _bind_batches(
        self,
        batches: BatchPlan,
        partition_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        if isinstance(batches, RegularBatchPlan):
            work_plan = {
                "kind": "regular_batches",
                "n_items": int(batches.n_items),
                "batch_size": int(batches.batch_size),
                "length": len(batches),
            }
        else:
            work_plan = {
                "kind": "boundary_batches",
                "boundaries": [int(value) for value in batches.boundaries],
                "length": len(batches),
            }
        if partition_metadata is not None:
            if len(partition_metadata) != len(batches):
                raise ValueError(
                    "partition_metadata must contain one record per table batch."
                )
            normalized = _normalize_json(
                list(partition_metadata), "partition_metadata",
            )
            for index, record in enumerate(normalized):
                if not isinstance(record, dict):
                    raise TypeError(
                        f"partition_metadata[{index}] must be a mapping."
                    )
                overlap = {"start", "stop"}.intersection(record)
                if overlap:
                    raise ValueError(
                        "partition_metadata must not redefine start or stop."
                    )
            work_plan["partition_metadata"] = normalized
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
        else:
            existing = record.get("work_plan")
            if partition_metadata is None and isinstance(existing, dict):
                existing = {
                    key: value for key, value in existing.items()
                    if key != "partition_metadata"
                }
            if existing != work_plan:
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

    def _batch_plan(self) -> BatchPlan:
        record = self._dataset_record()
        work = record.get("work_plan")
        if not isinstance(work, dict):
            raise ValueError("The table dataset has no supported batch work plan.")
        kind = work.get("kind")
        if kind == "regular_batches":
            plan: BatchPlan = RegularBatchPlan(
                int(work["n_items"]), int(work["batch_size"]),
            )
        elif kind == "boundary_batches":
            plan = BoundaryBatchPlan(tuple(int(value) for value in work["boundaries"]))
        else:
            raise ValueError("The table dataset has no supported batch work plan.")
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
        partition = {"start": int(batch.start), "stop": int(batch.stop)}
        work_plan = dataset.get("work_plan")
        if isinstance(work_plan, dict):
            metadata = work_plan.get("partition_metadata")
            if metadata is not None:
                partition.update(metadata[int(batch.batch_id)])
        return {
            "format_version": DATASET_FORMAT_VERSION,
            "dataset_identity": dataset["identity"],
            "batch_id": int(batch.batch_id),
            "partition": partition,
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
                partition_metadata={
                    key: value for key, value in expected["partition"].items()
                    if key not in ("start", "stop")
                },
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
            partition_metadata={
                key: value for key, value in expected["partition"].items()
                if key not in ("start", "stop")
            },
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
                partition_metadata={
                    key: value for key, value in partition.items()
                    if key not in ("start", "stop")
                },
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
        tables = [pq.read_table(part.path, columns=columns) for part in self.iter_parts()]
        if tables:
            table = pa.concat_tables(tables)
        else:
            schema = self.schema
            if columns is not None:
                schema = pa.schema([schema.field(name) for name in columns])
            table = pa.Table.from_batches([], schema=schema)
        return table.to_pandas()


def _schema_fingerprint(schema: Any) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _parquet_layout(path: str) -> Tuple[Any, Tuple[int, ...]]:
    try:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        counts = tuple(int(metadata.row_group(index).num_rows)
                       for index in range(metadata.num_row_groups))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Could not read Parquet metadata from {path!r}: {error}") from error
    if sum(counts) != int(metadata.num_rows):
        raise ValueError(f"Parquet row-group metadata is inconsistent in {path!r}.")
    return parquet.schema_arrow, counts


def _selected_schema(schema: Any, columns: Optional[Sequence[str]]) -> Any:
    if columns is None:
        return schema
    missing = [name for name in columns if name not in schema.names]
    if missing:
        raise ValueError(f"Parquet input is missing required columns {missing}.")
    return pa.schema([schema.field(name) for name in columns], metadata=schema.metadata)


def _table_dataset_input(
    path: str, columns: Optional[Sequence[str]],
) -> Tuple[Any, List[_ParquetInputPart], str]:
    dataset = TableDataset.open(path)
    definition = dataset._dataset_record()
    manifest = dataset._manifest_record()
    schema = dataset.schema
    parts = []
    identity_parts = []
    for record in manifest["parts"]:
        part_path = os.path.join(dataset.path, str(record["path"]))
        try:
            file_size = os.path.getsize(part_path)
        except OSError as error:
            raise ValueError(f"Could not read table part {part_path!r}.") from error
        if file_size != int(record["file_size"]):
            raise ValueError(f"Table part {part_path!r} has an incompatible file size.")
        checksum = record.get("checksum")
        if not isinstance(checksum, dict) or checksum.get("algorithm") != "sha256":
            raise ValueError(f"Table part {part_path!r} has invalid checksum metadata.")
        if _file_checksum(part_path) != checksum.get("value"):
            raise ValueError(f"Table part {part_path!r} does not match its checksum.")
        actual_schema, row_group_counts = _parquet_layout(part_path)
        if not actual_schema.equals(schema, check_metadata=True):
            raise ValueError(f"Table part {part_path!r} has an incompatible schema.")
        row_count = int(record["row_count"])
        if sum(row_group_counts) != row_count:
            raise ValueError(f"Table part {part_path!r} has an incompatible row count.")
        parts.append(_ParquetInputPart(part_path, row_count, row_group_counts))
        identity_parts.append({
            "path": str(record["path"]),
            "row_count": row_count,
            "file_size": int(record["file_size"]),
            "schema_fingerprint": str(record["schema_fingerprint"]),
            "checksum": record["checksum"],
        })
    identity = _fingerprint({
        "kind": "table_dataset",
        "dataset_identity": definition["identity"],
        "parts": identity_parts,
    })
    return _selected_schema(schema, columns), parts, identity


def _raw_parquet_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise ValueError(f"Parquet input path {path!r} does not exist.")
    files = []
    for folder, folders, names in os.walk(path):
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        for name in sorted(names):
            if not name.startswith(".") and name.lower().endswith(".parquet"):
                files.append(os.path.join(folder, name))
    if not files:
        raise ValueError(f"Parquet input directory {path!r} contains no Parquet files.")
    return sorted(files, key=lambda file_path: os.path.relpath(file_path, path))


def _raw_parquet_input(
    path: str, columns: Optional[Sequence[str]],
) -> Tuple[Any, List[_ParquetInputPart], str]:
    root = path if os.path.isdir(path) else os.path.dirname(path)
    files = _raw_parquet_files(path)
    schema = None
    parts = []
    identity_parts = []
    for file_path in files:
        file_schema, row_group_counts = _parquet_layout(file_path)
        selected_schema = _selected_schema(file_schema, columns)
        if schema is None:
            schema = selected_schema
        elif not selected_schema.equals(schema, check_metadata=False):
            raise ValueError(
                f"Raw Parquet fragment {file_path!r} has a different schema."
            )
        stat = os.stat(file_path)
        row_count = sum(row_group_counts)
        relative_path = os.path.relpath(file_path, root)
        parts.append(_ParquetInputPart(file_path, row_count, row_group_counts))
        identity_parts.append({
            "path": relative_path,
            "file_size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "row_count": row_count,
            "row_group_counts": list(row_group_counts),
            "schema_fingerprint": _schema_fingerprint(selected_schema.remove_metadata()),
        })
    assert schema is not None
    identity = _fingerprint({"kind": "raw_parquet", "parts": identity_parts})
    return schema, parts, identity


def _inspect_parquet_input(
    path: str, kind: str, columns: Optional[Sequence[str]],
) -> Tuple[Any, List[_ParquetInputPart], str]:
    if kind == "table_dataset":
        return _table_dataset_input(path, columns)
    if kind == "raw_parquet":
        return _raw_parquet_input(path, columns)
    raise ValueError(f"Unsupported Parquet input kind {kind!r}.")


def _describe_parquet_input(
    value: Any, columns: Optional[Sequence[str]] = None,
) -> Tuple[_ParquetInputDescriptor, Any, Dict[str, Any], Tuple[_ParquetInputPart, ...]]:
    """Describe a completed `TableDataset` or an ordered raw Parquet path."""
    if isinstance(value, TableDataset):
        path = value.path
        kind = "table_dataset"
    elif isinstance(value, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(value))
        if os.path.isdir(path) and os.path.exists(os.path.join(path, _DATASET_FILE)):
            kind = "table_dataset"
        else:
            kind = "raw_parquet"
    else:
        raise TypeError(
            "File-backed table input must be a TableDataset or a Parquet file or directory path."
        )

    schema, parts, identity = _inspect_parquet_input(path, kind, columns)
    row_count = sum(part.row_count for part in parts)
    schema_fingerprint = _schema_fingerprint(
        schema if kind == "table_dataset" else schema.remove_metadata()
    )
    descriptor = _ParquetInputDescriptor(
        kind=kind,
        path=path,
        identity=identity,
        row_count=row_count,
        schema_fingerprint=schema_fingerprint,
        columns=tuple(columns or ()),
    )
    input_identity = {
        "kind": kind,
        "path": path,
        "identity": identity,
        "row_count": row_count,
        "schema_fingerprint": schema_fingerprint,
    }
    return descriptor, schema, input_identity, tuple(parts)


def _persist_parquet_input_layout(
    descriptor: _ParquetInputDescriptor,
    schema: Any,
    parts: Sequence[_ParquetInputPart],
    dataset_path: str,
) -> _ParquetInputDescriptor:
    """Persist one verified input layout and return its worker descriptor."""
    record = {
        "format_version": _INPUT_LAYOUT_VERSION,
        "kind": descriptor.kind,
        "path": descriptor.path,
        "identity": descriptor.identity,
        "row_count": descriptor.row_count,
        "schema": _schema_record(schema),
        "schema_fingerprint": descriptor.schema_fingerprint,
        "columns": list(descriptor.columns),
        "parts": [
            {
                "path": part.path,
                "row_count": part.row_count,
                "row_group_counts": list(part.row_group_counts),
            }
            for part in parts
        ],
    }
    layout_fingerprint = _fingerprint(record)
    record["layout_fingerprint"] = layout_fingerprint
    layout_path = os.path.join(dataset_path, "inputs", "base-table-layout.json")
    if os.path.exists(layout_path):
        existing = _read_json(layout_path)
        if existing != record:
            raise ValueError(
                f"Parquet input layout {layout_path!r} does not match the planned input."
            )
    else:
        _atomic_write_json(layout_path, record)
    return replace(
        descriptor,
        layout_path=layout_path,
        layout_fingerprint=layout_fingerprint,
    )


def _load_parquet_input_layout(
    descriptor: _ParquetInputDescriptor,
) -> Tuple[Any, Tuple[_ParquetInputPart, ...]]:
    if descriptor.layout_path is None or descriptor.layout_fingerprint is None:
        raise ValueError("The Parquet input descriptor has no persisted layout.")
    record = _read_json(descriptor.layout_path)
    content = {
        key: value for key, value in record.items() if key != "layout_fingerprint"
    }
    fingerprint = _fingerprint(content)
    if (fingerprint != descriptor.layout_fingerprint
            or record.get("layout_fingerprint") != descriptor.layout_fingerprint):
        raise ValueError(f"Parquet input layout {descriptor.layout_path!r} is invalid.")
    expected = {
        "format_version": _INPUT_LAYOUT_VERSION,
        "kind": descriptor.kind,
        "path": descriptor.path,
        "identity": descriptor.identity,
        "row_count": descriptor.row_count,
        "schema_fingerprint": descriptor.schema_fingerprint,
        "columns": list(descriptor.columns),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Parquet input layout {descriptor.layout_path!r} does not match its descriptor."
        )
    schema_record = record.get("schema")
    if not isinstance(schema_record, dict):
        raise ValueError(f"Parquet input layout {descriptor.layout_path!r} has no schema.")
    schema = _schema_from_record(schema_record)
    schema_fingerprint = _schema_fingerprint(
        schema if descriptor.kind == "table_dataset" else schema.remove_metadata()
    )
    if schema_fingerprint != descriptor.schema_fingerprint:
        raise ValueError(f"Parquet input layout {descriptor.layout_path!r} has the wrong schema.")
    raw_parts = record.get("parts")
    if not isinstance(raw_parts, list):
        raise ValueError(f"Parquet input layout {descriptor.layout_path!r} has no parts.")
    try:
        parts = tuple(
            _ParquetInputPart(
                path=str(part["path"]),
                row_count=int(part["row_count"]),
                row_group_counts=tuple(int(value) for value in part["row_group_counts"]),
            )
            for part in raw_parts
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Parquet input layout {descriptor.layout_path!r} contains invalid parts."
        ) from error
    if any(
        part.row_count < 0
        or any(row_group_count < 0 for row_group_count in part.row_group_counts)
        or sum(part.row_group_counts) != part.row_count
        for part in parts
    ):
        raise ValueError(f"Parquet input layout {descriptor.layout_path!r} has invalid row counts.")
    if sum(part.row_count for part in parts) != descriptor.row_count:
        raise ValueError(f"Parquet input layout {descriptor.layout_path!r} has the wrong row count.")
    return schema, parts


class _ParquetRowReader:
    def __init__(self, descriptor: _ParquetInputDescriptor):
        if descriptor.layout_path is None:
            schema, raw_parts, identity = _inspect_parquet_input(
                descriptor.path, descriptor.kind, descriptor.columns or None,
            )
            parts = tuple(raw_parts)
            schema_fingerprint = _schema_fingerprint(
                schema if descriptor.kind == "table_dataset" else schema.remove_metadata()
            )
            row_count = sum(part.row_count for part in parts)
            if (identity != descriptor.identity or row_count != descriptor.row_count
                    or schema_fingerprint != descriptor.schema_fingerprint):
                raise ValueError(
                    f"Parquet input {descriptor.path!r} changed after planning."
                )
        else:
            schema, parts = _load_parquet_input_layout(descriptor)
        self.schema = schema
        self.parts = parts
        total = 0
        part_ends = []
        for part in parts:
            total += part.row_count
            part_ends.append(total)
        self.part_ends = tuple(part_ends)

    def _validate_range(self, start: int, stop: int, columns: Sequence[str]):
        start = int(start)
        stop = int(stop)
        row_count = self.part_ends[-1] if self.part_ends else 0
        if start < 0 or stop < start or stop > row_count:
            raise IndexError(f"Parquet row range [{start}, {stop}) is out of bounds.")
        missing = [name for name in columns if name not in self.schema.names]
        if missing:
            raise ValueError(f"Parquet input is missing required columns {missing}.")
        selected_schema = pa.schema(
            [self.schema.field(name) for name in columns], metadata=self.schema.metadata,
        )
        return start, stop, selected_schema

    def iter_rows(
        self,
        start: int,
        stop: int,
        columns: Sequence[str],
        batch_size: int,
    ) -> Iterator[pa.RecordBatch]:
        start, stop, selected_schema = self._validate_range(start, stop, columns)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if start == stop:
            return

        returned_rows = 0
        part_index = bisect.bisect_right(self.part_ends, start)
        while part_index < len(self.parts):
            part = self.parts[part_index]
            part_start = 0 if part_index == 0 else self.part_ends[part_index - 1]
            part_stop = self.part_ends[part_index]
            if part_start >= stop:
                break
            local_start = max(start, part_start) - part_start
            local_stop = min(stop, part_stop) - part_start
            parquet = pq.ParquetFile(part.path)
            row_group_start = 0
            for row_group, row_group_count in enumerate(part.row_group_counts):
                row_group_stop = row_group_start + row_group_count
                selected_start = max(local_start, row_group_start)
                selected_stop = min(local_stop, row_group_stop)
                if selected_start < selected_stop:
                    batch_start = row_group_start
                    for batch in parquet.iter_batches(
                        batch_size=batch_size, row_groups=[row_group], columns=list(columns),
                    ):
                        batch_stop = batch_start + batch.num_rows
                        take_start = max(selected_start, batch_start)
                        take_stop = min(selected_stop, batch_stop)
                        if take_start < take_stop:
                            piece = batch.slice(
                                take_start - batch_start, take_stop - take_start,
                            )
                            output = pa.RecordBatch.from_arrays(
                                list(piece.columns), schema=selected_schema,
                            )
                            returned_rows += output.num_rows
                            yield output
                        batch_start = batch_stop
                row_group_start = row_group_stop
            part_index += 1

        if returned_rows != stop - start:
            raise ValueError(
                f"Parquet row range [{start}, {stop}) returned {returned_rows} rows."
            )

    def read_rows(self, start: int, stop: int, columns: Sequence[str]):
        start, stop, selected_schema = self._validate_range(start, stop, columns)
        batches = list(self.iter_rows(start, stop, columns, DEFAULT_ROW_GROUP_ROWS))
        result = pa.Table.from_batches(batches, schema=selected_schema)
        return result


_PARQUET_READER_CACHE: Dict[_ParquetInputDescriptor, _ParquetRowReader] = {}
_PARQUET_READER_LOCK = threading.Lock()


def _read_parquet_rows(
    descriptor: _ParquetInputDescriptor,
    start: int,
    stop: int,
    columns: Sequence[str],
):
    with _PARQUET_READER_LOCK:
        reader = _PARQUET_READER_CACHE.get(descriptor)
        if reader is None:
            reader = _ParquetRowReader(descriptor)
            _PARQUET_READER_CACHE[descriptor] = reader
    return reader.read_rows(start, stop, columns)


def _iter_parquet_rows(
    descriptor: _ParquetInputDescriptor,
    start: int,
    stop: int,
    columns: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield one row range without decoding a row group for each compute chunk."""
    with _PARQUET_READER_LOCK:
        reader = _PARQUET_READER_CACHE.get(descriptor)
        if reader is None:
            reader = _ParquetRowReader(descriptor)
            _PARQUET_READER_CACHE[descriptor] = reader
    yield from reader.iter_rows(start, stop, columns, batch_size)


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
