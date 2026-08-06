"""Persistent diagnostic records for distributed runner attempts."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import TaskFailure


MANIFEST_SCHEMA_VERSION = 1
OUTCOME_SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str, value: Any) -> None:
    """Write JSON through a temporary file in the destination directory."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                                    dir=folder)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def create_manifest(tmp: str, *, backend: str, name: str, n_tasks: int,
                    python_executable: str) -> None:
    """Create the versioned run manifest before the first launch attempt."""
    atomic_write_json(
        os.path.join(tmp, "manifest.json"),
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "backend": backend,
            "created_at": utc_now(),
            "name": name,
            "n_tasks": int(n_tasks),
            "python_executable": python_executable,
            "attempts": [],
        },
    )


def load_manifest(tmp: str, *, expected_backend: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate a versioned run manifest."""
    path = os.path.join(tmp, "manifest.json")
    try:
        with open(path) as f:
            manifest = json.load(f)
    except FileNotFoundError as error:
        raise ValueError(
            f"Run folder {tmp!r} has no versioned manifest.json and cannot be resumed."
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read run manifest {path!r}: {error}") from error

    version = manifest.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema {version!r} in {path!r}; expected "
            f"{MANIFEST_SCHEMA_VERSION}."
        )
    backend = manifest.get("backend")
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(
            f"Run folder {tmp!r} uses backend {backend!r}, not {expected_backend!r}."
        )
    if not isinstance(manifest.get("attempts"), list):
        raise ValueError(f"Run manifest {path!r} has no valid attempts list.")
    return manifest


def relative_path(tmp: str, path: Optional[str]) -> Optional[str]:
    """Return a run-folder-relative path for a diagnostic artifact."""
    if path is None:
        return None
    return os.path.relpath(path, tmp)


def absolute_path(tmp: str, path: Optional[str]) -> Optional[str]:
    """Resolve a diagnostic artifact path from the run folder."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(tmp, path)


def attempt_folder(tmp: str, attempt_number: int) -> str:
    """Return the directory for one launch attempt."""
    return os.path.join(tmp, "attempts", f"{int(attempt_number):04d}")


def append_attempt(tmp: str, *, task_ids: Sequence[int], concurrency: int) -> Dict[str, Any]:
    """Append a launch attempt and create its diagnostic directories."""
    manifest = load_manifest(tmp)
    attempts = manifest["attempts"]
    attempt_number = len(attempts) + 1
    folder = attempt_folder(tmp, attempt_number)
    for subfolder in ("errors", "logs", "tasks", "timings"):
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    attempt = {
        "attempt_number": attempt_number,
        "task_ids": [int(task_id) for task_id in task_ids],
        "concurrency": int(concurrency),
        "started_at": utc_now(),
        "finished_at": None,
        "status": "launching",
        "job_id": None,
        "script_path": None,
        "accounting_path": None,
        "outcomes_path": relative_path(tmp, os.path.join(folder, "tasks")),
        "logs_path": relative_path(tmp, os.path.join(folder, "logs")),
        "tracebacks_path": relative_path(tmp, os.path.join(folder, "errors")),
        "timings_path": relative_path(tmp, os.path.join(folder, "timings")),
        "submission_error": None,
    }
    attempts.append(attempt)
    atomic_write_json(os.path.join(tmp, "manifest.json"), manifest)
    return dict(attempt)


def update_attempt(tmp: str, attempt_number: int, **changes: Any) -> Dict[str, Any]:
    """Update the current attempt without removing prior attempts."""
    manifest = load_manifest(tmp)
    index = int(attempt_number) - 1
    try:
        attempt = manifest["attempts"][index]
    except IndexError as error:
        raise ValueError(f"Run folder {tmp!r} has no attempt {attempt_number}.") from error
    if int(attempt.get("attempt_number", -1)) != int(attempt_number):
        raise ValueError(f"Run manifest {tmp!r} has inconsistent attempt numbering.")
    attempt.update(changes)
    atomic_write_json(os.path.join(tmp, "manifest.json"), manifest)
    return dict(attempt)


def task_outcome_path(tmp: str, attempt_number: int, task_id: int) -> str:
    """Return the normalized outcome path for one task attempt."""
    return os.path.join(attempt_folder(tmp, attempt_number), "tasks", f"{int(task_id)}.json")


def write_task_outcome(tmp: str, attempt_number: int, outcome: Mapping[str, Any]) -> None:
    """Persist one normalized task outcome."""
    record = {"schema_version": OUTCOME_SCHEMA_VERSION, **dict(outcome)}
    atomic_write_json(task_outcome_path(tmp, attempt_number, int(record["task_id"])), record)


def read_task_outcome(tmp: str, attempt_number: int, task_id: int) -> Optional[Dict[str, Any]]:
    """Read one normalized task outcome if it exists and has a supported schema."""
    path = task_outcome_path(tmp, attempt_number, task_id)
    try:
        with open(path) as f:
            record = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if record.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        return None
    return record


def latest_task_outcome(tmp: str, task_id: int) -> Optional[Dict[str, Any]]:
    """Return the newest persisted outcome for a task."""
    manifest = load_manifest(tmp)
    for attempt in reversed(manifest["attempts"]):
        if int(task_id) not in {int(value) for value in attempt.get("task_ids", [])}:
            continue
        outcome = read_task_outcome(tmp, int(attempt["attempt_number"]), int(task_id))
        if outcome is not None:
            return outcome
    return None


def outcome_to_failure(tmp: str, outcome: Mapping[str, Any]) -> TaskFailure:
    """Convert a persisted outcome to the public failure record."""
    return TaskFailure(
        task_id=int(outcome["task_id"]),
        backend=str(outcome["backend"]),
        attempt_number=int(outcome["attempt_number"]),
        scheduler_task_id=outcome.get("scheduler_task_id"),
        scheduler_state=outcome.get("scheduler_state"),
        exit_code=outcome.get("exit_code"),
        signal=outcome.get("signal"),
        elapsed_seconds=outcome.get("elapsed_seconds"),
        peak_rss_bytes=outcome.get("peak_rss_bytes"),
        failed_node=outcome.get("failed_node"),
        stdout_path=absolute_path(tmp, outcome.get("stdout_path")),
        stderr_path=absolute_path(tmp, outcome.get("stderr_path")),
        traceback_path=absolute_path(tmp, outcome.get("traceback_path")),
    )
