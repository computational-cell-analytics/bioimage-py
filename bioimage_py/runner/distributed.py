"""Distributed runners: a shared protocol base, the subprocess runner, and a slurm stub.

The protocol (cloudpickled payload + generated per-task work lists + result/sentinel files)
is shared so that :class:`SubprocessRunner` (here) and the future ``SlurmRunner`` differ
only in how tasks are launched and awaited.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from concurrent import futures
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import bioimage_cpp
import cloudpickle
import numpy
from bioimage_cpp.utils import Blocking
from tqdm import tqdm

from ..sources.base import Source
from ..util import ComputeFn, group_blocks_by_shard, maybe_warn_imbalance
from ._diagnostics import (append_attempt, atomic_write_json, attempt_folder, create_manifest,
                           latest_task_outcome, load_manifest, outcome_to_failure,
                           read_task_outcome, relative_path, update_attempt, utc_now,
                           write_task_outcome)
from .base import Runner, RunnerError, TaskFailure
from .config import RunnerConfig, SlurmConfig


def _env_versions() -> Dict[str, Any]:
    """Versions the distributed payload is stamped with (a submit-vs-runtime skew guard)."""
    return {
        "python": tuple(sys.version_info[:2]),
        "numpy": numpy.__version__,
        "bioimage_cpp": bioimage_cpp.__version__,
    }


def _version_major(lib: str, version: Any) -> Optional[int]:
    """Best-effort major-version integer for ``version``; ``None`` if it cannot be parsed."""
    try:
        return int(version[0]) if lib == "python" else int(str(version).split(".", 1)[0])
    except (ValueError, IndexError, TypeError):
        return None


def _check_versions(recorded: Mapping[str, Any], *, role: str) -> None:
    """Guard against submit-vs-runtime version skew stamped into the payload.

    Compares each recorded library version against the current environment: a differing
    *major* version raises :class:`RuntimeError`, while a differing minor/patch under a
    matching major (or a version that cannot be parsed) only warns. Libraries missing from
    ``recorded`` -- an older payload -- are skipped.

    Args:
        recorded: The ``versions`` mapping stamped into the payload at submit time.
        role: Where the check runs (``"worker"`` or ``"orchestrator"``), used in messages.
    """
    current = _env_versions()
    for lib, expected in recorded.items():
        if expected is None or lib not in current:
            continue
        actual = current[lib]
        if expected == actual:
            continue
        detail = (f"{lib} version skew ({role}): payload built with {expected!r}, "
                  f"environment has {actual!r}.")
        exp_major, act_major = _version_major(lib, expected), _version_major(lib, actual)
        if exp_major is not None and act_major is not None and exp_major != act_major:
            raise RuntimeError(
                detail + " Major versions differ; the runtime environment must match the "
                "submitting environment."
            )
        warnings.warn(detail + " Major versions agree; proceeding.", stacklevel=2)


def _partition(block_ids: Sequence[int], n_tasks: int) -> List[List[int]]:
    """Split ``block_ids`` into ``n_tasks`` contiguous, near-equal groups."""
    block_ids = list(block_ids)
    n = len(block_ids)
    base, extra = divmod(n, n_tasks)
    tasks, start = [], 0
    for t in range(n_tasks):
        size = base + (1 if t < extra else 0)
        tasks.append(block_ids[start:start + size])
        start += size
    return tasks


def _pack_groups(groups: Sequence[Sequence[int]], n_tasks: int, num_workers: int,
                 name: str) -> List[List[int]]:
    """Bin-pack whole shard-groups into at most ``n_tasks`` tasks (least-loaded first).

    ``n_tasks`` is the resolved target task count (from :meth:`_DistributedRunner._resolve_n_tasks`,
    i.e. ``num_workers * tasks_per_worker`` clamped); it is further capped at ``len(groups)`` since a
    shard-group is indivisible. ``num_workers`` is passed through only for the imbalance warning
    (which compares the shard-group count against the concurrency, not the task count).
    """
    groups = [list(g) for g in groups if g]
    if not groups:
        return [[]]
    n_tasks = max(1, min(int(n_tasks), len(groups)))
    tasks: List[List[int]] = [[] for _ in range(n_tasks)]
    loads = [0] * n_tasks
    for group in sorted(groups, key=len, reverse=True):
        t = min(range(n_tasks), key=lambda i: loads[i])
        tasks[t].extend(group)
        loads[t] += len(group)
    maybe_warn_imbalance(loads, num_workers, len(groups), name)
    return tasks


def _done_blocks(tmp: str, task_ids: Sequence[int]) -> set:
    """Union of completed block ids across the given tasks' done-logs.

    Parses only newline-terminated lines, so a torn final line (a worker crashed mid-append)
    is ignored -- safe to call while a worker is still appending. This is the authoritative
    "which blocks are done" set used for precise failure reporting and resume.
    """
    done: set = set()
    for t in task_ids:
        path = os.path.join(tmp, "progress", f"{t}.log")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.endswith("\n"):
                    s = line.strip()
                    if s:
                        try:
                            done.add(int(s))
                        except ValueError:  # defensive: ignore a malformed line
                            continue
    return done


class _DoneLogCounter:
    """Incremental processed-block count across the per-task done-logs (drives the progress bar).

    Each done-log is append-only and single-writer -- the worker harness appends one flushed
    ``{block_id}\\n`` per completed block (``_harness.py``) -- so a poll only needs the bytes
    appended since the previous tick. This tracks a per-task byte offset and running newline
    count and reads only each log's new tail per :meth:`count`, making a tick ``O(new bytes)``
    instead of ``O(total bytes)``. Only complete newline-terminated lines are counted, so a torn
    final line is deferred to the next tick; use :func:`_done_blocks` for the authoritative set.

    The counter spans all ``n_tasks`` (the bar credits prior progress on a resume): a completed
    task's full log is read once on the first :meth:`count`, then contributes an empty tail read.
    Instantiate one per poll loop; it is single-threaded (no locking).
    """

    def __init__(self, tmp: str, n_tasks: int) -> None:
        self._tmp = tmp
        self._n_tasks = n_tasks
        self._offsets = [0] * n_tasks
        self._counts = [0] * n_tasks

    def count(self) -> int:
        """Return the total processed-block count, reading only newly-appended bytes."""
        for t in range(self._n_tasks):
            path = os.path.join(self._tmp, "progress", f"{t}.log")
            try:
                with open(path, "rb") as f:
                    f.seek(self._offsets[t])
                    chunk = f.read()
            except FileNotFoundError:  # task has not started writing its log yet
                continue
            last_nl = chunk.rfind(b"\n")
            if last_nl == -1:  # no new complete line; leave a partial tail unread for next tick
                continue
            self._counts[t] += chunk[:last_nl + 1].count(b"\n")
            self._offsets[t] += last_nl + 1
        return sum(self._counts)


def _total_blocks(tmp: str, n_tasks: int) -> int:
    """Total assigned-block count across all tasks (from the per-task block lists)."""
    total = 0
    for t in range(n_tasks):
        with open(os.path.join(tmp, "blocks", f"{t}.json")) as f:
            total += len(json.load(f))
    return total


class _DistributedRunner(Runner):
    """Base for runners that ship the computation to separate worker processes."""

    backend_name = "distributed"

    @staticmethod
    def _require_reopenable(inputs: Sequence[Source], outputs: Sequence[Source],
                            mask: Optional[Source]) -> None:
        """Validate that every source is file-backed (reopenable on a worker).

        Args:
            inputs: The input sources.
            outputs: The output sources.
            mask: The mask source, or ``None``.

        Raises:
            ValueError: If any source cannot be reopened (e.g. an in-memory numpy array).
        """
        roles = [("input", inputs), ("output", outputs)]
        if mask is not None:
            roles.append(("mask", [mask]))
        for role, sources in roles:
            for source in sources:
                try:
                    spec = source.to_spec()
                except ValueError as error:
                    raise ValueError(
                        f"Distributed execution requires file-backed {role} arrays (zarr/n5). {error}"
                    ) from error
                if role == "output":
                    if not source.writable:
                        raise ValueError(
                            "Distributed outputs must be writable; got a read-only output "
                            f"({spec.kind!r}). Use a writable, file-backed output (zarr/n5)."
                        )
                    if spec.kind == "file" and spec.params.get("format") == "hdf5":
                        raise ValueError(
                            "HDF5 is not safe as a distributed output (concurrent multi-process writes "
                            "to one file corrupt it). Use zarr or n5 for distributed outputs."
                        )

    def _max_tasks(self) -> int:
        """Backend cap on the number of tasks a run may be partitioned into.

        The base (subprocess) has no hard limit -- only the number of schedulable units bounds
        it -- so this returns a sentinel. :class:`SlurmRunner` overrides it with the cluster's
        ``MaxArraySize``.
        """
        return sys.maxsize

    def _resolve_n_tasks(self, num_workers: int, n_items: int) -> int:
        """Resolve the task count for ``n_items`` schedulable units.

        Over-partitions into ``num_workers * config.tasks_per_worker`` tasks so a free worker
        pulls the next queued task (load-balancing), clamped to ``n_items`` (never more tasks
        than units) and to :meth:`_max_tasks` (the backend cap). ``tasks_per_worker == 1``
        (the default) reproduces the one-task-per-worker partition.

        Args:
            num_workers: The requested worker count (also the concurrency / array throttle).
            n_items: The number of schedulable units (block ids, or shard-groups).

        Returns:
            The number of tasks to partition the work into (at least 1).
        """
        if n_items <= 0:
            return 1
        tasks_per_worker = max(1, int(self.config.tasks_per_worker))
        desired = max(1, int(num_workers)) * tasks_per_worker
        return max(1, min(desired, int(n_items), self._max_tasks()))

    def _execute(
        self,
        *,
        function: ComputeFn,
        inputs: Sequence[Source],
        outputs: Sequence[Source],
        mask: Optional[Source],
        blocking: Blocking,
        block_ids: Sequence[int],
        halo: Optional[Sequence[int]],
        has_return_val: bool,
        num_workers: int,
        name: str,
        shape: Tuple[int, ...],
        block_shape: Tuple[int, ...],
        roi: Optional[Tuple[slice, ...]],
        pre_cleanup: Optional[Callable[[str], None]] = None,
    ) -> List[Any]:
        # Validate up front that every source can be reopened on a worker (file-backed).
        # to_spec() raises here for numpy inputs, the actionable "numpy is local-only" failure.
        self._require_reopenable(inputs, outputs, mask)
        payload_extra = {
            "mode": "block",
            "input_specs": [s.to_spec() for s in inputs],
            "output_specs": [s.to_spec() for s in outputs],
            "mask_spec": mask.to_spec() if mask is not None else None,
            "shape": tuple(shape),
            "block_shape": tuple(block_shape),
            "roi": roi,
            "halo": None if halo is None else [int(h) for h in halo],
        }
        # Sharded outputs: group blocks so each shard's blocks land in one task (the worker
        # runs them sequentially), preventing concurrent same-shard writes. None => no
        # sharded output, fall back to the default contiguous partition.
        groups = group_blocks_by_shard(blocking, outputs, block_ids)
        return self._run_ids(function, block_ids, payload_extra, has_return_val,
                             num_workers, name, pre_cleanup, groups=groups)

    def _execute_map(
        self,
        *,
        function: Callable[[int], Any],
        item_ids: Sequence[int],
        has_return_val: bool,
        num_workers: int,
        name: str,
        pre_cleanup: Optional[Callable[[str], None]] = None,
    ) -> List[Any]:
        """Ship ``function(index)`` over ``item_ids`` (no sources/blocking; closure-carried data)."""
        return self._run_ids(function, item_ids, {"mode": "map"}, has_return_val,
                             num_workers, name, pre_cleanup)

    def _run_ids(
        self,
        function: Callable[..., Any],
        ids: Sequence[int],
        payload_extra: Dict[str, Any],
        has_return_val: bool,
        num_workers: int,
        name: str,
        pre_cleanup: Optional[Callable[[str], None]],
        groups: Optional[List[List[int]]] = None,
    ) -> List[Any]:
        """Shared protocol: write the payload + per-task id lists, launch, and finalize.

        Used by both the block-wise :meth:`_execute` (``payload_extra`` carries the source specs
        and blocking) and :meth:`_execute_map` (``payload_extra = {"mode": "map"}``). The
        per-task work-list directory is still ``blocks/`` regardless of mode.

        Args:
            function: The cloudpickled per-block / per-item callable.
            ids: The block ids or item indices to process.
            payload_extra: Mode-specific payload keys (must include ``"mode"``).
            has_return_val: Whether the callable returns a value to collect.
            num_workers: Number of parallel tasks.
            name: A short name for progress display.
            pre_cleanup: Optional pre-cleanup callback forwarded to :meth:`_finalize`.
            groups: Optional shard-exclusive block groups (from
                :func:`bioimage_py.util.group_blocks_by_shard`); when given, whole groups are
                bin-packed into tasks so each shard is written by a single worker. ``None``
                uses the default contiguous partition. Result order is by ``ids`` regardless.

        Returns:
            The per-id return values in ``ids`` order if ``has_return_val``, else ``None``s.
        """
        ids = [int(b) for b in ids]
        tmp = tempfile.mkdtemp(prefix="bioimage_py_", dir=self.config.tmp_root)
        for sub in ("blocks", "results", "success", "error", "timings", "progress"):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)

        versions = _env_versions()
        payload = {
            "function": function,
            "has_return_val": bool(has_return_val),
            "num_workers": int(num_workers),  # persisted so resume() can relaunch without it
            "versions": versions,
            "python": versions["python"],  # legacy alias; an older harness still finds it
            **payload_extra,
        }
        with open(os.path.join(tmp, "payload.pkl"), "wb") as f:
            cloudpickle.dump(payload, f)

        # Human-readable debug artifact (never used for correctness).
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = f"# source unavailable for {getattr(function, '__name__', function)!r}\n"
        with open(os.path.join(tmp, "source.py"), "w") as f:
            f.write(source)

        if groups is None:
            n_tasks = self._resolve_n_tasks(num_workers, len(ids))
            tasks = _partition(ids, n_tasks)
        else:
            tasks = _pack_groups(groups, self._resolve_n_tasks(num_workers, len(groups)),
                                 num_workers, name)
            n_tasks = len(tasks)
        for task_id, task_ids in enumerate(tasks):
            with open(os.path.join(tmp, "blocks", f"{task_id}.json"), "w") as f:
                json.dump([int(b) for b in task_ids], f)

        create_manifest(
            tmp,
            backend=self.backend_name,
            name=name,
            n_tasks=n_tasks,
            python_executable=self.config.python_executable or sys.executable,
        )

        self._launch_and_wait(tmp, n_tasks, num_workers, name)
        return self._finalize(tmp, n_tasks, tasks, ids, has_return_val, name,
                              pre_cleanup=pre_cleanup)

    def _finalize(
        self,
        tmp: str,
        n_tasks: int,
        tasks: Sequence[Sequence[int]],
        block_ids: Sequence[int],
        has_return_val: bool,
        name: str,
        pre_cleanup: Optional[Callable[[str], None]] = None,
    ) -> List[Any]:
        """Check the per-task sentinels, then collect results or raise on failure.

        Shared by :meth:`_execute` and :meth:`SlurmRunner.reattach` so a detached run is
        finalized identically to an in-process one.

        Args:
            tmp: The job temp folder.
            n_tasks: The number of tasks the run was partitioned into.
            tasks: The per-task block-id lists (``tasks[task_id]``), used to map a failed
                task back to its block ids.
            block_ids: The full ordered block-id list (used to order collected results).
            has_return_val: Whether per-block return values were collected.
            name: A short name for the failure message.
            pre_cleanup: Optional ``pre_cleanup(tmp)`` callback invoked right before the temp
                folder is removed on the success path (best-effort; its failure is reported
                but does not abort cleanup or the run).

        Returns:
            The per-block return values in ``block_ids`` order if ``has_return_val``, else
            a list of ``None`` of the same length.

        Raises:
            RunnerError: If any block is incomplete. The error contains the preserved temp
                folder, failed block ids, and task diagnostics.
        """
        # Per-block done-logs are the authority for what completed, so failure reporting is
        # precise: only blocks not in any done-log are failed (not the whole task). A task that
        # finished all its blocks but died before writing its sentinel thus contributes nothing.
        done = _done_blocks(tmp, range(n_tasks))
        failed_block_ids = sorted(int(b) for b in block_ids if int(b) not in done)
        if failed_block_ids:
            failed_tasks = [t for t in range(n_tasks)
                            if not os.path.exists(os.path.join(tmp, "success", f"{t}.success"))]
            task_failures = self._task_failures(tmp, failed_tasks)
            raise RunnerError(self._failure_message(tmp, task_failures, name),
                              failed_block_ids=failed_block_ids, tmp_folder=tmp,
                              task_failures=task_failures)

        results = (self._collect(tmp, n_tasks, tasks, block_ids)
                   if has_return_val else [None] * len(block_ids))
        if pre_cleanup is not None:
            try:
                pre_cleanup(tmp)
            except Exception as err:  # noqa: BLE001 - best-effort: never fail the run on this
                print(f"pre_cleanup callback failed for {tmp}: {err!r}")
        shutil.rmtree(tmp, ignore_errors=True)
        return results

    def _collect(self, tmp: str, n_tasks: int, tasks: Sequence[Sequence[int]],
                 block_ids: Sequence[int]) -> List[Any]:
        """Load and order per-task results, reading length-framed per-block records.

        Each ``results/<task_id>`` file is a sequence of ``<8-byte little-endian length>
        <cloudpickled (bid, res)>`` records appended one per completed block (possibly across
        an original run and a resume). Reading stops at the first short/torn record (only the
        final record of a crashed write can be torn, since writes are flushed per record).
        Results are deduped by block id (last-wins) and ordered by ``block_ids``.
        """
        result_by_block: Dict[int, Any] = {}
        for task_id in range(n_tasks):
            path = os.path.join(tmp, "results", f"{task_id}")
            if not os.path.exists(path):  # a never-started task has no result file
                continue
            with open(path, "rb") as f:
                while True:
                    header = f.read(8)
                    if len(header) < 8:
                        break
                    (length,) = struct.unpack("<Q", header)
                    payload = f.read(length)
                    if len(payload) < length:  # torn final record
                        break
                    bid, res = cloudpickle.loads(payload)
                    result_by_block[int(bid)] = res
        missing = [int(b) for b in block_ids if int(b) not in result_by_block]
        if missing:
            missing_set = set(missing)
            failed_tasks = [task_id for task_id, task in enumerate(tasks)
                            if missing_set.intersection(int(value) for value in task)]
            task_failures = self._task_failures(tmp, failed_tasks)
            raise RunnerError(
                f"Result records missing for {len(missing)} block(s) after a successful run "
                f"(first: {missing[:5]}). Temp folder: {tmp}.",
                failed_block_ids=missing, tmp_folder=tmp, task_failures=task_failures)
        return [result_by_block[int(b)] for b in block_ids]

    def _task_failures(self, tmp: str, failed_tasks: Sequence[int]) -> List[TaskFailure]:
        """Load the newest persisted failure record for each unresolved task."""
        manifest = load_manifest(tmp, expected_backend=self.backend_name)
        attempts = manifest["attempts"]
        fallback_attempt = int(attempts[-1]["attempt_number"]) if attempts else 1
        failures: List[TaskFailure] = []
        for task_id in failed_tasks:
            outcome = latest_task_outcome(tmp, int(task_id))
            if outcome is None:
                traceback_path = os.path.join(tmp, "error", f"{int(task_id)}.txt")
                failures.append(TaskFailure(
                    task_id=int(task_id), backend=self.backend_name,
                    attempt_number=fallback_attempt,
                    traceback_path=traceback_path if os.path.exists(traceback_path) else None,
                ))
            else:
                failures.append(outcome_to_failure(tmp, outcome))
        return failures

    @staticmethod
    def _failure_message(tmp: str, task_failures: Sequence[TaskFailure], name: str) -> str:
        """Build an error message from persisted task diagnostics."""
        first = None
        err_files = [failure.traceback_path for failure in task_failures]
        err_files = [p for p in err_files if p is not None and os.path.exists(p)]
        if err_files:
            with open(err_files[0]) as f:
                lines = f.read().strip().splitlines()
            first = lines[-1] if lines else None
        if first is None and task_failures:
            failure = task_failures[0]
            details = []
            if failure.scheduler_state is not None:
                details.append(f"state={failure.scheduler_state}")
            if failure.exit_code is not None:
                details.append(f"exit_code={failure.exit_code}")
            if failure.signal is not None:
                details.append(f"signal={failure.signal}")
            first = ", ".join(details) or None
        n = len(task_failures) if task_failures else "some"
        return (
            f"{n} task(s) failed in '{name or 'run'}'. "
            f"Temp folder preserved for debugging: {tmp}. First error: {first!r}"
        )

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str,
                         task_ids: Optional[Sequence[int]] = None) -> None:
        """Launch the worker tasks and block until they have all finished.

        ``task_ids`` restricts the launch to a subset of task indices (used by :meth:`resume`
        to relaunch only the incomplete tasks); ``None`` launches all ``0 .. n_tasks - 1``.
        """
        raise NotImplementedError

    def _resume_entry(self, tmp_folder: str, *, name: str,
                      pre_cleanup: Optional[Callable[[str], None]]) -> Optional[list]:
        """Distributed override of :meth:`Runner._resume_entry`: resume from the temp folder."""
        return self.resume(tmp_folder, name=name or "resume", pre_cleanup=pre_cleanup)

    def resume(self, tmp_folder: str, *, name: str = "resume", num_workers: Optional[int] = None,
               pre_cleanup: Optional[Callable[[str], None]] = None) -> Optional[list]:
        """Resume a previously-failed run from its preserved temp folder.

        Reconstructs the original partition and payload, relaunches only the tasks that still
        have un-done blocks (the worker harness skips blocks already in its done-log), then
        finalizes over **all** persisted per-block results -- so a return-value op's reduction
        runs over the previously-completed blocks merged with the freshly re-run ones.

        Args:
            tmp_folder: The preserved temp folder (``RunnerError.tmp_folder``).
            name: A short name for the progress display.
            num_workers: Override for the worker count; defaults to the run's persisted value.
            pre_cleanup: Optional ``pre_cleanup(tmp)`` callback forwarded to :meth:`_finalize`.

        Returns:
            The per-block return values (if the run collected any), else ``None``.
        """
        manifest = load_manifest(tmp_folder, expected_backend=self.backend_name)
        with open(os.path.join(tmp_folder, "payload.pkl"), "rb") as f:
            payload = cloudpickle.load(f)
        has_return_val = bool(payload["has_return_val"])
        if num_workers is None:
            num_workers = int(payload.get("num_workers", 1))

        # Reconstruct the partition in numeric task order (never glob: it sorts lexically).
        tasks: List[List[int]] = []
        task_id = 0
        while os.path.exists(os.path.join(tmp_folder, "blocks", f"{task_id}.json")):
            with open(os.path.join(tmp_folder, "blocks", f"{task_id}.json")) as f:
                tasks.append([int(b) for b in json.load(f)])
            task_id += 1
        n_tasks = len(tasks)
        if n_tasks != int(manifest["n_tasks"]):
            raise ValueError(
                f"Run folder {tmp_folder!r} has {n_tasks} task files, but its manifest records "
                f"{manifest['n_tasks']}."
            )
        block_ids = [b for task in tasks for b in task]

        incomplete = [t for t, task_blocks in enumerate(tasks)
                      if set(task_blocks) - _done_blocks(tmp_folder, [t])]
        if incomplete:
            self._launch_and_wait(tmp_folder, n_tasks, num_workers, name, task_ids=incomplete)
        return self._finalize(tmp_folder, n_tasks, tasks, block_ids, has_return_val, name,
                              pre_cleanup=pre_cleanup)


class SubprocessRunner(_DistributedRunner):
    """Distributed runner that launches each task as a local subprocess.

    Exercises the full distributed protocol (cloudpickle payload, generated harness,
    result/sentinel files, ``block_ids`` re-run) without a scheduler.
    """

    backend_name = "subprocess"

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str,
                         task_ids: Optional[Sequence[int]] = None) -> None:
        """Run each task as a local subprocess, up to ``num_workers`` concurrently.

        The progress bar counts processed *blocks* (summed from the per-task done-logs) rather
        than tasks; a background thread polls the logs while the tasks run. ``task_ids``
        restricts the launch to a subset (resume); the bar still spans all tasks.
        """
        ids = list(range(n_tasks)) if task_ids is None else list(task_ids)
        python = self.config.python_executable or sys.executable
        attempt = append_attempt(tmp, task_ids=ids, concurrency=num_workers)
        attempt_number = int(attempt["attempt_number"])
        attempt_root = attempt_folder(tmp, attempt_number)
        cmd_base = [python, "-m", "bioimage_py.runner._harness", tmp]
        update_attempt(tmp, attempt_number, status="running")

        def _write_error(task_id: int, summary: str) -> None:
            """Write a traceback substitute when the harness cannot write one."""
            err_path = os.path.join(attempt_root, "errors", f"{task_id}.txt")
            if not os.path.exists(err_path):
                with open(err_path, "w") as f:
                    f.write(summary + "\n")
            shutil.copyfile(err_path, os.path.join(tmp, "error", f"{task_id}.txt"))

        def _run_task(task_id: int):
            # Persist launch failures and timeouts before finalization builds RunnerError.
            # The done-log keeps blocks that finished before a timeout.
            started = time.monotonic()
            stdout_path = os.path.join(attempt_root, "logs", f"{task_id}.out")
            stderr_path = os.path.join(attempt_root, "logs", f"{task_id}.err")
            exit_code: Optional[int] = None
            terminating_signal: Optional[int] = None
            succeeded = False
            summary: Optional[str] = None
            with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
                try:
                    proc = subprocess.run(
                        cmd_base + [str(task_id), str(attempt_number)],
                        stdout=stdout_file,
                        stderr=stderr_file,
                        text=True,
                        timeout=self.config.task_timeout,
                    )
                except subprocess.TimeoutExpired:
                    terminating_signal = int(signal.SIGKILL)
                    summary = (f"TimeoutError: worker for task {task_id} exceeded "
                               f"{self.config.task_timeout}s and was killed.")
                    _write_error(task_id, summary)
                except OSError as err:
                    summary = f"OSError: failed to launch worker for task {task_id}: {err!r}"
                    _write_error(task_id, summary)
                else:
                    if proc.returncode < 0:
                        terminating_signal = -int(proc.returncode)
                    else:
                        exit_code = int(proc.returncode)
                    succeeded = (proc.returncode == 0 and os.path.exists(
                        os.path.join(tmp, "success", f"{task_id}.success")
                    ))
                    if not succeeded:
                        summary = f"Worker for task {task_id} exited with code {proc.returncode}."
                        _write_error(task_id, summary)

            traceback_path = os.path.join(attempt_root, "errors", f"{task_id}.txt")
            write_task_outcome(
                tmp,
                attempt_number,
                {
                    "task_id": int(task_id),
                    "backend": self.backend_name,
                    "attempt_number": attempt_number,
                    "scheduler_task_id": None,
                    "scheduler_state": None,
                    "exit_code": exit_code,
                    "signal": terminating_signal,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_rss_bytes": None,
                    "failed_node": None,
                    "stdout_path": relative_path(tmp, stdout_path),
                    "stderr_path": relative_path(tmp, stderr_path),
                    "traceback_path": (relative_path(tmp, traceback_path)
                                       if os.path.exists(traceback_path) else None),
                    "succeeded": succeeded,
                    "summary": summary,
                },
            )
            return succeeded

        # Drive a block-counting progress bar from the done-logs (single source of truth, so no
        # double-counting); clamp to the total in case a resume re-reads prior lines.
        n_blocks = _total_blocks(tmp, n_tasks)
        stop = threading.Event()
        bar_thread = None
        if name:
            def _poll_bar() -> None:
                counter = _DoneLogCounter(tmp, n_tasks)
                with tqdm(total=n_blocks, desc=name, unit="block") as pbar:
                    while not stop.wait(0.5):
                        pbar.n = min(counter.count(), n_blocks)
                        pbar.refresh()
                    pbar.n = min(counter.count(), n_blocks)
                    pbar.refresh()
            bar_thread = threading.Thread(target=_poll_bar, daemon=True)
            bar_thread.start()
        try:
            with futures.ThreadPoolExecutor(max(1, int(num_workers))) as tp:
                succeeded = list(tp.map(_run_task, ids))
        except BaseException:
            update_attempt(tmp, attempt_number, status="interrupted", finished_at=utc_now())
            raise
        else:
            status = "succeeded" if all(succeeded) else "failed"
            update_attempt(tmp, attempt_number, status=status, finished_at=utc_now())
        finally:
            stop.set()
            if bar_thread is not None:
                bar_thread.join()


# Scheduler states from which a task will not progress further. The ground truth for
# success is still the per-task sentinel file; these are used only to detect *dead* tasks
# (terminal in the scheduler but with no sentinel -> failed).
_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "CANCELLED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT",
})
# Fallback array-size cap if the cluster's MaxArraySize cannot be queried.
_DEFAULT_MAX_ARRAY = 1001


class SlurmRunner(_DistributedRunner):
    """Distributed runner that submits one sbatch array job and polls it with ``sacct``.

    Reuses the full distributed protocol from :class:`_DistributedRunner` (cloudpickle
    payload, generated work-lists, per-task result + ``.success`` sentinel files, failure
    reporting and ``block_ids`` re-run) and overrides how tasks are launched and awaited.
    The per-task sentinel remains the source of truth for success. Lightweight ``sacct``
    queries detect dead tasks. A separate final query persists accounting diagnostics.
    """

    backend_name = "slurm"

    def __init__(self, config: Optional[RunnerConfig] = None):
        """Create the runner, requiring a :class:`SlurmConfig`.

        Args:
            config: The slurm configuration. ``None`` loads the user defaults from the config
                file via :meth:`SlurmConfig.load` (honoring ``BIOIMAGE_PY_NO_CONFIG`` /
                ``BIOIMAGE_PY_CONFIG``); ``tmp_root`` must still be set, here or in the file,
                before running.

        Raises:
            TypeError: If ``config`` is a non-slurm ``RunnerConfig``.
        """
        if config is None:
            config = SlurmConfig.load()
        if not isinstance(config, SlurmConfig):
            raise TypeError(
                f"SlurmRunner requires a SlurmConfig, got {type(config).__name__}. "
                "Pass job_config=SlurmConfig(...) (it carries partition/account/time/etc.)."
            )
        super().__init__(config)
        self._max_array_cache: Optional[int] = None

    def _max_tasks(self) -> int:
        """The cluster's ``MaxArraySize`` -- the hard cap on the number of tasks per array job.

        Uses ``config.max_array_size`` when set, else queries ``scontrol`` once and memoizes it
        (a cluster property, stable for this runner's lifetime). This bounds
        :meth:`_resolve_n_tasks` so over-partitioning degrades gracefully to the array limit
        instead of failing at submit.
        """
        if self.config.max_array_size is not None:
            return int(self.config.max_array_size)
        if self._max_array_cache is None:
            self._max_array_cache = self._max_array_size()
        return self._max_array_cache

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str,
                         task_ids: Optional[Sequence[int]] = None) -> None:
        """Submit an sbatch array job for the tasks and poll until they all finish.

        Args:
            tmp: The job temp folder (must live on a shared filesystem).
            n_tasks: The total number of tasks the run was partitioned into.
            num_workers: The array throttle (max tasks running concurrently).
            name: A short name used for the job name and progress display.
            task_ids: Restrict the submitted array to this subset of task indices (used by
                :meth:`resume` to resubmit only the incomplete tasks); ``None`` submits all
                ``0 .. n_tasks - 1``.
        """
        launch_ids = list(range(n_tasks)) if task_ids is None else sorted(set(int(t) for t in task_ids))
        is_resume = task_ids is not None

        def _guard_fail(message: str) -> None:
            # On a resume we must never remove the user's preserved temp folder.
            if not is_resume:
                shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(message)

        if self.config.tmp_root is None:
            _guard_fail(
                "SlurmRunner requires config.tmp_root to be set to a shared filesystem "
                "visible to all compute nodes (node-local /tmp is not usable)."
            )

        max_array = self._max_tasks()
        if len(launch_ids) > max_array:
            _guard_fail(
                f"Run partitioned into {len(launch_ids)} tasks exceeds the maximum array size "
                f"{max_array}. Lower num_workers/tasks_per_worker or use a larger block_shape."
            )

        throttle = max(1, min(int(num_workers), len(launch_ids)))
        attempt = append_attempt(tmp, task_ids=launch_ids, concurrency=throttle)
        attempt_number = int(attempt["attempt_number"])
        attempt_root = attempt_folder(tmp, attempt_number)
        script_path = os.path.join(attempt_root, "submit.sh")
        with open(script_path, "w") as f:
            f.write(self._build_script(tmp, launch_ids, throttle, name, attempt_number))
        shutil.copyfile(script_path, os.path.join(tmp, "submit.sh"))
        update_attempt(tmp, attempt_number, script_path=relative_path(tmp, script_path))

        # Unlike the tmp_root / max_array guards above, a submission failure deliberately does NOT
        # remove the temp folder: the generated submit.sh, payload, and per-task block lists are
        # exactly what's needed to diagnose why sbatch rejected the job. Re-raise naming the folder
        # so the user knows where to look.
        try:
            job_id = self._submit(script_path)
        except RuntimeError as err:
            update_attempt(
                tmp,
                attempt_number,
                status="submission_failed",
                finished_at=utc_now(),
                submission_error=str(err),
            )
            raise RuntimeError(f"{err} Temp folder preserved for debugging: {tmp}.") from err
        update_attempt(tmp, attempt_number, status="running", job_id=job_id)

        poll_states: Dict[int, str] = {}
        try:
            poll_states = self._poll(job_id, n_tasks, tmp, name, task_ids=launch_ids)
        except BaseException as error:
            accounting_path = self._persist_slurm_outcomes(
                tmp, attempt_number, job_id, launch_ids, poll_states
            )
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "orchestrator_error"
            update_attempt(
                tmp,
                attempt_number,
                status=status,
                finished_at=utc_now(),
                accounting_path=relative_path(tmp, accounting_path),
            )
            raise
        else:
            accounting_path = self._persist_slurm_outcomes(
                tmp, attempt_number, job_id, launch_ids, poll_states
            )
            succeeded = all(os.path.exists(os.path.join(tmp, "success", f"{task_id}.success"))
                            for task_id in launch_ids)
            update_attempt(
                tmp,
                attempt_number,
                status="succeeded" if succeeded else "failed",
                finished_at=utc_now(),
                accounting_path=relative_path(tmp, accounting_path),
            )

    @staticmethod
    def _format_array_indices(task_ids: Sequence[int], throttle: int) -> str:
        """Compress task ids into an sbatch ``--array`` spec, e.g. ``0,3,7-9%4``."""
        ids = sorted(set(int(t) for t in task_ids))
        parts: List[str] = []
        i = 0
        while i < len(ids):
            j = i
            while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
                j += 1
            parts.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
            i = j + 1
        return ",".join(parts) + f"%{throttle}"

    def _build_script(self, tmp: str, task_ids: Sequence[int], throttle: int, name: str,
                      attempt_number: int) -> str:
        """Render the sbatch array script for the given task indices."""
        cfg = self.config
        logs = os.path.join(attempt_folder(tmp, attempt_number), "logs")
        shebang, preamble = "#!/bin/bash", ""
        if cfg.shebang:
            lines = cfg.shebang.splitlines()
            if lines and lines[0].startswith("#!"):
                shebang, preamble = lines[0], "\n".join(lines[1:])
            else:
                preamble = cfg.shebang

        # Collapse whitespace/newlines so the name cannot break or inject directives.
        job_name = "_".join((name or "").split()) or "bioimage_py"
        directives = [
            f"--job-name={job_name}",
            f"--array={self._format_array_indices(task_ids, throttle)}",
            f"--cpus-per-task={int(cfg.cpus_per_task)}",
            f"--output={os.path.join(logs, 'slurm-%A_%a.out')}",
            f"--error={os.path.join(logs, 'slurm-%A_%a.err')}",
        ]
        if cfg.partition is not None:
            directives.append(f"--partition={cfg.partition}")
        if cfg.time is not None:
            directives.append(f"--time={cfg.time}")
        if cfg.mem is not None:
            directives.append(f"--mem={cfg.mem}")
        if int(cfg.gpus) > 0:
            directives.append(f"--gpus={int(cfg.gpus)}")
        if cfg.account is not None:
            directives.append(f"--account={cfg.account}")
        if cfg.qos is not None:
            directives.append(f"--qos={cfg.qos}")
        if cfg.constraint is not None:
            directives.append(f"--constraint={cfg.constraint}")

        python = shlex.quote(cfg.python_executable or sys.executable)
        command = (f'{python} -m bioimage_py.runner._harness {shlex.quote(tmp)} '
                   f'"${{SLURM_ARRAY_TASK_ID}}" {int(attempt_number)}')
        lines = [shebang]
        lines += [f"#SBATCH {d}" for d in directives]
        if preamble:
            lines.append(preamble)
        lines.append(command)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _submit(script_path: str) -> str:
        """Submit ``script_path`` with ``sbatch --parsable`` and return the job id."""
        sbatch = shutil.which("sbatch")
        if sbatch is None:
            raise RuntimeError("sbatch not found on PATH; the slurm CLI must be available.")
        try:
            proc = subprocess.run([sbatch, "--parsable", script_path],
                                  capture_output=True, text=True)
        except OSError as error:
            raise RuntimeError(f"Could not launch sbatch: {error!r}") from error
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch submission failed (exit {proc.returncode}): "
                               f"{proc.stderr.strip() or proc.stdout.strip()}")
        job_id = proc.stdout.strip().split(";")[0].strip()
        if not job_id.isdigit():
            raise RuntimeError(f"Could not parse job id from sbatch output: {proc.stdout!r}")
        return job_id

    @staticmethod
    def _max_array_size() -> int:
        """Return the cluster's ``MaxArraySize`` (or a safe fallback)."""
        scontrol = shutil.which("scontrol")
        if scontrol is None:
            return _DEFAULT_MAX_ARRAY
        try:
            proc = subprocess.run([scontrol, "show", "config"], capture_output=True, text=True)
        except OSError:
            return _DEFAULT_MAX_ARRAY
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", proc.stdout)
        return int(match.group(1)) if match else _DEFAULT_MAX_ARRAY

    @staticmethod
    def _parse_array_range(spec: str) -> List[int]:
        """Expand a pending-collapse range like ``[2-9,11%4]`` into its task indices."""
        body = spec.strip("[]").split("%", 1)[0]
        indices: List[int] = []
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                indices.extend(range(int(lo), int(hi) + 1))
            else:
                indices.append(int(part))
        return indices

    def _sacct_states(self, job_id: str) -> Optional[Dict[int, str]]:
        """Return ``{array_index: STATE}`` for the array job, or ``None`` on a poll error.

        ``None`` (a transient ``sacct`` failure) means *skip this poll*; an empty dict means
        the job is simply not registered with the scheduler yet. A task absent from the
        result is treated as pending, never as dead.
        """
        sacct = shutil.which("sacct")
        if sacct is None:
            raise RuntimeError("sacct not found on PATH; the slurm CLI must be available.")
        try:
            proc = subprocess.run(
                [sacct, "-X", "-n", "-P", "--format=JobID,State", "-j", str(job_id)],
                capture_output=True, text=True,
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None

        states: Dict[int, str] = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            jid, _, raw_state = line.partition("|")
            jid = jid.split(";", 1)[0]
            if "." in jid or "_" not in jid:  # step rows (defensive; -X already excludes them)
                continue
            # Take the first token: normalises e.g. "CANCELLED by 12345" -> "CANCELLED".
            state = self._normalise_slurm_state(raw_state) or ""
            suffix = jid.split("_", 1)[1]
            if suffix.startswith("["):
                for idx in self._parse_array_range(suffix):
                    states[idx] = state
            else:
                try:
                    states[int(suffix)] = state
                except ValueError:
                    continue
        return states

    @staticmethod
    def _normalise_slurm_state(value: Optional[str]) -> Optional[str]:
        """Return the canonical first token of a Slurm state."""
        if value is None:
            return None
        tokens = value.strip().split()
        return tokens[0].upper().rstrip("+") if tokens else None

    @staticmethod
    def _parse_slurm_exit_code(value: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        """Parse Slurm's ``exit:signal`` accounting value."""
        if value is None or not value.strip():
            return None, None
        code_text, _, signal_text = value.strip().partition(":")
        try:
            exit_code = int(code_text)
        except ValueError:
            exit_code = None
        try:
            terminating_signal = int(signal_text) if signal_text else None
        except ValueError:
            terminating_signal = None
        if terminating_signal == 0:
            terminating_signal = None
        return exit_code, terminating_signal

    @staticmethod
    def _parse_slurm_rss(value: Optional[str]) -> Optional[int]:
        """Parse a Slurm MaxRSS value requested with ``--units=K`` into bytes."""
        if value is None:
            return None
        text = value.strip().upper()
        if not text or text in {"N/A", "UNKNOWN"}:
            return None
        units = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3,
                 "T": 1024 ** 4, "P": 1024 ** 5}
        suffix = text[-1]
        multiplier = units.get(suffix, 1024)
        number = text[:-1] if suffix in units else text
        try:
            return int(float(number) * multiplier)
        except ValueError:
            return None

    @staticmethod
    def _parse_accounting_rows(stdout: str, fields: Sequence[str]) -> List[Dict[str, str]]:
        """Parse pipe-delimited sacct output."""
        rows: List[Dict[str, str]] = []
        for line in stdout.splitlines():
            if not line.strip():
                continue
            values = [value.strip() for value in line.rstrip("\n").split("|")]
            if len(values) != len(fields):
                continue
            rows.append(dict(zip(fields, values)))
        return rows

    def _query_slurm_accounting(self, job_id: str) -> Dict[str, Any]:
        """Run the final detailed accounting query with a core-field fallback."""
        rich_fields = [
            "JobID", "State", "ExitCode", "ElapsedRaw", "MaxRSS", "MaxRSSNode",
            "NodeList", "FailedNode", "StdOut", "StdErr",
        ]
        core_fields = ["JobID", "State", "ExitCode", "ElapsedRaw", "NodeList"]
        observation: Dict[str, Any] = {"queried_at": utc_now(), "queries": [], "rows": []}
        sacct = shutil.which("sacct")
        if sacct is None:
            observation["queries"].append({"error": "sacct not found on PATH"})
            return observation

        for fields in (rich_fields, core_fields):
            command = [
                sacct, "--array", "-n", "-P", "--units=K",
                f"--format={','.join(fields)}", "-j", str(job_id),
            ]
            try:
                proc = subprocess.run(command, capture_output=True, text=True)
            except OSError as error:
                observation["queries"].append({
                    "command": command,
                    "error": repr(error),
                })
                continue
            query = {
                "command": command,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            observation["queries"].append(query)
            if proc.returncode == 0:
                observation["fields"] = fields
                observation["rows"] = self._parse_accounting_rows(proc.stdout, fields)
                break
        return observation

    def _persist_slurm_outcomes(self, tmp: str, attempt_number: int, job_id: str,
                                task_ids: Sequence[int],
                                poll_states: Mapping[int, str]) -> str:
        """Persist raw Slurm accounting and normalized task outcomes."""
        observation = self._query_slurm_accounting(job_id)
        observation["job_id"] = str(job_id)
        observation["last_poll_states"] = {
            str(task_id): state for task_id, state in poll_states.items()
        }
        accounting_path = os.path.join(attempt_folder(tmp, attempt_number),
                                       "slurm-accounting.json")
        atomic_write_json(accounting_path, observation)

        task_rows: Dict[int, Dict[str, Dict[str, str]]] = {}
        pattern = re.compile(rf"^{re.escape(str(job_id))}_(\d+)(?:\.([^|]+))?$")
        for row in observation.get("rows", []):
            match = pattern.match(row.get("JobID", ""))
            if match is None:
                continue
            task_id = int(match.group(1))
            step = match.group(2) or "allocation"
            task_rows.setdefault(task_id, {})[step] = row

        attempt_root = attempt_folder(tmp, attempt_number)
        for task_id in task_ids:
            rows = task_rows.get(int(task_id), {})
            allocation = rows.get("allocation", {})
            batch = rows.get("batch", {})
            state = self._normalise_slurm_state(
                allocation.get("State") or batch.get("State") or poll_states.get(int(task_id))
            )
            exit_code, terminating_signal = self._parse_slurm_exit_code(
                allocation.get("ExitCode") or batch.get("ExitCode")
            )
            elapsed_text = allocation.get("ElapsedRaw") or batch.get("ElapsedRaw")
            try:
                elapsed_seconds = float(elapsed_text) if elapsed_text else None
            except ValueError:
                elapsed_seconds = None
            peak_rss_bytes = self._parse_slurm_rss(
                batch.get("MaxRSS") or allocation.get("MaxRSS")
            )
            failed_node = allocation.get("FailedNode") or batch.get("FailedNode") or None
            stdout_path = os.path.join(attempt_root, "logs", f"slurm-{job_id}_{task_id}.out")
            stderr_path = os.path.join(attempt_root, "logs", f"slurm-{job_id}_{task_id}.err")
            traceback_path = os.path.join(attempt_root, "errors", f"{task_id}.txt")
            succeeded = os.path.exists(os.path.join(tmp, "success", f"{task_id}.success"))
            write_task_outcome(
                tmp,
                attempt_number,
                {
                    "task_id": int(task_id),
                    "backend": self.backend_name,
                    "attempt_number": int(attempt_number),
                    "scheduler_task_id": f"{job_id}_{task_id}",
                    "scheduler_state": state,
                    "exit_code": exit_code,
                    "signal": terminating_signal,
                    "elapsed_seconds": elapsed_seconds,
                    "peak_rss_bytes": peak_rss_bytes,
                    "failed_node": failed_node,
                    "stdout_path": relative_path(tmp, stdout_path),
                    "stderr_path": relative_path(tmp, stderr_path),
                    # NFS visibility can lag the scheduler state. Keep the deterministic path
                    # even when the traceback is not visible to the orchestrator yet.
                    "traceback_path": relative_path(tmp, traceback_path),
                    "succeeded": succeeded,
                    "summary": None,
                },
            )
        return accounting_path

    def _job_known(self, job_id: str, attempts: int = 3) -> bool:
        """Whether the job is known to ``sacct``, retrying to tolerate post-submit lag.

        A transient ``sacct`` error (``None``) or any returned row counts as known; only a
        sustained empty result across ``attempts`` polls is treated as unknown.
        """
        for attempt in range(attempts):
            states = self._sacct_states(job_id)
            if states is None or states:
                return True
            if attempt + 1 < attempts:
                time.sleep(self.config.poll_interval)
        return False

    @staticmethod
    def _attempt_outcomes_are_terminal(tmp: str, attempt: Mapping[str, Any]) -> bool:
        """Return whether persisted outcomes resolve every task in an attempt."""
        attempt_number = int(attempt["attempt_number"])
        for task_id in attempt.get("task_ids", []):
            outcome = read_task_outcome(tmp, attempt_number, int(task_id))
            if outcome is None:
                return False
            state = outcome.get("scheduler_state")
            if not outcome.get("succeeded") and state not in _TERMINAL_STATES:
                return False
        return True

    def _poll(self, job_id: str, n_tasks: int, tmp: str, name: str,
              task_ids: Optional[Sequence[int]] = None) -> Dict[int, str]:
        """Poll ``sacct`` until every task has a visible sentinel or is confirmed dead.

        The scheduler ``State`` is not subject to NFS lag, but the ``.success`` sentinels the
        compute nodes write can take up to the mount's attribute-cache timeout to become
        visible here. So a ``COMPLETED`` task (its harness exited 0, hence wrote a sentinel)
        is given ``config.latency_wait`` for that sentinel to appear; any other terminal
        state means the harness did not succeed and the task is declared dead after a short
        confirmation grace. Tasks absent from ``sacct`` are pending, never dead.

        Args:
            job_id: The submitted array job id.
            n_tasks: The total number of tasks (spans the block-counting progress bar).
            tmp: The job temp folder (where sentinels are written).
            name: A short name for the progress bar (disables it when empty).
            task_ids: The subset of task indices this job actually runs (a resume submits only
                the incomplete tasks); resolution is over this subset, ``None`` means all tasks.
        """
        poll_ids = list(range(n_tasks)) if task_ids is None else sorted(set(int(t) for t in task_ids))

        def has_sentinel(t: int) -> bool:
            return os.path.exists(os.path.join(tmp, "success", f"{t}.success"))

        latency_wait = max(float(self.config.latency_wait), self.config.poll_interval)
        fail_grace = max(self.config.poll_interval, 5.0)
        terminal_since: Dict[int, float] = {}
        terminal_count: Dict[int, int] = {}
        resolved: set = set()
        # The bar counts processed blocks across ALL tasks (a resume credits prior progress).
        n_blocks = _total_blocks(tmp, n_tasks)
        counter = _DoneLogCounter(tmp, n_tasks)
        states: Dict[int, str] = {}
        with tqdm(total=n_blocks, desc=name or None, disable=not name, unit="block") as pbar:
            while len(resolved) < len(poll_ids):
                states = self._sacct_states(job_id)
                if states is None:  # transient sacct error: skip this poll.
                    time.sleep(self.config.poll_interval)
                    continue

                now = time.monotonic()
                ok = {t for t in poll_ids if has_sentinel(t)}
                running = sum(1 for s in states.values() if s == "RUNNING")
                dead = set()
                for t in poll_ids:
                    if t in ok:
                        terminal_since.pop(t, None)
                        terminal_count.pop(t, None)
                        continue
                    state = states.get(t)
                    if state in _TERMINAL_STATES:
                        terminal_since.setdefault(t, now)
                        terminal_count[t] = terminal_count.get(t, 0) + 1
                        # COMPLETED -> sentinel was written, just wait it out over NFS; any
                        # other terminal state -> the task will never produce a sentinel.
                        grace = latency_wait if state == "COMPLETED" else fail_grace
                        if (terminal_count[t] >= 2 and now - terminal_since[t] >= grace
                                and not has_sentinel(t)):
                            dead.add(t)
                    else:  # pending/running/requeued: reset the dead countdown.
                        terminal_since.pop(t, None)
                        terminal_count.pop(t, None)

                resolved = ok | dead
                pbar.n = min(counter.count(), n_blocks)
                pbar.set_postfix(ok=len(ok), failed=len(dead), run=running,
                                 pending=max(0, len(poll_ids) - len(resolved) - running), refresh=False)
                pbar.refresh()
                if len(resolved) >= len(poll_ids):
                    break
                try:
                    time.sleep(self.config.poll_interval)
                except KeyboardInterrupt:
                    print(f"\nInterrupted while waiting on slurm job {job_id}. The job was left "
                          f"running; reattach with SlurmRunner(...).reattach({tmp!r}).")
                    raise
        return states

    def reattach(self, tmp_folder: str, name: str = "reattach",
                 pre_cleanup: Optional[Callable[[str], None]] = None) -> Optional[list]:
        """Reattach to a previously submitted run and finalize it.

        Picks a run back up from its manifest (e.g. after the orchestrating login-node
        process was interrupted) instead of resubmitting. Only ``poll_interval`` is read
        from this runner's config, so a freshly constructed ``SlurmRunner`` can reattach.

        Args:
            tmp_folder: The job temp folder containing ``manifest.json`` and ``payload.pkl``.
            name: A short name for the progress display.
            pre_cleanup: Optional ``pre_cleanup(tmp)`` callback invoked right before the temp
                folder is removed (forwarded to :meth:`_finalize`).

        Returns:
            The per-block return values (if the run collected any), else ``None``.

        Raises:
            RunnerError: If any task failed (sentinel missing).
            RuntimeError: If the manifest's job is unknown to slurm and the run did not
                already complete.
        """
        manifest = load_manifest(tmp_folder, expected_backend=self.backend_name)
        n_tasks = int(manifest["n_tasks"])
        attempts = [attempt for attempt in manifest["attempts"] if attempt.get("job_id")]
        if not attempts:
            raise RuntimeError(f"Run folder {tmp_folder!r} has no submitted Slurm attempt.")
        attempt = attempts[-1]
        attempt_number = int(attempt["attempt_number"])
        job_id = str(attempt["job_id"])
        launch_ids = [int(task_id) for task_id in attempt.get("task_ids", [])]
        with open(os.path.join(tmp_folder, "payload.pkl"), "rb") as f:
            payload = cloudpickle.load(f)
        # A reattaching orchestrator unpickles/reduces the per-block results locally, so guard
        # its environment against skew from the submitting one just as the workers do.
        _check_versions(payload.get("versions", {"python": payload.get("python")}), role="orchestrator")
        has_return_val = bool(payload["has_return_val"])

        # Reconstruct the partition in numeric task order (never glob: it sorts lexically).
        tasks: List[List[int]] = []
        for task_id in range(n_tasks):
            with open(os.path.join(tmp_folder, "blocks", f"{task_id}.json")) as f:
                tasks.append(json.load(f))
        block_ids = [b for task in tasks for b in task]

        all_done = all(os.path.exists(os.path.join(tmp_folder, "success", f"{t}.success"))
                       for t in range(n_tasks))
        if not all_done:
            if not self._attempt_outcomes_are_terminal(tmp_folder, attempt):
                # Only a job that stays unknown to sacct across retries is unrecoverable.
                if not self._job_known(job_id):
                    raise RuntimeError(
                        f"Slurm job {job_id} is not known to the scheduler and the run did not "
                        f"complete. Inspect {tmp_folder} or resume it."
                    )
                poll_states = self._poll(job_id, n_tasks, tmp_folder, name, task_ids=launch_ids)
                accounting_path = self._persist_slurm_outcomes(
                    tmp_folder, attempt_number, job_id, launch_ids, poll_states
                )
                succeeded = all(os.path.exists(
                    os.path.join(tmp_folder, "success", f"{task_id}.success")
                ) for task_id in launch_ids)
                update_attempt(
                    tmp_folder,
                    attempt_number,
                    status="succeeded" if succeeded else "failed",
                    finished_at=utc_now(),
                    accounting_path=relative_path(tmp_folder, accounting_path),
                )

        results = self._finalize(tmp_folder, n_tasks, tasks, block_ids, has_return_val, name,
                                 pre_cleanup=pre_cleanup)
        return results if has_return_val else None
