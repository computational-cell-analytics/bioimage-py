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
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cloudpickle
from bioimage_cpp.utils import Blocking
from tqdm import tqdm

from ..sources.base import Source
from ..util import ComputeFn, group_blocks_by_shard, maybe_warn_imbalance
from .base import Runner, RunnerError
from .config import RunnerConfig, SlurmConfig


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


def _pack_groups(groups: Sequence[Sequence[int]], num_workers: int, name: str) -> List[List[int]]:
    """Bin-pack whole shard-groups into at most ``num_workers`` tasks (least-loaded first)."""
    groups = [list(g) for g in groups if g]
    if not groups:
        return [[]]
    n_tasks = max(1, min(int(num_workers), len(groups)))
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


def _count_done_blocks(tmp: str, n_tasks: int) -> int:
    """Total processed-block count across all done-logs (cheap newline count for the bar).

    Counts only newline-terminated lines, so a torn final line is not counted; use
    :func:`_done_blocks` for the authoritative set.
    """
    total = 0
    for t in range(n_tasks):
        path = os.path.join(tmp, "progress", f"{t}.log")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            total += f.read().count(b"\n")
    return total


def _total_blocks(tmp: str, n_tasks: int) -> int:
    """Total assigned-block count across all tasks (from the per-task block lists)."""
    total = 0
    for t in range(n_tasks):
        with open(os.path.join(tmp, "blocks", f"{t}.json")) as f:
            total += len(json.load(f))
    return total


class _DistributedRunner(Runner):
    """Base for runners that ship the computation to separate worker processes."""

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

        payload = {
            "function": function,
            "has_return_val": bool(has_return_val),
            "num_workers": int(num_workers),  # persisted so resume() can relaunch without it
            "python": tuple(sys.version_info[:2]),
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
            n_tasks = max(1, min(int(num_workers), len(ids))) if ids else 1
            tasks = _partition(ids, n_tasks)
        else:
            tasks = _pack_groups(groups, num_workers, name)
            n_tasks = len(tasks)
        for task_id, task_ids in enumerate(tasks):
            with open(os.path.join(tmp, "blocks", f"{task_id}.json"), "w") as f:
                json.dump([int(b) for b in task_ids], f)

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
            RunnerError: If any task is missing its success sentinel; the preserved temp
                folder and the failed block ids are attached.
        """
        # Per-block done-logs are the authority for what completed, so failure reporting is
        # precise: only blocks not in any done-log are failed (not the whole task). A task that
        # finished all its blocks but died before writing its sentinel thus contributes nothing.
        done = _done_blocks(tmp, range(n_tasks))
        failed_block_ids = sorted(int(b) for b in block_ids if int(b) not in done)
        if failed_block_ids:
            failed_tasks = [t for t in range(n_tasks)
                            if not os.path.exists(os.path.join(tmp, "success", f"{t}.success"))]
            raise RunnerError(self._failure_message(tmp, failed_tasks, name),
                              failed_block_ids=failed_block_ids, tmp_folder=tmp)

        results = self._collect(tmp, n_tasks, block_ids) if has_return_val else [None] * len(block_ids)
        if pre_cleanup is not None:
            try:
                pre_cleanup(tmp)
            except Exception as err:  # noqa: BLE001 - best-effort: never fail the run on this
                print(f"pre_cleanup callback failed for {tmp}: {err!r}")
        shutil.rmtree(tmp, ignore_errors=True)
        return results

    @staticmethod
    def _collect(tmp: str, n_tasks: int, block_ids: Sequence[int]) -> List[Any]:
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
            raise RunnerError(
                f"Result records missing for {len(missing)} block(s) after a successful run "
                f"(first: {missing[:5]}). Temp folder: {tmp}.",
                failed_block_ids=missing, tmp_folder=tmp)
        return [result_by_block[int(b)] for b in block_ids]

    @staticmethod
    def _failure_message(tmp: str, failed_tasks: Sequence[int], name: str) -> str:
        """Build an error message naming the preserved temp folder and first error."""
        first = None
        err_files = [os.path.join(tmp, "error", f"{t}.txt") for t in failed_tasks]
        err_files = [p for p in err_files if os.path.exists(p)]
        if err_files:
            with open(err_files[0]) as f:
                lines = f.read().strip().splitlines()
            first = lines[-1] if lines else None
        n = len(failed_tasks) if failed_tasks else "some"
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

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str,
                         task_ids: Optional[Sequence[int]] = None) -> None:
        """Run each task as a local subprocess, up to ``num_workers`` concurrently.

        The progress bar counts processed *blocks* (summed from the per-task done-logs) rather
        than tasks; a background thread polls the logs while the tasks run. ``task_ids``
        restricts the launch to a subset (resume); the bar still spans all tasks.
        """
        ids = list(range(n_tasks)) if task_ids is None else list(task_ids)
        python = self.config.python_executable or sys.executable
        cmd_base = [python, "-m", "bioimage_py.runner._harness", tmp]

        def _run_task(task_id: int):
            proc = subprocess.run(cmd_base + [str(task_id)], capture_output=True, text=True)
            # The harness writes its own error/<id>.txt on a caught exception. But a failure
            # *before* that try (e.g. an import error launching the module) would otherwise be
            # silent, so capture the subprocess output as a fallback error file.
            if proc.returncode != 0:
                err_path = os.path.join(tmp, "error", f"{task_id}.txt")
                if not os.path.exists(err_path):
                    with open(err_path, "w") as f:
                        f.write(f"Worker for task {task_id} exited with code {proc.returncode}.\n")
                        if proc.stdout:
                            f.write(f"--- stdout ---\n{proc.stdout}\n")
                        if proc.stderr:
                            f.write(f"--- stderr ---\n{proc.stderr}\n")
            return proc

        # Drive a block-counting progress bar from the done-logs (single source of truth, so no
        # double-counting); clamp to the total in case a resume re-reads prior lines.
        n_blocks = _total_blocks(tmp, n_tasks)
        stop = threading.Event()
        bar_thread = None
        if name:
            def _poll_bar() -> None:
                with tqdm(total=n_blocks, desc=name, unit="block") as pbar:
                    while not stop.wait(0.5):
                        pbar.n = min(_count_done_blocks(tmp, n_tasks), n_blocks)
                        pbar.refresh()
                    pbar.n = min(_count_done_blocks(tmp, n_tasks), n_blocks)
                    pbar.refresh()
            bar_thread = threading.Thread(target=_poll_bar, daemon=True)
            bar_thread.start()
        try:
            with futures.ThreadPoolExecutor(max(1, int(num_workers))) as tp:
                list(tp.map(_run_task, ids))
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
    reporting and ``block_ids`` re-run) and overrides only how tasks are launched and
    awaited. The per-task sentinel file remains the ground truth for success; ``sacct`` is
    queried only to detect tasks that died without writing a sentinel. A manifest is written
    at submission time so an interrupted run can be picked back up with :meth:`reattach`.
    """

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

        max_array = (self.config.max_array_size if self.config.max_array_size is not None
                     else self._max_array_size())
        if len(launch_ids) > max_array:
            _guard_fail(
                f"Run partitioned into {len(launch_ids)} tasks exceeds the maximum array size "
                f"{max_array}. Lower num_workers or use a larger block_shape."
            )

        os.makedirs(os.path.join(tmp, "logs"), exist_ok=True)
        throttle = max(1, min(int(num_workers), len(launch_ids)))
        script_path = os.path.join(tmp, "submit.sh")
        with open(script_path, "w") as f:
            f.write(self._build_script(tmp, launch_ids, throttle, name))

        # Unlike the tmp_root / max_array guards above, a submission failure deliberately does NOT
        # remove the temp folder: the generated submit.sh, payload, and per-task block lists are
        # exactly what's needed to diagnose why sbatch rejected the job. Re-raise naming the folder
        # so the user knows where to look.
        try:
            job_id = self._submit(script_path)
        except RuntimeError as err:
            raise RuntimeError(f"{err} Temp folder preserved for debugging: {tmp}.") from err
        manifest = {
            "job_id": job_id,
            "n_tasks": n_tasks,
            "launch_ids": launch_ids,
            "throttle": throttle,
            "name": name,
            "tmp": tmp,
            "script": script_path,
            "python_executable": self.config.python_executable or sys.executable,
            "submit_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_path = os.path.join(tmp, "manifest.json")
        if is_resume and os.path.exists(manifest_path):  # keep the prior job id for forensics
            try:
                with open(manifest_path) as f:
                    manifest["resumed_from_job_id"] = json.load(f).get("job_id")
            except (OSError, ValueError):
                pass
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self._poll(job_id, n_tasks, tmp, name, task_ids=launch_ids)

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

    def _build_script(self, tmp: str, task_ids: Sequence[int], throttle: int, name: str) -> str:
        """Render the sbatch array script for the given task indices."""
        cfg = self.config
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
            f"--output={os.path.join(tmp, 'logs', 'slurm-%A_%a.out')}",
            f"--error={os.path.join(tmp, 'logs', 'slurm-%A_%a.err')}",
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
        command = f'{python} -m bioimage_py.runner._harness {shlex.quote(tmp)} "${{SLURM_ARRAY_TASK_ID}}"'
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
        proc = subprocess.run([sbatch, "--parsable", script_path],
                              capture_output=True, text=True)
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
            tokens = raw_state.split()
            state = tokens[0].upper() if tokens else ""
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

    def _poll(self, job_id: str, n_tasks: int, tmp: str, name: str,
              task_ids: Optional[Sequence[int]] = None) -> None:
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
                pbar.n = min(_count_done_blocks(tmp, n_tasks), n_blocks)
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
        with open(os.path.join(tmp_folder, "manifest.json")) as f:
            manifest = json.load(f)
        job_id, n_tasks = str(manifest["job_id"]), int(manifest["n_tasks"])
        with open(os.path.join(tmp_folder, "payload.pkl"), "rb") as f:
            has_return_val = bool(cloudpickle.load(f)["has_return_val"])

        # Reconstruct the partition in numeric task order (never glob: it sorts lexically).
        tasks: List[List[int]] = []
        for task_id in range(n_tasks):
            with open(os.path.join(tmp_folder, "blocks", f"{task_id}.json")) as f:
                tasks.append(json.load(f))
        block_ids = [b for task in tasks for b in task]

        all_done = all(os.path.exists(os.path.join(tmp_folder, "success", f"{t}.success"))
                       for t in range(n_tasks))
        if not all_done:
            # Only a job that stays unknown to sacct across retries (not registration lag
            # right after submit, nor a transient error) is treated as unrecoverable.
            if not self._job_known(job_id):
                raise RuntimeError(
                    f"Slurm job {job_id} is not known to the scheduler and the run did not "
                    f"complete. Inspect {tmp_folder} or resubmit."
                )
            self._poll(job_id, n_tasks, tmp_folder, name)

        results = self._finalize(tmp_folder, n_tasks, tasks, block_ids, has_return_val, name,
                                 pre_cleanup=pre_cleanup)
        return results if has_return_val else None
