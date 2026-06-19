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
import subprocess
import sys
import tempfile
import time
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cloudpickle
from bioimage_cpp.utils import Blocking
from tqdm import tqdm

from ..sources.base import Source
from ..util import ComputeFn
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
        return self._run_ids(function, block_ids, payload_extra, has_return_val,
                             num_workers, name, pre_cleanup)

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

        Returns:
            The per-id return values in ``ids`` order if ``has_return_val``, else ``None``s.
        """
        ids = [int(b) for b in ids]
        tmp = tempfile.mkdtemp(prefix="bioimage_py_", dir=self.config.tmp_root)
        for sub in ("blocks", "results", "success", "error", "timings"):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)

        payload = {
            "function": function,
            "has_return_val": bool(has_return_val),
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

        n_tasks = max(1, min(int(num_workers), len(ids))) if ids else 1
        tasks = _partition(ids, n_tasks)
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
        # Ground truth for success is the per-task sentinel file, not the launcher's status.
        failed_tasks = [t for t in range(n_tasks)
                        if not os.path.exists(os.path.join(tmp, "success", f"{t}.success"))]
        if failed_tasks:
            failed_block_ids = sorted(int(b) for t in failed_tasks for b in tasks[t])
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
        """Load and order the per-task results by block id."""
        result_by_block = {}
        for task_id in range(n_tasks):
            with open(os.path.join(tmp, "results", f"{task_id}.pkl"), "rb") as f:
                for bid, res in cloudpickle.load(f):
                    result_by_block[int(bid)] = res
        return [result_by_block[int(b)] for b in block_ids]

    @staticmethod
    def _failure_message(tmp: str, failed_tasks: Sequence[int], name: str) -> str:
        """Build an error message naming the preserved temp folder and first error."""
        first = None
        err_path = os.path.join(tmp, "error", f"{failed_tasks[0]}.txt")
        if os.path.exists(err_path):
            with open(err_path) as f:
                lines = f.read().strip().splitlines()
            first = lines[-1] if lines else None
        return (
            f"{len(failed_tasks)} task(s) failed in '{name or 'run'}'. "
            f"Temp folder preserved for debugging: {tmp}. First error: {first!r}"
        )

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str) -> None:
        """Launch the worker tasks and block until they have all finished."""
        raise NotImplementedError


class SubprocessRunner(_DistributedRunner):
    """Distributed runner that launches each task as a local subprocess.

    Exercises the full distributed protocol (cloudpickle payload, generated harness,
    result/sentinel files, ``block_ids`` re-run) without a scheduler.
    """

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str) -> None:
        """Run each task as a local subprocess, up to ``num_workers`` concurrently."""
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

        with futures.ThreadPoolExecutor(max(1, int(num_workers))) as tp:
            list(tqdm(tp.map(_run_task, range(n_tasks)), total=n_tasks,
                      desc=name or None, disable=not name))


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
            config: The slurm configuration. ``None`` uses an all-default ``SlurmConfig``
                (which still requires ``tmp_root`` to be set before running).

        Raises:
            TypeError: If ``config`` is a non-slurm ``RunnerConfig``.
        """
        if config is None:
            config = SlurmConfig()
        if not isinstance(config, SlurmConfig):
            raise TypeError(
                f"SlurmRunner requires a SlurmConfig, got {type(config).__name__}. "
                "Pass job_config=SlurmConfig(...) (it carries partition/account/time/etc.)."
            )
        super().__init__(config)

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str) -> None:
        """Submit an sbatch array job for the tasks and poll until they all finish.

        Args:
            tmp: The job temp folder (must live on a shared filesystem).
            n_tasks: The number of tasks (array indices ``0 .. n_tasks - 1``).
            num_workers: The array throttle (max tasks running concurrently).
            name: A short name used for the job name and progress display.
        """
        if self.config.tmp_root is None:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(
                "SlurmRunner requires config.tmp_root to be set to a shared filesystem "
                "visible to all compute nodes (node-local /tmp is not usable)."
            )

        max_array = (self.config.max_array_size if self.config.max_array_size is not None
                     else self._max_array_size())
        if n_tasks > max_array:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(
                f"Run partitioned into {n_tasks} tasks exceeds the maximum array size "
                f"{max_array}. Lower num_workers or use a larger block_shape."
            )

        os.makedirs(os.path.join(tmp, "logs"), exist_ok=True)
        throttle = max(1, min(int(num_workers), n_tasks))
        script_path = os.path.join(tmp, "submit.sh")
        with open(script_path, "w") as f:
            f.write(self._build_script(tmp, n_tasks, throttle, name))

        # Unlike the tmp_root / max_array guards above, a submission failure deliberately does NOT
        # remove the temp folder: the generated submit.sh, payload, and per-task block lists are
        # exactly what's needed to diagnose why sbatch rejected the job. Re-raise naming the folder
        # so the user knows where to look.
        try:
            job_id = self._submit(script_path)
        except RuntimeError as err:
            raise RuntimeError(f"{err} Temp folder preserved for debugging: {tmp}.") from err
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump({
                "job_id": job_id,
                "n_tasks": n_tasks,
                "throttle": throttle,
                "name": name,
                "tmp": tmp,
                "script": script_path,
                "python_executable": self.config.python_executable or sys.executable,
                "submit_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

        self._poll(job_id, n_tasks, tmp, name)

    def _build_script(self, tmp: str, n_tasks: int, throttle: int, name: str) -> str:
        """Render the sbatch array script for the run."""
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
            f"--array=0-{n_tasks - 1}%{throttle}",
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

    def _poll(self, job_id: str, n_tasks: int, tmp: str, name: str) -> None:
        """Poll ``sacct`` until every task has a visible sentinel or is confirmed dead.

        The scheduler ``State`` is not subject to NFS lag, but the ``.success`` sentinels the
        compute nodes write can take up to the mount's attribute-cache timeout to become
        visible here. So a ``COMPLETED`` task (its harness exited 0, hence wrote a sentinel)
        is given ``config.latency_wait`` for that sentinel to appear; any other terminal
        state means the harness did not succeed and the task is declared dead after a short
        confirmation grace. Tasks absent from ``sacct`` are pending, never dead.

        Args:
            job_id: The submitted array job id.
            n_tasks: The number of tasks to await.
            tmp: The job temp folder (where sentinels are written).
            name: A short name for the progress bar (disables it when empty).
        """
        def has_sentinel(t: int) -> bool:
            return os.path.exists(os.path.join(tmp, "success", f"{t}.success"))

        latency_wait = max(float(self.config.latency_wait), self.config.poll_interval)
        fail_grace = max(self.config.poll_interval, 5.0)
        terminal_since: Dict[int, float] = {}
        terminal_count: Dict[int, int] = {}
        resolved: set = set()
        with tqdm(total=n_tasks, desc=name or None, disable=not name) as pbar:
            while len(resolved) < n_tasks:
                states = self._sacct_states(job_id)
                if states is None:  # transient sacct error: skip this poll.
                    time.sleep(self.config.poll_interval)
                    continue

                now = time.monotonic()
                ok = {t for t in range(n_tasks) if has_sentinel(t)}
                running = sum(1 for s in states.values() if s == "RUNNING")
                dead = set()
                for t in range(n_tasks):
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
                pbar.n = len(resolved)
                pbar.set_postfix(ok=len(ok), failed=len(dead), run=running,
                                 pending=max(0, n_tasks - len(resolved) - running), refresh=False)
                pbar.refresh()
                if len(resolved) >= n_tasks:
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
