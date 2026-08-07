"""Runner base class, the backend-independent ``run`` logic, and the local runner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent import futures
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Iterable, Iterator, List, Mapping,
                    Optional, Protocol, Sequence, Tuple)

from bioimage_cpp.utils import Blocking
from threadpoolctl import ThreadpoolController
from tqdm import tqdm

from ..sources.base import Source
from ..sources.dispatch import SourceLike, as_source
from ..util import (ComputeFn, derive_block_shape, get_blocking, group_blocks_by_shard,
                    maybe_warn_imbalance, normalize_halo)
from ._work import (Batch, ExplicitIdsSpec, RangeSpec, RegularBatchPlan, ShardRoutingPlan,
                    WorkSpec, assignment_length, iter_assignment, logical_size,
                    make_shard_routing)
from .config import RunnerConfig

if TYPE_CHECKING:
    from ..tables import TableDataset

# A single, process-wide threadpool controller, built once here at import time (in the main
# thread). Worker threads reuse it via ``_TP_CONTROLLER.limit(...)`` instead of constructing their
# own controller: ``ThreadpoolController()`` runs a ``dl_iterate_phdr`` library scan whose ctypes
# callback needs the GIL while the glibc loader lock is held, so constructing it inside worker
# threads deadlocks against any concurrent first-time import (dlopen) in another thread. Building it
# once at import avoids that; ``.limit()`` on an existing controller only reads/sets per-library
# thread counts and does no further scan.
_TP_CONTROLLER = ThreadpoolController()


@dataclass(frozen=True)
class TaskFailure:
    """Diagnostic information for one failed distributed task.

    Backend-specific values are ``None`` when the backend does not provide them. Paths refer to
    files in the preserved run folder.
    """

    task_id: int
    backend: str
    attempt_number: int
    scheduler_task_id: Optional[str] = None
    scheduler_state: Optional[str] = None
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    failed_node: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    traceback_path: Optional[str] = None


class _FailedIdSpec(Protocol):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[int]:
        ...


class RunnerError(RuntimeError):
    """Raised when one or more runner work units fail.

    Attributes:
        failed_block_ids: The failed IDs as a list. Access materializes compact ranges.
        failed_block_count: The number of failed IDs.
        failed_batches: The failed batches for :meth:`Runner.map_batches`.
        tmp_folder: The preserved temp folder for distributed jobs (``None`` for local).
        task_failures: Structured diagnostics for failed distributed tasks.
    """

    def __init__(self, message: str, failed_block_ids: Optional[Sequence[int]] = None,
                 tmp_folder: Optional[str] = None, *,
                 task_failures: Optional[Sequence[TaskFailure]] = None,
                 failed_id_specs: Optional[Sequence[_FailedIdSpec]] = None,
                 failed_batches: Optional[Sequence[Batch]] = None):
        super().__init__(message)
        self._failed_block_ids = ([int(value) for value in failed_block_ids]
                                  if failed_block_ids is not None else None)
        self._failed_id_specs = tuple(failed_id_specs or ())
        self.tmp_folder = tmp_folder
        self.task_failures: List[TaskFailure] = list(task_failures or [])
        self.failed_batches: Tuple[Batch, ...] = tuple(failed_batches or ())

    @property
    def failed_block_ids(self) -> List[int]:
        """Return failed IDs as a list, materializing compact specifications on access."""
        if self._failed_block_ids is None:
            self._failed_block_ids = list(self.iter_failed_block_ids())
            self._failed_id_specs = ()
        return self._failed_block_ids

    @property
    def failed_block_count(self) -> int:
        """Return the failed logical ID count without expanding compact ranges."""
        explicit = len(self._failed_block_ids) if self._failed_block_ids is not None else 0
        return explicit + sum(len(spec) for spec in self._failed_id_specs)

    def iter_failed_block_ids(self) -> Iterator[int]:
        """Iterate over failed IDs without expanding compact ranges."""
        if self._failed_block_ids is not None:
            yield from self._failed_block_ids
        for spec in self._failed_id_specs:
            yield from (int(value) for value in spec)


def run_block(function: ComputeFn, blocking: Blocking, block_id: int,
              inputs: Sequence[Source], outputs: Sequence[Source],
              mask: Optional[Source], halo: Optional[Sequence[int]]) -> Any:
    """Run the per-block ``function`` for a single block.

    This is the single per-block code path shared by every backend (local and
    distributed), which is what guarantees identical results across backends.

    Args:
        function: The per-block function ``function(block, inputs, outputs, mask)``.
        blocking: A ``bioimage_cpp.utils.Blocking``.
        block_id: The block id to process.
        inputs: Tuple of opened input sources.
        outputs: Tuple of opened output sources.
        mask: An opened mask source or ``None``.
        halo: A per-axis halo list, or ``None`` for no halo.

    Returns:
        The per-block return value of ``function`` (may be ``None``).
    """
    if halo is None:
        block = blocking.get_block(int(block_id))
    else:
        block = blocking.get_block_with_halo(int(block_id), [int(h) for h in halo])
    return function(block, inputs, outputs, mask)


class Runner(ABC):
    """Abstract runner. Subclasses implement :meth:`_execute` for a specific backend."""

    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()

    def run(
        self,
        function: ComputeFn,
        inputs: Sequence[SourceLike],
        outputs: Sequence[SourceLike] = (),
        *,
        block_shape: Optional[Tuple[int, ...]] = None,
        halo: Optional[Sequence[int]] = None,
        mask: Optional[SourceLike] = None,
        num_workers: int = 1,
        block_ids: Optional[Sequence[int]] = None,
        has_return_val: bool = False,
        name: str = "",
        roi: Optional[Tuple[slice, ...]] = None,
        pre_cleanup: Optional[Callable[[str], None]] = None,
        resume_from: Optional[str] = None,
    ) -> Optional[list]:
        """Run ``function`` block-wise over the inputs/outputs.

        Args:
            function: Per-block function ``function(block, inputs, outputs, mask)``.
            inputs: Input source-like objects (read).
            outputs: Output source-like objects (written in place).
            block_shape: Block shape; defaults to the domain source's chunks.
            halo: Per-axis halo; if given, ``function`` receives a ``BlockWithHalo``.
            mask: Optional binary mask source.
            num_workers: Number of parallel workers / tasks.
            block_ids: Restrict processing to these blocks (for re-running failures).
            has_return_val: Whether ``function`` returns a value to collect.
            name: A short name for progress display.
            roi: Region of interest to restrict the blocking to.
            pre_cleanup: Optional callback ``pre_cleanup(tmp_folder)`` invoked on the
                orchestrating process with the job temp folder right before it is deleted
                (distributed backends only, success path only). Use it to read out anything
                worth keeping from the temp folder (e.g. the per-task timing files under
                ``tmp_folder/timings/``) before cleanup. Ignored by the local runner, which
                has no temp folder.
            resume_from: Distributed backends only. Path to the preserved temp folder of a
                failed run (``RunnerError.tmp_folder``). Re-runs only the blocks that did not
                complete and merges them with the already-completed ones, so the result is
                correct and complete. The run is resumed from the serialized payload, so
                ``function``/``inputs``/``outputs``/``block_shape``/... from this call are
                **ignored** -- pass ``resume_from`` to *finish the same call*, not to start a
                new one. Mutually exclusive with ``block_ids``.

        Returns:
            The list of per-block return values (in ``block_ids`` order) if
            ``has_return_val``, else ``None``.
        """
        if resume_from is not None:
            if block_ids is not None:
                raise ValueError("resume_from and block_ids are mutually exclusive; resume_from "
                                 "re-runs the original partition's un-done blocks.")
            return self._resume_entry(resume_from, name=name, pre_cleanup=pre_cleanup)

        inputs = [as_source(i) for i in inputs]
        outputs = [as_source(o) for o in outputs]
        mask_source = as_source(mask) if mask is not None else None

        domain = inputs[0] if inputs else (outputs[0] if outputs else None)
        if domain is None:
            raise ValueError("run() requires at least one input or output source.")

        # Shape consistency: all inputs and the mask must match the domain shape.
        dom_shape = tuple(domain.shape)
        for src in inputs + ([mask_source] if mask_source is not None else []):
            if tuple(src.shape) != dom_shape:
                raise ValueError(
                    f"Shape mismatch: source with shape {src.shape} does not match the "
                    f"domain shape {domain.shape}."
                )
        # Outputs may carry a leading channel axis, but their trailing spatial dims must match
        # the domain so the per-block roi indexes them consistently.
        for out in outputs:
            out_shape = tuple(out.shape)
            if out_shape[-len(dom_shape):] != dom_shape:
                raise ValueError(
                    f"Output shape {out_shape} is incompatible with the domain shape "
                    f"{dom_shape}: its trailing dimensions must match the domain."
                )

        block_shape = derive_block_shape(domain, block_shape)
        halo_n = normalize_halo(halo, domain.ndim) if halo is not None else None
        self._validate_write_safety(outputs, block_shape)

        blocking = get_blocking(domain.shape, block_shape, roi)
        if block_ids is None:
            work: WorkSpec = RangeSpec(0, int(blocking.number_of_blocks))
        else:
            work = ExplicitIdsSpec(block_ids)

        results = self._execute(
            function=function, inputs=inputs, outputs=outputs, mask=mask_source,
            blocking=blocking, block_ids=work, halo=halo_n,
            has_return_val=has_return_val, num_workers=num_workers, name=name,
            shape=tuple(domain.shape), block_shape=block_shape, roi=roi,
            pre_cleanup=pre_cleanup,
        )
        return results if has_return_val else None

    def map(
        self,
        function: Callable[[int], Any],
        n_items: Optional[int] = None,
        *,
        item_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
        has_return_val: bool = True,
        name: str = "",
        pre_cleanup: Optional[Callable[[str], None]] = None,
        resume_from: Optional[str] = None,
    ) -> Optional[list]:
        """Map ``function(index)`` over item indices in parallel, across any backend.

        Unlike :meth:`run`, this is not block-wise: there is no domain, blocking, sources or
        mask. ``function`` takes a single integer index and returns its result; it must carry
        whatever data it needs in its (cloudpickled) closure — e.g. a `SourceSpec` it reopens
        and a file path it reads. This is the per-item counterpart used by per-object
        workflows.

        Args:
            function: The per-item function ``function(index) -> result``.
            n_items: The number of items; indices ``0 .. n_items - 1`` are processed. Ignored
                if ``item_ids`` is given.
            item_ids: Explicit item indices to process (e.g. to re-run failures). Defaults to
                ``range(n_items)``.
            num_workers: Number of parallel workers / tasks.
            has_return_val: Whether ``function`` returns a value to collect.
            name: A short name for progress display.
            pre_cleanup: Optional ``pre_cleanup(tmp_folder)`` callback (distributed backends
                only); see :meth:`run`.
            resume_from: Distributed backends only; the preserved temp folder of a failed run
                (see :meth:`run`). Re-runs only the incomplete items and merges with those
                already done. Mutually exclusive with ``item_ids``.

        Returns:
            The list of per-item return values (in ``item_ids`` order) if ``has_return_val``,
            else ``None``.

        Raises:
            ValueError: If neither ``n_items`` nor ``item_ids`` is given.
        """
        if resume_from is not None:
            if item_ids is not None:
                raise ValueError("resume_from and item_ids are mutually exclusive; resume_from "
                                 "re-runs the original partition's un-done items.")
            return self._resume_entry(resume_from, name=name, pre_cleanup=pre_cleanup)

        if item_ids is None:
            if n_items is None:
                raise ValueError("map() requires either n_items or item_ids.")
            if int(n_items) < 0:
                raise ValueError("n_items must be non-negative.")
            work: WorkSpec = RangeSpec(0, int(n_items))
        else:
            work = ExplicitIdsSpec(item_ids)

        results = self._execute_map(
            function=function, item_ids=work, has_return_val=has_return_val,
            num_workers=num_workers, name=name, pre_cleanup=pre_cleanup,
        )
        return results if has_return_val else None

    def map_batches(
        self,
        function: Callable[..., Any],
        n_items: Optional[int] = None,
        *,
        batch_size: Optional[int] = None,
        num_workers: int = 1,
        has_return_val: bool = False,
        name: str = "",
        pre_cleanup: Optional[Callable[[str], None]] = None,
        resume_from: Optional[str] = None,
        result_sink: Optional["TableDataset"] = None,
    ) -> Any:
        """Map ``function(batch)`` over deterministic contiguous batches.

        The completion, failure, and retry unit is one :class:`Batch`. A batch function can
        write output through its closure and return no in-memory result. With a table result
        sink, the runner calls ``function(batch, writer)`` and returns the completed dataset.

        Args:
            function: The function ``function(batch) -> result``.
            n_items: The non-negative logical item count.
            batch_size: The positive maximum number of items in each batch.
            num_workers: The local concurrency or distributed task throttle.
            has_return_val: Collect one ordered result per batch when true.
            name: A short progress display name.
            pre_cleanup: Distributed success callback. See :meth:`run`.
            resume_from: A preserved distributed run folder.
            result_sink: A table dataset that receives one typed part per batch.

        Returns:
            A table dataset when ``result_sink`` is set. Otherwise, returns one result per
            batch when ``has_return_val`` is true, or ``None``.
        """
        if resume_from is not None:
            return self._resume_entry(resume_from, name=name, pre_cleanup=pre_cleanup)
        if n_items is None:
            raise ValueError("map_batches() requires n_items.")
        if batch_size is None:
            raise ValueError("map_batches() requires batch_size.")
        batches = RegularBatchPlan(int(n_items), int(batch_size))
        if result_sink is not None:
            from ..tables import TableDataset

            if not isinstance(result_sink, TableDataset):
                raise TypeError("result_sink must be a TableDataset.")
            if has_return_val:
                raise ValueError("result_sink and has_return_val=True are mutually exclusive.")
            result_sink._bind_batches(batches)
        results = self._execute_batches(
            function=function, batches=batches, has_return_val=has_return_val,
            num_workers=num_workers, name=name, pre_cleanup=pre_cleanup,
            result_sink=result_sink,
        )
        return results if has_return_val or result_sink is not None else None

    def _resume_entry(self, tmp_folder: str, *, name: str,
                      pre_cleanup: Optional[Callable[[str], None]]) -> Optional[list]:
        """Resume a failed run from its temp folder; overridden by distributed runners.

        The local runner keeps no temp folder, so resuming is not possible here.
        """
        raise ValueError(
            "resume_from is only valid for distributed backends (subprocess/slurm); the local "
            "runner keeps no temp folder. Re-run the operation to recompute in-process "
            "(optionally with block_ids=err.failed_block_ids for a subset)."
        )

    @staticmethod
    def _validate_write_safety(outputs: Sequence[Source], block_shape: Sequence[int]) -> None:
        """Conservative guard: chunked output write-blocks must be a multiple of chunks.

        This prevents two blocks from concurrently writing the same chunk (which would
        corrupt it). Auto-derivation of a safe block shape is a flagged TODO.

        Sharded outputs (``out.shards is not None``) are exempt: for them the atomic write
        unit is the shard, and they are made safe by shard-exclusive routing (each shard's
        blocks go to one worker, run sequentially) rather than by constraining the block
        shape — see :func:`bioimage_py.util.group_blocks_by_shard`.

        ``block_shape`` is spatial-only, but an output may carry a leading channel axis (its
        ``chunks`` then have one extra leading entry); the block shape is aligned against the
        trailing (spatial) chunk axes, since the channel axis is fully written by every block.
        """
        for out in outputs:
            if out.shards is not None:
                continue
            chunks = out.chunks
            if chunks is None or len(chunks) < len(block_shape):
                continue
            spatial_chunks = tuple(chunks[-len(block_shape):])
            for bs, ch in zip(block_shape, spatial_chunks):
                if bs % ch != 0:
                    raise ValueError(
                        f"Unsafe block shape for writing: {tuple(block_shape)} is not a multiple "
                        f"of the output (spatial) chunk shape {spatial_chunks}. Concurrent writes "
                        "could corrupt shared chunks; use a block shape that is a chunk multiple."
                    )

    @abstractmethod
    def _execute(
        self,
        *,
        function: ComputeFn,
        inputs: Sequence[Source],
        outputs: Sequence[Source],
        mask: Optional[Source],
        blocking: Blocking,
        block_ids: WorkSpec,
        halo: Optional[Sequence[int]],
        has_return_val: bool,
        num_workers: int,
        name: str,
        shape: Tuple[int, ...],
        block_shape: Tuple[int, ...],
        roi: Optional[Tuple[slice, ...]],
        pre_cleanup: Optional[Callable[[str], None]] = None,
    ) -> Optional[List[Any]]:
        """Execute the per-block function over ``block_ids`` and return ordered results."""
        ...

    @abstractmethod
    def _execute_map(
        self,
        *,
        function: Callable[[int], Any],
        item_ids: WorkSpec,
        has_return_val: bool,
        num_workers: int,
        name: str,
        pre_cleanup: Optional[Callable[[str], None]] = None,
    ) -> Optional[List[Any]]:
        """Execute ``function(index)`` over ``item_ids`` and return ordered results."""
        ...

    @abstractmethod
    def _execute_batches(
        self,
        *,
        function: Callable[..., Any],
        batches: RegularBatchPlan,
        has_return_val: bool,
        num_workers: int,
        name: str,
        pre_cleanup: Optional[Callable[[str], None]] = None,
        result_sink: Optional["TableDataset"] = None,
    ) -> Any:
        """Execute ``function(batch)`` over a regular batch plan."""
        ...


class LocalRunner(Runner):
    """Run blocks locally with a thread pool."""

    def _execute(
        self,
        *,
        function: ComputeFn,
        inputs: Sequence[Source],
        outputs: Sequence[Source],
        mask: Optional[Source],
        blocking: Blocking,
        block_ids: WorkSpec,
        halo: Optional[Sequence[int]],
        has_return_val: bool,
        num_workers: int,
        name: str,
        shape: Tuple[int, ...],
        block_shape: Tuple[int, ...],
        roi: Optional[Tuple[slice, ...]],
        pre_cleanup: Optional[Callable[[str], None]] = None,
    ) -> Optional[List[Any]]:
        def call_one(bid: int) -> Any:
            return run_block(function, blocking, bid, inputs, outputs, mask, halo)

        shard_shapes = [output.shards for output in outputs if output.shards is not None]
        if shard_shapes and isinstance(block_ids, RangeSpec) and block_ids == RangeSpec(
                0, int(blocking.number_of_blocks)):
            routing = make_shard_routing(blocking, shard_shapes)
            assignments = _ComponentAssignments(routing)
            if routing.n_components <= 10_000:
                loads = [assignment_length(assignments[index])
                         for index in range(len(assignments))]
                maybe_warn_imbalance(loads, num_workers, routing.n_components, name)
            return self._run_assignment_pool(
                block_ids, assignments, call_one, num_workers, name,
                has_return_val=has_return_val, unit="block",
            )

        groups = group_blocks_by_shard(blocking, outputs, block_ids)
        if groups is not None:
            maybe_warn_imbalance([len(group) for group in groups], num_workers,
                                 len(groups), name)
            return self._run_legacy_groups(
                groups, block_ids, call_one, num_workers, name,
                has_return_val=has_return_val, unit="block",
            )
        return self._run_pool(
            block_ids, call_one, num_workers, name,
            has_return_val=has_return_val, unit="block",
        )

    def _execute_map(
        self,
        *,
        function: Callable[[int], Any],
        item_ids: WorkSpec,
        has_return_val: bool,
        num_workers: int,
        name: str,
        pre_cleanup: Optional[Callable[[str], None]] = None,
    ) -> Optional[List[Any]]:
        return self._run_pool(
            item_ids, lambda item_id: function(int(item_id)), num_workers, name,
            has_return_val=has_return_val, unit="item",
        )

    def _execute_batches(
        self,
        *,
        function: Callable[..., Any],
        batches: RegularBatchPlan,
        has_return_val: bool,
        num_workers: int,
        name: str,
        pre_cleanup: Optional[Callable[[str], None]] = None,
        result_sink: Optional["TableDataset"] = None,
    ) -> Any:
        if result_sink is not None:
            execution_error = None
            try:
                self._run_pool(
                    batches, lambda batch: result_sink._run_batch(function, batch),
                    num_workers, name, has_return_val=False, unit="batch",
                )
            except RunnerError as error:
                execution_error = error
            try:
                return result_sink._finalize(batches)
            except ValueError as error:
                from ..tables import TablePartsError

                if not isinstance(error, TablePartsError):
                    raise
                if execution_error is not None:
                    raise execution_error
                raise RunnerError(
                    f"{len(error.failed_batches)} table batch(es) have missing or invalid parts "
                    f"in '{name or 'run'}'.",
                    failed_batches=error.failed_batches,
                ) from error
        return self._run_pool(
            batches, function, num_workers, name, has_return_val=has_return_val,
            unit="batch",
        )

    @staticmethod
    def _run_pool(work: WorkSpec, call_one: Callable[[Any], Any], num_workers: int,
                  name: str, *, has_return_val: bool, unit: str) -> Optional[List[Any]]:
        """Run scalar work with a bounded pending-future window."""
        worker_count = max(1, int(num_workers))
        pending_limit = 2 * worker_count
        results = [None] * len(work) if has_return_val else None
        pending: dict[futures.Future[Any], Tuple[int, Any]] = {}
        next_position = 0
        failed_positions: List[int] = []
        first_error: Optional[BaseException] = None
        stop_submitting = False

        def invoke(value: Any) -> Any:
            with _TP_CONTROLLER.limit(limits=1):
                return call_one(value)

        def submit(executor: futures.ThreadPoolExecutor) -> None:
            nonlocal next_position
            while (not stop_submitting and len(pending) < pending_limit
                   and next_position < len(work)):
                value = work[next_position]
                pending[executor.submit(invoke, value)] = (next_position, value)
                next_position += 1

        total = work.n_items if isinstance(work, RegularBatchPlan) else len(work)
        with futures.ThreadPoolExecutor(worker_count) as executor:
            submit(executor)
            with tqdm(total=total, desc=name or None, disable=not name) as progress:
                while pending:
                    done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        position, value = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:  # noqa: BLE001
                            failed_positions.append(position)
                            stop_submitting = True
                            if first_error is None:
                                first_error = error
                        else:
                            if results is not None:
                                results[position] = result
                            progress.update(logical_size(value))
                    submit(executor)

        if failed_positions or next_position < len(work):
            if isinstance(work, RegularBatchPlan):
                failed_batches = tuple(work[position] for position in failed_positions)
                failed_batches += tuple(work[position] for position in
                                        range(next_position, len(work)))
                raise RunnerError(
                    f"{len(failed_batches)} batch(es) failed or were not started in "
                    f"'{name or 'run'}'. First error: {first_error!r}",
                    failed_batches=failed_batches,
                )
            failed_specs: List[_FailedIdSpec] = [
                ExplicitIdsSpec(tuple(int(work[position]) for position in failed_positions))
            ]
            if next_position < len(work):
                failed_specs.append(work[next_position:])  # type: ignore[arg-type]
            failed_count = sum(len(spec) for spec in failed_specs)
            preview = []
            for spec in failed_specs:
                for value in spec:
                    preview.append(int(value))
                    if len(preview) == 10:
                        break
                if len(preview) == 10:
                    break
            raise RunnerError(
                f"{failed_count} {unit}(s) failed or were not started in "
                f"'{name or 'run'}': {preview}. First error: {first_error!r}",
                failed_id_specs=failed_specs,
            )
        return results

    @staticmethod
    def _run_assignment_pool(
        work: WorkSpec,
        assignments: Sequence[Mapping[str, Any]],
        call_one: Callable[[Any], Any],
        num_workers: int,
        name: str,
        *,
        has_return_val: bool,
        unit: str,
    ) -> Optional[List[Any]]:
        """Run compact multi-item assignments with bounded pending futures."""
        worker_count = max(1, int(num_workers))
        pending_limit = 2 * worker_count
        results = [None] * len(work) if has_return_val else None
        pending: dict[futures.Future[Any], Tuple[int, Mapping[str, Any]]] = {}
        next_assignment = 0
        failed_specs: List[_FailedIdSpec] = []
        first_error: Optional[BaseException] = None
        stop_submitting = False

        def run_assignment(assignment: Mapping[str, Any]):
            local_results = [] if has_return_val else None
            with _TP_CONTROLLER.limit(limits=1):
                for offset, (position, value) in enumerate(iter_assignment(work, assignment)):
                    try:
                        result = call_one(value)
                    except Exception as error:  # noqa: BLE001
                        return local_results, offset, error
                    if local_results is not None:
                        local_results.append((position, result))
            return local_results, None, None

        def submit(executor: futures.ThreadPoolExecutor) -> None:
            nonlocal next_assignment
            while (not stop_submitting and len(pending) < pending_limit
                   and next_assignment < len(assignments)):
                assignment = assignments[next_assignment]
                pending[executor.submit(run_assignment, assignment)] = (
                    next_assignment, assignment)
                next_assignment += 1

        with futures.ThreadPoolExecutor(worker_count) as executor:
            submit(executor)
            with tqdm(total=len(work), desc=name or None, disable=not name) as progress:
                while pending:
                    done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        _, assignment = pending.pop(future)
                        local_results, failed_offset, error = future.result()
                        if results is not None:
                            for position, result in local_results:
                                results[position] = result
                        if failed_offset is not None:
                            stop_submitting = True
                            failed_specs.append(_AssignmentValues(work, assignment, failed_offset))
                            if first_error is None:
                                first_error = error
                        completed = (assignment_length(assignment) if failed_offset is None
                                     else failed_offset)
                        progress.update(completed)
                    submit(executor)

        if next_assignment < len(assignments):
            first = assignments[next_assignment]
            last = assignments[-1]
            if first.get("kind") == "shard_components":
                failed_specs.append(_AssignmentValues(
                    work,
                    {"kind": "shard_components", "start": first["start"],
                     "stop": last["stop"], "routing": first["routing"]},
                ))
            else:
                failed_specs.extend(_AssignmentValues(work, assignments[index])
                                    for index in range(next_assignment, len(assignments)))
        if failed_specs:
            failed_count = sum(len(spec) for spec in failed_specs)
            raise RunnerError(
                f"{failed_count} {unit}(s) failed or were not started in "
                f"'{name or 'run'}'. First error: {first_error!r}",
                failed_id_specs=failed_specs,
            )
        return results

    @classmethod
    def _run_legacy_groups(
        cls,
        groups: Sequence[Sequence[int]],
        work: WorkSpec,
        call_one: Callable[[Any], Any],
        num_workers: int,
        name: str,
        *,
        has_return_val: bool,
        unit: str,
    ) -> Optional[List[Any]]:
        positions: dict[int, List[int]] = {}
        for position, value in enumerate(work):
            positions.setdefault(int(value), []).append(position)
        position_groups = []
        for group in groups:
            position_groups.append(tuple(positions[int(value)].pop(0) for value in group))

        class LegacyAssignments(Sequence[Mapping[str, Any]]):
            def __len__(self) -> int:
                return len(position_groups)

            def __getitem__(self, index: int) -> Mapping[str, Any]:
                return {"kind": "positions", "positions": position_groups[index]}

        return cls._run_assignment_pool(
            work, LegacyAssignments(), call_one, num_workers, name,
            has_return_val=has_return_val, unit=unit,
        )


class _AssignmentValues(Iterable[int]):
    """A lazy failed-ID view over one compact task assignment."""

    def __init__(self, work: WorkSpec, assignment: Mapping[str, Any], skip: int = 0):
        self._work = work
        self._assignment = assignment
        self._skip = int(skip)

    def __len__(self) -> int:
        return assignment_length(self._assignment) - self._skip

    def __iter__(self) -> Iterator[int]:
        for _, value in iter_assignment(self._work, self._assignment, self._skip):
            yield int(value)


class _ComponentAssignments(Sequence[Mapping[str, Any]]):
    """A lazy sequence with one assignment per shard-routing component."""

    def __init__(self, routing: ShardRoutingPlan):
        self._routing = routing
        self._descriptor = routing.descriptor()

    def __len__(self) -> int:
        return self._routing.n_components

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return {"kind": "shard_components", "start": index, "stop": index + 1,
                "length": self._routing.component_size(index), "routing": self._descriptor}
