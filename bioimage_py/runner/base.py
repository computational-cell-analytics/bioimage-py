"""Runner base class, the backend-independent ``run`` logic, and the local runner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bioimage_cpp.utils import Blocking
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..sources.base import Source
from ..sources.dispatch import SourceLike, as_source
from ..util import (ComputeFn, derive_block_shape, get_blocking, group_blocks_by_shard,
                    maybe_warn_imbalance, normalize_halo)
from .config import RunnerConfig


class RunnerError(RuntimeError):
    """Raised when one or more blocks fail.

    Attributes:
        failed_block_ids: The ids of the blocks that failed (re-run with these).
        tmp_folder: The preserved temp folder for distributed jobs (``None`` for local).
    """

    def __init__(self, message: str, failed_block_ids: Optional[Sequence[int]] = None,
                 tmp_folder: Optional[str] = None):
        super().__init__(message)
        self.failed_block_ids: List[int] = [int(b) for b in (failed_block_ids or [])]
        self.tmp_folder = tmp_folder


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
            block_ids = list(range(int(blocking.number_of_blocks)))
        else:
            block_ids = [int(b) for b in block_ids]

        results = self._execute(
            function=function, inputs=inputs, outputs=outputs, mask=mask_source,
            blocking=blocking, block_ids=block_ids, halo=halo_n,
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
            item_ids = list(range(int(n_items)))
        else:
            item_ids = [int(i) for i in item_ids]

        results = self._execute_map(
            function=function, item_ids=item_ids, has_return_val=has_return_val,
            num_workers=num_workers, name=name, pre_cleanup=pre_cleanup,
        )
        return results if has_return_val else None

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
        """Execute the per-block function over ``block_ids`` and return ordered results."""
        ...

    @abstractmethod
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
        """Execute ``function(index)`` over ``item_ids`` and return ordered results."""
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
        """Run the blocks in a thread pool, collecting results and re-raising failures.

        ``pre_cleanup`` is accepted for interface parity but ignored: the local runner has
        no temp folder (and no per-worker concept) to read out before returning.
        """
        def call_one(bid: int) -> Any:
            return run_block(function, blocking, bid, inputs, outputs, mask, halo)

        # For sharded outputs, group blocks so each shard is written by a single thread
        # (a group runs sequentially) and never corrupted by concurrent writes; otherwise
        # each block is its own group, reproducing the plain one-future-per-block path.
        groups = group_blocks_by_shard(blocking, outputs, block_ids)
        if groups is None:
            groups = [[int(b)] for b in block_ids]
        else:
            maybe_warn_imbalance([len(g) for g in groups], num_workers, len(groups), name)
        return self._run_pool(groups, call_one, num_workers, name, unit="block")

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
        """Run ``function(index)`` over ``item_ids`` in a thread pool (``pre_cleanup`` ignored)."""
        groups = [[int(i)] for i in item_ids]
        return self._run_pool(groups, lambda i: function(int(i)), num_workers, name, unit="item")

    @staticmethod
    def _run_pool(groups: Sequence[Sequence[int]], call_one: Callable[[int], Any],
                  num_workers: int, name: str, *, unit: str = "block") -> List[Any]:
        """Run ``call_one(id)`` for each id in a thread pool, ordered, re-raising failures.

        The schedulable unit is a *group*: the ids in a group are run sequentially within one
        worker thread, while distinct groups run concurrently. Singleton groups reproduce the
        one-future-per-id behavior; multi-id groups serialize same-shard writes (see
        :func:`bioimage_py.util.group_blocks_by_shard`).

        Args:
            groups: The work groups; each is a list of ids (block ids or item indices) run
                sequentially. Results are returned in flattened ``groups`` order.
            call_one: The per-id callable returning that id's result.
            num_workers: Number of worker threads.
            name: A short name for the progress bar (disabled when empty).
            unit: The noun used in the failure message ("block" or "item").

        Returns:
            The per-id results in flattened ``groups`` order.

        Raises:
            RunnerError: If any id fails; the failed ids are attached for re-running. When an
                id in a group fails, the remaining (un-run) ids of that group are reported as
                failed too, since later same-shard writes cannot safely proceed.
        """
        groups = [list(g) for g in groups]
        flat_ids = [i for g in groups for i in g]
        result_by_id: Dict[int, Any] = {}
        failed: List[int] = []
        first_error: Optional[BaseException] = None

        @threadpool_limits.wrap(limits=1)
        def _run_group(group: List[int]):
            local: Dict[int, Any] = {}
            local_failed: List[int] = []
            err: Optional[BaseException] = None
            for k, bid in enumerate(group):
                try:
                    local[bid] = call_one(bid)
                except Exception as error:  # noqa: BLE001 - we re-raise as RunnerError
                    err = error
                    local_failed = list(group[k:])
                    break
            return local, local_failed, err

        with futures.ThreadPoolExecutor(max(1, int(num_workers))) as tp:
            fut_to_group = {tp.submit(_run_group, g): g for g in groups}
            with tqdm(total=len(flat_ids), desc=name or None, disable=not name) as pbar:
                for fut in futures.as_completed(fut_to_group):
                    group = fut_to_group[fut]
                    local, local_failed, err = fut.result()
                    result_by_id.update(local)
                    if local_failed:
                        failed.extend(local_failed)
                        if first_error is None:
                            first_error = err
                    pbar.update(len(group))

        if failed:
            failed = sorted(set(failed))
            raise RunnerError(
                f"{len(failed)} {unit}(s) failed in '{name or 'run'}': "
                f"{failed[:10]}. First error: {first_error!r}",
                failed_block_ids=failed,
            )
        return [result_by_id[i] for i in flat_ids]
