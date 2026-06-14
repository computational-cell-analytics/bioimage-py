"""Runner base class, the backend-independent ``run`` logic, and the local runner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent import futures
from typing import Any, Callable, List, Optional, Sequence, Tuple

from bioimage_cpp.utils import Blocking
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..sources.base import Source
from ..sources.dispatch import SourceLike, as_source
from ..util import ComputeFn, derive_block_shape, get_blocking, normalize_halo
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

        Returns:
            The list of per-block return values (in ``block_ids`` order) if
            ``has_return_val``, else ``None``.
        """
        inputs = [as_source(i) for i in inputs]
        outputs = [as_source(o) for o in outputs]
        mask_source = as_source(mask) if mask is not None else None

        domain = inputs[0] if inputs else (outputs[0] if outputs else None)
        if domain is None:
            raise ValueError("run() requires at least one input or output source.")

        # Shape consistency: all inputs and the mask must match the domain shape.
        for src in inputs + ([mask_source] if mask_source is not None else []):
            if tuple(src.shape) != tuple(domain.shape):
                raise ValueError(
                    f"Shape mismatch: source with shape {src.shape} does not match the "
                    f"domain shape {domain.shape}."
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

    @staticmethod
    def _validate_write_safety(outputs: Sequence[Source], block_shape: Sequence[int]) -> None:
        """Conservative guard: chunked output write-blocks must be a multiple of chunks.

        This prevents two blocks from concurrently writing the same chunk (which would
        corrupt it). Auto-derivation of a safe block shape is a flagged TODO.
        """
        for out in outputs:
            chunks = out.chunks
            if chunks is None or len(chunks) != len(block_shape):
                continue
            for bs, ch in zip(block_shape, chunks):
                if bs % ch != 0:
                    raise ValueError(
                        f"Unsafe block shape for writing: {tuple(block_shape)} is not a multiple "
                        f"of the output chunk shape {tuple(chunks)}. Concurrent writes could "
                        "corrupt shared chunks; use a block shape that is a chunk multiple."
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
        results: list = [None] * len(block_ids)
        failed: List[int] = []
        first_error: Optional[BaseException] = None

        @threadpool_limits.wrap(limits=1)
        def _run(idx: int):
            bid = block_ids[idx]
            return idx, run_block(function, blocking, bid, inputs, outputs, mask, halo)

        with futures.ThreadPoolExecutor(max(1, int(num_workers))) as tp:
            fut_to_idx = {tp.submit(_run, idx): idx for idx in range(len(block_ids))}
            for fut in tqdm(futures.as_completed(fut_to_idx), total=len(block_ids),
                            desc=name or None, disable=not name):
                idx = fut_to_idx[fut]
                try:
                    i, res = fut.result()
                    results[i] = res
                except Exception as error:  # noqa: BLE001 - we re-raise as RunnerError
                    failed.append(block_ids[idx])
                    if first_error is None:
                        first_error = error

        if failed:
            raise RunnerError(
                f"{len(failed)} block(s) failed in '{name or 'run'}': "
                f"{sorted(failed)[:10]}. First error: {first_error!r}",
                failed_block_ids=sorted(failed),
            )
        return results
