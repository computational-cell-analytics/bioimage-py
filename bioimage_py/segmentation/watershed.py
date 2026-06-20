"""Block-wise seeded watershed (single-stage, halo-based).

Ports ``elf.parallel.seeded_watershed``: the user supplies a height map and pre-computed integer
seed markers, and each block runs ``bioimage_cpp.segmentation.watershed`` over a halo-padded region,
writing back the (halo-free) inner block. Seed ids are an *input* -- they are preserved verbatim,
with no cross-block merge or global offsetting -- so, unlike the multi-stage ``label``, this is a
single-stage operation that supports ``block_ids`` / ``resume_from`` re-runs.

Because each block grows its catchment basins only within its halo-padded region, the block-wise
result equals a whole-array watershed only where the halo covers the relevant basins; pick a halo
large enough for the object extents at block boundaries. What is always exact is backend determinism:
for a fixed ``(block_shape, halo)`` every backend produces bit-identical output.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from ..util import BlockDescriptor, check_rerun_args, to_roi

__all__ = ["watershed"]

# Dtypes that bic.segmentation.watershed accepts directly (no copy needed).
_HMAP_DTYPES = (np.float32, np.float64)
_SEED_DTYPES = (np.uint32, np.uint64, np.int32, np.int64)


def _full_roi(ndim: int) -> Tuple[slice, ...]:
    """Return a slicing that selects the whole array."""
    return tuple(slice(None) for _ in range(ndim))


def _as_hmap(data: np.ndarray) -> np.ndarray:
    """Cast a height map to float32 unless it is already a float type."""
    return data if data.dtype in _HMAP_DTYPES else data.astype("float32")


def _as_seeds(data: np.ndarray) -> np.ndarray:
    """Cast seeds to uint32 unless they are already an integer type watershed accepts."""
    return data if data.dtype in _SEED_DTYPES else data.astype("uint32")


def _watershed_block(block: BlockDescriptor, inputs: Sequence[Source], outputs: Sequence[Source],
                     mask: Optional[Source]) -> None:
    """Per-block seeded watershed: read the halo'd region, run watershed, write the inner block."""
    hmap_src, seeds_src = inputs[0], inputs[1]
    out_src = outputs[0]
    outer = to_roi(block.outer_block)
    inner = to_roi(block.inner_block)
    inner_local = to_roi(block.inner_block_local)

    if mask is not None:
        block_mask = mask[outer].astype(bool)
        if not block_mask[inner_local].any():
            return None
    else:
        block_mask = None

    block_hmap = _as_hmap(hmap_src[outer])
    block_seeds = _as_seeds(seeds_src[outer])
    ws = bic.segmentation.watershed(block_hmap, block_seeds, mask=block_mask)
    out_src[inner] = ws[inner_local]
    return None


def watershed(
    input: SourceLike,
    seeds: SourceLike,
    output: Optional[SourceLike] = None,
    *,
    halo: Optional[Sequence[int]] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    num_workers: int = 1,
    mask: Optional[SourceLike] = None,
    block_ids: Optional[Sequence[int]] = None,
    resume_from: Optional[str] = None,
) -> SourceLike:
    """Compute a seeded watershed over a height map, block-wise.

    Each block runs a seeded watershed (``bioimage_cpp.segmentation.watershed``) on a halo-padded
    region and writes back the halo-free inner block. The ``seeds`` define the segments and their
    ids are preserved verbatim -- there is no cross-block merge, so pass globally consistent seeds
    (e.g. a connected-component labeling of a seed mask) for a coherent result.

    Being single-stage, this operation supports ``block_ids`` / ``resume_from`` re-runs. The
    block-wise output matches a whole-array watershed only where ``halo`` covers the relevant
    catchment basins, but is bit-identical across backends for a fixed ``(block_shape, halo)``.

    Args:
        input: The height map (a numpy/zarr/n5 array or a `Source`). It is cast to ``float32`` for
            the watershed if it is not already a float type.
        seeds: The pre-computed integer seed markers (``0`` = background), same shape as ``input``.
            A non-integer seed map is cast to ``uint32``.
        output: The integer output array to write the segmentation into. Optional for local
            execution -- a ``uint64`` numpy array is allocated and returned if omitted; **required**
            for distributed execution. It must be wide enough to hold the seed ids (``uint64``
            recommended; a ``uint32`` watershed result writes losslessly into a ``uint64`` output).
        halo: Per-axis halo enlarging each block; **required** for the block-wise path (there is no
            principled default for a watershed). Choose it large enough to cover object extents at
            block boundaries. Ignored by the direct (single-block) path.
        block_shape: Shape of the processing blocks. Defaults to the input chunk shape; required
            for unchunked data.
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed
            backends).
        mask: Optional binary mask; voxels outside the mask are excluded and stay ``0``.
        block_ids: Restrict processing to these block ids (e.g. to re-run previously failed blocks
            into the existing ``output``). Mutually exclusive with ``resume_from``.
        resume_from: Distributed only; the preserved temp folder of a failed run to resume (see
            ``runner.run``); the missing blocks are written into ``output``. Mutually exclusive
            with ``block_ids``.

    Returns:
        The output array (the provided ``output``, or a newly allocated ``uint64`` numpy array).
    """
    check_rerun_args(job_type, resume_from, block_ids)

    src = as_source(input)
    seeds_src = as_source(seeds)
    ndim = src.ndim
    if tuple(seeds_src.shape) != tuple(src.shape):
        raise ValueError(
            f"seeds shape {seeds_src.shape} does not match input shape {src.shape}."
        )

    # A subset/resume rerun is inherently block-wise, so it cannot use the direct (whole-array) path.
    direct = (job_type == "local" and num_workers == 1 and block_shape is None
              and block_ids is None and resume_from is None)

    if output is None:
        if job_type != "local":
            raise ValueError(
                f"'output' is required for distributed execution (job_type={job_type!r}); "
                "pass a file-backed (zarr/n5) output array."
            )
        out_array: SourceLike = np.zeros(tuple(src.shape), dtype="uint64")
    else:
        out_array = output

    out = as_source(out_array)
    if not np.issubdtype(out.dtype, np.integer):
        raise ValueError(f"output must have an integer dtype, got {out.dtype}.")

    if direct:
        block_hmap = _as_hmap(src[_full_roi(ndim)])
        block_seeds = _as_seeds(seeds_src[_full_roi(ndim)])
        block_mask = as_source(mask)[_full_roi(ndim)].astype(bool) if mask is not None else None
        out[_full_roi(ndim)] = bic.segmentation.watershed(block_hmap, block_seeds, mask=block_mask)
        return out_array

    if halo is None:
        raise ValueError(
            "halo is required for block-wise watershed; choose one large enough to cover object "
            "extents at block boundaries."
        )

    runner = get_runner(job_type, job_config)
    runner.run(_watershed_block, [input, seeds], outputs=[out_array], halo=halo,
               block_shape=block_shape, mask=mask, num_workers=num_workers,
               block_ids=block_ids, resume_from=resume_from, name="watershed")
    return out_array
