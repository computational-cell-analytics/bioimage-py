# Usage

Operations run block-wise and share a common interface: pass `block_shape` and `num_workers` for
parallel local execution, or `job_type="slurm"` to run distributed (one task per
block). For distributed runs the `output` must be a file-backed (zarr/n5) array.

## `copy` — block-wise copy of one source into another

Useful for converting between storage formats (e.g. a tiff stack to zarr) or for persisting an
on-the-fly wrapper transformation to file.

```python
import zarr
import bioimage_py as bp

# Convert a tiff stack (single multi-page file, or a folder of slices via bp.open_source(folder, "*.tif"))
# to a chunked zarr array.
src = bp.open_source("stack.tif")
out = zarr.open_array("out.zarr", mode="w", shape=src.shape, dtype=src.dtype, chunks=(64, 64, 64))
bp.copy(src, out, block_shape=(64, 64, 64), num_workers=8)

# Persist a wrapper (here a threshold) to file instead of recomputing it on every read.
from bioimage_py.wrapper import ThresholdSource
mask = zarr.open_array("mask.zarr", mode="w", shape=src.shape, dtype="bool", chunks=(64, 64, 64))
bp.copy(ThresholdSource(src, 128), mask, block_shape=(64, 64, 64), num_workers=8)

# Distributed: output must be file-backed (zarr/n5).
bp.copy(src, out, block_shape=(64, 64, 64), num_workers=8, job_type="slurm")
```

If `output` is omitted, a numpy array is allocated and returned (local execution only).

## `downsample` — block-wise downsampling by an integer factor

Defaults are label-safe (`order=0` nearest, no anti-aliasing). For intensity/image data pass
`order=1` (or higher) and `anti_aliasing=True` for a smooth, alias-free result.

```python
import zarr
import bioimage_py as bp

# Image data: smooth, anti-aliased 2x downsample into a new zarr array.
raw = zarr.open_array("raw.zarr", mode="r")
target = tuple(s // 2 for s in raw.shape)
out = zarr.open_array("raw_s1.zarr", mode="w", shape=target, dtype=raw.dtype, chunks=(64, 64, 64))
bp.downsample(raw, 2, out, order=1, anti_aliasing=True, block_shape=(64, 64, 64), num_workers=8)

# Label data: keep the defaults so no label ids are invented. Returns a numpy array when no output given.
seg = zarr.open_array("seg.zarr", mode="r")
small = bp.downsample(seg, 2)

# Anisotropic factor (downsample y/x only): bp.downsample(raw, (1, 2, 2), out, ...)
```

The downscaled shape is computed with `bioimage_py.util.downscale_shape` (ceil mode); under the hood
`downsample` wraps the input in a `bioimage_py.wrapper.ResizedSource` and copies it block-wise.

## Re-running failed blocks

A distributed run that loses some blocks (a transient node failure, an out-of-memory kill, a slurm
timeout) raises a `RunnerError`. Each worker persists progress per block, so the error reports the
*precise* `failed_block_ids` (only the blocks that did not complete, not the whole task) and, for
distributed backends, the preserved `tmp_folder` — the completed work is not thrown away.

```python
import bioimage_py as bp
from bioimage_py.runner import RunnerError

try:
    bp.filters.gaussian_smoothing(raw, 2.0, output=out, block_shape=(64, 64, 64),
                                  num_workers=64, job_type="slurm")
except RunnerError as e:
    print(e.failed_block_ids)  # e.g. [128, 129, 511]
    print(e.tmp_folder)        # /shared/tmp/bioimage_py_xxxx  (preserved for resume/debug)
```

**Recommended — `resume_from`** (distributed only). Re-issue the *same* call pointing at the
preserved temp folder: only the incomplete blocks are re-run, and the result is merged with the
blocks that already finished. This is correct for array-output ops (the missing blocks are written)
*and* return-value ops (`stats.mean`, `morphology.morphology`, …), which reduce over the full merged
set:

```python
bp.filters.gaussian_smoothing(raw, 2.0, output=out, block_shape=(64, 64, 64),
                              num_workers=64, job_type="slurm", resume_from=e.tmp_folder)
```

`resume_from` resumes from the original run's serialized payload, so pass it to *finish the same
call* — the input/output/parameters on the resuming call are ignored in favour of the originals.

**Simpler — `block_ids`** (a fresh re-run of just those blocks). For array-output and other
per-block-independent ops you can re-run the reported blocks directly; this works on every backend,
including `local`:

```python
bp.copy(src, out, block_shape=(64, 64, 64), num_workers=8, job_type="slurm",
        block_ids=e.failed_block_ids)
```

`resume_from` and `block_ids` are mutually exclusive. Two ops differ: `segmentation.label` has a
global cross-block merge, so a failed `label` is re-run **whole** (it accepts neither argument);
`morphology.regionprops` re-runs per object via `item_ids` / `resume_from`. A `local` run keeps no
temp folder, so re-run it (optionally with `block_ids=e.failed_block_ids`); `resume_from` is rejected
for `job_type="local"`.
