# bioimage-py

Efficient, parallel, and distributed implementation of image analysis and segmentation functionality.

## Installation

```bash
python -m pip install -e .
```

## Usage

Operations run block-wise and share a common interface: pass `block_shape` and `num_workers` for
parallel local execution, or `job_type="slurm"` to run distributed (one task per
block). For distributed runs the `output` must be a file-backed (zarr/n5) array.

### `copy` — block-wise copy of one source into another

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

### `downsample` — block-wise downsampling by an integer factor

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
