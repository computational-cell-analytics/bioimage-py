# Usage

Operations run block-wise and share a common interface: pass `block_shape` and `num_workers` for
parallel local execution, or `job_type="slurm"` to run distributed (one task per
worker by default). For distributed runs the `output` must be a file-backed (zarr/n5) array.

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
    print(e.failed_block_count)
    print(list(e.iter_failed_block_ids())[:10])
    print(e.tmp_folder)        # /shared/tmp/bioimage_py_xxxx  (preserved for resume/debug)
    for failure in e.task_failures:
        print(failure.task_id, failure.scheduler_state, failure.exit_code)
        print(failure.stderr_path, failure.traceback_path)
```

`task_failures` contains one immutable `TaskFailure` record for each failed distributed task.
Subprocess records include the exit code or terminating signal, elapsed time, and log paths. Slurm
records also include the scheduler task ID, scheduler state, peak resident memory, and failed node
when the cluster provides these values. An unavailable value is `None`.

The preserved folder contains a versioned `manifest.json`. Its `attempts` list keeps each launch,
resume, and Slurm submission failure. Attempt-specific directories keep logs, tracebacks, outcomes,
timings, and Slurm accounting observations. Successful runs still remove the folder after the
optional `pre_cleanup` callback runs.

The manifest also stores a compact work plan and deterministic task assignments. Default item and
block ranges do not create one manifest entry per item. Explicit ID sequences use JSON when small
and a memory-mapped int64 file when large.

**Recommended — `resume_from`** (distributed only). Re-issue the *same* call pointing at the
preserved temp folder: only incomplete work is re-run, and the result is merged with completed
work. This works for array-output operations, ordered return operations
such as `morphology.morphology`, and bounded reducer operations such as `stats.mean`. Ordered
returns merge the complete result list. Reducers reuse one durable accumulator per completed batch:

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
the DataFrame `morphology.regionprops` path re-runs per object via `item_ids` / `resume_from`. The
file-backed morphology paths resume durable table batches instead. A `local` run keeps no temp
folder, so re-run it (optionally with `block_ids=e.failed_block_ids`); `resume_from` is rejected for
`job_type="local"`.

Rerun and resume assume that inputs have not changed in place. Array identities describe the
source, shape, and data type, but they do not hash array values. Raw Parquet identities use file
metadata. After an in-place input change, use a new output path. For table-producing morphology
calls, you can also change `provenance` to create a new dataset identity.

## Batch mapping

Use `Runner.map_batches` when one function call can process several logical items. The runner passes
a frozen `Batch` with `batch_id`, `start`, `stop`, and `step` fields. Batch boundaries stay the same
when the worker or task count changes.

```python
from bioimage_py.runner import Batch, get_runner

def write_part(batch: Batch):
    rows = compute_rows(range(batch.start, batch.stop))
    write_batch_file(batch.batch_id, rows)

runner = get_runner("slurm", cfg)
runner.map_batches(write_part, n_items=10_000_000, batch_size=100_000,
                   num_workers=64, has_return_val=False)
```

The completion and retry unit is one batch. Distributed workers write one 16-byte completion record
after a batch succeeds. Progress reports logical items. A failed call exposes
`RunnerError.failed_batches`. Use `resume_from` to process only incomplete batches with the original
assignments.

Set `has_return_val=True` only when you need an ordered in-memory result for each batch. Ordered
results use memory proportional to the batch count.

Pass `batch_boundaries` instead of `n_items` and `batch_size` for deterministic irregular batches.
The boundaries must start at zero and increase strictly. The runner stores them in the run manifest,
so resume and reattach use the exact original plan.

```python
runner.map_batches(write_part, batch_boundaries=[0, 20_000, 75_000, 100_000],
                   num_workers=64, has_return_val=False)
```

## Associative reductions

Pass a `Reducer` to combine small logical results without collecting one result per item. A reducer
defines `initial`, `update`, `merge`, and `finalize`. Merge must be associative. The runner preserves
logical order, so merge does not need to be commutative. Keep reducer state in the accumulator. A
local runner can call the same reducer object concurrently.

```python
from bioimage_py.runner import Reducer, get_runner

class SumReducer:
    def initial(self):
        return 0

    def update(self, accumulator, value):
        return accumulator + value

    def merge(self, left, right):
        return left + right

    def finalize(self, accumulator):
        return accumulator

reducer: Reducer[int, int, int] = SumReducer()
total = get_runner("slurm", cfg).map(
    compute_value,
    n_items=10_000_000,
    num_workers=64,
    has_return_val=True,
    reducer=reducer,
    reduction_batch_size=1_000,
)
```

Each distributed batch writes one atomic accumulator under the runner folder. The worker records
completion only after the accumulator is durable. Resume validates the accumulator files and runs
only incomplete batches. Finalization reads accumulators in batch order and keeps memory bounded by
the accumulator size.

`Runner.run` and `Runner.map` default to 1,000 logical results per reducer batch. Reducer mode is
read-only and does not accept output arrays. Keep the ordered return path when an operation must
write arrays or return every logical result.

Use a `TableDataset` result sink for large typed tables. The batch function receives a
`TablePartWriter`, writes one Arrow table, and returns `None`.

```python
import pyarrow as pa

from bioimage_py import TableDataset
from bioimage_py.runner import Batch, get_runner

schema = pa.schema([("label", pa.uint64()), ("size", pa.uint64())])
dataset = TableDataset.create(
    "morphology.parquet",
    schema=schema,
    schema_version=1,
    operation="example-morphology",
    operation_version="1",
    input_identities={"segmentation": {"path": "/shared/segmentation.zarr"}},
    parameters={"resolution": [8.0, 8.0, 8.0]},
)

def write_table(batch: Batch, writer):
    columns = compute_columns(range(batch.start, batch.stop))
    writer.write(pa.table(columns, schema=schema))

runner = get_runner("slurm", cfg)
result = runner.map_batches(
    write_table,
    n_items=10_000_000,
    batch_size=100_000,
    num_workers=64,
    result_sink=dataset,
)
```

The output path is a dataset directory, even when it ends in `.parquet`. Each batch writes one
atomic Parquet part and one completion sidecar. A compatible fresh call reuses valid parts. Resume
also validates the parts and recomputes missing or invalid batches.

The runner returns a lightweight `TableDataset`. Use `result.iter_parts()` for part metadata. Call
`result.to_pandas()` only when you explicitly want to materialize the complete table. Runner cleanup
never removes the table dataset.

### File-backed base morphology

Set `output_table` to compute base morphology without returning all block tables to the
orchestrator. The segmentation and optional mask must be reopenable sources.

```python
base = bp.morphology.morphology(
    segmentation,
    output_table="morphology-base.parquet",
    block_shape=(64, 256, 256),
    blocks_per_batch=1_000,
    label_partition_size=1_000_000,
    num_workers=64,
    job_type="slurm",
    job_config=cfg,
    provenance={"block_mask_fingerprint": mask_fingerprint},
)
```

The first stage writes exact, sorted sufficient statistics for bounded batches of image blocks.
The second stage reduces each populated label range independently. Labels and internal sums remain
`uint64`; the final table uses `int64` for sizes and bounding boxes and `float64` for centers of
mass. Sparse and high-valued labels do not pass through floating-point values.

`blocks_per_batch` sets the durable retry unit for the first stage. `label_partition_size` bounds
the label arrays allocated by each reducer. Final parts include `label_start` and `label_stop`
partition metadata. A compatible call reuses completed parts. Use `resume_from` with the preserved
distributed runner folder to resume the failed stage.

The function retains `<output_table>.morphology-work` after a failure. This directory contains the
partial table and disk-backed label index. It removes the directory only after the final dataset
passes validation. The returned `TableDataset` has the same columns and bounding-box conventions as
the DataFrame path.

### File-backed `regionprops`

Set `output_table` to process a large base morphology table in bounded row batches. This path
accepts a completed `TableDataset`, a Parquet file, or a directory of Parquet files. It requires a
reopenable segmentation source for all execution backends.

```python
features = bp.morphology.regionprops(
    segmentation,
    "morphology-base.parquet",
    resolution=(40.0, 4.0, 4.0),
    compute_surface=False,
    output_table="regionprops.parquet",
    rows_per_batch=100_000,
    target_batch_cost=2_000_000_000,
    num_workers=64,
    job_type="slurm",
    job_config=cfg,
    provenance={"segmentation_revision": "final-v1"},
)
```

The function returns a `TableDataset`. The `output_table` path is a directory, including when its
name ends in `.parquet`. Each output part contains one input-row batch. Parts and rows retain input
order. Raw Parquet directories use lexical file order.

`target_batch_cost` is optional. When set, `regionprops` estimates each row's cost from its
bounding-box voxel count and packs contiguous rows up to that target. `rows_per_batch` remains a hard
row limit. An object whose bounding box exceeds the target gets a one-row batch. Planning reads only
the bounding-box columns in bounded chunks. Compatible reruns and resumes reuse the stored
boundaries without rescanning the input table.

The file-backed schema uses `uint64` for `label` and `n_voxels`, `int64` for bounding boxes, and
`float64` for measurements. It adds `surface_area` only for a 3D input when
`compute_surface=True`. Use `features.to_pandas()` only when the complete result fits in memory.

A compatible fresh call validates and skips completed parts. Use `resume_from` to resume a failed
distributed run with its original task assignments. The file-backed path does not accept
`item_ids`; its retry unit is one batch.

## Slurm configuration

Slurm settings are cluster- and user-specific (partition, account, qos, node constraint, the shared
`tmp_root`, ...). Pass them per call as a `SlurmConfig`:

```python
from bioimage_py import SlurmConfig

cfg = SlurmConfig(tmp_root="/scratch/shared/me", partition="gpu", account="myproj", time="01:00:00")
bp.copy(src, out, block_shape=(64, 64, 64), num_workers=64, job_type="slurm", job_config=cfg)
```

To avoid repeating these every time, store them once as user defaults in
`~/.config/bioimage-py/config.toml` (honoring `$XDG_CONFIG_HOME`). Use the helper rather than
editing the file by hand — it validates field names and preserves the rest of the file:

```python
from bioimage_py import write_slurm_config

write_slurm_config(tmp_root="/scratch/shared/me", partition="gpu", account="myproj")
```

These defaults are picked up automatically whenever a slurm run gets no explicit `job_config`
(e.g. `bp.copy(..., job_type="slurm")`). To combine the stored defaults with per-run tweaks, use
`SlurmConfig.load(**overrides)` (overrides win); a directly constructed `SlurmConfig(...)` is used
verbatim and does **not** read the file. Set `BIOIMAGE_PY_NO_CONFIG=1` to ignore the file
(reproducible CI), or `BIOIMAGE_PY_CONFIG=/path/to/config.toml` to point at a specific file (e.g. a
shared cluster-wide config).
