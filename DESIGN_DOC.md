# Design document

The goal of this repository is to make image analysis algorithms scalable --- and seamlessly switch between local and distributed execution on HPC or other backends.

This project is meant as an evolution of [elf](https://github.com/constantinpape/elf) and [cluster-tools](https://github.com/constantinpape/cluster_tools), providing similar functionality but in a significantly more scalable, convenient, and robust fashion.
To implement the underlying algorithms use standard scientific Python libraries (numpy, pandas, scipy, scikit-image, etc.) and [bioimage-cpp](https://github.com/computational-cell-analytics/bioimage-cpp). Prefer using the latter if it offers the required functionality.

The algorithms we implement are generally block-wise processing of (large) image data.
They should keep a fall-back for direct application of the respective algorithm so that the same function can be used for 'every day tasks'.
See details below.

## Sources / data specification

A `Source` is the unit of data the runner reads from and writes to. It is a thin, **serializable** description of how to obtain (or reopen) an array, plus array-like access:

- `__getitem__(index)` / `__setitem__(index, value)` — read/write a region of interest. The index is a numpy-style basic index (int / slice / ellipsis / tuple); the base class normalizes it to a full tuple of in-bounds slices and squeezes integer-indexed axes, then delegates to each source's `_getitem(roi)` / `_setitem(roi, value)`, which always receive a full tuple of slices.
- `shape`, `dtype`, and (where applicable) `chunks` and `shards` — metadata needed for blocking and write-alignment.
- `to_spec()` / `from_spec(spec)` — a serializable round-trip (path, internal path, storage options, …) so a worker on another node can reopen the same data. For distributed jobs this is what gets shipped, not the live handle.

**Users should rarely construct a `Source` themselves.** The public functions accept a `SourceLike`, and the runner converts it internally via a dispatch function:

```python
SourceLike = Union[Source, np.ndarray, "zarr.Array", "z5py.Dataset", ...]

def as_source(obj: SourceLike) -> Source:
    """Convert a supported object into a Source. Idempotent on Source inputs.

    Dispatches on type via a registry. Third parties can register converters for
    their own array types. A Source passed in is returned unchanged, which is the
    escape hatch for full control (custom storage options, S3 credentials, wrappers).
    """
```

This keeps the common case trivial — the user passes a zarr/n5 array and never sees `Source` — while still allowing power users to pass an explicit `Source` when they need to.

Conversion rules of note:
- A `numpy.ndarray` converts to an in-memory source. Its `to_spec()` raises a clear error, since an in-memory array cannot be reopened on another node. This is what enforces the "numpy arrays are local-execution only" rule, with an actionable message.
- Passing strings / file paths is currently not supported (it may be added later). A bare path is ambiguous: a zarr/n5 container typically holds multiple internal arrays (and groups / resolution levels), so "open the array at this path" is underspecified. Until we decide how to resolve that, callers open the array themselves and pass the handle.

Wrappers (on-the-fly transforms) are `Source`s that wrap another `Source`; see the wrapper section below.

## Runner

The core element is a flexible runner implementation that can dispatch to local execution (a thread pool), to local subprocesses (the `subprocess` backend), or to a distributed scheduler (slurm; more may be added). It is implemented with a `Runner` class hierarchy (an abstract base class + implementations for the different runners). The `subprocess` backend runs the *full* distributed protocol — cloudpickle payload, generated harness, per-task result/sentinel files — but launches tasks locally with no scheduler, so the local/distributed parity can be tested in CI and slurm becomes a thin layer on top (see "Implementing the slurm runner").

### Calling convention for the per-block function

The runner calls the per-block function defined for the specific algorithm with a fixed keyword signature, so there is no fragile dependency on argument count or order:

```python
def _compute(block, inputs, outputs, mask):
    # block:   a block descriptor from bioimage_cpp.utils -- a Block, or a BlockWithHalo
    #          when the operation requested a halo (see below).
    # inputs:  tuple of opened Sources (read).
    # outputs: tuple of opened Sources (write); empty when there is nothing to write.
    # mask:    an opened Source or None.
    ...
```

The block descriptor comes from `bioimage_cpp.utils` (the blocking utilities `Blocking`, `Block`, `BlockWithHalo`). The runner builds a `Blocking` over the array and, per block id, hands `_compute` one of two descriptors depending on whether the operation requested a halo:

- **No halo** (reductions like `max`, per-pixel ops): `block` is a `Block`, carrying the coordinate lists `block.begin`, `block.end`, `block.shape`, `block.ndim`. Read and write this one region.
- **With halo** (filters, distance transforms, watershed, connected components): `block` is a `BlockWithHalo` exposing three `Block`s — `block.outer_block` (the read region, extended by the halo and clipped to the array), `block.inner_block` (the write region, no overlap, in global coordinates), and `block.inner_block_local` (the inner block expressed relative to the outer block, used to crop the result before writing). The pattern is: read `inputs[i][to_roi(block.outer_block)]`, compute on the padded array, then `outputs[j][to_roi(block.inner_block)] = result[to_roi(block.inner_block_local)]`.

A `Block` has no `roi`/slice member — it carries `begin`/`end` coordinate lists. While a `Source` does accept numpy-style basic indices (normalized to slices internally), it does **not** accept a `Block`, so the compute function converts the block explicitly with a small helper `to_roi(block) -> Tuple[slice, ...]` (`tuple(slice(b, e) for b, e in zip(block.begin, block.end))`) and indexes with the result, e.g. `input_[to_roi(block)]`. Keeping the conversion in the compute function (rather than overloading `Source` to accept a `Block`) keeps it explicit which region (outer / inner / inner-local) is being indexed, which matters for halo ops where the same function touches all three.

There are two **orthogonal** output channels, and a given function may use either or both:
- **Output sources** (`outputs`): large array results written in place to storage. These never travel back to the master.
- **Return value** (`has_return_val=True`): a *small* value per block (e.g. a scalar or a tiny array) that the runner collects and the caller reduces locally. The local reduction must be associative and commutative, since block order is not guaranteed.

### Runner configuration

`job_config` is a typed dataclass rather than an opaque dict, so it can be validated and documented:

```python
@dataclass
class RunnerConfig:
    poll_interval: float = 10.0  # seconds between status polls (distributed runners).

@dataclass
class SlurmConfig(RunnerConfig):
    partition: Optional[str] = None
    time: Optional[str] = None
    mem: Optional[str] = None
    cpus_per_task: int = 1
    gpus: int = 0
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    # How to set up the environment in the generated job script (shebang / activation).
    # The package and all dependencies must be importable on the compute node.
    shebang: Optional[str] = None
```

### Serializing the computation

The per-block function and its (small) bound arguments are serialized with **`cloudpickle`**. cloudpickle captures the function *by value*, including its closure (free variables, module-level helpers, constants), so locally-defined functions just work as long as the libraries they reference are installed in the worker environment (they are — it is the same package/env).

For debuggability we write a human-readable artifact next to the payload: a best-effort `inspect.getsource(fn)` dump (and the call metadata) in the job temp folder. Correctness never depends on it; it is only there to make a failed job easy to inspect.

To guard against version skew, the payload is stamped with the Python (major, minor) version, validated by the worker before running so a mismatch fails loudly and early. (Stamping a fuller environment/library hash is a possible later hardening.)

### Validation in `run()`

Before dispatching, `run()` validates the job and fails early with actionable errors:

- **Shape consistency.** The mask and all inputs must describe the same shape (so a single block indexes them consistently). A mask at a different resolution is brought onto the input grid with an up-/down-sampling wrapper that *reports the wrapped (effective) shape* — so the check is simply "do the reported shapes match"; it does not special-case resolution. Outputs are allowed to differ in shape from the inputs (e.g. a downsampled output), and are blocked on their own grid.
- **Write safety (implemented as a conservative guard).** When there are `outputs`, `run()` must guarantee concurrency-safe writes. zarr/n5 are safe for concurrent writes to *different* chunks / shards but corrupt on concurrent writes to the *same* chunk / shard. The rule: the output write-blocks (`block.inner_block` under a halo, otherwise the block itself) must align to the output's chunk grid so that each chunk is written by exactly one block. `run()` currently validates that, for any chunked output, the block shape is a multiple of the output chunk shape, and raises otherwise. Auto-deriving a safe block shape (instead of raising) remains a future improvement.

### Optimization: skipping empty blocks (later)

For a sparse mask, many blocks are fully outside the mask and produce nothing. Pre-computing a block↔mask coverage map and only scheduling the non-empty blocks can be a large speedup. This is a *possible later optimization*, not part of the initial design, and it is not quite trivial: computing the coverage is itself a block-wise pass over the (potentially large) mask, so on slurm it may require submitting an extra preparatory job before the main array job (or folding coverage into a prior stage's output). Worth noting now so the block-scheduling logic leaves room for a pre-filter on `block_ids`.

### Example: computing the max

Below is a simple example for how the implementation for computing the max could look based on this design.

```python
import numpy as np


# This defines the per-block computation. `block` is a bioimage_cpp.utils.Block (no halo needed);
# to_roi turns it into a tuple of slices that indexes the sources.
def _compute(block, inputs, outputs, mask):
    input_ = inputs[0]
    roi = to_roi(block)
    if mask is None:
        return np.max(input_[roi])
    block_mask = mask[roi].astype(bool)
    mask_sum = block_mask.sum()
    if mask_sum == 0:  # Nothing is in the mask -> return early, without ever reading the input.
        return None
    if mask_sum == block_mask.size:  # Everything is in the mask, we don't need to apply it.
        return np.max(input_[roi])
    return np.max(input_[roi][block_mask])


def max(
    input: SourceLike,  # Any source-like object; numpy arrays restrict execution to local.
    num_workers: int = 1,  # Number of parallel workers.
    block_shape: Optional[Tuple[int, ...]] = None,  # Block shape; if None and single worker, compute directly.
    job_type: str = "local",  # 'local', 'subprocess', or 'slurm'.
    job_config: Optional[RunnerConfig] = None,  # Runner configuration, especially for slurm.
    mask: Optional[SourceLike] = None,  # Optional binary mask; only values within the mask are considered.
    block_ids: Optional[Sequence[int]] = None,  # Restrict to these blocks; relevant for re-running failed jobs.
) -> np.float64:
    """Compute the maximum value of an array, optionally restricted to a mask.

    Args:
        input: The input data.
        num_workers: The number of parallel workers.
        block_shape: The block shape for block-wise processing.
        job_type: The execution backend, 'local' or 'slurm'.
        job_config: Backend configuration.
        mask: Optional binary mask.
        block_ids: Optional subset of blocks to process.

    Returns:
        The maximum value.
    """
    # Direct computation (no blocking) is only valid for the trivial local single-worker case
    # and does not support masks or block subsetting.
    if job_type == "local" and num_workers == 1 and block_shape is None:
        if mask is not None or block_ids is not None:
            raise ValueError("Direct computation does not support 'mask' or 'block_ids'.")
        return np.max(as_source(input)[:])

    # get_runner returns the local or slurm runner depending on job_type. The number of inputs and
    # outputs is derived from the arguments to run(), so there is no separate counts step.
    runner = get_runner(job_type, config=job_config)

    # run() validates inputs/outputs, derives the block shape if None, builds the (cloudpickled) payload,
    # provides each block's inputs/outputs/mask and the block descriptor to the function, records per-block
    # success, updates progress, and returns the collected return values (here one per non-empty block).
    results = runner.run(
        function=_compute,
        inputs=[input],
        outputs=[],
        num_workers=num_workers,
        block_shape=block_shape,
        mask=mask,
        block_ids=block_ids,
        has_return_val=True,
        name="Compute max",
    )

    # Combine the per-block results locally, dropping empty blocks (None) from masking.
    results = [res for res in results if res is not None]
    if not results:  # Everything was masked out.
        raise ValueError("No values within the mask; cannot compute a maximum.")
    return np.max(results)
```

The implementation of this is straight-forward locally (futures.ThreadPoolExecutor and tqdm prog bar).

For the distributed case (slurm) it's more complicated. Current design idea:
- The runner converts each input/output/mask to a `Source` and obtains its `to_spec()`. If a source is not reopenable (e.g. a numpy array), it fails early with a clear message.
- It creates a temporary folder on a **shared filesystem** for the job, writes a slurm job config for an array job to it, and a small harness script that runs the computation per task. The harness loads the cloudpickled payload (function + bound args + source specs), reopens the sources, gets the blocks for this task, and runs them in a loop. It persists progress **per block** but into **per-task files** (so the file count stays bounded — one done-log and one result file per task, not per block): after each block succeeds it appends the block id to the task's done-log and, for a return-value op, a length-framed result record to the task's result file; the per-task `.success` sentinel is still written once at the end. This keeps a partially-failed task's completed blocks (so a straggler/failure costs only the un-done blocks, not the whole task) and lets a run be **resumed** — re-running only the incomplete blocks (the harness skips blocks already in its done-log) and merging over all persisted per-block results. It also writes a job **manifest** (submitted array/job ids, the block→task assignment, the temp folder) so a run can be *reattached* — if the orchestrating (login-node) process dies, a later call can pick the job back up from the manifest instead of resubmitting from scratch.
- After submitting the array, the runner watches the respective slurm jobs. It polls every 10 seconds
  (by default, make param in the config), and updates the progress report based on this
  (also reports how many jobs are pending, running, have succeeded, have failed; the bar itself counts processed **blocks**, summed from the per-task done-logs). The **per-task `.success` sentinel is the ground truth for a *task* being complete**, while the per-block **done-logs are the authority for which blocks finished** (so failure reporting is precise per block, not per task). The scheduler is queried only to detect *dead* tasks — using `sacct` rather than `squeue` (squeue drops finished jobs, while sacct reports terminal states reliably: COMPLETED, FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED, CANCELLED). A task that has left the scheduler in a terminal state but has no sentinel is treated as failed; `_finalize` then reports exactly the blocks missing from the done-logs.
- After everything is done the runner checkes if computation was overall successfull. If yes, it unpickles the return values (if any),
  cleans up the temp folder, and returns the list of return vals. Otherwise it gives an error report and throws a runtime error.
  In the latter case the temp folder is not cleared to enable debugging; the error report contains the path to the temp folder and the
  `block_ids` of the failed blocks, so the job can be re-run on exactly those blocks.


## Multi-stage workflows

Some algorithms are not a single map + local reduce. Connected components (label per block → merge labels across block boundaries → relabel) and seeded watershed need information exchanged *between* blocks and are genuinely map → reduce → map.

Statistics like mean and std are **not** in this category — they are single pass. A block returns a small tuple through the return-value channel (e.g. `sum + count` for the mean, or `sum + sum_of_squares + count` for the std), and these combine associatively and commutatively in the local reduction, exactly like `max`. No intermediate storage and no second map are needed.

These are expressed simply as **multiple sequential `run()` calls**, with intermediate results persisted to a `Source` or another suitable format that lives in the job's temp folder (e.g. a temporary zarr/n5 array, or a small table for per-block summaries). Stage *N+1* reads what stage *N* wrote. This needs no task-graph machinery (in contrast to the Luigi-based cluster-tools), and does not clash with the current single-`run()` design — the high-level function just orchestrates the stages and owns the temp source's lifecycle (created before the first stage, cleaned up on overall success, preserved on failure for debugging). The same `block_ids` re-run mechanism applies per stage.

## Wrapper / On-the-fly transformation

We want to also enable on-the-fly transformations of the input data, e.g. thresholding the data. With the `Source` abstraction this no longer needs a separate mechanism or any source parsing: **a wrapper is a `Source` that wraps another `Source`** and applies the transform in its `__getitem__`. You can refer to `elf.wrapper` for  similar design and example implementations.

```python
class ThresholdSource(Source):
    def __init__(self, source: SourceLike, threshold: float):
        self._source = as_source(source)
        self._threshold = threshold

    def __getitem__(self, roi):
        return self._source[roi] > self._threshold
    # shape/dtype/chunks delegate to the wrapped source; to_spec() records the wrapped
    # spec plus the transform parameters so the worker can rebuild the wrapper.
```

Because wrappers are `Source`s, they are serialized and reopened on workers exactly like any other source — no parsing or code rebuilding. For local execution the live wrapper object is used directly; for distributed execution it is cloudpickled (or reconstructed from `to_spec()`). Building out a small set of composable, serializable wrappers (threshold, cast, rescale, channel selection, resolution/scale adaptation) is the remaining work here.

The **resolution/scale-adapting wrapper** is what makes a differently-sampled mask usable: it wraps a low- (or high-) resolution source and resamples on read so that `__getitem__(roi)` returns data on the target (input) grid, while *reporting the effective, resampled `shape`*. This is what lets `run()`'s shape-consistency check stay simple — it only compares reported shapes, so a wrapped mask that matches the input shape passes, and an unwrapped mismatched mask is rejected with a clear error rather than silently mis-indexed.

## Implementation status and insights

The first slice is implemented and tested: `stats.max/min/mean/std`, `filters.apply_filter` (+ the gaussian-family convenience functions), and `segmentation.label`, on the `local`, `subprocess` and `slurm` backends. Key decisions and insights from building it:

- **One per-block code path for all backends.** `runner.run_block(...)` builds the `Block` / `BlockWithHalo` and calls the user function; both `LocalRunner` and the worker harness call it. This is what makes the `direct == local == subprocess` parity tests meaningful — there is no separate distributed code path to drift from.
- **The `subprocess` backend is the slurm dress rehearsal.** It implements the entire distributed protocol — temp folder, cloudpickle `payload.pkl`, generated per-task work-lists, per-block persistence into per-task files (a `progress/<task>.log` done-log + length-framed `results/<task>` records, appended after each block with the result written *before* the done-line so a crash leaves at most a harmless re-runnable tail), a final `.success` sentinel, result collection (dedup by block id, merges an original run with a resume), precise per-block failure reporting with a preserved temp folder + failed `block_ids`, a block-counting progress bar, and both `block_ids` re-run and `resume_from` (resume the incomplete tasks of a preserved run). All of this lives in a shared `_DistributedRunner` base; `SubprocessRunner` overrides only `_launch_and_wait` (a local `subprocess.Popen` per task, capped at `num_workers`, accepting a `task_ids` subset for resume). The worker entry point is `python -m bioimage_py.runner._harness <tmp> <task_id>`.
- **`output` is optional for local, required for distributed.** Array-output ops (`filters.*`, `segmentation.label`) allocate and return a fresh numpy array when `output` is omitted *and* the backend is local; distributed runs require a file-backed `output`. The runner enforces this up front via `_DistributedRunner._require_reopenable`, which calls `to_spec()` on every input/output/mask and raises a role-tagged, "file-backed (zarr/n5)" error for in-memory arrays. Always allocating a *fresh* array on omission (rather than filtering in place) also removes the halo-in-place hazard.
- **Connected components: relabel only over labels that exist.** Stage 1 uses the cluster_tools offset scheme (`offset = block_id * prod(block_shape)`), which is *sparse*. Relabeling the whole union-find element space (size `max_label + 1`) yields a correct partition but non-compact ids (max ≫ component count). The fix: stage 1 returns each block's actual labels, and stage 3 relabels the union-find roots of only those labels → compact, consecutive output ids. This is validated with a partition-equality (not id-equality) check against whole-array `bioimage_cpp.segmentation.label`.
- **Filters dispatch by name, not by function object.** The per-block closure looks up `bioimage_cpp.filters` functions by string in a module-level dict, so the cloudpickled closure captures only picklable values (a string, the sigma, small tuples). Halo per axis is `sigma_to_halo(sigma, order)` (mirrors elf / VIGRA: `2·ceil(3σ + 0.5·order + 0.5)`).
- **Multi-stage = sequential `run()` calls** sharing the same `block_shape` so block ids line up across stages; the labeled volume itself is the inter-stage state (no task graph needed). A compute function recovers its own block id with `blocking.coordinates_to_block_id(block.begin)`, keeping the `function(block, inputs, outputs, mask)` convention unchanged.
- **Slurm is `_launch_and_wait` + a manifest; the protocol is shared.** `SlurmRunner` adds only sbatch-array submission, `sacct` polling and reattach; payload/harness/`_finalize`/`block_ids` re-run are inherited (the post-launch tail of `_execute` was extracted into a shared `_finalize` so reattach finalizes a detached run identically). The one real-world subtlety, learned on the cluster: per-task `.success` sentinels written on compute nodes take **up to the NFS attribute-cache timeout (~60 s on a default v3 mount) to become visible to the orchestrating (login) node** — measured ~36 s, and neither `os.path.exists` nor a fresh `os.listdir` busts it sooner. Because `_finalize` re-checks the sentinels, the poll cannot return until they are actually visible. The fix keeps the sentinel as the success ground truth but uses the lag-free `sacct` `State` to classify: a `COMPLETED` task (harness exited 0 ⇒ sentinel written) is given a generous `latency_wait` for its sentinel to appear, while any other terminal state means the harness did not succeed and the task is declared dead after a short grace — so successes resolve the moment the sentinel lands and genuine failures are reported quickly.
- **Still deferred:** the resolution-adapting mask wrapper (only equal-shape masks are accepted now), empty-block skipping, write-safety auto-derivation (only a chunk-multiple guard is implemented), and the multichannel / structure-tensor filter paths beyond the scalar gaussian family covered by parity tests.

## The slurm runner

The protocol is shared; slurm is a thin layer. `SlurmRunner` subclasses `_DistributedRunner` and implements only `_launch_and_wait(tmp, n_tasks, num_workers, name, task_ids=None)` (plus `reattach`) — everything else (payload, harness, per-block persistence + result collection, `_finalize` failure handling, `block_ids` re-run, the shared `resume`, `_require_reopenable`) is inherited and exercised by the `subprocess` backend. How it works and the gotchas it handles:

- **One array job.** It renders an sbatch script from `SlurmConfig` (partition, time, mem, cpus_per_task, gpus, account, qos, constraint, shebang) with `--array=<indices>%<throttle>` — `0-(n_tasks-1)` for a normal run, or a compressed sparse list (e.g. `0,3,7-9`) of just the incomplete tasks on a `resume_from`. Array index = task id; the per-task command is exactly the harness invocation the subprocess backend uses, run with the absolute `python_executable` (default `sys.executable`). `num_workers` is the array throttle, decoupled from task count — the base partitions into `n_tasks` independently. `n_tasks` is guarded against the cluster's `MaxArraySize` (queried via `scontrol`, overridable with `SlurmConfig.max_array_size`) so an oversized run fails up front rather than at submit.
- **Shared filesystem.** The temp folder (`RunnerConfig.tmp_root`) must live on a filesystem visible to all compute nodes (not node-local `/tmp`); `SlurmRunner` requires `tmp_root` to be set and errors clearly otherwise. The harness reopens sources from their specs there and writes results/sentinels there.
- **Worker environment.** The generated script runs the absolute `python_executable`, so the submitting env (which must have `bioimage_py` / `bioimage_cpp` importable) is reused on the node with no activation when it lives on a shared filesystem; `SlurmConfig.shebang` supplies any extra preamble (module loads / `LD_LIBRARY_PATH`). The payload's Python `(major, minor)` stamp catches mismatches at run time.
- **Poll terminal states with `sacct`, not `squeue`.** `squeue` drops finished jobs; `sacct -X -n -P --format=JobID,State -j <jobid>` reports one clean row per array task in a terminal state reliably (COMPLETED / FAILED / TIMEOUT / OUT_OF_MEMORY / NODE_FAIL / PREEMPTED / CANCELLED / …). Pending tasks collapse into a `<jobid>_[a-b%thr]` range row that is parsed out; `CANCELLED by <uid>` is normalised by first token; a non-zero `sacct` exit is a transient hiccup (skip that poll), an empty result is "not registered yet" (pending), and a task absent from `sacct` is pending, never dead. The ground truth for success stays the per-task `.success` sentinel file — see the NFS-visibility insight above for how `COMPLETED` vs other terminal states are used to detect dead tasks without false failures. Polls every `config.poll_interval` seconds and updates a progress bar (ok / failed / running / pending).
- **Manifest + reattach.** A manifest (job id, n_tasks, throttle, temp folder, …) is written at submission time; `block→task` assignment stays the single source of truth in `blocks/*.json`. `reattach(tmp_folder)` reads the manifest + payload, reconstructs the partition in numeric task order, re-probes `sacct` (raising if the job is gone and the run is unfinished), resumes polling and runs the shared `_finalize`. The orchestrating process typically runs on a login node and may be interrupted during multi-hour jobs, so reattach matters more here than for `subprocess`. On `KeyboardInterrupt` the job is left running (not cancelled) so it can be reattached.
- **Re-run on failure.** Two paths, both carried by the `RunnerError` (`failed_block_ids` + preserved `tmp_folder`). `block_ids` re-partitions and re-runs only the failed blocks in a fresh temp folder (correct for array-output ops; for a return-value op it reduces over only those blocks). `resume_from=tmp_folder` instead reuses the preserved run: `resume(tmp_folder)` reconstructs the partition + payload and resubmits a sparse array for just the incomplete tasks (the harness skips blocks already done), then finalizes over **all** persisted per-block results — so return-value ops get a correct merged answer too. `resume` is shared in `_DistributedRunner` (subprocess relaunches the task subset locally; slurm resubmits the sparse array); it is distinct from `reattach`, which re-polls a still-running job rather than resubmitting a dead one.
- **Testing.** `subprocess` is the CI proxy for the protocol; `tests/test_slurm_runner.py` adds slurm-only tests (gated on `sbatch` + `BIOIMAGE_PY_SHARED_TMP`) asserting parity against `local`/`subprocess` plus failure+re-run, reattach, and resume end to end. Because pytest's `tmp_path` is node-local, both the test arrays and `tmp_root` are placed on the shared filesystem via the `shared_tmp_path` / `shared_zarr_factory` fixtures.
