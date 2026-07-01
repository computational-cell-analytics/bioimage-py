We are building `bioimage_py`, a Python library for efficient, parallel, and distributed (block-wise)
image analysis. It is an evolution of [elf](https://github.com/constantinpape/elf) and
[cluster-tools](https://github.com/constantinpape/cluster_tools). The full design rationale lives in
DESIGN_DOC.md — read it before making design changes.

Reference clones are available locally at `/home/pape/Work/my_projects/elf` and
`/home/pape/Work/my_projects/cluster_tools`; mirror their proven patterns. Prefer algorithms from
[bioimage-cpp](https://github.com/computational-cell-analytics/bioimage-cpp) (`import bioimage_cpp`)
when available; otherwise fall back to numpy / scipy / scikit-image.

# Library Structure

The package lives in `bioimage_py/`:

- `runner/` — execution backends. `base.py` (the `Runner` ABC + the backend-independent `run()` +
  `LocalRunner` + the shared `run_block`), `distributed.py` (`_DistributedRunner` base + the shared
  `_finalize`, `SubprocessRunner`, and `SlurmRunner` — sbatch array submission, `sacct` polling and
  reattach), `_harness.py` (worker entry point), `config.py` (`RunnerConfig` / `SlurmConfig` plus
  the user config file: `config_file_path`, `write_slurm_config`, and `SlurmConfig.load` — a
  TOML `[slurm]` table under `~/.config/bioimage-py/config.toml` supplies cluster-specific
  defaults; auto-loaded when `SlurmRunner`/`get_runner("slurm")` get no config, gated by
  `BIOIMAGE_PY_NO_CONFIG` / `BIOIMAGE_PY_CONFIG`), `factory.py` (`get_runner`).
- `sources/` — `Source` ABC + `SourceSpec` (`base.py`), `ArraySource` for numpy/zarr/z5py
  (`array_source.py`), the `as_source` / `from_spec` / `SourceLike` dispatch (`dispatch.py`),
  `FileSource` + `open_source` (`file_source.py`, the `kind="file"` spec over the `io/` layer),
  `CloudVolumeSource` + `open_cloudvolume` (`cloudvolume_source.py`, writable, ZYX-over-XYZ), and
  `WebKnossosSource` + `open_webknossos` (`webknossos_source.py`, read-only, remote).
- `io/` — native file-format IO layer (mirrors `elf.io`): `open_file` + a guarded extension→format
  registry (`files.py`, `registry.py`), array-like wrappers for mrc / nifti / knossos / image-stack
  (tif) / msr, and the ported indexing helpers (`_util.py`). Each backend's heavy import is guarded
  (optional dependency). `FileSource` opens these by path and adds the reopenable spec.
- `wrapper/` — on-the-fly transformation sources: `WrapperSource` (`base.py`), `ThresholdSource`
  (`generic.py`), and `ResizedSource` (`resize.py`, a shape-changing resize/resample wrapper that
  reads through a halo and delegates interpolation to `bioimage_cpp.transformation`).
- `stats/`, `filters/`, `segmentation/`, `morphology/` — the operations (`stats.max/min/mean/std`,
  `filters.apply_filter` + the gaussian family, `segmentation.label` + `segmentation.watershed`,
  `morphology.morphology` + `morphology.regionprops`). `segmentation/relabel.py` holds `relabel`
  (applies an externally-supplied labeling — a `{old_id: new_id}` dict or a dense 1D array/source —
  to a segmentation block-wise, in place by default; a numpy labeling array is persisted to a temp
  zarr under the runner temp root for distributed backends, cleaned on success via `pre_cleanup`,
  preserved on failure for `resume_from`) and `relabel_consecutive` (derives a consecutive mapping
  via a global `unique` reduction, then delegates the block-wise write to `relabel`; in place by
  default). `segmentation/multicut.py` ports the
  bioimage-cpp-backed multicut cost transform + solvers (`compute_edge_costs`,
  `multicut_decomposition` / `_gaec` / `_kernighan_lin`); meant to grow into multicut-based
  segmentation. `segmentation/stitching.py` (`stitch_segmentation` / `stitch_tiled_segmentation`)
  merges a tile-wise over-segmentation via a multicut over tile-interface object overlaps — the
  per-voxel phases (tile segmentation, overlap counting) run through the runner; RAG build +
  multicut solve are in-process private helpers (`_compute_rag` / `_project_node_labels_to_pixels`,
  with a TODO to move to dedicated distributed graph functionality).
- `evaluation/` — segmentation-comparison metrics built on a parallel contingency table. `contingency_table`
  returns the `ContingencyTable` dataclass primitive (sparse overlap counts via
  `bioimage_cpp.utils.segmentation_overlap`, additive across blocks with no halo; has a `drop_ignore`
  post-filter for ignore labels). Each metric — `variation_of_information` / `object_vi`, `rand_index`,
  `cremi_score`, `matching` / `mean_segmentation_accuracy`, `dice_score`, `symmetric_best_dice_score` —
  has a low-level `*_scores(table, …)` form and a high-level `(segmentation, groundtruth, …)` wrapper
  that builds the table in parallel then scores. `dice_score` is its own small sum-reduction (not table-based).
  `centroid_matching` / `coordinate_matching` are also non-table: they match objects by centroid distance
  under a threshold (high-level derives centroids via `morphology`'s center of mass, then defers to the
  coordinate-level form), reusing matching's precision/recall/f1/segmentation-accuracy formulas.
- `copy.py` — `copy` (block-wise copy of one source into another, e.g. format conversion or
  persisting a wrapper to file) plus the shared `_copy_source` core (output handling + direct path +
  runner dispatch).
- `downsample.py` — `downsample` (block-wise downsampling; wraps the input in a `ResizedSource` at the
  downscaled shape and reuses `_copy_source`).
- `util.py` — shared helpers: `to_roi`, `get_blocking`, `derive_block_shape`, `sigma_to_halo`,
  `downscale_shape`, `check_rerun_args`, the direct-fast-path trio `is_direct` / `check_direct` /
  `full_roi`, and the `BlockDescriptor` / `ComputeFn` type aliases.

Conventions (follow these):

- Every `__init__.py` is import-only; implementations live in dedicated modules and are re-exported.
- Blocking comes from `bioimage_cpp.utils` (`Blocking` / `Block` / `BlockWithHalo`); do not reimplement it.
- A `Source` accepts numpy-style basic indexing (int / slice / ellipsis / tuple): the base class
  normalizes the index to a full tuple of in-bounds slices and squeezes integer-indexed axes (via the
  shared `bioimage_py._indexing` helpers), then delegates to each source's `_getitem` / `_setitem`,
  which always receive a full tuple of slices. `Source` does not accept block objects: per-block
  functions convert a block with `to_roi(block)` (or `to_roi(block.outer_block / .inner_block /
  .inner_block_local)` under a halo) for explicitness, never relying on the index normalization.
- Per-block functions have the fixed signature `function(block, inputs, outputs, mask)` (the `ComputeFn`
  alias). They are cloudpickled, so capture only picklable values — dispatch heavy callables (e.g.
  `bioimage_cpp` functions) by name, not by object.
- Array-output ops (`filters.*`, `segmentation.label`, `segmentation.watershed`, `copy`, `downsample`)
  take an optional `output`: for local execution a numpy array is allocated and returned when omitted;
  for distributed execution `output` is required and the runner validates it is file-backed (reopenable).
  These ops return the output array object.
- Every block-wise op has a *direct* fast path (whole-array, no runner / no blocking) for
  `job_type="local"`, `num_workers==1`, `block_shape is None`, built on the shared `util.is_direct` /
  `util.check_direct` / `util.full_roi` — do not reimplement these per op. The two op families use them
  differently: reduction ops (`stats`, `morphology`, `evaluation.contingency_table`,
  `evaluation.dice_score`) call `check_direct(...)`, which *raises* when a `mask`/`block_ids` is passed
  alongside the direct conditions (the whole-array path can't honor them — pass a `block_shape` to go
  blocked); array-output ops compute `direct = is_direct(...) and <their mask/block_ids/resume_from are
  None>` and silently fall through to the blocked path instead (per-op extra conditions vary, e.g.
  `watershed`'s direct path does honor a mask). `full_roi(ndim)` builds the whole-array slicing for both.
- `segmentation.relabel` is the canonical way to write a *node labeling* (a relabeling of segment ids)
  onto pixels — any op that produces per-object labels (multicut, clustering, agglomeration, graph
  partitioning, size/property filters, …) should apply its result through `relabel` rather than a
  bespoke block-wise write. Performance findings (benchmarked): prefer a **dense 1D array/source**
  labeling (`labeling[old_id] = new_id`) over a dict — the per-block kernel is then `numpy.take`, which
  is O(block) *independent* of the labeling size (~0.3 ms/block even at 1e8 entries). A dict labeling
  uses `bioimage_cpp.utils.take_dict`, which rebuilds a hash map from the *whole* dict every block
  (O(dict size)/block); `relabel` mitigates this with **gated per-block subsampling** — for large dicts
  it restricts the mapping to the block's `np.unique` ids before `take_dict` (gated by
  `_RELABEL_SUBSAMPLE_MIN_DICT` / `_RELABEL_SUBSAMPLE_MAX_DIVERSITY` so it never regresses on small
  dicts or pathologically diverse blocks; ~7–8× faster over a large volume) — but a dense array is
  still fastest. `relabel_consecutive` derives its `{old_id: new_id}` mapping and delegates the write to
  `relabel` (kept as a dict, not a dense array, so it stays memory-safe for sparse/large input id
  spaces — the case it is most often used to compact). `bioimage_cpp` has no dense-array `take`.
- numpy arrays are local-only (their `to_spec()` raises); distributed backends need a reopenable source.
- A `Source` exposes a `writable` property (default `True`; `False` for wrappers, read-only `FileSource`s,
  and `WebKnossosSource`). The distributed runner rejects non-writable outputs, and rejects HDF5 as an
  output (concurrent multi-process writes corrupt it). Read-only formats (mrc/nifti/knossos/...) and
  HDF5 are valid distributed *inputs* (concurrent readers are safe); distributed *outputs* stay zarr/n5.

# Installation

Editable install: `python -m pip install -e .` (or `.[test,dev]`). Build/runtime metadata is in
`pyproject.toml`; `setup.cfg` holds the flake8 config (line length 120). Core deps: `bioimage_cpp`,
numpy, cloudpickle, tqdm, threadpoolctl (zarr / z5py for file-backed and distributed I/O).

# Tests

`python -m pytest -q` runs the suite under `tests/`. The headline `tests/test_runner_parity.py` asserts
`direct == LocalRunner == SubprocessRunner` for the ops — keep this parity green, it is the core
correctness guarantee. Use the `zarr_factory` / `rng` fixtures in `tests/conftest.py`.

# Coding standards etc.

Code should be PEP8-compliant (line limit 120), use type annotations on every function (parameters and
return type), and google-style doc strings. The documentation will later be built with pdoc (so you can
already use specific conventions from it if needed). Public functions document all parameters with
consistent wording; private helpers get a concise one-line docstring.

Use pyflakes and flake8 for linting: `python -m flake8 bioimage_py tests` and
`python -m pyflakes bioimage_py`.

# Status

Implemented and tested: the full `local` path, the `subprocess` backend (the real distributed protocol —
cloudpickle payload, generated harness, per-task result/sentinel files, `block_ids` re-run, failure
reporting), the `slurm` backend (sbatch array submission, `sacct` polling, reattach via a manifest), and
the operations above (`stats`, `filters`, `segmentation.label` + `segmentation.watershed`, `morphology`
+ `regionprops`, `copy`, and `downsample` — the latter built on the `ResizedSource` wrapper — the
`evaluation` package: the parallel `contingency_table` primitive plus the metrics built on it —
`segmentation.relabel` (apply a node labeling / relabeling map, in place by default; the canonical
node-label writer, see Conventions) + `segmentation.relabel_consecutive`, and
`segmentation.stitch_segmentation` / `stitch_tiled_segmentation` on the new `segmentation.multicut`
solvers). The slurm-only tests in `tests/test_slurm_runner.py` are skipped unless
`sbatch` is on `PATH` and `BIOIMAGE_PY_SHARED_TMP` points at a shared filesystem; `subprocess` stays the
CI proxy for the shared protocol. Note the slurm runner's key subtlety: per-task `.success` sentinels are
written on compute nodes but can take up to the NFS attribute-cache timeout (~60 s) to become visible to
the orchestrating node, so success is detected via the sentinel while the lag-free `sacct` `State`
distinguishes a `COMPLETED`-but-not-yet-visible task (wait `latency_wait`) from a genuinely dead one.
