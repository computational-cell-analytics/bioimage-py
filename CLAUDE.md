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
  reattach), `_harness.py` (worker entry point), `config.py` (`RunnerConfig` / `SlurmConfig`),
  `factory.py` (`get_runner`).
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
- `stats/`, `filters/`, `segmentation/` — the operations (`stats.max/min/mean/std`,
  `filters.apply_filter` + the gaussian family, `segmentation.label`).
- `copy.py` — `copy` (block-wise copy of one source into another, e.g. format conversion or
  persisting a wrapper to file) plus the shared `_copy_source` core (output handling + direct path +
  runner dispatch).
- `downsample.py` — `downsample` (block-wise downsampling; wraps the input in a `ResizedSource` at the
  downscaled shape and reuses `_copy_source`).
- `util.py` — shared helpers: `to_roi`, `get_blocking`, `derive_block_shape`, `sigma_to_halo`,
  `downscale_shape`, and the `BlockDescriptor` / `ComputeFn` type aliases.

Conventions (follow these):

- Every `__init__.py` is import-only; implementations live in dedicated modules and are re-exported.
- Blocking comes from `bioimage_cpp.utils` (`Blocking` / `Block` / `BlockWithHalo`); do not reimplement it.
- A `Source` is indexed only with a tuple of slices. Per-block functions convert a block with
  `to_roi(block)` (or `to_roi(block.outer_block / .inner_block / .inner_block_local)` under a halo);
  `Source` does not accept block objects.
- Per-block functions have the fixed signature `function(block, inputs, outputs, mask)` (the `ComputeFn`
  alias). They are cloudpickled, so capture only picklable values — dispatch heavy callables (e.g.
  `bioimage_cpp` functions) by name, not by object.
- Array-output ops (`filters.*`, `segmentation.label`, `copy`, `downsample`) take an optional `output`:
  for local execution a numpy array is allocated and returned when omitted; for distributed execution
  `output` is required and the runner validates it is file-backed (reopenable). These ops return the
  output array object.
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
the operations above (`stats`, `filters`, `segmentation.label`, `copy`, and `downsample` — the latter
built on the `ResizedSource` wrapper). The slurm-only tests in `tests/test_slurm_runner.py` are skipped unless
`sbatch` is on `PATH` and `BIOIMAGE_PY_SHARED_TMP` points at a shared filesystem; `subprocess` stays the
CI proxy for the shared protocol. Note the slurm runner's key subtlety: per-task `.success` sentinels are
written on compute nodes but can take up to the NFS attribute-cache timeout (~60 s) to become visible to
the orchestrating node, so success is detected via the sentinel while the lag-free `sacct` `State`
distinguishes a `COMPLETED`-but-not-yet-visible task (wait `latency_wait`) from a genuinely dead one.
