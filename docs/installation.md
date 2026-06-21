# Installation

Install the package in editable mode from a clone of the repository:

```bash
python -m pip install -e .
```

This pulls the core dependencies (numpy, pandas, scikit-image, cloudpickle, tqdm, threadpoolctl and
`bioimage-cpp`), which are enough for in-memory (numpy) workflows and the `local` execution backend.

## Optional dependencies

File-backed and distributed I/O, and the individual file-format backends, are optional extras. Install
the ones you need, e.g. `python -m pip install -e ".[io]"` or combine several
(`python -m pip install -e ".[io,nifti]"`):

| Extra | Pulls in | Enables |
| --- | --- | --- |
| `io` | `zarr>=3`, `z5py` | Chunked zarr / n5 arrays — required for file-backed and distributed (`subprocess`/`slurm`) runs. |
| `hdf5` | `h5py` | HDF5 input (read). HDF5 is rejected as a *distributed* output. |
| `mrc` | `mrcfile` | MRC / REC volumes (read-only). |
| `nifti` | `nibabel` | NIfTI volumes (read-only). |
| `imagestack` | `imageio`, `tifffile` | TIFF files and folders of image slices. |
| `msr` | `msr-reader` | MSR / OBF microscopy files (read-only). |
| `cloudvolume` | `cloud-volume` | `CloudVolume` (precomputed) layers — writable, Linux only. |
| `webknossos` | `webknossos` | WebKnossos layers — read-only, remote or local. |
| `io-all` | all of the above | Every supported I/O backend in one go. |
| `test` | `pytest`, `zarr>=3`, `scikit-image`, `scipy`, `openpyxl` | Running the test suite. |
| `dev` | `flake8`, `pyflakes` | Linting. |

Distributed (`subprocess` / `slurm`) execution always requires a file-backed output, so install at
least the `io` extra for those workflows.
