# Installation

This package can be installed via `pip`:
```bash
pip install bioimage-py
```
and `conda`:
```bash
conda install -c conda-forge bioimage-py
```

You can also install it from source by cloning the repository and then running 
```bash
python -m pip install -e .
```

This pulls the core dependencies, including numpy, pandas, pyarrow, scikit-image, and
`bioimage-cpp`. These dependencies support in-memory workflows, Parquet tables, and local runs.

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
| `tensorstore` | `tensorstore` | TensorStore neuroglancer-precomputed layers — local ZYX reads and shard-safe writes. |
| `webknossos` | `webknossos` | WebKnossos layers — read-only, remote or local. |
| `io-all` | all array I/O packages above | Every supported array I/O backend in one install. |
| `test` | `pytest`, `zarr>=3`, `scikit-image`, `scipy`, `openpyxl`, `pyarrow` | Running the test suite. |
| `dev` | `flake8`, `pyflakes` | Linting. |

Distributed array operations require a file-backed output. Install at least the `io` extra for
these operations.
