# bioimage-py

Efficient, parallel, and distributed implementation of image analysis and segmentation functionality for biomedical imaging.

**Note:** this package is in an early state and mainly provides support for data conversion, downsampling, and some initial segmentation functionality (connected components and watershed).
The functionality will be extended soon; the implementation of seamlessly switching between local and distributed execution (via slurm) is already in place. Any feedback on issues you find or on how to improve usability is welcome!

This package can be installed via `pip`:
```bash
pip install bioimage-py
```
and `conda`:
```bash
conda install -c conda-forge bioimage-py
```

See the [documentation](https://computational-cell-analytics.github.io/bioimage-py/) 
for more detailed [installation instructions](https://computational-cell-analytics.github.io/bioimage-py/bioimage_py.html#installation) 
and [usage examples](https://computational-cell-analytics.github.io/bioimage-py/bioimage_py.html#usage).
