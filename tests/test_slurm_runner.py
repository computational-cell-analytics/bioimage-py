"""Cluster-gated tests for the slurm runner.

These submit real sbatch array jobs, so they are skipped unless ``sbatch`` is on ``PATH``
and ``BIOIMAGE_PY_SHARED_TMP`` points at a shared filesystem (see ``requires_slurm`` in
``conftest.py``). The subprocess backend remains the CI proxy for the shared protocol; these
tests assert that the slurm backend produces identical results and that its slurm-specific
pieces (failure reporting + ``block_ids`` re-run, and reattach) work end to end.

Partition/account default to this cluster's CPU test queue and can be overridden via the
``BIOIMAGE_PY_SLURM_PARTITION`` / ``BIOIMAGE_PY_SLURM_ACCOUNT`` env vars.
"""
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bioimage_cpp as bic
import bioimage_py as bp
from bioimage_py.runner import RunnerError, SlurmConfig, SlurmRunner
from bioimage_py.util import to_roi

# Skip unless the slurm CLI and a shared filesystem (for the job temp + test arrays) are both
# available; computed here rather than imported from conftest to avoid a fragile module import.
pytestmark = pytest.mark.skipif(
    shutil.which("sbatch") is None or not os.environ.get("BIOIMAGE_PY_SHARED_TMP"),
    reason="needs sbatch on PATH and BIOIMAGE_PY_SHARED_TMP set to a shared-filesystem dir",
)


def _cfg(tmp_root, **overrides):
    """Build a small SlurmConfig for quick CPU test jobs."""
    params = dict(
        tmp_root=tmp_root,
        partition=os.environ.get("BIOIMAGE_PY_SLURM_PARTITION", "standard96:test"),
        account=os.environ.get("BIOIMAGE_PY_SLURM_ACCOUNT", "nim00007"),
        time="00:10:00",
        cpus_per_task=1,
        mem="2G",
        poll_interval=5.0,
    )
    params.update(overrides)
    return SlurmConfig(**params)


def _make_block_max():
    """Return a closure computing the per-block maximum.

    A closure (not a module-level function) is used on purpose: cloudpickle serializes it
    by value, so the worker need not import this test module — a module-level function would
    be pickled by reference and fail on the compute node with ModuleNotFoundError.
    """
    def fn(block, inputs, outputs, mask):
        return float(np.max(inputs[0][to_roi(block)]))
    return fn


def _fail_on_corner(should_fail):
    """Factory: a per-block fn that fails on the corner block (id 0) iff ``should_fail``."""
    def fn(block, inputs, outputs, mask):
        is_corner = all(int(b) == 0 for b in block.begin)
        if should_fail and is_corner:
            raise RuntimeError("boom on corner block")
        return int(block.begin[0])
    return fn


def _make_table_writer(schema):
    """Return a closure that writes one typed row per logical item."""
    def fn(batch, writer):
        writer.write(pa.table({
            "item": pa.array(list(batch), type=pa.int64()),
        }, schema=schema))
    return fn


class _Detached(Exception):
    """Carries the temp folder of a run whose orchestrator 'died' right after submit."""

    def __init__(self, tmp):
        super().__init__(tmp)
        self.tmp = tmp


class _DetachAfterSubmit(SlurmRunner):
    """Test double: submits + writes the manifest, then aborts before polling."""

    def _poll(self, job_id, n_tasks, tmp, name, task_ids=None):  # type: ignore[override]
        raise _Detached(tmp)


def test_slurm_max_parity(shared_zarr_factory, rng, shared_tmp_path):
    a = rng.random((64, 64)).astype("float32")
    z = shared_zarr_factory(a, chunks=(16, 16))
    got = bp.stats.max(z, num_workers=4, block_shape=(16, 16),
                       job_type="slurm", job_config=_cfg(shared_tmp_path))
    assert np.isclose(got, float(a.max()))


def test_slurm_gaussian_parity(shared_zarr_factory, rng, shared_tmp_path):
    a = rng.random((48, 48)).astype("float32")
    z = shared_zarr_factory(a, chunks=(16, 16))
    out = shared_zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
    ref = bic.filters.gaussian_smoothing(a, 2.0)

    bp.filters.gaussian_smoothing(z, 2.0, output=out, block_shape=(16, 16),
                                  num_workers=4, job_type="slurm", job_config=_cfg(shared_tmp_path))
    np.testing.assert_allclose(out[:], ref, atol=1e-4)


def test_slurm_failure_then_rerun(shared_zarr_factory, rng, shared_tmp_path):
    a = rng.random((32, 32)).astype("float32")
    z = shared_zarr_factory(a, chunks=(16, 16))
    runner = bp.get_runner("slurm", _cfg(shared_tmp_path))

    with pytest.raises(RunnerError) as excinfo:
        runner.run(_fail_on_corner(True), [z], block_shape=(16, 16), num_workers=4,
                   has_return_val=True, name="slurm-flaky")
    err = excinfo.value
    assert 0 in err.failed_block_ids  # the corner block is block id 0
    assert err.tmp_folder is not None and os.path.isdir(err.tmp_folder)
    assert err.task_failures
    failure = err.task_failures[0]
    assert failure.backend == "slurm"
    assert failure.scheduler_task_id is not None
    assert failure.stderr_path is not None
    if failure.scheduler_state is not None:
        assert failure.scheduler_state in {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL"}

    # Re-running the reported failed blocks with a non-failing fn now succeeds.
    results = runner.run(_fail_on_corner(False), [z], block_shape=(16, 16), num_workers=4,
                         has_return_val=True, block_ids=err.failed_block_ids, name="slurm-rerun")
    assert all(r is not None for r in results)


def test_slurm_reattach(shared_zarr_factory, rng, shared_tmp_path):
    a = rng.random((64, 64)).astype("float32")
    z = shared_zarr_factory(a, chunks=(16, 16))
    cfg = _cfg(shared_tmp_path)

    # Submit + write the manifest, then 'die' before polling (orchestrator interrupted).
    with pytest.raises(_Detached) as excinfo:
        _DetachAfterSubmit(cfg).run(_make_block_max(), [z], block_shape=(16, 16), num_workers=4,
                                    has_return_val=True, name="reattach-src")
    tmp = excinfo.value.tmp
    assert os.path.exists(os.path.join(tmp, "manifest.json"))

    # A fresh runner picks the still-running job back up from the manifest and finalizes it.
    results = SlurmRunner(cfg).reattach(tmp, name="reattach")
    assert np.isclose(max(r for r in results if r is not None), float(a.max()))
    assert not os.path.isdir(tmp)  # cleaned up on successful finalize


def test_slurm_table_sink_reattach(shared_tmp_path):
    cfg = _cfg(shared_tmp_path)
    schema = pa.schema([("item", pa.int64())])
    dataset = bp.TableDataset.create(
        os.path.join(shared_tmp_path, "table.parquet"),
        schema=schema,
        schema_version=1,
        operation="slurm-table-test",
        operation_version="1",
        input_identities={"input": "synthetic"},
        parameters={},
    )

    with pytest.raises(_Detached) as excinfo:
        _DetachAfterSubmit(cfg).map_batches(
            _make_table_writer(schema), 5, batch_size=2, num_workers=2,
            result_sink=dataset, name="table-reattach-src",
        )

    result = SlurmRunner(cfg).reattach(excinfo.value.tmp, name="table-reattach")
    assert isinstance(result, bp.TableDataset)
    assert result.to_pandas()["item"].tolist() == list(range(5))
    assert not os.path.isdir(excinfo.value.tmp)


def test_slurm_file_backed_regionprops(shared_zarr_factory, shared_tmp_path):
    seg = np.zeros((24, 28, 32), dtype="uint64")
    seg[2:8, 3:12, 4:14] = 7
    seg[10:19, 15:25, 18:29] = 2
    source = shared_zarr_factory(seg, chunks=(12, 14, 16))
    base = bp.morphology.morphology(seg).iloc[::-1].reset_index(drop=True)
    base_path = os.path.join(shared_tmp_path, "regionprops-base.parquet")
    output_path = os.path.join(shared_tmp_path, "regionprops-result.parquet")
    base.to_parquet(base_path, index=False)
    expected = bp.morphology.regionprops(seg, base)
    expected = expected.set_index("label").loc[base["label"].tolist()].reset_index()
    expected["label"] = expected["label"].astype("uint64")
    expected["n_voxels"] = expected["n_voxels"].astype("uint64")

    result = bp.morphology.regionprops(
        source,
        base_path,
        output_table=output_path,
        rows_per_batch=1,
        num_workers=2,
        job_type="slurm",
        job_config=_cfg(shared_tmp_path),
    )

    pd.testing.assert_frame_equal(result.to_pandas(), expected)


def test_slurm_file_backed_morphology(shared_zarr_factory, shared_tmp_path):
    seg = np.zeros((24, 28, 32), dtype="uint64")
    seg[2:8, 3:12, 4:14] = 17
    seg[10:19, 15:25, 18:29] = 2
    source = shared_zarr_factory(seg, chunks=(12, 14, 16))
    output_path = os.path.join(shared_tmp_path, "morphology-result.parquet")

    result = bp.morphology.morphology(
        source,
        output_table=output_path,
        block_shape=(12, 14, 16),
        blocks_per_batch=2,
        label_partition_size=10,
        num_workers=2,
        job_type="slurm",
        job_config=_cfg(shared_tmp_path),
    )

    pd.testing.assert_frame_equal(
        result.to_pandas(), bp.morphology.morphology(seg),
    )


def _make_shared_flaky(marker_path):
    """Per-block fn that fails the corner block once, via a marker on the shared filesystem.

    The marker must live on the shared FS so the compute node that re-runs the block (on resume)
    sees that the first attempt already happened and succeeds.
    """
    def fn(block, inputs, outputs, mask):
        is_corner = all(int(b) == 0 for b in block.begin)
        if is_corner and not os.path.exists(marker_path):
            open(marker_path, "w").close()
            raise RuntimeError("transient boom")
        return int(block.begin[0])
    return fn


def test_slurm_resume(shared_zarr_factory, rng, shared_tmp_path):
    a = rng.random((32, 32)).astype("float32")
    z = shared_zarr_factory(a, chunks=(16, 16))
    cfg = _cfg(shared_tmp_path)
    marker = os.path.join(shared_tmp_path, "marker.txt")
    fn = _make_shared_flaky(marker)

    with pytest.raises(RunnerError) as excinfo:
        bp.get_runner("slurm", cfg).run(fn, [z], block_shape=(16, 16), num_workers=4,
                                        has_return_val=True, name="slurm-flaky")
    tmp = excinfo.value.tmp_folder
    assert tmp is not None and os.path.isdir(tmp)

    # resume resubmits a sparse array for only the incomplete tasks; the corner block now succeeds
    # (marker exists), and the result merges with the blocks that completed in the first attempt.
    results = SlurmRunner(cfg).resume(tmp, name="slurm-resume")
    assert results is not None and all(r is not None for r in results)
    assert not os.path.isdir(tmp)  # cleaned up on successful finalize
