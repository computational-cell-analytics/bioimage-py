"""Cluster-gated slurm parity tests for the distributed RAG / edge-feature orchestration.

These submit real sbatch array jobs, so they are skipped unless ``sbatch`` is on ``PATH`` and
``BIOIMAGE_PY_SHARED_TMP`` points at a shared filesystem (same gate as ``tests/test_slurm_runner.py``).
They assert that the slurm backend reproduces the whole-volume graph / edge features exactly for the
region graph and the ``size/min/max`` columns, and to floating-point tolerance for ``mean/std`` --
extending the ``local`` / ``subprocess`` coverage in ``tests/graph/test_distributed_rag.py`` to slurm.

Partition/account default to this cluster's CPU test queue and can be overridden via the
``BIOIMAGE_PY_SLURM_PARTITION`` / ``BIOIMAGE_PY_SLURM_ACCOUNT`` env vars.
"""
import os
import shutil

import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp
from bioimage_py.runner import SlurmConfig

# Skip unless the slurm CLI and a shared filesystem (for the job temp + test arrays) are both
# available; computed here rather than imported from conftest to avoid a fragile module import.
pytestmark = pytest.mark.skipif(
    shutil.which("sbatch") is None or not os.environ.get("BIOIMAGE_PY_SHARED_TMP"),
    reason="needs sbatch on PATH and BIOIMAGE_PY_SHARED_TMP set to a shared-filesystem dir",
)

# Distributed complex feature columns are (mean, std, min, max, size); min/max/size are exact across
# backends / block order, mean/std are only float-order stable.
_EXACT_COMPLEX = [2, 3, 4]
_ALLCLOSE_COMPLEX = [0, 1]


def _cfg(tmp_root, **overrides):
    """Build a small SlurmConfig for quick CPU test jobs (duplicated from test_slurm_runner.py)."""
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


def _assert_features_backend_equal(a, b, complex_features):
    """Assert two distributed feature arrays agree (exact on size/min/max, allclose on mean/std)."""
    if complex_features:
        for col in _EXACT_COMPLEX:
            np.testing.assert_array_equal(a[:, col], b[:, col])
        for col in _ALLCLOSE_COMPLEX:
            np.testing.assert_allclose(a[:, col], b[:, col], rtol=1e-6, atol=1e-9)
    else:  # (mean, size)
        np.testing.assert_array_equal(a[:, 1], b[:, 1])
        np.testing.assert_allclose(a[:, 0], b[:, 0], rtol=1e-6, atol=1e-9)


def test_slurm_rag_parity(shared_zarr_factory, rng, shared_tmp_path):
    labels = rng.integers(0, 6, size=(24, 20)).astype("uint64")
    z = shared_zarr_factory(labels, chunks=(8, 8))
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()

    graph = bp.graph.distributed_rag(z, block_shape=(8, 8), num_workers=4, job_type="slurm",
                                     job_config=_cfg(shared_tmp_path))
    np.testing.assert_array_equal(graph.uv_ids(), whole)


@pytest.mark.parametrize("complex_features", [False, True])
def test_slurm_edge_map_features_parity(complex_features, shared_zarr_factory, rng, shared_tmp_path):
    labels = rng.integers(0, 6, size=(24, 20)).astype("uint64")
    edge_map = rng.standard_normal((24, 20)).astype("float64")
    zl = shared_zarr_factory(labels, chunks=(8, 8))
    ze = shared_zarr_factory(edge_map, chunks=(8, 8))
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()

    _, direct = bp.graph.distributed_edge_features(labels, edge_map, complex_features=complex_features)
    graph, feats = bp.graph.distributed_edge_features(
        zl, ze, complex_features=complex_features, block_shape=(8, 8), num_workers=4,
        job_type="slurm", job_config=_cfg(shared_tmp_path))
    np.testing.assert_array_equal(graph.uv_ids(), whole)
    _assert_features_backend_equal(direct, feats, complex_features)


def test_slurm_affinity_features_parity(shared_zarr_factory, rng, shared_tmp_path):
    offsets = [[1, 0], [0, 1], [3, 0], [0, 3]]  # includes long-range offsets (halo > block-face)
    labels = rng.integers(0, 6, size=(24, 20)).astype("uint64")
    aff = rng.standard_normal((len(offsets),) + labels.shape).astype("float64")
    zl = shared_zarr_factory(labels, chunks=(8, 8))
    za = shared_zarr_factory(aff, chunks=(len(offsets), 8, 8))
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()

    _, direct = bp.graph.distributed_edge_features(
        labels, aff, feature_type="affinity", offsets=offsets, complex_features=True)
    graph, feats = bp.graph.distributed_edge_features(
        zl, za, feature_type="affinity", offsets=offsets, complex_features=True,
        block_shape=(8, 8), num_workers=4, job_type="slurm", job_config=_cfg(shared_tmp_path))
    np.testing.assert_array_equal(graph.uv_ids(), whole)
    _assert_features_backend_equal(direct, feats, True)
