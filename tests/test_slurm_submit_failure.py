"""Offline test for the slurm submit-failure path (audit issue #12).

Unlike ``test_slurm_runner.py`` this needs no real ``sbatch``: it monkeypatches the slurm CLI
lookup and ``subprocess.run`` so :meth:`SlurmRunner._submit` fails deterministically. It asserts
that a submission failure (a) does **not** remove the job temp folder — preserved so the user can
inspect the generated ``submit.sh`` / payload / block lists — and (b) raises an error that names
that folder.
"""
import os
import subprocess

import pytest

from bioimage_py.runner import SlurmConfig, SlurmRunner
from bioimage_py.runner import distributed


def test_submit_failure_preserves_and_names_tmp(tmp_path, monkeypatch):
    """A failed sbatch submission keeps the temp folder and reports its location."""
    runner = SlurmRunner(SlurmConfig(tmp_root=str(tmp_path), max_array_size=1000))
    tmp = str(tmp_path / "job")
    os.makedirs(tmp)

    monkeypatch.setattr(distributed.shutil, "which",
                        lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="sbatch: error: boom")

    monkeypatch.setattr(distributed.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        runner._launch_and_wait(tmp, n_tasks=1, num_workers=1, name="t")

    msg = str(excinfo.value)
    assert "boom" in msg                              # original sbatch error preserved
    assert "Temp folder preserved for debugging" in msg
    assert tmp in msg                                 # points the user at the folder
    assert os.path.isdir(tmp)                         # folder deliberately not removed
