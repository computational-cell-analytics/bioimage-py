"""Tests for the user config file backing the slurm defaults.

These need no real ``sbatch``: they exercise only the config-file resolution, the
:func:`write_slurm_config` writer, the :meth:`SlurmConfig.load` reader, and the auto-load
that kicks in when a slurm runner is created without an explicit config. Each test isolates
itself from any real ``~/.config/bioimage-py/config.toml`` by pointing ``BIOIMAGE_PY_CONFIG``
at a temp file.
"""
import tomllib

import pytest

import bioimage_py as bp
from bioimage_py.runner import SlurmConfig, SlurmRunner, config_file_path, write_slurm_config


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    """Point the config-file resolution at an isolated temp path for the duration of a test."""
    path = tmp_path / "config.toml"
    monkeypatch.delenv("BIOIMAGE_PY_NO_CONFIG", raising=False)
    monkeypatch.setenv("BIOIMAGE_PY_CONFIG", str(path))
    return path


def test_write_then_load_round_trips(cfg_path):
    written = write_slurm_config(partition="gpu", account="proj", tmp_root="/scratch/shared", cpus_per_task=4)
    assert written == str(cfg_path)
    assert cfg_path.is_file()

    cfg = SlurmConfig.load()
    assert cfg.partition == "gpu"
    assert cfg.account == "proj"
    assert cfg.tmp_root == "/scratch/shared"
    assert cfg.cpus_per_task == 4
    # Unset fields keep their dataclass defaults.
    assert cfg.gpus == 0
    assert cfg.latency_wait == 120.0


def test_overrides_take_precedence_over_file(cfg_path):
    write_slurm_config(partition="cpu", time="01:00:00", tmp_root="/scratch/shared")
    cfg = SlurmConfig.load(partition="gpu", mem="16G")
    assert cfg.partition == "gpu"        # override wins
    assert cfg.mem == "16G"              # override-only field
    assert cfg.time == "01:00:00"        # file default retained
    assert cfg.tmp_root == "/scratch/shared"


def test_explicit_construction_ignores_file(cfg_path):
    write_slurm_config(partition="gpu", tmp_root="/scratch/shared")
    # A directly constructed config is used verbatim — the file is not consulted.
    cfg = SlurmConfig(time="00:30:00")
    assert cfg.partition is None
    assert cfg.tmp_root is None


def test_missing_file_loads_defaults(cfg_path):
    assert not cfg_path.is_file()
    cfg = SlurmConfig.load()
    assert cfg == SlurmConfig()


def test_write_merges_and_preserves_other_tables(cfg_path):
    write_slurm_config(partition="gpu")
    # A reserved table (future named profiles) must survive a subsequent writer call.
    data = tomllib.loads(cfg_path.read_text())
    data["profiles"] = {"big": {"slurm": {"mem": "64G"}}}
    cfg_path.write_text(_dump(data))

    write_slurm_config(account="proj")  # merge: keeps partition, adds account
    data = tomllib.loads(cfg_path.read_text())
    assert data["slurm"] == {"partition": "gpu", "account": "proj"}
    assert data["profiles"] == {"big": {"slurm": {"mem": "64G"}}}


def test_write_replace_resets_slurm_table(cfg_path):
    write_slurm_config(partition="gpu", account="proj")
    write_slurm_config(partition="cpu", replace=True)
    data = tomllib.loads(cfg_path.read_text())
    assert data["slurm"] == {"partition": "cpu"}


def test_write_skips_none_values(cfg_path):
    write_slurm_config(partition="gpu", account=None)
    data = tomllib.loads(cfg_path.read_text())
    assert data["slurm"] == {"partition": "gpu"}


def test_unknown_field_rejected_on_write(cfg_path):
    with pytest.raises(ValueError, match="partion"):
        write_slurm_config(partion="gpu")


def test_unknown_field_rejected_on_load(cfg_path):
    cfg_path.write_text('[slurm]\npartion = "gpu"\n')
    with pytest.raises(ValueError, match="partion"):
        SlurmConfig.load()


def test_unknown_field_rejected_in_overrides(cfg_path):
    with pytest.raises(ValueError, match="partion"):
        SlurmConfig.load(partion="gpu")


def test_no_config_env_disables_file(cfg_path, monkeypatch):
    write_slurm_config(partition="gpu", tmp_root="/scratch/shared")
    monkeypatch.setenv("BIOIMAGE_PY_NO_CONFIG", "1")
    cfg = SlurmConfig.load(mem="8G")
    assert cfg.partition is None          # file ignored
    assert cfg.tmp_root is None
    assert cfg.mem == "8G"                # overrides still apply


def test_config_file_path_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.toml"
    env = tmp_path / "env.toml"
    monkeypatch.setenv("BIOIMAGE_PY_CONFIG", str(env))
    # Explicit argument beats the env var.
    assert config_file_path(str(explicit)) == explicit
    # Env var beats the XDG default.
    assert config_file_path() == env
    # XDG default when nothing is set.
    monkeypatch.delenv("BIOIMAGE_PY_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_file_path() == tmp_path / "xdg" / "bioimage-py" / "config.toml"


def test_slurm_runner_auto_loads_file(cfg_path):
    write_slurm_config(partition="gpu", account="proj", tmp_root="/scratch/shared")
    # Creating the runner without a config pulls in the stored defaults.
    runner = SlurmRunner()
    assert runner.config.partition == "gpu"
    assert runner.config.account == "proj"
    assert runner.config.tmp_root == "/scratch/shared"
    # Same through the factory.
    runner = bp.get_runner("slurm")
    assert runner.config.partition == "gpu"


def test_slurm_runner_auto_load_respects_no_config(cfg_path, monkeypatch):
    write_slurm_config(partition="gpu")
    monkeypatch.setenv("BIOIMAGE_PY_NO_CONFIG", "1")
    runner = SlurmRunner()
    assert runner.config.partition is None


def _dump(data: dict) -> str:
    """Serialize a config dict back to TOML text (test helper)."""
    import tomli_w
    return tomli_w.dumps(data)
