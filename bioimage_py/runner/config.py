"""Runner configuration dataclasses and the user config file (``~/.config/bioimage-py``)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunnerConfig:
    """Base configuration shared by all runners.

    Attributes:
        poll_interval: Seconds between status polls (distributed runners).
        tmp_root: Root directory for job temp folders. ``None`` uses the system default.
            For distributed jobs this must be on a shared filesystem.
        python_executable: Interpreter used to launch worker tasks. ``None`` uses the
            current interpreter (``sys.executable``).
    """

    poll_interval: float = 10.0
    tmp_root: Optional[str] = None
    python_executable: Optional[str] = None


@dataclass
class SlurmConfig(RunnerConfig):
    """Configuration for the slurm runner.

    Inherits ``poll_interval``, ``tmp_root`` and ``python_executable`` from
    :class:`RunnerConfig`. For slurm, ``tmp_root`` is **required** and must point at a
    shared filesystem visible to all compute nodes (not node-local ``/tmp``), and
    ``num_workers`` (passed to the op / ``run``) is interpreted as the array throttle — the
    maximum number of tasks allowed to run concurrently — independently of how many tasks
    the work is partitioned into.

    Cluster-specific values (``partition``, ``account``, ``constraint``, ``tmp_root``, ...)
    can be stored once in a user config file and reused as defaults; see
    :meth:`load` and :func:`write_slurm_config`.

    Attributes:
        partition: The slurm partition to submit to.
        time: The per-task time limit (slurm time format, e.g. ``"01:00:00"``).
        mem: The per-task memory limit (e.g. ``"8G"``).
        cpus_per_task: Number of CPUs requested per task.
        gpus: Number of GPUs requested per task (emitted as ``--gpus`` only when > 0).
        account: The accounting project to charge.
        qos: The quality-of-service to request.
        constraint: A node feature constraint.
        shebang: Optional environment setup for the generated job script. If given, its
            first line must be an interpreter line (starting with ``#!``) which is placed at
            the top of the script; any remaining lines are emitted as an activation preamble
            *after* the ``#SBATCH`` directives (so the directives are still honoured). The
            preamble is for making the package importable on the node (e.g. ``module load``
            / ``LD_LIBRARY_PATH`` exports), not for choosing the interpreter: the worker is
            always launched with the absolute ``python_executable`` (defaulting to the
            submitting ``sys.executable``). ``None`` uses ``#!/bin/bash`` and that absolute
            interpreter, which needs no activation when the env lives on a shared
            filesystem. Example::

                shebang = "#!/bin/bash\\nmodule load gcc\\nexport LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH"

        max_array_size: Override for the maximum number of array tasks per job. ``None``
            queries the cluster's ``MaxArraySize`` (falling back to a safe default). A run
            partitioned into more tasks than this is rejected up front with a clear error.
        latency_wait: Seconds to wait for a finished task's ``.success`` sentinel to become
            visible on a shared (NFS) filesystem before giving up on it. A task that the
            scheduler reports ``COMPLETED`` wrote its sentinel, but the orchestrating node's
            attribute cache can lag the compute node by up to the mount's ``acdirmax``
            (typically 60 s); this must comfortably exceed that. It only bounds the wait on a
            ``COMPLETED``-but-not-yet-visible task — a task is resolved the moment its
            sentinel appears, so a generous value does not slow down successful runs.
    """

    partition: Optional[str] = None
    time: Optional[str] = None
    mem: Optional[str] = None
    cpus_per_task: int = 1
    gpus: int = 0
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    shebang: Optional[str] = None
    max_array_size: Optional[int] = None
    latency_wait: float = 120.0

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides: Any) -> "SlurmConfig":
        """Build a config from the user config file, with explicit overrides taking precedence.

        Precedence is ``overrides`` > config file ``[slurm]`` section > dataclass defaults.
        This is the way to combine the stored user defaults with per-run tweaks; constructing
        ``SlurmConfig(...)`` directly does **not** consult the file (an explicitly built config
        is used verbatim).

        Args:
            path: Path to the config file. ``None`` resolves the default location (see
                :func:`config_file_path`). A missing file is treated as empty.
            **overrides: Field values that override the file defaults. Each name must be a
                valid ``SlurmConfig`` field.

        Returns:
            A :class:`SlurmConfig` with file defaults filled in and overrides applied.

        Raises:
            ValueError: If the file or ``overrides`` contain an unknown field name.
        """
        _validate_keys(overrides, "load() overrides")
        merged: Dict[str, Any] = dict(_read_slurm_defaults(path))
        merged.update(overrides)
        return cls(**merged)


def config_file_path(path: Optional[str] = None) -> Path:
    """Resolve the path to the user config file.

    Resolution order: an explicit ``path`` argument, then the ``BIOIMAGE_PY_CONFIG``
    environment variable, then ``$XDG_CONFIG_HOME/bioimage-py/config.toml`` (falling back to
    ``~/.config/bioimage-py/config.toml``).

    Args:
        path: An explicit path that short-circuits the resolution. ``None`` resolves the
            default location.

    Returns:
        The resolved path (not guaranteed to exist).
    """
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("BIOIMAGE_PY_CONFIG")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "bioimage-py" / "config.toml"


def _slurm_field_names() -> set:
    """Return the set of valid ``SlurmConfig`` field names."""
    return {f.name for f in fields(SlurmConfig)}


def _validate_keys(keys: Any, where: str) -> None:
    """Raise ``ValueError`` if ``keys`` contains a name that is not a ``SlurmConfig`` field."""
    unknown = set(keys) - _slurm_field_names()
    if unknown:
        raise ValueError(
            f"Unknown SlurmConfig option(s) {sorted(unknown)} in {where}. "
            f"Valid options are {sorted(_slurm_field_names())}."
        )


def _parse_toml(fp: Path) -> Dict[str, Any]:
    """Parse a TOML config file, returning an empty dict if it does not exist."""
    if not fp.is_file():
        return {}
    with open(fp, "rb") as f:
        return tomllib.load(f)


def _read_slurm_defaults(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the validated ``[slurm]`` table from the config file.

    Honors ``BIOIMAGE_PY_NO_CONFIG`` (set to disable file lookup entirely, e.g. for
    reproducible CI runs), in which case an empty dict is returned.
    """
    if os.environ.get("BIOIMAGE_PY_NO_CONFIG"):
        return {}
    data = _parse_toml(config_file_path(path))
    section = data.get("slurm", {})
    if not isinstance(section, dict):
        raise ValueError(f"Expected a [slurm] table in {config_file_path(path)}, got {type(section).__name__}.")
    _validate_keys(section, f"the [slurm] section of {config_file_path(path)}")
    return section


def write_slurm_config(path: Optional[str] = None, *, replace: bool = False, **fields: Any) -> str:
    """Create or update the user config file with default slurm settings.

    This is the supported way to set up cluster-specific defaults (partition, account,
    constraint, ``tmp_root``, ...) instead of editing the file by hand. Provided fields are
    merged into the existing ``[slurm]`` table by default (so the file can be built up over
    several calls); ``None`` values are skipped, and any other top-level tables in the file
    (reserved for future named profiles) are preserved.

    Args:
        path: Path to write to. ``None`` resolves the default location (see
            :func:`config_file_path`); the parent directory is created if needed.
        replace: If ``True``, replace the whole ``[slurm]`` table instead of merging into it.
        **fields: Default field values to store. Each name must be a valid ``SlurmConfig``
            field.

    Returns:
        The path that was written.

    Raises:
        ValueError: If ``fields`` contains an unknown field name.
    """
    _validate_keys(fields, "write_slurm_config()")
    provided = {k: v for k, v in fields.items() if v is not None}
    fp = config_file_path(path)
    data = _parse_toml(fp)
    section = {} if replace else dict(data.get("slurm", {}))
    section.update(provided)
    data["slurm"] = section

    import tomli_w  # local import: only the writer needs the (optional-at-runtime) dependency.

    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "wb") as f:
        tomli_w.dump(data, f)
    return str(fp)
