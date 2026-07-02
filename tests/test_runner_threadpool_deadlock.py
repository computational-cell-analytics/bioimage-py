"""Regression test for a local-runner deadlock.

``LocalRunner._run_pool`` limits nested BLAS/OpenMP parallelism to one thread per worker. It used
to do this with ``@threadpool_limits.wrap(limits=1)``, which constructs a fresh
``threadpoolctl.ThreadpoolController`` *on every call* -- and that constructor runs a
``dl_iterate_phdr`` library scan whose ctypes callback needs the GIL while the glibc dynamic-loader
lock is held. In a nested local run (an outer ``map`` whose items each run an inner block-wise
``copy``), the inner pools run inside the outer worker threads, so the scan happens in worker
threads concurrently with first-time imports (e.g. tqdm lazily importing ``multiprocessing``),
which take the loader lock -> GIL/loader-lock deadlock.

The fix builds one process-wide controller at import (main thread) and reuses it via ``.limit()``,
so no controller is constructed inside worker threads. This test guards that invariant.
"""
import threading

import numpy as np
import threadpoolctl

import bioimage_py as bp
from bioimage_py.runner import get_runner


def _inner_copy(index):
    """Per-item work for the outer map: an inner block-wise copy (its own local pool)."""
    src = np.arange(256, dtype="float32").reshape(16, 16)
    out = np.zeros((16, 16), dtype="float32")
    # block_shape set + num_workers > 1 -> goes through the runner pool (not the direct fast path),
    # so the inner pool constructs its limiter inside the outer worker thread (the deadlock shape).
    bp.copy(src, out, block_shape=(8, 8), num_workers=2)
    return int(out.sum())


def test_no_threadpool_controller_built_in_worker_threads(monkeypatch):
    ctor_threads = []
    orig_init = threadpoolctl.ThreadpoolController.__init__

    def spy_init(self, *args, **kwargs):
        ctor_threads.append(threading.current_thread().name)
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(threadpoolctl.ThreadpoolController, "__init__", spy_init)

    # Nested local run mirroring mobie's HTM import path.
    runner = get_runner("local")
    result = runner.map(_inner_copy, n_items=4, num_workers=4, name="nested-map")

    # Completed (no hang) and every inner copy produced the full source sum.
    expected = int(np.arange(256).reshape(16, 16).sum())
    assert result == [expected, expected, expected, expected]

    worker_ctors = [t for t in ctor_threads if t != "MainThread"]
    assert not worker_ctors, (
        "threadpoolctl.ThreadpoolController was constructed inside worker thread(s) "
        f"{worker_ctors!r}; it must be built once at import and reused via .limit() to avoid the "
        "dl_iterate_phdr / loader-lock deadlock."
    )
