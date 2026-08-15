#!/usr/bin/env python
"""Empirically determine whether dolphin's phase linking runs on GPU or CPU.

Background (issue #8): an nsys trace of a full ``dolphin run`` captured **zero**
CUDA kernels and GPU memory stayed flat at 69 MiB, even though the config had
``gpu_enabled: true`` and a separate ``import dolphin; jax.default_backend()``
probe returned ``gpu``. This script settles the question by replicating
dolphin's exact init order and measuring, at each stage:

  * ``jax.default_backend()`` and ``jax.devices()``
  * which device a freshly-computed jnp array actually lives on
  * GPU memory in use (from JAX's allocator stats), to detect whether *any*
    GPU op ran in this process

JAX initialises its backend exactly once (lazily, on first op or device query),
so each stage must run in a **fresh process**. Select a stage with argv[1]:

  bare       : import jax only, then matmul            (sanity: GPU works)
  dolphin    : import dolphin, default_backend          (the original probe)
  threads    : import dolphin + set_num_threads(1), matmul   (XLA_FLAGS effect)
  phaselink  : full dolphin init order + run_phase_linking on a synthetic stack

Run via the benchmark container, e.g.:

  ./docker/run.sh python /dolphin-benchmark/scripts/probe_jax_backend.py phaselink
"""

from __future__ import annotations

import sys


def _gpu_mem_mib() -> str:
    """Bytes-in-use on the first GPU device, or a reason it's unavailable."""
    import jax

    gpu_devs = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devs:
        return "n/a (no gpu device)"
    try:
        stats = gpu_devs[0].memory_stats()
    except Exception as e:  # noqa: BLE001
        return f"n/a (memory_stats failed: {e})"
    if not stats:
        return "n/a (empty stats)"
    in_use = stats.get("bytes_in_use", 0) / 2**20
    peak = stats.get("peak_bytes_in_use", 0) / 2**20
    limit = stats.get("bytes_limit", 0) / 2**20
    # peak_bytes_in_use is the monotonic high-water mark — the decisive signal,
    # since run_phase_linking frees its jax arrays (converts to numpy) before
    # returning, which resets bytes_in_use to 0 even if the GPU *was* used.
    return f"in_use={in_use:.1f} MiB peak={peak:.1f} MiB / limit={limit:.1f} MiB"


def _report(tag: str) -> None:
    import jax

    print(f"[{tag}] default_backend = {jax.default_backend()!r}")
    print(f"[{tag}] devices         = {jax.devices()}")
    print(f"[{tag}] gpu memory      = {_gpu_mem_mib()}")


def _matmul_device(tag: str) -> None:
    """Run a real op and report which device the result array lives on."""
    import jax.numpy as jnp

    x = jnp.ones((2048, 2048), dtype=jnp.complex64)
    y = (x @ x).block_until_ready()
    # .devices() returns the set of devices the array is committed/placed on.
    print(f"[{tag}] matmul result device = {y.devices()}")
    print(f"[{tag}] gpu memory AFTER op  = {_gpu_mem_mib()}")


def stage_bare() -> None:
    _report("bare")
    _matmul_device("bare")


def stage_dolphin() -> None:
    import dolphin  # noqa: F401

    _report("dolphin")
    _matmul_device("dolphin")


def stage_threads() -> None:
    import dolphin  # noqa: F401
    from dolphin import utils

    # Mirror displacement.run(): gpu_enabled=True so disable_gpu() is NOT called;
    # set_num_threads() is, which overwrites XLA_FLAGS.
    print("[threads] calling utils.set_num_threads(1) ...")
    utils.set_num_threads(1)
    _report("threads")
    _matmul_device("threads")


def stage_phaselink() -> None:
    import numpy as np

    import dolphin  # noqa: F401
    from dolphin import utils
    from dolphin._types import HalfWindow, Strides
    from dolphin.phase_link import _core

    # Exact init order from workflows/displacement.run() with gpu_enabled=True:
    #   (disable_gpu skipped) -> set_num_threads(threads_per_worker=1)
    utils.set_num_threads(1)
    _report("phaselink/pre")

    print("[phaselink] gpu memory BEFORE run_phase_linking =", _gpu_mem_mib())

    # Synthetic SLC stack: 15 acquisitions, 256x256 — small but enough to force
    # the covariance + EMI jax kernels to execute.
    rng = np.random.default_rng(0)
    shape = (15, 256, 256)
    slc_stack = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)

    out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        strides=Strides(1, 1),
        compute_crlb=False,
    )
    # Touch the result so any lazy work is forced.
    _ = np.asarray(out.cpx_phase).sum()
    print("[phaselink] run_phase_linking completed, output dtype:",
          np.asarray(out.cpx_phase).dtype)
    print("[phaselink] gpu memory AFTER run_phase_linking  =", _gpu_mem_mib())


STAGES = {
    "bare": stage_bare,
    "dolphin": stage_dolphin,
    "threads": stage_threads,
    "phaselink": stage_phaselink,
}


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "phaselink"
    if stage not in STAGES:
        print(f"unknown stage {stage!r}; choose from {list(STAGES)}")
        raise SystemExit(2)
    print(f"=== stage: {stage} (python {sys.version.split()[0]}) ===")
    STAGES[stage]()


if __name__ == "__main__":
    main()
