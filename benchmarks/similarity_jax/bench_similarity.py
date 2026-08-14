#!/usr/bin/env python
"""Walltime benchmark: numba `similarity` reference vs the JAX rewrite.

Reference: vendored `similarity_numba_c2f7c24.py` (upstream numba version).
Under test: installed `dolphin.similarity` (feature/545-jax-similarity).

Run inside the dev container, once per JAX backend:

    JAX_PLATFORMS=cpu  python bench_similarity.py --out bench_cpu.json
    JAX_PLATFORMS=cuda python bench_similarity.py --out bench_gpu.json --skip-numba

The numba timings are backend-independent, so `--skip-numba` avoids
re-measuring them on the GPU run.

Each measurement reports the first (JIT/compile + run) call separately from
the steady-state repeats, since both numba and JAX pay a one-time
compilation cost per shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import similarity_numba_c2f7c24 as ref_sim  # noqa: E402

from dolphin import similarity as jax_sim  # noqa: E402

SEED = 42
REPS = 5

# (n_ifg, rows, cols, search_radius) -- 512x512 is the default
# `process_blocks` block size used by `create_similarities`
CONFIGS = [
    (20, 512, 512, 7),
    (20, 512, 512, 11),
    (30, 1024, 1024, 7),
]


def make_stack(rng, n, rows, cols):
    phases = rng.uniform(-np.pi, np.pi, size=(n, rows, cols)).astype("float32")
    return np.exp(1j * phases).astype("complex64")


def time_one(func, stack, radius, reps):
    """Time `func` on `stack`: first call, then `reps` steady-state calls."""
    t0 = time.perf_counter()
    func(stack, search_radius=radius)
    first_s = time.perf_counter() - t0

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        func(stack, search_radius=radius)
        times.append(time.perf_counter() - t0)
    return first_s, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-numba", action="store_true")
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()

    import jax
    import numba

    rng = np.random.default_rng(SEED)
    results = []
    for n, rows, cols, radius in CONFIGS:
        stack = make_stack(rng, n, rows, cols)
        entry = {
            "n_ifg": n,
            "rows": rows,
            "cols": cols,
            "radius": radius,
            "summary": "median",
        }

        if not args.skip_numba:
            first_s, times = time_one(
                ref_sim.median_similarity, stack, radius, args.reps
            )
            entry["numba"] = {"first_s": first_s, "times_s": times}
            print(
                f"numba          n={n} {rows}x{cols} r={radius}: "
                f"first={first_s:.3f}s steady={np.median(times):.3f}s"
            )

        first_s, times = time_one(jax_sim.median_similarity, stack, radius, args.reps)
        entry[f"jax_{jax.default_backend()}"] = {"first_s": first_s, "times_s": times}
        print(
            f"jax:{jax.default_backend():4s}       n={n} {rows}x{cols} r={radius}: "
            f"first={first_s:.3f}s steady={np.median(times):.3f}s"
        )
        results.append(entry)

    report = {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_version": jax.__version__,
        "numba_version": numba.__version__,
        "cpu_count": os.cpu_count(),
        "reps": args.reps,
        "seed": SEED,
        "results": results,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
