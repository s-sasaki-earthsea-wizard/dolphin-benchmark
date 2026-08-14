#!/usr/bin/env python
"""Attribute JAX-CPU walltime: shifted-slice machinery vs the median reduction.

Times, on the same (20, 512, 512) stack:
  1. full median_similarity / max_similarity (installed dolphin)
  2. jnp.nanmedian / jnp.nanmax / jnp.sort alone on a synthetic
     (n_neighbors, rows, cols) similarity cube

If max_similarity is much faster than median_similarity, the per-offset
slice+multiply machinery is fine and the nanmedian sort dominates.

    JAX_PLATFORMS=cpu python diag_cpu_breakdown.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp

from dolphin import similarity as jax_sim

SEED = 7
REPS = 3


def timeit(label, fn):
    fn()  # warmup / compile
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    print(f"{label:40s} {np.median(times):8.3f}s")
    return np.median(times)


def main():
    rng = np.random.default_rng(SEED)
    print(f"backend={jax.default_backend()} cpu_count={os.cpu_count()}")

    phases = rng.uniform(-np.pi, np.pi, size=(20, 512, 512)).astype("float32")
    stack = np.exp(1j * phases).astype("complex64")

    for radius in [7, 11]:
        n_neighbors = len(jax_sim.get_circle_idxs(radius))
        print(f"--- radius={radius} (n_neighbors={n_neighbors}) ---")
        timeit(
            f"median_similarity r={radius}",
            lambda r=radius: jax_sim.median_similarity(stack, search_radius=r),
        )
        timeit(
            f"max_similarity r={radius}",
            lambda r=radius: jax_sim.max_similarity(stack, search_radius=r),
        )

        # Isolated summary-stage costs on a synthetic similarity cube
        sims_np = rng.uniform(-1, 1, size=(n_neighbors, 512, 512)).astype("float32")
        sims_np[:, rng.random((512, 512)) < 0.05] = np.nan
        sims = jnp.asarray(sims_np)

        nanmedian_j = jax.jit(lambda x: jnp.nanmedian(x, axis=0))
        nanmax_j = jax.jit(lambda x: jnp.nanmax(x, axis=0))
        sort_j = jax.jit(lambda x: jnp.sort(x, axis=0))
        timeit(f"jnp.nanmedian cube K={n_neighbors}",
               lambda: nanmedian_j(sims).block_until_ready())
        timeit(f"jnp.nanmax cube K={n_neighbors}",
               lambda: nanmax_j(sims).block_until_ready())
        timeit(f"jnp.sort cube K={n_neighbors}",
               lambda: sort_j(sims).block_until_ready())


def median_variants():
    """Compare median strategies on the (K, rows, cols) cube."""
    rng = np.random.default_rng(SEED)
    for n_neighbors in [136, 348]:
        print(f"--- median variants, K={n_neighbors} ---")
        sims_np = rng.uniform(-1, 1, size=(n_neighbors, 512, 512)).astype("float32")
        sims_np[:, rng.random((512, 512)) < 0.05] = np.nan
        sims = jnp.asarray(sims_np)
        k_half = n_neighbors // 2 + 1

        @jax.jit
        def sort_last(x):
            return jnp.sort(x.transpose(1, 2, 0), axis=-1)

        @jax.jit
        def topk_last(x):
            return jax.lax.top_k(x.transpose(1, 2, 0), k_half)[0]

        @jax.jit
        def nanmedian_custom(x):
            xt = x.transpose(1, 2, 0)
            xs = jnp.sort(xt, axis=-1)  # NaNs sort to the end
            m = jnp.sum(~jnp.isnan(xt), axis=-1)
            lo = jnp.take_along_axis(xs, ((m - 1) // 2)[..., None], axis=-1)
            hi = jnp.take_along_axis(xs, (m // 2)[..., None], axis=-1)
            return ((lo + hi) / 2)[..., 0]

        timeit(f"sort last-axis (transposed) K={n_neighbors}",
               lambda: sort_last(sims).block_until_ready())
        timeit(f"lax.top_k k={k_half} K={n_neighbors}",
               lambda: topk_last(sims).block_until_ready())
        timeit(f"custom nanmedian (sort+gather) K={n_neighbors}",
               lambda: nanmedian_custom(sims).block_until_ready())

        # Correctness spot-check vs numpy
        got = np.asarray(nanmedian_custom(sims))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            want = np.nanmedian(sims_np, axis=0)
        both = ~np.isnan(got) & ~np.isnan(want)
        assert np.array_equal(np.isnan(got), np.isnan(want)), "nan pattern mismatch"
        print(f"  custom vs np.nanmedian max|diff| = "
              f"{np.max(np.abs(got[both] - want[both])):.2e}")


if __name__ == "__main__":
    main()
    median_variants()
