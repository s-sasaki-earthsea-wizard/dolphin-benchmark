#!/usr/bin/env python
"""Thread-pool throughput benchmark, mirroring `create_similarities`.

`dolphin.similarity.create_similarities` processes rasters as 512x512 blocks
through `io.process_blocks` with `num_threads=5` (its default). Single-call
latency therefore understates CPU throughput for both implementations:

- numba releases the GIL (`nogil=True, parallel=True`) and saturates cores
  from within one call;
- the JAX version's sort/top_k stage is single-threaded per call, but
  concurrent calls from the block thread pool run in parallel.

This benchmark times `n_blocks` blocks of (n_ifg, 512, 512) pushed through a
`ThreadPoolExecutor(num_threads)`, matching the workflow's concurrency shape.

    JAX_PLATFORMS=cpu  python bench_threaded.py --out bench_threaded_cpu.json
    JAX_PLATFORMS=cuda python bench_threaded.py --out bench_threaded_gpu.json --skip-numba
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import similarity_numba_c2f7c24 as ref_sim  # noqa: E402

from dolphin import similarity as jax_sim  # noqa: E402

SEED = 99
N_BLOCKS = 15
NUM_THREADS = 5  # create_similarities default
N_IFG = 20
BLOCK = 512
RADIUS = 7


def run_pool(func, blocks, num_threads):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(num_threads) as pool:
        futures = [
            pool.submit(func, b, search_radius=RADIUS) for b in blocks
        ]
        for f in futures:
            f.result()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-numba", action="store_true")
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--num-threads", type=int, default=NUM_THREADS)
    args = parser.parse_args()

    import jax
    import numba

    rng = np.random.default_rng(SEED)
    blocks = []
    for _ in range(args.n_blocks):
        phases = rng.uniform(-np.pi, np.pi, size=(N_IFG, BLOCK, BLOCK)).astype(
            "float32"
        )
        blocks.append(np.exp(1j * phases).astype("complex64"))

    results = {}
    print(
        f"{args.n_blocks} blocks of ({N_IFG}, {BLOCK}, {BLOCK}) r={RADIUS}, "
        f"{args.num_threads} threads, backend={jax.default_backend()}"
    )

    if not args.skip_numba:
        # Warmup (numba JIT compile) outside the timed region
        ref_sim.median_similarity(blocks[0], search_radius=RADIUS)
        elapsed = run_pool(ref_sim.median_similarity, blocks, args.num_threads)
        results["numba"] = elapsed
        print(f"numba pool:      {elapsed:8.3f}s "
              f"({elapsed / args.n_blocks:.3f}s/block)")

    jax_sim.median_similarity(blocks[0], search_radius=RADIUS)  # compile
    elapsed = run_pool(jax_sim.median_similarity, blocks, args.num_threads)
    results[f"jax_{jax.default_backend()}"] = elapsed
    print(f"jax:{jax.default_backend():4s} pool:   {elapsed:8.3f}s "
          f"({elapsed / args.n_blocks:.3f}s/block)")

    report = {
        "n_blocks": args.n_blocks,
        "num_threads": args.num_threads,
        "n_ifg": N_IFG,
        "block": BLOCK,
        "radius": RADIUS,
        "summary": "median",
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "numba_version": numba.__version__,
        "cpu_count": os.cpu_count(),
        "seed": SEED,
        "elapsed_s": results,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
