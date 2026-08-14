#!/usr/bin/env python
"""Caller-shaped benchmark: the configurations dolphin actually uses.

`bench_threaded.py` measured the `create_similarities` *default*
(num_threads=5, r=7) — which no in-repo caller uses. The real call sites are:

- `workflows/single.py`:  num_threads=1, search_radius=7 (default radius).
  num_threads=1 selects `DummyProcessPoolExecutor` (synchronous submit), so
  this path is fully sequential single calls.
- `workflows/sequential.py`: num_threads=2, search_radius=11.

This benchmark measures those two shapes (+ the default shape for
reference), on both synthetic all-valid blocks and real OPERA CSLC blocks
(which contain NaN borders — the numba loop skips masked/NaN centers, so
all-valid synthetic data is its worst case), with an optional
NUMBA_NUM_THREADS cap as a control for the "numba threads contend in a
pool" hypothesis.

Each (impl, config, dataset) cell runs in its own process so numba's
threading env and JIT state are isolated:

    python bench_callers.py prep --real-data-dir /cslc --out-dir /tmp/blocks
    python bench_callers.py run --impl numba --config single \
        --dataset real --blocks-dir /tmp/blocks --out result.json
    python bench_callers.py run --impl jax --config sequential ...

`--numba-threads N` (numba impl only) sets NUMBA_NUM_THREADS before numba
is imported.
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

SEED = 20260814
N_IFG = 20
BLOCK = 512

# label -> (num_threads, search_radius, n_blocks), mirroring the call sites
CONFIGS = {
    "single": (1, 7, 5),
    "sequential": (2, 11, 6),
    "default": (5, 7, 10),
}
MAX_BLOCKS = max(n for _, _, n in CONFIGS.values())


def prep(real_data_dir: Path | None, out_dir: Path):
    """Build synthetic and (optionally) real block sets, save as .npy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    synth = np.empty((MAX_BLOCKS, N_IFG, BLOCK, BLOCK), dtype="complex64")
    for i in range(MAX_BLOCKS):
        phases = rng.uniform(-np.pi, np.pi, size=(N_IFG, BLOCK, BLOCK))
        synth[i] = np.exp(1j * phases.astype("float32"))
    np.save(out_dir / "synthetic.npy", synth)
    print(f"synthetic: {synth.shape}")

    if real_data_dir is None:
        return
    import h5py

    files = sorted(Path(real_data_dir).glob("*.h5"))[: N_IFG + 1]
    if len(files) < N_IFG + 1:
        sys.exit(f"need {N_IFG + 1} granules in {real_data_dir}, found {len(files)}")

    # Find the valid-data boundary row (as in compare_similarity.py) so one
    # block straddles it; the rest tile the interior at two column offsets.
    col_starts = [10000, 14000]
    with h5py.File(files[0]) as f:
        strip = f["data/VV"][:, col_starts[0] : col_starts[0] + BLOCK]
    valid_frac = np.isfinite(strip).mean(axis=1)
    first_valid_row = int(np.argmax(valid_frac > 0.1))
    edge_start = max(0, first_valid_row - BLOCK // 2)

    windows = [(edge_start, col_starts[0])]
    row = 2000
    while len(windows) < MAX_BLOCKS:
        for c in col_starts:
            if len(windows) >= MAX_BLOCKS:
                break
            windows.append((row, c))
        row += BLOCK
    print(f"real windows (row, col): {windows}")

    real = np.empty((MAX_BLOCKS, N_IFG, BLOCK, BLOCK), dtype="complex64")
    slcs = np.empty((N_IFG + 1, BLOCK, BLOCK), dtype="complex64")
    for b, (r0, c0) in enumerate(windows):
        for gi, path in enumerate(files):
            with h5py.File(path) as f:
                slcs[gi] = f["data/VV"][r0 : r0 + BLOCK, c0 : c0 + BLOCK]
        real[b] = slcs[0] * np.conj(slcs[1:])
        nan_frac = np.isnan(real[b]).mean()
        print(f"  block {b} @ ({r0}, {c0}): nan_frac={nan_frac:.2f}")
    np.save(out_dir / "real.npy", real)


def run(args):
    if args.numba_threads:
        os.environ["NUMBA_NUM_THREADS"] = str(args.numba_threads)

    sys.path.insert(0, str(Path(__file__).parent))
    if args.impl == "numba":
        import similarity_numba_c2f7c24 as sim_mod
    else:
        from dolphin import similarity as sim_mod

    num_threads, radius, n_blocks = CONFIGS[args.config]
    blocks = np.load(Path(args.blocks_dir) / f"{args.dataset}.npy")[:n_blocks]
    func = sim_mod.median_similarity

    def one_pass():
        t0 = time.perf_counter()
        if num_threads == 1:
            # workflows/single.py path: DummyProcessPoolExecutor == a loop
            for b in blocks:
                func(b, search_radius=radius)
        else:
            with ThreadPoolExecutor(num_threads) as pool:
                futs = [pool.submit(func, b, search_radius=radius) for b in blocks]
                for f in futs:
                    f.result()
        return time.perf_counter() - t0

    func(blocks[0], search_radius=radius)  # JIT warmup, untimed
    times = [one_pass() for _ in range(args.reps)]

    backend = None
    if args.impl == "jax":
        import jax

        backend = jax.default_backend()
    result = {
        "impl": args.impl,
        "jax_backend": backend,
        "config": args.config,
        "dataset": args.dataset,
        "num_threads": num_threads,
        "radius": radius,
        "n_blocks": n_blocks,
        "numba_threads_cap": args.numba_threads,
        "reps": args.reps,
        "pass_times_s": times,
        "s_per_block": [t / n_blocks for t in times],
        "median_s_per_block": float(np.median(times)) / n_blocks,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    label = args.impl + (f"[cap{args.numba_threads}]" if args.numba_threads else "")
    if backend:
        label += f":{backend}"
    print(
        f"{label:16s} {args.config:10s} {args.dataset:9s} "
        f"median {result['median_s_per_block']:.3f} s/block "
        f"(passes: {', '.join(f'{t:.2f}' for t in times)})"
    )


def summarize(results_dir: Path):
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        rows.append(json.loads(p.read_text()))
    print(f"\n{'config':10s} {'dataset':9s} {'impl':16s} {'s/block':>8s}")
    for r in sorted(rows, key=lambda r: (r["config"], r["dataset"], r["impl"])):
        label = r["impl"] + (
            f"[cap{r['numba_threads_cap']}]" if r["numba_threads_cap"] else ""
        )
        if r.get("jax_backend"):
            label += f":{r['jax_backend']}"
        print(
            f"{r['config']:10s} {r['dataset']:9s} {label:16s} "
            f"{r['median_s_per_block']:8.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--real-data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)

    p = sub.add_parser("run")
    p.add_argument("--impl", choices=["numba", "jax"], required=True)
    p.add_argument("--config", choices=list(CONFIGS), required=True)
    p.add_argument("--dataset", choices=["synthetic", "real"], required=True)
    p.add_argument("--blocks-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--numba-threads", type=int, default=None)

    p = sub.add_parser("summarize")
    p.add_argument("--results-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "prep":
        prep(args.real_data_dir, args.out_dir)
    elif args.cmd == "run":
        run(args)
    else:
        summarize(args.results_dir)


if __name__ == "__main__":
    main()
