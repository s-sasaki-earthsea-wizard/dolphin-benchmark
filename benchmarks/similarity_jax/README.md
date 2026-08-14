# similarity.py: numba → JAX equivalence and benchmarks

Evidence package for replacing the numba implementation of
`dolphin/similarity.py` with a JAX implementation, as part of upstream issue
[isce-framework/dolphin#545](https://github.com/isce-framework/dolphin/issues/545)
(remove numba as a dependency).

## What is compared

- **Reference**: `similarity_numba_c2f7c24.py` — byte-identical copy of
  `src/dolphin/similarity.py` from upstream commit `c2f7c24`
  (git blob `d146b8e7e7af3f87d91bd0141d939262d414bd45`), the
  `numba.njit(parallel=True)` per-pixel loop.
- **Under test**: the `feature/545-jax-similarity` branch of the dolphin
  clone (editable install in the dev container), which replaces the loop with
  a jitted JAX kernel:
  - edge-replicating pad + `lax.dynamic_slice` per neighbor offset
    (mathematically identical to the reference's per-axis index clipping),
  - `lax.map` over offsets (bounds peak memory at one
    `(n_ifg, rows, cols)` intermediate),
  - median via `lax.top_k` partial selection (`_nanmedian_top_k`), computing
    the same two middle order statistics as `np.nanmedian`.

## Equivalence (`compare_similarity.py`)

Tolerance `rtol=1e-5, atol=5e-6`, justified by the reference accumulating in
float64 and casting to float32 on output, while JAX computes in float32
throughout. Observed differences are ~1e-7 (float32 rounding).

- CPU backend: **36/36 PASS** (`compare_cpu.json`)
- GPU backend: **36/36 PASS** (`compare_gpu.json`)

Both backends include two real-data cases built from OPERA CSLC tutorial
granules (track 78, T078-165573-IW2): a fully-valid interior window and a
window straddling the valid/invalid burst boundary (119k valid / 143k NaN
pixels). NaN patterns match exactly in every case.

Cases cover: multiple radii (2–11), median/max, random and pathological
masks, non-bool masks (int with value 2, float 0/1 — the numba loop used
truthiness semantics), all-NaN / all-zero invalid pixels, scattered
single-ifg NaNs, float32 and float64 phase (non-complex) inputs, images
smaller than the search radius, and single-row images. One caveat worth
stating in a PR: float64 phase input is computed in complex128 by the numba
reference but downcast to complex64 by the JAX version.

## Benchmarks

Environment: 16-core host, NVIDIA RTX 5080 (16 GB), CUDA 12.6 container,
JAX 0.10.1, numba from the dolphin conda env. Steady-state medians of 5 reps;
first-call (JIT compile) times recorded separately in the JSONs.

### Single call (`bench_similarity.py`)

`median_similarity`, latency of one call:

| n_ifg × rows × cols, radius | numba | JAX CPU | JAX GPU |
|---|---|---|---|
| 20 × 512², r=7  | 2.09 s | 3.41 s | 0.41 s |
| 20 × 512², r=11 | 3.03 s | 8.66 s | 0.44 s |
| 30 × 1024², r=7 | 9.76 s | 16.1 s | 3.56 s |

### Caller-shaped throughput (`bench_callers.py`, `run_callers_matrix.sh`)

`create_similarities` defaults to `num_threads=5`, but **neither in-repo
call site uses the default** (caught in VECR team review — an earlier
version of this README benchmarked only the default shape):

- `workflows/single.py` passes `num_threads=1`, which selects the
  synchronous `DummyProcessPoolExecutor` — fully sequential single calls —
  with the default `search_radius=7`;
- `workflows/sequential.py` passes `num_threads=2`, `search_radius=11`.

Median s/block over 3 passes, one process per cell (isolating numba's
threading env and JIT state). "real" blocks are tiled from 21 OPERA CSLC
granules (n_ifg=20) and include NaN borders (0–100% NaN per block);
"capped" sets `NUMBA_NUM_THREADS` so pool × numba threads ≈ 16 cores:

| config | dataset | numba | numba (capped) | JAX CPU | JAX GPU |
|---|---|---|---|---|---|
| single (1 thread, r=7) | synthetic | 2.17 | – | 3.26 | 0.42 |
| single (1 thread, r=7) | real | 2.01 | – | 2.99 | 0.37 |
| sequential (2 threads, r=11) | synthetic | 1.93 | 1.72 | 4.37 | 0.29 |
| sequential (2 threads, r=11) | real | 1.95 | 1.65 | 4.24 | 0.31 |
| default (5 threads, r=7) | synthetic | 1.21 | 1.18 | 0.83 | – |
| default (5 threads, r=7) | real | 1.10 | 1.04 | 0.66 | – |
| sequential5 (5 threads, r=11)* | synthetic | 1.60 | – | 2.17 | – |
| sequential5 (5 threads, r=11)* | real | 1.26 | – | 1.65 | – |

\* exploratory: sequential.py's radius at the default thread count — not an
existing call site; probes whether a thread bump alone would flip r=11.

Readings (ratios computed from the raw JSON medians, not the rounded table):

- **In the two configurations dolphin actually runs, the JAX version is
  1.5× (single) to 2.2–2.3× (sequential) slower on CPU**, and 5.1–6.7×
  faster on GPU. The sequential config is the worst CPU case because
  `top_k` cost grows superlinearly with the neighbor count (r=11 → K=348).
- The default 5-thread shape is the one place JAX CPU wins: 1.5–1.7× vs
  uncapped numba, and still 1.4–1.6× vs the `NUMBA_NUM_THREADS`-capped
  control — so the win is genuine, not just numba's pool-thread
  contention. But no in-repo caller uses that shape, and the win does
  **not** extend to r=11: at 5 threads the JAX version is still 1.3–1.4×
  slower there (sequential5 rows) — a caller-side thread bump flips the
  r=7 path only.
- Real NaN-bearing blocks shift both implementations only mildly (the numba
  loop skips masked/NaN centers; the JAX kernel computes densely).

`bench_threaded.py` / `bench_threaded_*.json` are the earlier default-shape
measurement, kept for provenance.

### Why single-call CPU is slower: XLA sort (`diag_cpu_breakdown.py`)

The per-offset slice+multiply machinery is on par with numba
(`max_similarity`: 1.4 s vs numba's 2.1 s total). The gap is entirely the
median reduction — XLA's CPU sort is single-threaded and slow:

| stage on (K, 512, 512) cube | K=136 | K=348 |
|---|---|---|
| `jnp.nanmax`               | 0.06 s | 0.15 s |
| `jnp.sort` (axis 0)        | 4.39 s | 16.1 s |
| `jnp.sort` (last axis)     | 3.97 s | 12.3 s |
| `lax.top_k` (k = K/2 + 1)  | 2.07 s | 6.23 s |

Hence the `lax.top_k`-based median in the implementation (bit-identical to
`np.nanmedian` on this cube, verified in the same script).

## Reproducing

Inside the dev container (`make shell`, or `./docker/run.sh '<cmd>'`):

```bash
cd /dolphin-benchmark/benchmarks/similarity_jax
JAX_PLATFORMS=cpu  python compare_similarity.py --out compare_cpu.json --real-data-dir /cslc
JAX_PLATFORMS=cuda python compare_similarity.py --out compare_gpu.json --real-data-dir /cslc
JAX_PLATFORMS=cpu  python bench_similarity.py   --out bench_cpu.json
JAX_PLATFORMS=cuda python bench_similarity.py   --out bench_gpu.json --skip-numba
bash run_callers_matrix.sh   # caller-shaped matrix -> results_callers/
```

The container image predates some current dolphin deps; each invocation
above assumes `pip install --no-deps -e /dolphin && pip install -U opera-utils`
has run in the container first.

## Disclosure

Implementation, harness, and measurements were produced with assistance from
Claude Code (model: Claude Fable 5). All results were generated by executing
the scripts in this directory; numbers in this README come from the JSON
files committed alongside it.
