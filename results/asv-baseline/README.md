# ASV baseline for dolphin's existing benchmark suite

Absolute-number baseline runs of upstream dolphin's `benchmarks/benchmarks.py`
on this project's reference machine, using asv with the container's existing
`dolphin-env`. Tracked in
[dolphin-benchmark#2](https://github.com/s-sasaki-earthsea-wizard/dolphin-benchmark/issues/2).

## Layout

```
gpu/   R2 — JAX_PLATFORMS=cuda
cpu/   R3 — JAX_PLATFORMS=cpu
```

The two backends get **physically separate results trees**. asv's `existing`
environment type ignores the requirement matrix entirely (`get_environments()`
in `asv/environment.py`: "Ignore requirement matrix"), so `env_nobuild` cannot
split env names, and both backends would otherwise write the identically named
result file — the second run silently overwriting the first.

Result JSONs exist at all only because the runs pass
`--set-commit-hash $(git -C /dolphin rev-parse HEAD)`: for `existing`
environments asv silently skips saving results otherwise.

## Reproduction

```bash
make asv-smoke          # R1: --quick sweep of all 7 benchmarks (GPU)
make asv-baseline-gpu   # R2: full run, working benchmarks only
make asv-baseline-cpu   # R3: same on CPU
```

Configs are generated from upstream `asv.conf.json` by `asv/generate-confs.sh`
(never hand-edited; regenerate with `make asv-confs`).

## Machine

| item | value |
|---|---|
| machine name | `dolphin-bench` (fixed via compose `hostname:`) |
| CPU | Intel Core Ultra 9 285H, 16 threads |
| RAM | 94 GiB |
| GPU | NVIDIA RTX 5080, 16 GiB |
| host driver / CUDA | 590.48 / CUDA 13.1 |
| container | `dolphin-benchmark:gpu` (CUDA 12.6.3, Ubuntu 22.04) |
| OS (host kernel) | Linux 6.17.0-29-generic |

## Software (existing `dolphin-env`)

| item | value |
|---|---|
| Python | 3.14 (upstream CI's mamba config pins 3.11 — see caveats) |
| dolphin | `49975fce` (`feature/545-jax-similarity`; identical to upstream `main` c2f7c24 except `similarity.py`, which no benchmark touches) |
| jax | 0.11.0 |
| asv / asv_runner | 0.6.6 / 0.3.0 |
| opera-utils | pip `git+…@main`, resolved `97f77d93d04e20cced1938c6282c30b547951600` |

## Environment controls

- `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` — same as
  upstream `benchmark.yml`.
- `NUMBA_NUM_THREADS` **unset** — matches upstream CI; also moot here, since
  none of the working benchmarks touch numba (the two numba-dependent ones are
  broken, see below).
- `JAX_PLATFORMS=cuda` (R2) / `cpu` (R3) — verified via
  `jax.default_backend()` → `gpu` / `cpu`.
- Image defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` — **non-default JAX GPU behavior**;
  upstream CI is CPU-only so this affects only the GPU absolute numbers.
- GPU checked idle before R2 (`nvidia-smi`: 0 %, 69 MiB, 51 °C).

## Broken benchmarks (excluded from R2/R3, recorded by the R1 smoke)

1. `ShpBenchmark.time_estimate_neighbors` — `TypeError: tuple indices must be
   integers or slices, not str` at `benchmarks.py:134` (`HALF_WINDOW["y"]`;
   `HalfWindow` became a NamedTuple in upstream #203). Upstream bug,
   asv-level `failed` marker confirmed in R1.
2. `SingleMinistackBenchmark.time_single_ministack` — setup_cache phase dies:
   asv_runner 0.3.0's `do_setup_cache` (asv#1592 follow-up) now runs
   parameter-free `setup()` hooks *before* `setup_cache()`, and this
   benchmark's `setup` asserts on files that `setup_cache` creates. Breaks on
   any current asv_runner install, including upstream CI's.

## Caveats on the numbers

- Python 3.14 here vs 3.11 in upstream CI's mamba environment — absolute
  numbers are not directly comparable to a CI run (R4/R5 cover that).
- Benchmark inputs are generated in-process (random arrays); no NAS I/O is on
  the measured path for the working benchmarks.
- `--quick` R1 numbers are single-sample; only R2/R3 numbers below are
  asv-standard repeated timings.

## Results

Commit `49975fce`, asv-standard repeated timings (R2/R3). Wall-clock:
R2 ≈ 2 min, R3 ≈ 3.5 min.

### Timings

| benchmark | params | GPU (R2) | CPU (R3) | GPU speedup |
|---|---|---|---|---|
| CovarianceBenchmark.time_covariance_single | nslc=10 | 0.54 ms | 0.14 ms | 0.27× |
| | nslc=20 | 0.45 ms | 0.14 ms | 0.30× |
| | nslc=30 | 0.42 ms | 0.15 ms | 0.35× |
| CovarianceBenchmark.time_covariance_stack | nslc=10 | 4.6 ms | 4.3 ms | 0.92× |
| | nslc=20 | 11.8 ms | 10.1 ms | 0.86× |
| | nslc=30 | 17.4 ms | 15.6 ms | 0.89× |
| PhaseLinkingBenchmark.time_phase_link | 10, evd=True | 30.0 ms | 448 ms | 14.9× |
| | 10, evd=False | 31.9 ms | 1.56 s | 48.8× |
| | 20, evd=True | 64.5 ms | 1.07 s | 16.6× |
| | 20, evd=False | 70.1 ms | 1.22 s | 17.4× |
| | 30, evd=True | 106 ms | 2.14 s | 20.1× |
| | 30, evd=False | 110 ms | 3.35 s | 30.6× |

### Peak memory (process RSS)

| benchmark | params | GPU (R2) | CPU (R3) |
|---|---|---|---|
| CovarianceBenchmark.peakmem_covariance_stack | nslc=10/20/30 | 1.25 / 1.34 / 1.37 GB | 0.96 / 0.98 / 1.00 GB |
| PhaseLinkingBenchmark.peakmem_phase_link | 10, evd=T/F | 1.40 / 1.40 GB | 1.77 / 1.75 GB |
| | 20, evd=T/F | 1.52 / 1.53 GB | 2.65 / 2.67 GB |
| | 30, evd=T/F | 1.57 / 1.60 GB | 3.51 / 3.53 GB |

### Reading the numbers

- The R1 `--quick` value for `time_covariance_single` was ~396 ms — a single
  sample that includes JAX JIT compilation. The repeated-timing value here
  (0.4–0.5 ms) is the amortized per-call cost. Keep this in mind for any
  `--quick` number.
- The two small covariance benchmarks favor CPU (transfer overhead dominates
  at this size); phase linking is where the GPU pays off (15–49×), and CPU
  peak RSS also grows much faster with stack size there.

## asv compare demo

The GPU tree also contains a run labeled `c2f7c24` (upstream `main`), taken
from the same working tree: the branch's only diff vs `c2f7c24` is
`similarity.py` (+ its tests), which no benchmark imports, so the measured
code paths are identical. This doubles as a miniature no-regression check and
as the tooling demo for the dolphin#545 PR workflow:

```bash
docker/run.sh 'asv compare c2f7c24 49975fce --machine dolphin-bench \
    --config /dolphin-benchmark/asv/asv-existing-gpu.conf.json'
```

Output (all within noise; the empty Change column means asv flags no
significant difference):

```
| Change   | Before [c2f7c24a] <ci-baseline/upstream-c2f7c24>   | After [49975fce] <feature/545-jax-similarity>   | Ratio   | Benchmark (Parameter)                                          |
|----------|----------------------------------------------------|-------------------------------------------------|---------|----------------------------------------------------------------|
|          | 1.25G                                              | 1.25G                                           | 1.00    | benchmarks.CovarianceBenchmark.peakmem_covariance_stack(10)    |
|          | 1.34G                                              | 1.34G                                           | 1.00    | benchmarks.CovarianceBenchmark.peakmem_covariance_stack(20)    |
|          | 1.37G                                              | 1.37G                                           | 1.00    | benchmarks.CovarianceBenchmark.peakmem_covariance_stack(30)    |
|          | 374±60μs                                           | 538±60μs                                        | ~1.44   | benchmarks.CovarianceBenchmark.time_covariance_single(10)      |
|          | 434±80μs                                           | 454±60μs                                        | 1.05    | benchmarks.CovarianceBenchmark.time_covariance_single(20)      |
|          | 433±7μs                                            | 419±20μs                                        | 0.97    | benchmarks.CovarianceBenchmark.time_covariance_single(30)      |
|          | 4.52±0.1ms                                         | 4.64±0.2ms                                      | 1.02    | benchmarks.CovarianceBenchmark.time_covariance_stack(10)       |
|          | 11.9±0.2ms                                         | 11.8±0.09ms                                     | 0.99    | benchmarks.CovarianceBenchmark.time_covariance_stack(20)       |
|          | 17.1±0.1ms                                         | 17.4±0.3ms                                      | 1.01    | benchmarks.CovarianceBenchmark.time_covariance_stack(30)       |
|          | 1.41G                                              | 1.4G                                            | 0.99    | benchmarks.PhaseLinkingBenchmark.peakmem_phase_link(10, False) |
|          | 1.4G                                               | 1.4G                                            | 1.00    | benchmarks.PhaseLinkingBenchmark.peakmem_phase_link(10, True)  |
|          | 1.54G                                              | 1.53G                                           | 1.00    | benchmarks.PhaseLinkingBenchmark.peakmem_phase_link(20, False) |
|          | 1.52G                                              | 1.52G                                           | 1.00    | benchmarks.PhaseLinkingBenchmark.peakmem_phase_link(20, True)  |
|          | 1.59G                                              | 1.6G                                            | 1.00    | benchmarks.PhaseLinkingBenchmark.peakmem_phase_link(30, False) |
|          | 1.57G                                              | 1.57G                                           | 1.00    | benchmarks.PhaseLinkingBenchmark.peakmem_phase_link(30, True)  |
|          | 33.0±1ms                                           | 31.9±2ms                                        | 0.97    | benchmarks.PhaseLinkingBenchmark.time_phase_link(10, False)    |
|          | 31.7±0.8ms                                         | 30.0±0.6ms                                      | 0.95    | benchmarks.PhaseLinkingBenchmark.time_phase_link(10, True)     |
|          | 69.7±4ms                                           | 70.1±3ms                                        | 1.00    | benchmarks.PhaseLinkingBenchmark.time_phase_link(20, False)    |
|          | 66.4±2ms                                           | 64.5±3ms                                        | 0.97    | benchmarks.PhaseLinkingBenchmark.time_phase_link(20, True)     |
|          | 107±4ms                                            | 110±2ms                                         | 1.03    | benchmarks.PhaseLinkingBenchmark.time_phase_link(30, False)    |
|          | 99.9±2ms                                           | 106±2ms                                         | 1.06    | benchmarks.PhaseLinkingBenchmark.time_phase_link(30, True)     |
| !        | n/a                                                | failed                                          | n/a     | benchmarks.ShpBenchmark.time_estimate_neighbors                |
| !        | n/a                                                | failed                                          | n/a     | benchmarks.SingleMinistackBenchmark.time_single_ministack      |
```

Notes on the table:

- The `Before` branch label `<ci-baseline/upstream-c2f7c24>` is asv resolving
  c2f7c24 to the fork branch of the same name.
- The two `!` rows are a bookkeeping artifact: the R1 smoke recorded the
  broken benchmarks' `failed` status under `49975fce`, while the
  c2f7c24-labeled run used `-b` to select working benchmarks only, so the
  Before side shows `n/a` rather than an equally failed result.
- `time_covariance_single(10)`'s ~1.44 ratio sits inside its ±60 μs noise
  band (asv prints `~` and no significance marker for exactly that reason).

