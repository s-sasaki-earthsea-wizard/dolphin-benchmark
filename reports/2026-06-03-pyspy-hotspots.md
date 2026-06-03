# py-spy hotspot report — end-to-end `dolphin run` (issue #1)

**Date**: 2026-06-03
**Question**: are `goldstein` and `calc_ps_block` real hotspots on a realistic
`dolphin run`, i.e. is it worth porting them to JAX/GPU? If something else
dominates, redirect.

## TL;DR

- The workload is **I/O-bound**: GDAL raster reads are ~40 % of wall time and the
  top-3 self-time leaves are all I/O.
- **`calc_ps_block` is negligible (0.8 %)** → porting PS detection is not worth it
  on this workload. **Pivot PR#2.**
- **`goldstein` is modest but real (7.6 %)** → a JAX port has a single-digit-% wall
  ceiling. Worth doing as a clean standalone PR, eyes open.
- **The GPU was never used** (see [GPU section](#gpu-side-nsys)): nsys captured
  zero CUDA kernels and GPU memory stayed at 69 MiB even though `gpu_enabled=true`.
  The whole pipeline ran on CPU. **This is the most important follow-up** — the
  GPU-acceleration premise needs the existing JAX path to actually engage first.

## Run configuration

| | |
|---|---|
| Data | OPERA CSLC tutorial stack, 26 scenes, burst `T078-165573-IW2` (West Texas) |
| Scene shape | (4890, 20646) complex64, subdataset `/data/VV` |
| Config | `--sx 6 --sy 3`, `--worker-settings.gpu-enabled`, `--unwrap-options.run-goldstein`, bbox `735000 3470000 745000 3480000` / EPSG 32613 |
| Parallelism | single process (`n_parallel_bursts=1`, `threads_per_worker=1`, `n_parallel_jobs=1`) |
| Wall time | ~12–13 min cold (phase-linking ≈ 6 min + stitch/unwrap/timeseries ≈ 6–7 min) |
| Peak host memory | 3.69 GB |
| Host | RTX 5080 (16 GB), CUDA 12.6.3 container, JAX 0.10.1, **data on NAS** |

The exact config used is committed next to this report (`dolphin_config.yaml`).

> **`--output-options.bounds` does not reduce compute.** It crops only the output
> extent; phase-linking still ingests the full frame
> (`Total stack size (in pixels): (15, 4890, 20646)`). A real subset needs
> spatially cropped input CSLCs (`gdal_translate`).

## Method

`py-spy 0.4.2`, launch mode, **non-blocking** sampling:

```
py-spy record --nonblocking --rate 100 --format raw \
       -o profile-fullrun.folded -- dolphin run dolphin_config.yaml
```

9845 samples, 443 errors (non-blocking torn reads, ~4.5 %). The `--nonblocking`
flag is **mandatory** here: default stop-the-world sampling deadlocks dolphin's
multithreaded JAX/GDAL process (it gets stuck in `t`/stopped state) on Python 3.14.

Self / total tables: `scripts/folded_hotspots.py`. Flame graph
(`profile-fullrun.svg`): `scripts/flamegraph_from_folded.py` (a dependency-free
`flamegraph.pl` replacement; the raw `.folded` is committed so it can be
regenerated or re-analysed).

## Results

### Top by SELF time (on-CPU leaf)

| rank | self % | function |
|---|---|---|
| 1 | **38.8 %** | `DatasetIONumPy` (osgeo/gdal_array.py) — GDAL raster read into numpy |
| 2 | 6.4 % | `BandRasterIONumPy` (osgeo/gdal_array.py) — GDAL read |
| 3 | 6.3 % | `repack_raster` (dolphin/io/_utils.py) — recompress on write |
| 4 | 3.9 % | `backend_compile_and_load` (jax compiler) — one-time JIT compile |
| 5 | 2.6 % | `open` (rasterio) |
| 6 | 2.6 % | `read_stack` (dolphin/io/_readers.py) |
| 7 | 2.5 % | `Open` (osgeo/gdal.py) |
| 19 | 0.73 % | `apply_pspec` (dolphin/goldstein.py:38) — Goldstein FFT |

### Top by TOTAL time (frame + callees = speed-up ceiling)

| total % | area |
|---|---|
| ~40 % | **GDAL `ReadAsArray` / `load_gdal`** — raster reads (single biggest cost) |
| 41.6 % | `read_stack` — SLC stack load for phase linking (overlaps the above) |
| 26.7 % | unwrap stage (snaphu + goldstein + interpolation) |
| **7.6 %** | **`goldstein`** (PR#1 candidate) |
| 6.7 % | snaphu |
| **0.8 %** | **`calc_ps_block`** (PR#2 candidate) |
| 0.01 % | `estimate_stack_covariance` (already JAX; ran on CPU here, see below) |

## Interpretation

1. **I/O-bound.** Reading the 26 full-frame CSLCs dominates. Even the
   phase-linking stage is read-limited, not compute-limited — its ~6 min is mostly
   `read_stack` time, with the covariance/EVD math barely registering (0.01 %).
2. **PS detection (PR#2) is a rounding error (0.8 %).** Not worth a JAX port on
   this workload. Revisit only if a PS-heavy config or the standalone `create_ps`
   flow shows otherwise.
3. **Goldstein (PR#1) is the only candidate with real (if modest) weight (7.6 %)**,
   and it only runs because we forced `run_goldstein=true` (default is `false`).
   The FFT (`apply_pspec`) is the accelerable core.

### Caveats

- I/O dominance is partly environmental: CSLCs on a **NAS** mount, single process.
  Faster storage / more `n_parallel_bursts` / `threads_per_worker` would raise the
  compute fraction.
- Goldstein's 7.6 % only exists in configs that enable it.

## GPU side (nsys)

Nsight Systems 2024.5.1, `nsys profile -t cuda,nvtx dolphin run ...` (12.2 min run):

- **`nsys stats` reports zero CUDA kernels** (`SQLite does not contain CUDA kernel
  data`) and zero GPU memory ops over the full 12-min run.
- **GPU memory stayed at 69 MiB and utilisation at 0 %** for the entire run (sampled
  every 2 s). JAX preallocates ~75 % of VRAM the first time it runs an op on the GPU;
  the flat 69 MiB means **JAX never executed a single op on the GPU**.
- **Yet `jax.default_backend()` is `gpu` (`CudaDevice(id=0)`) in the dolphin process**
  right after `import dolphin`. A standalone `nsys profile python -c "import jax; (x@x)…"`
  *does* capture GPU kernels, so nsys/CUPTI and JAX-GPU both work in this container.

So `gpu_enabled=true` is honoured (`disable_gpu()` is correctly *not* called —
[displacement.py:195](../../dolphin/src/dolphin/workflows/displacement.py),
[utils.py:135](../../dolphin/src/dolphin/utils.py)) and the parent process selects
the GPU backend — **but the phase-linking math still ran entirely on CPU**.

**This is the headline open question.** The parent has the GPU backend, so the most
likely culprit is the **executor**: dolphin runs phase linking through worker
processes, and JAX-on-GPU does not survive a `fork()` (CUDA can't be reinitialised in
a forked child → JAX silently falls back to CPU). That is a well-known JAX gotcha and
fits the evidence (parent backend = gpu, zero kernels system-wide). It needs
confirming against dolphin's executor/parallelism code before any GPU-port PR: if the
GPU path doesn't engage in the normal multiprocessing workflow, porting goldstein to
JAX/GPU has no payoff regardless of its 7.6 %. (Alternative hypotheses: the covariance
path doesn't dispatch to JAX for this data, or device placement is reset downstream.)

## Recommendation (go / pivot)

- **PR#2 (PS → JAX): pivot.** 0.8 % ceiling.
- **PR#1 (Goldstein → JAX): proceed, eyes open.** 7.6 % ceiling, clean and
  FFT-based — but gated on the GPU actually being used.
- **New P0: investigate why dolphin ran on CPU despite `gpu_enabled=true`.** This
  outranks both ports.

## Artifacts

Tracked here (curated):

- `2026-06-03-pyspy-hotspots.md` — this report
- `profile-fullrun.svg` — flame graph

Everything else is regenerable measurement output and lives under `results/`
(git-ignored scratch): the raw `profile-fullrun.folded`, the full ranked
`profile-hotspots.txt` (`scripts/folded_hotspots.py`), `nsys-fullrun.nsys-rep`,
the `dolphin_config.yaml` actually run, and the run logs. Regenerate the SVG /
rankings from a `.folded` with the two scripts in `scripts/`; reproduce the run
from the `dolphin config …` command above.
