# py-spy hotspot report — end-to-end `dolphin run` (issue #1)

**Date**: 2026-06-03

> **⚠️ Correction (added the same day).** The "GPU side" section of this report
> reached the **wrong conclusion**: the GPU *was* being used. The flat 69 MiB reading
> was a measurement artifact, not an idle GPU. Every affected claim below is struck
> through in place rather than deleted, with a pointer to the verified result — see
> [Correction: the GPU was used](#correction-the-gpu-was-used). ~~**The py-spy hotspot
> numbers are unaffected** and stand as written.~~ *(Also corrected — see below.)*

> **⚠️ Correction 2 (2026-08-16).** The `goldstein` "7.6 %" headline is an
> aggregation artifact — `folded_hotspots.py` summed the totals of *nested* frames,
> counting single samples up to 3×. The correct total is **2.8 %**, and it is a share
> of *Python-active sample time*, not wall time. Wall-clock follow-up measurements
> (`benchmarks/unwrap_breakdown/`) settle the underlying question: `goldstein` +
> `interpolation` together are **< 0.5 %** of the unwrap step at realistic scale.
> Details in [Correction 2](#correction-2-goldstein-totals-were-overcounted).

**Question**: are `goldstein` and `calc_ps_block` real hotspots on a realistic
`dolphin run`, i.e. is it worth porting them to JAX/GPU? If something else
dominates, redirect.

## TL;DR

- The workload is **I/O-bound**: GDAL raster reads are ~40 % of wall time and the
  top-3 self-time leaves are all I/O.
- **`calc_ps_block` is negligible (0.8 %)** → porting PS detection is not worth it
  on this workload. **Pivot PR#2.**
- ~~**`goldstein` is modest but real (7.6 %)** → a JAX port has a single-digit-% wall
  ceiling. Worth doing as a clean standalone PR, eyes open.~~
  *(Overcounted — the correct total is **2.8 %** of Python-active sample time; see
  [Correction 2](#correction-2-goldstein-totals-were-overcounted).)*
- ~~**The GPU was never used**: nsys captured zero CUDA kernels and GPU memory stayed
  at 69 MiB even though `gpu_enabled=true`. The whole pipeline ran on CPU. **This is
  the most important follow-up** — the GPU-acceleration premise needs the existing JAX
  path to actually engage first.~~
  *(This was wrong — the verified result is in the
  [follow-up correction](#correction-the-gpu-was-used) below: phase linking does run on
  the GPU, peak 2581 MiB.)*

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
| ~~26.7 %~~ **21.5 %** | unwrap stage (snaphu wrapper + goldstein; *corrected — see [Correction 2](#correction-2-goldstein-totals-were-overcounted)*) |
| ~~**7.6 %**~~ **2.8 %** | **`goldstein`** (PR#1 candidate; *corrected — nested frames were summed*) |
| 6.7 % | snaphu — *Python wrapper only: the SNAPHU C subprocess is invisible to py-spy* |
| **0.8 %** | **`calc_ps_block`** (PR#2 candidate; unaffected by the overcount) |
| 0.01 % | `estimate_stack_covariance` (already JAX; ~~ran on CPU here~~ *it ran on the GPU — see the [correction](#correction-the-gpu-was-used); its Python-level self time is near-zero either way*) |

## Interpretation

1. **I/O-bound.** Reading the 26 full-frame CSLCs dominates. Even the
   phase-linking stage is read-limited, not compute-limited — its ~6 min is mostly
   `read_stack` time, with the covariance/EVD math barely registering (0.01 %).
2. **PS detection (PR#2) is a rounding error (0.8 %).** Not worth a JAX port on
   this workload. Revisit only if a PS-heavy config or the standalone `create_ps`
   flow shows otherwise.
3. ~~**Goldstein (PR#1) is the only candidate with real (if modest) weight (7.6 %)**~~
   *(corrected to **2.8 %** — see
   [Correction 2](#correction-2-goldstein-totals-were-overcounted))*,
   and it only runs because we forced `run_goldstein=true` (default is `false`).
   The FFT (`apply_pspec`) is the accelerable core.

### Caveats

- I/O dominance is partly environmental: CSLCs on a **NAS** mount, single process.
  Faster storage / more `n_parallel_bursts` / `threads_per_worker` would raise the
  compute fraction.
- Goldstein's ~~7.6 %~~ 2.8 % only exists in configs that enable it.

## GPU side (nsys) — retracted

> **Everything in this section is wrong.** *(The conclusion below was a measurement
> artifact; the verified result is in the
> [follow-up correction](#correction-the-gpu-was-used) that follows this section.)*
> It is struck through rather than deleted, because the way it went wrong is worth
> knowing about.

~~Nsight Systems 2024.5.1, `nsys profile -t cuda,nvtx dolphin run ...` (12.2 min run):~~

- ~~**`nsys stats` reports zero CUDA kernels** (`SQLite does not contain CUDA kernel
  data`) and zero GPU memory ops over the full 12-min run.~~
- ~~**GPU memory stayed at 69 MiB and utilisation at 0 %** for the entire run (sampled
  every 2 s). JAX preallocates ~75 % of VRAM the first time it runs an op on the GPU;
  the flat 69 MiB means **JAX never executed a single op on the GPU**.~~
- ~~**Yet `jax.default_backend()` is `gpu` (`CudaDevice(id=0)`) in the dolphin process**
  right after `import dolphin`. A standalone `nsys profile python -c "import jax; (x@x)…"`
  *does* capture GPU kernels, so nsys/CUPTI and JAX-GPU both work in this container.~~

~~So `gpu_enabled=true` is honoured (`disable_gpu()` is correctly *not* called —
[displacement.py:195](../../dolphin/src/dolphin/workflows/displacement.py),
[utils.py:135](../../dolphin/src/dolphin/utils.py)) and the parent process selects
the GPU backend — **but the phase-linking math still ran entirely on CPU**.~~

~~**This is the headline open question.** The parent has the GPU backend, so the most
likely culprit is the **executor**: dolphin runs phase linking through worker
processes, and JAX-on-GPU does not survive a `fork()` (CUDA can't be reinitialised in
a forked child → JAX silently falls back to CPU). That is a well-known JAX gotcha and
fits the evidence (parent backend = gpu, zero kernels system-wide). It needs
confirming against dolphin's executor/parallelism code before any GPU-port PR: if the
GPU path doesn't engage in the normal multiprocessing workflow, porting goldstein to
JAX/GPU has no payoff regardless of its 7.6 %. (Alternative hypotheses: the covariance
path doesn't dispatch to JAX for this data, or device placement is reset downstream.)~~

---

## Correction: the GPU *was* used

The follow-up that disproved the section above ran the same day.

### What actually happens

A bare `dolphin run` with the same config (fresh work directory, so nothing is skipped
as `already_processed`), sampled with
`nvidia-smi --query-gpu=memory.used,utilization.gpu -lms 200` — i.e. **200 ms instead
of 2 s, and not under a profiler**:

| | |
|---|---|
| Samples | 1799 |
| GPU memory, min | **69 MiB** |
| GPU memory, peak | **2581 MiB** |
| Samples in 0–500 MiB | 965 (I/O phases: stack reads, stitching, PS) |
| Samples in 2500–3000 MiB | 834 (phase linking) |

GPU usage is **bimodal**. 69 MiB is the *idle baseline* — a live CUDA context holding
no arrays — not an unused GPU. It jumps to ~2.5 GB only while phase linking runs, and
the run log shows JAX's `jax.scipy.linalg.cho_solve` warning firing from the EMI step
during exactly those windows.

An independent in-process probe agrees: `run_phase_linking` reports
`peak_bytes_in_use` = **2184 MiB** on `CudaDevice(id=0)`
(`scripts/probe_jax_backend.py`).

### The two mistakes

1. **Sampling too coarsely, under a profiler.** At 2 s intervals a bimodal signal that
   is high for a minority of the run is easy to miss entirely. Worse, the original
   numbers were collected *under `nsys`* — a standalone `nsys profile python -c
   "import jax; (x@x)"` does capture kernels, but under the full dolphin run CUPTI
   appears to have suppressed JAX-GPU execution. **Do not infer "the GPU is idle" from
   a profiled run.**
2. **Reading the wrong memory counter.** `jax.Device.memory_stats()["bytes_in_use"]`
   is a *live* value. `run_phase_linking` converts its result to numpy before
   returning, which frees the JAX arrays, so the counter reads 0 immediately after the
   call — which initially looked like "no GPU work happened". The right counter is
   **`peak_bytes_in_use`**, a monotonic high-water mark.

### The `fork()` hypothesis was also wrong — structurally

The retracted text guessed that phase linking ran in a forked worker where JAX-on-GPU
silently falls back to CPU. That cannot apply here: a **single-burst OPERA run never
forks or spawns**. Both executor levels resolve to `DummyProcessPoolExecutor` — the
outer one in `workflows/displacement.py` and the block loop in `workflows/single.py` —
so `run_phase_linking` is called in-process, on the main thread. The
`mp.get_context("spawn")` nearby in `displacement.py` is only reached when the executor
is a real pool (multiple bursts, or `n_parallel_bursts > 1`).

The fork-after-CUDA gotcha is real in general, but it is not what happened here.
Multi-burst runs use **spawn**, not fork, and spawn re-imports in the child, so JAX-GPU
should survive — still unmeasured.

### What this changes, and what it doesn't

**Unchanged** (py-spy measures Python-level sample time and is unaffected by any of
this): the workload is I/O-bound, ~~`goldstein` is 7.6 %~~ *(later corrected to
2.8 % — see [Correction 2](#correction-2-goldstein-totals-were-overcounted))*,
`calc_ps_block` is 0.8 %.

**Changed**: the "New P0" below is **resolved and closed** — the GPU path engages
normally, so the premise behind the GPU-port work still stands. `goldstein` is
currently numpy FFT on the CPU during unwrapping, and there is genuine room to move it
onto the GPU; the modest wall-time ceiling (~~7.6 %~~ 2.8 %) is a separate matter.

Measurement artifacts: `results/gpu-verify-mem.csv`, `results/gpu_verify_run.sh`
(regenerable scratch), and `scripts/probe_jax_backend.py` (tracked).

## Correction 2: `goldstein` totals were overcounted

**Date**: 2026-08-16

The candidate totals in this report came from `scripts/folded_hotspots.py`,
which summed the per-frame *total* counts of every frame whose name matched
the candidate. For `goldstein` the matched frames are **nested** —
`goldstein` → `apply_goldstein_filter` → `patch_goldstein_filter` — so a
single sample with all three on its stack was counted up to three times:
2.77 % + 1.87 % + 0.74 % + … ≈ 7.6 %.

Counting each sample once (any matched frame on its stack) gives the correct
subsystem total, which necessarily equals the entry frame's own total:

| candidate | reported | corrected |
|---|---|---|
| `goldstein` (total) | 7.6 % | **2.8 %** (274 / 9845 samples; matches `goldstein (goldstein.py:129)` at 2.77 %) |
| unwrap stage (total) | 26.7 % | **21.5 %** (samples passing through `unwrap/_unwrap.py`) |
| `calc_ps_block` (total) | 0.8 % | 0.8 % — unchanged: its matched frames are the same function at different lines, mutually exclusive per sample |

`folded_hotspots.py` is fixed alongside this correction (union counting; the
per-frame breakdown lines were always individually correct and are unchanged).
The PR#2 pivot call is unaffected.

### Two structural limits of this profile, found while correcting

1. **The denominator is Python-active sample time, not wall time.** 9845
   samples at 100 Hz ≈ 98 s, against ~13 min of wall: threads blocked on NAS
   reads or subprocess waits are idle and dropped (py-spy default). Every "%"
   in this report is a share of that ~98 s, and I/O-heavy stages are
   *under*-represented in it.
2. **SNAPHU is invisible.** `snaphu-py` runs the bundled `snaphu` C executable
   in a subprocess (`snaphu._snaphu.run_snaphu`); the "snaphu 6.7 %" row is
   Python-side wrapper I/O only. The unwrap step's dominant cost never appears
   in this profile at all.

Both limits are why the follow-up measured the unwrap step with wall-clock
timers instead: see `benchmarks/unwrap_breakdown/` (2026-08-16). Headline
result: at realistic scale (Sentinel-1 burst and NISAR GUNW frame) the unwrap
step is **98–100 % SNAPHU**, and `goldstein` + `interpolation` together are
**< 0.5 %** of it — there is no performance case for JAX-porting these two
stages. The practical levers are SNAPHU tiling (7.4× measured) and enabling
preprocessing on noisy data (29–34 % faster SNAPHU).

## Recommendation (go / pivot)

- **PR#2 (PS → JAX): pivot.** 0.8 % ceiling.
- ~~**PR#1 (Goldstein → JAX): proceed, eyes open.** 7.6 % ceiling, clean and
  FFT-based — but gated on the GPU actually being used.~~
  *(Superseded. The gate concern was lifted by the
  [GPU correction](#correction-the-gpu-was-used), but the ceiling was
  overcounted (2.8 %, of Python-active time) and the 2026-08-16 wall-clock
  breakdown (`benchmarks/unwrap_breakdown/`) puts goldstein at **< 0.5 %** of
  the unwrap step — **pivot PR#1 as well**. See
  [Correction 2](#correction-2-goldstein-totals-were-overcounted).)*
- ~~**New P0: investigate why dolphin ran on CPU despite `gpu_enabled=true`.** This
  outranks both ports.~~
  *(Wrong premise — it didn't run on CPU. Resolved the same day; the verified result
  is in the [follow-up correction](#correction-the-gpu-was-used) above.)*

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
