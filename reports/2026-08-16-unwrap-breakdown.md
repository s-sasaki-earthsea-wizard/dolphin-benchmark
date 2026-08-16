# Unwrap-step wall-clock breakdown — SNAPHU is 98–100 % of the step

**Date**: 2026-08-16

**Question**: within dolphin's unwrap step, how is wall time actually split
between the Python-side preprocessing stages (`goldstein.py`,
`interpolation.py` — both long-standing JAX-port candidates), the SNAPHU
core, and post-processing? This decides whether accelerating the Python
stages is worth anything.

## TL;DR

- **The unwrap step is SNAPHU**: 98.4–99.9 % of step wall time in every
  measured configuration, on both a Sentinel-1 burst and a NISAR frame.
- **Goldstein + interpolation together are < 0.5 % of the step.** There is
  no performance case for JAX-porting either. PR#1 (Goldstein → JAX) joins
  PR#2 (PS → JAX) as **pivot**.
- The measured wall-time levers are elsewhere, and both are configuration:
  **SNAPHU tiling** (7.4× on the same interferogram) and **enabling
  preprocessing on noisy L-band data** (SNAPHU itself runs 29–34 % faster
  on filtered+interpolated input).

## Why a new measurement

The 2026-06-03 py-spy profile could not answer this question, for three
reasons documented in
[its Correction 2](2026-06-03-pyspy-hotspots.md#correction-2-goldstein-totals-were-overcounted):

1. `snaphu-py` runs the bundled SNAPHU C executable in a **subprocess** —
   invisible to py-spy (the waiting thread is idle and dropped).
2. The profile's denominator is **Python-active sample time** (~98 s of a
   ~13 min run), not wall time.
3. The report's `goldstein` "7.6 %" was additionally an **aggregation
   artifact** (nested frames summed; correct value 2.8 %).

So this measurement wraps wall-clock timers around every stage function the
driver (`dolphin/unwrap/_unwrap.py::unwrap`) calls, and runs the real
`dolphin.unwrap.run()` on real interferograms. Caller shape replicates the
displacement workflow: `nlooks` from the run config, numba/numpy threads
capped exactly as `displacement.py:223` does (tutorial config: 1 thread),
sequential jobs, same correlation/similarity rasters, same scratch layout.
Inputs are copied to container-local NVMe first so NAS latency does not
pollute the attribution. Harness: `benchmarks/unwrap_breakdown/`.

## Data

| dataset | raster | notes |
|---|---|---|
| Sentinel-1 burst | 3441 × 1630 px (5.6 M px), 72 stitched ifgs, 3 measured | OPERA CSLC tutorial stack (West Texas), full-extent re-run @ 49975fce, strides x=6/y=3, nlooks=435 |
| NISAR frame | 4401 × 4338 px (19.1 M px, 27 % valid swath), 3 pairs extracted, 2 measured | **Real NISAR PROVISIONAL wrapped interferograms** from L2 GUNW products (Boso, descending 118D): the GUNW carries a wrapped igram + coherence at 20 m; 4×4 complex multilook reproduces the mission's 80 m unwrap grid. nlooks=16 (lower bound; no looks metadata in the product). Provisional products are pre-cal/val — fine for timing, not for science. |

SNAPHU config is dolphin's default (`init: mcf`, `cost: smooth`, single
tile) unless noted. Preprocessing modes are real user configs —
`run_goldstein` / `run_interpolation` are opt-in flags.

## Results

### Sentinel-1 burst (3 ifgs × 4 modes)

| stage | per-ifg | share of step |
|---|---|---|
| SNAPHU core | 313–428 s (±15 % input-dependent) | **98.4–99.8 %** |
| Goldstein filter | 1.3–1.6 s | 0.3–0.4 % |
| interpolation (numba, 1 thread) | 0.03–0.05 s (+ ~1 s one-time JIT) | 0.1 % |
| conncomp regrow (2nd SNAPHU, regrow mode) | ~1.6 s | 0.5 % |
| everything else (I/O, nodata, ambiguity transfer) | < 2 s | < 0.5 % |

The interpolation kernel is far cheaper in caller shape than its loop
structure suggests: the coherent-pixel mask plus early exit leave ~40 ms of
work per 5.6 M px interferogram, single-threaded.

### NISAR GUNW frame (2 ifgs per mode)

| run | SNAPHU per-ifg | preproc per-ifg | SNAPHU share |
|---|---|---|---|
| no preprocessing, single tile | 2635 / 3223 s (44–54 min) | — | 99.9 % |
| goldstein + interpolation, single tile | 1879 / 2125 s | 5 s + 10.5 s | 98.5 % |
| no preprocessing, 3×3 tiles / 4 parallel | **354 s** | — | 99.5 % |

Two findings beyond the headline:

1. **Preprocessing pays for itself ~50× over on noisy L-band data.** With
   Goldstein + interpolation enabled, SNAPHU itself ran 29–34 % faster on
   *both* interferograms (smoother phase → easier optimization): ~15 s of
   preprocessing bought ~15 min of SNAPHU. A configuration insight, not an
   argument for accelerating the stages.
2. **Tiling is the real lever: 7.4× on the same interferogram**
   (2635 s → 354 s with `ntiles=[3,3]`, `n_parallel_tiles=4`) —
   configuration dolphin already exposes, on top of the across-ifg
   `n_parallel_jobs` parallelism.

## Implications

- **PR#1 (Goldstein → JAX): pivot.** Even zeroing both Python stages moves
  the unwrap step < 0.5 %. The 2026-06-03 report's recommendation is
  superseded (its Correction 2 says so in place).
- **Upstream #545 (numba removal), `interpolation` item**: the remaining
  motivation is dependency hygiene only. The numba kernel's early-exit +
  mask structure is exactly what JAX's all-candidates formulation cannot
  express cheaply on CPU — the similarity trade-off, but more extreme, for
  a stage worth 0.1–0.5 % of its step. Worth stating plainly if/when a JAX
  port of `interpolation.py` is discussed.
- Wall-time acceleration of unwrapping, if pursued, means SNAPHU-side
  configuration (tiling defaults/documentation), alternative unwrappers, or
  preprocessing guidance for noisy data — not Python-stage ports.

## Artifacts

- `benchmarks/unwrap_breakdown/` — harness (`profile_unwrap.py`), NISAR GUNW
  extractor (`extract_gunw_wrapped.py`), method README
- `results/unwrap-breakdown/breakdown-*.json` — summaries + per-call records
  (7 runs; the first `interpolate` call carries the numba JIT cost, visible
  in the records)
- Extracted NISAR rasters and the full-extent S1 run are regenerable
  scratch under `results/` (git-ignored): `results/nisar-unwrap-bench/`,
  `results/run-full/`
