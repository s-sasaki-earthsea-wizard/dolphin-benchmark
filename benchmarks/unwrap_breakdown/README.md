# Unwrap-step wall-clock breakdown

**Question**: within dolphin's unwrap step, how is wall time split between the
preprocessing stages (`goldstein.py`, `interpolation.py` — both candidates for
JAX ports), the SNAPHU core, and post-processing? This decides where (and
whether) acceleration work on the unwrap step is worth doing.

## Why not py-spy

Two independent reasons the existing full-run profile
(`results/profile-fullrun.folded`, 2026-06-03) cannot answer this question:

1. **SNAPHU is a subprocess.** `snaphu-py` invokes the bundled `snaphu` C
   executable via `subprocess` (see `snaphu._snaphu.run_snaphu`). The Python
   thread that waits on it is idle and py-spy drops idle threads by default,
   so SNAPHU core time simply does not exist in the profile.
2. **The sampled denominator is not wall time.** The profile holds 9845
   samples at 100 Hz ≈ 98 s of Python-active time, against ~13 min of wall —
   NAS-blocked reads and subprocess waits are invisible. Any "% of run" from
   that profile is a share of *Python-active time only*.

Additionally, the candidate-function totals in `results/profile-hotspots.txt`
("`goldstein`: total=7.56%") sum the per-frame totals of *nested* frames
(`goldstein` → `apply_goldstein_filter` → `patch_goldstein_filter`), counting
the same sample up to 3×. The correct subsystem total is the entry frame's:
**2.77%** (`goldstein (dolphin/goldstein.py:129)`). A correction to
`reports/2026-06-03-pyspy-hotspots.md` is tracked separately.

## Method

`profile_unwrap.py` wraps wall-clock timers around every stage function called
by `dolphin.unwrap._unwrap.unwrap()` (monkeypatched in the namespaces the
driver resolves at call time), then runs the real driver `dolphin.unwrap.run()`
on real stitched interferograms produced by the displacement workflow.

Caller shape is replicated from `dolphin/workflows/unwrapping.py::run` /
`displacement.py`:

- `nlooks` = `(2·hw_y+1)·(2·hw_x+1)` from the run config's phase-linking half
  window (tutorial config: 15 × 29 = 435)
- numba/numpy threads capped via `dolphin.utils.set_num_threads(
  worker_settings.threads_per_worker)` exactly as `displacement.py:223` does
  (tutorial config: **1 thread**)
- same correlation + full similarity rasters, same scratchdir layout,
  `n_parallel_jobs=1` (tutorial default), SNAPHU single tile, `init: mcf`,
  `cost: smooth`
- inputs copied to container-local disk first (default) so NAS latency does
  not pollute stage attribution; `--no-copy-local` measures in place

Preprocessing modes: `none` (dolphin default), `goldstein`, `interp`, `both` —
`run_goldstein` / `run_interpolation` are opt-in config flags, so each mode is
a real user configuration.

Data: full-extent re-run of the OPERA CSLC tutorial stack (West Texas,
T078-165573-IW2, 26 scenes, nearest-3 network), stitched single-burst
interferograms at strides x=6/y=3.

## Usage

```bash
# from dolphin-benchmark/, after a full-extent run landed in results/run-full
./docker/run.sh 'pip install --no-deps --quiet -e /dolphin && \
  for m in none goldstein interp both; do \
    python /dolphin-benchmark/benchmarks/unwrap_breakdown/profile_unwrap.py \
      --work /work/run-full --preproc $m -n 3 --out /work/unwrap-breakdown; \
  done'
```

Outputs land in `results/unwrap-breakdown/breakdown-<mode>.json` (summary +
per-call records; the first `interpolate` call includes numba JIT compilation,
visible in the per-ifg records).

## Results

Raw JSONs in `results/unwrap-breakdown/` (summary + per-call records).

### Tier 1 — Sentinel-1 burst (OPERA CSLC tutorial stack)

Stitched single-burst interferograms, 3441 × 1630 px (~5.6 M px), strides
x=6/y=3, nlooks=435, SNAPHU single tile, `init: mcf`, `cost: smooth`,
1 thread (tutorial-config caller shape), inputs on container-local NVMe.
3 interferograms per mode; RTX 5080 host, container CUDA 12.6.3.

| stage | per-ifg | share of unwrap step |
|---|---|---|
| SNAPHU core | 313–428 s | **98.4–99.8 %** |
| Goldstein filter | 1.3–1.6 s | 0.3–0.4 % |
| interpolation (numba, 1 thread) | 0.03–0.05 s (+ ~1 s one-time JIT) | 0.1 % |
| conncomp regrow (2nd SNAPHU, regrow mode) | ~1.6 s | 0.5 % |
| everything else (I/O, nodata fixup, ambiguity transfer) | < 2 s | < 0.5 % |

Takeaways:

1. **The unwrap step is SNAPHU.** At realistic burst scale the two
   Python-side preprocessing stages that were candidates for JAX ports are
   noise: accelerating both to zero would shave **< 0.5 %** off the step.
2. The interpolation kernel is far cheaper in caller shape than its numba
   loop structure suggests — the coherent-pixel mask + early exit leave
   little work (~40 ms per 5.6 M px ifg, single-threaded).
3. SNAPHU run-to-run spread across interferograms is ±15 % (optimization is
   input-dependent), so mode-to-mode differences in SNAPHU time are noise.
4. Wall-time leverage on the unwrap step in dolphin therefore comes from
   SNAPHU-side parallelism (`ntiles` / `n_parallel_tiles` / `n_parallel_jobs`)
   or a different unwrapper, not from accelerating the Python stages.

### Tier 2 — NISAR GUNW frame (Boso, descending 118D)

Real NISAR PROVISIONAL wrapped interferograms extracted from L2 GUNW
products (`extract_gunw_wrapped.py`), multilooked 4×4 to the official 80 m
unwrap grid: 4401 × 4338 px (19.1 M px, ~27 % valid swath, mean coherence
0.30), nlooks=16 (lower bound — the 20 m layer carries no numberOfLooks
metadata), 2 interferograms per mode, 1 thread, inputs on NVMe.

| run | SNAPHU per-ifg | preproc per-ifg | SNAPHU share |
|---|---|---|---|
| `none`, single tile | 2635 / 3223 s (44–54 min) | — | 99.9 % |
| `both`, single tile | 1879 / 2125 s | goldstein 5 s + interp 10.5 s | 98.5 % |
| `none`, 3×3 tiles, 4 parallel | 354 s | — | 99.5 % |

Takeaways:

1. **SNAPHU dominance holds at frame scale** (98.5–99.9 %), and the absolute
   cost balloons: ~49 min per frame single-tile.
2. **Preprocessing pays for itself ~50× over on noisy L-band data**: with
   Goldstein + interpolation enabled, SNAPHU ran 29–34 % faster on both
   interferograms (smoother phase → easier optimization), cutting ~15 min
   from ~49 min — while the preprocessing itself costs 15 s. (This is a
   *configuration* insight, not an argument for accelerating the stages.)
3. Even with far more masked pixels to fill (73 % invalid + low coherence),
   interpolation stays at 10.5 s single-threaded — 0.5 % of the step.
4. **SNAPHU tiling is the real wall-time lever: 7.4× on the same
   interferogram** (2635 s → 354 s with `ntiles=[3,3]`,
   `n_parallel_tiles=4`) — configuration dolphin already exposes.

### Conclusion

Across both datasets the unwrap step is 98–100 % SNAPHU (an external C
subprocess). JAX-porting the Python-side stages (`goldstein.py`,
`interpolation.py`) would shave **< 0.5 %** off the step in every measured
configuration; the practical levers are SNAPHU tiling/parallelism (already
exposed by dolphin's config) and preprocessing-as-configuration on noisy
data. Any acceleration effort aimed at wall time should target the SNAPHU
side (tiling defaults, docs) or alternative unwrappers, not the Python
stages.
