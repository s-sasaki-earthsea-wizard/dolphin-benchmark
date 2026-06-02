# dolphin-benchmark

A sibling repository for benchmarking and experimentation targeting GPU acceleration PRs against
[dolphin](https://github.com/isce-framework/dolphin).

This repo lives **next to** the upstream `dolphin/` clone (technically inside it, but excluded via
`.git/info/exclude`) so we can keep profiling artifacts, draft benchmark scripts, and working notes
out of the upstream tree.

## Layout

```
third-party-projects/dolphin/                 # upstream clone
├── (upstream tracked files)
├── .claude-notes/                            # session notes (excluded in upstream .git/info/exclude)
├── CLAUDE.md                                 # project context for Claude sessions (excluded)
└── dolphin-benchmark/                        # THIS repo (excluded)
    ├── .git/
    ├── benchmarks/                           # draft ASV-compatible benchmark scripts
    └── results/                              # profile output, benchmark numbers, charts
```

Task tracking lives on GitHub Issues:
<https://github.com/s-sasaki-earthsea-wizard/dolphin-benchmark/issues>

dolphin's `.git/info/exclude` lists `dolphin-benchmark/`, so it never appears in dolphin's
`git status` and cannot leak into an upstream PR.

## Strategy

### Principles

1. **Respect existing workflow** — do not change the implicit GPU/CPU detection or fallback
   behavior. Performance work only.
2. **Numerical equivalence** — every JAX rewrite is verified against the existing NumPy/Numba
   implementation with `np.testing.assert_allclose`.
3. **Quantified with ASV** — every PR ships a benchmark a reviewer can rerun.
4. **Small, focused PRs** — one concept per PR.

### Planned upstream PRs

| # | Scope | Files touched | Priority |
|---|---|---|---|
| 1 | Goldstein FFT filter → JAX | `src/dolphin/goldstein.py` | high |
| 2 | PS detection → JAX | `src/dolphin/ps.py` (`calc_ps_block`) | high |
| 3 | Log active JAX backend on startup | logging path TBD | medium |

Details and progress are tracked on
[GitHub Issues](https://github.com/s-sasaki-earthsea-wizard/dolphin-benchmark/issues).

## Existing benchmark infra in dolphin

dolphin already ships ASV (`asv.conf.json` + `benchmarks/benchmarks.py`). `CovarianceBenchmark`
is the established pattern; new benchmarks should follow it.

This sibling repo's `benchmarks/` is for **drafts and experiments** before they are merged into
dolphin's own `benchmarks/benchmarks.py` as part of a PR.

## Profiling tools

| Tool | Phase | Purpose |
|---|---|---|
| py-spy | exploration | confirm hotspots on a real end-to-end workflow |
| Nsight Systems (nsys) | post-GPU-rewrite | inspect CUDA kernel time, H↔D transfers, cuFFT cost |
| ASV | PR evidence | reproducible before/after numbers |
