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
    ├── docker/                               # GPU-enabled dev container (CUDA + JAX-GPU)
    ├── benchmarks/                           # scratch/experiment scripts (PRs go upstream)
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
is the established pattern; new benchmarks for Goldstein / PS detection are added directly to
that file (and become part of the upstream PR). This is intentional: it keeps the maintainer's
review experience aligned — one set of benchmarks, one ASV invocation.

This sibling repo's `benchmarks/` directory is reserved for one-off experiments and scratch
work that *won't* go upstream (e.g. exploratory profiling scripts, comparison harnesses).

## Profiling tools

| Tool | Phase | Purpose |
|---|---|---|
| py-spy | exploration | confirm hotspots on a real end-to-end workflow |
| Nsight Systems (nsys) | post-GPU-rewrite | inspect CUDA kernel time, H↔D transfers, cuFFT cost |
| ASV | PR evidence | reproducible before/after numbers |

## Dev environment (Docker)

A GPU-enabled container is defined under [`docker/`](docker/). It:

- builds on `nvidia/cuda:12.6.3-devel-ubuntu22.04` (Nsight Systems CLI included)
- creates a conda env matching dolphin's `conda-env.yml`
- installs `jax[cuda12]`, `asv`, `py-spy`, `pynvml` on top
- mounts the host's `dolphin/` clone as `/dolphin` and `pip install -e .` on container start,
  so source edits to dolphin are picked up live without rebuilds

### Host prerequisites

- NVIDIA driver supporting CUDA 12.x
- Docker with `nvidia-container-toolkit` and the `nvidia` runtime registered
  (`docker info | grep -i runtime` should mention `nvidia`)

### Setup `.env`

Required before first run:

```bash
cp .env.example .env
$EDITOR .env   # fill in ASF credentials and absolute paths
```

`.env` lives at the repo root (next to this README), is gitignored, and is read by both
the compose-time path substitution and the in-container environment. Required variables:

| Variable | Purpose |
|---|---|
| `ASF_USERNAME` / `ASF_PASSWORD` | NASA Earthdata login for `opera_utils.download` |
| `DATA_DIR` | host path to raw Sentinel-1 SAFE tree (mounted `/data` ro) |
| `OPERA_CSLC_DIR` | host path where downloaded OPERA CSLCs go (mounted `/cslc` rw) |

### First-time build and interactive shell

```bash
cd dolphin-benchmark/docker
./run.sh
```

This builds the image (5–10 min first time, cached afterwards) and drops into an interactive
shell with `/dolphin` editable-installed and JAX configured for GPU. The banner reports the
GPU and JAX backend so you can verify GPU is actually in use.

### Override mount paths ad-hoc

```bash
DOLPHIN_SRC=/other/clone ./run.sh
```

This works for variables that *aren't* user-specific. User-specific paths
(`DATA_DIR`, `OPERA_CSLC_DIR`) must come from `.env` — there are no
hardcoded defaults in the compose file.

### One-shot invocations

```bash
./run.sh asv run --quick
./run.sh py-spy record -o /work/profile.svg -- dolphin run /work/config.yaml
./run.sh nsys profile -o /work/run python -m my_bench
```

### Notes

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set in the image to avoid pre-allocating 75% of GPU
  memory. Unset (or set to `true`) for max throughput when not sharing the GPU.
- The `docker/conda-env.yml` is a snapshot of `../dolphin/conda-env.yml` — sync date is noted
  in the file header. Diff and update when upstream changes its deps.
- `/data` is mounted read-only on purpose. Write results to `/work` (host: `results/`).
