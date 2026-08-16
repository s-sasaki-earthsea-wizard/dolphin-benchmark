# asv_runner 0.3.0 → 0.3.1 A/B

Raw logs behind the confirmation posted on
[airspeed-velocity/asv_runner#52](https://github.com/airspeed-velocity/asv_runner/issues/52):
the setup-ordering regression that ran a class's bound `setup(self)` before
`setup_cache()` is gone in 0.3.1.

Both runs are the same image, the same command, and the same benchmark. The
only difference is which `asv_runner` the benchmark process imports.

## The benchmark

`SingleMinistackBenchmark` from [isce-framework/dolphin](https://github.com/isce-framework/dolphin)'s
`benchmarks/benchmarks.py` — a class whose parameter-free `setup` consumes
files that `setup_cache` creates:

```python
def setup_cache(self):
    _make_slc_stack(Path("slcs"))       # writes 10 rasters

def setup(self):
    ...
    self.slc_file_list = sorted(Path("slcs").glob("20*.slc.tif"))
    assert len(self.slc_file_list) > 0, f"No SLC files found: ..."
```

That assert is the tripwire: it can only fail if `setup` runs first.

## Environment

Python 3.14.6, asv 0.6.6, jax 0.11.0 (GPU backend), Linux 6.17.0-29-generic.
dolphin at `49975fce`, whose `benchmarks/` and `src/dolphin/workflows/` are
identical to upstream `main` at `c2f7c24`.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda
asv machine --yes --config <conf>

# B: 0.3.1 — the image default
asv run --quick --show-stderr --python=same \
    --set-commit-hash $(git -C /dolphin rev-parse HEAD) \
    -b SingleMinistackBenchmark --config <conf>

# A: 0.3.0 — forced onto the benchmark process, image otherwise untouched
pip install --target /tmp/runner030 asv_runner==0.3.0
PYTHONPATH=/tmp/runner030 asv run --quick --show-stderr --python=same \
    --set-commit-hash $(git -C /dolphin rev-parse HEAD) \
    -b SingleMinistackBenchmark --config <conf>
```

`<conf>` is this repo's generated `asv/asv-existing-gpu.conf.json`
(`make asv-confs`). `PYTHONPATH` is enough to redirect the import because asv
spawns the benchmark in a subprocess that inherits the environment. The 0.3.0
log records the module that was actually loaded
(`runner in use: /tmp/runner030/asv_runner/__init__.py`, line 38); the 0.3.1
side is the image default, which `docker/Dockerfile` pins to `>=0.3.1`.

## What to look for

`asv_runner-0.3.0.log`:

```
[50.00%] ··· Setting up benchmarks:177                                   failed
              AssertionError: No SLC files found: []
              asv: setup_cache failed (exit status 1)
[50.00%] ··· ...SingleMinistackBenchmark.time_single_ministack  skipped (setup_cache failed)
```

`asv_runner-0.3.1.log`:

```
[50.00%] ··· Setting up benchmarks:177                                       ok
                 Files: 11   Evicted Pages: 20501 (80M)
```

The `Files: 11` block is `vmtouch -e .` from the benchmark's own `setup`,
reporting the rasters `setup_cache` had by then written — the ordering is
correct again.

On 0.3.1 the benchmark still ends in `failed`, but with a different error:
a `TypeError` from a dolphin API that changed signature in 2024 and left the
benchmark behind. Unrelated to this issue, and fixed separately.

## Note on the logs

Unedited console capture, so each starts with the CUDA image banner and the
`asv machine` prompts before reaching `· Discovering benchmarks`.
