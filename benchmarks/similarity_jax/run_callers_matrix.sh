#!/usr/bin/env bash
# Driver for the caller-shaped benchmark matrix (run inside the dev container).
# Each cell is its own process so numba threading env and JIT state are isolated.
set -euo pipefail
cd "$(dirname "$0")"
RES=results_callers
mkdir -p "$RES"

echo "=== prep blocks ==="
python bench_callers.py prep --real-data-dir /cslc --out-dir /tmp/blocks

echo "=== CPU matrix ==="
for ds in synthetic real; do
  for cfg in single sequential default; do
    JAX_PLATFORMS=cpu python bench_callers.py run --impl numba --config "$cfg" \
      --dataset "$ds" --blocks-dir /tmp/blocks --out "$RES/numba-$cfg-$ds.json"
    JAX_PLATFORMS=cpu python bench_callers.py run --impl jax --config "$cfg" \
      --dataset "$ds" --blocks-dir /tmp/blocks --out "$RES/jaxcpu-$cfg-$ds.json"
  done
  # NUMBA_NUM_THREADS caps: total threads = pool threads x numba threads ~ 16
  JAX_PLATFORMS=cpu python bench_callers.py run --impl numba --config sequential \
    --dataset "$ds" --numba-threads 8 --blocks-dir /tmp/blocks \
    --out "$RES/numbacap8-sequential-$ds.json"
  JAX_PLATFORMS=cpu python bench_callers.py run --impl numba --config default \
    --dataset "$ds" --numba-threads 3 --blocks-dir /tmp/blocks \
    --out "$RES/numbacap3-default-$ds.json"
done

echo "=== GPU cells ==="
for ds in synthetic real; do
  for cfg in single sequential; do
    JAX_PLATFORMS=cuda python bench_callers.py run --impl jax --config "$cfg" \
      --dataset "$ds" --blocks-dir /tmp/blocks --out "$RES/jaxgpu-$cfg-$ds.json"
  done
done

echo "=== summary ==="
python bench_callers.py summarize --results-dir "$RES"
