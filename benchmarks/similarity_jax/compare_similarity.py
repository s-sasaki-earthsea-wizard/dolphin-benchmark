#!/usr/bin/env python
"""Numerical equivalence: JAX `dolphin.similarity` vs the upstream numba version.

The implementation under test is whatever `dolphin` is installed (expected:
the `feature/545-jax-similarity` working tree, editable install). The
reference is a vendored, byte-identical copy of `src/dolphin/similarity.py`
from upstream commit c2f7c24 (numba implementation), stored next to this
script as `similarity_numba_c2f7c24.py`
(git blob d146b8e7e7af3f87d91bd0141d939262d414bd45).

Run inside the dev container, once per JAX backend:

    JAX_PLATFORMS=cpu  python compare_similarity.py --out compare_cpu.json
    JAX_PLATFORMS=cuda python compare_similarity.py --out compare_gpu.json \
        --real-data-dir /cslc

Exit code is non-zero if any case fails the tolerance check.

Tolerance rationale: the numba reference accumulates per-pixel similarity in
float64 and casts the summary to float32 on output; the JAX version computes
in float32 throughout (complex64 inputs, default no-x64 JAX config). Values
are bounded in [-1, 1], so we compare with atol=5e-6, rtol=1e-5 and also
report the observed max abs difference per case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import similarity_numba_c2f7c24 as ref_sim  # noqa: E402

from dolphin import similarity as jax_sim  # noqa: E402

RTOL = 1e-5
ATOL = 5e-6
SEED = 1234


def random_unit_stack(rng, n, rows, cols):
    phases = rng.uniform(-np.pi, np.pi, size=(n, rows, cols)).astype("float32")
    return np.exp(1j * phases).astype("complex64")


def build_cases(rng):
    """Yield (name, ifg_stack, mask, radius) tuples covering the semantics."""
    cases = []

    # 1. Basic random stacks, several radii, odd sizes to stress edge clipping
    base = random_unit_stack(rng, 10, 64, 65)
    for radius in [2, 5, 7, 11]:
        cases.append((f"basic_r{radius}", base, None, radius))

    # 2. Random mask (~50% False)
    mask = rng.random((64, 65)) > 0.5
    for radius in [5, 7]:
        cases.append((f"masked_r{radius}", base, mask, radius))

    # 3. Fully-NaN pixels (invalid) + scattered single-ifg NaNs
    nan_stack = base.copy()
    invalid = rng.random((64, 65)) < 0.05
    nan_stack[:, invalid] = np.nan
    scattered = rng.random((10, 64, 65)) < 0.02
    nan_stack[scattered] = np.nan
    cases.append(("nan_pixels_r7", nan_stack, None, 7))

    # 4. All-zero pixels (invalid path via nan_to_num sum == 0)
    zero_stack = base.copy()
    zero_stack[:, rng.random((64, 65)) < 0.05] = 0
    cases.append(("zero_pixels_r7", zero_stack, None, 7))

    # 5. Floating-point phase input (not complex)
    phase32 = rng.uniform(-np.pi, np.pi, size=(8, 40, 41)).astype("float32")
    cases.append(("float32_phase_r5", phase32, None, 5))
    phase64 = rng.uniform(-np.pi, np.pi, size=(8, 40, 41))
    cases.append(("float64_phase_r5", phase64, None, 5))

    # 6. Search radius larger than the image (heavy boundary clipping)
    tiny = random_unit_stack(rng, 5, 8, 9)
    cases.append(("tiny_image_r7", tiny, None, 7))

    # 7. Single-row image (extreme clipping)
    row = random_unit_stack(rng, 3, 1, 12)
    cases.append(("single_row_r3", row, None, 3))

    # 8. Everything invalid -> all-NaN output
    cases.append(("all_zero_r2", np.zeros((10, 4, 5), dtype="complex64"), None, 2))

    # 9. One unmasked pixel: no valid neighbors -> NaN at that pixel too
    lone_mask = np.zeros((16, 16), dtype=bool)
    lone_mask[8, 8] = True
    cases.append(("lone_pixel_r3", random_unit_stack(rng, 6, 16, 16), lone_mask, 3))

    # 10. Non-bool masks: the numba loop used truthiness (nonzero == valid).
    # int value 2 has no low bit set, catching bitwise-AND mask handling
    small = random_unit_stack(rng, 4, 20, 21)
    nonbool = (rng.random((20, 21)) > 0.3).astype("int16") * 2
    cases.append(("int2_mask_r3", small, nonbool, 3))
    cases.append(("float_mask_r3", small, nonbool.astype("float32") / 2, 3))

    return cases


def load_real_cases(real_data_dir, n_granules=6):
    """Build single-reference ifg stacks from the first OPERA CSLC granules."""
    import h5py

    files = sorted(Path(real_data_dir).glob("*.h5"))[:n_granules]
    if len(files) < 2:
        print(f"real-data: fewer than 2 .h5 files in {real_data_dir}, skipping")
        return []
    # Find the first mostly-valid row in a column strip of the first granule,
    # so the "edge" window straddles the valid/invalid data boundary
    col_window = slice(10000, 10512)
    with h5py.File(files[0]) as f:
        strip = f["data/VV"][:, col_window]
    valid_frac = np.isfinite(strip).mean(axis=1)
    first_valid_row = int(np.argmax(valid_frac > 0.1))
    edge_start = max(0, first_valid_row - 256)
    print(f"real-data: valid rows start at {first_valid_row}, "
          f"edge window rows {edge_start}:{edge_start + 512}")

    windows = {
        "real_interior": (slice(2000, 2512), col_window),
        "real_valid_edge": (slice(edge_start, edge_start + 512), col_window),
    }
    cases = []
    for name, (rw, cw) in windows.items():
        slcs = []
        for path in files:
            with h5py.File(path) as f:
                slcs.append(f["data/VV"][rw, cw])
        ref = slcs[0]
        ifg_stack = np.stack(
            [(ref * np.conj(s)).astype("complex64") for s in slcs[1:]]
        )
        cases.append((f"{name}_r7", ifg_stack, None, 7))
    return cases


def run_one(name, stack, mask, radius, summary):
    func_ref = getattr(ref_sim, f"{summary}_similarity")
    func_jax = getattr(jax_sim, f"{summary}_similarity")

    # The reference mutates the passed mask in place -- give each its own copy
    out_ref = func_ref(stack, search_radius=radius, mask=None if mask is None else mask.copy())
    out_jax = func_jax(stack, search_radius=radius, mask=None if mask is None else mask.copy())

    nan_ref = np.isnan(out_ref)
    nan_jax = np.isnan(out_jax)
    nan_match = bool(np.array_equal(nan_ref, nan_jax))
    both_valid = ~nan_ref & ~nan_jax
    if both_valid.any():
        max_abs_diff = float(np.max(np.abs(out_ref[both_valid] - out_jax[both_valid])))
        close = bool(
            np.allclose(out_ref[both_valid], out_jax[both_valid], rtol=RTOL, atol=ATOL)
        )
    else:
        max_abs_diff = 0.0
        close = True

    passed = nan_match and close
    return {
        "case": name,
        "summary": summary,
        "shape": list(stack.shape),
        "radius": radius,
        "dtype": str(stack.dtype),
        "n_valid": int(both_valid.sum()),
        "n_nan_ref": int(nan_ref.sum()),
        "n_nan_jax": int(nan_jax.sum()),
        "nan_pattern_match": nan_match,
        "max_abs_diff": max_abs_diff,
        "allclose": close,
        "pass": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--real-data-dir", type=Path, default=None)
    args = parser.parse_args()

    import jax

    rng = np.random.default_rng(SEED)
    cases = build_cases(rng)
    if args.real_data_dir is not None:
        cases += load_real_cases(args.real_data_dir)

    results = []
    for name, stack, mask, radius in cases:
        for summary in ["median", "max"]:
            res = run_one(name, stack, mask, radius, summary)
            results.append(res)
            status = "PASS" if res["pass"] else "FAIL"
            print(
                f"{status}  {name:20s} {summary:6s} shape={tuple(stack.shape)!s:14s}"
                f" r={radius:2d} max|diff|={res['max_abs_diff']:.2e}"
                f" nan_match={res['nan_pattern_match']}"
            )

    all_pass = all(r["pass"] for r in results)
    report = {
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "numba_version": ref_sim.numba.__version__,
        "numpy_version": np.__version__,
        "rtol": RTOL,
        "atol": ATOL,
        "seed": SEED,
        "n_cases": len(results),
        "all_pass": all_pass,
        "results": results,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    print(f"\n{'ALL PASS' if all_pass else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)}) "
          f"on backend={jax.default_backend()}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
