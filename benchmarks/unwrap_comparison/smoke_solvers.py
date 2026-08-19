#!/usr/bin/env python
"""Smoke-test every unwrapper reachable in the `unwrap-cmp` env.

Purpose is to fail fast, on synthetic data in seconds, on the two things that
would otherwise only surface after a 45-minute run on a real NISAR frame:

1. **tophu <-> isce3 API drift.** conda-forge `tophu` 0.2.1 is a 2024 release
   pinned only to `isce3>=0.12`; the env resolves isce3 0.25.x. Nothing
   guarantees the ICU/PHASS bindings still match.
2. **whirlwind <-> dolphin API drift.** dolphin's `unwrap/_whirlwind.py` is
   written against `whirlwind-insar`, not the similarly named
   `isce-framework/whirlwind`. This script checks the signature it actually
   got rather than asserting the match from memory, then exercises both the
   direct call and dolphin's own `unwrap_whirlwind` wrapper.

Run inside the comparison container. `run.sh` with arguments bypasses the
compose `command:`, which is what installs dolphin, so do it here::

    SERVICE=unwrap ./docker/run.sh \
        'pip install --no-deps --quiet -e /dolphin &&
         python /dolphin-benchmark/benchmarks/unwrap_comparison/smoke_solvers.py'

Exit status is 0 if every *reachable* solver produced a congruent solution;
non-zero if a solver that should work did not.
"""

from __future__ import annotations

import inspect
import tempfile
import time
from pathlib import Path

import numpy as np

# Shape is small enough to stay a smoke test but large enough that ICU's
# tiling and PHASS's region growing both have something to chew on.
SHAPE = (512, 512)
NLOOKS = 10.0
SEED = 20260816

# What dolphin's `unwrap/_whirlwind.py` passes to `whirlwind.unwrap` beyond
# the positional (igram, corr, nlooks), with the defaults from
# `unwrap_whirlwind`. Passing the real values (not just probing the names)
# is what proves the call dolphin makes is the call whirlwind accepts.
DOLPHIN_WHIRLWIND_KWARGS = {
    "mask": None,
    "interpolate": False,
    "interp_cutoff": 0.5,
    "interp_num_neighbors": 20,
    "interp_max_radius": 51,
    "interp_min_radius": 0,
    "interp_alpha": 0.75,
    "cost_threshold": 50,
    "conncomp_sigma": None,
    "conncomp_cycle_prob": None,
    "min_size_px": 100,
    "max_ncomps": 1024,
}


def make_synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a noisy phase ramp with a known unwrapped truth.

    Returns
    -------
    igram, coherence, truth
        Wrapped interferogram (complex64), coherence (float32) and the
        unwrapped phase the solvers should recover up to a global constant.

    """
    rng = np.random.default_rng(SEED)
    ny, nx = SHAPE
    y, x = np.mgrid[0:ny, 0:nx]

    # ~8 fringes across each axis: enough wrapping to be a real test, few
    # enough that any correct solver recovers it.
    truth = (0.10 * x + 0.06 * y).astype(np.float32)

    coherence = np.full(SHAPE, 0.9, dtype=np.float32)
    noise_std = np.sqrt((1 - coherence**2) / (2 * NLOOKS * coherence**2))
    noisy = truth + rng.normal(0.0, noise_std).astype(np.float32)

    igram = np.exp(1j * noisy).astype(np.complex64)
    return igram, coherence, truth


def congruence(unw: np.ndarray, igram: np.ndarray) -> float:
    """Max deviation of ``(unw - wrapped) / 2pi`` from an integer.

    A solver returning a congruent solution scores ~0. This is the
    solver-internal check of the #13 metric contract, in its cheapest form.
    """
    k = (unw - np.angle(igram)) / (2 * np.pi)
    return float(np.max(np.abs(k - np.round(k))))


def report(name: str, unw, ccl, seconds: float, igram: np.ndarray, truth) -> bool:
    """Print one solver's smoke result; return whether it looks sane."""
    dev = congruence(unw, igram)
    # Solvers are free to differ by a global 2pi*k offset, so compare the
    # de-meaned residual against the known truth.
    resid = (unw - truth) - np.mean(unw - truth)
    rms = float(np.sqrt(np.mean(resid**2)))
    ncomp = "n/a" if ccl is None else str(len(np.unique(ccl)))
    ok = dev < 1e-3 and rms < 1.0
    print(
        f"  {name:22s} {seconds:7.2f}s  congruence_dev={dev:.2e}"
        f"  rms_vs_truth={rms:.4f} rad  ncomponents={ncomp}"
        f"  {'OK' if ok else 'SUSPECT'}"
    )
    return ok


def main() -> int:
    info = Path("/opt/conda/unwrap-solver-info.txt")
    if info.exists():
        print(info.read_text())

    igram, coherence, truth = make_synthetic()
    print(f"synthetic ifg {SHAPE}, nlooks={NLOOKS}, coherence=0.9\n")

    results: dict[str, bool] = {}

    # --- tophu-backed solvers (isce3 ICU / PHASS, plus isce3's SNAPHU) ------
    try:
        import isce3
        import tophu
    except ImportError as e:
        print(f"tophu/isce3 unavailable: {e}")
    else:
        print(f"isce3 {isce3.__version__} | tophu {tophu.__version__}")
        callbacks = {
            "tophu.ICUUnwrap": tophu.ICUUnwrap,
            "tophu.PhassUnwrap": tophu.PhassUnwrap,
            "tophu.SnaphuUnwrap": tophu.SnaphuUnwrap,
        }
        for name, factory in callbacks.items():
            # conda-forge isce3 is built with ICU and Phass only — no `snaphu`
            # binding, because SNAPHU's license prohibits commercial use and
            # conda-forge cannot redistribute it. tophu.SnaphuUnwrap therefore
            # has nothing to call here. That is a property of the environment,
            # not a failure, so it is skipped rather than counted.
            if name == "tophu.SnaphuUnwrap" and not hasattr(isce3.unwrap, "snaphu"):
                print(
                    f"  {name:22s} SKIP: isce3 {isce3.__version__} exposes"
                    f" {sorted(a for a in dir(isce3.unwrap) if not a.startswith('_'))}"
                    " — no snaphu binding (license). Use snaphu-py directly."
                )
                continue
            try:
                cb = factory()
                # tophu 0.2.1's UnwrapCallback takes a required `scratchdir`
                # that the protocol documented on GitHub main does not. Probe
                # rather than assume, so this keeps working across tophu
                # releases.
                extra = {}
                if "scratchdir" in inspect.signature(cb.__call__).parameters:
                    extra["scratchdir"] = Path(tempfile.mkdtemp(prefix=f"{name}-"))
                t0 = time.perf_counter()
                unw, ccl = cb(igram, coherence, NLOOKS, **extra)
                dt = time.perf_counter() - t0
            except Exception as e:  # noqa: BLE001 - smoke test reports, not raises
                print(f"  {name:22s} FAILED: {type(e).__name__}: {e}")
                results[name] = False
            else:
                results[name] = report(name, np.asarray(unw), np.asarray(ccl), dt, igram, truth)

    # --- whirlwind, called the way dolphin calls it -------------------------
    print()
    try:
        import whirlwind as ww
    except ImportError as e:
        print(f"whirlwind unavailable: {e}")
    else:
        print(f"whirlwind {ww.__version__}")
        sig = inspect.signature(ww.unwrap)
        missing = [k for k in DOLPHIN_WHIRLWIND_KWARGS if k not in sig.parameters]
        if missing:
            print(f"  signature: unwrap{sig}")
            print(
                f"  dolphin passes {len(missing)} kwarg(s) this build does not"
                f" accept: {', '.join(missing)}"
            )
            print("  -> wrong whirlwind package; expected `whirlwind-insar`.")
            results["whirlwind.unwrap"] = False
        else:
            print(f"  accepts all {len(DOLPHIN_WHIRLWIND_KWARGS)} kwargs dolphin passes")
            try:
                t0 = time.perf_counter()
                unw, ccl = ww.unwrap(
                    igram, coherence, NLOOKS, **DOLPHIN_WHIRLWIND_KWARGS
                )
                dt = time.perf_counter() - t0
            except Exception as e:  # noqa: BLE001
                print(f"  {'whirlwind.unwrap':22s} FAILED: {type(e).__name__}: {e}")
                results["whirlwind.unwrap"] = False
            else:
                results["whirlwind.unwrap"] = report(
                    "whirlwind.unwrap", np.asarray(unw), np.asarray(ccl), dt,
                    igram, truth,
                )

    # --- dolphin's own wrapper, end to end through files --------------------
    # The direct call above proves the API matches; this proves dolphin's
    # `unwrap_method: whirlwind` dispatch actually runs, raster I/O included.
    print()
    try:
        import rasterio
        from dolphin.unwrap import unwrap_whirlwind
    except ImportError as e:
        print(f"dolphin whirlwind path unavailable: {e}")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="dolphin-ww-"))
        prof = {
            "driver": "GTiff",
            "height": SHAPE[0],
            "width": SHAPE[1],
            "count": 1,
        }
        try:
            with rasterio.open(tmp / "ifg.tif", "w", dtype="complex64", **prof) as f:
                f.write(igram, 1)
            with rasterio.open(tmp / "corr.tif", "w", dtype="float32", **prof) as f:
                f.write(coherence, 1)

            t0 = time.perf_counter()
            unw_path, ccl_path = unwrap_whirlwind(
                ifg_filename=tmp / "ifg.tif",
                corr_filename=tmp / "corr.tif",
                unw_filename=tmp / "out.unw.tif",
                nlooks=NLOOKS,
            )
            dt = time.perf_counter() - t0

            with rasterio.open(unw_path) as f:
                unw = f.read(1)
            with rasterio.open(ccl_path) as f:
                ccl = f.read(1)
        except Exception as e:  # noqa: BLE001
            print(f"  {'dolphin.unwrap_whirlwind':22s} FAILED: {type(e).__name__}: {e}")
            results["dolphin.unwrap_whirlwind"] = False
        else:
            results["dolphin.unwrap_whirlwind"] = report(
                "dolphin.unwrap_whirlwind", unw, ccl, dt, igram, truth
            )

    print()
    bad = [k for k, v in results.items() if not v]
    if bad:
        print(f"SUSPECT/FAILED: {', '.join(bad)}")
        return 1
    print(f"all {len(results)} solver(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
