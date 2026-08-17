#!/usr/bin/env python
"""Wall-clock breakdown of dolphin's unwrap step, stage by stage.

Why wall-clock wrapping instead of a sampling profiler: snaphu-py runs the
bundled ``snaphu`` C executable in a subprocess, so SNAPHU core time never
appears in py-spy profiles of the Python process (the waiting thread is idle
and excluded by default). Wrapping the stage functions called by
``dolphin.unwrap._unwrap.unwrap`` is the only reliable way to attribute the
unwrap step's wall time.

The script replicates the displacement-workflow caller shape
(``dolphin/workflows/unwrapping.py::run`` -> ``dolphin.unwrap.run``): same
nlooks (derived from the run config's phase-linking half window), same
correlation/similarity rasters, same scratchdir layout, sequential jobs
(``n_parallel_jobs=1``, the tutorial-config default).

Timed call sites in ``dolphin/unwrap/_unwrap.py::unwrap``:

- ``goldstein``             Goldstein filter (numpy FFT)
- ``interpolate``           numba nearest-scatterer interpolation
- ``unwrap_snaphu_py``      SNAPHU core incl. snaphu-py wrapper I/O
- ``multiscale_unwrap``     tophu wrapper around the isce3 ICU/PHASS solvers
                            (includes tophu's downsample/upsample + tiling)
- ``unwrap_whirlwind``      whirlwind native MCF solve
- ``transfer_ambiguities``  ambiguity transfer back to the original ifg
- ``grow_conncomp_snaphu``  connected-component regrow (a second SNAPHU run;
                            only when goldstein/interpolation ran)
- ``set_nodata_values``     nodata fixup on unw + conncomp rasters
- ``create_combined_mask``  nodata mask combine (when a mask applies)
- ``io.load_gdal`` / ``io.write_arr``  raster I/O issued by the driver itself

``--method`` selects the solver (issue #13). Two things to know before
comparing methods:

- ``grow_conncomp_snaphu`` is **not** solver-specific: dolphin regrows
  connected components with SNAPHU whenever goldstein/interpolation ran,
  whatever ``unwrap_method`` is. An ICU/PHASS run with ``--preproc both``
  therefore still contains a SNAPHU invocation. Use ``--preproc none`` for a
  SNAPHU-free measurement of the alternative solvers.
- the ``tophu`` stage is an *adapter total*, not a solver core: it covers
  tophu's coarse downsample, tiling and upsample around the isce3 call.

Raster I/O that happens *inside* a timed stage (e.g. snaphu-py's raster
copies) is included in that stage's time and additionally recorded with a
``within`` marker in the raw records, so nothing is double-counted in the
summary.

Run inside the dev container::

    python profile_unwrap.py --work /work/run-full --preproc both -n 3

By default inputs are copied to container-local disk first so NAS latency
does not pollute the attribution; pass --no-copy-local to measure in place.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import yaml

# In the unwrap-cmp env, `dolphin.io` -> `opera_utils` -> `isce3.core` pulls in
# pyre, whose CommandLineParser parses sys.argv at *import* time and fails on
# any flag it does not recognise (`--config`, `--method`, ...). The failure
# surfaces as a misleading `AttributeError: partially initialized module
# 'journal'`. dolphin-env has no isce3, which is why this never fired before.
# Stash argv across the whole dolphin import block; main() sees the real argv.
_REAL_ARGV = sys.argv
sys.argv = sys.argv[:1]
try:
    import dolphin.unwrap._unwrap as unwrap_mod
    from dolphin import io
    from dolphin.unwrap import run as unwrap_run
    from dolphin.workflows import UnwrapOptions
finally:
    sys.argv = _REAL_ARGV

# `list.append` is atomic under the GIL, so the record sink can be shared.
RECORDS: list[dict] = []

# The nesting stack and the current interferogram label must NOT be shared:
# dolphin runs `--n-parallel-jobs > 1` on a ThreadPoolExecutor, so several
# `unwrap()` calls are in flight at once. A module-level stack would have each
# thread popping frames pushed by another, mislabelling `within` (and with it
# every stage the summary reads) and attributing records to the wrong ifg.
_TLS = threading.local()


def _stack() -> list[str]:
    stack = getattr(_TLS, "stack", None)
    if stack is None:
        stack = _TLS.stack = []
    return stack

# Stages summed in the summary table. Everything else under "total" becomes
# "other" (scratch cleanup, intermediate moves, mask handling, ...).
SUMMARY_STAGES = [
    "goldstein",
    "interpolate",
    "snaphu",
    "tophu",
    "whirlwind",
    "transfer_ambiguities",
    "conncomp_regrow",
    "set_nodata",
    "combined_mask",
]

# The stage that carries the solver for each --method, i.e. the row that is
# actually being compared across methods.
SOLVER_STAGE = {
    "snaphu": "snaphu",
    "icu": "tophu",
    "phass": "tophu",
    "whirlwind": "whirlwind",
}


def _import_with_clean_argv(name: str):
    """Import ``name`` with ``sys.argv`` temporarily stripped to ``argv[0]``.

    isce3 (pulled in by tophu for the ICU/PHASS solvers) imports pyre, whose
    ``CommandLineParser`` parses ``sys.argv`` eagerly at import time. Any flag
    it does not recognise -- ``--config`` and ``--method`` among them -- makes
    it fail deep inside its own logging bootstrap, surfacing as a misleading
    ``AttributeError: partially initialized module 'journal'``.

    dolphin imports tophu lazily, at unwrap time, long after argparse has run,
    so the import has to happen here behind a stashed argv.
    """
    saved = sys.argv
    sys.argv = saved[:1]
    try:
        return importlib.import_module(name)
    finally:
        sys.argv = saved


def _timed(func, stage: str):
    def wrapper(*args, **kwargs):
        stack = _stack()
        stack.append(stage)
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            stack.pop()
            RECORDS.append(
                {
                    "ifg": getattr(_TLS, "ifg", None),
                    "stage": stage,
                    "within": stack[-1] if stack else None,
                    "seconds": dt,
                }
            )

    wrapper.__wrapped__ = func
    return wrapper


def install_probes() -> None:
    """Patch timing wrappers into the namespaces the driver resolves at call time."""
    for name, stage in [
        ("goldstein", "goldstein"),
        ("interpolate", "interpolate"),
        ("unwrap_snaphu_py", "snaphu"),
        ("multiscale_unwrap", "tophu"),
        ("unwrap_whirlwind", "whirlwind"),
        ("transfer_ambiguities", "transfer_ambiguities"),
        ("grow_conncomp_snaphu", "conncomp_regrow"),
        ("set_nodata_values", "set_nodata"),
        ("create_combined_mask", "combined_mask"),
    ]:
        setattr(unwrap_mod, name, _timed(getattr(unwrap_mod, name), stage))

    # The driver calls these as `io.<name>`, so patch the io module attribute.
    for name, stage in [("load_gdal", "io.load_gdal"), ("write_arr", "io.write_arr")]:
        setattr(io, name, _timed(getattr(io, name), stage))

    orig_unwrap = unwrap_mod.unwrap

    def unwrap_with_label(*args, **kwargs):
        _TLS.ifg = Path(kwargs["ifg_filename"]).name
        stack = _stack()
        stack.append("total")
        t0 = time.perf_counter()
        try:
            return orig_unwrap(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            stack.pop()
            RECORDS.append(
                {
                    "ifg": getattr(_TLS, "ifg", None),
                    "stage": "total",
                    "within": None,
                    "seconds": dt,
                }
            )
            _TLS.ifg = None

    unwrap_mod.unwrap = unwrap_with_label


def summarize() -> dict:
    """Aggregate top-level stage times; nested io records are excluded."""
    by_stage: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    for r in RECORDS:
        if r["stage"] == "total":
            by_stage["total"] += r["seconds"]
            calls["total"] += 1
        elif r["stage"] in SUMMARY_STAGES and r["within"] == "total":
            by_stage[r["stage"]] += r["seconds"]
            calls[r["stage"]] += 1
        elif r["stage"].startswith("io.") and r["within"] == "total":
            by_stage["driver_io"] += r["seconds"]
            calls["driver_io"] += 1
    accounted = sum(v for k, v in by_stage.items() if k != "total")
    by_stage["other"] = by_stage["total"] - accounted
    return {
        "seconds": dict(by_stage),
        "calls": dict(calls),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Wall-clock breakdown of dolphin's unwrap step"
    )
    p.add_argument("--work", default="/work/run-full", help="dolphin work dir")
    p.add_argument(
        "--config",
        default=None,
        help="dolphin run config YAML (default: <work>/dolphin_config.yaml)",
    )
    p.add_argument(
        "--ifg-files",
        nargs="+",
        default=None,
        help=(
            "explicit interferogram paths (bypasses --work/--config discovery;"
            " correlation defaults to <ifg>.int.tif -> <ifg>.int.cor.tif)"
        ),
    )
    p.add_argument(
        "--similarity",
        default=None,
        help=(
            "similarity raster for --ifg-files mode; omit if none exists"
            " (interpolation then masks on correlation only)"
        ),
    )
    p.add_argument(
        "--nlooks",
        type=float,
        default=None,
        help="effective looks; required with --ifg-files, else from the config",
    )
    p.add_argument(
        "--label",
        default=None,
        help="dataset label used in the output filename and metadata",
    )
    p.add_argument(
        "--method",
        choices=["snaphu", "icu", "phass", "whirlwind"],
        default="snaphu",
        help="unwrapper to dispatch to (dolphin's unwrap_options.unwrap_method)",
    )
    p.add_argument(
        "--ntiles",
        type=int,
        nargs=2,
        default=None,
        metavar=("NY", "NX"),
        help="SNAPHU internal tiling (default: single tile, the dolphin default)",
    )
    p.add_argument(
        "--n-parallel-tiles",
        type=int,
        default=1,
        help="SNAPHU tiles unwrapped in parallel (with --ntiles)",
    )
    p.add_argument(
        "--tophu-ntiles",
        type=int,
        nargs=2,
        default=None,
        metavar=("NY", "NX"),
        help="tophu tiling for --method icu/phass (default: dolphin's (1, 1))",
    )
    p.add_argument(
        "--tophu-downsample",
        type=int,
        nargs=2,
        default=None,
        metavar=("NY", "NX"),
        help=(
            "tophu coarse-unwrap downsample factor for --method icu/phass."
            " dolphin's default is (1, 1), i.e. the multiscale path is off."
        ),
    )
    p.add_argument(
        "--ww-threads",
        type=int,
        default=None,
        help=(
            "whirlwind rayon pool size (unwrap_options.whirlwind_options"
            ".num_threads). Default: ww's own default (all CPUs). The pool is"
            " shared across --n-parallel-jobs concurrent unwraps."
        ),
    )
    p.add_argument(
        "--n-parallel-jobs",
        type=int,
        default=1,
        help=(
            "interferograms unwrapped concurrently. The harness default of 1"
            " measures single-IG latency; dolphin's own default resolves -1 to"
            " cpu_count//4 for whirlwind and 1 for snaphu/spurt."
        ),
    )
    p.add_argument(
        "--preproc", choices=["none", "goldstein", "interp", "both"], default="both"
    )
    p.add_argument("-n", "--num-ifgs", type=int, default=3)
    p.add_argument("--out", default="/work/unwrap-breakdown")
    p.add_argument(
        "--no-copy-local",
        action="store_true",
        help="measure in place instead of copying inputs to local disk",
    )
    p.add_argument("--local-dir", default="/tmp/unwrap-bench")
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "numba/numpy thread cap. Default: worker_settings.threads_per_worker"
            " from the run config, replicating displacement.py's set_num_threads call."
        ),
    )
    args = p.parse_args()

    # Force tophu (and through it isce3/pyre) in before anything else can
    # trigger it lazily with argparse flags still on sys.argv.
    if args.method in ("icu", "phass"):
        _import_with_clean_argv("tophu")

    if args.ifg_files:
        # Direct-file mode (e.g. NISAR GUNW extractions).
        if args.nlooks is None:
            raise SystemExit("--nlooks is required with --ifg-files")
        ifgs = [Path(f) for f in args.ifg_files][: args.num_ifgs]
        cors = [Path(str(f).replace(".int.tif", ".int.cor.tif")) for f in ifgs]
        similarity = Path(args.similarity) if args.similarity else None
        nlooks = args.nlooks
        cfg_threads = 1  # displacement.py default caller shape
        label = args.label or "direct"
    else:
        work = Path(args.work)
        ifg_dir = work / "interferograms"
        ifgs = sorted(ifg_dir.glob("*.int.tif"))[: args.num_ifgs]
        if not ifgs:
            raise FileNotFoundError(f"no *.int.tif under {ifg_dir}")
        cors = [Path(str(f).replace(".int.tif", ".int.cor.tif")) for f in ifgs]
        sims = sorted(ifg_dir.glob("similarity_full_*.tif")) or sorted(
            ifg_dir.glob("similarity_*.tif")
        )
        similarity = sims[-1]

        cfg_path = Path(args.config) if args.config else work / "dolphin_config.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        hw = cfg["phase_linking"]["half_window"]
        # Same as displacement.py: nlooks = row_looks * col_looks from half window.
        nlooks = float((2 * hw["y"] + 1) * (2 * hw["x"] + 1))
        cfg_threads = int(cfg["worker_settings"]["threads_per_worker"])
        label = args.label or work.name
    for f in [*cors, *([similarity] if similarity else [])]:
        if not f.exists():
            raise FileNotFoundError(f)

    # Replicate displacement.py:223 — the workflow caps numba/numpy threads at
    # worker_settings.threads_per_worker before unwrapping (tutorial config: 1).
    from dolphin import utils

    num_threads = args.threads if args.threads is not None else cfg_threads
    utils.set_num_threads(num_threads)

    if not args.no_copy_local:
        local = Path(args.local_dir) / "inputs"
        local.mkdir(parents=True, exist_ok=True)

        def _stage(f: Path) -> Path:
            dst = local / Path(f).name
            if not dst.exists():
                shutil.copy2(f, dst)
            return dst

        ifgs = [_stage(f) for f in ifgs]
        cors = [_stage(f) for f in cors]
        if similarity is not None:
            similarity = _stage(similarity)
        base = Path(args.local_dir)
    else:
        base = Path(args.out)

    out_dir = base / f"unw-{label}-{args.method}-{args.preproc}"
    scratch = base / f"scratch-{label}-{args.method}-{args.preproc}"
    for d in (out_dir, scratch):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    shape = io.get_raster_xysize(ifgs[0])[::-1]
    opts = UnwrapOptions(
        run_unwrap=True,
        run_goldstein=args.preproc in ("goldstein", "both"),
        run_interpolation=args.preproc in ("interp", "both"),
        unwrap_method=args.method,
        n_parallel_jobs=args.n_parallel_jobs,
    )
    if args.ww_threads is not None:
        opts.whirlwind_options.num_threads = args.ww_threads
    if args.ntiles:
        opts.snaphu_options.ntiles = tuple(args.ntiles)
        opts.snaphu_options.n_parallel_tiles = args.n_parallel_tiles
    if args.tophu_ntiles:
        opts.tophu_options.ntiles = tuple(args.tophu_ntiles)
    if args.tophu_downsample:
        opts.tophu_options.downsample_factor = tuple(args.tophu_downsample)

    install_probes()
    t0 = time.perf_counter()
    unwrap_run(
        ifg_filenames=[str(f) for f in ifgs],
        cor_filenames=[str(f) for f in cors],
        output_path=out_dir,
        unwrap_options=opts,
        nlooks=nlooks,
        similarity_filename=str(similarity) if similarity else None,
        scratchdir=scratch,
    )
    wall = time.perf_counter() - t0

    try:
        commit = subprocess.run(
            ["git", "-C", "/dolphin", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = "unknown"

    import numba
    import numpy
    import snaphu

    # Record the versions of every solver actually reachable in this env, not
    # just the one selected: a comparison table is only readable if the reader
    # can tell which builds were in the room. Absent solvers are recorded as
    # None rather than omitted, so "not installed" is distinguishable from
    # "forgot to record".
    solver_versions: dict[str, str | None] = {
        "snaphu_py": getattr(snaphu, "__version__", "unknown")
    }
    for mod_name in ("isce3", "tophu", "whirlwind"):
        try:
            mod = _import_with_clean_argv(mod_name)
        except ImportError:
            solver_versions[mod_name] = None
        else:
            solver_versions[mod_name] = getattr(mod, "__version__", "unknown")

    # Per-method options, so a run stays self-describing (issue #13).
    method_options = {
        "snaphu": lambda: opts.snaphu_options.model_dump(mode="json"),
        "icu": lambda: opts.tophu_options.model_dump(mode="json"),
        "phass": lambda: opts.tophu_options.model_dump(mode="json"),
        "whirlwind": lambda: opts.whirlwind_options.model_dump(mode="json"),
    }[args.method]()

    # Written by docker/Dockerfile.unwrap: the resolved version of every solver
    # in the image, so a result names the stack it came from.
    solver_info = Path("/opt/conda/unwrap-solver-info.txt")

    summary = summarize()
    result = {
        "meta": {
            "label": label,
            "method": args.method,
            "solver_stage": SOLVER_STAGE[args.method],
            "method_options": method_options,
            "solver_versions": solver_versions,
            "solver_info": (
                solver_info.read_text() if solver_info.exists() else None
            ),
            "preproc": args.preproc,
            "num_ifgs": len(ifgs),
            "similarity_used": similarity is not None,
            "raster_shape_yx": list(shape),
            "nlooks": nlooks,
            "num_threads": num_threads,
            "snaphu_ntiles": list(args.ntiles) if args.ntiles else [1, 1],
            "snaphu_n_parallel_tiles": args.n_parallel_tiles,
            "n_parallel_jobs": args.n_parallel_jobs,
            "ww_threads_requested": args.ww_threads,
            "ww_thread_env": {
                k: os.environ.get(k)
                for k in ("WHIRLWIND_NUM_THREADS", "RAYON_NUM_THREADS")
            },
            "cpu_count": os.cpu_count(),
            "inputs_on_local_disk": not args.no_copy_local,
            "dolphin_commit": commit,
            "numpy": numpy.__version__,
            "numba": numba.__version__,
            "python": platform.python_version(),
            "hostname": platform.node(),
            "wall_seconds": wall,
        },
        "summary": summary,
        "records": RECORDS,
    }

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    json_file = out_path / f"breakdown-{label}-{args.method}-{args.preproc}.json"
    json_file.write_text(json.dumps(result, indent=2))

    total = summary["seconds"]["total"]
    n = len(ifgs)
    print(f"\n=== unwrap breakdown: method={args.method}, preproc={args.preproc}, {n} ifgs, ")
    print(f"    shape={shape}, nlooks={nlooks:.0f}, wall={wall:.1f}s ===")
    print(f"{'stage':22s} {'total_s':>9s} {'per_ifg_s':>10s} {'share':>7s} {'calls':>6s}")
    order = ["total", *SUMMARY_STAGES, "driver_io", "other"]
    for stage in order:
        if stage not in summary["seconds"]:
            continue
        sec = summary["seconds"][stage]
        ncalls = summary["calls"].get(stage, 0)
        print(
            f"{stage:22s} {sec:9.2f} {sec / n:10.2f} {100 * sec / total:6.1f}% {ncalls:6d}"
        )
    print(f"\nwritten: {json_file}")


if __name__ == "__main__":
    main()
