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
- ``transfer_ambiguities``  ambiguity transfer back to the original ifg
- ``grow_conncomp_snaphu``  connected-component regrow (a second SNAPHU run;
                            only when goldstein/interpolation ran)
- ``set_nodata_values``     nodata fixup on unw + conncomp rasters
- ``create_combined_mask``  nodata mask combine (when a mask applies)
- ``io.load_gdal`` / ``io.write_arr``  raster I/O issued by the driver itself

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
import json
import platform
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import yaml

import dolphin.unwrap._unwrap as unwrap_mod
from dolphin import io
from dolphin.unwrap import run as unwrap_run
from dolphin.workflows import UnwrapOptions

RECORDS: list[dict] = []
CURRENT: dict = {"ifg": None}
_STACK: list[str] = []

# Stages summed in the summary table. Everything else under "total" becomes
# "other" (scratch cleanup, intermediate moves, mask handling, ...).
SUMMARY_STAGES = [
    "goldstein",
    "interpolate",
    "snaphu",
    "transfer_ambiguities",
    "conncomp_regrow",
    "set_nodata",
    "combined_mask",
]


def _timed(func, stage: str):
    def wrapper(*args, **kwargs):
        _STACK.append(stage)
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            _STACK.pop()
            RECORDS.append(
                {
                    "ifg": CURRENT["ifg"],
                    "stage": stage,
                    "within": _STACK[-1] if _STACK else None,
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
        CURRENT["ifg"] = Path(kwargs["ifg_filename"]).name
        _STACK.append("total")
        t0 = time.perf_counter()
        try:
            return orig_unwrap(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            _STACK.pop()
            RECORDS.append(
                {"ifg": CURRENT["ifg"], "stage": "total", "within": None, "seconds": dt}
            )
            CURRENT["ifg"] = None

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

    out_dir = base / f"unw-{label}-{args.preproc}"
    scratch = base / f"scratch-{label}-{args.preproc}"
    for d in (out_dir, scratch):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    shape = io.get_raster_xysize(ifgs[0])[::-1]
    opts = UnwrapOptions(
        run_unwrap=True,
        run_goldstein=args.preproc in ("goldstein", "both"),
        run_interpolation=args.preproc in ("interp", "both"),
        unwrap_method="snaphu",
        n_parallel_jobs=1,
    )
    if args.ntiles:
        opts.snaphu_options.ntiles = tuple(args.ntiles)
        opts.snaphu_options.n_parallel_tiles = args.n_parallel_tiles

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

    summary = summarize()
    result = {
        "meta": {
            "label": label,
            "preproc": args.preproc,
            "num_ifgs": len(ifgs),
            "similarity_used": similarity is not None,
            "raster_shape_yx": list(shape),
            "nlooks": nlooks,
            "num_threads": num_threads,
            "snaphu_ntiles": list(args.ntiles) if args.ntiles else [1, 1],
            "snaphu_n_parallel_tiles": args.n_parallel_tiles,
            "inputs_on_local_disk": not args.no_copy_local,
            "dolphin_commit": commit,
            "numpy": numpy.__version__,
            "numba": numba.__version__,
            "snaphu_py": getattr(snaphu, "__version__", "unknown"),
            "hostname": platform.node(),
            "wall_seconds": wall,
        },
        "summary": summary,
        "records": RECORDS,
    }

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    json_file = out_path / f"breakdown-{label}-{args.preproc}.json"
    json_file.write_text(json.dumps(result, indent=2))

    total = summary["seconds"]["total"]
    n = len(ifgs)
    print(f"\n=== unwrap breakdown: preproc={args.preproc}, {n} ifgs, ")
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
