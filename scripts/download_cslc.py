"""Download an OPERA CSLC-S1 stack from ASF for benchmarking dolphin.

Defaults reproduce the configuration from dolphin's basic walkthrough notebook
(West Texas, Sentinel-1 track 78, burst T078-165573-IW2, 12 months starting
2021-06-01). Skips files that already exist in the output directory, so re-runs
are cheap.

Credentials are read from the environment (ASF_USERNAME, ASF_PASSWORD), which
the Makefile / run.sh wire up from .env.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import asf_search as asf
from opera_utils.download import search_cslcs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--track", type=int, default=78,
                   help="Sentinel-1 relative orbit / track number. Default: 78")
    p.add_argument("--burst", default="t078_165573_iw2",
                   help="OPERA burst id (lowercase). Default: t078_165573_iw2")
    p.add_argument("--start", default="2021-06-01",
                   help="Acquisition start date (YYYY-MM-DD). Default: 2021-06-01")
    p.add_argument("--end", default="2022-06-01",
                   help="Acquisition end date (YYYY-MM-DD). Default: 2022-06-01")
    p.add_argument("--output", default="/cslc",
                   help="Output directory (inside the container). Default: /cslc")
    p.add_argument("--max-results", type=int, default=None,
                   help="Cap on number of files to download (None = all matches).")
    p.add_argument("--dry-run", action="store_true",
                   help="Search only; report what would be downloaded.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    user = os.environ.get("ASF_USERNAME")
    pwd = os.environ.get("ASF_PASSWORD")
    if not (user and pwd):
        print("error: ASF_USERNAME and ASF_PASSWORD must be set (use .env)",
              file=sys.stderr)
        return 2

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"search: track={args.track} burst={args.burst} "
          f"{args.start} .. {args.end}")
    results, _ = search_cslcs(
        track=args.track,
        burst_ids=[args.burst],
        start=args.start,
        end=args.end,
        max_results=args.max_results,
        check_missing_data=True,
    )
    if not results:
        print("no CSLCs match the query")
        return 1
    print(f"found {len(results)} CSLC(s)")

    to_download = []
    skipped = 0
    for r in results:
        dest = out / r.properties["fileName"]
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
        else:
            to_download.append(r)
    print(f"already-on-disk: {skipped}, will download: {len(to_download)}")

    if args.dry_run:
        for r in to_download:
            print(f"  would download: {r.properties['fileName']} "
                  f"({_size_mb(r):.0f} MB)")
        return 0

    if not to_download:
        print("nothing to do")
        return 0

    session = asf.ASFSession().auth_with_creds(user, pwd)
    t0 = time.time()
    total_bytes = 0
    for i, r in enumerate(to_download, 1):
        fname = r.properties["fileName"]
        size_mb = _size_mb(r)
        print(f"[{i}/{len(to_download)}] {fname} ({size_mb:.0f} MB) ...",
              flush=True)
        t_start = time.time()
        r.download(path=str(out), session=session)
        dt = time.time() - t_start
        dest = out / fname
        if not dest.exists():
            print(f"  ERROR: {fname} did not land on disk", file=sys.stderr)
            return 3
        actual_mb = dest.stat().st_size / 1024 / 1024
        total_bytes += dest.stat().st_size
        speed = actual_mb / dt if dt > 0 else 0
        print(f"  done in {dt:.1f}s ({speed:.0f} MB/s)")

    total_gb = total_bytes / 1024 ** 3
    elapsed = time.time() - t0
    print(f"\ndownloaded {len(to_download)} file(s), {total_gb:.2f} GB total, "
          f"elapsed {elapsed/60:.1f} min")
    return 0


def _size_mb(result) -> float:
    """Extract the H5 data file's size in MB from asf_search result metadata.

    Recent asf_search nests bytes per-file:
        {'<name>.h5': {'bytes': 273737588, 'format': 'HDF5'},
         '<name>.iso.xml': {'bytes': 221703, 'format': 'XML'}}
    We pull the entry whose format is HDF5 (falling back to the largest).
    """
    b = result.properties.get("bytes")
    if isinstance(b, (int, float)):
        return float(b) / 1024 / 1024
    if not isinstance(b, dict):
        return 0.0
    candidates = []
    for entry in b.values():
        if isinstance(entry, dict) and isinstance(entry.get("bytes"), (int, float)):
            candidates.append((entry.get("format") == "HDF5", entry["bytes"]))
    if not candidates:
        return 0.0
    candidates.sort(reverse=True)  # HDF5 first, then largest
    return float(candidates[0][1]) / 1024 / 1024


if __name__ == "__main__":
    raise SystemExit(main())
