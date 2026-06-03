#!/usr/bin/env python3
"""Rank hotspots from py-spy collapsed/folded stacks.

py-spy `record --format raw` emits one line per unique stack:

    frame0;frame1;...;frameN <count>

where ``count`` is the number of samples whose leaf was ``frameN``. This script
aggregates those samples into:

* **self time** -- samples where a frame is the leaf (the frame actually on-CPU).
* **total time** -- samples where a frame appears anywhere in the stack (the
  frame plus everything it called). For a candidate we plan to accelerate, the
  total time under its frame is the ceiling on the speed-up.

Usage:
    folded_hotspots.py profile.folded [--top 20] [--grep goldstein,calc_ps_block]
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_folded(path: Path):
    """Return (self_counts, total_counts, total_samples).

    Frame labels are normalised to ``function (file:line)`` -> we keep the raw
    py-spy label, which already looks like ``func (path/file.py:123)``.
    """
    self_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    total_samples = 0

    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # Split trailing count off the last space.
            stack, _, count_str = line.rpartition(" ")
            try:
                count = int(count_str)
            except ValueError:
                continue
            frames = stack.split(";")
            if not frames:
                continue
            total_samples += count
            # self == leaf frame
            self_counts[frames[-1]] += count
            # total == each unique frame in the stack (dedupe per stack so
            # recursion doesn't double-count a single sample).
            for frame in set(frames):
                total_counts[frame] += count

    return self_counts, total_counts, total_samples


def strip_location(frame: str) -> str:
    """``func (a/b/c.py:12)`` -> ``func`` for grep matching."""
    return re.sub(r"\s*\(.*\)\s*$", "", frame)


def fmt_row(rank, label, count, total_samples):
    pct = 100.0 * count / total_samples if total_samples else 0.0
    return f"{rank:>3}. {pct:6.2f}%  {count:>8}  {label}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folded", type=Path)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--grep",
        default="goldstein,calc_ps_block",
        help="comma-separated function names to report explicitly",
    )
    args = ap.parse_args()

    self_counts, total_counts, total = parse_folded(args.folded)
    if not total:
        raise SystemExit(f"no samples parsed from {args.folded}")

    print(f"# total samples: {total}\n")

    print(f"## Top {args.top} by SELF time (on-CPU leaf)")
    for i, (frame, c) in enumerate(
        sorted(self_counts.items(), key=lambda kv: kv[1], reverse=True)[: args.top], 1
    ):
        print(fmt_row(i, frame, c, total))

    print(f"\n## Top {args.top} by TOTAL time (frame + callees)")
    for i, (frame, c) in enumerate(
        sorted(total_counts.items(), key=lambda kv: kv[1], reverse=True)[: args.top], 1
    ):
        print(fmt_row(i, frame, c, total))

    targets = [t.strip() for t in args.grep.split(",") if t.strip()]
    if targets:
        print("\n## Candidate functions (matched by name)")
        for name in targets:
            matched_self = {
                f: c for f, c in self_counts.items() if name in strip_location(f)
            }
            matched_total = {
                f: c for f, c in total_counts.items() if name in strip_location(f)
            }
            self_sum = sum(matched_self.values())
            total_sum = sum(matched_total.values())
            print(
                f"\n* {name!r}: self={self_sum} "
                f"({100.0 * self_sum / total:.2f}%)  "
                f"total={total_sum} ({100.0 * total_sum / total:.2f}%)"
            )
            for f, c in sorted(matched_total.items(), key=lambda kv: kv[1], reverse=True)[:6]:
                print(f"    total {100.0 * c / total:5.2f}%  {f}")


if __name__ == "__main__":
    main()
