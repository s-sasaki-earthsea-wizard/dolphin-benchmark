#!/usr/bin/env python3
"""Render a flame graph SVG from py-spy collapsed/folded stacks.

A self-contained replacement for Brendan Gregg's ``flamegraph.pl`` so the
benchmark repo has no Perl dependency. Reads ``frame0;frame1;... <count>``
lines (``py-spy record --format raw``) and writes an interactive SVG: hover a
frame for its sample count / percent, click to zoom is *not* implemented to
keep this dependency-free, but the static graph is enough for hotspot triage.

Usage:
    flamegraph_from_folded.py profile.folded -o profile.svg [--title "..."]
"""

from __future__ import annotations

import argparse
import colorsys
import html
from pathlib import Path

WIDTH = 1200
FRAME_H = 16
PAD = 10
FONT = "Verdana, sans-serif"
MIN_PX = 0.1  # skip frames narrower than this many px


class Node:
    __slots__ = ("name", "value", "children")

    def __init__(self, name: str):
        self.name = name
        self.value = 0
        self.children: dict[str, "Node"] = {}

    def child(self, name: str) -> "Node":
        n = self.children.get(name)
        if n is None:
            n = self.children[name] = Node(name)
        return n


def build_tree(path: Path) -> Node:
    root = Node("all")
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            stack, _, count_str = line.rpartition(" ")
            try:
                count = int(count_str)
            except ValueError:
                continue
            root.value += count
            node = root
            for frame in stack.split(";"):
                node = node.child(frame)
                node.value += count
    return root


def color_for(name: str) -> str:
    """Warm 'hot' palette, hashed by name so the same frame keeps its color."""
    h = (hash(name) & 0xFFFFFFFF) / 0xFFFFFFFF
    hue = 0.02 + 0.10 * h          # red -> orange
    sat = 0.65 + 0.20 * ((hash(name) >> 8 & 0xFF) / 255.0)
    val = 0.85 + 0.10 * ((hash(name) >> 16 & 0xFF) / 255.0)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, min(val, 1.0))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folded", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--title", default="py-spy flame graph")
    args = ap.parse_args()

    root = build_tree(args.folded)
    total = root.value
    if not total:
        raise SystemExit(f"no samples parsed from {args.folded}")

    px_per_sample = (WIDTH - 2 * PAD) / total

    rects: list[str] = []
    max_depth = 0

    def walk(node: Node, depth: int, x: float):
        nonlocal max_depth
        w = node.value * px_per_sample
        if w < MIN_PX:
            return
        max_depth = max(max_depth, depth)
        if depth > 0:  # skip drawing the synthetic root bar
            pct = 100.0 * node.value / total
            label = node.name
            title = f"{html.escape(label)} ({node.value} samples, {pct:.2f}%)"
            # y grows downward; deepest at bottom would be classic icicle, but
            # py-spy/Gregg draw root at bottom. We draw root at top for simplicity.
            y = PAD + (depth - 1) * FRAME_H
            text = ""
            if w > 28:  # only label frames wide enough to read
                short = html.escape(label.split(" (")[0])[: int(w / 7)]
                text = (
                    f'<text x="{x + 2:.1f}" y="{y + FRAME_H - 4}" '
                    f'font-size="11" font-family="{FONT}">{short}</text>'
                )
            rects.append(
                f'<g><title>{title}</title>'
                f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{FRAME_H - 1}" '
                f'fill="{color_for(label)}" rx="1" ry="1"/>{text}</g>'
            )
        # children laid left-to-right, largest first for stable look
        cx = x
        for ch in sorted(node.children.values(), key=lambda c: c.name):
            walk(ch, depth + 1, cx)
            cx += ch.value * px_per_sample

    walk(root, 0, PAD)

    height = 2 * PAD + (max_depth + 1) * FRAME_H + 30
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height}" font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{height}" fill="#f8f8f8"/>',
        f'<text x="{PAD}" y="{height - 12}" font-size="13" font-weight="bold">'
        f'{html.escape(args.title)} — {total} samples</text>',
        *rects,
        "</svg>",
    ]
    args.output.write_text("\n".join(svg))
    print(f"wrote {args.output} ({total} samples, depth {max_depth})")


if __name__ == "__main__":
    main()
