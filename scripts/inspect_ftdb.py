#!/usr/bin/env python3
"""
Quick sanity check for an FTDB file.

Usage:
  python scripts/inspect_ftdb.py /path/to/file.ftdb [--fonts 20] [--char A] [--no-validate]

Import strategy for FTDB reader:
  1) Try `data.ftdb_reader` (repo layout)
  2) Try local `ftdb_reader.py` next to this script

Outputs size, point/contour/glyph counts, and optional font/glyph info.
"""

from __future__ import annotations

from pathlib import Path
import sys
# Ensure the project root (parent of scripts/) is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from rich import print  # type: ignore
    from rich.table import Table  # type: ignore
    from rich.console import Console  # type: ignore
except Exception:  # fallback to plain prints
    Table = None
    class _C:  # type: ignore
        def print(self, *a, **k):
            print(*a, **k)
    Console = _C  # type: ignore

# ---------------------------------------------
# FTDB import shim
# ---------------------------------------------
FTDB = None
err: Exception | None = None
try:
    from data.ftdb_reader import FTDB  # type: ignore
except Exception as e:
    err = e
    try:
        # Add script dir to path and try local ftdb_reader.py
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        from ftdb_reader import FTDB  # type: ignore
    except Exception as e2:
        err = e2

if FTDB is None:
    msg = (
        "Could not import FTDB reader. Place `ftdb_reader.py` next to this script "
        "or under `data/ftdb_reader.py`.\nLast import error: " + repr(err)
    )
    raise ImportError(msg)


def human_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser(description="FTDB inspector")
    ap.add_argument("path", help="Path to .ftdb file")
    ap.add_argument("--fonts", type=int, default=10, help="Show first N fonts")
    ap.add_argument("--char", type=str, default=None, help="Optional character or codepoint (e.g. 'A' or 0x41)")
    ap.add_argument("--no-validate", action="store_true", help="Skip lightweight validation")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"[red]File not found:[/red] {p}")
        sys.exit(2)

    t0 = time.time()
    with FTDB(str(p)) as db:
        open_ms = (time.time() - t0) * 1000
        print(f"[bold]Opened:[/bold] {p}")
        print(f"  Size:     {human_size(db.size)}")
        print(f"  Points:   {db.point_count:,}")
        print(f"  Contours: {db.contour_count:,}")
        print(f"  Glyphs:   {db.glyph_count:,}")
        print(f"  Fonts:    {len(db.fonts):,}")
        print(f"  Open time: {open_ms:.1f} ms\n")

        if not args.no_validate:
            try:
                db.validate()
                print("[green]Validation: OK[/green]")
            except Exception as e:
                print(f"[red]Validation FAILED[/red]: {e}")
        print()

        # Show fonts
        N = max(0, int(args.fonts))
        if N:
            fonts = db.fonts[:N]
            if Table:
                table = Table(title=f"First {N} fonts")
                table.add_column("#", justify="right")
                table.add_column("Family")
                table.add_column("Subfamily")
                table.add_column("Glyphs")
                table.add_column("Path")
                for f in fonts:
                    table.add_row(str(f.index), f.family, f.subfamily,
                                  f"{f.first_glyph}..{f.first_glyph+f.num_glyphs-1}", f.path)
                Console().print(table)
            else:
                print(f"First {N} fonts:")
                for f in fonts:
                    print(f"[{f.index}] {f.family} — {f.subfamily}  (glyphs {f.first_glyph}..{f.first_glyph+f.num_glyphs-1})")
            print()

        # Optional glyph lookup by char/codepoint
        if args.char is not None:
            tok = args.char.strip()
            try:
                cp = ord(tok) if len(tok) == 1 else int(tok, 0)
            except Exception:
                print(f"[red]Bad --char value[/red]: {tok}")
                cp = None
            if cp is not None:
                gi = db.glyphs_for_codepoint(cp)
                print(f"U+{cp:04X} maps to {len(gi)} glyph(s): {gi[:16]}{' ...' if len(gi)>16 else ''}")
                if gi:
                    g0 = db.glyph(gi[0])
                    print(f"  First glyph #{g0.index}: codepoint=U+{g0.codepoint:04X}, contours={g0.num_contours}")
                    if g0.num_contours:
                        c0 = db.contour(g0.first_contour)
                        print(f"  First contour points: {c0.num_points}")

if __name__ == "__main__":
    main()