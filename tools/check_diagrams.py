#!/usr/bin/env python3
"""Render every Mermaid diagram in the repo to prove it actually renders.

    python3 tools/check_diagrams.py          # validate only
    python3 tools/check_diagrams.py --svg    # also write docs/diagrams/*.svg

A diagram that fails to parse renders as a red error box. On a screen in front of
a client that is worse than having no diagram at all, so this runs in CI.

Sources:
    docs/**/*.md         fenced ```mermaid blocks, named <file>-<n>
    docs/diagrams/*.mmd  standalone diagrams

Requires node. Uses @mermaid-js/mermaid-cli via npx; the first run downloads it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = DOCS / "diagrams"
FENCE = re.compile(r"```mermaid\n(.*?)```", re.S)


def collect() -> list[tuple[str, str, Path]]:
    """Return (name, source, origin) for every diagram in the repo."""
    found = []
    for md in sorted(DOCS.rglob("*.md")):
        blocks = FENCE.findall(md.read_text(encoding="utf-8"))
        for i, block in enumerate(blocks, 1):
            stem = md.stem
            name = f"{stem}-{i}" if len(blocks) > 1 else stem
            found.append((name, block, md))
    for mmd in sorted(OUT.glob("*.mmd")):
        found.append((mmd.stem, mmd.read_text(encoding="utf-8"), mmd))
    return found


def render(source: str, dest: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "d.mmd"
        src.write_text(source, encoding="utf-8")
        proc = subprocess.run(
            ["npx", "--yes", "@mermaid-js/mermaid-cli",
             "-i", str(src), "-o", str(dest), "-b", "transparent"],
            capture_output=True, text=True,
        )
    if proc.returncode != 0 or not dest.exists():
        return False, (proc.stderr or proc.stdout).strip()
    return True, ""


def main() -> int:
    if not shutil.which("npx"):
        print("npx not found — cannot check diagrams. Install node.", file=sys.stderr)
        return 2

    keep_svg = "--svg" in sys.argv
    diagrams = collect()
    if not diagrams:
        print("no diagrams found")
        return 0

    if keep_svg:
        OUT.mkdir(parents=True, exist_ok=True)

    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        for name, source, origin in diagrams:
            dest = (OUT / f"{name}.svg") if keep_svg else Path(scratch) / f"{name}.svg"
            ok, err = render(source, dest)
            rel = origin.relative_to(ROOT)
            if ok:
                where = f" -> docs/diagrams/{name}.svg" if keep_svg else ""
                print(f"  ok    {name:<22} ({rel}){where}")
            else:
                print(f"  FAIL  {name:<22} ({rel})")
                failures.append((name, rel, err))

    print()
    if failures:
        for name, rel, err in failures:
            print(f"{name} in {rel}:", file=sys.stderr)
            for line in err.splitlines()[:8]:
                print(f"    {line}", file=sys.stderr)
        print(f"{len(failures)} of {len(diagrams)} diagram(s) failed.", file=sys.stderr)
        return 1

    print(f"{len(diagrams)} diagram(s) render.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
