"""`qs build`: one raw dataset session → the dataset tree and the frozen golden set.

    bin/qs build --session DIR --product P [--data data] [--name mvp-YYYY-MM-DD]

Reads `plan.yml`, `session.yml`, `frames.csv` and the top-level `*.png` of DIR, exactly as
`qs shoot` and `qs import-phone` wrote them, checks the whole build (docs/12 §2, docs/13 §6;
FR-52, FR-56), and only then writes `datasets/<product>/<geometry>/` and the frozen golden
set `golden/<name>/` under `--data`. A refusal names the unit, file or field and writes
nothing; the raw session is never modified; an existing target is never overwritten.

Exit codes: 0 done (or already built) · 1 refused · 2 could not run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qs.eval.golden import NAME_RE, Build, BuildRefused, GoldenRefused, plan_build, write_build
from qs.session import FRAMES_FILE, PLAN_FILE, SESSION_FILE


def golden_name(text: str) -> str:
    """argparse type: the golden set's name, e.g. `mvp-2026-09-25`."""
    if NAME_RE.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(
            f"{text!r}: lower-case words joined by '-', e.g. mvp-2026-09-25"
        )
    return text


def main(args: argparse.Namespace) -> int:
    """Build; 0 done or already built, 1 refused, 2 could not run."""
    session = Path(args.session)
    data = Path(args.data)
    for what in (session, session / SESSION_FILE, session / PLAN_FILE,
                 session / FRAMES_FILE):
        if not what.exists():
            print(f"build: {what} is missing — a raw session as qs shoot or qs import-phone "
                  "wrote it")
            return 2
    try:
        build = plan_build(session, data, args.product, args.name)
    except BuildRefused as exc:
        print(f"build: {exc} — nothing written")
        return 1
    try:
        write_build(build)
    except (OSError, GoldenRefused) as exc:
        print(f"build: {exc}")
        return 1
    _report(build)
    return 0


def _report(build: Build) -> None:
    counts = " · ".join(
        f"{name} {frames} frames" if name == "placement" else f"{name} {units} units / "
        f"{frames} frames"
        for name, units, frames in build.counts
    )
    print(f"build: {counts}")
    print(f"dataset: {build.dataset.root} — "
          f"{'already built' if build.dataset_done else 'written'}")
    print(f"golden: {build.golden.root} — "
          f"{'already built' if build.golden_done else 'written, verified'}")
    print(f"golden sha256: {build.golden_hash}")
    print(f"next: qs train --dataset {build.dataset.root}")
    print(f"      qs calibrate --dataset {build.dataset.root}")
    print(f"      qs evaluate --golden {build.golden.root}")
