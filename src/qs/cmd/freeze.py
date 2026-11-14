"""`qs freeze`: pin the model, the threshold, the lane, the split and the commit (#160).

    qs freeze --lane-bundle ID --threshold threshold.yml --lane lane.yml --split split.yml
              --out freeze.yml [--patchcore MODEL_ID --patchcore-threshold X] [--models models]

After this, `qs compare` may score the test clips, and only while every hash here still matches
and HEAD is the commit recorded. The tree must be clean (tracked files) and the output file new.

Exit codes: 0 written · 2 refused, one line.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from qs.errors import ConfigError
from qs.eval.scoring import utc_now
from qs.lane.freeze import (
    Freeze,
    model_file_sha256,
    run_git,
    threshold_of,
    write_freeze,
)
from qs.lane.split import file_sha256


def add_parser(commands: Any) -> None:
    """Register `freeze` on the `qs` subcommands."""
    parser = commands.add_parser("freeze", help="pin what the test block will be scored with")
    parser.add_argument("--lane-bundle", required=True, metavar="ID", help="the lane bundle")
    parser.add_argument("--threshold", required=True, metavar="PATH", help="the threshold file")
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--out", required=True, metavar="PATH", help="the freeze file to write")
    parser.add_argument("--patchcore", metavar="MODEL_ID", help="method B's model")
    parser.add_argument("--patchcore-threshold", type=float, metavar="X",
                        help="method B's threshold")
    parser.add_argument("--models", default="models", metavar="DIR",
                        help="where bundles are (models)")


def _freeze(
    args: argparse.Namespace, git: Callable[[Sequence[str]], str], clock: Callable[[], datetime]
) -> None:
    out = Path(args.out)
    if out.exists():
        raise ConfigError(f"{out} already exists; a freeze is never overwritten")
    dirty = [line for line in git(["status", "--porcelain", "--untracked-files=no"]).splitlines()
             if line.strip()]
    if dirty:
        raise ConfigError(f"the working tree has uncommitted changes to {len(dirty)} tracked "
                          f"file(s); commit them first")
    if (args.patchcore is None) != (args.patchcore_threshold is None):
        raise ConfigError("--patchcore and --patchcore-threshold go together")
    patchcore: tuple[str, float] | None = None
    if args.patchcore is not None:
        if not math.isfinite(args.patchcore_threshold):
            raise ConfigError(f"--patchcore-threshold must be finite, not "
                              f"{args.patchcore_threshold}")
        patchcore = (args.patchcore, float(args.patchcore_threshold))
    threshold = Path(args.threshold)
    threshold_model, _ = threshold_of(threshold)
    if threshold_model != args.lane_bundle:
        raise ConfigError(f"{threshold} was made for {threshold_model}, not {args.lane_bundle}")
    try:
        hashes = [file_sha256(Path(path)) for path in (threshold, args.lane, args.split)]
    except OSError as exc:
        raise ConfigError(f"cannot read a frozen file: {exc}") from exc
    model_sha = model_file_sha256(Path(args.models), args.lane_bundle)
    write_freeze(out, Freeze(
        git_commit=git(["rev-parse", "HEAD"]).strip(),
        model_id=args.lane_bundle,
        model_sha256=model_sha,
        threshold_sha256=hashes[0],
        lane_sha256=hashes[1],
        split_sha256=hashes[2],
        patchcore=patchcore,
        created_utc=clock().isoformat(),
    ))
    print(f"froze {args.lane_bundle} at {out}", file=sys.stderr)


def main(
    args: argparse.Namespace,
    *,
    git: Callable[[Sequence[str]], str] = run_git,
    clock: Callable[[], datetime] = utc_now,
) -> int:
    """Write the freeze file; 0 written, 2 refused."""
    try:
        _freeze(args, git, clock)
    except ConfigError as exc:
        print(f"freeze: {exc}")
        return 2
    return 0
