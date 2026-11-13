"""`qs lane-split`: give each recorded clip one role and pin its hash (#154).

    qs lane-split --footage DIR --train T-1 [T-2 ...] --calibrate C-1 --test E-1 --out split.yml

Each clip's sidecar (`DIR/<clip>.json`) is read and `DIR/<clip>.mkv` is hashed; a clip in two
roles, a missing sidecar, a changed hash or a drifted clip is refused (`--allow-drifted` keeps
the last). Every role needs at least one clip.

Exit codes: 0 written · 2 refused.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qs.errors import ConfigError
from qs.lane.split import build_split, file_sha256, write_split


def add_parser(commands: Any) -> None:
    """Register `lane-split` on the `qs` subcommands."""
    parser = commands.add_parser("lane-split", help="give each recorded clip one role")
    parser.add_argument("--footage", required=True, metavar="DIR",
                        help="the folder holding the clips and their sidecars")
    parser.add_argument("--train", nargs="+", default=[], metavar="CLIP")
    parser.add_argument("--calibrate", nargs="+", default=[], metavar="CLIP")
    parser.add_argument("--test", nargs="+", default=[], metavar="CLIP")
    parser.add_argument("--out", required=True, metavar="PATH", help="the split file to write")
    parser.add_argument("--allow-drifted", action="store_true",
                        help="keep a clip whose camera settings drifted while recording")


def main(args: argparse.Namespace) -> int:
    try:
        split = build_split(Path(args.footage), args.train, args.calibrate, args.test,
                            file_sha256, args.allow_drifted)
        write_split(split, Path(args.out))
    except ConfigError as exc:
        print(f"lane-split: {exc}")
        return 2
    except OSError as exc:
        print(f"lane-split: {args.out} cannot be written: {exc}")
        return 2
    print(f"lane-split: {len(split.clips)} clips written to {args.out}")
    return 0
