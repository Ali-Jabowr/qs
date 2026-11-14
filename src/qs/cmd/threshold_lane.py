"""`qs threshold-lane`: the threshold from block C's full passes (#158)."""

from __future__ import annotations

import argparse
from typing import Any

from qs.cmd.train import positive_int


def add_parser(commands: Any) -> None:
    """Register `threshold-lane` on the `qs` subcommands."""
    parser = commands.add_parser("threshold-lane", help="the threshold from the calibrate clips")
    parser.add_argument("--bundle", required=True, metavar="MODEL_ID", help="the lane bundle")
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--labels", required=True, metavar="PATH", help="the labels file")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the clips' folder")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--out", required=True, metavar="PATH", help="the threshold file")
    parser.add_argument("--far", type=positive_int, default=1, metavar="PCT",
                        help="the share of full passes allowed above the threshold (1)")
    parser.add_argument("--window", type=positive_int, default=15, metavar="N",
                        help="frames a pass may sit from its run (15)")
    parser.add_argument("--models", default="models", metavar="DIR", help="the bundles' folder")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="cuda never falls back to the CPU")


def main(args: argparse.Namespace, **kwargs: Any) -> int:
    return 99  # stub
