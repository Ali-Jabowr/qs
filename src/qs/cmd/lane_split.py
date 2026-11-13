"""Stub."""

from __future__ import annotations

import argparse
from typing import Any


def add_parser(commands: Any) -> None:
    parser = commands.add_parser("lane-split", help="give each recorded clip one role")
    parser.add_argument("--footage", required=True)
    parser.add_argument("--train", nargs="+", default=[])
    parser.add_argument("--calibrate", nargs="+", default=[])
    parser.add_argument("--test", nargs="+", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-drifted", action="store_true")


def main(args: argparse.Namespace) -> int:
    return 1
