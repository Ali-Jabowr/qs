"""`qs train-lane` (#156)."""

from __future__ import annotations

import argparse
from typing import Any


def add_parser(commands: Any) -> None:
    parser = commands.add_parser("train-lane", help="train the lane classifier")
    parser.add_argument("--lane", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--footage", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--input-px", type=int, default=224)
    parser.add_argument("--models", default="models")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")


def main(args: argparse.Namespace, **_: Any) -> int:
    return 99
