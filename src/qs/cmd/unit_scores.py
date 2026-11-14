"""`qs unit-scores`: the off-colour scores of a block's full passes (#175)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import numpy as np

from qs.inspector.appearance import real_cuda_available, real_scorer
from qs.inspector.model import Scorer


def add_parser(commands: Any) -> None:
    """Register `unit-scores` on the `qs` subcommands."""
    parser = commands.add_parser("unit-scores", help="off-colour scores of the cut-out units")
    parser.add_argument("--lane", required=True, metavar="PATH")
    parser.add_argument("--unit", required=True, metavar="PATH")
    parser.add_argument("--labels", required=True, metavar="PATH")
    parser.add_argument("--footage", required=True, metavar="DIR")
    parser.add_argument("--split", required=True, metavar="PATH")
    parser.add_argument("--block", required=True, choices=("train", "calibrate"))
    parser.add_argument("--patchcore", action="append", default=[], metavar="STATE=MODEL_ID")
    parser.add_argument("--out", required=True, metavar="PATH")
    parser.add_argument("--models", default="models", metavar="DIR")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")


def main(
    args: argparse.Namespace,
    *,
    make_scorer: Callable[[str, np.ndarray], Scorer] = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    raise NotImplementedError
