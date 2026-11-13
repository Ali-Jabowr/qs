"""`qs train-lane-b`: PatchCore fitted on the lane crop of every full pass of the train clips (#159).

    qs train-lane-b --lane lane.yml --labels L.json --footage DIR --split split.yml --product P
                    [--seed 42] [--coreset 0.1] [--input-px 256] [--models models]
                    [--device cuda|cpu]

Method B needs no empty cup. Each full pass (`full`, `full_white`, `full_pink`) of a clip the
split marks `train` gives one image: the lane crop at the labelled frame. The rim finder is
never called. Only train clips are hashed and read.

Exit codes: 0 written · 1 refused (no full pass) · 2 could not run.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from qs.cmd.train import _first_line, _real_patchcore, coreset, positive_int, product, seed
from qs.errors import ConfigError, InferenceError
from qs.inspector.appearance import real_cuda_available
from qs.inspector.bundle import BundleInfo, ImageRecord, pixels_sha256, write_bundle
from qs.inspector.model import FitSettings, PatchCore, Scorer, device_for
from qs.inspector.window import to_input
from qs.labels import CupState, FrameReadError, LabelsError, VideoFrames, load_labels
from qs.lane.lanefile import Lane, crop, load_lane
from qs.lane.split import Split, file_sha256, load_split, verify

_FULL = (CupState.FULL, CupState.FULL_WHITE, CupState.FULL_PINK)


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


class _CannotRun(Exception):
    """The run could not start: exit 2, one line."""


def add_parser(commands: Any) -> None:
    """Register `train-lane-b` on the `qs` subcommands."""
    parser = commands.add_parser(
        "train-lane-b", help="PatchCore on the lane crop of every full pass of the train clips"
    )
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--labels", required=True, metavar="PATH", help="the labels file (qs label)")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the folder of clips")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--product", required=True, type=product, metavar="P",
                        help="for the model_id, e.g. mochi-lane")
    parser.add_argument("--seed", type=seed, default=42, metavar="N", help="coreset seed (42)")
    parser.add_argument("--coreset", type=coreset, default=0.1, metavar="R",
                        help="coreset share (0.1)")
    parser.add_argument("--input-px", type=positive_int, default=256, metavar="N",
                        help="the model input (256)")
    parser.add_argument("--models", default="models", metavar="DIR",
                        help="where bundles are written (models)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="the torch device (cuda; never a silent fallback to cpu)")


def lane_b_score(pixels: np.ndarray, lane: Lane, scorer: Scorer, input_px: int) -> float:
    return 0.0  # stub


def main(
    args: argparse.Namespace,
    *,
    make_patchcore: Callable[[str], PatchCore] = _real_patchcore,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    return 99  # stub
