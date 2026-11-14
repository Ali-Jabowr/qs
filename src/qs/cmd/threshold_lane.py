"""`qs threshold-lane`: the threshold from block C's full passes (#158)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from qs.cmd.train import positive_int
from qs.errors import ConfigError
from qs.inspector.appearance import real_cuda_available
from qs.labels import CupState, FrameReadError, LabelsError, load_labels
from qs.lane.lanefile import load_lane
from qs.lane.model import load_lane_bundle
from qs.lane.passes import frame_probs, pair_runs, runs_from_probs
from qs.lane.split import Split, file_sha256, load_split, verify
from qs.lane.threshold import HALF, Threshold, lane_threshold, write_threshold

_FULL = (CupState.FULL, CupState.FULL_WHITE, CupState.FULL_PINK)
Probs = Callable[..., np.ndarray]


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


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


def _run(args: argparse.Namespace, probs: Probs) -> int:
    out = Path(args.out)
    if out.exists():
        raise ConfigError(f"{out} already exists: delete it to write a new threshold")
    if args.device == "cuda" and not real_cuda_available():
        raise ConfigError("device: --device cuda, but torch sees no CUDA device")
    lane = load_lane(Path(args.lane))
    split = load_split(Path(args.split))
    calibrate = split.videos("calibrate")
    if not calibrate:
        raise ConfigError("no calibrate clip in the split file")
    footage = Path(args.footage)
    # Only the calibrate clips are hashed and opened: a test clip is never touched.
    verify(Split({v: split.clips[v] for v in calibrate}), footage, file_sha256)
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise ConfigError(f"labels file {labels_path} not found")
    try:
        labels = load_labels(labels_path, labels_path.stem)
    except LabelsError as exc:
        raise ConfigError(f"labels: {exc}") from exc
    bundle = load_lane_bundle(Path(args.models), args.bundle)
    input_px = bundle.record["input_px"]
    scores: list[float] = []
    unseen = 0
    for video in calibrate:
        cups = labels.cups_of(video)
        if not cups:
            continue
        rows = probs(footage / video, lane, bundle.model, args.device, input_px=input_px)
        pairs, _ = pair_runs(cups, runs_from_probs(rows.tolist()), args.window)
        for cup, run in pairs:
            if cup.state not in _FULL:
                continue
            if run is None or run.score is None:
                unseen += 1
            else:
                scores.append(run.score)
    try:
        value = lane_threshold(scores, args.far)
    except ValueError as exc:
        raise _Refused(str(exc)) from exc
    above = sum(1 for s in scores if s > HALF)
    write_threshold(
        Threshold(bundle.model_id, args.far, value, len(scores), above, unseen,
                  file_sha256(labels_path)),
        out,
    )
    print(f"full passes used {len(scores)} of {len(scores) + unseen} in "
          f"{len(calibrate)} calibrate clips")
    print(f"above {HALF}: {above}")
    print(f"n_unseen {unseen}")
    print(f"threshold {value:g} (far {args.far}%)")
    print(f"threshold file: {out}")
    return 0


def main(args: argparse.Namespace, *, probs: Probs = frame_probs) -> int:
    """Write the threshold file; 0 written, 1 refused, 2 could not run."""
    try:
        return _run(args, probs)
    except _Refused as exc:
        print(f"threshold-lane: refused: {exc}")
        return 1
    except (ConfigError, FrameReadError, OSError) as exc:
        print(f"threshold-lane: {exc}")
        return 2
    except KeyboardInterrupt:
        print("threshold-lane: stopped by the operator — no file written")
        return 2
