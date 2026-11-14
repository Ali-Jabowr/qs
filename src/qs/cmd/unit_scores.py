"""`qs unit-scores`: the off-colour scores of a block's full passes (#175).

    qs unit-scores --lane lane.yml --unit unit.yml --labels L.json --footage DIR --split split.yml
                   --block train|calibrate --out scores.csv
                   [--patchcore STATE=MODEL_ID ...] [--models models] [--device cuda|cpu]

For every full pass of the block's clips the crops of frames f-2 ... f+2 are cut, the best cut is
scored and one CSV row is written: colour spread, the unit's median Lab colour and, for a state
with a `--patchcore` bank (fitted by `qs train-lane-b --states`), the highest anomaly value
inside the unit. A pass with no cut-out is a row with the fault and no scores. The test block is
not offered: `qs unit-report` opens it, and only the block's clips are hashed and read.

Exit codes: 0 done · 1 refused (no full pass) · 2 could not run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from qs.cmd.train import _first_line
from qs.cmd.train_lane_b import _FULL
from qs.errors import ConfigError, InferenceError
from qs.inspector.appearance import real_cuda_available, real_scorer
from qs.inspector.bundle import load_bundle
from qs.inspector.model import Scorer, device_for
from qs.labels import CupState, FrameReadError, LabelsError, load_labels
from qs.lane.lanefile import load_lane
from qs.lane.split import Split, file_sha256, load_split, verify
from qs.lane.unit import load_unit
from qs.lane.unit_scores import score_block, write_scores


def add_parser(commands: Any) -> None:
    """Register `unit-scores` on the `qs` subcommands."""
    parser = commands.add_parser(
        "unit-scores", help="colour spread, unit colour and PatchCore read inside the cut-out"
    )
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--unit", required=True, metavar="PATH", help="the unit settings file")
    parser.add_argument("--labels", required=True, metavar="PATH",
                        help="the labels file (qs label)")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the folder of clips")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--block", required=True, choices=("train", "calibrate"),
                        help="whose clips to score (never test)")
    parser.add_argument("--patchcore", action="append", default=[], metavar="STATE=MODEL_ID",
                        help="the lane-b bundle for one state; repeat for each state")
    parser.add_argument("--out", required=True, metavar="PATH", help="the CSV to write")
    parser.add_argument("--models", default="models", metavar="DIR",
                        help="where bundles are read (models)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="the torch device (cuda; never a silent fallback to cpu)")


def _scorers(
    args: argparse.Namespace,
    make_scorer: Callable[[str, np.ndarray], Scorer],
    cuda_available: Callable[[], bool],
) -> dict[CupState, tuple[Scorer, int]]:
    """One scorer and its input size per `--patchcore STATE=MODEL_ID`."""
    chosen: dict[CupState, str] = {}
    for entry in args.patchcore:
        name, separator, model_id = entry.partition("=")
        try:
            state = CupState(name)
        except ValueError:
            state = None
        if not separator or state not in _FULL:
            raise ConfigError(f"--patchcore {entry!r}: expected STATE=MODEL_ID with STATE one of "
                              f"{', '.join(s.value for s in _FULL)}")
        if state in chosen:
            raise ConfigError(f"--patchcore: state {name!r} is given twice")
        chosen[state] = model_id
    if not chosen:
        return {}
    device = device_for("torch-cuda" if args.device == "cuda" else "torch-cpu",
                        cuda_available=cuda_available())
    scorers: dict[CupState, tuple[Scorer, int]] = {}
    for state, model_id in chosen.items():
        try:
            bundle = load_bundle(Path(args.models), model_id)
        except ConfigError as exc:
            raise ConfigError(f"--patchcore {state.value}: {exc}") from exc
        scorers[state] = (make_scorer(device, bundle.bank), bundle.input_px)
    return scorers


def _run(
    args: argparse.Namespace,
    make_scorer: Callable[[str, np.ndarray], Scorer],
    cuda_available: Callable[[], bool],
) -> int:
    try:
        lane = load_lane(Path(args.lane))
        unit = load_unit(Path(args.unit))
        scorers = _scorers(args, make_scorer, cuda_available)
    except ConfigError as exc:
        print(exc)
        return 2
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        print(f"labels file {labels_path} not found")
        return 2
    footage = Path(args.footage)
    try:
        labels = load_labels(labels_path, labels_path.stem)
        split = load_split(Path(args.split))
        videos = split.videos(args.block)
        verify(Split({video: split.clips[video] for video in videos}), footage, file_sha256)
    except (LabelsError, ConfigError) as exc:
        print(exc)
        return 2
    if not any(cup.state in _FULL for video in videos for cup in labels.cups_of(video)):
        print(f"refused: no full pass in the {args.block} clips")
        return 1
    try:
        rows = score_block(labels, footage, videos, lane, unit, scorers)
        write_scores(rows, Path(args.out))
    except FrameReadError as exc:
        print(f"footage: {exc}")
        return 2
    except (ConfigError, InferenceError, RuntimeError, OSError) as exc:
        print(_first_line(exc))
        return 2
    faults = Counter(row.fault for row in rows if row.fault)
    print(f"scored {len(rows)} passes of {len(videos)} clips; "
          f"{sum(faults.values())} without a cut-out")
    for detail, count in faults.items():
        print(f"{detail}: {count}")
    return 0


def main(
    args: argparse.Namespace,
    *,
    make_scorer: Callable[[str, np.ndarray], Scorer] = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    """Score one block; 0 done, 1 refused, 2 could not run."""
    try:
        return _run(args, make_scorer, cuda_available)
    except KeyboardInterrupt:
        print("stopped by the operator")
        return 2
