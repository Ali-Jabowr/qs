"""`qs line-alarms`: the line model's false alarms on held-out good frames (#110).

    qs line-alarms --config C --labels PATH --footage DIR --model MODEL_ID
                   [--models models] [--gap-s 3] [--far 5]

The frames of one camera position labelled `unit` are split by time exactly as `qs train` split
them (`qs.eval.line_split`, the same frames and `--gap-s`), so the model never saw the two
held-out blocks; a held-out frame the bundle lists among its training images is refused. Each
held-out frame is scored as the line station scores it: the cup wholly in the zone
(`find_cup`), the window of the bundle's size centred on the rim, the map masked to the cup's
inside (`check_appearance`). The threshold is chosen on the calibrate block at `--far`
(`choose_threshold`), and the test block's frames above it are the false alarms. Only numbers are
printed: no image, config or record is written, so the threshold reaches a station config by
hand.

Exit codes: 0 counted · 1 refused (a held-out frame with no cup or trained on, an empty block)
· 2 could not run (config, labels, footage, bundle).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from qs.cmd.train import DEFAULT_GAP_S, non_negative_float
from qs.config import LineConfig, StationConfig, load_config
from qs.errors import ConfigError
from qs.eval.line_split import LineSplitError, split_units, video_frame
from qs.eval.stats import upper_bound
from qs.eval.threshold import UnitScore, choose_threshold
from qs.inspector.appearance import (
    LoadedModel,
    check_appearance,
    load_model,
    real_cuda_available,
    real_scorer,
)
from qs.inspector.cup import find_cup
from qs.inspector.model import Scorer
from qs.labels import FrameReadError, LabelsError, VideoFrames, load_labels


class _Refused(Exception):
    """The run is refused: exit 1, one line."""


class _CannotRun(Exception):
    """The run could not start: exit 2, one line."""


def add_parser(commands: Any) -> None:
    """Register `line-alarms` on the `qs` subcommands."""
    parser = commands.add_parser(
        "line-alarms", help="false alarms of a line model on held-out good frames"
    )
    parser.add_argument("--config", required=True, metavar="PATH", help="the line station config")
    parser.add_argument("--labels", required=True, metavar="PATH",
                        help="the labels file of one camera position (from qs label)")
    parser.add_argument("--footage", required=True, metavar="DIR",
                        help="the folder holding the labelled videos")
    parser.add_argument("--model", required=True, metavar="MODEL_ID",
                        help="the bundle `qs train` wrote from the same labels")
    parser.add_argument("--models", default="models", metavar="DIR",
                        help="where the bundles are (models)")
    parser.add_argument("--gap-s", type=non_negative_float, default=DEFAULT_GAP_S, metavar="S",
                        help=f"seconds dropped before each held-out block, as in training "
                        f"({DEFAULT_GAP_S:g})")
    parser.add_argument("--far", type=int, default=5, metavar="PCT",
                        help="the false-alarm target the threshold is set at (5)")


def _line_config(path: Path) -> tuple[StationConfig, LineConfig, float]:
    config = load_config(path).config
    line = config.inspector.line
    if config.inspector.type != "line" or line is None:
        raise _CannotRun(f"{path}: not a line config")
    if line.fps is None:
        raise _CannotRun("inspector.line.fps: the time split needs this key")
    return config, line, line.fps


def _scores(
    frame_ids: Sequence[str], footage: Path, config: StationConfig, line: LineConfig,
    model: LoadedModel,
) -> list[float]:
    """Each frame's appearance score, as the line station computes it."""
    videos: dict[str, VideoFrames] = {}
    scores: list[float] = []
    try:
        for frame_id in frame_ids:
            video, index = video_frame(frame_id)
            if video not in videos:
                videos[video] = VideoFrames(footage / video)
            pixels = videos[video].read(index)
            cup = find_cup(pixels, line)
            if cup is None:
                raise _Refused(f"{frame_id}: no cup wholly in the zone")
            check, _, _ = check_appearance(pixels, cup.rim[0], cup.inside, model, config)
            if check.value is None:  # check_appearance always sets the score
                raise RuntimeError(f"{frame_id}: the appearance check carries no score")
            scores.append(check.value)
    finally:
        for frames in videos.values():
            frames.close()
    return scores


def _report(args: argparse.Namespace, make_scorer: Callable[[str, np.ndarray], Scorer],
            cuda_available: Callable[[], bool]) -> list[str]:
    config, line, fps = _line_config(Path(args.config))
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise _CannotRun(f"labels file {labels_path} not found")
    labels = load_labels(labels_path, labels_path.stem)
    try:
        split = split_units(labels, fps=fps, gap_s=args.gap_s)
    except LineSplitError as exc:
        raise _Refused(str(exc)) from None
    if not split.calibrate or not split.test:
        raise _Refused(f"{labels_path.name}: the calibrate or test block holds no frame")
    config = replace(config, inspector=replace(config.inspector, model=args.model))
    model = load_model(Path(args.models), config, validation=False,
                       make_scorer=make_scorer, cuda_available=cuda_available)
    trained = {image.file for image in model.bundle.images}
    seen = [frame_id for frame_id in (*split.calibrate, *split.test) if frame_id in trained]
    if seen:
        raise _Refused(f"{seen[0]}: the model was trained on this held-out frame; use the "
                       "labels and --gap-s it was trained with")
    footage = Path(args.footage)
    calibrate = _scores(split.calibrate, footage, config, line, model)
    test = _scores(split.test, footage, config, line, model)
    choice = choose_threshold(
        [UnitScore(f, s, geometry_failed=False) for f, s in zip(split.calibrate, calibrate,
                                                                 strict=True)],
        args.far,
    )
    threshold = choice.value
    if threshold is None:  # no unit is geometry-failed, so the rule always chooses one
        raise RuntimeError("choose_threshold chose no threshold with no unit flagged")
    n, k = len(test), sum(score > threshold for score in test)
    return [
        "# INDICATIVE — false alarms of the line model on held-out good frames",
        "",
        "These numbers are **INDICATIVE**: one visit's footage, labelled by hand; not Gate A and "
        "not a client claim (ADR-0007).",
        "Recall on real defects is unmeasured: no labelled defect exists, so this report counts "
        "false alarms only.",
        "Frames, not cups: neighbouring frames often show the same cup, so they are not "
        "independent and the bound is tighter than the truth.",
        "",
        f"- model: `{args.model}`; {len(split.train)} frames trained on, "
        f"{len(split.dropped)} dropped in the {args.gap_s:g} s gaps",
        f"- threshold: {threshold:.6g}",
        f"- threshold set on {choice.n} calibrate frames at a {args.far}% target, "
        f"{choice.a} allowed above it",
        f"- false alarms on test frames: {k} of {n} = {k / n * 100:.1f}%, "
        f"upper bound {upper_bound(n, k) * 100:.1f}%",
    ]


def main(
    args: argparse.Namespace,
    *,
    make_scorer: Callable[[str, np.ndarray], Scorer] = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    try:
        lines = _report(args, make_scorer, cuda_available)
    except _Refused as exc:
        print(f"refused: {exc}")
        return 1
    except (_CannotRun, ConfigError, LabelsError, FrameReadError) as exc:
        print(f"line-alarms: {exc}")
        return 2
    print("\n".join(lines))
    return 0
