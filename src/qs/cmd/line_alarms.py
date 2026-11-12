"""`qs line-alarms`: the line model's false alarms on held-out good frames (#110).

    qs line-alarms --config C --labels PATH --footage DIR --model MODEL_ID
                   [--models models] [--gap-s 3] [--far 5]

The frames of one camera position labelled `unit` are split by time exactly as `qs train` split
them (`qs.eval.line_split`, the same frames and `--gap-s`), so the model never saw the two
held-out blocks. A held-out frame the bundle lists among its training images is refused, and so is
a `--gap-s` or an `inspector.line.fps` other than the one the bundle records (#112), since either
changes the gap in frames; a bundle that records none, made before #112, gets a one-line WARNING
per missing record and is counted. Every frame the bundle was trained on is read again
from `--footage` and must hash to the bundle's SHA-256, or the run is refused: another video under
the same name could hold training cups at held-out ids, and re-encoded footage is refused the same
way; retrain on the footage used. A video renamed or missing under `--footage` cannot be opened,
so the run cannot run (exit 2) rather than being refused. The labels' training block must then
be exactly the bundle's training frames, or the run is refused: relabelled frames move the cuts,
and a cup beside a trained frame could enter a held-out block. Each held-out frame is scored as
the line station scores it: the cup wholly in the zone (`find_cup`), the window of the bundle's
size centred on the rim, the map masked to the cup's inside (`check_appearance`). The threshold
is chosen on the calibrate block at `--far` (`choose_threshold`), and the test block's frames
above it are the false alarms. Only numbers are printed: no image, config or record is written,
so the threshold reaches a station config by hand.

Exit codes: 0 counted · 1 refused (a held-out frame with no cup or trained on, an empty block, a
gap or fps other than the bundle's, a trained frame whose pixels differ, another training block)
· 2 could not run (config, labels, footage or a video missing from it, bundle, a model scoring
not a number).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from qs.cmd.train import DEFAULT_GAP_S, non_negative_float
from qs.config import LineConfig, StationConfig, load_config
from qs.errors import ConfigError, InferenceError
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
from qs.inspector.bundle import ImageRecord, pixels_sha256
from qs.inspector.cup import find_cup, no_cup_reason
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
    parser.add_argument("--far", type=int, choices=(1, 5, 10), default=5, metavar="PCT",
                        help="the false-alarm target the threshold is set at: 1, 5 or 10 (5)")


def _line_config(path: Path) -> tuple[StationConfig, LineConfig, float]:
    config = load_config(path).config
    line = config.inspector.line
    if config.inspector.type != "line" or line is None:
        raise _CannotRun(f"{path}: not a line config")
    return config, line, line.fps


@contextmanager
def _footage_frames(footage: Path) -> Iterator[Callable[[str], np.ndarray]]:
    """A reader of `<video>#<index>` frames under `footage`, each video opened once."""
    videos: dict[str, VideoFrames] = {}

    def read(frame_id: str) -> np.ndarray:
        video, index = video_frame(frame_id)
        if video not in videos:
            videos[video] = VideoFrames(footage / video)
        return videos[video].read(index)

    try:
        yield read
    finally:
        for frames in videos.values():
            frames.close()


def _check_trained_pixels(
    images: Sequence[ImageRecord], read: Callable[[str], np.ndarray]
) -> None:
    """Refuse footage whose frame under a trained id is not the frame the bundle hashed: the
    same id over other pixels would let training cups into the held-out blocks unseen."""
    for image in images:
        try:
            pixels = read(image.file)
        except LineSplitError as exc:
            raise _Refused(f"the bundle's training image {exc}") from None
        if pixels_sha256(pixels) != image.sha256:
            raise _Refused(f"{image.file}: its pixels differ from the frame the model was "
                           "trained on (SHA-256); re-encoded or renamed footage also does this "
                           "— retrain on the footage used")


def _scores(
    frame_ids: Sequence[str], read: Callable[[str], np.ndarray], config: StationConfig,
    line: LineConfig, model: LoadedModel,
) -> list[float]:
    """Each frame's appearance score, as the line station computes it."""
    scores: list[float] = []
    for frame_id in frame_ids:
        pixels = read(frame_id)
        cup = find_cup(pixels, line)
        if cup is None:
            raise _Refused(f"{frame_id}: {no_cup_reason(pixels, line)}")
        check, _, _ = check_appearance(pixels, cup.rim[0], cup.inside, model, config)
        if check.value is None:  # check_appearance always sets the score
            raise RuntimeError(f"{frame_id}: the appearance check carries no score")
        scores.append(check.value)
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
    recorded = model.bundle.line
    if recorded is None:  # a bundle from before #112: warn rather than refuse, so it stays usable
        print("WARNING: the bundle records no gap_s (made before #112), so --gap-s is not "
              "checked against training")
    elif recorded.gap_s != args.gap_s:
        raise _Refused(f"--gap-s {args.gap_s:g} is not the {recorded.gap_s:g} s the model was "
                       "split with; use the gap it was trained with")
    trained_fps = model.bundle.line_fps
    if trained_fps is None:  # likewise: a bundle from before #112 stays usable
        print("WARNING: the bundle records no line_fps (made before #112), so inspector.line.fps "
              "is not checked against training")
    elif trained_fps != fps:
        raise _Refused(f"inspector.line.fps {fps:g} is not the {trained_fps:g} fps the model was "
                       "split at, so the gap in frames would differ from training; use the fps "
                       "it was trained with")
    trained = {image.file for image in model.bundle.images}
    seen = [frame_id for frame_id in (*split.calibrate, *split.test) if frame_id in trained]
    if seen:
        raise _Refused(f"{seen[0]}: the model was trained on this held-out frame; use the "
                       "labels and --gap-s it was trained with")
    with _footage_frames(Path(args.footage)) as read:
        _check_trained_pixels(model.bundle.images, read)
        # The train block depends on the labels alone; if it moved, a cup beside a trained frame
        # may sit in a held-out block although no trained frame id is held out.
        if set(split.train) != trained:
            raise _Refused(f"{labels_path.name}: its training block is not the frames the model "
                           "was trained on, so the labels differ from training; retrain, or use "
                           "the labels it was trained with")
        calibrate = _scores(split.calibrate, read, config, line, model)
        test = _scores(split.test, read, config, line, model)
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
    except (_CannotRun, ConfigError, LabelsError, FrameReadError, InferenceError) as exc:
        print(f"line-alarms: {exc}")
        return 2
    print("\n".join(lines))
    return 0
