"""`qs score-line`: line footage scored against hand-counted cups (#105).

    qs score-line --config C --footage DIR --labels POSITION=PATH [--labels ...] [--window 15]

Each labels file (from `qs label`) covers one camera position. Every video its cups name is
played from `--footage` through the line station's own path: the cup trigger, the line replay and
a real `Pipeline` with the config's inspector. The video plays on a clock that moves one frame per
turn of the loop, so a slow inspection never skips a frame and the same footage always scores the
same. Each record is paired with a labelled cup (`qs.eval.line_score`) and only numbers are
printed: no image is kept or written.

Exit codes: 0 scored · 2 refused (config, labels or footage).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qs.annunciator import NullAnnunciator
from qs.config import LoadedConfig, load_config
from qs.contracts import Label
from qs.errors import ConfigError
from qs.eval.line_score import (
    CupScore,
    Judged,
    combined,
    hand_line,
    header_lines,
    position_lines,
    score_cups,
)
from qs.eval.line_split import LineSplitError, video_frame
from qs.eval.scoring import MemorySink, utc_now
from qs.labels import Labels, LabelsError, load_labels
from qs.pipeline import Pipeline
from qs.sources.line_replay import video_replay_pair
from qs.sources.replay import ReplayError
from qs.station import StartupRefused, build_inspector


class _FrameClock:
    """The scoring clock: it moves only when the loop says a frame has passed."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def add_parser(commands: Any) -> None:
    """Register `score-line` on the `qs` subcommands."""
    parser = commands.add_parser("score-line", help="score line footage against hand-counted cups")
    parser.add_argument("--config", required=True, metavar="PATH", help="the line station config")
    parser.add_argument("--footage", required=True, metavar="DIR",
                        help="the folder holding the videos the labels name")
    parser.add_argument("--labels", required=True, action="append", metavar="POSITION=PATH",
                        help="a labels file and the camera position it was shot from; repeatable")
    parser.add_argument("--window", type=int, default=15, metavar="FRAMES",
                        help="how far, in frames, a record may be from its cup (default 15)")


def _judged(loaded: LoadedConfig, video: Path) -> list[Judged]:
    """Every record the line station makes on `video`, as the frame it judged and its verdict."""
    config = loaded.config
    clock = _FrameClock()
    trigger, source = video_replay_pair(config, video, clock=clock)
    inspector = build_inspector(loaded)
    inspector.load(config)
    sink = MemorySink()
    pipeline = Pipeline(
        config=config,
        config_hash=loaded.config_hash,
        run_id="score-line",
        source=source,
        inspector=inspector,
        annunciator=NullAnnunciator(),
        sink=sink,
        clock=clock,
        wall=utc_now,
    )
    step = 1 / source.fps
    pipeline.start()
    while not trigger.exhausted and not pipeline.stopped:
        event = trigger.poll()
        if event is not None:
            pipeline.handle(event)
        clock.now += step
    stopped_early = pipeline.stopped
    pipeline.stop()
    if stopped_early:
        # The cups after this point would count as never seen: refuse rather than score them.
        raise ReplayError(f"{video.name}: the station stopped before the video ended "
                          f"({pipeline.refusal})")
    return [Judged(source.frame_index(r.event), r.verdict.label) for r in sink.records]


def _position(item: str) -> tuple[str, Path]:
    name, sep, path = item.partition("=")
    if not sep or not name or not path:
        raise LabelsError(f"--labels {item!r}: expected POSITION=PATH")
    return name, Path(path)


def _labels(path: Path) -> Labels:
    if not path.is_file():
        raise LabelsError(f"labels file {path} not found")
    return load_labels(path, path.stem)


def _score(loaded: LoadedConfig, footage: Path, labels: Labels, window: int) -> list[str]:
    """One position's lines: its cups' counts, then the frames labelled as showing a hand."""
    cup_videos = {cup.video for cup in labels.cups}
    if not cup_videos:
        raise LabelsError(f"labels file {labels.path} counts no cup")
    hands = {video_frame(i) for i, frame in labels.frames.items() if frame.label == "hand"}
    scores: list[CupScore] = []
    on_hands: list[Label] = []
    # A video with hands but no counted cup is played for its hands only: its records are not
    # scored against cups, or every one would count as a record paired with no cup.
    for video in sorted(cup_videos | {video for video, _ in hands}):
        judged = _judged(loaded, footage / video)
        if video in cup_videos:
            scores.append(score_cups(labels.cups_of(video), judged, window))
        on_hands += [j.label for j in judged if (video, j.frame) in hands]
    return [*position_lines(combined(scores), window), hand_line(len(hands), on_hands)]


def main(args: argparse.Namespace) -> int:
    try:
        loaded = load_config(Path(args.config))
        positions = [_position(item) for item in args.labels]
        lines = header_lines(loaded.config_hash, args.window)
        for name, path in positions:
            lines += ["", f"## position {name}"]
            lines += _score(loaded, Path(args.footage), _labels(path), args.window)
    except (ConfigError, LabelsError, LineSplitError, ReplayError, StartupRefused) as exc:
        print(f"score-line: {exc}")
        return 2
    print("\n".join(lines))
    return 0
