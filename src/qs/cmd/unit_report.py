"""`qs unit-report`: the test block scored once, behind the thresholds file (#177).

    qs unit-report --thresholds unit-thresholds.yml --lane lane.yml --unit unit.yml
                   --labels L.json --footage DIR --split split.yml
                   [--defects defects.yml] [--out table.md] [--models models] [--device cuda|cpu]

It opens no test clip unless the thresholds file exists and the lane file, `unit.yml` and every
PatchCore bundle's `model.pt` still hash as the file records. Only the clips the split marks
`test` are scored, with `score_block`, once; the table of each state is printed (and written to
`--out`). A dirty tracked tree is refused, as in `qs compare`. Each scoring is appended to
`<thresholds>.scored.json`; from the second on the table's first line says `RESCORED (run N)`.
No image is written.

Exit codes: 0 scored · 2 refused (thresholds, labels, footage, defects or models).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from qs.cmd.compare import _prior_runs
from qs.cmd.train_lane_b import _FULL
from qs.cmd.unit_scores import _scorers
from qs.errors import ConfigError, InferenceError
from qs.eval.scoring import utc_now
from qs.inspector.appearance import real_cuda_available, real_scorer
from qs.labels import FrameReadError, Labels, LabelsError, load_labels
from qs.lane.defects import load_defects
from qs.lane.freeze import model_file_sha256, require_clean_tree, run_git
from qs.lane.lanefile import load_lane
from qs.lane.split import Split, file_sha256, load_split, verify
from qs.lane.unit import load_unit
from qs.lane.unit_limits import LimitsFile, from_score_rows, load_limits, render_report
from qs.lane.unit_scores import score_block


def add_parser(commands: Any) -> None:
    """Register `unit-report` on the `qs` subcommands."""
    parser = commands.add_parser(
        "unit-report", help="score the test clips once, behind the thresholds file"
    )
    parser.add_argument("--thresholds", required=True, metavar="PATH",
                        help="the file of `qs unit-thresholds`")
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--unit", required=True, metavar="PATH", help="the unit settings file")
    parser.add_argument("--labels", required=True, metavar="PATH", help="the labels file")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the folder of clips")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--defects", metavar="PATH", help="the defects file (none: no defect)")
    parser.add_argument("--out", metavar="PATH", help="also write the table here")
    parser.add_argument("--models", default="models", metavar="DIR", help="where bundles are")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="the torch device (cuda; never a silent fallback to cpu)")


def _check_hashes(limits: LimitsFile, lane: Path, unit: Path, models: Path) -> None:
    """Raise, naming the file, unless the lane file, unit file and bundles are as recorded."""
    for path, recorded in [(lane, limits.lane_sha256), (unit, limits.unit_sha256)]:
        try:
            actual = file_sha256(path)
        except OSError as error:
            raise ConfigError(f"{path} cannot be read: {error}") from error
        if actual != recorded:
            raise ConfigError(f"{path} changed since the thresholds: "
                              f"{actual[:8]} now, {recorded[:8]} recorded")
    for state, (model_id, recorded) in limits.patchcore.items():
        try:
            actual = model_file_sha256(models, model_id)
        except ConfigError as error:
            raise ConfigError(f"{state}: {error}") from error
        if actual != recorded:
            raise ConfigError(f"{models / model_id / 'model.pt'} changed since the thresholds: "
                              f"{actual[:8]} now, {recorded[:8]} recorded")


def _test_videos(split: Split, labels: Labels) -> list[str]:
    videos = [video for video in split.videos("test")
              if any(cup.state in _FULL for cup in labels.cups_of(video))]
    if not videos:
        raise ConfigError("no labelled full pass in any test clip of the split file")
    return videos


def _report(
    args: argparse.Namespace,
    git: Callable[[Sequence[str]], str],
    clock: Callable[[], datetime],
    score: Callable[..., list[Any]],
    make_scorer: Callable[..., Any],
    cuda_available: Callable[[], bool],
) -> None:
    require_clean_tree(git)
    thresholds = Path(args.thresholds)
    if not thresholds.is_file():
        raise ConfigError(f"{thresholds}: no thresholds file; run `qs unit-thresholds` first")
    limits = load_limits(thresholds)
    lane_path, unit_path = Path(args.lane), Path(args.unit)
    _check_hashes(limits, lane_path, unit_path, Path(args.models))
    lane, unit = load_lane(lane_path), load_unit(unit_path)
    footage = Path(args.footage)
    split = load_split(Path(args.split))
    verify(Split({video: split.clips[video] for video in split.videos("test")}), footage,
           file_sha256)
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise LabelsError(f"labels file {labels_path} not found")
    labels = load_labels(labels_path, labels_path.stem)
    videos = _test_videos(split, labels)
    defects = load_defects(Path(args.defects), labels, split) if args.defects else ()
    banks = argparse.Namespace(
        patchcore=[f"{state}={model_id}" for state, (model_id, _) in limits.patchcore.items()],
        device=args.device, models=args.models)
    scorers = _scorers(banks, make_scorer, cuda_available)

    rows = from_score_rows(score(labels, footage, videos, lane, unit, scorers))
    record = thresholds.with_name(thresholds.name + ".scored.json")
    earlier = _prior_runs(record)
    run_number = len(earlier) + 1
    plain = render_report(limits, rows, defects)
    stamped = render_report(limits, rows, defects, run_number if run_number > 1 else None)
    entry = {"scored_utc": clock().isoformat(),
             "labels_sha256": file_sha256(labels_path),
             "table_sha256": hashlib.sha256(plain.encode("utf-8")).hexdigest()}
    try:
        record.write_text(json.dumps({"runs": [*earlier, entry]}, indent=2) + "\n",
                          encoding="utf-8")
        if args.out:
            Path(args.out).write_text(stamped + "\n", encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot write the result: {error}") from error
    print(stamped)


def main(
    args: argparse.Namespace,
    *,
    git: Callable[[Sequence[str]], str] = run_git,
    clock: Callable[[], datetime] = utc_now,
    score: Callable[..., list[Any]] = score_block,
    make_scorer: Callable[..., Any] = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> int:
    """Score the test clips once; 0 scored, 2 refused."""
    try:
        _report(args, git, clock, score, make_scorer, cuda_available)
    except (ConfigError, LabelsError, FrameReadError, InferenceError, ValueError,
            RuntimeError) as error:
        print(f"unit-report: {error}")
        return 2
    return 0
