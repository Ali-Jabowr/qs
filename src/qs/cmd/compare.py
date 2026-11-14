"""`qs compare`: the test block scored once, behind the freeze (#160).

    qs compare --freeze freeze.yml --labels L.json --footage DIR [--a-config A.yml] [--window 15]
               [--out table.md] [--models models] [--device cuda|cpu]
               [--threshold PATH] [--lane PATH] [--split PATH]

It opens no test clip unless the freeze file exists, the hash of every file in it still matches
(the lane bundle's `model.pt`, the threshold, lane and split files, which are looked for beside
the freeze file unless a flag says where) and HEAD is the commit it records. Only clips the split
marks `test` are scored; a labelled pass in any other clip is left alone.

Method C is the lane classifier (`qs.lane.passes`) at the frozen threshold. Method A, only with
`--a-config`, is the line station's own records scored by `qs.eval.line_score`. Method B, only if
the freeze names it, is PatchCore at the labelled frames, judge only. One Markdown table is
printed (and written to `--out`); no image is written. Each scoring is appended to
`<freeze>.scored.json`; from the second on, the table's first line says `RESCORED (run N)`.

Exit codes: 0 scored · 2 refused (freeze, labels, footage or models).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from qs.contracts import Label
from qs.errors import ConfigError, InferenceError
from qs.eval.line_score import CupScore, Judged, Tally, _tally, combined, score_cups
from qs.eval.scoring import utc_now
from qs.labels import Cup, CupState, FrameReadError, Labels, LabelsError, VideoFrames, load_labels
from qs.lane.freeze import Freeze, check_freeze, load_freeze, run_git, threshold_of
from qs.lane.lanefile import Lane, load_lane
from qs.lane.passes import pair_runs, runs_from_probs, verdict
from qs.lane.split import Split, file_sha256, load_split, verify
from qs.lane.table import MethodResult, render_table

ProbsOf = Callable[[Path], Sequence[Sequence[float]]]
JudgeA = Callable[[Path, Path], tuple[list[Judged], list[int]]]
ScoreB = Callable[[Path, Sequence[Cup]], list[float]]


def add_parser(commands: Any) -> None:
    """Register `compare` on the `qs` subcommands."""
    parser = commands.add_parser("compare", help="score the test clips once, behind the freeze")
    parser.add_argument("--freeze", required=True, metavar="PATH", help="the freeze file")
    parser.add_argument("--labels", required=True, metavar="PATH", help="the labels file")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the folder of clips")
    parser.add_argument("--a-config", metavar="PATH",
                        help="the line station config: adds method A")
    parser.add_argument("--window", type=int, default=15, metavar="FRAMES",
                        help="how far, in frames, a run may be from its pass (15)")
    parser.add_argument("--out", metavar="PATH", help="also write the table here")
    parser.add_argument("--models", default="models", metavar="DIR", help="where bundles are")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="the torch device (cuda; never a silent fallback to cpu)")
    parser.add_argument("--threshold", metavar="PATH",
                        help="the threshold file (default threshold.yml beside the freeze file)")
    parser.add_argument("--lane", metavar="PATH",
                        help="the lane file (default lane.yml beside the freeze file)")
    parser.add_argument("--split", metavar="PATH",
                        help="the split file (default split.yml beside the freeze file)")


def _device(args: argparse.Namespace) -> str:
    from qs.inspector.appearance import real_cuda_available
    from qs.inspector.model import device_for

    return device_for("torch-cuda" if args.device == "cuda" else "torch-cpu",
                      cuda_available=real_cuda_available())


def _default_probs(args: argparse.Namespace, lane: Lane, frozen: Freeze) -> ProbsOf:
    from qs.lane.model import load_lane_bundle

    # `frame_probs` is ticket #157's; looked up by name so this module imports without it.
    frame_probs = getattr(importlib.import_module("qs.lane.passes"), "frame_probs", None)
    if frame_probs is None:
        raise ConfigError("qs.lane.passes.frame_probs is missing from this build (#157)")
    device = _device(args)
    model = load_lane_bundle(Path(args.models), frozen.model_id).model

    def probs(video: Path) -> Sequence[Sequence[float]]:
        return np.asarray(frame_probs(video, lane, model, device)).tolist()  # type: ignore[no-any-return]

    return probs


def _default_judge_a(config: Path, video: Path) -> tuple[list[Judged], list[int]]:
    from qs.cmd.score_line import _judged
    from qs.config import load_config

    return _judged(load_config(config), video)


def _default_score_b(
    args: argparse.Namespace, lane: Lane, patchcore_id: str
) -> ScoreB:
    from qs.cmd.train import _real_patchcore
    from qs.cmd.train_lane_b import lane_b_score
    from qs.inspector.bundle import load_bundle

    bundle = load_bundle(Path(args.models), patchcore_id)
    scorer = _real_patchcore(_device(args)).scorer(bundle.bank)

    def scores(video: Path, cups: Sequence[Cup]) -> list[float]:
        reader = VideoFrames(video)
        try:
            last = len(reader.ids) - 1
            return [
                statistics.median(
                    lane_b_score(reader.read(i), lane, scorer, bundle.input_px)
                    for i in sorted({max(0, min(cup.frame + d, last)) for d in (-1, 0, 1)})
                )
                for cup in cups
            ]
        finally:
            reader.close()

    return scores


def _score_of(pairs: Sequence[tuple[Cup, Judged | None]]) -> CupScore:
    """The counts per kind of labelled pass; a pass given no judgement counts as unseen."""
    def of(*states: CupState) -> Tally:
        return _tally([pair for pair in pairs if pair[0].state in states])

    full = tuple(s for s in CupState if s not in (CupState.EMPTY, CupState.GAP))
    return CupScore(empty=of(CupState.EMPTY), full=of(*full), white=Tally(), pink=Tally(),
                    unpaired=0, offsets=(), gap=of(CupState.GAP))


def _method_c(
    cups: dict[str, tuple[Cup, ...]], footage: Path, probs_of: ProbsOf, threshold: float,
    window: int,
) -> CupScore:
    scores: list[CupScore] = []
    for video, video_cups in cups.items():
        runs = runs_from_probs(probs_of(footage / video))
        pairs, _ = pair_runs(video_cups, runs, window)
        scores.append(_score_of([
            (cup, None if run is None else Judged(run.first, verdict(run, threshold)))
            for cup, run in pairs
        ]))
    return combined(scores)


def _method_a(
    cups: dict[str, tuple[Cup, ...]], footage: Path, config: Path, judge_a: JudgeA, window: int
) -> CupScore:
    scores = []
    for video, video_cups in cups.items():
        judged, _ = judge_a(config, footage / video)
        scores.append(score_cups(video_cups, judged, window))
    return combined(scores)


def _method_b(
    cups: dict[str, tuple[Cup, ...]], footage: Path, score_b: ScoreB, threshold: float
) -> CupScore:
    scores: list[CupScore] = []
    for video, video_cups in cups.items():
        judged_cups = [cup for cup in video_cups if cup.state != CupState.GAP]
        values = score_b(footage / video, judged_cups)
        if len(values) != len(judged_cups):
            raise InferenceError(f"{video}: {len(values)} scores for {len(judged_cups)} passes")
        scores.append(_score_of([
            (cup, Judged(cup.frame, Label.DEFECT if value > threshold else Label.PASS))
            for cup, value in zip(judged_cups, values, strict=True)
        ]))
    return combined(scores)


def _prior_runs(record: Path) -> list[dict[str, str]]:
    if not record.exists():
        return []
    try:
        runs = json.loads(record.read_text(encoding="utf-8"))["runs"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"{record}: cannot read the earlier scorings: {exc!r}") from exc
    if not isinstance(runs, list):
        raise ConfigError(f"{record}: 'runs' must be a list")
    return runs


def _test_cups(split: Split, labels: Labels) -> dict[str, tuple[Cup, ...]]:
    cups = {video: labels.cups_of(video) for video in split.videos("test")}
    cups = {video: found for video, found in cups.items() if found}
    if not cups:
        raise ConfigError("no labelled pass in any test clip of the split file")
    return cups


def _compare(
    args: argparse.Namespace,
    git: Callable[[Sequence[str]], str],
    clock: Callable[[], datetime],
    probs_of: ProbsOf | None,
    judge_a: JudgeA | None,
    score_b: ScoreB | None,
) -> None:
    freeze_path = Path(args.freeze)
    if not freeze_path.is_file():
        raise ConfigError(f"{freeze_path}: no freeze file; run `qs freeze` first")
    frozen = load_freeze(freeze_path)
    beside = freeze_path.parent
    threshold_path = Path(args.threshold) if args.threshold else beside / "threshold.yml"
    lane_path = Path(args.lane) if args.lane else beside / "lane.yml"
    split_path = Path(args.split) if args.split else beside / "split.yml"
    check_freeze(frozen, models=Path(args.models), threshold=threshold_path, lane=lane_path,
                 split=split_path, head=git(["rev-parse", "HEAD"]).strip())
    footage = Path(args.footage)
    split = load_split(split_path)
    verify(Split({video: split.clips[video] for video in split.videos("test")}), footage,
           file_sha256)
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise LabelsError(f"labels file {labels_path} not found")
    cups = _test_cups(split, load_labels(labels_path, labels_path.stem))
    threshold_model, threshold = threshold_of(threshold_path)
    if threshold_model != frozen.model_id:
        raise ConfigError(f"{threshold_path} was made for {threshold_model}, "
                          f"the freeze is of {frozen.model_id}")
    lane = load_lane(lane_path)
    record = freeze_path.with_name(freeze_path.name + ".scored.json")
    earlier = _prior_runs(record)

    results = [MethodResult("C", _method_c(
        cups, footage, probs_of or _default_probs(args, lane, frozen), threshold, args.window))]
    if args.a_config:
        results.append(MethodResult("A", _method_a(
            cups, footage, Path(args.a_config), judge_a or _default_judge_a, args.window)))
    if frozen.patchcore is not None:
        patchcore_id, patchcore_threshold = frozen.patchcore
        results.append(MethodResult("B (judge only)", _method_b(
            cups, footage, score_b or _default_score_b(args, lane, patchcore_id),
            patchcore_threshold), gap_scored=False))

    run_number = len(earlier) + 1
    plain = render_table(results)
    stamped = render_table(results, run_number if run_number > 1 else None)
    entry = {"scored_utc": clock().isoformat(),
             "table_sha256": hashlib.sha256(plain.encode("utf-8")).hexdigest()}
    try:
        record.write_text(json.dumps({"runs": [*earlier, entry]}, indent=2) + "\n",
                          encoding="utf-8")
        if args.out:
            Path(args.out).write_text(stamped + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot write the result: {exc}") from exc
    print(stamped)


def main(
    args: argparse.Namespace,
    *,
    git: Callable[[Sequence[str]], str] = run_git,
    clock: Callable[[], datetime] = utc_now,
    probs_of: ProbsOf | None = None,
    judge_a: JudgeA | None = None,
    score_b: ScoreB | None = None,
) -> int:
    """Score the test clips once; 0 scored, 2 refused."""
    try:
        _compare(args, git, clock, probs_of, judge_a, score_b)
    except (ConfigError, LabelsError, FrameReadError, InferenceError) as exc:
        print(f"compare: {exc}")
        return 2
    return 0
