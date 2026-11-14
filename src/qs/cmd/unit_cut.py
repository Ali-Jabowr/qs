"""`qs unit-cut`: the cut-out of the unit, drawn on the crop of every full pass (#174).

    qs unit-cut --lane lane.yml --unit unit.yml --labels L.json --footage DIR --split split.yml
                --block train|calibrate --out DIR [--limit 40]

For each full pass (`full`, `full_white`, `full_pink`) of a clip the split puts in the block, the
crops of frames f-2 ... f+2 are cut and the one whose unit is nearest the crop centre is drawn
with its outline and written as `<video>-<frame>.png`. For the eye check of `unit.yml`. The test
block is not offered, and only the block's clips are hashed and read.

Exit codes: 0 done · 1 refused (no full pass) · 2 could not run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from qs.cmd.train import positive_int
from qs.cmd.train_lane_b import _FULL, _evenly
from qs.errors import ConfigError, PlacementFault
from qs.labels import FrameReadError, LabelsError, VideoFrames, load_labels
from qs.lane.lanefile import Lane, crop, load_lane
from qs.lane.split import Split, file_sha256, load_split, verify
from qs.lane.unit import UnitSettings, best_cut, load_unit

_NEIGHBOURS = 2
_OUTLINE = (0, 255, 0)


def add_parser(commands: Any) -> None:
    """Register `unit-cut` on the `qs` subcommands."""
    parser = commands.add_parser(
        "unit-cut", help="cut the unit out of its cup and draw the outline for an eye check"
    )
    parser.add_argument("--lane", required=True, metavar="PATH", help="the lane file")
    parser.add_argument("--unit", required=True, metavar="PATH", help="the unit settings file")
    parser.add_argument("--labels", required=True, metavar="PATH",
                        help="the labels file (qs label)")
    parser.add_argument("--footage", required=True, metavar="DIR", help="the folder of clips")
    parser.add_argument("--split", required=True, metavar="PATH", help="the split file")
    parser.add_argument("--block", required=True, choices=("train", "calibrate"),
                        help="whose clips to draw (never test)")
    parser.add_argument("--out", required=True, metavar="DIR", help="where the pictures go")
    parser.add_argument("--limit", type=positive_int, default=40, metavar="N",
                        help="draw at most N full passes, evenly spaced (40)")


def _block_videos(split: Split, block: str, footage: Path) -> tuple[str, ...]:
    """The block's clips, after checking only their bytes against the split file."""
    videos = split.videos(block)
    verify(Split({video: split.clips[video] for video in videos}), footage, file_sha256)
    return videos


def _draw_pass(
    reader: VideoFrames, frame: int, lane: Lane, unit: UnitSettings, out: Path, video: str
) -> PlacementFault | None:
    """Cut the pass at `frame`, write its overlay; the fault if there was no cut."""
    last = len(reader.ids) - 1
    positions = range(max(0, frame - _NEIGHBOURS), min(last, frame + _NEIGHBOURS) + 1)
    crops = [crop(reader.read(position), lane) for position in positions]
    try:
        index, cut = best_cut(crops, unit)
    except PlacementFault as fault:
        return fault
    picture = crops[index].copy()
    cv2.drawContours(picture, [cut.contour], -1, _OUTLINE, 2)
    target = out / f"{video}-{frame}.png"
    if not cv2.imwrite(str(target), picture):
        raise OSError(f"cannot write {target}")
    return None


def _run(args: argparse.Namespace) -> int:
    try:
        lane = load_lane(Path(args.lane))
        unit = load_unit(Path(args.unit))
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
        videos = _block_videos(load_split(Path(args.split)), args.block, footage)
    except (LabelsError, ConfigError) as exc:
        print(exc)
        return 2
    found = [(video, cup.frame) for video in videos for cup in labels.cups_of(video)
             if cup.state in _FULL]
    if not found:
        print(f"refused: no full pass in the {args.block} clips")
        return 1
    chosen = _evenly(found, args.limit)
    out = Path(args.out)
    faults: Counter[str] = Counter()
    try:
        out.mkdir(parents=True, exist_ok=True)
        for video in dict.fromkeys(v for v, _ in chosen):
            reader = VideoFrames(footage / video)
            try:
                for _, frame in (c for c in chosen if c[0] == video):
                    fault = _draw_pass(reader, frame, lane, unit, out, video)
                    if fault is not None:
                        faults[fault.detail] += 1
            finally:
                reader.close()
    except FrameReadError as exc:
        print(f"footage: {exc}")
        return 2
    except (ConfigError, OSError) as exc:
        print(exc)
        return 2
    print(f"cut {len(chosen) - sum(faults.values())} of {len(chosen)}")
    for detail, count in faults.items():
        print(f"{detail}: {count}")
    return 0


def main(args: argparse.Namespace) -> int:
    """Draw the cut-outs; 0 done, 1 refused, 2 could not run."""
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("stopped by the operator")
        return 2
