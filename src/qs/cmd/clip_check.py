"""`qs clip-check VIDEO`: numbers about one recorded clip, caught at the line (#152).

    qs clip-check VIDEO [--sharp-min 30.0] [--static X,Y,W,H]

Reads the clip and its sidecar (`<same name>.json`, written by `qs record`) and writes nothing.
Prints the numbers and one `WARN` line for each fault.

Two numbers are built to ignore cups passing through the picture, which real footage always has:
`shift` compares the per-pixel median of frames sampled over the first and the last 10 s (a cup in
view for under half of a window drops out of the median), and the frame count warns on gaps between
frame timestamps, not on the wall-clock total, which includes the recorder's start-up.

With `--static X,Y,W,H`, both medians are cut to that region (a place where nothing moves, such
as the channel wall) before the phase correlation. Without it a cup standing in view at both ends
of a clip can read as a shift, and the WARN line says so.

Exit codes: 0 clean · 1 at least one WARN · 2 could not run (no sidecar, unreadable clip).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SAMPLE_EVERY = 15
EDGE_SAMPLES = 30
EDGE_SECONDS = 10.0
SWING_FRAMES = 60
SATURATED_MAX = 0.02
SHIFT_MAX_PX = 2.0
SATURATION_LEVEL = 250
RATE_TOLERANCE = 0.05  # measured against the sidecar's fps
GAP_FACTOR = 1.5  # a step over 1.5 frame periods means frames are missing
GAP_MAX_S = 0.2
MISSING_MAX_SHARE = 0.001
UNREADABLE_TOLERANCE = 2  # a container's frame count is an estimate; ignore +-2 frames


class ClipCheckError(ValueError):
    """The clip or its sidecar could not be read; the check cannot run."""


@dataclass(frozen=True, slots=True)
class GapStats:
    missing: int
    longest_s: float


@dataclass(frozen=True, slots=True)
class ClipStats:
    decoded: int
    sharpness: float
    saturated: float
    shift: float
    swing: float
    gaps: GapStats
    unreadable_at: int | None
    rate_fps: float | None = None
    whole_picture: bool = True


Region = tuple[int, int, int, int]


def _region(text: str) -> Region:
    """`X,Y,W,H` as four non-negative integers with a positive width and height."""
    try:
        x, y, w, h = (int(part) for part in text.split(","))
    except ValueError as exc:
        raise ClipCheckError(f"--static {text!r} is not X,Y,W,H of whole numbers") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ClipCheckError(f"--static {text!r} must have X, Y >= 0 and W, H > 0")
    return x, y, w, h


def add_parser(commands: Any) -> None:
    """Register `clip-check` on the `qs` subcommands."""
    parser = commands.add_parser("clip-check", help="check one recorded clip for faults")
    parser.add_argument("video", metavar="VIDEO", help="the clip; its sidecar sits beside it")
    parser.add_argument("--static", default=None, metavar="X,Y,W,H",
                        help="a region where nothing moves; the shift is measured only there")
    parser.add_argument("--sharp-min", type=float, default=30.0, metavar="VARIANCE",
                        help="WARN below this median Laplacian variance (30.0, a starting value)")


def expected_frames(sidecar: dict[str, Any]) -> int:
    """`round((stopped_utc - started_utc) * fps)` from a sidecar."""
    try:
        started = datetime.fromisoformat(sidecar["started_utc"])
        stopped = datetime.fromisoformat(sidecar["stopped_utc"])
        fps = float(sidecar["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClipCheckError(
            f"the sidecar lacks a usable started_utc, stopped_utc or fps: {exc!r}"
        ) from exc
    return round((stopped - started).total_seconds() * fps)


def frame_gaps(timestamps_ms: Sequence[float]) -> GapStats:
    """Frames missing where a step exceeds 1.5 frame periods (the median step), and the longest
    step in seconds. `ClipCheckError` if the timestamps do not advance."""
    if len(timestamps_ms) < 2:
        return GapStats(missing=0, longest_s=0.0)
    steps = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
    period = float(np.median(steps))
    if period <= 0:
        raise ClipCheckError("the video's frame timestamps do not advance")
    missing = sum(round(step / period) - 1 for step in steps if step > GAP_FACTOR * period)
    return GapStats(missing=missing, longest_s=float(steps.max()) / 1000.0)


def frame_rate(timestamps_ms: Sequence[float]) -> float:
    """Frames per second from the median step between timestamps. `ClipCheckError` if they do
    not advance."""
    steps = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
    if steps.size == 0 or float(np.median(steps)) <= 0:
        raise ClipCheckError("the video's frame timestamps do not advance")
    return 1000.0 / float(np.median(steps))


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian of the grey picture."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def saturated_share(frame: np.ndarray) -> float:
    """Share of pixels with any channel at or above 250."""
    return float((frame.max(axis=2) >= SATURATION_LEVEL).mean())


def _median_grey(frames: Sequence[np.ndarray], region: Region | None) -> np.ndarray:
    greys = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) for frame in frames]
    if region is not None:
        x, y, w, h = region
        height, width = greys[0].shape
        if x + w > width or y + h > height:
            raise ClipCheckError(
                f"the static region {x},{y},{w},{h} does not fit the {width}x{height} picture"
            )
        greys = [grey[y : y + h, x : x + w] for grey in greys]
    return np.median(greys, axis=0).astype(np.float32)


def shift_px(
    head: Sequence[np.ndarray], tail: Sequence[np.ndarray], region: Region | None = None
) -> float:
    """Length in pixels of the phase-correlation shift between the per-pixel median grey pictures
    of two sets of frames, cut to `region` (X, Y, W, H) when given."""
    (dx, dy), _response = cv2.phaseCorrelate(
        _median_grey(head, region), _median_grey(tail, region)
    )
    return float(np.hypot(dx, dy))


def swing(brightness: list[float]) -> float:
    """Standard deviation over mean of the frame-mean brightness."""
    values = np.asarray(brightness, dtype=np.float64)
    return float(values.std() / values.mean()) if values.mean() > 0 else 0.0


def measure(video: Path, fps: float, region: Region | None = None) -> ClipStats:
    """Decode the clip once, in order, and take every number.
    `ClipCheckError` if it cannot be read."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ClipCheckError(f"cannot open the video {video}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ClipCheckError(f"the video {video} reports no frames")
        window = max(1, min(round(EDGE_SECONDS * fps), count // 3))
        every = max(1, window // EDGE_SAMPLES)
        sharp: list[float] = []
        sat: list[float] = []
        brightness: list[float] = []
        stamps: list[float] = []
        head: list[np.ndarray] = []
        tail: deque[np.ndarray] = deque(maxlen=EDGE_SAMPLES)
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
            if decoded % SAMPLE_EVERY == 0:
                sharp.append(sharpness(frame))
                sat.append(saturated_share(frame))
            if decoded < SWING_FRAMES:
                brightness.append(float(frame.mean()))
            if decoded % every == 0:
                if decoded < window and len(head) < EDGE_SAMPLES:
                    head.append(frame)
                tail.append(frame)
            decoded += 1
    finally:
        capture.release()
    if decoded == 0:
        raise ClipCheckError(f"no frame of {video} could be decoded")
    return ClipStats(
        decoded=decoded,
        sharpness=float(np.median(sharp)),
        saturated=float(np.mean(sat)),
        shift=shift_px(head, list(tail), region),
        swing=swing(brightness),
        gaps=frame_gaps(stamps),
        unreadable_at=decoded if decoded < count - UNREADABLE_TOLERANCE else None,
        rate_fps=frame_rate(stamps),
        whole_picture=region is None,
    )


def _sidecar(video: Path) -> dict[str, Any]:
    path = video.with_suffix(".json")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ClipCheckError(f"the sidecar {path} is missing") from exc
    except OSError as exc:
        raise ClipCheckError(f"cannot read the sidecar {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClipCheckError(f"the sidecar {path} is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ClipCheckError(f"the sidecar {path} is not a JSON object")
    return data


def report(
    stats: ClipStats, expected: int, sharp_min: float, drifted: bool = False,
    fps: float | None = None,
) -> list[str]:
    """The number lines, then one WARN line per fault."""
    lines = [
        f"frames {stats.decoded} of {expected}",
        f"gaps {stats.gaps.missing} missing, longest {stats.gaps.longest_s:.3f} s",
        f"sharpness {stats.sharpness:.1f}",
        f"saturated {stats.saturated:.3f}",
        f"shift {stats.shift:.1f}",
        f"swing {stats.swing:.3f}",
    ]
    if stats.rate_fps is not None:
        lines.insert(2, f"rate {stats.rate_fps:.1f} fps")
    total = stats.decoded + stats.gaps.missing
    if stats.gaps.longest_s > GAP_MAX_S or stats.gaps.missing > MISSING_MAX_SHARE * total:
        lines.append(
            f"WARN gaps {stats.gaps.missing} frames missing, longest {stats.gaps.longest_s:.3f} s"
        )
    if (
        fps is not None
        and stats.rate_fps is not None
        and abs(stats.rate_fps - fps) > RATE_TOLERANCE * fps
    ):
        lines.append(f"WARN rate {stats.rate_fps:.1f} fps, sidecar says {fps:g}")
    if stats.unreadable_at is not None:
        lines.append(f"WARN unreadable at frame {stats.unreadable_at}")
    if stats.sharpness < sharp_min:
        lines.append(f"WARN sharpness {stats.sharpness:.1f} below {sharp_min}: blurred")
    if stats.saturated > SATURATED_MAX:
        lines.append(f"WARN saturated {stats.saturated:.3f} above {SATURATED_MAX}: glare")
    if stats.shift > SHIFT_MAX_PX:
        note = (
            " (whole picture: a cup standing in view can cause this; use --static)"
            if stats.whole_picture
            else ""
        )
        lines.append(f"WARN shift {stats.shift:.1f} px above {SHIFT_MAX_PX}: scene moved{note}")
    if drifted:
        lines.append("WARN drifted: a camera setting changed during the recording")
    return lines


def main(args: argparse.Namespace) -> int:
    video = Path(args.video)
    try:
        sidecar = _sidecar(video)
        expected = expected_frames(sidecar)
        fps = float(sidecar["fps"])
        region = _region(args.static) if args.static is not None else None
        stats = measure(video, fps, region)
        lines = report(stats, expected, args.sharp_min, sidecar.get("drifted") is True, fps)
    except ClipCheckError as exc:
        print(f"clip-check: {exc}")
        return 2
    print("\n".join(lines))
    return 1 if any(line.startswith("WARN") for line in lines) else 0
