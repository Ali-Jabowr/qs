"""`qs clip-check VIDEO`: numbers about one recorded clip, caught at the line (#152).

    qs clip-check VIDEO [--sharp-min 30.0]

Reads the clip and its sidecar (`<same name>.json`, written by `qs record`) and writes nothing.
Prints the five numbers and one `WARN` line for each fault.

Exit codes: 0 clean · 1 at least one WARN · 2 could not run (no sidecar, unreadable clip).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qs.labels import FrameReadError, VideoFrames

SAMPLE_EVERY = 15
EDGE_FRAMES = 30
SWING_FRAMES = 60
FRAMES_MIN_SHARE = 0.99
SATURATED_MAX = 0.02
SHIFT_MAX_PX = 2.0
SATURATION_LEVEL = 250


class ClipCheckError(ValueError):
    """The clip or its sidecar could not be read; the check cannot run."""


@dataclass(frozen=True, slots=True)
class ClipStats:
    decoded: int
    sharpness: float
    saturated: float
    shift: float
    swing: float


def add_parser(commands: Any) -> None:
    """Register `clip-check` on the `qs` subcommands."""
    parser = commands.add_parser("clip-check", help="check one recorded clip for faults")
    parser.add_argument("video", metavar="VIDEO", help="the clip; its sidecar sits beside it")
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


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian of the grey picture."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def saturated_share(frame: np.ndarray) -> float:
    """Share of pixels with any channel at or above 250."""
    return float((frame.max(axis=2) >= SATURATION_LEVEL).mean())


def shift_px(first: np.ndarray, last: np.ndarray) -> float:
    """Length in pixels of the phase-correlation shift between two float32 grey pictures."""
    (dx, dy), _response = cv2.phaseCorrelate(first, last)
    return float(np.hypot(dx, dy))


def swing(brightness: list[float]) -> float:
    """Standard deviation over mean of the frame-mean brightness."""
    values = np.asarray(brightness, dtype=np.float64)
    return float(values.std() / values.mean()) if values.mean() > 0 else 0.0


def _grey32(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def measure(video: Path) -> ClipStats:
    """Decode the clip once, in order, and take every number.
    `ClipCheckError` if it cannot be read."""
    try:
        frames = VideoFrames(video)
    except FrameReadError as exc:
        raise ClipCheckError(str(exc)) from exc
    sharp: list[float] = []
    sat: list[float] = []
    brightness: list[float] = []
    head: list[np.ndarray] = []
    tail: deque[np.ndarray] = deque(maxlen=EDGE_FRAMES)
    decoded = 0
    try:
        for position in range(len(frames.ids)):
            try:
                frame = frames.read(position)
            except FrameReadError:
                break  # the rest is not decodable; the frame count shows it
            if decoded % SAMPLE_EVERY == 0:
                sharp.append(sharpness(frame))
                sat.append(saturated_share(frame))
            grey = _grey32(frame)
            if decoded < SWING_FRAMES:
                brightness.append(float(frame.mean()))
            if decoded < EDGE_FRAMES:
                head.append(grey)
            tail.append(grey)
            decoded += 1
    finally:
        frames.close()
    if decoded == 0:
        raise ClipCheckError(f"no frame of {video} could be decoded")
    return ClipStats(
        decoded=decoded,
        sharpness=float(np.median(sharp)),
        saturated=float(np.mean(sat)),
        shift=shift_px(np.mean(head, axis=0), np.mean(tail, axis=0)),
        swing=swing(brightness),
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


def report(stats: ClipStats, expected: int, sharp_min: float) -> list[str]:
    """The number lines, then one WARN line per fault."""
    lines = [
        f"frames {stats.decoded} of {expected}",
        f"sharpness {stats.sharpness:.1f}",
        f"saturated {stats.saturated:.2f}",
        f"shift {stats.shift:.1f}",
        f"swing {stats.swing:.3f}",
    ]
    if stats.decoded < FRAMES_MIN_SHARE * expected:
        lines.append(f"WARN frames {stats.decoded} of {expected}: frames were dropped")
    if stats.sharpness < sharp_min:
        lines.append(f"WARN sharpness {stats.sharpness:.1f} below {sharp_min}: blurred")
    if stats.saturated > SATURATED_MAX:
        lines.append(f"WARN saturated {stats.saturated:.2f} above {SATURATED_MAX}: glare")
    if stats.shift > SHIFT_MAX_PX:
        lines.append(f"WARN shift {stats.shift:.1f} px above {SHIFT_MAX_PX}: scene moved")
    return lines


def main(args: argparse.Namespace) -> int:
    video = Path(args.video)
    try:
        expected = expected_frames(_sidecar(video))
        lines = report(measure(video), expected, args.sharp_min)
    except ClipCheckError as exc:
        print(f"clip-check: {exc}")
        return 2
    print("\n".join(lines))
    return 1 if any(line.startswith("WARN") for line in lines) else 0
