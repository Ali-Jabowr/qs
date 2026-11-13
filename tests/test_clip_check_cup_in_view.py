"""`qs clip-check`: a cup in view for part of the first and last seconds is not a mount shift.

A low-contrast background (so a bright disc standing in for a cup dominates a mean picture;
sharpness is switched off here) and the disc.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.__main__ import run

GONE = -100  # a disc centre off the picture


def scene(disc_x: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    grey = cv2.GaussianBlur(rng.integers(90, 110, (240, 320), dtype=np.uint8), (5, 5), 0)
    frame = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    cv2.circle(frame, (disc_x, 120), 25, (200, 200, 200), -1)
    return frame


def draw(tmp_path: Path, positions: list[int]) -> Path:
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"MJPG"), 30, (320, 240))
    assert writer.isOpened()
    for x in positions:
        writer.write(scene(x))
    writer.release()
    (tmp_path / "clip.json").write_text(
        json.dumps({
            "version": 1, "fps": 30, "drifted": False,
            "started_utc": "2026-11-16T10:00:00.000Z", "stopped_utc": "2026-11-16T10:00:03.000Z",
        }),
        encoding="utf-8",
    )
    return video


def test_cups_in_view_for_part_of_the_edges_do_not_warn(tmp_path, capsys):
    first = [100] * 12 + [GONE] * 18
    last = [220] * 12 + [GONE] * 18
    video = draw(tmp_path, first + [GONE] * 30 + last)
    code = run(["clip-check", str(video), "--sharp-min", "0"])
    out = capsys.readouterr().out
    shift_line = next(line for line in out.splitlines() if line.startswith("shift "))
    assert float(shift_line.split()[1]) < 2.0
    assert "WARN" not in out
    assert code == 0


@pytest.mark.parametrize("roll_px", [6])
def test_the_whole_scene_moving_still_warns_with_cups_in_view(tmp_path, capsys, roll_px):
    first = [100] * 12 + [GONE] * 18
    last = [220] * 12 + [GONE] * 18
    video = draw(tmp_path, first + [GONE] * 30 + last)
    frames = cv2.VideoCapture(str(video))
    shifted = []
    for index in range(90):
        _, frame = frames.read()
        shifted.append(frame if index < 60 else np.roll(frame, roll_px, axis=1))
    frames.release()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"MJPG"), 30, (320, 240))
    for frame in shifted:
        writer.write(frame)
    writer.release()
    code = run(["clip-check", str(video), "--sharp-min", "0"])
    out = capsys.readouterr().out
    value = float(out.split("WARN shift ")[1].split()[0])
    assert code == 1
    assert 5.0 <= value <= 7.0
