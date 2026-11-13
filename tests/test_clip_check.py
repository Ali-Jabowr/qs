"""`qs clip-check`: a clip drawn in tmp_path, then the numbers and WARN lines it earns."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.__main__ import run

START = "2026-11-16T10:00:00+00:00"
STOP = "2026-11-16T10:00:03+00:00"


def textured() -> np.ndarray:
    """A 320x240 picture of mild noise, bright enough to see, never near saturation."""
    rng = np.random.default_rng(7)
    grey = rng.integers(40, 160, (240, 320), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(grey, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def draw_clip(tmp_path: Path, frames: list[np.ndarray], sidecar: bool = True) -> Path:
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (320, 240))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    if sidecar:
        (tmp_path / "clip.json").write_text(
            json.dumps({"version": 1, "fps": 30.0, "started_utc": START, "stopped_utc": STOP}),
            encoding="utf-8",
        )
    return video


def check(video: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = run(["clip-check", str(video)])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_clean_clip_has_no_warning(tmp_path, capsys):
    video = draw_clip(tmp_path, [textured()] * 90)
    code, out = check(video, capsys)
    assert code == 0
    assert "frames 90 of 90" in out
    assert "WARN" not in out


def test_missing_frames_warn(tmp_path, capsys):
    video = draw_clip(tmp_path, [textured()] * 80)
    code, out = check(video, capsys)
    assert code == 1
    assert "WARN frames 80 of 90" in out


def test_blurred_clip_warns_on_sharpness(tmp_path, capsys):
    blurred = cv2.GaussianBlur(textured(), (31, 31), 0)
    video = draw_clip(tmp_path, [blurred] * 90)
    code, out = check(video, capsys)
    assert code == 1
    assert "WARN sharpness" in out


def test_white_patch_warns_on_saturation(tmp_path, capsys):
    frame = textured()
    frame[0:80, 0:96] = 255
    video = draw_clip(tmp_path, [frame] * 90)
    code, out = check(video, capsys)
    assert code == 1
    assert "WARN saturated 0.10" in out


def test_shifted_scene_warns_with_its_length(tmp_path, capsys):
    still = textured()
    moved = np.roll(still, 6, axis=1)
    video = draw_clip(tmp_path, [still] * 60 + [moved] * 30)
    code, out = check(video, capsys)
    value = float(out.split("WARN shift ")[1].split()[0])
    assert code == 1
    assert 5.0 <= value <= 7.0


def test_flicker_is_printed_never_warned(tmp_path, capsys):
    dim = textured()
    bright = np.clip(dim.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    video = draw_clip(tmp_path, [dim, bright] * 45)
    code, out = check(video, capsys)
    swing_line = next(line for line in out.splitlines() if line.startswith("swing "))
    assert code == 0
    assert float(swing_line.split()[1]) > 0.05


def test_missing_sidecar_is_refused(tmp_path, capsys):
    video = draw_clip(tmp_path, [textured()] * 90, sidecar=False)
    code, out = check(video, capsys)
    assert code == 2
    assert out.strip().count("\n") == 0
    assert "clip.json" in out
