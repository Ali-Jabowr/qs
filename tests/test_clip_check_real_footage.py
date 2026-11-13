"""`qs clip-check` on what real footage does: moving cups, container timestamps, recorder sidecars.

Added after review of the first version; the sidecar here is written the way `qs record` writes it
(`Z` suffix, integer fps, milliseconds).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.__main__ import run
from qs.cmd.clip_check import ClipStats, GapStats, frame_gaps, report

CLEAN = GapStats(missing=0, longest_s=0.034)


def streaked() -> np.ndarray:
    """A static background with strong structure, so only a real mount shift moves it."""
    rng = np.random.default_rng(11)
    grey = cv2.GaussianBlur(rng.integers(90, 110, (240, 320), dtype=np.uint8), (5, 5), 0)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def disc_frames(positions: list[int]) -> list[np.ndarray]:
    frames = []
    for x in positions:
        frame = streaked()
        cv2.circle(frame, (x, 120), 25, (200, 200, 200), -1)
        frames.append(frame)
    return frames


def draw(tmp_path: Path, frames: list[np.ndarray], sidecar: dict | None = None) -> Path:
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"MJPG"), 30, (320, 240))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    body = {
        "version": 1,
        "fps": 30,
        "started_utc": "2026-11-16T10:00:00.000Z",
        "stopped_utc": "2026-11-16T10:00:03.000Z",
        "drifted": False,
    }
    body.update(sidecar or {})
    (tmp_path / "clip.json").write_text(json.dumps(body), encoding="utf-8")
    return video


def check(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = run(["clip-check", *argv])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_a_cup_moving_across_a_still_scene_is_not_a_shift(tmp_path, capsys):
    gone = -100  # off the picture
    first = [100] * 12 + [gone] * 18  # a cup in view for 12 of the first 30 frames
    middle = [gone] * 30
    last = [220] * 12 + [gone] * 18  # another cup, elsewhere, in the last 30
    video = draw(tmp_path, disc_frames(first + middle + last))
    code, out = check([str(video)], capsys)
    shift_line = next(line for line in out.splitlines() if line.startswith("shift "))
    assert float(shift_line.split()[1]) < 2.0
    assert "WARN" not in out
    assert code == 0


@pytest.mark.parametrize(
    ("stamps_ms", "missing", "longest_s"),
    [
        ([0.0, 33.0, 66.0, 99.0, 132.0], 0, 0.033),
        ([0.0, 33.0, 66.0, 99.0, 330.0, 363.0], 6, 0.231),
        ([0.0, 33.0, 99.0, 132.0], 1, 0.066),
    ],
)
def test_frame_gaps_count_the_frames_missing_between_timestamps(stamps_ms, missing, longest_s):
    stats = frame_gaps(stamps_ms)
    assert stats.missing == missing
    assert stats.longest_s == pytest.approx(longest_s, abs=0.001)


def stats(decoded: int, gaps: GapStats, unreadable_at: int | None = None) -> ClipStats:
    return ClipStats(
        decoded=decoded, sharpness=100.0, saturated=0.0, shift=0.0, swing=0.0,
        gaps=gaps, unreadable_at=unreadable_at,
    )


@pytest.mark.parametrize(
    ("decoded", "gaps", "warns"),
    [
        (1000, GapStats(missing=0, longest_s=0.034), False),
        (1000, GapStats(missing=1, longest_s=0.25), True),
        (100000, GapStats(missing=3, longest_s=0.1), False),
        (50000, GapStats(missing=100, longest_s=0.1), True),
    ],
)
def test_gaps_warn_on_a_long_gap_or_a_tenth_of_a_percent_missing(decoded, gaps, warns):
    lines = report(stats(decoded, gaps), expected=decoded, sharp_min=30.0)
    assert any(line.startswith("WARN gaps") for line in lines) is warns


def test_fewer_frames_than_the_wall_clock_expects_is_only_a_number():
    lines = report(stats(80, CLEAN), expected=90, sharp_min=30.0)
    assert "frames 80 of 90" in lines
    assert not any(line.startswith("WARN") for line in lines)


def test_an_unreadable_frame_is_warned():
    lines = report(stats(40, CLEAN, unreadable_at=40), expected=90, sharp_min=30.0)
    assert "WARN unreadable at frame 40" in lines


def test_saturated_share_prints_three_decimals():
    pixels = ClipStats(
        decoded=90, sharpness=100.0, saturated=0.0204, shift=0.0, swing=0.0,
        gaps=CLEAN, unreadable_at=None,
    )
    lines = report(pixels, expected=90, sharp_min=30.0)
    assert "saturated 0.020" in lines
    assert any(line.startswith("WARN saturated 0.020 above") for line in lines)


def test_a_drifted_clip_is_warned(tmp_path, capsys):
    video = draw(tmp_path, disc_frames([100] * 90), {"drifted": True})
    code, out = check([str(video)], capsys)
    assert code == 1
    assert "WARN drifted" in out


def test_a_video_that_cannot_be_opened_is_refused(tmp_path, capsys):
    video = draw(tmp_path, disc_frames([100] * 90))
    video.write_bytes(b"not a video")
    code, out = check([str(video)], capsys)
    assert code == 2
    assert out.startswith("clip-check:")
    assert "clip.avi" in out


def test_a_sidecar_that_is_not_json_is_refused(tmp_path, capsys):
    video = draw(tmp_path, disc_frames([100] * 90))
    (tmp_path / "clip.json").write_text("{nope", encoding="utf-8")
    code, out = check([str(video)], capsys)
    assert code == 2
    assert "clip.json" in out


def test_sharp_min_moves_the_sharpness_limit(tmp_path, capsys):
    blurred = [cv2.GaussianBlur(f, (31, 31), 0) for f in disc_frames([100] * 90)]
    video = draw(tmp_path, blurred)
    code, out = check([str(video), "--sharp-min", "0"], capsys)
    assert code == 0
    assert "WARN" not in out


def test_a_clip_cut_short_warns_where_decoding_stopped(tmp_path, capsys):
    video = draw(tmp_path, disc_frames([100] * 90))
    video.write_bytes(video.read_bytes()[: video.stat().st_size // 2])
    code, out = check([str(video)], capsys)
    assert code == 1
    assert "WARN unreadable at frame" in out
