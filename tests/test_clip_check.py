"""`qs clip-check`: clips drawn in tmp_path, then the numbers and WARN lines they earn.

The sidecar is written the way `qs record` writes it (`Z` suffix, integer fps, milliseconds).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gapped_clip import make_gapped_clip

from qs.__main__ import run
from qs.cmd.clip_check import ClipStats, GapStats, frame_gaps, frame_rate, report

CLEAN = GapStats(missing=0, longest_s=0.034)
GONE = -100  # a disc centre off the picture


def textured() -> np.ndarray:
    """A 320x240 picture of mild noise, bright enough to see, never near saturation."""
    rng = np.random.default_rng(7)
    grey = rng.integers(40, 160, (240, 320), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(grey, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def streaked() -> np.ndarray:
    """A low-contrast static background, so a bright disc (a cup) dominates a mean picture."""
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


def write_video(video: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"MJPG"), 30, (320, 240))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def draw(
    tmp_path: Path, frames: list[np.ndarray], sidecar: dict | None = None, with_sidecar: bool = True
) -> Path:
    video = tmp_path / "clip.avi"
    write_video(video, frames)
    if with_sidecar:
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


def test_clean_clip_has_no_warning(tmp_path, capsys):
    video = draw(tmp_path, [textured()] * 90)
    code, out = check([str(video)], capsys)
    assert code == 0
    assert "frames 90 of 90" in out
    assert "WARN" not in out


def test_fewer_frames_than_the_clock_expects_is_a_number_not_a_warning(tmp_path, capsys):
    video = draw(tmp_path, [textured()] * 80)
    code, out = check([str(video)], capsys)
    assert code == 0
    assert "frames 80 of 90" in out
    assert "WARN" not in out


def test_blurred_clip_warns_on_sharpness(tmp_path, capsys):
    blurred = cv2.GaussianBlur(textured(), (31, 31), 0)
    video = draw(tmp_path, [blurred] * 90)
    code, out = check([str(video)], capsys)
    assert code == 1
    assert "WARN sharpness" in out


def test_white_patch_warns_on_saturation(tmp_path, capsys):
    frame = textured()
    frame[0:80, 0:96] = 255
    video = draw(tmp_path, [frame] * 90)
    code, out = check([str(video)], capsys)
    assert code == 1
    assert "WARN saturated 0.100" in out


def test_shifted_scene_warns_with_its_length(tmp_path, capsys):
    still = textured()
    moved = np.roll(still, 6, axis=1)
    video = draw(tmp_path, [still] * 60 + [moved] * 30)
    code, out = check([str(video)], capsys)
    value = float(out.split("WARN shift ")[1].split()[0])
    assert code == 1
    assert 5.0 <= value <= 7.0


def test_cups_in_view_for_part_of_the_edges_are_not_a_shift(tmp_path, capsys):
    first = [100] * 12 + [GONE] * 18  # a cup in view for 12 of the first 30 frames
    last = [220] * 12 + [GONE] * 18  # another cup, elsewhere, in the last 30
    video = draw(tmp_path, disc_frames(first + [GONE] * 30 + last))
    code, out = check([str(video), "--sharp-min", "0"], capsys)
    shift_line = next(line for line in out.splitlines() if line.startswith("shift "))
    assert float(shift_line.split()[1]) < 2.0
    assert "WARN" not in out
    assert code == 0


def test_the_whole_scene_moving_still_warns_with_cups_in_view(tmp_path, capsys):
    first = [100] * 12 + [GONE] * 18
    last = [220] * 12 + [GONE] * 18
    frames = disc_frames(first + [GONE] * 30 + last)
    moved = frames[:60] + [np.roll(frame, 6, axis=1) for frame in frames[60:]]
    video = draw(tmp_path, moved)
    code, out = check([str(video), "--sharp-min", "0"], capsys)
    value = float(out.split("WARN shift ")[1].split()[0])
    assert code == 1
    assert 5.0 <= value <= 7.0


def test_a_chain_standing_still_in_both_windows_is_a_shift_without_a_static_region(
    tmp_path, capsys
):
    video = draw(tmp_path, disc_frames([100] * 30 + [GONE] * 30 + [220] * 30))
    code, out = check([str(video), "--sharp-min", "0"], capsys)
    assert code == 1
    assert "(whole picture: a cup standing in view can cause this; use --static)" in out


def test_a_static_region_away_from_a_standing_cup_gives_no_shift_warning(tmp_path, capsys):
    video = draw(tmp_path, disc_frames([100] * 30 + [GONE] * 30 + [220] * 30))
    code, out = check([str(video), "--sharp-min", "0", "--static", "0,160,320,80"], capsys)
    assert code == 0
    assert "WARN" not in out


def test_the_whole_scene_moving_still_warns_inside_the_static_region(tmp_path, capsys):
    frames = disc_frames([100] * 30 + [GONE] * 30 + [220] * 30)
    moved = frames[:60] + [np.roll(frame, 6, axis=1) for frame in frames[60:]]
    video = draw(tmp_path, moved)
    code, out = check([str(video), "--sharp-min", "0", "--static", "0,160,320,80"], capsys)
    value = float(out.split("WARN shift ")[1].split()[0])
    assert code == 1
    assert 5.0 <= value <= 7.0
    assert "whole picture" not in out


@pytest.mark.parametrize("region", ["0,160,320", "a,b,c,d", "0,160,320,900", "-5,0,10,10"])
def test_a_static_region_that_does_not_fit_the_picture_is_refused(tmp_path, capsys, region):
    video = draw(tmp_path, [textured()] * 90)
    code, out = check([str(video), f"--static={region}"], capsys)
    assert code == 2
    assert out.startswith("clip-check:")


@pytest.mark.parametrize(
    ("stamps_ms", "fps"),
    [
        ([0.0, 33.3, 66.6, 99.9, 133.2], 30.0),
        ([0.0, 66.7, 133.4, 200.1], 15.0),
    ],
)
def test_frame_rate_is_the_median_step_inverted(stamps_ms, fps):
    assert frame_rate(stamps_ms) == pytest.approx(fps, abs=0.1)


@pytest.mark.parametrize(
    ("rate", "warns"),
    [(15.0, True), (28.0, True), (28.6, False), (30.0, False), (31.4, False), (32.0, True)],
)
def test_a_rate_more_than_five_percent_off_the_sidecar_warns(rate, warns):
    measured = ClipStats(
        decoded=90, sharpness=100.0, saturated=0.0, shift=0.0, swing=0.0,
        gaps=CLEAN, unreadable_at=None, rate_fps=rate,
    )
    lines = report(measured, expected=90, sharp_min=30.0, fps=30.0)
    assert any(line.startswith("WARN rate") for line in lines) is warns


def test_the_rate_warning_names_both_rates():
    measured = ClipStats(
        decoded=90, sharpness=100.0, saturated=0.0, shift=0.0, swing=0.0,
        gaps=CLEAN, unreadable_at=None, rate_fps=15.0,
    )
    lines = report(measured, expected=90, sharp_min=30.0, fps=30.0)
    assert "WARN rate 15.0 fps, sidecar says 30" in lines


def test_flicker_is_printed_never_warned(tmp_path, capsys):
    dim = textured()
    bright = np.clip(dim.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    video = draw(tmp_path, [dim, bright] * 45)
    code, out = check([str(video)], capsys)
    swing_line = next(line for line in out.splitlines() if line.startswith("swing "))
    assert code == 0
    assert float(swing_line.split()[1]) > 0.05


def test_missing_sidecar_is_refused(tmp_path, capsys):
    video = draw(tmp_path, [textured()] * 90, with_sidecar=False)
    code, out = check([str(video)], capsys)
    assert code == 2
    assert out.strip().count("\n") == 0
    assert "clip.json" in out


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
        (100000, GapStats(missing=3, longest_s=0.1), True),
        (50000, GapStats(missing=100, longest_s=0.1), True),
    ],
)
def test_gaps_warn_on_any_missing_frame(decoded, gaps, warns):
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


def test_one_step_of_two_periods_warns_and_names_the_fix():
    lines = report(stats(1000, GapStats(missing=1, longest_s=0.066)), expected=1000, sharp_min=30.0)
    assert (
        "WARN gaps 1 frames missing, longest 0.066 s: "
        "frame numbers are not reliable: run qs clip-fix"
    ) in lines


def test_a_decoded_count_that_differs_from_the_containers_warns():
    clip = ClipStats(
        decoded=110, sharpness=100.0, saturated=0.0, shift=0.0, swing=0.0, gaps=CLEAN,
        unreadable_at=None, container_frames=120,
    )
    lines = report(clip, expected=110, sharp_min=30.0)
    assert (
        "WARN gaps decoded 110 frames, the container says 120: "
        "frame numbers are not reliable: run qs clip-fix"
    ) in lines


def test_a_clip_with_ten_frames_removed_exits_one_with_the_warning(tmp_path, capsys):
    video = make_gapped_clip(tmp_path)
    body = {"version": 1, "fps": 30, "started_utc": "2026-11-16T10:00:00.000Z",
            "stopped_utc": "2026-11-16T10:00:04.000Z", "drifted": False}
    (tmp_path / "clip.json").write_text(json.dumps(body), encoding="utf-8")
    code, out = check([str(video)], capsys)
    assert code == 1
    assert (
        "WARN gaps 10 frames missing, longest 0.366 s: "
        "frame numbers are not reliable: run qs clip-fix"
    ) in out
