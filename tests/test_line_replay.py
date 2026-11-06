"""`LineReplaySource` on drawn files and a fake clock: frames at a fixed pace by the clock, previews
that consume nothing, and `capture` taking the first frame later than the trigger. 8 fps, a frame
every 0.125 s, keeps the clock's arithmetic exact; the config's exposure is 10 ms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from qs.config import load_config
from qs.errors import CaptureError
from qs.sources.line_replay import LineReplaySource
from station_fixtures import FakeClock, config_mapping, event, write_png
from test_cup import drawn_frame

START = 1000.0
PERIOD = 0.125
FIRST_X, SECOND_X, THIRD_X = 70, 130, 190


def _source(root: Path, clock: FakeClock) -> LineReplaySource:
    frames = [drawn_frame("pale", centre=(x, 120)) for x in (FIRST_X, SECOND_X, THIRD_X)]
    files = []
    for index, pixels in enumerate(frames):
        path = root / f"f{index}.png"
        write_png(path, pixels)
        files.append(path)
    data = config_mapping(root)
    path = root / "station.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return LineReplaySource(load_config(path).config, files, fps=8.0, clock=clock)


def _cup_x(pixels: np.ndarray) -> int:
    """The column midway between the rim's two dark crossings of the row through the cup."""
    return round(float(np.flatnonzero(pixels[120, :, 0] < 130).mean()))


def test_no_preview_before_it_is_opened(tmp_path: Path) -> None:
    source = _source(tmp_path, FakeClock(START))
    assert source.preview() is None
    assert source.finished is False


def test_preview_follows_the_clock_and_consumes_nothing(tmp_path: Path) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    source.open()
    first = source.preview()
    assert first is not None
    assert source.preview() is first
    clock.advance(PERIOD)
    second = source.preview()
    assert second is not first
    assert second is not None
    assert abs(_cup_x(first) - FIRST_X) <= 3
    assert abs(_cup_x(second) - SECOND_X) <= 3


def test_past_the_last_frame_there_is_no_preview_and_it_is_finished(tmp_path: Path) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    source.open()
    clock.advance(2 * PERIOD)
    assert source.preview() is not None
    assert source.finished is False
    clock.advance(PERIOD)
    assert source.preview() is None
    assert source.finished is True


def test_capture_takes_the_first_frame_later_than_the_trigger_and_the_exposure(
    tmp_path: Path,
) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    source.open()
    trigger = event(7, START)
    frame = source.capture(trigger)
    assert frame.event == trigger
    assert frame.grabbed_at == 1000.125
    assert abs(_cup_x(frame.pixels) - SECOND_X) <= 3


def test_capture_of_a_trigger_just_before_a_frame_skips_it_by_the_exposure(tmp_path: Path) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    source.open()
    frame = source.capture(event(1, START + 0.120))
    assert frame.grabbed_at == 1000.25
    assert abs(_cup_x(frame.pixels) - THIRD_X) <= 3


def test_capture_does_not_hand_out_the_preview_array(tmp_path: Path) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    source.open()
    clock.advance(PERIOD)
    shown = source.preview()
    frame = source.capture(event(1, START))
    assert frame.grabbed_at == 1000.125
    assert frame.pixels is not shown


def test_capture_past_the_last_frame_is_a_capture_error(tmp_path: Path) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    source.open()
    with pytest.raises(CaptureError, match="the recording ended"):
        source.capture(event(1, START + 2 * PERIOD))


def test_capture_before_open_is_a_capture_error(tmp_path: Path) -> None:
    source = _source(tmp_path, FakeClock(START))
    with pytest.raises(CaptureError, match="before open"):
        source.capture(event(1, START))


def test_reopening_after_a_fault_does_not_restart_the_recording(tmp_path: Path) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    source.open()
    clock.advance(2 * PERIOD)
    source.close()
    source.open()
    shown = source.preview()
    assert shown is not None
    assert abs(_cup_x(shown) - THIRD_X) <= 3
    assert source.capture(event(1, START + PERIOD)).grabbed_at == 1000.25


def test_an_unreadable_frame_gives_no_preview_and_a_capture_error(tmp_path: Path) -> None:
    clock = FakeClock(START)
    source = _source(tmp_path, clock)
    (tmp_path / "f1.png").write_bytes(b"not a png")
    source.open()
    clock.advance(PERIOD)
    assert source.preview() is None
    assert source.finished is False
    with pytest.raises(CaptureError, match="not a PNG"):
        source.capture(event(1, START))
