"""The burst contract (#130): a source returns N frames after an event, an inspector judges a
burst, and a burst of 1 is today's single capture."""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.contracts import Label
from qs.errors import CaptureError
from qs.inspector.stub import StubInspector
from qs.sources.replay import ReplaySource
from station_fixtures import FakeClock, event, loaded_config, write_frames
from test_cup import drawn_frame
from test_line_inspector import HASH, _frame, _inspector
from test_line_replay import SECOND_X, START, _cup_x, _source


@pytest.mark.parametrize(
    ("count", "grabbed_at"),
    [
        (1, (1000.125,)),
        (2, (1000.125, 1000.25)),
    ],
)
def test_a_burst_is_consecutive_frames_after_the_event_and_one_exposure(
    tmp_path: Path, count: int, grabbed_at: tuple[float, ...]
) -> None:
    source = _source(tmp_path, FakeClock(START))
    source.open()
    burst = source.capture_burst(event(1, START), count)
    assert tuple(frame.grabbed_at for frame in burst) == grabbed_at
    assert {frame.event for frame in burst} == {event(1, START)}


def test_a_burst_of_one_is_the_frame_a_single_capture_returned_before_bursts(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, FakeClock(START))
    source.open()
    (frame,) = source.capture_burst(event(1, START), 1)
    assert frame.grabbed_at == 1000.125  # the pre-burst value of capture(event(1, START))
    assert abs(_cup_x(frame.pixels) - SECOND_X) <= 3


@pytest.mark.parametrize("count", [0, -1])
def test_a_burst_of_less_than_one_frame_is_refused(tmp_path: Path, count: int) -> None:
    source = _source(tmp_path, FakeClock(START))
    source.open()
    with pytest.raises(ValueError, match="at least one frame"):
        source.capture_burst(event(1, START), count)


def test_a_burst_running_past_the_recording_is_a_capture_error(tmp_path: Path) -> None:
    source = _source(tmp_path, FakeClock(START))
    source.open()
    with pytest.raises(CaptureError, match="the recording ended"):
        source.capture_burst(event(1, START), 3)  # frames 1, 2, 3 of a recording of 3


def test_a_source_that_cannot_burst_refuses_more_than_one_frame(tmp_path: Path) -> None:
    files = write_frames(tmp_path / "shoot", 2)
    source = ReplaySource(loaded_config(tmp_path).config, files, clock=FakeClock(step=0.001))
    source.open()
    with pytest.raises(ValueError, match="burst of 2"):
        source.capture_burst(event(1), 2)


def test_the_line_inspector_judges_a_burst_of_three_identical_frames_as_that_frame(
    tmp_path: Path,
) -> None:
    frame = _frame(drawn_frame("pale"))
    verdict = _inspector(tmp_path).inspect_burst((frame, frame, frame))
    (check,) = verdict.checks
    assert verdict.label is Label.PASS
    assert check.value == pytest.approx(10.0, abs=2.0)  # a pale моти's warmth
    assert dict(check.measures) == {"frames": 3.0, "spread": 0.0}
    assert verdict.unit_box == pytest.approx((78, 68, 105, 105), abs=4)


def test_an_inspector_that_cannot_judge_a_burst_refuses_more_than_one_frame() -> None:
    inspector = StubInspector(HASH)
    frame = _frame(drawn_frame("pale"))
    with pytest.raises(ValueError, match="burst of 2"):
        inspector.inspect_burst((frame, frame))
