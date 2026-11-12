"""`CupTracker` with no images: positions along the direction of travel and a clock go in, the
number of events to fire comes out. The patience for a cup gone from view is 0.25 s. Then
`CupTrigger` over a scripted source, for what the whole-station tests cannot reach."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qs.config import LineConfig
from qs.cup_trigger import CupTracker, CupTrigger, SeenCup
from station_fixtures import FakeClock, ScriptedSource, StepWall
from test_cup import drawn_frame

TOLERANCE = 25.0
LINE = LineConfig(
    zone=(20, 30, 240, 180), rim_size_px=(105, 105), direction="right", fps=8.0,
    warmth_min=5.0, burst=3, warmth_spread_max=40.0, max_speed_px_s=320.0,
)


def _tracker() -> CupTracker:
    return CupTracker(TOLERANCE)


def test_no_cups_fire_nothing() -> None:
    tracker = _tracker()
    assert tracker.update([], 0.0) == 0
    assert tracker.update([], 1.0) == 0
    assert tracker.finish() == 0


def test_a_cup_fires_once_on_first_being_in_the_zone() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(50.0, False)], 0.0) == 0
    assert tracker.update([SeenCup(90.0, True)], 0.1) == 1
    assert tracker.update([SeenCup(130.0, True)], 0.2) == 0
    assert tracker.update([SeenCup(170.0, False)], 0.3) == 0


def test_a_cup_that_stops_in_the_zone_does_not_fire_again() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(90.0, True)], 0.0) == 1
    assert tracker.update([SeenCup(90.0, True)], 5.0) == 0
    assert tracker.update([SeenCup(90.0, True)], 500.0) == 0


def test_a_position_that_jitters_backwards_is_the_same_cup() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(90.0, True)], 0.0) == 1
    assert tracker.update([SeenCup(70.0, True)], 0.1) == 0
    assert tracker.update([SeenCup(90.0, True)], 0.2) == 0


def test_a_position_behind_by_more_than_the_tolerance_is_a_new_cup() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(190.0, True)], 0.0) == 1
    assert tracker.update([SeenCup(190.0, True), SeenCup(90.0, True)], 0.1) == 1


def test_two_cups_in_view_are_followed_apart() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(170.0, True), SeenCup(50.0, False)], 0.0) == 1
    assert tracker.update([SeenCup(210.0, False), SeenCup(90.0, True)], 0.1) == 1
    assert tracker.update([SeenCup(250.0, False), SeenCup(130.0, True)], 0.2) == 0


def test_a_cup_missed_for_a_moment_does_not_fire_again() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(90.0, True)], 0.0) == 1
    assert tracker.update([], 0.1) == 0
    assert tracker.update([], 0.2) == 0
    assert tracker.update([SeenCup(210.0, True)], 0.24) == 0


def test_a_cup_gone_without_being_in_the_zone_fires_once_after_the_patience() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(50.0, False)], 0.0) == 0
    assert tracker.update([], 0.25) == 0
    assert tracker.update([], 0.26) == 1
    assert tracker.update([], 0.27) == 0


def test_a_fired_cup_that_leaves_fires_nothing_more() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(90.0, True)], 0.0) == 1
    assert tracker.update([], 1.0) == 0
    assert tracker.update([], 2.0) == 0


def test_the_next_cup_after_one_has_left_fires_for_itself() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(90.0, True)], 0.0) == 1
    assert tracker.update([], 1.0) == 0
    assert tracker.update([SeenCup(90.0, True)], 2.0) == 1


def test_finishing_fires_each_cup_still_followed_and_not_yet_fired() -> None:
    tracker = _tracker()
    assert tracker.update([SeenCup(250.0, True), SeenCup(50.0, False)], 0.0) == 1
    assert tracker.finish() == 1
    assert tracker.finish() == 0


def _trigger(source: ScriptedSource, ended: Callable[[], bool] = lambda: False) -> CupTrigger:
    return CupTrigger(
        LINE, source, source_id="replay:shoot", clock=source.clock, wall=StepWall(), ended=ended
    )


def _source(pixels: np.ndarray | None) -> ScriptedSource:
    return ScriptedSource(clock=FakeClock(start=500.0), preview_pixels=pixels)


def _two_cups_in_the_zone() -> np.ndarray:
    pixels = drawn_frame("pale", centre=(200, 120))
    pixels[:, :140] = drawn_frame("empty", centre=(80, 120))[:, :140]
    return pixels


def test_two_cups_firing_on_one_frame_give_an_event_on_each_of_two_polls() -> None:
    trigger = _trigger(_source(_two_cups_in_the_zone()))
    first, second, third = trigger.poll(), trigger.poll(), trigger.poll()
    assert first is not None and second is not None
    assert (first.seq, second.seq) == (1, 2)
    assert first.t0 == 500.0
    assert first.source_id == "replay:shoot"
    assert third is None


def test_a_frozen_preview_tells_nothing_so_a_cup_is_gone_only_on_a_new_frame() -> None:
    clock = FakeClock(start=500.0)
    source = ScriptedSource(clock=clock, preview_pixels=drawn_frame("pale", centre=(235, 120)))
    trigger = _trigger(source)
    assert trigger.poll() is None
    source.preview_pixels = drawn_frame("none")
    assert trigger.poll() is None
    clock.advance(1.0)
    assert trigger.poll() is None
    source.preview_pixels = drawn_frame("none", seed=2)
    fired = trigger.poll()
    assert fired is not None
    assert fired.t0 == 501.0


def test_the_configured_axis_is_the_one_that_tells_a_new_cup_from_a_moved_one() -> None:
    """Travelling down, a rim further up by more than the tolerance is a cup coming behind."""
    line = LineConfig(
        zone=(20, 30, 240, 180), rim_size_px=(105, 105), direction="down", fps=8.0,
        warmth_min=5.0, burst=3, warmth_spread_max=40.0, max_speed_px_s=320.0,
    )
    clock = FakeClock(start=500.0)
    source = ScriptedSource(clock=clock, preview_pixels=drawn_frame("pale", centre=(130, 90)))
    trigger = CupTrigger(
        line, source, source_id="s", clock=clock, wall=StepWall(), ended=lambda: False
    )
    assert trigger.poll() is not None
    source.preview_pixels = drawn_frame("pale", centre=(130, 150))
    assert trigger.poll() is None
    source.preview_pixels = drawn_frame("pale", centre=(130, 90), seed=2)
    assert trigger.poll() is not None


def test_no_preview_is_not_the_end_until_the_source_says_so() -> None:
    trigger = _trigger(_source(None))
    assert trigger.poll() is None
    assert trigger.exhausted is False


def test_the_end_of_the_frames_fires_a_cup_not_yet_fired_then_is_exhausted() -> None:
    source = _source(drawn_frame("pale", centre=(235, 120)))
    trigger = _trigger(source, ended=lambda: source.preview_pixels is None)
    assert trigger.poll() is None
    source.preview_pixels = None
    assert trigger.exhausted is False
    last = trigger.poll()
    assert last is not None
    assert last.seq == 1
    assert trigger.exhausted is True
    assert trigger.poll() is None


def test_closing_ends_it_and_drops_events_not_yet_returned() -> None:
    trigger = _trigger(_source(_two_cups_in_the_zone()))
    assert trigger.poll() is not None
    trigger.close()
    assert trigger.exhausted is True
    assert trigger.poll() is None
