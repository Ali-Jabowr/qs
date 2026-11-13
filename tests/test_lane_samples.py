"""Which frames of a labelled pass train which class."""

from __future__ import annotations

from qs.labels import Cup, CupState
from qs.lane.samples import Sample, samples


def cup(video: str, order: int, state: CupState, frame: int) -> Cup:
    return Cup(video, order, state, frame)


def frames(video: str, cls: str, order: int, *numbers: int) -> list[Sample]:
    return [Sample(video, n, cls, order) for n in numbers]


def test_a_labelled_pass_gives_its_frames_and_the_gaps_between() -> None:
    cups = [
        cup("a.mkv", 1, CupState.FULL, 100),
        cup("a.mkv", 2, CupState.EMPTY, 132),
        cup("a.mkv", 3, CupState.GAP, 164),
        cup("a.mkv", 4, CupState.FULL_PINK, 400),
    ]
    expected = (
        frames("a.mkv", "full", 1, 98, 99, 100, 101, 102)
        + frames("a.mkv", "none", 2, 110, 116, 122)
        + frames("a.mkv", "empty", 2, 130, 131, 132, 133, 134)
        + frames("a.mkv", "none", 3, 142, 148, 154)
        + frames("a.mkv", "none", 3, 162, 163, 164, 165, 166)
        + frames("a.mkv", "full", 4, 398, 399, 400, 401, 402)
    )
    assert samples(cups, {"a.mkv": 1000}) == expected


def test_frames_before_the_start_are_dropped() -> None:
    assert samples([cup("a.mkv", 1, CupState.FULL, 1)], {"a.mkv": 1000}) == frames(
        "a.mkv", "full", 1, 0, 1, 2, 3
    )


def test_frames_past_the_end_are_dropped() -> None:
    assert samples([cup("a.mkv", 1, CupState.FULL_WHITE, 99)], {"a.mkv": 100}) == frames(
        "a.mkv", "full", 1, 97, 98, 99
    )


def test_the_between_rule_never_spans_two_videos() -> None:
    cups = [cup("a.mkv", 1, CupState.FULL, 100), cup("b.mkv", 1, CupState.FULL, 120)]
    expected = frames("a.mkv", "full", 1, 98, 99, 100, 101, 102) + frames(
        "b.mkv", "full", 1, 118, 119, 120, 121, 122
    )
    assert samples(cups, {"a.mkv": 1000, "b.mkv": 1000}) == expected


def test_a_short_interval_gives_no_between_sample() -> None:
    cups = [cup("a.mkv", 1, CupState.FULL, 100), cup("a.mkv", 2, CupState.FULL, 110)]
    assert [s.cls for s in samples(cups, {"a.mkv": 1000})] == ["full"] * 10
