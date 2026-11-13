"""Which frames of a labelled pass train which class."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(("interval", "count"), [(15, 0), (16, 3), (48, 3), (49, 0)])
def test_the_edges_of_the_interval_rule(interval: int, count: int) -> None:
    cups = [cup("a.mkv", 1, CupState.FULL, 100), cup("a.mkv", 2, CupState.FULL, 100 + interval)]
    assert len([s for s in samples(cups, {"a.mkv": 1000}) if s.cls == "none"]) == count


@pytest.mark.parametrize(
    ("cups", "kwargs", "match"),
    [
        (
            [cup("a.mkv", 1, CupState.EMPTY, 100), cup("a.mkv", 2, CupState.FULL, 100)],
            {},
            r"a\.mkv.*pass 1.*pass 2.*frame 98",
        ),
        (
            [cup("a.mkv", 1, CupState.FULL, 100), cup("a.mkv", 2, CupState.EMPTY, 103)],
            {},
            r"a\.mkv.*pass 1.*pass 2.*frame 101",
        ),
        (
            [cup("a.mkv", 1, CupState.FULL, 100), cup("a.mkv", 2, CupState.FULL, 120)],
            {"half": 8},
            r"a\.mkv.*pass 1.*pass 2.*frame 106",
        ),
    ],
)
def test_a_frame_with_two_classes_raises(
    cups: list[Cup], kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        samples(cups, {"a.mkv": 1000}, **kwargs)


def test_a_pass_in_an_unknown_video_raises_with_context() -> None:
    with pytest.raises(ValueError, match=r"b\.mkv.*frame_counts"):
        samples([cup("b.mkv", 1, CupState.FULL, 10)], {"a.mkv": 1000})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"half": -1}, "half"),
        ({"fractions": (0.3, 1.2)}, "fractions"),
        ({"fractions": (-0.1,)}, "fractions"),
        ({"between": (48, 16)}, "between"),
    ],
)
def test_bad_parameters_raise(kwargs: dict[str, object], match: str) -> None:
    one = [cup("a.mkv", 1, CupState.FULL, 100)]
    with pytest.raises(ValueError, match=match):
        samples(one, {"a.mkv": 1000}, **kwargs)  # type: ignore[arg-type]
