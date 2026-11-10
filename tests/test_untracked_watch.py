"""`UntrackedWatch`: warmth in a thin slit that rises and falls with no tracked cup crossing (#134).

Each case is a run of frames, `(warmth, crossed)`: the slit's warmth, and whether a tracked rim
centre crossed the slit's centre line in that frame. The expected value is the frames on which an
untracked stretch has just ended. A stretch starts at warmth of 10.0 or more and ends after two
frames in a row below it.
"""

from __future__ import annotations

import pytest

from qs.cup_trigger import UntrackedWatch

THRESHOLD = 10.0

CASES = [
    pytest.param(
        [(0.0, False), (30.0, False), (30.0, False), (0.0, False), (0.0, False)],
        [False, False, False, False, True],
        id="a warm stretch with no crossing ends once, after two cold frames",
    ),
    pytest.param(
        [(0.0, False), (30.0, False), (30.0, True), (0.0, False), (0.0, False)],
        [False, False, False, False, False],
        id="a tracked rim crossing during the stretch credits it",
    ),
    pytest.param(
        [(0.0, True), (30.0, False), (0.0, False), (0.0, False)],
        [False, False, False, True],
        id="a crossing while the slit is cold credits nothing (a tracked empty cup)",
    ),
    pytest.param(
        [(10.0, False), (10.0, False), (0.0, False), (0.0, False)],
        [False, False, False, True],
        id="warmth equal to the threshold is warm",
    ),
    pytest.param(
        [(30.0, False), (30.0, False), (0.0, False)],
        [False, False, False],
        id="one cold frame does not end the stretch",
    ),
    pytest.param(
        [(30.0, False), (30.0, False)],
        [False, False],
        id="a stretch still warm at the last frame is silent",
    ),
    pytest.param(
        [(0.0, False), (11.0, False), (9.0, False), (11.0, False), (9.0, False), (0.0, False),
         (0.0, False)],
        [False, False, False, False, False, True, False],
        id="warmth flickering around the threshold is one event, not two",
    ),
    pytest.param(
        [(30.0, False), (0.0, False), (0.0, False), (30.0, False), (0.0, False), (0.0, False)],
        [False, False, True, False, False, True],
        id="two stretches are two events",
    ),
    pytest.param(
        [(30.0, True), (0.0, False), (0.0, False), (30.0, False), (0.0, False), (0.0, False)],
        [False, False, False, False, False, True],
        id="a credit is used by its own stretch and not the next",
    ),
    pytest.param(
        [(0.0, False), (0.0, False)],
        [False, False],
        id="a slit that stays cold gives nothing",
    ),
    pytest.param(
        [(40.0, False), (40.0, False), (40.0, True), (0.0, False), (0.0, False)],
        [False, False, False, False, False],
        id="a missed mochi and a tracked one in one unbroken stretch: one stretch, credited",
    ),
]


@pytest.mark.parametrize(("frames", "ended"), CASES)
def test_an_untracked_stretch_is_reported_when_warmth_has_stayed_down(
    frames: list[tuple[float, bool]], ended: list[bool]
) -> None:
    watch = UntrackedWatch(THRESHOLD)

    assert [watch.update(warmth, crossed) for warmth, crossed in frames] == ended
