"""`UntrackedWatch`: warmth in a fixed window that rises and falls with no tracked cup (#134).

Each case is a run of frames, `(warmth, cup_in_window)`, and the frames on which the watch says an
untracked passage has just ended. The threshold is 5.0: warmth must be above it to count as warm.
"""

from __future__ import annotations

import pytest

from qs.cup_trigger import UntrackedWatch

THRESHOLD = 5.0

CASES = [
    pytest.param(
        [(0.0, False), (30.0, False), (30.0, False), (0.0, False)],
        [False, False, False, True],
        id="a warm passage with no cup ends once",
    ),
    pytest.param(
        [(0.0, False), (30.0, True), (30.0, True), (0.0, False)],
        [False, False, False, False],
        id="a cup found in the window while it is warm is tracked",
    ),
    pytest.param(
        [(0.0, True), (30.0, False), (0.0, False)],
        [False, False, False],
        id="a cup found in the window just before it warms is tracked",
    ),
    pytest.param(
        [(5.0, False), (5.0, False), (0.0, False)],
        [False, False, False],
        id="warmth equal to the threshold is not warm",
    ),
    pytest.param(
        [(30.0, False), (30.0, False)],
        [False, False],
        id="warmth that has not fallen again is not yet a passage",
    ),
    pytest.param(
        [(30.0, False), (0.0, False), (30.0, False), (0.0, False)],
        [False, True, False, True],
        id="two warm passages are two events",
    ),
    pytest.param(
        [(30.0, True), (0.0, False), (0.0, False), (30.0, False), (0.0, False)],
        [False, False, False, False, True],
        id="a tracked passage does not excuse the next one",
    ),
    pytest.param(
        [(0.0, False), (0.0, False)],
        [False, False],
        id="a window that stays cold gives nothing",
    ),
]


@pytest.mark.parametrize(("frames", "ended"), CASES)
def test_an_untracked_passage_is_reported_when_warmth_falls(
    frames: list[tuple[float, bool]], ended: list[bool]
) -> None:
    watch = UntrackedWatch(THRESHOLD)

    assert [watch.update(warmth, cup) for warmth, cup in frames] == ended
