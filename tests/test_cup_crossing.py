"""`CupTracker.crossed`: how many followed cups crossed a line across the travel in a frame (#134).

Each case is a run of frames, the positions along the travel of the cups found in it, taken 0.125 s
apart (8 fps), and the crossings counted per frame. The line is at 150 and a cup crosses when its
position goes from below it to at or above it. The tolerance is 25.
"""

from __future__ import annotations

import pytest

from qs.cup_trigger import CupTracker, SeenCup

LINE = 150.0

CASES = [
    pytest.param([[110.0], [150.0], [190.0]], [0, 1, 0], id="a cup moving over the line"),
    pytest.param([[190.0], [230.0]], [0, 0], id="a cup first seen beyond the line"),
    pytest.param([[110.0], [], [190.0]], [0, 0, 1], id="a cup missed for a frame is the same"),
    pytest.param([[110.0, 70.0], [150.0, 110.0]], [0, 1], id="only the cup that reaches the line"),
    pytest.param([[110.0, 30.0], [150.0, 70.0], [190.0, 110.0]], [0, 1, 0], id="one at a time"),
    pytest.param([[60.0], [100.0], [140.0]], [0, 0, 0], id="a cup that stays short of the line"),
    pytest.param([[149.0], [149.0], [150.0]], [0, 0, 1], id="a cup that stops short then goes on"),
]


@pytest.mark.parametrize(("frames", "crossed"), CASES)
def test_a_cup_is_counted_in_the_frame_it_reaches_the_line(
    frames: list[list[float]], crossed: list[int]
) -> None:
    tracker = CupTracker(25.0)
    counted = []
    for index, positions in enumerate(frames):
        tracker.fires([SeenCup(along, False) for along in positions], index * 0.125, LINE)
        counted.append(tracker.crossed)

    assert counted == crossed
