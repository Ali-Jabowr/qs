"""A cup seen in view but never wholly in the zone is judged as itself, not as the cup behind it
(#103, code review of 06.11). Such a cup fires only once it has been gone from view for a while,
and by then the next cup may be wholly in the zone: the frame captured for the lost cup shows
that next cup, which must not be judged in its place.

The drawings: cup A rides off the zone (y 65), cup B 110 px behind it in the zone's band (y 120),
both moving 10 px a frame at 8 fps. A is last found near x 260 and fires about three frames later;
B is then near x 190, still wholly in the zone (centre x 73–207), so the frame captured for A shows
B wholly in the zone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qs.contracts import Label
from test_cup import drawn_frame
from test_line_station import OFF_THE_ZONE_Y, _labels, _reasons, _Station, _steel

GAP = 110
STEPS = range(0, 28)


def _a_ahead_of_b(step: int) -> np.ndarray:
    """A full cup A off the zone, and a full cup B behind it in the zone's band."""
    a_x = 170 + 10 * step
    pixels = drawn_frame("pale", centre=(a_x, OFF_THE_ZONE_Y), seed=step)
    behind = drawn_frame("pale", centre=(a_x - GAP, 120), seed=500 + step)
    split = a_x - GAP // 2
    pixels[:, :split] = behind[:, :split]
    return pixels


def test_a_cup_lost_without_entering_the_zone_is_not_judged_on_the_cup_behind_it(
    tmp_path: Path,
) -> None:
    frames = [_a_ahead_of_b(step) for step in STEPS] + _steel(4)

    records = _Station(tmp_path, frames).run()

    assert _labels(records) == [Label.PASS, Label.NO_VERDICT]
    assert _reasons(records) == [None, "placement"]
