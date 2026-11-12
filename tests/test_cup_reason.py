"""Why no cup was found: a rim of the wrong size says so, with what it measured.

A remounted camera sees the cups at another scale. Its rims then fail `rim_size_px`, and a reason
that said only "no cup wholly in the zone" would point at the cup's position instead of the
camera's scale. Drawn frames only, as in tests/test_cup.py.
"""

from __future__ import annotations

import cv2

from qs.inspector.cup import no_cup_reason
from test_cup import LINE, RIM_GREY, drawn_frame, line_config

NARROW = line_config((20, 0, 240, 240), (130, 130))


def test_rims_all_the_wrong_size_name_the_size_and_what_they_measured() -> None:
    # The drawn rim's two edges measure about 95 and 105 px across; both are under 130 - 5%.
    assert no_cup_reason(drawn_frame("pale"), NARROW) == (
        "no cup wholly in the zone: no rim the size of rim_size_px 130×130 ±5%, "
        "the largest seen is 105×105 px"
    )


def test_a_rim_of_the_right_size_out_of_the_zone_still_says_position() -> None:
    # A cup cut by the zone's right edge, and a rim too small for a cup (about 30 px) inside it.
    pixels = drawn_frame("pale", centre=(235, 120))
    cv2.ellipse(pixels, (80, 120), (15, 15), 0.0, 0, 360, (RIM_GREY,) * 3, 3)
    assert no_cup_reason(pixels, LINE) == "no cup wholly in the zone"
