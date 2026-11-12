"""Why no cup was found: a rim of the wrong size says so, with what it measured.

A remounted camera sees the cups at another scale. Its rims then fail `rim_size_px`, and a reason
that said only "no cup wholly in the zone" would point at the cup's position instead of the
camera's scale. Drawn frames only, as in tests/test_cup.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from qs.config import LineConfig
from qs.errors import PlacementFault
from qs.inspector.cup import no_cup_reason
from test_cup import LINE, RIM_GREY, drawn_frame
from test_line_inspector import _frame, _inspector

NARROW = LineConfig(zone=(20, 0, 240, 240), rim_size_px=(130, 130), blob_share_min=0.3)


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


def test_the_line_inspector_reports_a_rim_too_large_for_rim_size_px(tmp_path: Path) -> None:
    # An empty cup's rim of about 160 px, wholly in the zone, against rim_size_px 105×105.
    with pytest.raises(PlacementFault) as raised:
        _inspector(tmp_path).inspect(_frame(drawn_frame("empty", axes=(80, 80))))
    assert str(raised.value) == (
        "no cup wholly in the zone: no rim the size of rim_size_px 105×105 ±5%, "
        "the largest seen is 165×165 px"
    )
