"""`find_rims` on drawn frames: every cup in view, each once, whether or not it is in the zone.
Every value here proves the finder on these drawings only, never on real footage."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from qs.config import LineConfig
from qs.inspector.cup import find_rims
from test_cup import RIM_GREY, drawn_frame

LINE = LineConfig(zone=(20, 30, 240, 180), rim_axis_px=(50, 140), blob_share_min=0.3)


def _two_cups() -> np.ndarray:
    """The frame of one cup on the left of column 150 and of another on the right of it."""
    pixels = drawn_frame("pale", centre=(210, 120))
    pixels[:, :150] = drawn_frame("empty", centre=(90, 120))[:, :150]
    return pixels


@pytest.mark.parametrize("contents", ["empty", "pale", "pink"])
def test_a_cup_in_the_zone_is_one_rim_whatever_it_holds(contents: str) -> None:
    [rim] = find_rims(drawn_frame(contents), LINE)
    (cx, cy), (first, second), _ = rim.ellipse
    assert (cx, cy) == pytest.approx((130.0, 120.0), abs=2.0)
    assert (first, second) == pytest.approx((100.0, 100.0), abs=8.0)
    assert rim.in_zone is True


@pytest.mark.parametrize(
    ("centre", "expected"),
    [((235, 120), (235.0, 120.0)), ((130, 65), (130.0, 65.0))],
    ids=["past_the_right_of_the_zone", "past_the_top_of_the_zone"],
)
def test_a_cup_partly_out_of_the_zone_is_found_and_says_so(
    centre: tuple[int, int], expected: tuple[float, float]
) -> None:
    [rim] = find_rims(drawn_frame("pale", centre=centre), LINE)
    assert rim.ellipse[0] == pytest.approx(expected, abs=2.0)
    assert rim.in_zone is False


def test_two_cups_are_two_rims() -> None:
    rims = find_rims(_two_cups(), LINE)
    centres = sorted(rim.ellipse[0][0] for rim in rims)
    assert centres == pytest.approx([90.0, 210.0], abs=2.0)


def test_steel_has_no_rim() -> None:
    assert find_rims(drawn_frame("none"), LINE) == ()


def test_an_arc_with_no_edge_along_most_of_its_ellipse_is_no_rim() -> None:
    pixels = drawn_frame("none")
    cv2.ellipse(pixels, (130, 120), (50, 50), 0.0, 0, 200, (RIM_GREY,) * 3, 1)
    assert find_rims(pixels, LINE) == ()


def test_the_frame_is_only_read() -> None:
    pixels = drawn_frame("pale")
    pixels.setflags(write=False)
    assert len(find_rims(pixels, LINE)) == 1
