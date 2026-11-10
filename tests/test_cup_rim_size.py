"""`rim_size_px` is the rim's [minor, major] axes, each accepted within 5% of its own value
(#133). A camera not straight overhead sees an ellipse. The drawn cup has half-axes 35 and 50,
which the finder fits as about 75 x 105 px."""

from __future__ import annotations

import pytest

from qs.config import LineConfig
from qs.inspector.cup import _rim_sized, find_cup, no_cup_reason
from test_cup import drawn_frame


def _line(rim_size_px: tuple[int, int] | None) -> LineConfig:
    return LineConfig(
        zone=(20, 0, 240, 240), rim_axis_px=(50, 140), blob_share_min=0.3, rim_size_px=rim_size_px
    )


@pytest.mark.parametrize(
    ("rim_size_px", "found"),
    [
        ((75, 105), True),
        ((77, 108), True),   # each axis 2-3% off its own value
        ((75, 116), False),  # the major axis 10% off
        ((83, 105), False),  # the minor axis 10% off
        (None, True),        # no key: the axis range alone
    ],
)
def test_an_elliptical_rim_is_accepted_when_each_axis_is_within_5_percent_of_its_own(
    rim_size_px: tuple[int, int] | None, found: bool
) -> None:
    pixels = drawn_frame("pink", axes=(35, 50))

    assert (find_cup(pixels, _line(rim_size_px)) is not None) is found


def test_the_reason_names_the_measured_rim_size_when_it_is_set() -> None:
    reason = no_cup_reason(drawn_frame("pink", axes=(35, 50)), _line((75, 116)))

    assert reason == (
        "no cup wholly in the zone: no rim the size of rim_size_px 75×116 ±5%, "
        "the largest seen is 75×105 px"
    )


@pytest.mark.parametrize(
    ("axes", "sized"),
    [
        ((75.0, 105.0), True),   # minor first
        ((105.0, 75.0), True),   # major first: the axes are sorted before they are compared
        ((116.0, 75.0), False),  # major first, the major 10% off
    ],
)
def test_a_rim_is_judged_by_its_sorted_axes_whichever_the_fit_gives_first(
    axes: tuple[float, float], sized: bool
) -> None:
    assert _rim_sized(((100.0, 100.0), axes, 0.0), _line((75, 105))) is sized
