"""A burst that finds too few cups says why, naming a wrong-size rim (#144).

A remounted camera sees cups at another scale, and every rim then fails `rim_size_px`. The burst
fault adds the reason of the first frame where no cup was found, so the log names the size
instead of only counting frames. The drawn rim measures about 105 px across.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qs.config import load_config
from qs.contracts import Frame
from qs.errors import PlacementFault
from qs.inspector.line import LineInspector
from station_fixtures import PATCH_GREY, VERIFIED, event
from test_cup import drawn_frame
from warmth_fixtures import warmth_config

HASH = "d" * 64


def _burst(contents: str) -> tuple[Frame, ...]:
    return tuple(
        Frame(event(1), drawn_frame(contents, centre=(x, 120), seed=i), 0.0, VERIFIED,
              PATCH_GREY / 255, 5.0)
        for i, x in enumerate((130, 170, 210))
    )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("pink", "the cup was found on 0 of 3 frames, fewer than the 3 needed; no cup wholly in "
                 "the zone: no rim the size of rim_size_px 130×130 ±5%, the largest seen is "
                 "105×105 px"),
        ("none", "the cup was found on 0 of 3 frames, fewer than the 3 needed; no cup wholly in "
                 "the zone"),
    ],
    ids=["wrong-size rim", "no rim"],
)
def test_a_burst_with_no_cup_names_why_the_first_frame_had_none(
    tmp_path: Path, contents: str, message: str
) -> None:
    inspector = LineInspector(HASH)
    inspector.load(load_config(warmth_config(tmp_path, rim_size_px=[130, 130])).config)

    with pytest.raises(PlacementFault, match=f"^{re.escape(message)}$"):
        inspector.inspect_burst(_burst(contents))
