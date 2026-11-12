"""The PlacementFault of a burst with too few measured frames keeps its wording, and names the lens
mask only when the mask caused a frame to be skipped (#133). Frames are a pink cup at x 130, 170
and 210, or steel where a row says the cup is missing."""

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


def _frames(cups: tuple[int | None, ...]) -> tuple[Frame, ...]:
    return tuple(
        Frame(event(1), drawn_frame("none" if x is None else "pink", centre=(x or 0, 120), seed=i),
              0.0, VERIFIED, PATCH_GREY / 255, 5.0)
        for i, x in enumerate(cups)
    )


@pytest.mark.parametrize(
    ("lens_mask_px", "cups", "message"),
    [
        ([170, 120, 28], (130, 170, 210),
         "the cup was found on 3 of 3 frames, the lens mask covered the whole inner half on 1, "
         "fewer than the 3 needed"),
        ([300, 200, 5], (130, None, None),
         "the cup was found on 1 of 3 frames, fewer than the 3 needed"),
    ],
)
def test_the_fault_names_the_lens_mask_only_when_it_skipped_a_frame(
    tmp_path: Path, lens_mask_px: list[int], cups: tuple[int | None, ...], message: str
) -> None:
    inspector = LineInspector(HASH)
    inspector.load(load_config(warmth_config(tmp_path, lens_mask_px=lens_mask_px)).config)

    with pytest.raises(PlacementFault, match=re.escape(message)):
        inspector.inspect_burst(_frames(cups))
