"""The cup finder on drawn frames: a clear cup's rim on bright steel, with or without a моти.

`drawn_frame` is also what tests/test_line_inspector.py draws with. Every value here proves the
finder on these drawings only, never on real footage.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from qs.config import LineConfig
from qs.inspector.cup import find_cup
from station_fixtures import HEIGHT, PATCH_GREY, PATCH_ROI, WIDTH


def line_config(zone: tuple[int, int, int, int], rim_size_px: tuple[int, int]) -> LineConfig:
    """A line block for the finder: `zone` and `rim_size_px`, and the keys every line block
    needs, which the finder never reads."""
    return LineConfig(
        zone=zone, rim_size_px=rim_size_px, direction="right", fps=8.0,
        warmth_min=5.0, burst=3, warmth_spread_max=40.0, max_speed_px_s=320.0,
    )


LINE = line_config((20, 0, 240, 240), (105, 105))
CENTRE = (130, 120)
ROUND = (50, 50)
COLOURS = {"pale": (222, 228, 232), "pink": (190, 175, 240), "green": (60, 180, 60)}
STEEL_GREY = 185
RIM_GREY = 105
FLOOR_DIM = 8  # a clear cup's floor dims the steel seen through it by this many grey levels
MOTI_SCALE = 0.62  # the моти's axes as a share of the rim's
GLARE: tuple[int, int, int] = (255, 255, 255)
GLARE_APART = ([(230, 20), (258, 30), (250, 62), (222, 48)], [(30, 190), (64, 176), (70, 222)])
GLARE_ACROSS = ([(60, 90), (92, 96), (88, 138), (58, 130)],)


def _steel(seed: int) -> np.ndarray:
    """Bright steel with the fine streaks of a polished channel."""
    rng = np.random.default_rng(seed)
    pixels = np.full((HEIGHT, WIDTH, 3), STEEL_GREY, dtype=np.uint8)
    for y in range(4, HEIGHT, 9):
        grey = STEEL_GREY + int(rng.integers(-12, 13))
        cv2.line(pixels, (0, y), (WIDTH - 1, y), (grey, grey, grey), 1)
    return pixels


def _ellipse_mask(centre: tuple[int, int], axes: tuple[int, int], angle: float) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.ellipse(mask, centre, axes, angle, 0, 360, 255, -1)
    return mask.astype(bool)


def drawn_frame(
    contents: str = "empty",
    *,
    centre: tuple[int, int] = CENTRE,
    axes: tuple[int, int] = ROUND,
    angle: float = 0.0,
    glare_on_steel: tuple[list[tuple[int, int]], ...] = (),
    glare_inside: int = 0,
    seed: int = 1,
) -> np.ndarray:
    """A 320×240 BGR frame with the grey card: `contents` is `none` (no cup), `empty`, or a
    colour of `COLOURS` for a моти. `axes` are the rim's half-lengths. `glare_on_steel` lists
    white polygons drawn over everything, `glare_inside` the number of small white patches
    drawn inside the cup."""
    pixels = _steel(seed)
    if contents != "none":
        floor = _ellipse_mask(centre, (int(axes[0] * 0.9), int(axes[1] * 0.9)), angle)
        pixels[floor] = pixels[floor] - FLOOR_DIM
        if contents in COLOURS:
            moti_axes = (int(axes[0] * MOTI_SCALE), int(axes[1] * MOTI_SCALE))
            cv2.ellipse(pixels, centre, moti_axes, angle, 0, 360, COLOURS[contents], -1)
        for index in range(glare_inside):
            offset = (centre[0] - axes[0] // 2 + index * 14, centre[1] - 12 + (index % 2) * 24)
            cv2.ellipse(pixels, offset, (5, 3), 30.0, 0, 360, GLARE, -1)
        cv2.ellipse(pixels, centre, axes, angle, 0, 360, (RIM_GREY,) * 3, 3)
    for polygon in glare_on_steel:
        cv2.fillPoly(pixels, [np.array(polygon, dtype=np.int32)], GLARE)
    pixels = cv2.GaussianBlur(pixels, (3, 3), 0)
    rng = np.random.default_rng(seed)
    noise = rng.integers(-2, 3, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    x, y, w, h = PATCH_ROI
    pixels[y : y + h, x : x + w] = PATCH_GREY
    return pixels


@pytest.mark.parametrize("contents", ["empty", "pale", "pink", "green"])
def test_the_same_rim_is_found_whatever_the_cup_holds(contents: str) -> None:
    cup = find_cup(drawn_frame(contents), LINE)
    assert cup is not None
    (cx, cy), (first, second), _ = cup.rim
    assert (cx, cy) == pytest.approx((130.0, 120.0), abs=2.0)
    assert (first, second) == pytest.approx((100.0, 100.0), abs=8.0)


def test_an_elliptical_rim_is_found() -> None:
    tilted = line_config((20, 0, 240, 240), (73, 105))
    cup = find_cup(drawn_frame("pale", axes=(50, 34), angle=25.0), tilted)
    assert cup is not None
    (cx, cy), axes, _ = cup.rim
    assert (cx, cy) == pytest.approx((130.0, 120.0), abs=2.0)
    assert sorted(axes) == pytest.approx([68.0, 100.0], abs=8.0)


@pytest.mark.parametrize("contents", ["empty", "pale", "pink"])
def test_glare_on_the_steel_apart_from_the_cup_does_not_move_the_rim(contents: str) -> None:
    cup = find_cup(drawn_frame(contents, glare_on_steel=GLARE_APART), LINE)
    assert cup is not None
    (cx, cy), (first, second), _ = cup.rim
    assert (cx, cy) == pytest.approx((130.0, 120.0), abs=2.0)
    assert (first, second) == pytest.approx((100.0, 100.0), abs=8.0)


@pytest.mark.parametrize("contents", ["empty", "pale", "pink"])
def test_a_glare_patch_across_the_rim_still_finds_it(contents: str) -> None:
    cup = find_cup(drawn_frame(contents, glare_on_steel=GLARE_ACROSS), LINE)
    assert cup is not None
    (cx, cy), (first, second), _ = cup.rim
    assert (cx, cy) == pytest.approx((130.0, 120.0), abs=3.0)
    assert (first, second) == pytest.approx((100.0, 100.0), abs=8.0)


@pytest.mark.parametrize("contents", ["empty", "pale", "pink"])
def test_glare_patches_inside_the_cup_do_not_move_the_rim(contents: str) -> None:
    cup = find_cup(drawn_frame(contents, glare_inside=5), LINE)
    assert cup is not None
    (cx, cy), (first, second), _ = cup.rim
    assert (cx, cy) == pytest.approx((130.0, 120.0), abs=2.0)
    assert (first, second) == pytest.approx((100.0, 100.0), abs=8.0)


def test_steel_with_no_cup_gives_none() -> None:
    assert find_cup(drawn_frame("none"), LINE) is None


def test_glare_alone_is_not_a_cup() -> None:
    assert find_cup(drawn_frame("none", glare_on_steel=GLARE_APART + GLARE_ACROSS), LINE) is None


@pytest.mark.parametrize(
    "centre",
    [(25, 120), (235, 120), (130, 25)],
    ids=["past_the_left_of_the_zone", "past_the_right_of_the_zone", "off_the_top_of_the_frame"],
)
def test_a_cup_partly_out_of_the_zone_gives_none(centre: tuple[int, int]) -> None:
    assert find_cup(drawn_frame("pale", centre=centre), LINE) is None


def test_a_larger_neighbour_cut_by_the_zone_does_not_hide_the_cup_inside_it() -> None:
    pixels = drawn_frame("pale", centre=(80, 120), axes=(30, 30))
    # The neighbour is within the 5% a rim may differ from rim_size_px, so both are rims.
    cv2.ellipse(pixels, (255, 120), (32, 32), 0.0, 0, 360, (RIM_GREY,) * 3, 3)
    small = line_config((20, 0, 240, 240), (67, 67))
    cup = find_cup(pixels, small)
    assert cup is not None
    (cx, cy), _, _ = cup.rim
    assert (cx, cy) == pytest.approx((80.0, 120.0), abs=2.0)


def test_a_rim_outside_the_size_limits_gives_none() -> None:
    narrow = line_config((20, 0, 240, 240), (130, 130))
    assert find_cup(drawn_frame("pale"), narrow) is None


def test_the_inside_holds_the_middle_and_not_the_rim() -> None:
    cup = find_cup(drawn_frame("empty"), LINE)
    assert cup is not None
    assert cup.inside.shape == (240, 320)
    assert cup.inside.dtype == bool
    assert cup.inside.flags.writeable is False
    assert bool(cup.inside[120, 130]) is True
    assert bool(cup.inside[120, 130 + 49]) is False
    assert bool(cup.inside[120 + 49, 130]) is False


def test_the_frame_is_only_read() -> None:
    pixels = drawn_frame("pale")
    pixels.setflags(write=False)
    assert find_cup(pixels, LINE) is not None
