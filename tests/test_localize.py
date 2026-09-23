"""The Localizer: a frame in, the silhouette out — or the placement fault naming the rule.

Every acceptance rule of docs/15 §2, on synthetic frames drawn in memory: the disc's box,
centroid and area, the three faults, the ROI, the fill, the specks, the border-margin
boundary, the fixed threshold, determinism and purity (C1, C2).
"""

import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FRAME_SHAPE, Paint
from qs.config import LocalizeConfig, load_config
from qs.errors import PlacementFault
from qs.inspector.localize import Silhouette, localize

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"

UNIT_CENTRE = (960, 540)  # the centre of the example ROI [360, 0, 1200, 1080]
UNIT_RADIUS = 180
UNIT_GREY = 200
DARK_GREY = 30
DISC_BOX = (780, 360, 360, 360)
DISC_AREA = math.pi * UNIT_RADIUS**2
BLOB_BOX = (500, 900, 40, 40)  # 1600 px, far below min_area_px, clear of the disc
EDGES = ("left", "right", "top", "bottom")
_TOWARDS = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}


def _fault(
    pixels: np.ndarray, config: LocalizeConfig, detail: str
) -> PlacementFault:
    """`localize` raises the fault with exactly this detail, and the placement reason."""
    with pytest.raises(PlacementFault) as raised:
        localize(pixels, config)
    assert raised.value.detail == detail
    assert raised.value.reason == "placement"
    return raised.value


def _gap_to(config: LocalizeConfig, edge: str, box: tuple[int, int, int, int]) -> int:
    """The detected box's gap to one ROI edge, in pixels."""
    rx, ry, rw, rh = config.roi
    x, y, w, h = box
    return {
        "left": x - rx,
        "right": rx + rw - (x + w),
        "top": y - ry,
        "bottom": ry + rh - (y + h),
    }[edge]


def _disc_at_gap(
    paint: Paint, config: LocalizeConfig, edge: str, gap: int, *, towards: int = 0
) -> np.ndarray:
    """A pale disc whose detected unit_box sits exactly `gap` px from one ROI edge.

    Blur, threshold and the morphology move the detected box a pixel or so off the drawn
    circle, so the frame is drawn once at a safe distance, the difference measured, and the
    disc redrawn in the corrected place. `towards` then nudges it that many pixels towards
    the edge — for the "one pixel closer" case.
    """
    rx, ry, rw, rh = config.roi
    mid_x, mid_y = rx + rw // 2, ry + rh // 2
    safe = UNIT_RADIUS + config.border_margin_px + 40
    nominal = {
        "left": (rx + safe + UNIT_RADIUS, mid_y),
        "right": (rx + rw - safe - UNIT_RADIUS, mid_y),
        "top": (mid_x, ry + safe + UNIT_RADIUS),
        "bottom": (mid_x, ry + rh - safe - UNIT_RADIUS),
    }[edge]
    drawn = paint.floor()
    paint.disc(drawn, nominal, UNIT_RADIUS)
    bx, by, bw, bh = localize(drawn, config).unit_box
    if edge == "left":
        cx, cy = rx + gap - (bx - (rx + safe)) + UNIT_RADIUS, mid_y
    elif edge == "right":
        cx, cy = rx + rw - gap + ((rx + rw - safe) - (bx + bw)) - UNIT_RADIUS, mid_y
    elif edge == "top":
        cx, cy = mid_x, ry + gap - (by - (ry + safe)) + UNIT_RADIUS
    else:
        cx, cy = mid_x, ry + rh - gap + ((ry + rh - safe) - (by + bh)) - UNIT_RADIUS
    tx, ty = _TOWARDS[edge]
    placed = paint.floor()
    paint.disc(placed, (cx + tx * towards, cy + ty * towards), UNIT_RADIUS)
    return placed


def test_a_pale_disc_gives_the_discs_silhouette(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    found = localize(pixels, localize_config)
    assert isinstance(found, Silhouette)
    for got, want in zip(found.unit_box, DISC_BOX, strict=True):
        assert abs(got - want) <= 1
    assert math.dist(found.centroid, UNIT_CENTRE) <= 0.5
    assert abs(int(found.mask.sum()) - DISC_AREA) <= 0.01 * DISC_AREA
    assert found.mask.shape == FRAME_SHAPE
    assert found.mask.dtype == bool
    assert not found.mask.flags.writeable
    assert tuple(cv2.boundingRect(found.contour)) == found.unit_box


def test_the_same_frame_twice_is_the_same_silhouette_and_the_input_is_untouched(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    before = pixels.copy()
    first = localize(pixels, localize_config)
    second = localize(pixels, localize_config)
    assert np.array_equal(first.mask, second.mask)
    assert first.unit_box == second.unit_box
    assert first.centroid == second.centroid
    assert np.array_equal(pixels, before)
    pixels.setflags(write=False)
    third = localize(pixels, localize_config)
    assert np.array_equal(third.mask, first.mask)
    with pytest.raises(ValueError):
        first.mask[0, 0] = True


def test_an_empty_floor_is_not_a_unit(localize_config: LocalizeConfig, paint: Paint) -> None:
    _fault(paint.floor(), localize_config, "no unit")


def test_bright_specks_are_not_a_unit(localize_config: LocalizeConfig, paint: Paint) -> None:
    pixels = paint.floor()
    paint.specks(pixels, 300)
    _fault(pixels, localize_config, "no unit")


def test_a_blob_below_min_area_px_is_not_a_unit(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.blob(pixels, BLOB_BOX)
    _fault(pixels, localize_config, "no unit")


def test_two_discs_are_more_than_one_unit(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, (600, 540), UNIT_RADIUS, UNIT_GREY)
    paint.disc(pixels, (1320, 540), UNIT_RADIUS, UNIT_GREY)
    _fault(pixels, localize_config, "more than one unit")


def test_a_blob_below_min_area_px_beside_the_disc_is_ignored(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    alone = paint.floor()
    paint.disc(alone, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    paint.blob(pixels, BLOB_BOX)
    found = localize(pixels, localize_config)
    reference = localize(alone, localize_config)
    assert found.unit_box == reference.unit_box
    assert found.centroid == reference.centroid
    assert np.array_equal(found.mask, reference.mask)


def test_a_bright_square_over_the_grey_card_is_ignored(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    card = load_config(EXAMPLE).config.lighting.reference_patch.roi
    alone = paint.floor()
    paint.disc(alone, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    paint.blob(pixels, card)
    found = localize(pixels, localize_config)
    reference = localize(alone, localize_config)
    assert found.unit_box == reference.unit_box
    assert np.array_equal(found.mask, reference.mask)


def test_a_bright_square_over_the_grey_card_alone_is_not_a_unit(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    card = load_config(EXAMPLE).config.lighting.reference_patch.roi
    pixels = paint.floor()
    paint.blob(pixels, card)
    _fault(pixels, localize_config, "no unit")


def test_a_dark_interior_blob_is_part_of_the_unit(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    solid = paint.floor()
    paint.disc(solid, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    paint.disc(pixels, UNIT_CENTRE, 80, DARK_GREY)  # (80 / 180) ** 2 — about 20% of the disc
    found = localize(pixels, localize_config)
    reference = localize(solid, localize_config)
    assert int(found.mask.sum()) == int(reference.mask.sum())
    assert np.array_equal(found.mask, reference.mask)


def test_specks_away_from_the_disc_change_nothing(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    alone = paint.floor()
    paint.disc(alone, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    paint.specks(pixels, 300, avoid=((UNIT_CENTRE[0], UNIT_CENTRE[1], UNIT_RADIUS),))
    found = localize(pixels, localize_config)
    reference = localize(alone, localize_config)
    assert np.array_equal(found.mask, reference.mask)
    assert found.unit_box == reference.unit_box


def test_two_touching_discs_are_one_silhouette(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, (760, 540), 220, UNIT_GREY)
    paint.disc(pixels, (1160, 540), 220, UNIT_GREY)
    found = localize(pixels, localize_config)
    assert isinstance(found, Silhouette)
    assert found.unit_box[2] > 600  # the one box spans both discs


def test_a_disc_exactly_at_the_threshold_is_not_a_unit(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, localize_config.threshold)
    _fault(pixels, localize_config, "no unit")


def test_a_disc_above_the_threshold_is_found(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, localize_config.threshold + 10)
    found = localize(pixels, localize_config)
    assert isinstance(found, Silhouette)


@pytest.mark.parametrize("edge", EDGES)
def test_a_disc_exactly_at_the_border_margin_is_accepted(
    localize_config: LocalizeConfig, paint: Paint, edge: str
) -> None:
    margin = localize_config.border_margin_px
    pixels = _disc_at_gap(paint, localize_config, edge, margin)
    found = localize(pixels, localize_config)
    assert _gap_to(localize_config, edge, found.unit_box) == margin


def test_a_pale_square_localizes_with_the_full_boundary(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.blob(pixels, (810, 390, 300, 300), UNIT_GREY)  # centred in the example ROI
    found = localize(pixels, localize_config)
    assert isinstance(found, Silhouette)
    points = found.contour.reshape(-1, 2)
    assert len(points) >= 5  # a compressed contour would be the square's 4 corners
    assert len(np.unique(points, axis=0)) == len(points)  # one point per boundary pixel
    closed = np.vstack([points, points[:1]])
    steps = np.abs(np.diff(closed, axis=0)).max(axis=1)
    assert (steps == 1).all()  # every consecutive pair 8-adjacent, the wrap-around included


@pytest.mark.parametrize("edge", EDGES)
def test_a_disc_one_pixel_closer_to_an_edge_is_refused(
    localize_config: LocalizeConfig, paint: Paint, edge: str
) -> None:
    margin = localize_config.border_margin_px
    pixels = _disc_at_gap(paint, localize_config, edge, margin, towards=1)
    borderless = replace(localize_config, border_margin_px=0)
    measured = localize(pixels, borderless)  # the same chain, the border rule switched off
    assert _gap_to(localize_config, edge, measured.unit_box) == margin - 1
    _fault(pixels, localize_config, "not wholly in view")


@pytest.mark.parametrize("edge", EDGES)
def test_a_disc_cut_by_an_roi_edge_is_refused(
    localize_config: LocalizeConfig, paint: Paint, edge: str
) -> None:
    rx, ry, rw, rh = localize_config.roi
    centre = {
        "left": (rx, ry + rh // 2),
        "right": (rx + rw, ry + rh // 2),
        "top": (rx + rw // 2, ry),
        "bottom": (rx + rw // 2, ry + rh),
    }[edge]
    pixels = paint.floor()
    paint.disc(pixels, centre, UNIT_RADIUS, UNIT_GREY)
    _fault(pixels, localize_config, "not wholly in view")
