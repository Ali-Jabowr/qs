"""The cut-out: the моти found inside its cup in a lane-window crop (#174).

The drawn crop is 276 x 230: blurred grey noise, a white rim circle of radius 80 and a filled
ellipse of half-axes 52 x 46 at (138, 115), whose true area is about 7 515 px.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.errors import ConfigError, PlacementFault
from qs.lane.unit import UnitSettings, best_cut, cut_unit, load_unit

PINK = (180, 150, 235)
CREAM = (190, 225, 235)
WHITE = (232, 232, 236)
TRUE_AREA = 7515
SETTINGS = UnitSettings(seed_px=20, reach_px=60, border_px=30, area_px=(3000, 12000), scale=0.5)
GOOD = "version: 1\nseed_px: 20\nreach_px: 60\nborder_px: 30\narea_px: [3000, 12000]\nscale: 0.5\n"


def draw_crop(
    colour: tuple[int, int, int] | None = PINK,
    dx: int = 0,
    cream_right_third: bool = False,
    bite: bool = False,
) -> np.ndarray:
    """The drawn crop: noise, the rim, and (unless `colour` is None) the ellipse shifted by `dx`."""
    rng = np.random.default_rng(7)
    grey = rng.integers(80, 140, size=(230, 276), dtype=np.uint8)
    crop = cv2.GaussianBlur(cv2.merge([grey, grey, grey]), (0, 0), 2)
    cv2.circle(crop, (138, 115), 80, (240, 240, 240), 3)
    if colour is not None:
        shape = np.zeros((230, 276), dtype=np.uint8)
        cv2.ellipse(shape, (138 + dx, 115), (52, 46), 0, 0, 360, 255, -1)
        if bite:
            cv2.circle(shape, (190 + dx, 115), 22, 0, -1)
        crop[shape > 0] = colour
        if cream_right_third:
            columns = np.arange(276)[None, :] > 138 + dx + 17
            crop[(shape > 0) & columns] = CREAM
    return crop


def test_a_pink_unit_is_cut_where_it_is() -> None:
    cut = cut_unit(draw_crop(PINK), SETTINGS)
    assert abs(int(cut.mask.sum()) - TRUE_AREA) < 0.1 * TRUE_AREA
    assert cut.mask.shape == (230, 276)
    assert abs(cut.centroid[0] - 138) < 3
    assert abs(cut.centroid[1] - 115) < 3


def test_a_cream_part_of_the_unit_stays_inside_the_cut() -> None:
    cut = cut_unit(draw_crop(PINK, cream_right_third=True), SETTINGS)
    assert abs(int(cut.mask.sum()) - TRUE_AREA) < 0.1 * TRUE_AREA


def test_a_white_unit_is_cut() -> None:
    cut = cut_unit(draw_crop(WHITE), SETTINGS)
    assert abs(int(cut.mask.sum()) - TRUE_AREA) < 0.1 * TRUE_AREA


def _solidity(contour: np.ndarray) -> float:
    return float(cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour)))


def test_a_bite_out_of_the_edge_is_kept_as_a_bite() -> None:
    bitten = _solidity(cut_unit(draw_crop(PINK, bite=True), SETTINGS).contour)
    whole = _solidity(cut_unit(draw_crop(PINK), SETTINGS).contour)
    assert bitten < 0.95
    assert whole > 0.97


def test_the_same_crop_cut_twice_gives_the_same_mask() -> None:
    crop = draw_crop(PINK)
    first = cut_unit(crop, SETTINGS)
    second = cut_unit(crop, SETTINGS)
    assert np.array_equal(first.mask, second.mask)


def test_the_cut_never_writes_into_its_input() -> None:
    crop = draw_crop(PINK)
    before = crop.copy()
    cut_unit(crop, SETTINGS)
    assert np.array_equal(crop, before)


def test_a_crop_without_a_unit_is_a_placement_fault() -> None:
    with pytest.raises(PlacementFault) as caught:
        cut_unit(draw_crop(None), SETTINGS)
    assert caught.value.detail in {"no unit", "unit size out of range"}


def test_a_unit_touching_the_border_band_is_not_wholly_in_view() -> None:
    wide_band = UnitSettings(seed_px=20, reach_px=60, border_px=60, area_px=(3000, 12000),
                             scale=0.5)
    with pytest.raises(PlacementFault) as caught:
        cut_unit(draw_crop(PINK, dx=30), wide_band)
    assert caught.value.detail == "not wholly in view"


def test_a_unit_outside_the_area_range_is_a_size_fault() -> None:
    small = UnitSettings(seed_px=20, reach_px=60, border_px=30, area_px=(100, 2000), scale=0.5)
    with pytest.raises(PlacementFault) as caught:
        cut_unit(draw_crop(PINK), small)
    assert caught.value.detail == "unit size out of range"


def test_best_cut_takes_the_crop_whose_unit_is_nearest_the_centre() -> None:
    crops = [draw_crop(PINK, dx=30), draw_crop(PINK, dx=4), draw_crop(PINK, dx=20)]
    index, cut = best_cut(crops, SETTINGS)
    assert index == 1
    assert abs(cut.centroid[0] - 142) < 3


def test_best_cut_skips_a_crop_that_faults() -> None:
    index, _ = best_cut([draw_crop(None), draw_crop(PINK, dx=10)], SETTINGS)
    assert index == 1


def test_best_cut_raises_a_fault_when_every_crop_faults() -> None:
    crops = [draw_crop(None), draw_crop(None), draw_crop(None)]
    with pytest.raises(PlacementFault):
        best_cut(crops, SETTINGS)


def test_load_unit_reads_the_file(tmp_path: Path) -> None:
    path = tmp_path / "unit.yml"
    path.write_text(GOOD, encoding="utf-8")
    assert load_unit(path) == SETTINGS


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (GOOD.replace("reach_px: 60\n", ""), "reach_px"),
        (GOOD.replace("seed_px: 20\n", ""), "seed_px"),
        (GOOD + "extra: 1\n", "extra"),
        (GOOD.replace("scale: 0.5", "scale: 1.5"), "scale"),
        (GOOD.replace("scale: 0.5", "scale: 0"), "scale"),
        (GOOD.replace("border_px: 30", "border_px: 0"), "border_px"),
        (GOOD.replace("[3000, 12000]", "[12000, 3000]"), "area_px"),
        (GOOD.replace("[3000, 12000]", "[3000]"), "area_px"),
        (GOOD.replace("seed_px: 20", "seed_px: 90"), "seed_px"),
        (GOOD.replace("version: 1", "version: 2"), "version"),
    ],
)
def test_load_unit_names_the_bad_key(tmp_path: Path, text: str, key: str) -> None:
    path = tmp_path / "unit.yml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_unit(path)


def test_load_unit_on_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unit.yml"):
        load_unit(tmp_path / "unit.yml")
