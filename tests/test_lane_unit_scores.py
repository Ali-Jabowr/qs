"""The off-colour scores of a cut-out unit: colour spread, whole-unit colour and the PatchCore
read inside the mask (#175).

Colour images are built in 8-bit Lab and converted to BGR, so (a, b) are known. The mask is a
filled disc of radius 40 at the centre of a 276 x 230 crop. The drawn calibrate clip C-1.mkv
(320 x 260, lane window [20, 10, 276, 230], 10 fps, lossless) has full_pink@50, empty@82,
full_white@114 and full@146 with no unit drawn; at each pass the unit is centred on frame f and
shifted 8 px on the four neighbours, so the nearest-to-centre frame is f itself.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from qs.errors import InferenceError
from qs.inspector.model import Scored, Scorer
from qs.labels import CupState, load_labels
from qs.lane.lanefile import Lane
from qs.lane.split import Split, file_sha256, write_split
from qs.lane.unit import UnitSettings
from qs.lane.unit_scores import (
    ScoreRow,
    colour_shift,
    colour_spread,
    patch_score,
    score_block,
    unit_colour,
    write_scores,
)
from test_lane_unit import PINK, WHITE, draw_crop

HEIGHT, WIDTH = 230, 276
BASE = (180, 150, 120)


def _disc(radius: int = 40) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(mask, (138, 115), radius, 1, -1)
    return mask.astype(bool)


def _lab_crop(colour: tuple[int, int, int] = BASE) -> np.ndarray:
    lab = np.full((HEIGHT, WIDTH, 3), colour, dtype=np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _with_patch(crop: np.ndarray, lab_colour: tuple[int, int, int]) -> np.ndarray:
    """A centred disc of radius 20 (29.2% of the eroded unit) in another Lab colour."""
    patch = _lab_crop(lab_colour)
    inside = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(inside, (138, 115), 20, 1, -1)
    out = crop.copy()
    out[inside > 0] = patch[inside > 0]
    return out


def test_one_colour_has_no_spread() -> None:
    assert colour_spread(_lab_crop(), _disc()) == pytest.approx(0.0, abs=1.0)


def test_a_patch_of_another_colour_adds_its_share_of_the_distance() -> None:
    crop = _with_patch(_lab_crop(), (180, 130, 150))
    assert colour_spread(crop, _disc()) == pytest.approx(10.8, abs=1.0)


def test_brightness_alone_is_not_spread() -> None:
    crop = _lab_crop((180, 150, 120))
    crop[:, 138:] = _lab_crop((120, 150, 120))[:, 138:]
    assert colour_spread(crop, _disc()) == pytest.approx(0.0, abs=2.0)


def test_glare_is_left_out() -> None:
    crop = _lab_crop()
    crop[100:130, 120:160] = (250, 250, 250)
    assert colour_spread(crop, _disc()) == pytest.approx(0.0, abs=1.0)


def _forty_pixels() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[100:104, 100:110] = True
    return mask


def test_colour_spread_refuses_a_mask_of_forty_pixels() -> None:
    with pytest.raises(InferenceError, match="pixels"):
        colour_spread(_lab_crop(), _forty_pixels())


def test_unit_colour_refuses_a_mask_of_forty_pixels() -> None:
    with pytest.raises(InferenceError, match="pixels"):
        unit_colour(_lab_crop(), _forty_pixels())


def test_unit_colour_is_the_median_lab_of_the_unit() -> None:
    assert unit_colour(_lab_crop(), _disc()) == pytest.approx((70.6, 150.0, 120.0), abs=1.0)


def test_a_patch_does_not_move_the_median_colour() -> None:
    crop = _with_patch(_lab_crop(), (180, 130, 150))
    assert unit_colour(crop, _disc()) == pytest.approx((70.6, 150.0, 120.0), abs=1.0)


REFERENCE = (70.0, 150.0, 120.0)


@pytest.mark.parametrize(
    ("colour", "expected", "tolerance"),
    [
        ((70.0, 150.0, 120.0), 0.0, 1e-9),
        ((70.0, 153.0, 124.0), 5.0, 1e-9),
        ((50.0, 150.0, 120.0), 20.0, 1e-9),
        ((70.0, 130.0, 150.0), 36.06, 0.01),
    ],
)
def test_colour_shift_is_the_euclidean_distance(
    colour: tuple[float, float, float], expected: float, tolerance: float
) -> None:
    assert colour_shift(colour, REFERENCE) == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_colour_shift_refuses_a_non_finite_colour(bad: float) -> None:
    with pytest.raises(InferenceError, match="colour"):
        colour_shift((70.0, bad, 120.0), REFERENCE)


def _map() -> np.ndarray:
    anomaly = np.zeros((4, 4), dtype=np.float32)
    anomaly[0, 0] = 9.0
    anomaly[2, 2] = 5.0
    return anomaly


def _lower_right_quarter() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[HEIGHT // 2 :, WIDTH // 2 :] = True
    return mask


def test_patch_score_is_the_highest_value_inside_the_mask() -> None:
    assert patch_score(_map(), _lower_right_quarter()) == 5.0


def test_patch_score_refuses_a_non_finite_value_inside_the_mask() -> None:
    anomaly = _map()
    anomaly[3, 3] = np.nan
    with pytest.raises(InferenceError, match="finite"):
        patch_score(anomaly, _lower_right_quarter())


def test_patch_score_refuses_a_mask_that_vanishes_at_the_map_size() -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[1:3, 1:3] = True
    with pytest.raises(InferenceError, match="mask"):
        patch_score(_map(), mask)


# score_block on a drawn clip

LANE = Lane((20, 10, WIDTH, HEIGHT), 10.0)
SETTINGS = UnitSettings(seed_px=20, reach_px=70, border_px=30, area_px=(3000, 12000), scale=0.5)
PASSES = [("full_pink", 50), ("empty", 82), ("full_white", 114), ("full", 146)]


class CentreScorer:
    """A map of zeros with 7.5 in the four centre cells; `pred_score` is not used."""

    def score(self, window_input: np.ndarray) -> Scored:
        anomaly = np.zeros((8, 8), dtype=np.float32)
        anomaly[3:5, 3:5] = 7.5
        return Scored(anomaly, 0.0, 1.0, 1.0)


def make_clip(path: Path) -> None:
    """The drawn calibrate clip; frame n is the unit of the nearest pass, or nothing."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (320, 260))
    assert writer.isOpened()
    colours = {50: PINK, 114: WHITE}
    empty = draw_crop(None)
    for n in range(160):
        frame = np.full((260, 320, 3), 100, dtype=np.uint8)
        near = [f for f in colours if abs(n - f) <= 2]
        crop = draw_crop(colours[near[0]], dx=0 if n == near[0] else 8) if near else empty
        frame[10:240, 20:296] = crop
        writer.write(frame)
    writer.release()


def make_rig(root: Path) -> dict[str, Path]:
    """The footage folder with C-1 (calibrate, drawn) and byte-file T-1 (train) and E-1 (test),
    plus the labels, lane, unit and split files."""
    footage = root / "footage"
    footage.mkdir()
    make_clip(footage / "C-1.mkv")
    (footage / "T-1.mkv").write_bytes(b"train")
    (footage / "E-1.mkv").write_bytes(b"test")
    cups = [{"video": "C-1.mkv", "order": i + 1, "state": s, "frame": f}
            for i, (s, f) in enumerate(PASSES)]
    cups += [{"video": "T-1.mkv", "order": 1, "state": "full", "frame": 5},
             {"video": "E-1.mkv", "order": 1, "state": "full", "frame": 5}]
    labels = root / "L.json"
    labels.write_text(json.dumps({"version": 1, "source": "L", "frames": {}, "cups": cups}))
    lane = root / "lane.yml"
    lane.write_text(yaml.safe_dump({"version": 1, "window": list(LANE.window), "fps": 10}))
    unit = root / "unit.yml"
    unit.write_text("version: 1\nseed_px: 20\nreach_px: 70\nborder_px: 30\n"
                    "area_px: [3000, 12000]\nscale: 0.5\n")
    split = root / "split.yml"
    roles = {"C-1.mkv": "calibrate", "T-1.mkv": "train", "E-1.mkv": "test"}
    write_split(Split({v: (r, file_sha256(footage / v)) for v, r in roles.items()}), split)
    return {"footage": footage, "labels": labels, "lane": lane, "unit": unit, "split": split}


def _score(tmp_path: Path, scorers: Mapping[CupState, tuple[Scorer, int]]) -> list[ScoreRow]:
    paths = make_rig(tmp_path)
    labels = load_labels(paths["labels"], "L")
    return score_block(labels, paths["footage"], ["C-1.mkv"], LANE, SETTINGS, scorers)


def test_a_block_gives_one_row_per_full_pass_and_a_fault_row_for_a_failed_cut(
    tmp_path: Path,
) -> None:
    rows = _score(tmp_path, {CupState.FULL_PINK: (CentreScorer(), 32)})
    assert [(r.state, r.frame, r.order) for r in rows] == [
        ("full_pink", 50, 1), ("full_white", 114, 3), ("full", 146, 4)]
    assert [r.used_frame for r in rows] == [50, 114, None]
    assert [r.fault for r in rows] == ["", "", "unit size out of range"]
    last = rows[2]
    assert (last.colour_spread, last.median_l, last.median_a, last.median_b, last.patch_score) == (
        None, None, None, None, None)


def test_a_state_without_a_patchcore_has_colour_scores_and_no_patch_score(
    tmp_path: Path,
) -> None:
    pink, white, _ = _score(tmp_path, {CupState.FULL_PINK: (CentreScorer(), 32)})
    assert pink.patch_score == 7.5
    assert white.patch_score is None
    assert white.colour_spread == pytest.approx(0.0, abs=1.0)
    assert (white.median_l, white.median_a, white.median_b) == pytest.approx(
        (92.5, 129.0, 128.0), abs=1.0)


def test_the_scores_are_written_to_a_csv_with_empty_cells_for_no_answer(
    tmp_path: Path,
) -> None:
    rows = _score(tmp_path, {CupState.FULL_PINK: (CentreScorer(), 32)})
    out = tmp_path / "scores.csv"
    write_scores(rows, out)
    table = list(csv.reader(out.open(encoding="utf-8")))
    assert table[0] == ["video", "frame", "order", "state", "used_frame", "fault",
                        "colour_spread", "median_l", "median_a", "median_b", "patch_score"]
    assert table[3][:6] == ["C-1.mkv", "146", "4", "full", "", "unit size out of range"]
    assert table[3][6:] == ["", "", "", "", ""]
    assert table[1][10] == "7.5"
    assert table[2][10] == ""
