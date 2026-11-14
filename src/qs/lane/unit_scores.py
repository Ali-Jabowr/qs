"""The off-colour scores of a cut-out unit (#175): colour spread, whole-unit colour and PatchCore.

Off-colour means any colour not usual for a good unit. Three reads, all inside the cut-out of
`qs.lane.unit`: `colour_spread` (a unit against itself: a patch), `unit_colour` with `colour_shift`
(a unit against the usual colour of good units: an evenly off-colour unit passes the spread), and
`patch_score` (PatchCore's anomaly map, highest inside the unit). This module only measures; the
reference colour and the limits come from block C.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from qs.cmd.train_lane_b import _FULL
from qs.errors import InferenceError, PlacementFault
from qs.inspector.model import Scorer
from qs.inspector.window import to_input
from qs.labels import CupState, Labels, VideoFrames
from qs.lane.lanefile import Lane, crop
from qs.lane.unit import UnitSettings, best_cut

_ERODE_PX = 3
_GLARE = 245
_MIN_PIXELS = 50
_NEIGHBOURS = 2


@dataclass(frozen=True, slots=True)
class ScoreRow:
    """One full pass: the frame its cut-out came from, and its scores. Scores are None, and
    `fault` names the reason, when the pass had no cut-out; `patch_score` is also None when the
    pass's state has no PatchCore bank."""

    video: str
    frame: int
    order: int
    state: str
    used_frame: int | None
    fault: str
    colour_spread: float | None
    median_l: float | None
    median_a: float | None
    median_b: float | None
    patch_score: float | None


def _unit_pixels(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The Lab pixels (N x 3, 8-bit) of `mask` eroded by 3 px, glare left out; under 50 raise."""
    size = 2 * _ERODE_PX + 1
    inside = cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
    inside &= ~(crop_bgr >= _GLARE).all(axis=2)
    if int(inside.sum()) < _MIN_PIXELS:
        raise InferenceError(
            f"fewer than {_MIN_PIXELS} unit pixels left after the erosion and the glare"
        )
    lab: np.ndarray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    return np.asarray(lab[inside])


def colour_spread(crop: np.ndarray, mask: np.ndarray) -> float:
    """The mean distance of each unit pixel's (a, b) from the median (a, b); brightness is left
    out, so a unit of one colour in uneven light has no spread."""
    ab = _unit_pixels(crop, mask)[:, 1:].astype(np.float64)
    return float(np.linalg.norm(ab - np.median(ab, axis=0), axis=1).mean())


def unit_colour(crop: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """The median (L, a, b) of the unit pixels, L rescaled to 0-100, a and b in OpenCV's units."""
    lab_l, lab_a, lab_b = (float(v) for v in np.median(_unit_pixels(crop, mask), axis=0))
    return lab_l * 100 / 255, lab_a, lab_b


def colour_shift(
    colour: tuple[float, float, float], reference: tuple[float, float, float]
) -> float:
    """The Euclidean distance between two (L, a, b) colours (ΔE 1976)."""
    if not all(math.isfinite(v) for v in (*colour, *reference)):
        raise InferenceError(f"colour_shift: a colour is not finite: {colour}, {reference}")
    return math.dist(colour, reference)


def patch_score(anomaly_map: np.ndarray, mask: np.ndarray) -> float:
    """The highest anomaly-map value inside `mask`, which is brought to the map's shape."""
    height, width = anomaly_map.shape
    small = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    inside = anomaly_map[small > 0]
    if inside.size == 0:
        raise InferenceError(f"the unit mask is empty at the map's size {width}x{height}")
    score = float(inside.max()) if np.isfinite(inside).all() else math.nan
    if not math.isfinite(score):
        raise InferenceError("the anomaly map holds a value that is not finite inside the mask")
    return score


def _fault_row(video: str, frame: int, order: int, state: CupState, fault: str) -> ScoreRow:
    return ScoreRow(video, frame, order, state.value, None, fault, None, None, None, None, None)


def score_block(
    labels: Labels,
    footage: Path,
    videos: Sequence[str],
    lane: Lane,
    unit: UnitSettings,
    scorers: Mapping[CupState, tuple[Scorer, int]],
) -> list[ScoreRow]:
    """One row for every full pass of `videos`: the crops of frames f-2 ... f+2 go to `best_cut`,
    and the winning crop is scored. `scorers` maps a state to its PatchCore scorer and input
    size; a state absent from it gets no `patch_score`. A pass with no cut-out gets a row with
    the fault and no scores, never dropped."""
    rows: list[ScoreRow] = []
    for video in videos:
        reader = VideoFrames(footage / video)
        try:
            last = len(reader.ids) - 1
            for cup in labels.cups_of(video):
                if cup.state not in _FULL:
                    continue
                positions = range(max(0, cup.frame - _NEIGHBOURS),
                                  min(last, cup.frame + _NEIGHBOURS) + 1)
                crops = [crop(reader.read(p), lane) for p in positions]
                try:
                    index, cut = best_cut(crops, unit)
                except PlacementFault as fault:
                    rows.append(_fault_row(video, cup.frame, cup.order, cup.state, fault.detail))
                    continue
                best = crops[index]
                lab_l, lab_a, lab_b = unit_colour(best, cut.mask)
                patch = None
                if cup.state in scorers:
                    scorer, input_px = scorers[cup.state]
                    scored = scorer.score(to_input(best, input_px))
                    patch = patch_score(scored.anomaly_map, cut.mask)
                rows.append(ScoreRow(
                    video, cup.frame, cup.order, cup.state.value, positions[index], "",
                    colour_spread(best, cut.mask), lab_l, lab_a, lab_b, patch,
                ))
        finally:
            reader.close()
    return rows


def write_scores(rows: Sequence[ScoreRow], path: Path) -> None:
    """The rows as a CSV, a missing score as an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(field.name for field in fields(ScoreRow))
        for row in rows:
            writer.writerow("" if value is None else value for value in astuple(row))
