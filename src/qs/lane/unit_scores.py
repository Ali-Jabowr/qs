"""Stub: filled in after the tests are committed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qs.inspector.model import Scorer
from qs.labels import CupState, Labels
from qs.lane.lanefile import Lane
from qs.lane.unit import UnitSettings


@dataclass(frozen=True, slots=True)
class ScoreRow:
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


def colour_spread(crop: np.ndarray, mask: np.ndarray) -> float:
    raise NotImplementedError


def unit_colour(crop: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    raise NotImplementedError


def colour_shift(colour: tuple[float, float, float], reference: tuple[float, float, float]) -> float:
    raise NotImplementedError


def patch_score(anomaly_map: np.ndarray, mask: np.ndarray) -> float:
    raise NotImplementedError


def score_block(
    labels: Labels, footage: Path, videos: Sequence[str], lane: Lane, unit: UnitSettings,
    scorers: Mapping[CupState, tuple[Scorer, int]],
) -> list[ScoreRow]:
    raise NotImplementedError


def write_scores(rows: Sequence[ScoreRow], path: Path) -> None:
    raise NotImplementedError
