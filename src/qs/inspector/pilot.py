"""The pilot report's pure arithmetic: separation, the window suggestion, the ranking.

`separation` turns one geometry's leave-one-out scores into gap, spread and separation
(docs/13 §5): gap is the lowest damaged score over all folds minus the highest held-out good
score, spread is the sample standard deviation (n − 1) of the held-out good scores, and
separation = gap / spread. `suggest_window` is the pilot's `window_px = ⌈2 × median
diameter_px⌉` of the good frames (docs/15 §4). `rank` orders the geometries that pass the
silhouette by separation and applies ADR-0013's tie rule: a tie — equal to two decimals —
goes to G2, or to G3 when the session compares G1 and G3 only. Pure: no clock, no disk, no
network; the inputs are only ever read, never written.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Separation:
    """One geometry's leave-one-out numbers: gap, spread, separation and the fold count n."""

    gap: float
    spread: float
    separation: float
    n: int


def separation(
    good_scores: Sequence[float], damaged_by_fold: Sequence[Sequence[float]]
) -> Separation:
    """Gap, spread and separation from the leave-one-out scores (docs/13 §5).

    `good_scores[i]` is fold i's held-out good unit's score, its maximum over rotations;
    `damaged_by_fold[i]` lists the damaged units' scores scored against the same fold's fit.
    Gap is the lowest damaged score over all folds minus the highest held-out good score,
    spread the sample standard deviation (n − 1) of the good scores, separation = gap / spread.
    Refused with a `ValueError`: fewer than 2 folds, where no spread exists, or a fold with
    no damaged score, where the gap would quietly shrink. Zero spread with a positive gap is
    a perfect separation (inf); with a negative gap, −inf — it must not rank above a
    geometry whose gap is less negative but has a real spread; with no gap, 0.
    """
    if len(good_scores) < 2:
        raise ValueError(
            f"{len(good_scores)} held-out good score(s); a spread needs at least 2 folds"
        )
    for index, damaged in enumerate(damaged_by_fold):
        if not damaged:
            raise ValueError(f"fold {index}: no damaged score — the gap would quietly shrink")
    gap = min(min(damaged) for damaged in damaged_by_fold) - max(good_scores)
    spread = statistics.stdev(good_scores)
    if spread == 0.0:
        perfect = math.inf if gap > 0.0 else -math.inf if gap < 0.0 else 0.0
        return Separation(gap, spread, perfect, len(good_scores))
    return Separation(gap, spread, gap / spread, len(good_scores))


def suggest_window(diameters: Sequence[float]) -> tuple[float, int]:
    """The median `diameter_px` of the good frames and `window_px = ⌈2 × median⌉` (docs/15 §4).

    Refused with a `ValueError` when no diameters are given.
    """
    if not diameters:
        raise ValueError("no diameters given: the window is suggested from the good frames")
    median = statistics.median(diameters)
    return median, math.ceil(2.0 * median)


def rank(separations: Mapping[str, float], geometries: Sequence[str]) -> list[str]:
    """The geometries that pass the silhouette, by separation, with the tie rule applied.

    Highest separation first; geometries whose separations are equal to two decimals are
    ordered by ADR-0013's tie rule — G2 wins the tie, or G3 when the session compares G1 and
    G3 only. Any tie the rule does not name falls back to the session's geometry order.
    """
    one_lamp = set(geometries) == {"G1", "G3"}
    preference = ("G3", "G1") if one_lamp else ("G2", "G1", "G3")
    order = {geometry: index for index, geometry in enumerate(geometries)}
    return sorted(
        separations,
        key=lambda g: (-round(separations[g], 2), preference.index(g), order[g]),
    )
