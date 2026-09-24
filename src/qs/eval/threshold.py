"""The threshold rule (docs/14 §4 step 4, docs/15 §7 step 3; FR-54).

`choose_threshold` is pure integer arithmetic, shared by `qs calibrate` (one target) and
`qs evaluate` (1, 5 and 10). With `n` calibration units, `a = ⌊target × n / 100⌋` flags are
allowed and `g` units are already geometry-flagged. `g > a` is not reachable at this target
(assumption A2, #78): the tolerances are the problem, not the threshold. Otherwise the
threshold is the `(a − g + 1)`-th highest appearance score among the geometry-passed units —
the lowest value at which no more than `a − g` of them lie strictly above it, since a unit
passes exactly when its score is at or below the threshold (#49). Ties therefore only lower
the combined count below the budget, never above it. A unit's score is the maximum over its
rotations; the caller aggregates, this function knows no rotations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnitScore:
    """One calibration unit: its appearance score and whether geometry flagged it."""

    unit: str
    score: float
    geometry_failed: bool


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """What the rule chose at one target.

    `value` is None when geometry alone already flags more than `a` units; `flagged` then
    carries the geometry-flagged units. Otherwise `flagged` carries every combined-flagged
    unit — geometry-flagged, above the threshold, or both — and `k` counts them.
    """

    value: float | None
    n: int
    a: int
    g: int
    k: int
    below_one_over_n: bool
    flagged: tuple[str, ...]

    @property
    def reachable(self) -> bool:
        return self.value is not None


def choose_threshold(units: Sequence[UnitScore], target_pct: int) -> ThresholdChoice:
    """The threshold at `target_pct`, or a not-reachable result carrying the flagged units."""
    if not units:
        raise ValueError("no units given: the threshold is chosen on at least one unit")
    n = len(units)
    a = target_pct * n // 100
    below_one_over_n = target_pct * n < 100
    failed = sorted(u.unit for u in units if u.geometry_failed)
    g = len(failed)
    if g > a:
        return ThresholdChoice(
            value=None, n=n, a=a, g=g, k=g, below_one_over_n=below_one_over_n,
            flagged=tuple(failed),
        )
    passed = sorted((u for u in units if not u.geometry_failed), key=lambda u: u.score,
                    reverse=True)
    value = passed[a - g].score
    above = sorted(u.unit for u in passed if u.score > value)
    return ThresholdChoice(
        value=value, n=n, a=a, g=g, k=g + len(above), below_one_over_n=below_one_over_n,
        flagged=tuple(sorted((*failed, *above))),
    )
