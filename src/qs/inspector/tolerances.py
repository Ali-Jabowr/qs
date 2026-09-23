"""The derived tolerances and the pilot's two numbers (docs/15 §3).

`derive_tolerances(per_unit)` turns per-unit, per-rotation measures of good units into a
`GeometryConfig` plus the report that explains it: μᵤ is each unit's mean over its rotations,
σₙ the pooled within-unit standard deviation √(Σᵤ Σᵣ (xᵤᵣ − μᵤ)² / Σᵤ (nᵤ − 1)) — the
measurement noise — and every tolerance the observed range of unit means widened by 3σₙ,
the area range rounded outward. A derivation that would produce an invalid tolerance is
refused with the unit or the measure named. `rotation_spread` gives each unit's area spread
across rotations for the pilot's silhouette criterion (docs/13 §5). `suggest_threshold` pools
the 5 × 5-blurred grey ROI of many frames into one 256-bin histogram and computes Otsu's
value on it once — a suggestion, never a written config (docs/15 §2). Pure: nothing here
reads the clock, the disk or the network, and the inputs are only ever read, never written.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import cv2
import numpy as np

from qs.config import GeometryConfig
from qs.inspector.geometry import GeometryMeasures

_BLUR_KSIZE = (5, 5)
_BINS = 256
_MEASURES = ("area_px", "axis_ratio", "solidity")

_VALUES: Mapping[str, Callable[[GeometryMeasures], float]] = {
    "area_px": lambda m: float(m.area_px),
    "axis_ratio": lambda m: m.axis_ratio,
    "solidity": lambda m: m.solidity,
}


@dataclass(frozen=True, slots=True)
class Tolerances:
    """The derived geometry tolerances and the report that explains them (docs/15 §3).

    `config` is what the calibration writes into `threshold.geometry`; `units` and `frames`
    count what it was derived from; `means` is μᵤ per unit and measure; `sigma` is σₙ and
    `mean_range` the (min, max) of the unit means, per measure. All three mappings are
    read-only and keyed in sorted unit order, so the same input gives the same report.
    """

    config: GeometryConfig
    units: int
    frames: int
    means: Mapping[str, Mapping[str, float]]
    sigma: Mapping[str, float]
    mean_range: Mapping[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class ThresholdSuggestion:
    """A suggested localize threshold and the pooled histogram it was computed from."""

    value: float
    histogram: np.ndarray


def _pooled(
    per_unit: Mapping[str, Sequence[GeometryMeasures]],
) -> tuple[int, int, dict[str, float], dict[str, dict[str, float]]]:
    """Frames, Σ(nᵤ − 1), the per-measure pooled σₙ and μᵤ, accumulated in sorted unit order."""
    denominator = 0
    squared: dict[str, float] = dict.fromkeys(_MEASURES, 0.0)
    means: dict[str, dict[str, float]] = {}
    frames = 0
    for unit_id in sorted(per_unit):
        rotations = per_unit[unit_id]
        frames += len(rotations)
        denominator += len(rotations) - 1
        means[unit_id] = {}
        for name in _MEASURES:
            values = [_VALUES[name](m) for m in rotations]
            mean = sum(values) / len(values)
            means[unit_id][name] = mean
            squared[name] += sum((value - mean) ** 2 for value in values)
    sigma = {name: math.sqrt(squared[name] / denominator) for name in _MEASURES}
    return frames, denominator, sigma, means


def derive_tolerances(
    per_unit: Mapping[str, Sequence[GeometryMeasures]],
) -> Tolerances:
    """The geometry tolerances derived from good units, never typed (docs/15 §3, FR-54).

    For each of `area_px`, `axis_ratio` and `solidity`: μᵤ is the unit's mean over its
    rotations and σₙ the pooled within-unit standard deviation, with the denominator
    Σ(nᵤ − 1). The area range is [⌊min μᵤ − 3σₙ⌋, ⌈max μᵤ + 3σₙ⌉], rounded outward so the
    integer range contains the real one; the two minimums are min μᵤ − 3σₙ. Refused with a
    `ValueError` naming the unit or the measure: an empty input, a unit measured in fewer
    than 2 rotations, a spread that drives `axis_ratio_min` or `solidity_min` to ≤ 0, an
    area minimum ≤ 0. The caller chooses the units; this function knows no splits.
    """
    if not per_unit:
        raise ValueError(
            "no units given: the tolerances are derived from at least one unit "
            "measured in at least two rotations"
        )
    for unit_id in sorted(per_unit):
        if len(per_unit[unit_id]) < 2:
            raise ValueError(
                f"unit {unit_id}: measured in {len(per_unit[unit_id])} rotation(s); "
                "at least 2 are needed to estimate the measurement noise"
            )
    frames, _, sigma, means = _pooled(per_unit)
    mean_range = {
        name: (
            min(means[unit_id][name] for unit_id in means),
            max(means[unit_id][name] for unit_id in means),
        )
        for name in _MEASURES
    }
    axis_min = mean_range["axis_ratio"][0] - 3.0 * sigma["axis_ratio"]
    if axis_min <= 0.0:
        raise ValueError(
            f"axis_ratio_min: {axis_min} is not above 0 — the spread of axis_ratio "
            "across rotations is too wide for a tolerance"
        )
    solidity_min = mean_range["solidity"][0] - 3.0 * sigma["solidity"]
    if solidity_min <= 0.0:
        raise ValueError(
            f"solidity_min: {solidity_min} is not above 0 — the spread of solidity "
            "across rotations is too wide for a tolerance"
        )
    area_low = math.floor(mean_range["area_px"][0] - 3.0 * sigma["area_px"])
    if area_low <= 0:
        raise ValueError(f"area_px: the derived minimum {area_low} is not above 0")
    area_high = math.ceil(mean_range["area_px"][1] + 3.0 * sigma["area_px"])
    config = GeometryConfig(
        area_px=(area_low, area_high),
        axis_ratio_min=axis_min,
        solidity_min=solidity_min,
    )
    return Tolerances(
        config=config,
        units=len(per_unit),
        frames=frames,
        means=MappingProxyType(
            {unit_id: MappingProxyType(means[unit_id]) for unit_id in sorted(means)}
        ),
        sigma=MappingProxyType(sigma),
        mean_range=MappingProxyType(mean_range),
    )


def rotation_spread(
    per_unit: Mapping[str, Sequence[GeometryMeasures]],
) -> dict[str, float]:
    """Each unit's area spread across its rotations: (max − min) / mean of `area_px`.

    The pilot's silhouette criterion is that this stays under 2 % for every good unit
    (docs/13 §5). A unit measured in fewer than 2 rotations is refused, and so is a unit
    whose mean area is not above 0, where the spread is undefined.
    """
    spread: dict[str, float] = {}
    for unit_id in sorted(per_unit):
        rotations = per_unit[unit_id]
        if len(rotations) < 2:
            raise ValueError(
                f"unit {unit_id}: measured in {len(rotations)} rotation(s); "
                "a spread needs at least 2"
            )
        areas = [float(m.area_px) for m in rotations]
        mean = sum(areas) / len(areas)
        if mean <= 0.0:
            raise ValueError(
                f"unit {unit_id}: the mean of area_px over its rotations is {mean}; "
                "the spread is undefined"
            )
        spread[unit_id] = (max(areas) - min(areas)) / mean
    return spread


def _otsu(hist: np.ndarray) -> float:
    """Otsu's threshold from one 256-bin histogram: the split with the widest gap.

    The value t splits the bins into [0..t] and [t+1..255]; the t maximizing the
    between-class variance q₀·q₁·(μ₀ − μ₁)² wins, the first on a tie. A histogram with
    fewer than two occupied classes has no split and is refused.
    """
    total = int(hist.sum())
    if total <= 0:
        raise ValueError("the pooled histogram is empty: no pixels were pooled")
    if int(np.count_nonzero(hist)) < 2:
        raise ValueError(
            "the pooled histogram has one occupied bin; Otsu's split is undefined"
        )
    left_weight = 0
    left_sum = 0
    weighted_sum = sum(int(hist[i]) * i for i in range(_BINS))
    best: tuple[float, float] | None = None
    for t in range(_BINS - 1):
        left_weight += int(hist[t])
        left_sum += t * int(hist[t])
        right_weight = total - left_weight
        if left_weight == 0 or right_weight == 0:
            continue
        mu_left = left_sum / left_weight
        mu_right = (weighted_sum - left_sum) / right_weight
        between = (left_weight / total) * (right_weight / total) * (mu_left - mu_right) ** 2
        if best is None or between > best[0]:
            best = (between, float(t))
    assert best is not None
    return best[1]


def suggest_threshold(
    frames: Sequence[np.ndarray], roi: tuple[int, int, int, int]
) -> ThresholdSuggestion:
    """A suggested localize threshold from the pooled histogram of many frames (docs/15 §2).

    Every frame's ROI is cropped, greyed, blurred 5 × 5 and pooled into one 256-bin
    histogram; Otsu's value is computed on that pooled histogram exactly once. It suggests —
    the config is written by a person, and the value is recorded by the config version. A
    frame's own histogram is never taken. Empty frames are refused.
    """
    if not frames:
        raise ValueError(
            "no frames given: the suggested threshold is pooled over the pilot frames"
        )
    rx, ry, rw, rh = roi
    pooled = np.zeros(_BINS, dtype=np.int64)
    for frame in frames:
        grey = cv2.cvtColor(frame[ry : ry + rh, rx : rx + rw], cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, _BLUR_KSIZE, 0)
        pooled += np.bincount(blurred.ravel().astype(np.intp), minlength=_BINS)
    pooled.setflags(write=False)
    return ThresholdSuggestion(value=_otsu(pooled), histogram=pooled)
