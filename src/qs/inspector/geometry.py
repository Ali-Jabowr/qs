"""The geometry check: four measures of the silhouette, checked against the tolerances (docs/15 §3).

`measure(silhouette)` reads the four numbers of docs/15 §3 off the filled mask and its outer
contour — `area_px`, `diameter_px` (reported for people, never tested), `axis_ratio` as the
minor over the major axis of `cv2.fitEllipse` on the contour, and `solidity` as the contour
area over the convex-hull area. Fewer than 5 contour points, a failed ellipse fit, a zero
hull area or any non-finite value raises `InferenceError` (C8): a measure that cannot be
computed never passes. `check_geometry(measures, config)` turns the four numbers into the
geometry `CheckResult` — passed exactly when all three rules hold, inclusively, `failed_on`
naming every failing rule, comma-separated, in the fixed order. Pure: the silhouette is only
ever read, never written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from qs.config import GeometryConfig
from qs.contracts import CHECK_GEOMETRY, CheckResult
from qs.errors import InferenceError
from qs.inspector.localize import Silhouette

_MIN_CONTOUR_POINTS = 5


@dataclass(frozen=True, slots=True)
class GeometryMeasures:
    """The four geometry numbers of one silhouette (docs/15 §3)."""

    area_px: int
    diameter_px: float
    axis_ratio: float
    solidity: float

    @property
    def measures(self) -> tuple[tuple[str, float], ...]:
        """The ordered measures tuple of the contract: area, diameter, axis ratio, solidity."""
        return (
            ("area_px", float(self.area_px)),
            ("diameter_px", self.diameter_px),
            ("axis_ratio", self.axis_ratio),
            ("solidity", self.solidity),
        )


def measure(silhouette: Silhouette) -> GeometryMeasures:
    """Measure one silhouette by docs/15 §3, or raise `InferenceError` (C8)."""
    contour = silhouette.contour
    if len(contour) < _MIN_CONTOUR_POINTS:
        raise InferenceError(
            f"the contour has {len(contour)} points; an ellipse fit needs at least "
            f"{_MIN_CONTOUR_POINTS}"
        )
    area_px = int(np.count_nonzero(silhouette.mask))
    diameter_px = math.sqrt(4.0 * area_px / math.pi)
    try:
        _, axes, _ = cv2.fitEllipse(contour)
    except cv2.error as exc:
        raise InferenceError("the ellipse fit failed on this contour") from exc
    major_axis = max(axes)
    if not math.isfinite(major_axis) or major_axis <= 0.0:
        raise InferenceError("the fitted ellipse has a zero or non-finite major axis")
    axis_ratio = min(axes) / major_axis
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0.0:
        raise InferenceError("the convex hull of this contour has no area")
    solidity = cv2.contourArea(contour) / hull_area
    if not all(
        math.isfinite(value)
        for value in (float(area_px), diameter_px, axis_ratio, solidity)
    ):
        raise InferenceError("a geometry measure is not finite")
    return GeometryMeasures(
        area_px=area_px,
        diameter_px=diameter_px,
        axis_ratio=axis_ratio,
        solidity=solidity,
    )


def check_geometry(measures: GeometryMeasures, config: GeometryConfig) -> CheckResult:
    """The geometry `CheckResult` against `threshold.geometry`: every rule holds or it fails."""
    if not all(math.isfinite(value) for _, value in measures.measures):
        raise InferenceError("a geometry measure is not finite")
    failed: list[str] = []
    if measures.area_px < config.area_px[0]:
        failed.append("area_px<min")
    if measures.area_px > config.area_px[1]:
        failed.append("area_px>max")
    if measures.axis_ratio < config.axis_ratio_min:
        failed.append("axis_ratio<min")
    if measures.solidity < config.solidity_min:
        failed.append("solidity<min")
    return CheckResult(
        name=CHECK_GEOMETRY,
        passed=not failed,
        value=None,
        limit=None,
        measures=measures.measures,
        failed_on=", ".join(failed) if failed else None,
    )
