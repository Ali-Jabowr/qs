"""The Localizer: a frame in, the unit's filled silhouette out (docs/15 §2).

`localize(pixels, LocalizeConfig)` runs the chain of docs/15 §2 — crop to the localize ROI,
grey, a 5 × 5 Gaussian, the fixed config threshold (a pixel is unit when it is above it),
open then close with a 5 × 5 ellipse, external contours of at least `min_area_px`, traced
over the full boundary (`CHAIN_APPROX_NONE`, so a straight-sided outline still carries the
points an ellipse is fitted to) — and
draws the one remaining contour filled into a frame-sized mask, so interior holes (the dark
filling of a tear) are part of the unit. Zero counted contours, or two or more, or one whose
bounding box is closer than `border_margin_px` to any ROI edge, is a `PlacementFault` naming
which rule broke (FR-25). Pure: the input is only ever read, never written.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qs.config import LocalizeConfig
from qs.errors import PlacementFault

_BLUR_KSIZE = (5, 5)
_KERNEL_KSIZE = (5, 5)


@dataclass(frozen=True, slots=True)
class Silhouette:
    """The unit found in a frame, everything in frame coordinates.

    `mask` is frame-sized (H×W), bool and read-only; `contour` is the outer contour;
    `unit_box` is (x, y, w, h); `centroid` is (cx, cy), from the filled mask's moments.
    """

    mask: np.ndarray
    contour: np.ndarray
    unit_box: tuple[int, int, int, int]
    centroid: tuple[float, float]


def localize(pixels: np.ndarray, config: LocalizeConfig) -> Silhouette:
    """Find the one unit in a frame: its filled silhouette, or a `PlacementFault` (FR-25).

    The chain is exactly docs/15 §2, inside `inspector.localize.roi` only. The faults, in
    the order the placement rules are checked: "no unit", "more than one unit", "not wholly
    in view" — the last one when the unit's bounding box is closer than `border_margin_px`
    to any ROI edge; a gap of exactly the margin is accepted.
    """
    rx, ry, rw, rh = config.roi
    crop = pixels[ry : ry + rh, rx : rx + rw]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, _BLUR_KSIZE, 0)
    _, binary = cv2.threshold(blurred, config.threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _KERNEL_KSIZE)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(rx, ry)
    )
    counted = [contour for contour in contours if cv2.contourArea(contour) >= config.min_area_px]
    if not counted:
        raise PlacementFault("no unit")
    if len(counted) > 1:
        raise PlacementFault("more than one unit")
    contour = counted[0]
    x, y, w, h = cv2.boundingRect(contour)
    margin = config.border_margin_px
    if (
        x - rx < margin
        or y - ry < margin
        or rx + rw - (x + w) < margin
        or ry + rh - (y + h) < margin
    ):
        raise PlacementFault("not wholly in view")
    filled = np.zeros(pixels.shape[:2], dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    moments = cv2.moments(filled)
    mask = filled.astype(bool)
    mask.setflags(write=False)
    return Silhouette(
        mask=mask,
        contour=contour,
        unit_box=(x, y, w, h),
        centroid=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
    )
