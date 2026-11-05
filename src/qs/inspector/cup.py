"""The cup finder: a frame in, the one cup wholly inside the inspection zone out, or None.

`find_cup(pixels, LineConfig)` looks only at edges and shape, so an empty cup and a full one are
found the same way, whatever colour the моти is: grey, a 5 × 5 Gaussian, Canny, one ellipse
fitted to each traced edge. An ellipse is the rim when both its axes are in `rim_axis_px`, all
of it is inside `zone`, and edges lie along most of it, so a rim may be broken by glare. Shape
is all it knows: a round glare patch of a rim's size passes for a rim. Of those the largest
wins: the outline of a моти lies inside its cup's rim. Pure: the input is only ever read.

A cup partly outside the zone is not returned at all, so a neighbouring cup cut by the zone's
edge can never hide the cup wholly inside it. These numbers have only been tried on drawn
frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qs.config import LineConfig

Ellipse = tuple[tuple[float, float], tuple[float, float], float]

_BLUR_KSIZE = (5, 5)
_CANNY = (30, 90)
_MIN_EDGE_POINTS = 20  # a shorter edge fixes no ellipse worth trying
_EDGE_TOLERANCE_KSIZE = (5, 5)  # an edge within 2 px counts as lying on the ellipse
_RIM_EDGE_SHARE_MIN = 0.75
_INSIDE_SCALE = 0.85  # the inside stops short of the rim, so it never holds the rim's own pixels


@dataclass(frozen=True, slots=True)
class Cup:
    """The cup found in a frame, everything in frame coordinates.

    `rim` is the ellipse as `cv2.fitEllipse` gives it: ((cx, cy), (axis, axis), angle), with
    full axis lengths. `box` is the rim's (x, y, w, h). `inside` is frame-sized (H×W), bool and
    read-only.
    """

    rim: Ellipse
    box: tuple[int, int, int, int]
    inside: np.ndarray


def _outline(ellipse: Ellipse) -> np.ndarray:
    (cx, cy), (first, second), angle = ellipse
    return np.asarray(
        cv2.ellipse2Poly(
            (round(cx), round(cy)), (round(first / 2), round(second / 2)), round(angle), 0, 360, 2
        )
    )


def _in_zone(outline: np.ndarray, zone: tuple[int, int, int, int]) -> bool:
    x, y, w, h = zone
    xs, ys = outline[:, 0], outline[:, 1]
    return bool(xs.min() >= x and ys.min() >= y and xs.max() < x + w and ys.max() < y + h)


def _edge_share(outline: np.ndarray, near_edge: np.ndarray) -> float:
    height, width = near_edge.shape
    xs = np.clip(outline[:, 0], 0, width - 1)
    ys = np.clip(outline[:, 1], 0, height - 1)
    return float(np.count_nonzero(near_edge[ys, xs])) / len(outline)


def _inside(rim: Ellipse, shape: tuple[int, int]) -> np.ndarray:
    (cx, cy), (first, second), angle = rim
    half_axes = (round(first / 2 * _INSIDE_SCALE), round(second / 2 * _INSIDE_SCALE))
    filled = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(filled, (round(cx), round(cy)), half_axes, angle, 0, 360, 255, cv2.FILLED)
    mask = filled.astype(bool)
    mask.setflags(write=False)
    return mask


def find_cup(pixels: np.ndarray, config: LineConfig) -> Cup | None:
    """The cup wholly inside `config.zone`, or None when no rim qualifies."""
    grey = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, _BLUR_KSIZE, 0), *_CANNY)
    near_edge = cv2.dilate(edges, np.ones(_EDGE_TOLERANCE_KSIZE, dtype=np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    low, high = config.rim_axis_px
    best: Ellipse | None = None
    for contour in contours:
        if len(contour) < _MIN_EDGE_POINTS:
            continue
        (cx, cy), (first, second), angle = cv2.fitEllipse(contour)
        ellipse: Ellipse = ((cx, cy), (first, second), angle)
        if not (low <= first <= high and low <= second <= high):
            continue
        outline = _outline(ellipse)
        if not _in_zone(outline, config.zone):
            continue
        if _edge_share(outline, near_edge) < _RIM_EDGE_SHARE_MIN:
            continue
        if best is None or first * second > best[1][0] * best[1][1]:
            best = ellipse
    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(_outline(best))
    return Cup(rim=best, box=(x, y, w, h), inside=_inside(best, grey.shape))
