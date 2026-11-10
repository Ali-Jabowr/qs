"""The cup finder: a frame in, the one cup wholly inside the inspection zone out, or None.

`find_cup(pixels, LineConfig)` looks only at edges and shape, so an empty cup and a full one are
found the same way, whatever colour the моти is: grey, a 5 × 5 Gaussian, Canny, one ellipse
fitted to each traced edge. An ellipse is the rim when both its axes are in `rim_axis_px`, all
of it is inside `zone`, and edges lie along most of it, so a rim may be broken by glare. With
`rim_size_px` set each axis must also be within 5% of its own measured value. Shape is all it
knows: a round glare patch of a rim's size passes for a rim. Of those the largest wins: the
outline of a моти lies inside its cup's rim. Pure: the input is only ever read.

A cup partly outside the zone is not returned at all, so a neighbouring cup cut by the zone's
edge can never hide the cup wholly inside it. These numbers have only been tried on drawn
frames.

`find_rims(pixels, LineConfig)` applies the same rim rules without the zone: every cup in the
frame, each once, saying whether it is wholly in the zone. The cup trigger follows cups with it.

`no_cup_reason(pixels, LineConfig)` says why `find_cup` found nothing. When every traced ellipse
failed `rim_axis_px`, it names the size and the largest one's axes, so a camera remounted at
another scale is not logged as a cup out of position. A round glare patch counts here too.
"""

from __future__ import annotations

from collections.abc import Callable
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


@dataclass(frozen=True, slots=True)
class Rim:
    """One rim seen in a frame: `ellipse` as on `Cup`, and whether all of it is in the zone."""

    ellipse: Ellipse
    in_zone: bool


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


def inside_of(
    rim: Ellipse, shape: tuple[int, int], scale: float = _INSIDE_SCALE
) -> np.ndarray:
    """The frame-sized read-only mask of the rim's ellipse shrunk to `scale` of its axes."""
    (cx, cy), (first, second), angle = rim
    half_axes = (round(first / 2 * scale), round(second / 2 * scale))
    filled = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(filled, (round(cx), round(cy)), half_axes, angle, 0, 360, 255, cv2.FILLED)
    mask = filled.astype(bool)
    mask.setflags(write=False)
    return mask


def _traced_ellipses(
    pixels: np.ndarray, sized: Callable[[Ellipse], bool] = lambda _: True
) -> list[Ellipse]:
    """Every traced edge that fits an ellipse `sized` accepts, with edges along most of it.
    `sized` runs before the edge share, which is the costly test, so most edges skip it."""
    grey = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, _BLUR_KSIZE, 0), *_CANNY)
    near_edge = cv2.dilate(edges, np.ones(_EDGE_TOLERANCE_KSIZE, dtype=np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ellipses: list[Ellipse] = []
    for contour in contours:
        if len(contour) < _MIN_EDGE_POINTS:
            continue
        (cx, cy), (first, second), angle = cv2.fitEllipse(contour)
        ellipse: Ellipse = ((cx, cy), (first, second), angle)
        if not sized(ellipse):
            continue
        if _edge_share(_outline(ellipse), near_edge) < _RIM_EDGE_SHARE_MIN:
            continue
        ellipses.append(ellipse)
    return ellipses


_RIM_SIZE_TOLERANCE = 0.05  # ADR-0029 §3: a rim within ±5% of the size measured on the mount


def _rim_sized(ellipse: Ellipse, config: LineConfig) -> bool:
    low, high = config.rim_axis_px
    _, axes, _ = ellipse
    if not all(low <= axis <= high for axis in axes):
        return False
    if config.rim_size_px is None:
        return True
    return all(
        size * (1 - _RIM_SIZE_TOLERANCE) <= axis <= size * (1 + _RIM_SIZE_TOLERANCE)
        for axis, size in zip(sorted(axes), config.rim_size_px, strict=True)
    )


def _rim_ellipses(pixels: np.ndarray, config: LineConfig) -> list[Ellipse]:
    """Every traced edge that passes the rim rules, wherever it is. A full cup gives several: its
    rim's two edges, and often the outline of the моти inside."""
    return _traced_ellipses(pixels, lambda ellipse: _rim_sized(ellipse, config))


def _area(ellipse: Ellipse) -> float:
    return ellipse[1][0] * ellipse[1][1]


def find_cup(pixels: np.ndarray, config: LineConfig) -> Cup | None:
    """The cup wholly inside `config.zone`, or None when no rim qualifies."""
    best: Ellipse | None = None
    for ellipse in _rim_ellipses(pixels, config):
        if not _in_zone(_outline(ellipse), config.zone):
            continue
        if best is None or _area(ellipse) > _area(best):
            best = ellipse
    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(_outline(best))
    return Cup(rim=best, box=(x, y, w, h), inside=inside_of(best, pixels.shape[:2]))


def no_cup_reason(pixels: np.ndarray, config: LineConfig) -> str:
    """Why `find_cup` found no cup in `pixels`, for the log of a frame it returned None for.

    When ellipses were traced but none is the size of `rim_axis_px`, the reason says so with the
    largest one's axes, as a remounted camera sees cups at another scale. Otherwise the reason is
    position, or no rim at all, as before.
    """
    reason = "no cup wholly in the zone"
    traced = _traced_ellipses(pixels)
    if not traced or any(_rim_sized(e, config) for e in traced):
        return reason
    low, high = config.rim_axis_px
    _, (first, second), _ = max(traced, key=_area)
    sized = (
        f"rim_axis_px {low}–{high}" if config.rim_size_px is None
        else f"rim_size_px {config.rim_size_px[0]}×{config.rim_size_px[1]} ±5%"
    )
    return (
        f"{reason}: no rim the size of {sized}, "
        f"the largest seen is {first:.0f}×{second:.0f} px"
    )


def find_rims(pixels: np.ndarray, config: LineConfig) -> tuple[Rim, ...]:
    """Every cup in the frame, in or out of `config.zone`, each once: an ellipse centred inside a
    larger one belongs to the larger one's cup."""
    kept: list[Ellipse] = []
    for ellipse in sorted(_rim_ellipses(pixels, config), key=_area, reverse=True):
        (cx, cy), _, _ = ellipse
        if not any(
            cv2.pointPolygonTest(_outline(larger).astype(np.float32), (cx, cy), False) >= 0
            for larger in kept
        ):
            kept.append(ellipse)
    return tuple(Rim(e, _in_zone(_outline(e), config.zone)) for e in kept)
