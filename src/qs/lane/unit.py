"""The cut-out: the pixels of the моти inside its cup, found in a lane-window crop (#174).

The unit is not found by colour likeness: a pale patch on a pink unit would be cut away by a
colour rule. GrabCut is seeded by geometry only — a disc at the crop's centre is surely unit,
nothing beyond `reach_px` is (sure background, so the cut can never leak past it), an edge band
is surely background — so a patch of any colour inside the outline stays inside the cut-out.

Choosing the values: put `reach_px` just inside the cup rim (a unit that reaches the reach circle
is "not wholly in view", never a clipped outline); `seed_px` about a third of the unit's radius;
`area_px` max well under the area inside the rim, so a leak up to the rim is refused as "unit
size out of range".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from qs.errors import ConfigError, PlacementFault
from qs.inspector.localize import Silhouette

_KEYS = {"version", "seed_px", "reach_px", "border_px", "area_px", "scale"}
_GRABCUT_ITERATIONS = 5


@dataclass(frozen=True, slots=True)
class UnitSettings:
    """`seed_px`: radius of the centre disc that is surely unit; `reach_px`: radius beyond which
    nothing is unit (a unit that reaches it faults); `border_px`: edge band that is surely
    background; `area_px`: the range of a unit's area in crop pixels; `scale`: the factor the crop
    is shrunk by for the cut."""

    seed_px: int
    reach_px: int
    border_px: int
    area_px: tuple[int, int]
    scale: float


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_unit(path: Path) -> UnitSettings:
    """The settings in `path`; a missing or extra key, or a value out of range, raises
    ConfigError naming the key."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: cannot read the unit file: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the unit file must be a mapping")
    for key in sorted(_KEYS - data.keys()):
        raise ConfigError(f"{path}: missing key {key!r}")
    for key in sorted(data.keys() - _KEYS, key=str):
        raise ConfigError(f"{path}: unknown key {key!r}")
    if data["version"] != 1:
        raise ConfigError(f"{path}: key 'version' must be 1, not {data['version']!r}")
    for key in ("seed_px", "reach_px", "border_px"):
        if not _is_int(data[key]) or data[key] <= 0:
            raise ConfigError(f"{path}: key {key!r} must be a whole number above 0, "
                              f"not {data[key]!r}")
    if data["seed_px"] >= data["reach_px"]:
        raise ConfigError(f"{path}: key 'seed_px' must be below 'reach_px': "
                          f"{data['seed_px']} and {data['reach_px']}")
    area = data["area_px"]
    if (
        not isinstance(area, list)
        or len(area) != 2
        or not all(_is_int(v) for v in area)
        or not 0 < area[0] <= area[1]
    ):
        raise ConfigError(f"{path}: key 'area_px' must be [min, max], 0 < min <= max: {area!r}")
    scale = data["scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, int | float)
        or not math.isfinite(scale)
        or not 0 < scale <= 1
    ):
        raise ConfigError(f"{path}: key 'scale' must be a number above 0 and at most 1, "
                          f"not {scale!r}")
    return UnitSettings(
        data["seed_px"], data["reach_px"], data["border_px"], (area[0], area[1]), float(scale)
    )


def _silhouette(filled: np.ndarray) -> Silhouette:
    """The `Silhouette` of a filled uint8 mask, built as `inspector.localize` builds its own."""
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(filled)
    mask = filled.astype(bool)
    mask.setflags(write=False)
    x, y, w, h = cv2.boundingRect(contour)
    return Silhouette(
        mask=mask,
        contour=contour,
        unit_box=(x, y, w, h),
        centroid=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
    )


def cut_unit(crop: np.ndarray, settings: UnitSettings) -> Silhouette:
    """The unit's filled silhouette in `crop` coordinates, or a `PlacementFault`.

    Faults: "no unit" (the centre is not foreground), "not wholly in view" (the unit reaches the
    border band), "unit size out of range" (its area is outside `area_px`). The input is only
    read. GrabCut draws random numbers, so the generator is reseeded before every call: the same
    crop always gives the same cut.
    """
    height, width = crop.shape[:2]
    small = cv2.resize(crop, None, fx=settings.scale, fy=settings.scale,
                       interpolation=cv2.INTER_AREA)
    small_h, small_w = small.shape[:2]
    centre = (small_w // 2, small_h // 2)
    band = max(1, round(settings.border_px * settings.scale))
    grab = np.full((small_h, small_w), cv2.GC_BGD, dtype=np.uint8)
    reach = round(settings.reach_px * settings.scale)
    cv2.circle(grab, centre, reach, cv2.GC_PR_FGD, -1)
    disc = (grab != cv2.GC_BGD).astype(np.uint8)
    rim = disc - cv2.erode(disc, np.ones((3, 3), np.uint8))
    cv2.circle(grab, centre, max(1, round(settings.seed_px * settings.scale)), cv2.GC_FGD, -1)
    grab[:band] = grab[-band:] = cv2.GC_BGD
    grab[:, :band] = grab[:, -band:] = cv2.GC_BGD
    cv2.setRNGSeed(0)
    cv2.grabCut(small, grab, (0, 0, 0, 0), np.zeros((1, 65)), np.zeros((1, 65)),
                _GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    foreground = ((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD)).astype(np.uint8)
    if not foreground[centre[1], centre[0]]:
        raise PlacementFault("no unit")
    _, labels = cv2.connectedComponents(foreground, connectivity=4)
    part = (labels == labels[centre[1], centre[0]]).astype(np.uint8)
    if (part & rim).any():
        raise PlacementFault("not wholly in view")
    inner = part[band : small_h - band, band : small_w - band]
    if inner[0].any() or inner[-1].any() or inner[:, 0].any() or inner[:, -1].any():
        raise PlacementFault("not wholly in view")
    contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(part)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    full = cv2.resize(filled * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    low, high = settings.area_px
    if not low <= int(cv2.countNonZero(full)) <= high:
        raise PlacementFault("unit size out of range")
    return _silhouette(full)


def best_cut(crops: list[np.ndarray], settings: UnitSettings) -> tuple[int, Silhouette]:
    """Of several crops of the same pass, the index and cut-out whose centroid is nearest the
    crop centre; crops that fault are skipped, and when all fault the middle crop's fault is
    raised."""
    if not crops:
        raise ValueError("best_cut needs at least one crop")
    best: tuple[float, int, Silhouette] | None = None
    faults: list[PlacementFault] = []
    for index, crop in enumerate(crops):
        try:
            cut = cut_unit(crop, settings)
        except PlacementFault as fault:
            faults.append(fault)
            continue
        height, width = crop.shape[:2]
        distance = math.hypot(cut.centroid[0] - width / 2, cut.centroid[1] - height / 2)
        if best is None or distance < best[0]:
            best = (distance, index, cut)
    if best is None:
        raise faults[len(faults) // 2]
    return best[1], best[2]
