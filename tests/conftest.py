"""Shared fixtures for the Inspector's synthetic frames (docs/15 §2).

Pale discs, ellipses and blobs — and bright starch specks — drawn in memory on a black
1920×1080 BGR floor, against the example config's `inspector.localize`. Nothing here needs
a camera, a display or a GPU, and nothing is written anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.config import GeometryConfig, LocalizeConfig, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"

FRAME_SHAPE = (1080, 1920)
FLOOR_GREY = 20  # the black floor of the acceptance frames
UNIT_GREY = 200  # a pale unit on it
DARK_GREY = 30  # the dark filling of a tear, seen from above
SPECK_GREY = 255


def _bgr(grey: int) -> tuple[int, int, int]:
    return (grey, grey, grey)


def _clear_of(
    box: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    avoid: tuple[tuple[int, int, int], ...],
) -> bool:
    """Whether a speck's box clears the placed specks by 8 px and every avoid circle by 12 px."""
    x0, y0, x1, y1 = box
    for ox0, oy0, ox1, oy1 in placed:
        if x0 - 8 <= ox1 and ox0 <= x1 + 8 and y0 - 8 <= oy1 and oy0 <= y1 + 8:
            return False
    for cx, cy, radius in avoid:
        near_x = min(max(x0, cx), x1)
        near_y = min(max(y0, cy), y1)
        if (cx - near_x) ** 2 + (cy - near_y) ** 2 <= (radius + 12) ** 2:
            return False
    return True


@dataclass(frozen=True, slots=True)
class Paint:
    """Drawing primitives for synthetic frames: pale shapes on a black floor, in place."""

    def floor(self, grey: int = FLOOR_GREY) -> np.ndarray:
        """A black 1920×1080 BGR floor at `grey`."""
        return np.full(FRAME_SHAPE + (3,), grey, dtype=np.uint8)

    def disc(
        self,
        pixels: np.ndarray,
        centre: tuple[int, int],
        radius: int,
        grey: int = UNIT_GREY,
    ) -> None:
        """A filled disc."""
        cv2.circle(pixels, centre, radius, _bgr(grey), -1)

    def ellipse(
        self,
        pixels: np.ndarray,
        centre: tuple[int, int],
        axes: tuple[int, int],
        angle: int,
        grey: int = UNIT_GREY,
    ) -> None:
        """A filled ellipse: `axes` the half-sizes, `angle` the rotation in degrees."""
        cv2.ellipse(pixels, centre, axes, angle, 0, 360, _bgr(grey), -1)

    def wedge_disc(
        self,
        pixels: np.ndarray,
        centre: tuple[int, int],
        radius: int,
        degrees: int = 30,
        grey: int = UNIT_GREY,
    ) -> None:
        """A disc with a wedge of `degrees` cut out to its centre — the notched unit."""
        half = degrees / 2
        cv2.ellipse(pixels, centre, (radius, radius), 0, half, 360 - half, _bgr(grey), -1)

    def blob(
        self, pixels: np.ndarray, box: tuple[int, int, int, int], grey: int = UNIT_GREY
    ) -> None:
        """A filled rectangle, `box` = (x, y, w, h) exactly `w` × `h` pixels."""
        x, y, w, h = box
        cv2.rectangle(pixels, (x, y), (x + w - 1, y + h - 1), _bgr(grey), -1)

    def specks(
        self,
        pixels: np.ndarray,
        count: int,
        *,
        grey: int = SPECK_GREY,
        seed: int = 0,
        avoid: tuple[tuple[int, int, int], ...] = (),
    ) -> None:
        """`count` bright 1–2 px specks scattered on the floor; the same seed, the same layout.

        Specks stay at least 8 px from each other and 12 px outside every `avoid`
        (centre, radius) circle, so no speck survives the blur, the threshold and the
        open as part of a unit, and none bridges to a unit in the close.
        """
        rng = np.random.default_rng(seed)
        placed: list[tuple[int, int, int, int]] = []
        attempts = 0
        while len(placed) < count:
            attempts += 1
            if attempts > 10_000 + 100 * count:
                raise RuntimeError("could not scatter the specks")
            x = int(rng.integers(8, FRAME_SHAPE[1] - 10))
            y = int(rng.integers(8, FRAME_SHAPE[0] - 10))
            side = int(rng.integers(1, 3))
            box = (x, y, x + side - 1, y + side - 1)
            if not _clear_of(box, placed, avoid):
                continue
            placed.append(box)
            pixels[y : y + side, x : x + side] = grey


@pytest.fixture(scope="session")
def localize_config() -> LocalizeConfig:
    """The example config's `inspector.localize`: roi [360, 0, 1200, 1080], threshold 70,
    min_area_px 20000, border_margin_px 10."""
    return load_config(EXAMPLE).config.inspector.localize


@pytest.fixture(scope="session")
def geometry_config() -> GeometryConfig:
    """The example config's `threshold.geometry`: area_px [98000, 131000], axis_ratio_min 0.90,
    solidity_min 0.96."""
    return load_config(EXAMPLE).config.threshold.geometry


@pytest.fixture
def paint() -> Paint:
    """The synthetic-frame drawing primitives."""
    return Paint()


@pytest.fixture(scope="session")
def mask_dilate_px() -> int:
    """The example config's `inspector.mask_dilate_px`: 12."""
    return load_config(EXAMPLE).config.inspector.mask_dilate_px
