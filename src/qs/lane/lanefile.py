"""The lane file (`lane.yml`) and the crop it describes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Lane:
    """The window `(x, y, w, h)` of a frame the lane occupies, and the camera's frame rate."""

    window: tuple[int, int, int, int]
    fps: float


def load_lane(path: Path) -> Lane:
    raise NotImplementedError


def crop(pixels: np.ndarray, lane: Lane) -> np.ndarray:
    raise NotImplementedError
