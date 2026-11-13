"""The lane file (`lane.yml`) and the crop it describes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from qs.errors import ConfigError

_KEYS = {"version", "window", "fps"}


@dataclass(frozen=True, slots=True)
class Lane:
    """The window `(x, y, w, h)` of a frame the lane occupies, and the camera's frame rate."""

    window: tuple[int, int, int, int]
    fps: float


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def load_lane(path: Path) -> Lane:
    """The lane in `path`; a missing or extra key, a window without area, or a fps that is not
    a positive finite number raises ConfigError naming the key."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: cannot read the lane file: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the lane file must be a mapping")
    for key in sorted(_KEYS - data.keys()):
        raise ConfigError(f"{path}: missing key {key!r}")
    for key in sorted(data.keys() - _KEYS, key=str):
        raise ConfigError(f"{path}: unknown key {key!r}")
    if data["version"] != 1:
        raise ConfigError(f"{path}: key 'version' must be 1, not {data['version']!r}")
    window = data["window"]
    if (
        not isinstance(window, list)
        or len(window) != 4
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in window)
        or window[0] < 0
        or window[1] < 0
        or window[2] <= 0
        or window[3] <= 0
    ):
        raise ConfigError(
            f"{path}: key 'window' must be [x, y, w, h], x and y >= 0, w and h > 0: {window!r}"
        )
    fps = data["fps"]
    if not _is_number(fps) or not math.isfinite(fps) or fps <= 0:
        raise ConfigError(f"{path}: key 'fps' must be a positive finite number, not {fps!r}")
    x, y, w, h = window
    return Lane((x, y, w, h), float(fps))


def crop(pixels: np.ndarray, lane: Lane) -> np.ndarray:
    """The lane's rectangle of `pixels`, as a copy; a window reaching outside the frame raises."""
    x, y, w, h = lane.window
    height, width = pixels.shape[:2]
    if x + w > width or y + h > height:
        raise ConfigError(
            f"lane window {lane.window} reaches outside the frame of {width}x{height}"
        )
    return pixels[y : y + h, x : x + w].copy()
