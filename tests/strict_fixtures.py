"""Drawn video for the strict cup finder (#133), at 320x240 and 8 fps like `warmth_fixtures`.

A crossing is a cup at the nine steps of `PATH_X` (40 px a frame) then four frames of steel, as in
`test_line_warmth_score`. `ring` is a thin dark ring drawn over every frame at a fixed place, as a
reflection of the lens hole would be: the finder sees it as a rim of its size.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from test_cup import drawn_frame
from warmth_fixtures import FPS

PATH_X = (10, 50, 90, 130, 170, 210, 250, 290, 330)
ROW = 120
RIM_GREY = 105


def steel(count: int, *, seed: int = 100) -> list[np.ndarray]:
    return [drawn_frame("none", seed=seed + i) for i in range(count)]


def crossing(contents: str = "pink", *, half_axis: int = 50, seed: int = 0) -> list[np.ndarray]:
    frames = [
        drawn_frame(contents, centre=(x, ROW), axes=(half_axis, half_axis), seed=seed + x)
        for x in PATH_X
    ]
    return frames + steel(4, seed=seed + 100)


def standing(count: int, *, contents: str = "pink", x: int = 130, half_axis: int = 50,
             seed: int = 0) -> list[np.ndarray]:
    """A cup standing at x for `count` frames: the chain is stopped."""
    return [
        drawn_frame(contents, centre=(x, ROW), axes=(half_axis, half_axis), seed=seed + i)
        for i in range(count)
    ]


def with_ring(frames: list[np.ndarray], centre: tuple[int, int], half_axis: int) -> list[np.ndarray]:
    """The same frames with a still ring drawn over each, so it is in the picture whatever moves."""
    out = []
    for pixels in frames:
        drawn = pixels.copy()
        cv2.ellipse(drawn, centre, (half_axis, half_axis), 0.0, 0, 360, (RIM_GREY,) * 3, 3)
        out.append(drawn)
    return out


def write_video(path: Path, frames: list[np.ndarray]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (320, 240))
    assert writer.isOpened()
    for pixels in frames:
        writer.write(pixels)
    writer.release()
    return path
