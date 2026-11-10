"""Drawn footage for the self-check of #134: a warm disc that crosses with no rim drawn, and a
full cup that is tracked, on the 320x240, 8 fps line of `warmth_fixtures`.

A crossing is the nine steps of `PATH_X`, 40 px apart, and four frames of steel. The window is the
box [130, 80, 80, 80], centred on x 170, so a thing is over it at x 130, 170 and 210. A pink моти
has warmth 50 over the inner half of its rim, and the disc is drawn with the same 50 at its
centre, fading to nothing over a radius of 40, so that no edge is traced: the finder sees nothing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from test_cup import drawn_frame
from warmth_fixtures import FPS, warmth_config

PATH_X = (10, 50, 90, 130, 170, 210, 250, 290, 330)
WINDOW = [130, 80, 80, 80]
VIDEO = "line.avi"


def steel(count: int) -> list[np.ndarray]:
    return [drawn_frame("none", seed=100 + i) for i in range(count)]


def _disc(pixels: np.ndarray, centre: tuple[int, int]) -> np.ndarray:
    """`pixels` with a soft red disc over them: 50 more red at the centre, none at 40 px."""
    height, width = pixels.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    distance = np.hypot(xs - centre[0], ys - centre[1])
    fade = np.clip(1.0 - distance / 40.0, 0.0, 1.0)
    warmed = pixels.astype(np.float32)
    warmed[..., 2] += 50.0 * fade
    return np.clip(warmed, 0, 255).astype(np.uint8)


def disc_crossing() -> list[np.ndarray]:
    """A warm disc and no cup."""
    return [_disc(drawn_frame("none", seed=x), (x, 120)) for x in PATH_X] + steel(4)


def cup_crossing(contents: str) -> list[np.ndarray]:
    return [drawn_frame(contents, centre=(x, 120), seed=x) for x in PATH_X] + steel(4)


def write_video(path: Path, frames: list[np.ndarray]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (320, 240))
    assert writer.isOpened()
    for pixels in frames:
        writer.write(pixels)
    writer.release()
    return path


def config(root: Path, **line: object) -> Path:
    """The warmth line config with the window set; `untracked_window=None` leaves it out."""
    return warmth_config(root, **{"untracked_window": WINDOW, **line})
