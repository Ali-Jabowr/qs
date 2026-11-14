"""A drawn clip with frames missing, made with the real `ffmpeg` (#173).

120 frames, each with its own number written large; frames 30-39 are removed and the other
frames keep their timestamps, as a camera that dropped frames would leave them. The container
still says 120; 110 decode.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

FRAMES = 120
FIRST_DROPPED = 30
DROPPED = 10
FPS = 30


def numbered_frame(index: int) -> np.ndarray:
    frame = np.full((240, 320, 3), 50, np.uint8)
    cv2.putText(frame, str(index), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    return frame


def make_gapped_clip(folder: Path, name: str = "clip") -> Path:
    """`<folder>/<name>.mkv`, the gapped clip. Needs `ffmpeg` on the path."""
    full = folder / f"{name}-full.avi"
    writer = cv2.VideoWriter(str(full), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (320, 240))
    assert writer.isOpened()
    for index in range(FRAMES):
        writer.write(numbered_frame(index))
    writer.release()
    video = folder / f"{name}.mkv"
    last = FIRST_DROPPED + DROPPED - 1
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(full),
         "-vf", f"select='not(between(n,{FIRST_DROPPED},{last}))'",
         "-fps_mode", "passthrough", "-c:v", "mjpeg", str(video)],
        check=True,
    )
    full.unlink()
    return video


def make_off_rate_clip(
    folder: Path, name: str = "clip", frames: int = 91, rate: float = 30.2
) -> Path:
    """`<folder>/<name>.mkv`: every frame present, timestamps at a true `rate` although the
    clip is declared 30 fps, so the container's count is an estimate that is off by one."""
    full = folder / f"{name}-full.avi"
    writer = cv2.VideoWriter(str(full), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (320, 240))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(numbered_frame(index))
    writer.release()
    video = folder / f"{name}.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(full),
         "-c", "copy", "-bsf:v", f"setts=ts=N/({rate}*TB)", str(video)],
        check=True,
    )
    full.unlink()
    return video
