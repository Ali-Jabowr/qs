"""`frame_probs`: every frame of a clip, cropped to the lane, scored by the lane classifier."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from qs.errors import ConfigError
from qs.labels import FrameReadError
from qs.lane.lanefile import Lane
from qs.lane.passes import frame_probs

WIDTH, HEIGHT = 64, 48
LANE = Lane((16, 8, 32, 24), 30.0)
RED = (0, 0, 255)  # BGR
BLUE = (255, 0, 0)


class _RedDetector(torch.nn.Module):
    """Untrained stand-in: logits (x, -x, 0) with x from the mean red channel of the input, so a
    red image scores class 0 (full) and a non-red one class 1 (empty)."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = 100.0 * batch[:, 0].mean(dim=(1, 2))
        return torch.stack([x, -x, torch.zeros_like(x)], dim=1)


def _clip(path: Path, inside: list[tuple[int, int, int]]) -> Path:
    """One frame per colour: that colour inside the lane window, the opposite outside it."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"FFV1"), 30.0, (WIDTH, HEIGHT))
    for colour in inside:
        frame = np.full((HEIGHT, WIDTH, 3), BLUE if colour == RED else RED, dtype=np.uint8)
        x, y, w, h = LANE.window
        frame[y : y + h, x : x + w] = colour
        writer.write(frame)
    writer.release()
    return path


def test_one_row_per_frame_in_frame_order_from_the_lane_crop(tmp_path):
    clip = _clip(tmp_path / "a.avi", [RED, BLUE, RED, RED, BLUE])
    probs = frame_probs(clip, LANE, _RedDetector(), "cpu", input_px=16)
    assert probs.shape == (5, 3)
    assert probs.argmax(axis=1).tolist() == [0, 1, 0, 0, 1]


def test_rows_are_probabilities(tmp_path):
    clip = _clip(tmp_path / "a.avi", [RED, BLUE])
    probs = frame_probs(clip, LANE, _RedDetector(), "cpu", input_px=16)
    assert probs.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("batch", [1, 2, 3, 64])
def test_the_batch_size_does_not_change_the_result(tmp_path, batch):
    clip = _clip(tmp_path / "a.avi", [RED, BLUE, RED, RED, BLUE])
    probs = frame_probs(clip, LANE, _RedDetector(), "cpu", input_px=16, batch=batch)
    assert probs.argmax(axis=1).tolist() == [0, 1, 0, 0, 1]


def test_a_missing_video_raises(tmp_path):
    with pytest.raises(FrameReadError, match="cannot open the video"):
        frame_probs(tmp_path / "none.avi", LANE, _RedDetector(), "cpu", input_px=16)


def test_a_window_outside_the_frame_raises(tmp_path):
    clip = _clip(tmp_path / "a.avi", [RED])
    with pytest.raises(ConfigError, match="reaches outside the frame"):
        frame_probs(clip, Lane((40, 0, 32, 24), 30.0), _RedDetector(), "cpu", input_px=16)


@pytest.mark.parametrize("batch", [0, -1])
def test_a_batch_below_one_raises(tmp_path, batch):
    clip = _clip(tmp_path / "a.avi", [RED])
    with pytest.raises(ValueError, match="batch must be at least 1"):
        frame_probs(clip, LANE, _RedDetector(), "cpu", input_px=16, batch=batch)


@pytest.mark.env
def test_the_real_model_on_the_gpu_gives_one_row_per_frame(tmp_path):
    from qs.lane.model import build

    clip = _clip(tmp_path / "a.avi", [RED, BLUE, RED, BLUE, RED, RED, BLUE])
    probs = frame_probs(clip, LANE, build(None), "cuda", input_px=64, batch=3)
    assert probs.shape == (7, 3)
