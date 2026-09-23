"""The window PatchCore sees, and the masked score of its anomaly map (docs/15 §4, §6).

`cut_window` cuts the square of `window_px` centred on P06's centroid, padding with black
where it runs past the frame edge — the background the window would have seen — and reports
where it sits as an unclipped `window_box`, so the heatmap and the box always align.
`to_input` is the one resize every unit gets: `INTER_AREA` to `input_px`, one constant
factor, then BGR → RGB, ÷ 255 and the ImageNet mean and std, as a CHW float32 array that
is still numpy. `score_map` brings the model's anomaly map back to window coordinates,
zeroes it outside the silhouette dilated by an elliptical kernel of `2 · mask_dilate_px +
1`, and takes the maximum as the score: dust on the floor cannot raise an alarm
(ADR-0009). A non-finite map value or score raises `InferenceError` (C8): the model is
never trusted to have passed. Pure: every input is only ever read, never written, and
read-only inputs work.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from qs.errors import InferenceError

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cut_window(
    pixels: np.ndarray, centroid: tuple[float, float], window_px: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Cut the `window_px` square centred on `centroid`, black outside the frame (docs/15 §4).

    The centre is rounded half up: `x0 = ⌊cx + 0.5⌋ − window_px // 2`, the same for `y0`.
    The box is **not clipped** — it may start outside the frame. The window never aliases
    the frame.
    """
    cx, cy = centroid
    x0 = math.floor(cx + 0.5) - window_px // 2
    y0 = math.floor(cy + 0.5) - window_px // 2
    window = np.zeros((window_px, window_px, *pixels.shape[2:]), dtype=pixels.dtype)
    height, width = pixels.shape[:2]
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + window_px, width), min(y0 + window_px, height)
    if fx1 > fx0 and fy1 > fy0:
        window[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = pixels[fy0:fy1, fx0:fx1]
    return window, (x0, y0, window_px, window_px)


def to_input(window: np.ndarray, input_px: int) -> np.ndarray:
    """Resize the BGR window once and normalise it into the model's input (docs/15 §4).

    `INTER_AREA` to `input_px × input_px`, BGR → RGB, ÷ 255, the ImageNet mean and std,
    CHW float32. The tensor conversion stays with the model.
    """
    resized = cv2.resize(window, (input_px, input_px), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    scaled = rgb.astype(np.float32) / 255.0
    scaled -= _IMAGENET_MEAN
    scaled /= _IMAGENET_STD
    return np.ascontiguousarray(scaled.transpose(2, 0, 1))


def score_map(
    map_input: np.ndarray,
    silhouette_mask: np.ndarray,
    window_box: tuple[int, int, int, int],
    window_px: int,
    mask_dilate_px: int,
) -> tuple[np.ndarray, float]:
    """Mask the anomaly map to the dilated silhouette and take the maximum (docs/15 §6).

    The map at `input_px` is brought back to `window_px` with `INTER_LINEAR` and stays in
    window coordinates. The frame-sized `silhouette_mask` is cut with the same
    `window_box` (0 outside the frame) and dilated by an elliptical kernel of
    `2 · mask_dilate_px + 1`; the map is set to 0 outside it. The heatmap comes back
    read-only, float32, in score units. A non-finite map value or score raises
    `InferenceError` (C8) — never a score.
    """
    map_32 = np.asarray(map_input, dtype=np.float32)
    finite = np.isfinite(map_32)
    if not finite.all():
        raise InferenceError(
            f"the anomaly map holds {int(np.count_nonzero(~finite))} non-finite value(s)"
        )
    x0, y0 = window_box[0], window_box[1]
    mask_window = np.zeros((window_px, window_px), dtype=np.uint8)
    height, width = silhouette_mask.shape[:2]
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + window_px, width), min(y0 + window_px, height)
    if fx1 > fx0 and fy1 > fy0:
        mask_window[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = (
            silhouette_mask[fy0:fy1, fx0:fx1] != 0
        ) * np.uint8(255)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * mask_dilate_px + 1, 2 * mask_dilate_px + 1)
    )
    dilated = cv2.dilate(mask_window, kernel)
    upsampled = cv2.resize(map_32, (window_px, window_px), interpolation=cv2.INTER_LINEAR)
    heatmap = upsampled * (dilated != 0)
    heatmap.setflags(write=False)
    score = float(heatmap.max())
    if not math.isfinite(score):
        raise InferenceError(f"the masked score {score} is not finite")
    return heatmap, score
