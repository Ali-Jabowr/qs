"""The heatmap's two encodings: the store's PNG and the screen's overlay (docs/15 §6).

Both divide the masked heatmap by the threshold, so a ratio of 1.0 means exactly at the
threshold and the colour means the same thing on every unit, every day.
`heatmap_png` clips the ratio to [0, 2] and encodes it as `ratio × 127.5`, rounded, uint8,
lossless PNG (1.0 × threshold = 128). `heatmap_overlay` clips to [0, 1.5], colours it with
`cv2.COLORMAP_JET` and blends it 50/50 over the window where the dilated mask is set,
leaving every other pixel of the window unchanged. A threshold that is not a finite
number above 0 is refused with `ValueError` by both. Pure: every input is only ever read,
never written, and read-only inputs work.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from qs.errors import InferenceError

_STORE_MAX_RATIO = 2.0
_STORE_SCALE = 127.5
_SCREEN_MAX_RATIO = 1.5
_SCREEN_SCALE = 255.0


def _checked_threshold(threshold: float) -> float:
    """The threshold as a float, or `ValueError`: it must be a finite number above 0."""
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"threshold {threshold!r} is not a finite number above 0")
    value = float(threshold)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"threshold {threshold!r} is not a finite number above 0")
    return value


def heatmap_png(heatmap: np.ndarray, threshold: float) -> bytes:
    """Encode the heatmap for the store: the threshold ratio as a lossless uint8 PNG.

    `clip(heatmap / threshold, 0, 2) × 127.5`, rounded, uint8. The decoded PNG has the
    heatmap's shape.
    """
    checked = _checked_threshold(threshold)
    ratio = np.clip(heatmap / checked, 0.0, _STORE_MAX_RATIO)
    encoded = np.round(ratio * _STORE_SCALE).astype(np.uint8)
    ok, buffer = cv2.imencode(".png", encoded)
    if not ok:
        raise InferenceError("the heatmap could not be encoded as a PNG")
    return buffer.tobytes()


def heatmap_overlay(
    window: np.ndarray, heatmap: np.ndarray, threshold: float, mask: np.ndarray
) -> np.ndarray:
    """Colour the heatmap for the screen and blend it over the masked window pixels.

    `clip(heatmap / threshold, 0, 1.5) / 1.5 × 255` → `cv2.COLORMAP_JET`, blended 50/50
    with the window where `mask` is set; every other pixel of the window is unchanged.
    The overlay is RGB, and never aliases the window.
    """
    checked = _checked_threshold(threshold)
    ratio = np.clip(heatmap / checked, 0.0, _SCREEN_MAX_RATIO) / _SCREEN_MAX_RATIO
    grey = np.round(ratio * _SCREEN_SCALE).astype(np.uint8)
    coloured = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
    overlay = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
    blended = (
        overlay.astype(np.uint16) + cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB).astype(np.uint16) + 1
    ) // 2
    keep = mask != 0
    overlay[keep] = blended[keep].astype(np.uint8)
    return overlay
