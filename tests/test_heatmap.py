"""The heatmap's two encodings (docs/15 §6): the store's PNG and the screen's overlay.

The threshold ratio at 0, 0.5, 1, 2 and 3 × threshold decodes to 0, 64, 128, 255, 255 with
the heatmap's shape; the overlay blends JET's top and bottom colours 50/50 over the masked
pixels and leaves every other pixel of the window unchanged; a threshold that is not a
finite number above 0 is refused with `ValueError` by both; and everything is pure
(C1, C2), on the read-only heatmap the score produces.
"""

import cv2
import numpy as np
import pytest

from conftest import Paint
from qs.config import LocalizeConfig
from qs.inspector.heatmap import heatmap_overlay, heatmap_png
from qs.inspector.localize import localize
from qs.inspector.window import cut_window, score_map

THRESHOLD = 1.83  # the example config's threshold.value
JET_TOP_RGB = (128, 0, 0)  # cv2.COLORMAP_JET at grey 255, as RGB
JET_BOTTOM_RGB = (0, 0, 128)  # cv2.COLORMAP_JET at grey 0, as RGB
BAD_THRESHOLDS = [0, -1.83, float("nan"), float("inf")]


def test_the_store_png_encodes_the_threshold_ratio() -> None:
    heatmap = np.zeros((5, 5), dtype=np.float32)
    values = [0.0, 0.5 * THRESHOLD, THRESHOLD, 2.0 * THRESHOLD, 3.0 * THRESHOLD]
    for row, value in enumerate(values):
        heatmap[row, row] = value
    decoded = cv2.imdecode(
        np.frombuffer(heatmap_png(heatmap, THRESHOLD), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    assert decoded is not None
    assert decoded.shape == heatmap.shape
    assert decoded.dtype == np.uint8
    for row, want in enumerate((0, 64, 128, 255, 255)):
        assert decoded[row, row] == want
    assert int(decoded.sum()) == 0 + 64 + 128 + 255 + 255  # every other pixel is 0


@pytest.mark.parametrize("bad", BAD_THRESHOLDS)
def test_a_threshold_that_is_not_a_finite_number_above_0_is_refused(bad: float) -> None:
    heatmap = np.full((4, 4), 1.0, dtype=np.float32)
    window = np.full((4, 4, 3), 90, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="not a finite number above 0"):
        heatmap_png(heatmap, bad)
    with pytest.raises(ValueError, match="not a finite number above 0"):
        heatmap_overlay(window, heatmap, bad, mask)


def test_the_overlay_blends_jets_ends_50_50_over_the_masked_pixels() -> None:
    grey = 100
    window = np.full((3, 3, 3), grey, dtype=np.uint8)
    heatmap = np.zeros((3, 3), dtype=np.float32)
    heatmap[0, 0] = 2.0 * THRESHOLD  # ≥ 1.5 × threshold: JET's top colour
    heatmap[1, 1] = 0.0  # JET's bottom colour
    heatmap[2, 2] = 1.5 * THRESHOLD  # exactly at the clip: still the top colour
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True
    overlay = heatmap_overlay(window, heatmap, THRESHOLD, mask)
    assert overlay.shape == (3, 3, 3)
    assert overlay.dtype == np.uint8
    top = tuple((grey + channel + 1) // 2 for channel in JET_TOP_RGB)
    bottom = tuple((grey + channel + 1) // 2 for channel in JET_BOTTOM_RGB)
    assert tuple(overlay[0, 0]) == top
    assert tuple(overlay[2, 2]) == top
    assert tuple(overlay[1, 1]) == bottom
    assert (overlay[~mask] == grey).all()  # every pixel outside the mask is the window's


def test_the_encodings_are_pure_and_deterministic() -> None:
    window = np.full((8, 8, 3), 120, dtype=np.uint8)
    window[2:5, 2:5] = (40, 160, 210)
    rng = np.random.default_rng(3)
    heatmap = (rng.random((8, 8)) * 3.0 * THRESHOLD).astype(np.float32)
    mask = rng.random((8, 8)) < 0.5
    window_before, heatmap_before, mask_before = window.copy(), heatmap.copy(), mask.copy()
    first_png = heatmap_png(heatmap, THRESHOLD)
    first_overlay = heatmap_overlay(window, heatmap, THRESHOLD, mask)
    assert heatmap_png(heatmap, THRESHOLD) == first_png
    assert np.array_equal(heatmap_overlay(window, heatmap, THRESHOLD, mask), first_overlay)
    assert np.array_equal(window, window_before)
    assert np.array_equal(heatmap, heatmap_before)
    assert np.array_equal(mask, mask_before)
    assert not np.shares_memory(first_overlay, window)
    heatmap.setflags(write=False)
    window.setflags(write=False)
    mask.setflags(write=False)
    assert heatmap_png(heatmap, THRESHOLD) == first_png
    assert np.array_equal(heatmap_overlay(window, heatmap, THRESHOLD, mask), first_overlay)


def test_the_scored_heatmap_encodes_end_to_end(
    localize_config: LocalizeConfig, paint: Paint, mask_dilate_px: int
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, (960, 540), 180)
    frame_before = pixels.copy()
    silhouette = localize(pixels, localize_config)
    window, box = cut_window(pixels, silhouette.centroid, 800)
    amap = np.random.default_rng(5).standard_normal((256, 256))
    amap = (amap * 0.5 + 1.8).astype(np.float32)
    heatmap, score = score_map(amap, silhouette.mask, box, 800, mask_dilate_px)
    assert not heatmap.flags.writeable
    decoded = cv2.imdecode(
        np.frombuffer(heatmap_png(heatmap, THRESHOLD), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    assert decoded is not None
    assert decoded.shape == heatmap.shape
    overlay = heatmap_overlay(window, heatmap, THRESHOLD, heatmap > 0)
    assert overlay.shape == (800, 800, 3)
    assert overlay.dtype == np.uint8
    assert np.array_equal(pixels, frame_before)  # the whole chain never touched the frame
    assert 0.0 < score < 10.0
