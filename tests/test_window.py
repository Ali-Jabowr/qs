"""The window PatchCore sees, and the masked score of its anomaly map (docs/15 §4, §6).

Synthetic discs drawn in memory and localized with the example config: the centred window
and its unclipped box, the black padding at every frame edge, one resize scale for every
unit, the ImageNet normalisation, the dilated mask cutting a planted hot spot off the
floor (ADR-0009), the non-finite raise (C8), purity (C1, C2) and the preprocess timing
budget (docs/10 §5).
"""

import math
from time import perf_counter

import cv2
import numpy as np
import pytest

from conftest import FRAME_SHAPE, Paint
from qs.config import LocalizeConfig
from qs.errors import InferenceError
from qs.inspector.localize import localize
from qs.inspector.window import cut_window, score_map, to_input

UNIT_CENTRE = (960, 540)  # the centre of the example ROI [360, 0, 1200, 1080]
UNIT_RADIUS = 180
WINDOW_PX = 800
INPUT_PX = 256
CENTRED_BOX = (560, 140, 800, 800)
EDGES = ("left", "right", "top", "bottom")
EDGE_CENTROID = {
    "left": (100, 540),  # 100 px from the left edge: the box starts at x0 = -300
    "right": (1820, 540),
    "top": (960, 100),
    "bottom": (960, 980),
}
EDGE_BOX = {
    "left": (-300, 140, 800, 800),
    "right": (1420, 140, 800, 800),
    "top": (560, -300, 800, 800),
    "bottom": (560, 580, 800, 800),
}


def _localized_disc(
    paint: Paint, config: LocalizeConfig, centre: tuple[int, int] = UNIT_CENTRE
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[int, int, int, int]]:
    """A pale disc on the floor, localized: the frame, its mask, centroid and window box."""
    pixels = paint.floor()
    paint.disc(pixels, centre, UNIT_RADIUS)
    silhouette = localize(pixels, config)
    _, box = cut_window(pixels, silhouette.centroid, WINDOW_PX)
    return pixels, silhouette.mask, silhouette.centroid, box


def test_a_disc_at_the_frames_centre_gives_the_centred_window(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels, _, centroid, box = _localized_disc(paint, localize_config)
    window, box = cut_window(pixels, centroid, WINDOW_PX)
    assert box == CENTRED_BOX
    assert window.shape == (WINDOW_PX, WINDOW_PX, 3)
    assert window.dtype == np.uint8
    assert np.array_equal(window, pixels[140:940, 560:1360])
    ys, xs = np.nonzero(window[:, :, 0] > 100)
    assert math.dist((xs.mean(), ys.mean()), (400, 400)) <= 0.5


@pytest.mark.parametrize("edge", EDGES)
def test_a_disc_100_px_from_an_edge_gives_a_black_padded_window(
    paint: Paint, edge: str
) -> None:
    height, width = FRAME_SHAPE
    cx, cy = EDGE_CENTROID[edge]
    pixels = paint.floor()
    paint.disc(pixels, (cx, cy), 200)
    window, box = cut_window(pixels, (float(cx), float(cy)), WINDOW_PX)
    assert box == EDGE_BOX[edge]
    assert window.shape == (WINDOW_PX, WINDOW_PX, 3)
    x0, y0 = box[0], box[1]
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + WINDOW_PX, width), min(y0 + WINDOW_PX, height)
    outside = np.ones((WINDOW_PX, WINDOW_PX), dtype=bool)
    outside[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = False
    assert not window[outside].any()  # every out-of-frame pixel is 0 in all three channels
    assert np.array_equal(
        window[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0], pixels[fy0:fy1, fx0:fx1]
    )


def test_the_centre_is_rounded_half_up_and_a_far_window_is_all_black() -> None:
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    window, box = cut_window(pixels, (100.5, 99.49), 40)
    assert box == (81, 79, 40, 40)  # floor(100.5 + 0.5) = 101, floor(99.49 + 0.5) = 99
    assert window.shape == (40, 40, 3)
    assert not window.any()


def test_cut_window_never_touches_the_frame(paint: Paint) -> None:
    pixels = paint.floor()
    paint.disc(pixels, (300, 300), 120)
    before = pixels.copy()
    window, box = cut_window(pixels, (300.4, 299.6), 500)
    assert box == (50, 50, 500, 500)
    assert np.array_equal(pixels, before)
    assert not np.shares_memory(window, pixels)
    pixels.setflags(write=False)
    again, same_box = cut_window(pixels, (300.4, 299.6), 500)
    assert np.array_equal(again, window)
    assert same_box == box


def test_to_input_normalises_a_pure_red_window() -> None:
    window = np.zeros((64, 64, 3), dtype=np.uint8)
    window[:, :, 2] = 255  # pure red in BGR
    inp = to_input(window, INPUT_PX)
    assert inp.shape == (3, INPUT_PX, INPUT_PX)
    assert inp.dtype == np.float32
    assert np.allclose(inp[0], (1 - 0.485) / 0.229, atol=1e-6)
    assert np.allclose(inp[1], (0 - 0.456) / 0.224, atol=1e-6)
    assert np.allclose(inp[2], (0 - 0.406) / 0.225, atol=1e-6)


def test_to_input_is_pure_and_deterministic() -> None:
    window = np.full((100, 100, 3), 137, dtype=np.uint8)
    window[40:60, 40:60] = (30, 200, 90)
    before = window.copy()
    first = to_input(window, INPUT_PX)
    assert np.array_equal(to_input(window, INPUT_PX), first)
    assert np.array_equal(window, before)
    assert not np.shares_memory(first, window)
    window.setflags(write=False)
    assert np.array_equal(to_input(window, INPUT_PX), first)


def test_every_unit_is_seen_at_one_scale(localize_config: LocalizeConfig, paint: Paint) -> None:
    counts = []
    for radius in (150, 200):
        pixels = paint.floor()
        paint.disc(pixels, UNIT_CENTRE, radius)
        silhouette = localize(pixels, localize_config)
        window, _ = cut_window(pixels, silhouette.centroid, WINDOW_PX)
        inp = to_input(window, INPUT_PX)
        counts.append(int((inp[0] > 0).sum()))  # channel 0 above 0: the disc, not the floor
    ratio = counts[0] / counts[1]
    assert abs(ratio - (150 / 200) ** 2) <= 0.02 * (150 / 200) ** 2


def _cut_mask(
    silhouette_mask: np.ndarray, box: tuple[int, int, int, int], window_px: int
) -> np.ndarray:
    """The frame-sized silhouette mask cut with the window box, 0 outside the frame (docs/15 §6)."""
    x0, y0 = box[0], box[1]
    cut = np.zeros((window_px, window_px), dtype=np.uint8)
    height, width = silhouette_mask.shape[:2]
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + window_px, width), min(y0 + window_px, height)
    cut[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = (
        silhouette_mask[fy0:fy1, fx0:fx1] != 0
    ) * np.uint8(255)
    return cut


def _dilated_mask(
    silhouette_mask: np.ndarray, box: tuple[int, int, int, int], window_px: int, dilate_px: int
) -> np.ndarray:
    """The cut silhouette dilated by the elliptical kernel of `2 · dilate_px + 1` (docs/15 §6)."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    return cv2.dilate(_cut_mask(silhouette_mask, box, window_px), kernel) > 0


def test_a_constant_map_scores_c_on_the_dilated_silhouette(
    localize_config: LocalizeConfig, paint: Paint, mask_dilate_px: int
) -> None:
    pixels, mask, centroid, box = _localized_disc(paint, localize_config)
    c = 2.5
    amap = np.full((INPUT_PX, INPUT_PX), c, dtype=np.float32)
    heatmap, score = score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)
    assert heatmap.shape == (WINDOW_PX, WINDOW_PX)
    assert heatmap.dtype == np.float32
    assert score == c
    dilated = _dilated_mask(mask, box, WINDOW_PX, mask_dilate_px)
    assert (heatmap[dilated] == c).all()
    assert (heatmap[~dilated] == 0).all()


def test_dust_just_outside_the_dilated_silhouette_cannot_raise_the_alarm(
    localize_config: LocalizeConfig, paint: Paint, mask_dilate_px: int
) -> None:
    pixels, mask, centroid, box = _localized_disc(paint, localize_config)
    dilated = _dilated_mask(mask, box, WINDOW_PX, mask_dilate_px)
    row = int(centroid[1] + 0.5) - box[1]
    edge = int(np.nonzero(_cut_mask(mask, box, WINDOW_PX)[row])[0][0])
    inside = edge - mask_dilate_px  # mask_dilate_px from the silhouette's edge, along the axis
    outside = inside - 1  # one pixel further
    assert dilated[row, inside]
    assert not dilated[row, outside]

    amap = np.ones((WINDOW_PX, WINDOW_PX), dtype=np.float32)  # input_px == window_px: identity
    amap[row, inside] = 5.0
    heatmap, score = score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)
    assert score == 5.0
    assert heatmap[row, inside] == 5.0

    amap[row, inside] = 1.0
    amap[row, outside] = 5.0
    heatmap, score = score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)
    assert score == 1.0
    assert heatmap[row, outside] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_map_value_raises_before_any_score(
    localize_config: LocalizeConfig, paint: Paint, mask_dilate_px: int, bad: float
) -> None:
    pixels, mask, centroid, box = _localized_disc(paint, localize_config)
    amap = np.ones((INPUT_PX, INPUT_PX), dtype=np.float32)
    amap[0, 0] = bad  # on the floor, far outside the mask: it still refuses
    with pytest.raises(InferenceError) as raised:
        score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)
    assert "non-finite" in str(raised.value)
    assert raised.value.reason == "inference"


def test_score_map_is_pure_and_deterministic(
    localize_config: LocalizeConfig, paint: Paint, mask_dilate_px: int
) -> None:
    pixels, mask, centroid, box = _localized_disc(paint, localize_config)
    amap = np.random.default_rng(7).standard_normal((INPUT_PX, INPUT_PX))
    amap = (amap + 2.0).astype(np.float32)
    map_before = amap.copy()
    frame_before = pixels.copy()
    first_heat, first_score = score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)
    second_heat, second_score = score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)
    assert np.array_equal(first_heat, second_heat)
    assert first_score == second_score
    assert np.array_equal(amap, map_before)
    assert np.array_equal(pixels, frame_before)
    assert not first_heat.flags.writeable
    assert not np.shares_memory(first_heat, amap)
    amap.setflags(write=False)
    third_heat, third_score = score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)
    assert np.array_equal(third_heat, first_heat)
    assert third_score == first_score


def test_the_preprocess_chain_fits_its_budget(
    localize_config: LocalizeConfig, paint: Paint, mask_dilate_px: int
) -> None:
    pixels, mask, centroid, box = _localized_disc(paint, localize_config)
    amap = np.random.default_rng(11).standard_normal((INPUT_PX, INPUT_PX)).astype(np.float32)

    def chain() -> None:
        window, box = cut_window(pixels, centroid, WINDOW_PX)
        to_input(window, INPUT_PX)
        score_map(amap, mask, box, WINDOW_PX, mask_dilate_px)

    chain()  # one warm-up
    times = []
    for _ in range(20):
        start = perf_counter()
        chain()
        times.append((perf_counter() - start) * 1000.0)
    median = sorted(times)[len(times) // 2]
    print(
        f"\ncut_window + to_input + score_map, 1920x1080, window {WINDOW_PX}: "
        f"median {median:.2f} ms over 20 runs (the 40 ms preprocess budget, docs/10 §5)"
    )
    assert median < 20.0
