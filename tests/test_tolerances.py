"""The derived tolerances: the formula against a hand-computed example, every refusal, the
order-independence and determinism of the report, the pilot's rotation spread, and — end to
end — the round trip: synthetic good units drawn, localized and measured, the derived config
loaded through `load_config`, the good frames passing `check_geometry` and the deformed ones
failing. Pure inputs throughout; the only file written is the round-trip config in `tmp_path`.
"""

import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FLOOR_GREY, FRAME_SHAPE, UNIT_GREY, Paint
from qs.config import LocalizeConfig, load_config
from qs.inspector.geometry import GeometryMeasures, check_geometry, measure
from qs.inspector.localize import Silhouette, localize
from qs.inspector.tolerances import (
    ThresholdSuggestion,
    Tolerances,
    derive_tolerances,
    rotation_spread,
    suggest_threshold,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"

UNIT_CENTRE = (960, 540)  # the centre of the example ROI [360, 0, 1200, 1080]


def _m(area: int, axis: float, solidity: float) -> GeometryMeasures:
    """Measures with the reported diameter from the area — the derivation never reads it."""
    return GeometryMeasures(
        area_px=area,
        diameter_px=math.sqrt(4.0 * area / math.pi),
        axis_ratio=axis,
        solidity=solidity,
    )


# 3 units × 3 rotations (u3 in 2), the measures given by hand:
#   area_px  u1 1000/1020/1040 · u2 1100/1120/1140 · u3 1200/1240
#   axis     u1 .98/.98/.98     · u2 .96/.98/1.00   · u3 .98/1.00
#   solidity u1 .99/.99/.99     · u2 .98/.99/1.00   · u3 .99/.97
HAND: dict[str, list[GeometryMeasures]] = {
    "u1": [_m(1000, 0.98, 0.99), _m(1020, 0.98, 0.99), _m(1040, 0.98, 0.99)],
    "u2": [_m(1100, 0.96, 0.98), _m(1120, 0.98, 0.99), _m(1140, 1.00, 1.00)],
    "u3": [_m(1200, 0.98, 0.99), _m(1240, 1.00, 0.97)],
}

# The same measures, entered in another unit order.
HAND_REORDERED: dict[str, list[GeometryMeasures]] = {
    "u3": HAND["u3"],
    "u1": list(reversed(HAND["u1"])),
    "u2": list(reversed(HAND["u2"])),
}


def test_a_hand_computed_example_gives_the_worked_values() -> None:
    # Pooled within-unit σₙ, by hand, per measure:
    #   area_px — deviations from μᵤ: u1 (−20, 0, +20) · u2 (−20, 0, +20) · u3 (−20, +20),
    #     Σᵤ Σᵣ (x − μᵤ)² = 800 + 800 + 800 = 2400, Σᵤ (nᵤ − 1) = 2 + 2 + 1 = 5,
    #     σₙ = √(2400/5) = √480 ≈ 21.9089, 3σₙ ≈ 65.7267
    #   axis_ratio — 0 + (.02² + .02²) + (.01² + .01²) = .0008 + .0002 = .0010 → σₙ = √.0002
    #   solidity — 0 + (.01² + .01²) + (.01² + .01²) = .0002 + .0002 = .0004 → σₙ = √.00008
    # Unit means: area 1020 · 1120 · 1220 (min 1020, max 1220); axis .98 · .98 · .99 (min .98);
    # solidity .99 · .99 · .98 (min .98).
    #   area_px  = [⌊1020 − 3√480⌋, ⌈1220 + 3√480⌉] = [⌊954.2733⌋, ⌈1285.7267⌉] = [954, 1286]
    #   axis_ratio_min   = .98 − 3√.0002  ≈ .98 − .0424264 = .9375736
    #   solidity_min     = .98 − 3√.00008 = .98 − .0268328 = .9531672
    sigma_area = math.sqrt(2400 / 5)
    assert sigma_area == pytest.approx(21.908902300206645, abs=1e-9)
    sigma_axis = math.sqrt(0.0010 / 5)
    sigma_solidity = math.sqrt(0.0004 / 5)
    derived = derive_tolerances(HAND)
    assert isinstance(derived, Tolerances)
    assert derived.units == 3
    assert derived.frames == 8
    assert derived.means["u1"]["area_px"] == pytest.approx(1020.0, abs=1e-9)  # (1000+1020+1040)/3
    assert derived.means["u2"]["area_px"] == pytest.approx(1120.0, abs=1e-9)
    assert derived.means["u3"]["area_px"] == pytest.approx(1220.0, abs=1e-9)  # (1200+1240)/2
    assert derived.means["u3"]["axis_ratio"] == pytest.approx(0.99, abs=1e-9)  # (.98+1.00)/2
    assert derived.means["u3"]["solidity"] == pytest.approx(0.98, abs=1e-9)  # (.99+.97)/2
    assert derived.sigma["area_px"] == pytest.approx(sigma_area, abs=1e-9)
    assert derived.sigma["axis_ratio"] == pytest.approx(sigma_axis, abs=1e-9)
    assert derived.sigma["solidity"] == pytest.approx(sigma_solidity, abs=1e-9)
    assert derived.mean_range["area_px"] == pytest.approx((1020.0, 1220.0), abs=1e-9)
    assert derived.mean_range["axis_ratio"][0] == pytest.approx(0.98, abs=1e-9)
    assert derived.mean_range["solidity"][1] == pytest.approx(0.99, abs=1e-9)
    assert derived.config.area_px == (
        math.floor(1020.0 - 3.0 * sigma_area),  # ⌊min μᵤ − 3σₙ⌋, rounded outward
        math.ceil(1220.0 + 3.0 * sigma_area),  # ⌈max μᵤ + 3σₙ⌉
    )
    assert derived.config.area_px == (954, 1286)
    assert derived.config.axis_ratio_min == pytest.approx(0.98 - 3.0 * sigma_axis, abs=1e-9)
    assert derived.config.solidity_min == pytest.approx(0.98 - 3.0 * sigma_solidity, abs=1e-9)
    # The rounded-outward integer range contains the real one: 954 ≤ 954.2733 and 1285.7267 ≤ 1286.
    assert derived.config.area_px[0] <= 1020.0 - 3.0 * sigma_area
    assert derived.config.area_px[1] >= 1220.0 + 3.0 * sigma_area


def test_an_empty_input_is_refused() -> None:
    with pytest.raises(ValueError, match="no units"):
        derive_tolerances({})


def test_a_unit_with_one_rotation_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unit solo"):
        derive_tolerances({"solo": [_m(1000, 0.98, 0.99)], "u2": HAND["u2"]})


def test_a_spread_that_drives_a_minimum_to_zero_is_refused_by_measure() -> None:
    wide = [_m(1000, 0.05, 0.99), _m(1000, 0.15, 0.99)]
    with pytest.raises(ValueError, match="axis_ratio_min"):
        derive_tolerances({"a": wide, "b": wide})
    notched = [_m(1000, 0.98, 0.05), _m(1000, 0.98, 0.15)]
    with pytest.raises(ValueError, match="solidity_min"):
        derive_tolerances({"a": notched, "b": notched})


def test_an_area_minimum_at_or_below_zero_is_refused() -> None:
    tiny = [_m(10, 0.98, 0.99), _m(30, 0.98, 0.99)]
    with pytest.raises(ValueError, match="area_px"):
        derive_tolerances({"a": tiny, "b": tiny})


def test_unit_order_does_not_change_the_result_and_twice_is_equal() -> None:
    first = derive_tolerances(HAND)
    assert derive_tolerances(HAND_REORDERED) == first
    assert derive_tolerances(HAND) == first
    assert derive_tolerances(HAND_REORDERED) == derive_tolerances(HAND_REORDERED)


def test_the_report_mappings_are_read_only() -> None:
    derived = derive_tolerances(HAND)
    with pytest.raises(TypeError):
        derived.sigma["area_px"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        derived.means["u1"]["area_px"] = 0.0  # type: ignore[index]


def test_spread_of_equal_areas_is_zero() -> None:
    flat = [_m(1000, 0.98, 0.99), _m(1000, 0.98, 0.99), _m(1000, 0.98, 0.99)]
    assert rotation_spread({"u": flat}) == {"u": 0.0}


def test_spread_is_the_range_over_the_mean() -> None:
    areas = [_m(990, 0.98, 0.99), _m(1000, 0.98, 0.99), _m(1010, 0.98, 0.99)]
    got = rotation_spread({"u": areas})
    # (1010 − 990) / 1000 = 0.02 — exactly the pilot's silhouette bound of docs/13 §5.
    assert got == {"u": 0.02}


def test_a_one_rotation_unit_has_no_spread_and_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unit solo"):
        rotation_spread({"solo": [_m(1000, 0.98, 0.99)]})


def test_a_unit_whose_mean_area_is_zero_has_no_spread() -> None:
    empty = [_m(0, 0.98, 0.99), _m(0, 0.98, 0.99)]
    with pytest.raises(ValueError, match="unit empty"):
        rotation_spread({"empty": empty})


def _noisy_floor(rng: np.random.Generator) -> np.ndarray:
    """A grey-20 BGR floor with sensor noise, drawn in memory."""
    noise = np.clip(rng.normal(FLOOR_GREY, 4.0, FRAME_SHAPE), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def test_a_suggested_threshold_separates_disc_from_noisy_floor(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    rng = np.random.default_rng(0)
    frames: list[np.ndarray] = []
    for _ in range(10):
        pixels = _noisy_floor(rng)
        paint.disc(pixels, UNIT_CENTRE, 190, UNIT_GREY)
        frames.append(pixels)
    got = suggest_threshold(frames, localize_config.roi)
    assert isinstance(got, ThresholdSuggestion)
    assert 20 < got.value < 200
    histogram = got.histogram
    assert histogram.shape == (256,)
    assert int(histogram.sum()) == 1200 * 1080 * 10  # ROI pixels × frames
    assert not histogram.flags.writeable
    at_suggestion = replace(localize_config, threshold=int(got.value))
    for frame in frames:
        assert isinstance(localize(frame, at_suggestion), Silhouette)


def test_a_bright_square_outside_the_roi_changes_nothing(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    rng = np.random.default_rng(0)
    plain: list[np.ndarray] = []
    squared: list[np.ndarray] = []
    for _ in range(10):
        base = _noisy_floor(rng)
        paint.disc(base, UNIT_CENTRE, 190, UNIT_GREY)
        plain.append(base)
        pixels = base.copy()
        paint.blob(pixels, (1780, 40, 100, 100), 255)  # the grey card's corner, outside the ROI
        squared.append(pixels)
    got = suggest_threshold(plain, localize_config.roi)
    again = suggest_threshold(squared, localize_config.roi)
    assert again.value == got.value
    assert np.array_equal(again.histogram, got.histogram)


def test_no_frames_are_refused(localize_config: LocalizeConfig) -> None:
    with pytest.raises(ValueError, match="no frames"):
        suggest_threshold([], localize_config.roi)


_SHIFTS = ((0.0, 0.0, 0), (0.4, -0.3, 1), (-0.5, 0.3, 1))


def _good_frame(radius: int, shift: tuple[float, float, int]) -> np.ndarray:
    """A good unit drawn in one "rotation": a disc of `radius`, `minor` px flatter,
    shifted sub-pixel by (dx, dy)."""
    dx, dy, minor = shift
    cx, cy = UNIT_CENTRE[0] + dx, UNIT_CENTRE[1] + dy
    x0, y0 = int(cx - radius) - 2, int(cy - radius) - 2
    x1, y1 = int(cx + radius) + 3, int(cy + radius) + 3
    pixels = np.full(FRAME_SHAPE + (3,), FLOOR_GREY, np.uint8)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = ((xx - cx) / radius) ** 2 + ((yy - cy) / (radius - minor)) ** 2 <= 1.0
    pixels[y0:y1, x0:x1][inside] = UNIT_GREY
    return pixels


def test_the_derived_config_round_trips_through_the_loader_and_the_check(
    tmp_path: Path, localize_config: LocalizeConfig
) -> None:
    per_unit: dict[str, list[GeometryMeasures]] = {}
    for i in range(20):
        radius = 185 + (i * 7) % 11  # 20 units spread over r = 185..195
        per_unit[f"u{i:02d}"] = [
            measure(localize(_good_frame(radius, shift), localize_config)) for shift in _SHIFTS
        ]
    derived = derive_tolerances(per_unit)

    geometry = derived.config
    text = EXAMPLE.read_text(encoding="utf-8")
    for old, new in (
        (
            "area_px:        [98000, 131000]",
            f"area_px:        [{geometry.area_px[0]}, {geometry.area_px[1]}]",
        ),
        ("axis_ratio_min: 0.90", f"axis_ratio_min: {geometry.axis_ratio_min}"),
        ("solidity_min:   0.96", f"solidity_min:   {geometry.solidity_min}"),
    ):
        assert old in text
        text = text.replace(old, new)
    config_path = tmp_path / "derived.yml"
    config_path.write_text(text, encoding="utf-8")
    loaded = load_config(config_path)
    assert loaded.config.threshold.geometry == geometry

    for rotations in per_unit.values():
        for frame_measures in rotations:
            assert check_geometry(frame_measures, geometry).passed is True

    flattened = check_geometry(
        measure(localize(_good_frame(240, (0.0, 0.0, 0)), localize_config)), geometry
    )
    assert flattened.passed is False
    assert flattened.failed_on is not None and "area_px>max" in flattened.failed_on

    oval = check_geometry(
        measure(localize(_good_frame(240, (0.0, 0.0, 60)), localize_config)), geometry
    )
    assert oval.passed is False
    assert oval.failed_on is not None and "axis_ratio<min" in oval.failed_on
