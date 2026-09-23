"""The geometry check: the four measures against their analytic values, the rules and their
combinations against the example tolerances, the inclusive boundaries, the raise paths, the
label (C8), determinism (C1, C2) and the bench budget (docs/15 §3, docs/10 §5).

Every frame is drawn in memory with the shared `Paint` primitives on the black 1920×1080
floor, measured through the real chain — `localize` → `measure` → `check_geometry` — against
the example config's `threshold.geometry`. Nothing here needs a camera, a display or a GPU.
"""

import math
import statistics
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from conftest import FRAME_SHAPE, Paint
from qs.config import GeometryConfig, LocalizeConfig
from qs.contracts import CHECK_GEOMETRY, CheckResult, Label, derive_label
from qs.errors import InferenceError
from qs.inspector.geometry import GeometryMeasures, check_geometry, measure
from qs.inspector.localize import Silhouette, localize

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"

UNIT_CENTRE = (960, 540)  # the centre of the example ROI [360, 0, 1200, 1080]
UNIT_RADIUS = 180
UNIT_GREY = 200
DARK_GREY = 30
WEDGE_SOLIDITY = 2.880 / 3.130  # the analytic value for a 30° wedge cut to the centre


def _checked(
    pixels: np.ndarray,
    localize_config: LocalizeConfig,
    geometry_config: GeometryConfig,
) -> CheckResult:
    """The full chain over one drawn frame: localize, measure, check."""
    return check_geometry(measure(localize(pixels, localize_config)), geometry_config)


def _label(
    pixels: np.ndarray,
    localize_config: LocalizeConfig,
    geometry_config: GeometryConfig,
) -> Label:
    """The label `derive_label` reads off the geometry check alone (C8)."""
    result = _checked(pixels, localize_config, geometry_config)
    return derive_label((result,), frozenset({CHECK_GEOMETRY}))


def _silhouette_with(contour: np.ndarray) -> Silhouette:
    """A silhouette whose mask, box and centroid do not matter — only its contour."""
    return Silhouette(
        mask=np.zeros(FRAME_SHAPE, dtype=bool),
        contour=contour.reshape(-1, 1, 2).astype(np.int32),
        unit_box=(0, 0, 9, 9),
        centroid=(4.0, 4.0),
    )


def test_a_discs_measures_match_the_analytic_values(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    got = measure(localize(pixels, localize_config))
    disc_area = math.pi * UNIT_RADIUS**2
    assert isinstance(got.area_px, int)
    assert abs(got.area_px - disc_area) <= 0.01 * disc_area
    assert abs(got.diameter_px - 2 * UNIT_RADIUS) <= 0.01 * 2 * UNIT_RADIUS
    assert got.diameter_px == math.sqrt(4.0 * got.area_px / math.pi)
    assert got.axis_ratio >= 0.98
    assert got.solidity >= 0.98
    with pytest.raises(FrozenInstanceError):
        got.area_px = 0


def test_the_result_carries_the_measures_in_the_fixed_order(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    result = _checked(pixels, localize_config, geometry_config)
    assert result.name == CHECK_GEOMETRY
    assert result.value is None
    assert result.limit is None
    assert [name for name, _ in result.measures] == [
        "area_px",
        "diameter_px",
        "axis_ratio",
        "solidity",
    ]


def test_an_ellipses_axis_ratio_matches_its_axes(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.ellipse(pixels, UNIT_CENTRE, (200, 140), 0, UNIT_GREY)
    got = measure(localize(pixels, localize_config))
    assert abs(got.axis_ratio - 140 / 200) <= 0.02


def test_a_wedge_cut_discs_solidity_matches_the_analytic_value(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.wedge_disc(pixels, UNIT_CENTRE, UNIT_RADIUS, degrees=30)
    got = measure(localize(pixels, localize_config))
    assert abs(got.solidity - WEDGE_SOLIDITY) <= 0.01


def test_a_disc_of_r190_passes_the_example_config(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, 190, UNIT_GREY)
    result = _checked(pixels, localize_config, geometry_config)
    assert result.passed is True
    assert result.failed_on is None


def test_a_disc_of_r170_is_undersized(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, 170, UNIT_GREY)
    result = _checked(pixels, localize_config, geometry_config)
    assert result.passed is False
    assert result.failed_on == "area_px<min"


def test_a_disc_of_r210_is_oversized(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, 210, UNIT_GREY)
    result = _checked(pixels, localize_config, geometry_config)
    assert result.passed is False
    assert result.failed_on == "area_px>max"


def test_a_squeezed_ellipse_fails_the_axis_ratio(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.ellipse(pixels, UNIT_CENTRE, (220, 154), 0, UNIT_GREY)  # area π·220·154 ≈ 106 000
    result = _checked(pixels, localize_config, geometry_config)
    assert result.passed is False
    assert result.failed_on == "axis_ratio<min"


def test_a_wedge_cut_disc_fails_only_the_solidity(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.wedge_disc(pixels, UNIT_CENTRE, 190, degrees=30)
    result = _checked(pixels, localize_config, geometry_config)
    assert result.passed is False
    assert "solidity<min" in result.failed_on
    assert "area_px" not in result.failed_on


def test_a_large_oval_fails_area_and_axis_ratio(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.ellipse(pixels, UNIT_CENTRE, (260, 180), 0, UNIT_GREY)
    result = _checked(pixels, localize_config, geometry_config)
    assert result.passed is False
    assert result.failed_on == "area_px>max, axis_ratio<min"


def test_a_measure_exactly_at_the_tolerances_passes(
    geometry_config: GeometryConfig,
) -> None:
    area_min, area_max = geometry_config.area_px
    at_min = GeometryMeasures(
        area_px=area_min,
        diameter_px=math.sqrt(4.0 * area_min / math.pi),
        axis_ratio=geometry_config.axis_ratio_min,
        solidity=geometry_config.solidity_min,
    )
    result = check_geometry(at_min, geometry_config)
    assert result.passed is True
    assert result.failed_on is None
    at_max = replace(at_min, area_px=area_max)
    assert check_geometry(at_max, geometry_config).passed is True


def test_two_touching_discs_are_one_silhouette_that_fails_geometry(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, (760, 540), 220, UNIT_GREY)
    paint.disc(pixels, (1160, 540), 220, UNIT_GREY)
    found = localize(pixels, localize_config)
    assert isinstance(found, Silhouette)
    result = check_geometry(measure(found), geometry_config)
    assert result.passed is False
    assert result.failed_on is not None


def test_a_dark_interior_blob_changes_no_measure(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    solid = paint.floor()
    paint.disc(solid, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    torn = paint.floor()
    paint.disc(torn, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    paint.disc(torn, UNIT_CENTRE, 80, DARK_GREY)
    assert measure(localize(torn, localize_config)) == measure(
        localize(solid, localize_config)
    )


def test_a_four_point_contour_cannot_be_measured() -> None:
    square = np.array([[0, 0], [9, 0], [9, 9], [0, 9]])
    with pytest.raises(InferenceError):
        measure(_silhouette_with(square))


def test_a_degenerate_ellipse_fit_is_refused_not_passed(
    localize_config: LocalizeConfig, monkeypatch: pytest.MonkeyPatch, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    found = localize(pixels, localize_config)
    monkeypatch.setattr(
        "qs.inspector.geometry.cv2.fitEllipse",
        lambda contour: ((0.0, 0.0), (0.0, 0.0), 0.0),
    )
    with pytest.raises(InferenceError):
        measure(found)


@pytest.mark.parametrize("field", ("area_px", "diameter_px", "axis_ratio", "solidity"))
@pytest.mark.parametrize("bad", (math.nan, math.inf))
def test_a_non_finite_measure_is_refused_not_passed(
    geometry_config: GeometryConfig, field: str, bad: float
) -> None:
    base = GeometryMeasures(area_px=113_000, diameter_px=379.4, axis_ratio=0.99, solidity=0.99)
    with pytest.raises(InferenceError):
        check_geometry(replace(base, **{field: bad}), geometry_config)


def test_the_label_passes_a_good_disc_and_defects_every_failing_shape(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    good = paint.floor()
    paint.disc(good, UNIT_CENTRE, 190, UNIT_GREY)
    assert _label(good, localize_config, geometry_config) is Label.PASS

    undersized = paint.floor()
    paint.disc(undersized, UNIT_CENTRE, 170, UNIT_GREY)
    oversized = paint.floor()
    paint.disc(oversized, UNIT_CENTRE, 210, UNIT_GREY)
    squeezed = paint.floor()
    paint.ellipse(squeezed, UNIT_CENTRE, (220, 154), 0, UNIT_GREY)
    notched = paint.floor()
    paint.wedge_disc(notched, UNIT_CENTRE, 190, degrees=30)
    oval = paint.floor()
    paint.ellipse(oval, UNIT_CENTRE, (260, 180), 0, UNIT_GREY)
    stuck = paint.floor()
    paint.disc(stuck, (760, 540), 220, UNIT_GREY)
    paint.disc(stuck, (1160, 540), 220, UNIT_GREY)
    for pixels in (undersized, oversized, squeezed, notched, oval, stuck):
        assert _label(pixels, localize_config, geometry_config) is Label.DEFECT


def test_the_same_frame_measured_twice_gives_the_same_measures(
    localize_config: LocalizeConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)
    found = localize(pixels, localize_config)
    mask_before = found.mask.copy()
    contour_before = found.contour.copy()
    first = measure(found)
    second = measure(found)
    assert first == second
    assert np.array_equal(found.mask, mask_before)
    assert np.array_equal(found.contour, contour_before)


def test_localize_measure_and_check_fit_the_bench_budget(
    localize_config: LocalizeConfig, geometry_config: GeometryConfig, paint: Paint
) -> None:
    pixels = paint.floor()
    paint.disc(pixels, UNIT_CENTRE, UNIT_RADIUS, UNIT_GREY)

    def run() -> None:
        check_geometry(measure(localize(pixels, localize_config)), geometry_config)

    run()  # one warm-up before the 20 measured runs
    times_ms: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        run()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    median_ms = statistics.median(times_ms)
    print(f"bench: localize + measure + check median of 20 runs = {median_ms:.2f} ms")
    assert median_ms < 20.0
