"""ScreenAnnunciator (#87): every verdict becomes an operator view, shown on a fake screen."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeScreen
from qs.config import StationConfig, load_config
from qs.contracts import (
    CHECK_APPEARANCE,
    CHECK_GEOMETRY,
    CameraReadback,
    CheckResult,
    Frame,
    Label,
    TriggerEvent,
    Verdict,
)
from qs.inspector.heatmap import heatmap_overlay
from qs.operator_view import AMBER, GREEN, RED, Status, View, reason_line, render
from qs.screen_annunciator import ScreenAnnunciator

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config" / "station-A-bench.example.yml"
FONT = ROOT / "assets" / "DejaVuSans.ttf"
BASE = load_config(EXAMPLE).config
THRESHOLD = BASE.threshold.value  # 1.83
READY = Status("ready", None, frozenset(), None)
REASONS = ("placement", "lighting", "capture", "camera_settings", "inference", "camera", "busy")
NO_PICTURE = ("camera", "capture", "camera_settings", "busy")


def config(font: Path = FONT) -> StationConfig:
    display = dataclasses.replace(BASE.display, font=str(font))
    return dataclasses.replace(BASE, display=display)


def frame(pixels: np.ndarray, seq: int = 5) -> Frame:
    event = TriggerEvent(seq, 1.0, datetime(2026, 9, 27, 10, 0, tzinfo=UTC), "keyboard:space")
    readback = CameraReadback(10000, 0, 4500, 30, True)
    return Frame(event, pixels, 1.01, readback, 0.62, 12.0)


def uniform(bgr: tuple[int, int, int], shape: tuple[int, int] = (300, 400)) -> np.ndarray:
    return np.full((*shape, 3), bgr, dtype=np.uint8)


def geometry(failed_on: str | None = None, *, area: float = 110000.0,
             ratio: float = 0.97, solidity: float = 0.98) -> CheckResult:
    return CheckResult(
        name=CHECK_GEOMETRY,
        passed=failed_on is None,
        value=None,
        limit=None,
        measures=(("area_px", area), ("diameter_px", 374.0), ("axis_ratio", ratio),
                  ("solidity", solidity)),
        failed_on=failed_on,
    )


def appearance(value: float | None = 0.9, limit: float | None = THRESHOLD) -> CheckResult:
    passed = value is not None and limit is not None and value <= limit
    return CheckResult(
        name=CHECK_APPEARANCE, passed=passed, value=value, limit=limit,
        measures=(("pred_score", 0.5),), failed_on=None if passed else "score>threshold",
    )


def verdict(label: Label, checks: tuple[CheckResult, ...] = (), *, reason: str | None = None,
            heatmap: np.ndarray | None = None,
            window_box: tuple[int, int, int, int] | None = None,
            unit_box: tuple[int, int, int, int] | None = None) -> Verdict:
    if label is not Label.NO_VERDICT and not checks:
        checks = (geometry(), appearance())
    return Verdict(
        label=label, checks=checks, heatmap=heatmap, window_box=window_box,
        unit_box=unit_box, model_id="m", threshold_id="t", config_hash="h", inspect_ms=5.0,
        reason=reason,
    )


def no_verdict(reason: str) -> Verdict:
    return verdict(Label.NO_VERDICT, (), reason=reason)


def has_colour(image: np.ndarray, colour: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(image == np.array(colour, dtype=np.uint8), axis=-1)))


def make() -> tuple[ScreenAnnunciator, FakeScreen]:
    screen = FakeScreen()
    return ScreenAnnunciator(config(), screen), screen


# AC1 ------------------------------------------------------------------------------------


def test_a_pass_is_shown_green_with_its_frame() -> None:
    annunciator, screen = make()
    annunciator.show(frame(uniform((30, 200, 10))), verdict(Label.PASS))
    assert len(screen.presented) == 1
    presented = screen.presented[0]
    assert presented.getpixel((4, 40)) == GREEN
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.seq == 5
    assert shown.detail == ""
    assert shown.image is not None
    assert np.array_equal(shown.image, uniform((10, 200, 30)))
    assert annunciator.counts == {Label.PASS: 1}
    assert annunciator.recent == ((5, Label.PASS),)


# AC2 ------------------------------------------------------------------------------------


def test_an_appearance_defect_shows_the_heatmap_over_the_window() -> None:
    pixels = np.random.default_rng(3).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:100, 0:100]
    heatmap = np.where((yy - 50) ** 2 + (xx - 50) ** 2 <= 30**2, 2 * THRESHOLD, 0.0)
    heatmap = heatmap.astype(np.float32)
    cut = np.zeros((100, 100, 3), np.uint8)
    cut[:, 20:] = pixels[50:150, 0:80]  # the box starts 20 px left of the frame
    annunciator, screen = make()
    annunciator.show(
        frame(pixels),
        verdict(Label.DEFECT, (geometry(), appearance(2 * THRESHOLD)), heatmap=heatmap,
                window_box=(-20, 50, 100, 100), unit_box=(0, 60, 70, 80)),
    )
    shown = annunciator.last_shown
    assert shown is not None and shown.image is not None
    assert np.array_equal(shown.image, heatmap_overlay(cut, heatmap, THRESHOLD, heatmap != 0))
    assert screen.presented[-1].getpixel((4, 40)) == RED


# AC3 ------------------------------------------------------------------------------------


def test_a_geometry_defect_names_the_failed_measures_and_outlines_the_unit() -> None:
    annunciator, _ = make()
    check = geometry("area_px<min, solidity<min", area=90000.0, solidity=0.95)
    annunciator.show(
        frame(uniform((30, 200, 10))),
        verdict(Label.DEFECT, (check, appearance()), unit_box=(100, 80, 120, 100)),
    )
    shown = annunciator.last_shown
    assert shown is not None and shown.image is not None
    assert shown.detail == "площадь 90000 px < 98000 · выпуклость 0.950 < 0.960"
    assert tuple(shown.image[130, 100]) == RED  # the left edge of the outline
    assert tuple(shown.image[80, 160]) == RED  # the top edge
    assert tuple(shown.image[130, 160]) == (10, 200, 30)  # inside, the frame


def test_every_geometry_item_is_worded() -> None:
    annunciator, _ = make()
    check = geometry("area_px>max, axis_ratio<min", area=140000.0, ratio=0.8)
    annunciator.show(frame(uniform((30, 200, 10))),
                     verdict(Label.DEFECT, (check, appearance()), unit_box=(1, 1, 5, 5)))
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.detail == "площадь 140000 px > 131000 · соотношение осей 0.80 < 0.90"


# AC4 ------------------------------------------------------------------------------------


def test_appearance_comes_first_when_both_checks_failed() -> None:
    annunciator, _ = make()
    checks = (geometry("area_px<min", area=90000.0), appearance(2.10))
    annunciator.show(
        frame(uniform((30, 200, 10))),
        verdict(Label.DEFECT, checks, heatmap=np.zeros((100, 100), np.float32),
                window_box=(10, 10, 100, 100), unit_box=(20, 20, 80, 80)),
    )
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.detail == "внешний вид: 2.10 > порог 1.83 · площадь 90000 px < 98000"


# AC5 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", REASONS)
def test_no_verdict_is_amber_with_its_reason_and_never_green(reason: str) -> None:
    annunciator, screen = make()
    annunciator.show(frame(uniform((30, 200, 10))), no_verdict(reason))
    presented = np.asarray(screen.presented[-1])
    assert screen.presented[-1].getpixel((4, 40)) == AMBER
    assert not has_colour(presented, GREEN)
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.detail == reason_line(reason)
    if reason in NO_PICTURE:
        assert shown.image is None
    else:
        assert shown.image is not None


def test_a_no_verdict_without_a_frame_has_no_seq_and_no_image() -> None:
    annunciator, _ = make()
    annunciator.show(None, no_verdict("capture"))
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.seq is None
    assert shown.image is None


# AC6 ------------------------------------------------------------------------------------

MAGENTA_BGR = (255, 0, 255)
MAGENTA_RGB = (255, 0, 255)


def test_the_live_picture_never_becomes_the_verdict_image() -> None:
    annunciator, screen = make()
    annunciator.refresh(READY, uniform(MAGENTA_BGR))
    assert screen.presented[-1].getpixel((320, 456)) == MAGENTA_RGB
    annunciator.show(frame(uniform((30, 200, 10))), verdict(Label.PASS))
    shown = annunciator.last_shown
    assert shown is not None and shown.image is not None
    assert not has_colour(shown.image, MAGENTA_RGB)
    right_box = np.asarray(screen.presented[-1])[236:676, 640:1280]
    assert not has_colour(right_box, MAGENTA_RGB)


# AC7 ------------------------------------------------------------------------------------


def test_a_verdict_during_a_fault_shows_the_fault_screen_and_is_counted() -> None:
    annunciator, screen = make()
    annunciator.refresh(Status("fault", "capture", frozenset(), None), None)
    annunciator.show(None, no_verdict("capture"))
    presented = screen.presented[-1]
    for point in ((4, 4), (1275, 4), (4, 795), (1275, 795)):
        assert presented.getpixel(point) == AMBER
    assert not has_colour(np.asarray(presented), GREEN)
    assert annunciator.counts[Label.NO_VERDICT] == 1


# AC8 ------------------------------------------------------------------------------------


def test_recent_keeps_the_newest_ten_and_counts_every_call() -> None:
    annunciator, _ = make()
    for seq in range(1, 12):
        label_verdict = verdict(Label.PASS) if seq % 2 else no_verdict("placement")
        annunciator.show(frame(uniform((30, 200, 10)), seq), label_verdict)
    assert [seq for seq, _ in annunciator.recent] == list(range(2, 12))
    assert sum(annunciator.counts.values()) == 11


# AC9 ------------------------------------------------------------------------------------


def test_a_drawing_failure_leaves_a_plain_amber_screen_and_still_counts(
    tmp_path: Path,
) -> None:
    screen = FakeScreen()
    annunciator = ScreenAnnunciator(config(tmp_path / "missing.ttf"), screen)
    with pytest.raises(OSError):
        annunciator.show(frame(uniform((30, 200, 10))), verdict(Label.PASS))
    last = np.asarray(screen.presented[-1])
    assert last.shape == (800, 1280, 3)
    assert np.all(last == np.array(AMBER, dtype=np.uint8))
    assert annunciator.counts == {Label.PASS: 1}


def test_a_failed_show_never_brings_the_previous_verdict_back() -> None:
    annunciator, screen = make()
    annunciator.show(frame(uniform((30, 200, 10)), 1), verdict(Label.PASS))
    broken = verdict(Label.DEFECT, (geometry(), appearance(None, None)))  # no value to word
    with pytest.raises(TypeError):
        annunciator.show(frame(uniform((30, 200, 10)), 2), broken)
    annunciator.refresh(READY, None)
    band = screen.presented[-1].getpixel((4, 40))
    assert band != GREEN
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.seq == 2
    assert shown.label is Label.NO_VERDICT


# AC10 -----------------------------------------------------------------------------------


def test_close_twice_closes_the_screen_once() -> None:
    annunciator, screen = make()
    annunciator.close()
    annunciator.close()
    assert screen.closed
    assert screen.close_calls == 1


# REVIEW-P09 m1: the verdict image is fitted once per show(), and no pixel moves ----------


def full_hd() -> np.ndarray:
    """A 1920 × 1080 BGR frame with detail at every scale, so any change in the fitting
    filter would change the drawn pixels."""
    yy, xx = np.mgrid[0:1080, 0:1920]
    return np.dstack([xx % 256, yy % 256, (xx * 7 + yy * 3) % 256]).astype(np.uint8)


def full_hd_verdict(kind: str) -> Verdict:
    if kind == "pass":
        return verdict(Label.PASS)
    if kind == "no_verdict":
        return no_verdict("placement")
    yy, xx = np.mgrid[0:448, 0:448]
    heatmap = np.where((yy - 224) ** 2 + (xx - 224) ** 2 <= 120**2, 2 * THRESHOLD, 0.0)
    return verdict(
        Label.DEFECT, (geometry(), appearance(2 * THRESHOLD)),
        heatmap=heatmap.astype(np.float32), window_box=(736, 316, 448, 448),
        unit_box=(760, 340, 400, 400),
    )


# sha256 of the images presented by show() and by the refresh after it (READY, no live
# picture), recorded with the code before the m1 fix (b9d5992). Pinned to this venv's
# Pillow and assets/DejaVuSans.ttf: a Pillow upgrade may redraw the text.
BEFORE_M1 = {
    "pass": (
        "8f1fcf8f78589e4c71fc55e5ac3526f1f6d7913874cb309397198c9564bdfd9e",
        "537e875c604170bc123f9eeca0b3bea0770f87759e17a3692405e9c784a65115",
    ),
    "defect_heatmap": (
        "dc124353338829d0fb4e47866bab099dd02f3c98a7adeae91690946090255e1b",
        "a20e78d10ace2a3437d0e61d3c45ba6cfdc87af6fe1fc556acc36329bd1456c4",
    ),
    "no_verdict": (
        "fbdf2adf219696564ef5eb259774b242b9282a54f47bc24e0f1ec0bff5a78b4c",
        "ff1e83f936ad62a9df172c97fd96e5944ce1843ed2dc25791fa75923e1054cab",
    ),
}


def digest(image: object) -> str:
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def render_uncached(annunciator: ScreenAnnunciator) -> object:
    """The refresh screen drawn from scratch with no cache: what every refresh must match."""
    view = View(READY, None, annunciator.last_shown, annunciator.counts, annunciator.recent)
    return render(view, BASE.display.labels, FONT)


@pytest.mark.parametrize("kind", sorted(BEFORE_M1))
def test_a_full_hd_verdict_draws_the_same_pixels_as_before_the_fit_cache(kind: str) -> None:
    annunciator, screen = make()
    annunciator.show(frame(full_hd(), 3), full_hd_verdict(kind))
    for _ in range(3):
        annunciator.refresh(READY, None)
    shown, refreshed = BEFORE_M1[kind]
    drawn = [digest(image) for image in screen.presented]
    assert drawn == [shown, refreshed, refreshed, refreshed]
    assert digest(render_uncached(annunciator)) == refreshed


def test_a_new_verdict_is_never_drawn_with_the_previous_verdicts_image() -> None:
    annunciator, screen = make()
    green, magenta = (10, 200, 30), MAGENTA_RGB
    annunciator.show(frame(uniform((30, 200, 10)), 1), verdict(Label.PASS))
    annunciator.refresh(READY, None)
    first = np.asarray(screen.presented[-1])[236:676, 640:1280]
    annunciator.show(frame(uniform(MAGENTA_BGR), 1), verdict(Label.PASS))  # same seq, too
    second_show = np.asarray(screen.presented[-1])[236:676, 640:1280]
    annunciator.refresh(READY, None)
    second = np.asarray(screen.presented[-1])[236:676, 640:1280]
    assert has_colour(first, green) and not has_colour(first, magenta)
    for box in (second_show, second):
        assert has_colour(box, magenta) and not has_colour(box, green)
    assert digest(screen.presented[-1]) == digest(render_uncached(annunciator))
