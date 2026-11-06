"""The operator screen on the line (#104): ГОДЕН expires, and the line's own wording.

Every test runs on a fake screen and a fake clock, with drawn frames only."""

from __future__ import annotations

import dataclasses
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeScreen
from qs.config import ConfigError, StationConfig, load_config
from qs.contracts import (
    CHECK_EMPTY_CUP,
    CameraReadback,
    CheckResult,
    Frame,
    Label,
    TriggerEvent,
    Verdict,
)
from qs.operator_view import AMBER, GREEN, GREY, RED, Status, status_line
from qs.screen_annunciator import ScreenAnnunciator

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "DejaVuSans.ttf"
BENCH = load_config(ROOT / "config" / "station-A-bench.example.yml").config
READY = Status("ready", None, frozenset(), None)
BAND = (4, 40)  # a pixel of the verdict band


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def line_config(pass_expiry_s: float | None = 1.0) -> StationConfig:
    display = dataclasses.replace(
        BENCH.display, font=str(FONT), pass_expiry_s=pass_expiry_s,
        ready_line="ГОТОВ", placement_line="стаканчик не проверен",
    )
    return dataclasses.replace(BENCH, display=display)


def bench_config() -> StationConfig:
    return dataclasses.replace(BENCH, display=dataclasses.replace(BENCH.display, font=str(FONT)))


def frame() -> Frame:
    event = TriggerEvent(7, 1.0, datetime(2026, 11, 6, 10, 0, tzinfo=UTC), "cup")
    pixels = np.full((300, 400, 3), (30, 200, 10), dtype=np.uint8)
    return Frame(event, pixels, 1.01, CameraReadback(10000, 0, 4500, 30, True), 0.62, 12.0)


def empty_cup(passed: bool) -> CheckResult:
    return CheckResult(
        name=CHECK_EMPTY_CUP, passed=passed, value=0.6 if passed else 0.1, limit=0.35,
        measures=(("rim_axis_a_px", 120.0), ("rim_axis_b_px", 80.0)),
        failed_on=None if passed else "blob_share<min",
    )


def verdict(label: Label, reason: str | None = None) -> Verdict:
    checks = () if label is Label.NO_VERDICT else (empty_cup(label is Label.PASS),)
    return Verdict(
        label=label, checks=checks, heatmap=None, window_box=None, unit_box=None, model_id="m",
        threshold_id="t", config_hash="h", inspect_ms=5.0, reason=reason,
    )


@pytest.mark.parametrize(
    ("expiry_s", "label", "reason", "elapsed_s", "band"),
    [
        (1.0, Label.PASS, None, 0.9, GREEN),
        (1.0, Label.PASS, None, 1.0, GREY),
        (1.0, Label.PASS, None, 60.0, GREY),
        (1.0, Label.DEFECT, None, 60.0, RED),
        (1.0, Label.NO_VERDICT, "placement", 60.0, AMBER),
        (None, Label.PASS, None, 3600.0, GREEN),  # the bench: ГОДЕН never expires
    ],
)
def test_only_a_pass_expires_and_only_where_the_display_sets_an_expiry(
    expiry_s: float | None, label: Label, reason: str | None, elapsed_s: float,
    band: tuple[int, int, int],
) -> None:
    clock = FakeClock(100.0)
    screen = FakeScreen()
    annunciator = ScreenAnnunciator(line_config(expiry_s), screen, clock=clock)
    annunciator.show(frame(), verdict(label, reason))
    clock.now = 100.0 + elapsed_s
    annunciator.refresh(READY, None)
    assert screen.presented[-1].getpixel(BAND) == band


@pytest.mark.parametrize(
    ("station", "detail"),
    [
        (line_config(), "стаканчик не проверен"),
        (bench_config(), "положите изделие в рамку"),
    ],
)
def test_a_cup_seen_but_not_judged_is_worded_by_the_display_config(
    station: StationConfig, detail: str
) -> None:
    annunciator = ScreenAnnunciator(station, FakeScreen(), clock=FakeClock(0.0))
    annunciator.show(frame(), verdict(Label.NO_VERDICT, "placement"))
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.detail == detail


@pytest.mark.parametrize(
    ("station", "line"),
    [
        (line_config(), "ГОТОВ"),
        (bench_config(), "ГОТОВ — положите изделие в рамку и нажмите ПРОБЕЛ"),
    ],
)
def test_the_ready_status_line_comes_from_the_display_config(
    station: StationConfig, line: str
) -> None:
    display = station.display
    assert status_line(READY, display.labels, ready=display.ready_line) == line


def status_bar(station: StationConfig) -> np.ndarray:
    screen = FakeScreen()
    ScreenAnnunciator(station, screen, clock=FakeClock(0.0)).refresh(READY, None)
    return np.asarray(screen.presented[-1])[740:800]


def test_the_line_ready_line_reaches_the_screen() -> None:
    assert not np.array_equal(status_bar(line_config()), status_bar(bench_config()))


def test_an_empty_cup_defect_says_so() -> None:
    annunciator = ScreenAnnunciator(line_config(), FakeScreen(), clock=FakeClock(0.0))
    annunciator.show(frame(), verdict(Label.DEFECT))
    shown = annunciator.last_shown
    assert shown is not None
    assert shown.detail == "пустой стаканчик"


@pytest.mark.parametrize(
    ("label", "reason"), [(Label.DEFECT, None), (Label.NO_VERDICT, "placement")]
)
def test_the_line_draws_no_green_for_anything_but_a_pass(label: Label, reason: str | None) -> None:
    screen = FakeScreen()
    annunciator = ScreenAnnunciator(line_config(), screen, clock=FakeClock(0.0))
    annunciator.show(frame(), verdict(label, reason))
    annunciator.refresh(READY, None)
    presented = np.asarray(screen.presented[-1])
    assert not np.any(np.all(presented == np.array(GREEN, dtype=np.uint8), axis=-1))


LINE_EXAMPLE = ROOT / "config" / "station-line.example.yml"
BENCH_EXAMPLE = ROOT / "config" / "station-A-bench.example.yml"


@pytest.mark.parametrize(
    ("path", "expiry_s", "ready", "placement"),
    [
        (LINE_EXAMPLE, 1.0, "ГОТОВ", "стаканчик не проверен"),
        (BENCH_EXAMPLE, None, None, None),
    ],
)
def test_the_display_config_loads_the_line_wording_and_leaves_the_bench_alone(
    path: Path, expiry_s: float | None, ready: str | None, placement: str | None
) -> None:
    display = load_config(path).config.display
    assert (display.pass_expiry_s, display.ready_line, display.placement_line) == (
        expiry_s, ready, placement,
    )


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("pass_expiry_s: 1.0", "pass_expiry_s: 0", "display.pass_expiry_s"),
        ("pass_expiry_s: 1.0", "pass_expiry_s: -1", "display.pass_expiry_s"),
        ("pass_expiry_s: 1.0", "pass_expiry_s: .nan", "display.pass_expiry_s"),
        ("pass_expiry_s: 1.0", "pass_expiry_s: .inf", "display.pass_expiry_s"),
        ("pass_expiry_s: 1.0", "pass_expiry_s: soon", "display.pass_expiry_s"),
        ("ready_line: ГОТОВ", "ready_line: '  '", "display.ready_line"),
        ("placement_line: стаканчик не проверен", "placement_line: ''",
         "display.placement_line"),
    ],
)
def test_a_bad_line_display_value_is_refused_at_load(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    text = LINE_EXAMPLE.read_text(encoding="utf-8")
    path = tmp_path / "station.yml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape(match)):
        load_config(path)
