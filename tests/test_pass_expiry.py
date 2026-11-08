"""ГОДЕН expiry (#104) at its edges: a ГОДЕН that failed to draw, and the station's clock.

Every test runs on a fake screen and a fake clock, with drawn frames only."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import FakeScreen
from evidence_records import fast_sqlite
from qs.config import StationConfig, load_config
from qs.contracts import (
    CHECK_EMPTY_CUP,
    CameraReadback,
    CheckResult,
    Frame,
    Label,
    TriggerEvent,
    Verdict,
)
from qs.operator_view import GREY, Status
from qs.screen_annunciator import ScreenAnnunciator
from screen_station import BAND, Rig, closing_after, presses
from station_fixtures import FakeClock, config_mapping, write_frames

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "DejaVuSans.ttf"
BENCH = load_config(ROOT / "config" / "station-A-bench.example.yml").config
READY = Status("ready", None, frozenset(), None)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def line_config(font: Path) -> StationConfig:
    display = dataclasses.replace(BENCH.display, font=str(font), pass_expiry_s=1.0)
    return dataclasses.replace(BENCH, display=display)


def a_pass() -> Verdict:
    check = CheckResult(
        name=CHECK_EMPTY_CUP, passed=True, value=0.6, limit=0.35,
        measures=(("rim_axis_a_px", 120.0), ("rim_axis_b_px", 80.0)), failed_on=None,
    )
    return Verdict(
        label=Label.PASS, checks=(check,), heatmap=None, window_box=None, unit_box=None,
        model_id="m", threshold_id="t", config_hash="h", inspect_ms=5.0, reason=None,
    )


def frame() -> Frame:
    event = TriggerEvent(7, 1.0, datetime(2026, 11, 6, 10, 0, tzinfo=UTC), "cup")
    pixels = np.full((300, 400, 3), (30, 200, 10), dtype=np.uint8)
    return Frame(event, pixels, 1.01, CameraReadback(10000, 0, 4500, 30, True), 0.62, 12.0)


def test_a_pass_that_failed_to_draw_stays_no_verdict_after_the_expiry(tmp_path: Path) -> None:
    clock = FakeClock(100.0)
    annunciator = ScreenAnnunciator(
        line_config(tmp_path / "missing.ttf"), FakeScreen(), clock=clock
    )
    with pytest.raises(OSError):
        annunciator.show(frame(), a_pass())
    clock.advance(60.0)
    with pytest.raises(OSError):
        annunciator.refresh(READY, None)
    shown = annunciator.last_shown
    assert shown is not None
    assert (shown.seq, shown.label) == (7, Label.NO_VERDICT)


def test_the_station_screen_expires_a_pass_on_the_station_clock(tmp_path: Path) -> None:
    data = config_mapping(tmp_path)
    data["display"]["pass_expiry_s"] = 50.0  # far beyond the test's real run time
    config = tmp_path / "station.yml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    shoot = tmp_path / "shoot"
    write_frames(shoot, 1)
    rig = Rig(oversleep=100.0)  # each 30 ms turn of the loop moves the station clock 100 s
    code = rig.run(
        config, shoot, screen_options=closing_after(presses(1)), make_screen_annunciator=None
    )
    assert code == 0
    assert rig.screens[0].presented[-1].getpixel(BAND) == GREY
