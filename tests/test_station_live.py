"""Live `qs station --config C` (#90): the webcam station, built and tested with fakes until the
webcam arrives (#85, A1).

Through `main(args, *, factories)` with a fake camera (scripted `open()`, uniform `capture()`
and `preview()` frames), a recording stub inspector under an `appearance` config, a
`FakeScreen` on the fake clock and the real `SqliteSink` on `tmp_path`."""

from __future__ import annotations

import json
import tkinter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from evidence_records import fast_sqlite
from qs.config import DisplayConfig, LoadedConfig
from qs.contracts import CameraReadback, Frame, Health, TriggerEvent, Verdict
from qs.errors import CameraLost, CameraSettingsDrift
from qs.inspector.stub import StubInspector
from qs.operator_view import AMBER, BANNER, banner_line, fault_line
from qs.pipeline import CAMERA_SETTINGS
from qs.screen import KeyPress, ScreenLike
from qs.sources.factory import frame_source
from qs.sources.v4l2 import ControlReading
from screen_station import (
    GAP,
    RecordingSqlite,
    Rig,
    closing_after,
    database,
    presses,
    recording,
    run_row,
    space,
)
from station_fixtures import HEIGHT, PATCH_GREY, UNVERIFIED, VERIFIED, WIDTH, config_mapping, rows

MAGENTA = (255, 0, 255)  # BGR and RGB alike
CAPTURED = (90, 140, 110)  # BGR: a green-grey
CAPTURED_RGB = CAPTURED[::-1]


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def live_config(root: Path, name: str = "station.yml", **changes: Any) -> Path:
    """P04's small config as a live one: keyboard trigger, webcam, `appearance` inspector;
    `changes` set `section.key` values."""
    data = config_mapping(root, inspector_type="appearance")
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        data[section][key] = value
    path = root / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def uniform(bgr: tuple[int, int, int]) -> np.ndarray:
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[:] = bgr
    return pixels


@dataclass
class FakeCamera:
    """A live `FrameSource`: `opens` scripts successive `open()` calls (None succeeds);
    `capture()` gives a uniform green-grey frame; `preview()` a uniform magenta one, or
    raises on the calls named in `preview_fails`. `readings` stands in for the webcam's."""

    clock: Callable[[], float]
    opens: list[BaseException | None] = field(default_factory=list)
    camera: CameraReadback = VERIFIED
    preview_fails: set[int] = field(default_factory=set)
    readings: tuple[ControlReading, ...] = ()
    previews: int = 0
    open_calls: int = 0

    def open(self) -> None:
        self.open_calls += 1
        outcome = self.opens.pop(0) if self.opens else None
        if outcome is not None:
            raise outcome

    def capture(self, event: TriggerEvent) -> Frame:
        return Frame(
            event=event,
            pixels=uniform(CAPTURED),
            grabbed_at=self.clock(),
            camera=self.camera,
            reference_patch=PATCH_GREY / 255,
            capture_ms=40.0,
        )

    def preview(self) -> np.ndarray | None:
        self.previews += 1
        if self.previews in self.preview_fails:
            raise RuntimeError("the grab thread's newest frame is gone (injected)")
        return uniform(MAGENTA)

    def health(self) -> Health:
        return Health(True, "fake")

    def close(self) -> None:
        pass


class RecordingInspector(StubInspector):
    """The stub's verdicts, keeping the mean colour of every frame it was given."""

    def __init__(self, config_hash: str, script: dict[int, str] | None = None) -> None:
        super().__init__(config_hash, script or {})
        self.seen: list[tuple[float, ...]] = []

    def inspect(self, frame: Frame) -> Verdict:
        self.seen.append(tuple(float(v) for v in frame.pixels.reshape(-1, 3).mean(axis=0)))
        return super().inspect(frame)


@dataclass
class Live:
    """A live run's fakes."""

    rig: Rig = field(default_factory=Rig)
    camera: FakeCamera | None = None
    inspectors: list[RecordingInspector] = field(default_factory=list)
    sinks: list[RecordingSqlite] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.camera is None:
            self.camera = FakeCamera(self.rig.clock)

    def inspector(self, loaded: LoadedConfig) -> RecordingInspector:
        inspector = RecordingInspector(loaded.config_hash)
        self.inspectors.append(inspector)
        return inspector

    def run(self, config: Path, **options: Any) -> int:
        options.setdefault("make_inspector", self.inspector)
        options.setdefault("make_source", lambda config: self.camera)
        options.setdefault("make_sink", recording(self.sinks))
        return self.rig.run(config, None, **options)


# AC1 ------------------------------------------------------------------------------------


def test_three_presses_make_three_live_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    live = Live()
    code = live.run(live_config(tmp_path), screen_options=closing_after(presses(3)))
    assert code == 0
    records = rows(database(tmp_path), "SELECT * FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert [r.event.source_id for r in live.sinks[0].written] == ["keyboard:space"] * 3
    run = run_row(tmp_path)
    assert run["ended_utc"] is not None
    counters = json.loads(run["counters"])
    assert counters["source_id"] == "keyboard:space" and counters["keys_discarded"] == 0
    spy = live.rig.spies[0]
    assert all(s.caption is None for s in spy.statuses)  # no replay caption live
    assert live.rig.screens[0].close_calls == 1
    assert "events 3 · pass 3" in capsys.readouterr().out


# AC2 ------------------------------------------------------------------------------------


def test_a_preview_frame_is_shown_live_and_never_inspected_recorded_or_shown(
    tmp_path: Path,
) -> None:
    live = Live()
    assert live.run(live_config(tmp_path), screen_options=closing_after(presses(3))) == 0
    spy = live.rig.spies[0]
    ready = [i for i, s in enumerate(spy.statuses) if s.mode == "ready"]
    refreshes = [i for i, (kind, _, _) in enumerate(spy.timeline) if kind == "refresh"]
    assert ready and all(spy.live_box[refreshes[i]] == MAGENTA for i in ready)
    for record in rows(database(tmp_path), "SELECT * FROM inspection"):
        image = cv2.imread(str(tmp_path / "evidence" / record["image_path"]))
        mean = image.reshape(-1, 3).mean(axis=0)
        assert np.allclose(mean, CAPTURED, atol=10), mean
    for shown in spy.shown:
        assert shown is not None and shown.image is not None
        assert not bool(np.all(shown.image == MAGENTA, axis=-1).any())
    assert live.inspectors[0].seen == [tuple(float(v) for v in CAPTURED)] * 3


# AC3 ------------------------------------------------------------------------------------


def test_a_camera_not_found_at_start_shows_the_fault_screen_until_it_is_back(
    tmp_path: Path,
) -> None:
    live = Live()
    assert live.camera is not None
    live.camera.opens = [CameraLost("/dev/video0: cannot open"), None]
    # One press in FAULT, then 40 idle iterations (1.2 s: past the 0.5 s backoff), one more.
    script: list[list[KeyPress]] = [[], [space()]]
    script.extend([] for _ in range(40))
    script.append([space()])
    script.extend([] for _ in range(GAP))
    code = live.run(live_config(tmp_path), screen_options=closing_after(script))
    assert code == 0  # back to READY before the stop
    spy = live.rig.spies[0]
    refreshes = [i for i, (kind, _, _) in enumerate(spy.timeline) if kind == "refresh"]
    first = spy.statuses[1]  # the first refresh after start-up (the 0th is ЗАПУСК)
    assert first.mode == "fault" and fault_line(first.fault_reason) == "камера не подключена"
    assert spy.corners[refreshes[1]] == {AMBER}  # the FAULT screen
    records = rows(database(tmp_path), "SELECT * FROM inspection ORDER BY seq")
    assert [(r["seq"], r["label"], r["reason"]) for r in records] == [
        (1, "no_verdict", "camera"),
        (2, "pass", None),
    ]
    assert live.camera.open_calls == 2
    ready = [i for i, s in enumerate(spy.statuses) if s.mode == "ready"]
    assert ready and spy.live_box[refreshes[ready[0]]] == MAGENTA  # the live view is back
    assert json.loads(run_row(tmp_path)["counters"])["reconnect_attempts"] == 1


# AC4 ------------------------------------------------------------------------------------


def test_a_read_back_mismatch_at_open_stops_with_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    live = Live()
    assert live.camera is not None
    live.camera.opens = [CameraSettingsDrift("exposure: set 10000, reads 9984")]
    assert live.run(live_config(tmp_path), screen_options=closing_after(presses(1))) == 2
    assert "exposure: set 10000, reads 9984" in capsys.readouterr().out
    assert live.rig.screens[0].close_calls == 1
    assert rows(database(tmp_path), "SELECT * FROM inspection") == []


# AC5 ------------------------------------------------------------------------------------


def test_an_unverified_read_back_raises_the_camera_settings_banner_for_good(
    tmp_path: Path,
) -> None:
    live = Live()
    assert live.camera is not None
    live.camera.camera = UNVERIFIED
    assert live.run(live_config(tmp_path), screen_options=closing_after(presses(2))) == 0
    spy = live.rig.spies[0]
    [first] = [i for i, (kind, seq, _) in enumerate(spy.timeline) if kind == "show" and seq == 1]
    after = [i for i in range(first + 1, len(spy.timeline)) if spy.timeline[i][0] == "refresh"]
    statuses = spy.statuses[-len(after):]
    assert all(CAMERA_SETTINGS in s.warnings for s in statuses)
    assert "настройки камеры не подтверждены" in banner_line(statuses[0].warnings)
    assert all(spy.banner[i] == BANNER for i in after)


def test_settings_that_cannot_be_read_at_open_start_with_the_banner(tmp_path: Path) -> None:
    """docs/16 §2 step 3 (REVIEW-P04 n3): the banner is up from the first READY refresh,
    before any unit is checked."""
    live = Live()
    assert live.camera is not None
    live.camera.readings = (
        ControlReading("exposure_us", 10000, 10000),
        ControlReading("gain", 0, None),  # cannot be read
    )
    assert live.run(live_config(tmp_path), screen_options=closing_after([[], []])) == 0
    spy = live.rig.spies[0]
    ready = [s for s in spy.statuses if s.mode == "ready"]
    assert ready and all(CAMERA_SETTINGS in s.warnings for s in ready)


def test_settings_all_read_back_at_open_raise_no_banner(tmp_path: Path) -> None:
    live = Live()
    assert live.camera is not None
    live.camera.readings = tuple(
        ControlReading(name, 1, 1)
        for name in ("a", "b", "c", "d", "e", "f", "g", "h")  # the webcam reads eight
    )
    assert live.run(live_config(tmp_path), screen_options=closing_after([[], []])) == 0
    assert all(CAMERA_SETTINGS not in s.warnings for s in live.rig.spies[0].statuses)


# AC6 ------------------------------------------------------------------------------------


def test_a_preview_that_raises_draws_no_picture_and_the_loop_goes_on(tmp_path: Path) -> None:
    live = Live()
    assert live.camera is not None
    live.camera.preview_fails = {3}  # call 1 is start-up's reference patch; 3 the 2nd refresh
    assert live.run(live_config(tmp_path), screen_options=closing_after(presses(1))) == 0
    spy = live.rig.spies[0]
    lives = [live_ for s, live_ in zip(spy.statuses, spy.lives, strict=True) if s.mode == "ready"]
    assert lives[1] is None and lives[0] is not None and lives[2] is not None
    refreshes = [i for i, (kind, _, _) in enumerate(spy.timeline) if kind == "refresh"]
    assert spy.live_box[refreshes[2]] != MAGENTA  # the refresh of the second preview call
    records = rows(database(tmp_path), "SELECT * FROM inspection")
    assert [(r["seq"], r["label"]) for r in records] == [(1, "pass")]


# AC7 ------------------------------------------------------------------------------------


def test_live_refusals_exit_2_with_one_line_and_no_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_display(display: DisplayConfig, title: str, clock: Any) -> ScreenLike:
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    phone = live_config(tmp_path, "phone.yml", camera__type="phone")
    real = {"make_source": frame_source}  # the real factory: nothing it builds is opened
    cases: list[tuple[Path, dict[str, Any], tuple[str, ...]]] = [
        (phone, real, ("ADR-0022", "--replay", "--screen", str(phone))),
        (live_config(tmp_path, "t.yml", trigger__type="replay"), real,
         ("trigger.type 'replay'",)),
        (live_config(tmp_path, "c.yml", camera__type="replay"), real, ("camera.type 'replay'",)),
        (live_config(tmp_path, "s.yml", inspector__type="stub"), real, ("stub",)),
        (live_config(tmp_path), {"make_screen": no_display}, ("ADR-0018",)),
    ]
    capsys.readouterr()
    for config, options, words in cases:
        live = Live()
        assert live.run(config, **options) == 2
        out = capsys.readouterr().out
        assert out.count("\n") == 1, out
        assert all(word in out for word in words), out
        assert "Traceback" not in out
        assert live.rig.screens == []  # refused before any window opened
    assert not (tmp_path / "evidence").exists()


# AC8 ------------------------------------------------------------------------------------


def test_closing_the_window_in_fault_stops_cleanly_with_exit_1(tmp_path: Path) -> None:
    live = Live()
    assert live.camera is not None
    live.camera.opens = [CameraLost("/dev/video0: cannot open")] * 20
    assert live.run(live_config(tmp_path), screen_options=closing_after(presses(1))) == 1
    run = run_row(tmp_path)
    assert run["ended_utc"] is not None
    assert json.loads(run["counters"])["final_state"] == "FAULT(camera)"
    assert live.rig.screens[0].close_calls == 1
