import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from qs.cmd.camera_test import LockTestResult, Sample, lock_test_verdict, main
from qs.config import StationConfig
from qs.contracts import CameraReadback, Frame, Health, TriggerEvent
from qs.screen import Screen
from qs.sources.v4l2 import ControlReading

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"
EXPOSURE = 10000
GOOD_READINGS = (
    ControlReading("power_line_hz", 50, 50),
    ControlReading("exposure_auto", 1, 1),
    ControlReading("exposure_us", 10000, 10000),
    ControlReading("gain", 0, 0),
    ControlReading("auto_white_balance", 0, 0),
    ControlReading("wb_kelvin", 4500, 4500),
    ControlReading("focus_auto", 0, 0),
    ControlReading("focus", 30, 30),
)
VERIFIED = CameraReadback(10000, 0, 4500, 30, verified=True)


def sample(seq: int, patch: float = 0.62, **changes: object) -> Sample:
    t0 = 100.0 + seq * 6
    good = Sample(seq, t0, t0 + 0.03, 35.0, patch, VERIFIED, None)
    return dataclasses.replace(good, **changes)  # type: ignore[arg-type]


def verdict(samples: list[Sample], readings: tuple = GOOD_READINGS) -> LockTestResult:
    return lock_test_verdict(readings, tuple(samples), 3, EXPOSURE)


def failed_lines(result: LockTestResult) -> list[str]:
    return [line for line in result.lines if line.startswith("FAIL")]


def test_everything_good_passes() -> None:
    result = verdict([sample(1), sample(2), sample(3, patch=0.623)])
    assert result.passed is True
    assert failed_lines(result) == []
    assert sum(line.startswith("PASS") for line in result.lines) == 5
    assert result.lines[-1].startswith("CHECK BY EYE banding")


def test_a_patch_varying_two_percent_fails_and_says_so() -> None:
    result = verdict([sample(1, patch=0.60), sample(2, patch=0.61), sample(3, patch=0.612)])
    assert result.passed is False
    [line] = failed_lines(result)
    assert line.startswith("FAIL reference patch")
    assert "varies 1.98%" in line


def test_one_stale_sample_fails() -> None:
    stale = sample(2)
    stale = dataclasses.replace(stale, grabbed_at=stale.t0 + EXPOSURE / 1e6)
    result = verdict([sample(1), stale, sample(3)])
    assert result.passed is False
    [line] = failed_lines(result)
    assert line.startswith("FAIL freshness") and "#2" in line


def test_timestamps_that_do_not_increase_fail() -> None:
    late = sample(3, grabbed_at=50.0, t0=40.0)
    result = verdict([sample(1), sample(2), late])
    assert result.passed is False
    assert "not strictly increasing" in failed_lines(result)[0]


def test_a_mismatched_reading_fails() -> None:
    readings = (*GOOD_READINGS[:3], ControlReading("gain", 0, 12), *GOOD_READINGS[4:])
    result = verdict([sample(1), sample(2), sample(3)], readings)
    assert result.passed is False
    [line] = failed_lines(result)
    assert line.startswith("FAIL controls") and "gain" in line


def test_an_unreadable_reading_fails() -> None:
    readings = (*GOOD_READINGS[:3], ControlReading("gain", 0, None), *GOOD_READINGS[4:])
    result = verdict([sample(1), sample(2), sample(3)], readings)
    assert result.passed is False
    assert "unreadable" in failed_lines(result)[0]


def test_fewer_samples_than_wanted_fails() -> None:
    result = verdict([sample(1), sample(2)])
    assert result.passed is False
    [line] = failed_lines(result)
    assert line.startswith("FAIL frames: 2 of 3")


def test_a_failed_capture_fails() -> None:
    broken = Sample(2, 112.0, None, None, None, None, "no frame newer than the trigger")
    result = verdict([sample(1), broken, sample(3)])
    assert result.passed is False
    assert any(line.startswith("FAIL frames") for line in result.lines)


def test_an_unverified_sample_fails() -> None:
    unverified = CameraReadback(10000, None, 4500, 30, verified=False)
    result = verdict([sample(1), sample(2, readback=unverified), sample(3)])
    assert result.passed is False
    [line] = failed_lines(result)
    assert line.startswith("FAIL read-back: 2 of 3")


class FakeSource:
    def __init__(self, config: StationConfig) -> None:
        self.config = config
        self.readings = GOOD_READINGS
        self.closed = 0
        rng = np.random.default_rng(0)
        self.pixels = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)

    def open(self) -> None:
        pass

    def capture(self, event: TriggerEvent) -> Frame:
        return Frame(event, self.pixels, event.t0 + 0.03, VERIFIED, 0.62, 30.0)

    def preview(self) -> np.ndarray | None:
        return self.pixels

    def health(self) -> Health:
        return Health(True, "streaming")

    def close(self) -> None:
        self.closed += 1


def args(tmp_path: Path, config: Path = EXAMPLE, frames: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config),
        frames=frames,
        interval_s=0.0,
        out=str(tmp_path / "frames"),
        report_dir=str(tmp_path / "reports"),
        no_screen=True,
    )


def test_main_with_a_good_source_saves_frames_and_a_report(tmp_path: Path) -> None:
    made: list[FakeSource] = []

    def make_source(config: StationConfig) -> FakeSource:
        made.append(FakeSource(config))
        return made[-1]

    assert main(args(tmp_path), make_source=make_source) == 0
    frames = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert frames == ["lock_01.png", "lock_02.png", "lock_03.png"]
    [report] = list((tmp_path / "reports").iterdir())
    assert report.name == f"{datetime.now().date().isoformat()}-lock-test.md"
    text = report.read_text(encoding="utf-8")
    assert "/dev/video0" in text and "PASS reference patch" in text
    assert f"CHECK BY EYE banding — look at the saved frames in {tmp_path / 'frames'}" in text
    assert made[0].closed == 1


def test_main_with_no_camera_prints_one_camera_line_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "station.yml"
    config.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace("/dev/video0", "/dev/video99"),
        encoding="utf-8",
    )
    assert main(args(tmp_path, config)) == 2
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("camera:") and "/dev/video99" in lines[0]


def test_main_with_a_bad_config_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(tmp_path, tmp_path / "missing.yml")) == 2
    assert capsys.readouterr().out.startswith("config:")


def test_main_without_a_display_says_so_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import tkinter as tk

    made: list[FakeSource] = []

    def make_source(config: StationConfig) -> FakeSource:
        made.append(FakeSource(config))
        return made[-1]

    def no_display(display: object, title: str) -> Screen:
        raise tk.TclError("no display name and no $DISPLAY environment variable")

    run = args(tmp_path)
    run.no_screen = False
    assert main(run, make_source=make_source, make_screen=no_display) == 2
    assert capsys.readouterr().out.splitlines()[-1].startswith("screen: no display")
    assert made[0].closed == 1


def test_main_stops_at_a_lost_camera_and_fails(tmp_path: Path) -> None:
    from qs.errors import CameraLost

    class LosingSource(FakeSource):
        calls = 0

        def capture(self, event: TriggerEvent) -> Frame:
            LosingSource.calls += 1
            if LosingSource.calls >= 2:
                raise CameraLost("/dev/video0: camera stopped delivering frames")
            return super().capture(event)

    assert main(args(tmp_path, frames=5), make_source=LosingSource) == 1
    assert LosingSource.calls == 2
    [report] = list((tmp_path / "reports").iterdir())
    assert "FAIL frames: 2 of 5 captured, 1 failed" in report.read_text(encoding="utf-8")


def test_main_with_the_grey_card_outside_the_frame_says_so_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "station.yml"
    config.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace("[1780, 40, 100, 100]", "[1880, 40, 100, 100]"),
        encoding="utf-8",
    )
    opened: list[FakeSource] = []
    assert main(args(tmp_path, config), make_source=lambda c: opened.append(FakeSource(c))) == 2
    [line] = capsys.readouterr().out.strip().splitlines()
    assert line == (
        "config: lighting.reference_patch.roi: [1880, 40, 100, 100] is outside the "
        "1920x1080 frame"
    )
    assert opened == []  # refused before the camera is touched


def test_main_records_a_lighting_fault_as_a_failed_frame_not_a_traceback(tmp_path: Path) -> None:
    from qs.errors import LightingFault

    class OffFrameSource(FakeSource):
        def capture(self, event: TriggerEvent) -> Frame:
            raise LightingFault("reference patch: [1780, 40, 100, 100] is outside the 64x48 frame")

    assert main(args(tmp_path), make_source=OffFrameSource) == 1
    [report] = list((tmp_path / "reports").iterdir())
    text = report.read_text(encoding="utf-8")
    assert "FAIL frames: 3 of 3 captured, 3 failed (first: #1 reference patch: [1780" in text


NO_FOCUS = (*GOOD_READINGS[:7], ControlReading("focus", 30, None))


def test_a_missing_focus_control_fails_and_says_the_phone_takes_over() -> None:
    unverified = CameraReadback(10000, 0, 4500, None, verified=False)
    samples = [sample(n, readback=unverified) for n in (1, 2, 3)]
    result = verdict(samples, NO_FOCUS)
    assert result.passed is False
    controls, readback = failed_lines(result)
    assert controls == (
        "FAIL controls: focus unreadable (set 30; the camera does not report this control, "
        "so it cannot be locked) — a missing control fails ADR-0010: the phone takes over"
    )
    assert readback == "FAIL read-back: 0 of 3 frames verified (unreadable: focus)"


def test_main_names_a_missing_control_before_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class NoFocusSource(FakeSource):
        def __init__(self, config: StationConfig) -> None:
            super().__init__(config)
            self.readings = NO_FOCUS

        def capture(self, event: TriggerEvent) -> Frame:
            unverified = CameraReadback(10000, 0, 4500, None, verified=False)
            return Frame(event, self.pixels, event.t0 + 0.03, unverified, 0.62, 30.0)

    assert main(args(tmp_path), make_source=NoFocusSource) == 1
    out = capsys.readouterr().out.splitlines()
    warning = out.index(
        "WARNING focus: the camera does not report this control, so it cannot be locked — "
        "this run will FAIL (ADR-0010: the phone takes over). It goes on to measure the "
        "grey card; Q quits."
    )
    assert warning < next(i for i, line in enumerate(out) if line.startswith("FAIL"))


def test_main_with_a_refused_value_at_open_fails_and_points_at_the_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from qs.errors import CameraSettingsDrift

    class RefusingSource(FakeSource):
        def open(self) -> None:
            raise CameraSettingsDrift("gain: set 400, camera reports 255")

    assert main(args(tmp_path), make_source=RefusingSource) == 1
    out = capsys.readouterr().out.splitlines()
    assert "FAIL gain: set 400, camera reports 255" in out
    assert (
        "Before calling it the camera's fail: is the value within the control's min/max "
        "(v4l2-ctl -d /dev/video0 --list-ctrls) and the mode one the camera lists "
        "(--list-formats-ext)?"
    ) in out
