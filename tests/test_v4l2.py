import dataclasses
import struct
from pathlib import Path

import pytest

from qs.config import CameraConfig, load_config
from qs.errors import CameraSettingsDrift
from qs.sources import v4l2
from qs.sources.v4l2 import (
    CID_AUTO_WHITE_BALANCE,
    CID_EXPOSURE_ABSOLUTE,
    CID_EXPOSURE_AUTO,
    CID_FOCUS_ABSOLUTE,
    CID_FOCUS_AUTO,
    CID_GAIN,
    CID_POWER_LINE_FREQUENCY,
    CID_WHITE_BALANCE_TEMPERATURE,
    VIDIOC_G_CTRL,
    VIDIOC_S_CTRL,
    ControlReading,
    V4L2Controls,
    apply_controls,
    check_readings,
    read_controls,
    to_readback,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"


class FakeControls:
    """A camera's controls as a dict of raw V4L2 values; records every set in order."""

    def __init__(
        self,
        *,
        fail_get: frozenset[int] = frozenset(),
        fail_set: frozenset[int] = frozenset(),
        stuck: dict[int, int] | None = None,
        log: list[str] | None = None,
    ) -> None:
        self.values: dict[int, int] = {
            CID_POWER_LINE_FREQUENCY: 2,
            CID_EXPOSURE_AUTO: 3,
            CID_EXPOSURE_ABSOLUTE: 156,
            CID_GAIN: 64,
            CID_AUTO_WHITE_BALANCE: 1,
            CID_WHITE_BALANCE_TEMPERATURE: 4600,
            CID_FOCUS_AUTO: 1,
            CID_FOCUS_ABSOLUTE: 0,
        }
        self.values.update(stuck or {})
        self.stuck = dict(stuck or {})
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.sets: list[tuple[int, int]] = []
        self.log = log
        self.closed = 0

    def get(self, cid: int) -> int:
        if cid in self.fail_get:
            raise OSError(22, "cannot read")
        return self.values[cid]

    def set(self, cid: int, value: int) -> None:
        self.sets.append((cid, value))
        if self.log is not None:
            self.log.append(f"control {cid:#x}")
        if cid in self.fail_set:
            raise OSError(13, "refused")
        if cid not in self.stuck:
            self.values[cid] = value

    def close(self) -> None:
        self.closed += 1


def camera(**changes: object) -> CameraConfig:
    return dataclasses.replace(load_config(EXAMPLE).config.camera, **changes)  # type: ignore[arg-type]


def test_apply_controls_sets_every_control_in_order_with_its_value() -> None:
    controls = FakeControls()
    apply_controls(controls, camera(gain=3, wb_kelvin=5000, focus=40))
    assert controls.sets == [
        (CID_POWER_LINE_FREQUENCY, 1),
        (CID_EXPOSURE_AUTO, 1),
        (CID_EXPOSURE_ABSOLUTE, 100),
        (CID_GAIN, 3),
        (CID_AUTO_WHITE_BALANCE, 0),
        (CID_WHITE_BALANCE_TEMPERATURE, 5000),
        (CID_FOCUS_AUTO, 0),
        (CID_FOCUS_ABSOLUTE, 40),
    ]


def test_readings_are_named_and_in_config_units() -> None:
    readings = apply_controls(FakeControls(), camera())
    assert [r.name for r in readings] == [
        "power_line_hz",
        "exposure_auto",
        "exposure_us",
        "gain",
        "auto_white_balance",
        "wb_kelvin",
        "focus_auto",
        "focus",
    ]
    assert all(r.ok for r in readings)


def test_exposure_is_set_in_100_us_units_and_read_back_in_microseconds() -> None:
    controls = FakeControls()
    readings = apply_controls(controls, camera(exposure_us=10000))
    assert (CID_EXPOSURE_ABSOLUTE, 100) in controls.sets
    assert controls.values[CID_EXPOSURE_ABSOLUTE] == 100
    exposure = next(r for r in readings if r.name == "exposure_us")
    assert exposure == ControlReading("exposure_us", 10000, 10000)


def test_power_line_50_is_menu_1_and_reads_back_as_50() -> None:
    controls = FakeControls()
    readings = apply_controls(controls, camera(power_line_hz=50))
    assert controls.sets[0] == (CID_POWER_LINE_FREQUENCY, 1)
    assert readings[0] == ControlReading("power_line_hz", 50, 50)


def test_power_line_60_is_menu_2() -> None:
    controls = FakeControls()
    readings = apply_controls(controls, camera(power_line_hz=60, exposure_us=10000))
    assert controls.sets[0] == (CID_POWER_LINE_FREQUENCY, 2)
    assert readings[0].reported == 60


def test_power_line_55_is_refused_before_any_set() -> None:
    controls = FakeControls()
    with pytest.raises(CameraSettingsDrift, match="power_line_hz"):
        apply_controls(controls, camera(power_line_hz=55))
    assert controls.sets == []


def test_exposure_not_a_multiple_of_100_us_is_refused_before_any_set() -> None:
    controls = FakeControls()
    with pytest.raises(CameraSettingsDrift, match="exposure_us"):
        apply_controls(controls, camera(exposure_us=10050))
    assert controls.sets == []


def test_an_unreadable_control_is_none_and_the_readback_unverified() -> None:
    readings = apply_controls(FakeControls(fail_get=frozenset({CID_GAIN})), camera())
    gain = next(r for r in readings if r.name == "gain")
    assert gain.reported is None
    assert not gain.ok
    readback = to_readback(readings)
    assert readback.gain is None
    assert readback.verified is False


def test_every_control_readable_gives_a_verified_readback() -> None:
    readback = to_readback(apply_controls(FakeControls(), camera()))
    assert readback.verified is True
    assert (readback.exposure_us, readback.gain, readback.wb_kelvin, readback.focus) == (
        10000,
        0,
        4500,
        30,
    )


def test_a_refused_set_does_not_stop_the_others() -> None:
    controls = FakeControls(fail_set=frozenset({CID_EXPOSURE_AUTO}))
    readings = apply_controls(controls, camera())
    assert len(controls.sets) == 8
    assert controls.values[CID_FOCUS_ABSOLUTE] == 30
    auto = next(r for r in readings if r.name == "exposure_auto")
    assert auto.reported == 3
    assert not auto.ok


def test_read_controls_never_raises() -> None:
    every = frozenset(FakeControls().values)
    readings = read_controls(FakeControls(fail_get=every), camera())
    assert len(readings) == 8
    assert all(r.reported is None for r in readings)


def test_check_readings_names_the_first_mismatch() -> None:
    readings = (
        ControlReading("power_line_hz", 50, 50),
        ControlReading("gain", 0, 7),
        ControlReading("focus", 30, 31),
    )
    with pytest.raises(CameraSettingsDrift, match=r"^gain: set 0, camera reports 7$"):
        check_readings(readings)


def test_check_readings_does_not_raise_for_an_unreadable_control() -> None:
    check_readings((ControlReading("gain", 0, None), ControlReading("focus", 30, 30)))


def test_v4l2_controls_speak_the_kernel_ioctls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, bytes]] = []

    def fake_ioctl(fd: int, request: int, buf: bytearray) -> int:
        calls.append((fd, request, bytes(buf)))
        if request == VIDIOC_G_CTRL:
            cid, _ = struct.unpack("=Ii", buf)
            buf[:] = struct.pack("=Ii", cid, 123)
        return 0

    closed: list[int] = []
    monkeypatch.setattr(v4l2.os, "open", lambda path, flags: 42)
    monkeypatch.setattr(v4l2.os, "close", closed.append)
    monkeypatch.setattr(v4l2.fcntl, "ioctl", fake_ioctl)

    controls = V4L2Controls("/dev/video0")
    assert controls.get(CID_GAIN) == 123
    controls.set(CID_FOCUS_ABSOLUTE, -5)
    assert calls == [
        (42, VIDIOC_G_CTRL, struct.pack("=Ii", CID_GAIN, 0)),
        (42, VIDIOC_S_CTRL, struct.pack("=Ii", CID_FOCUS_ABSOLUTE, -5)),
    ]
    controls.close()
    controls.close()
    assert closed == [42]


def test_v4l2_controls_on_a_missing_device_is_camera_lost() -> None:
    from qs.errors import CameraLost

    with pytest.raises(CameraLost, match="/dev/video99"):
        V4L2Controls("/dev/video99")
