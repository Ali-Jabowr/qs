"""Camera controls over V4L2 ioctls: set every control to manual, then read each one back.

Controls go through a file descriptor of their own, never through OpenCV (ADR-0010): the
grab thread owns the `VideoCapture`, and V4L2 controls belong to the device, not to the
descriptor. A value that reads back different is drift; one that cannot be read is
unverified, never assumed (FR-13).
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from qs.config import CameraConfig
from qs.contracts import CameraReadback
from qs.errors import CameraLost, CameraSettingsDrift

CID_EXPOSURE_AUTO = 0x009A0901
CID_EXPOSURE_ABSOLUTE = 0x009A0902
CID_GAIN = 0x00980913
CID_AUTO_WHITE_BALANCE = 0x0098090C
CID_WHITE_BALANCE_TEMPERATURE = 0x0098091A
CID_FOCUS_AUTO = 0x009A090C
CID_FOCUS_ABSOLUTE = 0x009A090A
CID_POWER_LINE_FREQUENCY = 0x00980918

VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C

_CONTROL = "=Ii"
_POWER_LINE_MENU = {50: 1, 60: 2}
_POWER_LINE_HZ = {menu: hz for hz, menu in _POWER_LINE_MENU.items()}


class Controls(Protocol):
    """Raw access to a camera's controls, in V4L2 units."""

    def get(self, cid: int) -> int: ...

    def set(self, cid: int, value: int) -> None: ...

    def close(self) -> None: ...


class V4L2Controls:
    """`Controls` over `VIDIOC_G_CTRL`/`VIDIOC_S_CTRL` on the device node."""

    def __init__(self, device: str) -> None:
        try:
            self._fd: int | None = os.open(device, os.O_RDWR)
        except OSError as exc:
            raise CameraLost(f"{device}: {exc}") from exc

    def _require_fd(self) -> int:
        if self._fd is None:
            raise OSError("controls are closed")
        return self._fd

    def get(self, cid: int) -> int:
        """Read one control; raises OSError if it cannot be read."""
        buf = bytearray(struct.pack(_CONTROL, cid, 0))
        fcntl.ioctl(self._require_fd(), VIDIOC_G_CTRL, buf)
        _, value = struct.unpack(_CONTROL, buf)
        return int(value)

    def set(self, cid: int, value: int) -> None:
        """Write one control; raises OSError if the driver refuses it."""
        buf = bytearray(struct.pack(_CONTROL, cid, value))
        fcntl.ioctl(self._require_fd(), VIDIOC_S_CTRL, buf)

    def close(self) -> None:
        """Close the descriptor; safe to call twice."""
        if self._fd is not None:
            fd, self._fd = self._fd, None
            os.close(fd)


@dataclass(frozen=True, slots=True)
class ControlReading:
    """One control: what the config wants and what the camera reports, in config units."""

    name: str
    wanted: int
    reported: int | None

    @property
    def ok(self) -> bool:
        """True if the camera reports exactly the wanted value."""
        return self.reported == self.wanted


def _raw(value: int) -> int:
    return value


def _hundreds(value: int) -> int:
    return value * 100


def _hz(value: int) -> int:
    return _POWER_LINE_HZ.get(value, value)


def _table(camera: CameraConfig) -> tuple[tuple[int, int, str, int, Callable[[int], int]], ...]:
    """(cid, value to set, reading name, wanted in config units, raw → config units)."""
    return (
        (
            CID_POWER_LINE_FREQUENCY,
            _POWER_LINE_MENU.get(camera.power_line_hz, 0),
            "power_line_hz",
            camera.power_line_hz,
            _hz,
        ),
        (CID_EXPOSURE_AUTO, 1, "exposure_auto", 1, _raw),
        (CID_EXPOSURE_ABSOLUTE, camera.exposure_us // 100, "exposure_us", camera.exposure_us,
         _hundreds),
        (CID_GAIN, camera.gain, "gain", camera.gain, _raw),
        (CID_AUTO_WHITE_BALANCE, 0, "auto_white_balance", 0, _raw),
        (CID_WHITE_BALANCE_TEMPERATURE, camera.wb_kelvin, "wb_kelvin", camera.wb_kelvin, _raw),
        (CID_FOCUS_AUTO, 0, "focus_auto", 0, _raw),
        (CID_FOCUS_ABSOLUTE, camera.focus, "focus", camera.focus, _raw),
    )


def apply_controls(controls: Controls, camera: CameraConfig) -> tuple[ControlReading, ...]:
    """Set every control, auto off before its value (ADR-0010), then read them all back."""
    if camera.power_line_hz not in _POWER_LINE_MENU:
        raise CameraSettingsDrift(
            f"power_line_hz: {camera.power_line_hz} is not 50 or 60"
        )
    if camera.exposure_us % 100 != 0:
        raise CameraSettingsDrift(
            f"exposure_us: {camera.exposure_us} is not a multiple of 100 µs (the V4L2 unit)"
        )
    for cid, value, _, _, _ in _table(camera):
        with contextlib.suppress(OSError):  # the read-back shows it
            controls.set(cid, value)
    return read_controls(controls, camera)


def read_controls(controls: Controls, camera: CameraConfig) -> tuple[ControlReading, ...]:
    """Read every control back; an unreadable one is `None`. Never raises."""
    readings: list[ControlReading] = []
    for cid, _, name, wanted, convert in _table(camera):
        try:
            reported: int | None = convert(controls.get(cid))
        except OSError:
            reported = None
        readings.append(ControlReading(name, wanted, reported))
    return tuple(readings)


def check_readings(readings: tuple[ControlReading, ...]) -> None:
    """Raise `CameraSettingsDrift` for the first readable control that differs from the config."""
    for reading in readings:
        if reading.reported is not None and not reading.ok:
            raise CameraSettingsDrift(
                f"{reading.name}: set {reading.wanted}, camera reports {reading.reported}"
            )


def to_readback(readings: tuple[ControlReading, ...]) -> CameraReadback:
    """The frame's `CameraReadback`; verified only if every control could be read."""
    by_name = {reading.name: reading.reported for reading in readings}
    return CameraReadback(
        exposure_us=by_name.get("exposure_us"),
        gain=by_name.get("gain"),
        wb_kelvin=by_name.get("wb_kelvin"),
        focus=by_name.get("focus"),
        verified=len(readings) == 8 and all(r.reported is not None for r in readings),
    )
