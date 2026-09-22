"""`WebcamSource`: a USB webcam over V4L2, its settings read back, never a stale frame.

A grab thread drains the camera into a short queue of (kernel timestamp, pixels). `capture()`
returns the earliest frame stamped after the trigger plus one exposure (C4), refuses a
corrupt one (FR-12), re-reads every control (FR-13) and measures the reference patch (FR-14).
The stream and the controls are opened through two factories, so tests need no camera.
"""

from __future__ import annotations

import collections
import re
import threading
import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from qs.config import StationConfig
from qs.contracts import Frame, Health, TriggerEvent
from qs.errors import CameraLost, CameraSettingsDrift, CaptureError
from qs.sources.v4l2 import (
    ControlReading,
    Controls,
    V4L2Controls,
    apply_controls,
    check_readings,
    read_controls,
    to_readback,
)

_DEVICE = re.compile(r"^/dev/video(\d+)$")
_QUEUE = 8
_FAILURES_LOST = 3
_STALE_S = 2.0
_JOIN_S = 2.0


class CaptureLike(Protocol):
    """The slice of `cv2.VideoCapture` used here."""

    def set(self, prop: int, value: float) -> bool: ...

    def get(self, prop: int) -> float: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def isOpened(self) -> bool: ...  # noqa: N802 — OpenCV's name

    def release(self) -> None: ...


def open_v4l2_capture(device: str) -> CaptureLike:
    """Open `/dev/videoN` with OpenCV's V4L2 backend, by number (it cannot open by path)."""
    match = _DEVICE.fullmatch(device)
    if match is None:
        raise CameraLost(f"{device}: not a /dev/videoN path")
    capture: CaptureLike = cv2.VideoCapture(int(match.group(1)), cv2.CAP_V4L2)
    return capture


def measure_patch(pixels: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    """Mean luminance of the reference patch's ROI [x, y, w, h], from 0 to 1."""
    x, y, w, h = roi
    crop = pixels[y : y + h, x : x + w]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(grey.mean()) / 255.0


def _fourcc(code: float) -> str:
    value = int(code)
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


class WebcamSource:
    """`FrameSource` over a V4L2 webcam: settings forced to manual and read back on every frame."""

    def __init__(
        self,
        config: StationConfig,
        *,
        open_capture: Callable[[str], CaptureLike] = open_v4l2_capture,
        open_controls: Callable[[str], Controls] = V4L2Controls,
    ) -> None:
        self._config = config
        self._camera = config.camera
        self._open_capture = open_capture
        self._open_controls = open_controls
        self._cap: CaptureLike | None = None
        self._controls: Controls | None = None
        self._readings: tuple[ControlReading, ...] = ()
        self._frames: collections.deque[tuple[float, np.ndarray]] = collections.deque(
            maxlen=_QUEUE
        )
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._lost = False
        self._thread: threading.Thread | None = None

    @property
    def readings(self) -> tuple[ControlReading, ...]:
        """The control readings taken by `open()`; empty before it."""
        return self._readings

    def open(self) -> None:
        """Open the stream, set its format, force every control to manual, start grabbing."""
        camera = self._camera
        try:
            cap = self._open_capture(camera.device)
            self._cap = cap
            if not cap.isOpened():
                raise CameraLost(f"{camera.device}: cannot open")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*camera.format))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
            fmt = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (fmt, width, height) != (camera.format, camera.width, camera.height):
                raise CameraSettingsDrift(
                    f"format: set {camera.format} {camera.width}x{camera.height}, "
                    f"camera reports {fmt} {width}x{height}"
                )
            self._controls = self._open_controls(camera.device)
            self._readings = apply_controls(self._controls, camera)
            check_readings(self._readings)
            self._start_grabbing(cap)
        except BaseException:
            self.close()
            raise

    def _start_grabbing(self, cap: CaptureLike) -> None:
        self._stop.clear()
        self._lost = False
        self._frames.clear()
        self._thread = threading.Thread(
            target=self._grab, args=(cap,), name="qs-webcam-grab", daemon=True
        )
        self._thread.start()
        deadline = time.perf_counter() + self._camera.capture_timeout_ms / 1000
        with self._cond:
            while not self._frames and not self._lost:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            if not self._frames:
                raise CaptureError(
                    f"no frame within {self._camera.capture_timeout_ms} ms of opening"
                )

    def _grab(self, cap: CaptureLike) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, img = cap.read()
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ok and img is not None:
                img.setflags(write=False)
                with self._cond:
                    self._frames.append((ts, img))
                    self._cond.notify_all()
                failures = 0
                continue
            failures += 1
            if failures >= _FAILURES_LOST:
                with self._cond:
                    self._lost = True
                    self._cond.notify_all()
                return

    def capture(self, event: TriggerEvent) -> Frame:
        """The earliest frame stamped after the trigger plus one exposure, checked and read back."""
        camera = self._camera
        if self._lost:
            raise CameraLost(f"{camera.device}: camera stopped delivering frames")
        controls = self._controls
        if controls is None:
            raise CaptureError("capture() called before open()")
        not_before = event.t0 + camera.exposure_us / 1e6
        deadline = event.t0 + camera.capture_timeout_ms / 1000
        with self._cond:
            while True:
                fresh = [(ts, img) for ts, img in self._frames if ts > not_before]
                if fresh:
                    ts, img = min(fresh, key=lambda item: item[0])
                    break
                if self._lost:
                    raise CameraLost(f"{camera.device}: camera stopped delivering frames")
                if self._frames and self._frames[-1][0] <= 0:
                    raise CaptureError("camera gives no kernel timestamps")
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise CaptureError(
                        f"no frame newer than the trigger within {camera.capture_timeout_ms} ms"
                    )
                self._cond.wait(remaining)
        if ts <= 0:
            raise CaptureError("camera gives no kernel timestamps")
        if ts > time.perf_counter() + 1.0:
            raise CaptureError("kernel timestamp is not on the monotonic clock")
        if img.shape != (camera.height, camera.width, 3) or img.dtype != np.uint8:
            raise CaptureError(f"corrupt frame: {img.shape} {img.dtype}")
        if int(img.max()) - int(img.min()) <= 1:
            raise CaptureError("corrupt frame: flat")
        readings = read_controls(controls, camera)
        check_readings(readings)
        return Frame(
            event=event,
            pixels=img,
            grabbed_at=ts,
            camera=to_readback(readings),
            reference_patch=measure_patch(img, self._config.lighting.reference_patch.roi),
            capture_ms=(time.perf_counter() - event.t0) * 1000,
        )

    def preview(self) -> np.ndarray | None:
        """The newest frame, for the live view only — never inspected (C9)."""
        with self._cond:
            return self._frames[-1][1] if self._frames else None

    def health(self) -> Health:
        """Streaming, or which of lost / thread stopped / frames stale is wrong."""
        problems: list[str] = []
        if self._lost:
            problems.append("camera lost")
        if self._thread is None or not self._thread.is_alive():
            problems.append("grab thread not running")
        with self._cond:
            newest = self._frames[-1][0] if self._frames else None
        if newest is None or time.perf_counter() - newest >= _STALE_S:
            problems.append(f"no frame in the last {_STALE_S:g} s")
        if problems:
            return Health(ok=False, detail="; ".join(problems))
        return Health(ok=True, detail="streaming")

    def close(self) -> None:
        """Stop the grab thread, release the camera and the controls; safe to call twice."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_JOIN_S)
            self._thread = None
        if self._cap is not None:
            cap, self._cap = self._cap, None
            cap.release()
        if self._controls is not None:
            controls, self._controls = self._controls, None
            controls.close()
