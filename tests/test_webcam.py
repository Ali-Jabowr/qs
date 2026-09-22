import dataclasses
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.config import StationConfig, load_config
from qs.contracts import TriggerEvent
from qs.errors import CameraLost, CameraSettingsDrift, CaptureError
from qs.sources.v4l2 import CID_GAIN
from qs.sources.webcam import WebcamSource, measure_patch, open_v4l2_capture
from test_v4l2 import FakeControls

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"
W, H = 64, 48
FAIL = None  # a script entry: read() fails


def small_config() -> StationConfig:
    config = load_config(EXAMPLE).config
    lighting = dataclasses.replace(
        config.lighting,
        reference_patch=dataclasses.replace(config.lighting.reference_patch, roi=(0, 0, 8, 8)),
    )
    return dataclasses.replace(
        config, camera=dataclasses.replace(config.camera, width=W, height=H), lighting=lighting
    )


def image(level: int = 120, shape: tuple[int, ...] = (H, W, 3)) -> np.ndarray:
    img = np.full(shape, level, dtype=np.uint8)
    img[-1, -1] = 0  # never flat
    return img


class FakeCapture:
    """Serves a script of (timestamp, image) entries; FAIL entries fail read().

    When the script runs out it either streams fresh frames stamped now (`stream=True`) or
    waits for more entries or `finish()`, after which every read() fails.
    """

    def __init__(
        self,
        script: list[tuple[float, np.ndarray] | None],
        *,
        opened: bool = True,
        stream: bool = False,
        report: dict[int, float] | None = None,
        log: list[str] | None = None,
    ) -> None:
        self.script = list(script)
        self.opened = opened
        self.stream = stream
        self.props: dict[int, float] = {}
        self.report = report or {}
        self.log = log
        self.ts = 0.0
        self.finished = threading.Event()
        self.released = 0

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        if self.log is not None:
            self.log.append(f"cap {prop}")
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.ts * 1000.0
        return self.report.get(prop, self.props.get(prop, 0.0))

    def read(self) -> tuple[bool, np.ndarray | None]:
        while not self.script:
            if self.finished.is_set():
                return False, None
            if self.stream:
                time.sleep(0.005)
                self.ts = time.perf_counter()
                return True, image()
            self.finished.wait(0.002)
        entry = self.script.pop(0)
        if entry is FAIL:
            return False, None
        self.ts, img = entry
        return True, img.copy()

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def release(self) -> None:
        self.released += 1

    def finish(self) -> None:
        self.finished.set()


@pytest.fixture
def sources() -> Iterator[list[tuple[WebcamSource, FakeCapture]]]:
    made: list[tuple[WebcamSource, FakeCapture]] = []
    yield made
    for source, cap in made:
        cap.finish()
        source.close()


def make(
    made: list[tuple[WebcamSource, FakeCapture]],
    cap: FakeCapture,
    controls: FakeControls | None = None,
    config: StationConfig | None = None,
) -> tuple[WebcamSource, FakeControls]:
    controls = controls or FakeControls()
    source = WebcamSource(
        config or small_config(),
        open_capture=lambda device: cap,
        open_controls=lambda device: controls,
    )
    made.append((source, cap))
    return source, controls


def event(t0: float, seq: int = 1) -> TriggerEvent:
    return TriggerEvent(seq, t0, datetime.now(UTC), "test")


def test_open_sets_the_format_then_applies_the_controls(sources: list) -> None:
    log: list[str] = []
    cap = FakeCapture([(time.perf_counter(), image())], log=log)
    source, controls = make(sources, cap, FakeControls(log=log))
    assert source.readings == ()
    source.open()
    assert log[:4] == [
        f"cap {cv2.CAP_PROP_READ_TIMEOUT_MSEC}",
        f"cap {cv2.CAP_PROP_FOURCC}",
        f"cap {cv2.CAP_PROP_FRAME_WIDTH}",
        f"cap {cv2.CAP_PROP_FRAME_HEIGHT}",
    ]
    assert all(entry.startswith("control") for entry in log[4:]) and len(log) == 12
    assert cap.props[cv2.CAP_PROP_READ_TIMEOUT_MSEC] == 1000
    assert cap.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter.fourcc(*"YUYV")
    assert (cap.props[cv2.CAP_PROP_FRAME_WIDTH], cap.props[cv2.CAP_PROP_FRAME_HEIGHT]) == (W, H)
    assert len(source.readings) == 8
    assert all(r.ok for r in source.readings)


def test_a_camera_that_cannot_open_is_camera_lost(sources: list) -> None:
    source, _ = make(sources, FakeCapture([], opened=False))
    with pytest.raises(CameraLost, match="cannot open"):
        source.open()


def test_a_format_read_back_mismatch_is_drift(sources: list) -> None:
    cap = FakeCapture([(1.0, image())], report={cv2.CAP_PROP_FRAME_WIDTH: 1280.0})
    source, _ = make(sources, cap)
    with pytest.raises(CameraSettingsDrift, match=r"^format: set YUYV 64x48, camera reports"):
        source.open()
    assert cap.released == 1


def test_a_control_mismatch_at_open_is_drift_and_readings_are_kept(sources: list) -> None:
    controls = FakeControls(stuck={CID_GAIN: 16})
    source, _ = make(sources, FakeCapture([(1.0, image())]), controls)
    with pytest.raises(CameraSettingsDrift, match=r"^gain: set 0, camera reports 16$"):
        source.open()
    assert len(source.readings) == 8
    assert controls.closed == 1


def test_no_first_frame_within_the_timeout_is_a_capture_error(sources: list) -> None:
    source, _ = make(sources, FakeCapture([FAIL, FAIL, FAIL]))
    with pytest.raises(CaptureError, match="no frame within 3000 ms of opening"):
        source.open()


def test_capture_returns_the_earliest_frame_after_trigger_plus_exposure(sources: list) -> None:
    base = time.perf_counter()
    script = [
        (base - 0.05, image(10)),
        (base + 0.005, image(20)),  # inside the exposure: still stale
        (base + 0.010, image(30)),  # exactly t0 + exposure: stale
        (base + 0.020, image(40)),
        (base + 0.030, image(50)),
    ]
    source, _ = make(sources, FakeCapture(script))
    source.open()
    frame = source.capture(event(base))
    assert frame.grabbed_at == base + 0.020
    assert int(frame.pixels[0, 0, 0]) == 40
    assert frame.camera.verified is True
    assert frame.capture_ms >= 0


def test_no_fresh_frame_within_the_timeout_is_a_capture_error(sources: list) -> None:
    base = time.perf_counter()
    source, _ = make(sources, FakeCapture([(base - 0.1, image()), (base, image())]))
    source.open()
    with pytest.raises(CaptureError, match="newer than the trigger within 500 ms"):
        source.capture(event(base))


def test_a_flat_frame_is_corrupt(sources: list) -> None:
    base = time.perf_counter()
    flat = np.full((H, W, 3), 128, dtype=np.uint8)
    flat[0, 0] = 129  # within one grey level: still flat
    source, _ = make(sources, FakeCapture([(base + 0.02, flat)]))
    source.open()
    with pytest.raises(CaptureError, match="corrupt frame: flat"):
        source.capture(event(base))


def test_a_wrong_shape_frame_is_corrupt(sources: list) -> None:
    base = time.perf_counter()
    source, _ = make(sources, FakeCapture([(base + 0.02, image(shape=(H, W // 2, 3)))]))
    source.open()
    with pytest.raises(CaptureError, match=r"corrupt frame: \(48, 32, 3\) uint8"):
        source.capture(event(base))


def test_a_zero_timestamp_is_refused(sources: list) -> None:
    source, _ = make(sources, FakeCapture([(0.0, image())]))
    source.open()
    with pytest.raises(CaptureError, match="no kernel timestamps"):
        source.capture(event(time.perf_counter()))


def test_a_timestamp_in_the_future_is_refused(sources: list) -> None:
    base = time.perf_counter()
    source, _ = make(sources, FakeCapture([(base + 5.0, image())]))
    source.open()
    with pytest.raises(CaptureError, match="not on the monotonic clock"):
        source.capture(event(base))


def test_a_control_changed_after_open_is_drift_on_the_next_capture(sources: list) -> None:
    base = time.perf_counter()
    source, controls = make(sources, FakeCapture([(base + 0.02, image())]))
    source.open()
    controls.values[CID_GAIN] = 5
    with pytest.raises(CameraSettingsDrift, match="gain: set 0, camera reports 5"):
        source.capture(event(base))


def test_three_read_failures_lose_the_camera(sources: list) -> None:
    source, _ = make(sources, FakeCapture([(time.perf_counter(), image()), FAIL, FAIL, FAIL]))
    source.open()
    deadline = time.perf_counter() + 1.0
    while source.health().ok is not False and time.perf_counter() < deadline:
        time.sleep(0.005)
    health = source.health()
    assert health.ok is False
    assert "camera lost" in health.detail
    with pytest.raises(CameraLost, match="stopped delivering frames"):
        source.capture(event(time.perf_counter()))


def test_two_read_failures_do_not_lose_the_camera(sources: list) -> None:
    base = time.perf_counter()
    script = [(base, image()), FAIL, FAIL, (base + 0.02, image())]
    source, _ = make(sources, FakeCapture(script))
    source.open()
    assert source.capture(event(base)).grabbed_at == base + 0.02


def test_the_reference_patch_is_the_mean_luminance_of_its_roi(sources: list) -> None:
    base = time.perf_counter()
    img = image(30)
    img[0:8, 0:8] = 158
    source, _ = make(sources, FakeCapture([(base + 0.02, img)]))
    source.open()
    frame = source.capture(event(base))
    assert frame.reference_patch == pytest.approx(158 / 255, abs=1e-6)


def test_measure_patch_reads_only_the_roi() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:10, 2:6] = 200
    assert measure_patch(img, (2, 5, 4, 5)) == pytest.approx(200 / 255, abs=1e-6)


def test_the_returned_pixels_are_read_only(sources: list) -> None:
    base = time.perf_counter()
    source, _ = make(sources, FakeCapture([(base + 0.02, image())]))
    source.open()
    frame = source.capture(event(base))
    assert frame.pixels.flags.writeable is False
    with pytest.raises(ValueError):
        frame.pixels[0, 0, 0] = 1


def test_preview_health_and_close(sources: list) -> None:
    cap = FakeCapture([], stream=True)
    source, controls = make(sources, cap)
    assert source.preview() is None
    source.open()
    preview = source.preview()
    assert isinstance(preview, np.ndarray) and preview.shape == (H, W, 3)
    assert source.health().ok is True
    assert source.health().detail == "streaming"
    thread = source._thread
    assert thread is not None and thread.is_alive()
    source.close()
    source.close()
    assert not thread.is_alive()
    assert cap.released == 1
    assert controls.closed == 1
    assert source.health().ok is False


def test_open_v4l2_capture_refuses_a_path_that_is_not_a_video_device() -> None:
    with pytest.raises(CameraLost, match="/dev/sda: not a /dev/videoN path"):
        open_v4l2_capture("/dev/sda")


class CrashingCapture(FakeCapture):
    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.script:
            raise cv2.error("driver fell over")
        return super().read()


def test_a_crashed_grab_thread_loses_the_camera(sources: list) -> None:
    source, _ = make(sources, CrashingCapture([(time.perf_counter(), image())]))
    source.open()
    with pytest.raises(CameraLost, match="stopped delivering frames"):
        source.capture(event(time.perf_counter() + 0.1))


def test_close_never_releases_a_capture_that_is_still_being_read(
    sources: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qs.sources import webcam

    monkeypatch.setattr(webcam, "_JOIN_S", 0.05)
    cap = FakeCapture([(time.perf_counter(), image())])  # then blocks in read()
    source, controls = make(sources, cap)
    source.open()
    source.close()
    assert cap.released == 0
    assert controls.closed == 1


def test_a_reference_patch_outside_the_frame_is_a_lighting_fault() -> None:
    from qs.errors import LightingFault

    pixels = np.full((48, 64, 3), 128, dtype=np.uint8)
    with pytest.raises(LightingFault, match=r"\[60, 40, 10, 10\] is outside the 64x48 frame"):
        measure_patch(pixels, (60, 40, 10, 10))
