import argparse
import json
from pathlib import Path

import pytest

from qs.cmd import record
from qs.sources import v4l2

CAMERA_YML = """\
device: /dev/video0
width: 1280
height: 720
format: MJPG
fps: 30
power_line_hz: 50
exposure_us: 2000
gain: 64
wb_kelvin: 4600
focus: 30
"""

# What a camera that obeyed every control reports, by control id.
OBEDIENT = {
    v4l2.CID_POWER_LINE_FREQUENCY: 1,
    v4l2.CID_EXPOSURE_AUTO: 1,
    v4l2.CID_EXPOSURE_ABSOLUTE: 20,
    v4l2.CID_GAIN: 64,
    v4l2.CID_AUTO_WHITE_BALANCE: 0,
    v4l2.CID_WHITE_BALANCE_TEMPERATURE: 4600,
    v4l2.CID_FOCUS_AUTO: 0,
    v4l2.CID_FOCUS_ABSOLUTE: 30,
}

FFMPEG_60 = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-input_format", "mjpeg",
    "-video_size", "1280x720", "-framerate", "30", "-i", "/dev/video0", "-an", "-c:v", "copy",
    "-t", "60", "d/T-1.mkv",
]


class FakeControls:
    def __init__(self, values: dict[int, int]) -> None:
        self.values = dict(values)

    def get(self, cid: int) -> int:
        if cid not in self.values:
            raise OSError("not reported")
        return self.values[cid]

    def set(self, cid: int, value: int) -> None:
        pass  # the fake camera keeps whatever it was given

    def close(self) -> None:
        pass


class FakeProcess:
    def __init__(self, controls: FakeControls, out: Path, *, after: dict[int, int] | None = None,
                 code: int = 0, interrupt_first: bool = False) -> None:
        self.controls = controls
        self.out = out
        self.after = after or {}
        self.code = code
        self.interrupt_first = interrupt_first
        self.interrupted = 0
        self.waits = 0

    def interrupt(self) -> None:
        self.interrupted += 1

    def wait(self) -> int:
        self.waits += 1
        if self.interrupt_first and self.interrupted == 0:
            raise KeyboardInterrupt
        self.controls.values.update(self.after)
        self.out.write_bytes(b"video")
        return self.code


class Rig:
    """The parameters of `main`, faked, and what they saw."""

    def __init__(self, tmp_path: Path, **process: object) -> None:
        self.out = tmp_path / "d"
        self.camera = tmp_path / "record.yml"
        self.camera.write_text(CAMERA_YML)
        self.controls = FakeControls(OBEDIENT)
        self.started: list[list[str]] = []
        self.process_kwargs = process
        self.process: FakeProcess | None = None
        self.times = iter(["2026-11-14T10:00:00Z", "2026-11-14T10:01:00Z"])

    def start(self, argv: list[str]) -> FakeProcess:
        self.started.append(argv)
        self.process = FakeProcess(
            self.controls, Path(argv[-1]), **self.process_kwargs  # type: ignore[arg-type]
        )
        return self.process

    def run(self, seconds: int | None = 60, block: str = "train") -> int:
        args = argparse.Namespace(camera=str(self.camera), out=str(self.out), clip="T-1",
                                  block=block, seconds=seconds)
        return record.main(
            args,
            start_process=self.start,
            make_controls=lambda device: self.controls,
            clock=lambda: next(self.times),
            file_hash=lambda path: "fakehash",
        )

    def sidecar(self) -> dict:
        return json.loads((self.out / "T-1.json").read_text())


def test_ffmpeg_arguments_with_seconds(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    expected = [a.replace("d/T-1.mkv", str(rig.out / "T-1.mkv")) for a in FFMPEG_60]
    assert rig.started == [expected]


def test_ctrl_c_without_seconds_interrupts_and_waits(tmp_path: Path) -> None:
    rig = Rig(tmp_path, interrupt_first=True, code=255)
    code = rig.run(seconds=None)
    assert "-t" not in rig.started[0]
    assert rig.process is not None
    assert (rig.process.interrupted, rig.process.waits) == (1, 2)
    assert code == 0
    assert (rig.out / "T-1.json").is_file()


def test_clean_clip_writes_sidecar(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.run(block="calibrate") == 0
    side = rig.sidecar()
    assert side["version"] == 1
    assert side["clip"] == "T-1"
    assert side["drifted"] is False
    assert side["block"] == "calibrate"
    assert side["video"] == "T-1.mkv"
    assert side["device"] == "/dev/video0"
    assert (side["width"], side["height"], side["fps"], side["format"]) == (1280, 720, 30, "MJPG")
    assert side["started_utc"] == "2026-11-14T10:00:00Z"
    assert side["stopped_utc"] == "2026-11-14T10:01:00Z"
    assert side["sha256"] == "fakehash"
    assert side["controls_before"] == side["controls_after"]
    assert {"name": "gain", "wanted": 64, "reported": 64} in side["controls_before"]


def test_drift_after_recording(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rig = Rig(tmp_path, after={v4l2.CID_GAIN: 80})
    assert rig.run() == 1
    assert rig.sidecar()["drifted"] is True
    assert {"name": "gain", "wanted": 64, "reported": 80} in rig.sidecar()["controls_after"]
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert "gain" in lines[0] and "64" in lines[0] and "80" in lines[0]


def test_drift_before_recording_refuses(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.controls.values[v4l2.CID_GAIN] = 80
    assert rig.run() == 2
    assert rig.started == []
    assert not (rig.out / "T-1.json").exists()


def test_unreported_control_is_null_not_drift(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    del rig.controls.values[v4l2.CID_FOCUS_ABSOLUTE]
    assert rig.run() == 0
    side = rig.sidecar()
    assert side["drifted"] is False
    assert {"name": "focus", "wanted": 30, "reported": None} in side["controls_after"]


def test_ffmpeg_failure_writes_no_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = Rig(tmp_path, code=1)
    assert rig.run() == 2
    assert not (rig.out / "T-1.json").exists()
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "1" in out[0]


@pytest.mark.parametrize("existing", ["T-1.mkv", "T-1.json"])
def test_existing_output_is_never_overwritten(tmp_path: Path, existing: str) -> None:
    rig = Rig(tmp_path)
    rig.out.mkdir()
    (rig.out / existing).write_text("keep")
    assert rig.run() == 2
    assert (rig.out / existing).read_text() == "keep"
    assert rig.started == []
