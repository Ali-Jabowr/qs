"""`qs record`: one clip from the USB webcam, every control forced to manual (V2-02, #151).

    qs record --camera record.yml --out DIR --clip T-1 --block train|calibrate|test|variant|ref
              [--seconds N]

The controls are set and read back, `ffmpeg` copies the camera's own JPEG frames into
`DIR/<clip>.mkv` (no re-compression, no sound), the controls are read back again when recording
stops, and `DIR/<clip>.json` says whether the camera held its settings. `record.yml` is read
here, not through `load_config`, whose flicker rule refuses the short exposures this command
exists for.

Exit codes: 0 recorded, controls held · 1 recorded, a control drifted · 2 refused or failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from qs.config import CameraConfig
from qs.errors import CameraLost
from qs.sources.v4l2 import (
    ControlReading,
    Controls,
    V4L2Controls,
    apply_controls,
    read_controls,
)

BLOCKS = ("train", "calibrate", "test", "variant", "ref")
_KEYS = (
    "device", "width", "height", "format", "fps", "power_line_hz", "exposure_us", "gain",
    "wb_kelvin", "focus",
)
_INPUT_FORMAT = {"MJPG": "mjpeg"}
_SIGINT_EXIT = 255  # what ffmpeg returns after a clean stop on SIGINT


class _Refused(Exception):
    """A reason to stop with exit code 2; its message is the one printed line."""


class Process(Protocol):
    def interrupt(self) -> None: ...

    def wait(self) -> int: ...


class _FfmpegProcess:
    def __init__(self, argv: list[str]) -> None:
        try:
            self._popen = subprocess.Popen(argv)
        except OSError as exc:
            raise _Refused(f"cannot start ffmpeg: {exc}") from exc

    def interrupt(self) -> None:
        self._popen.send_signal(signal.SIGINT)

    def wait(self) -> int:
        return self._popen.wait()


def _start_process(argv: list[str]) -> Process:
    return _FfmpegProcess(argv)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _Record:
    camera: CameraConfig
    fps: int


def add_parser(commands: Any) -> None:
    """Register `record` on the `qs` subcommands."""
    parser = commands.add_parser("record", help="record one clip with the camera locked")
    parser.add_argument("--camera", required=True, metavar="PATH", help="record.yml")
    parser.add_argument("--out", required=True, metavar="DIR", help="where the clip goes")
    parser.add_argument("--clip", required=True, metavar="NAME", help="the clip's name, e.g. T-1")
    parser.add_argument("--block", required=True, choices=BLOCKS, help="the clip's role")
    parser.add_argument("--seconds", type=int, default=None, metavar="N",
                        help="stop after N seconds (default: until Ctrl-C)")


def _load_camera(path: Path) -> _Record:
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise _Refused(f"camera file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise _Refused(f"camera file {path}: not a mapping")
    missing = [key for key in _KEYS if key not in raw]
    extra = sorted(set(raw) - set(_KEYS))
    if missing or extra:
        raise _Refused(f"camera file {path}: missing {missing}, unknown {extra}")
    if raw["format"] not in _INPUT_FORMAT:
        raise _Refused(f"camera file {path}: format {raw['format']!r} is not MJPG")
    fps = raw["fps"]
    if not isinstance(fps, int) or fps <= 0:
        raise _Refused(f"camera file {path}: fps {fps!r} is not a positive whole number")
    camera = CameraConfig(
        type="webcam",
        device=str(raw["device"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        format=str(raw["format"]),
        power_line_hz=int(raw["power_line_hz"]),
        exposure_us=int(raw["exposure_us"]),
        gain=int(raw["gain"]),
        wb_kelvin=int(raw["wb_kelvin"]),
        focus=int(raw["focus"]),
        capture_timeout_ms=500,
    )
    return _Record(camera, fps)


def _drift(readings: tuple[ControlReading, ...]) -> list[ControlReading]:
    """The controls the camera reports with a different value; an unreadable one is not drift."""
    return [r for r in readings if r.reported is not None and not r.ok]


def _readings(readings: tuple[ControlReading, ...]) -> list[dict[str, Any]]:
    return [{"name": r.name, "wanted": r.wanted, "reported": r.reported} for r in readings]


def _ffmpeg_argv(record: _Record, video: Path, seconds: int | None) -> list[str]:
    camera = record.camera
    argv = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
        "-input_format", _INPUT_FORMAT[camera.format],
        "-video_size", f"{camera.width}x{camera.height}", "-framerate", str(record.fps),
        "-i", camera.device, "-an", "-c:v", "copy",
    ]
    if seconds is not None:
        argv += ["-t", str(seconds)]
    return argv + [str(video)]


def _wait(process: Process) -> tuple[int, bool]:
    """Wait for ffmpeg; on Ctrl-C send it SIGINT and wait again. Returns (code, interrupted)."""
    try:
        return process.wait(), False
    except KeyboardInterrupt:
        process.interrupt()
        return process.wait(), True


def main(
    args: argparse.Namespace,
    *,
    start_process: Callable[[list[str]], Process] = _start_process,
    make_controls: Callable[[str], Controls] = V4L2Controls,
    clock: Callable[[], str] = _now,
    file_hash: Callable[[Path], str] = _sha256,
) -> int:
    """Record one clip; 0 held, 1 drifted, 2 refused or failed."""
    try:
        return _record(args, start_process, make_controls, clock, file_hash)
    except _Refused as exc:
        print(str(exc))
        return 2


def _record(
    args: argparse.Namespace,
    start_process: Callable[[list[str]], Process],
    make_controls: Callable[[str], Controls],
    clock: Callable[[], str],
    file_hash: Callable[[Path], str],
) -> int:
    record = _load_camera(Path(args.camera))
    out = Path(args.out)
    video = out / f"{args.clip}.mkv"
    sidecar = out / f"{args.clip}.json"
    for path in (video, sidecar):
        if path.exists():
            raise _Refused(f"{path} already exists; nothing overwritten")
    try:
        controls = make_controls(record.camera.device)
    except CameraLost as exc:
        raise _Refused(f"camera: {exc}") from exc
    try:
        before = apply_controls(controls, record.camera)
        if bad := _drift(before):
            first = bad[0]
            raise _Refused(
                f"{first.name}: set {first.wanted}, camera reports {first.reported} "
                "before recording; nothing recorded"
            )
        out.mkdir(parents=True, exist_ok=True)
        started = clock()
        process = start_process(_ffmpeg_argv(record, video, args.seconds))
        code, interrupted = _wait(process)
        stopped = clock()
        after = read_controls(controls, record.camera)
    finally:
        controls.close()
    if code != 0 and not (interrupted and code == _SIGINT_EXIT):
        raise _Refused(f"ffmpeg exited with code {code}; no sidecar written")
    drifted = _drift(after)
    body = {
        "version": 1,
        "clip": args.clip,
        "block": args.block,
        "video": video.name,
        "device": record.camera.device,
        "width": record.camera.width,
        "height": record.camera.height,
        "fps": record.fps,
        "format": record.camera.format,
        "started_utc": started,
        "stopped_utc": stopped,
        "controls_before": _readings(before),
        "controls_after": _readings(after),
        "drifted": bool(drifted),
        "sha256": file_hash(video),
    }
    try:
        sidecar.write_text(json.dumps(body, indent=2) + "\n")
    except OSError as exc:
        raise _Refused(f"cannot write {sidecar}: {exc}") from exc
    for reading in drifted:
        print(f"{reading.name}: set {reading.wanted}, camera reports {reading.reported} "
              "after recording; clip marked drifted")
    return 1 if drifted else 0
