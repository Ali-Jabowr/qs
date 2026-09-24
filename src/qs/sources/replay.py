"""Replay (docs/11 §6, FR-15): `ReplayTrigger` and `ReplaySource`, built from one ordered list
of PNG files, so that a frozen image set, a model bundle and a config give the same verdicts
every time.

- `ReplayTrigger` gives event *k* for file *k*: `seq` from 1, gapless; `t0` the clock at
  `poll()`; `wall` UTC now; `source_id` `replay:<name>`. It never blocks.
- `ReplaySource.capture(event)` reads file `seq`. When the file's folder has a `frames.csv`
  (#29) with a row naming the file, the bytes' SHA-256 must equal the row's, and the row gives
  the camera read-back. With no row, every value is `None` and `verified` is False: a read-back
  that did not happen is never assumed (FR-13). The frame is validated as the webcam validates
  it (shape from the config, uint8, not flat) and the reference patch is measured from the
  pixels, never read from `frames.csv`: replay is a pure function of the pixels and the config.

Every refusal of one frame is a `CaptureError`: its unit becomes NO_VERDICT(`capture`).
Anything that stops replay from starting at all is a `ReplayError` (exit 2).
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from qs.config import StationConfig
from qs.contracts import CameraReadback, Frame, Health, TriggerEvent
from qs.errors import CaptureError
from qs.session import FRAMES_FILE
from qs.sources.webcam import measure_patch
from qs.store import PNG_MAGIC

Clock = Callable[[], float]
WallClock = Callable[[], datetime]

_READBACK = ("exposure_us", "gain", "wb_kelvin", "focus")
_NEEDED = ("file", *_READBACK, "verified", "sha256")


class ReplayError(Exception):
    """Replay cannot start: no such directory, no PNG in it, or an unreadable `frames.csv`.

    Not a `QSError`: those are the Pipeline's NO_VERDICT reasons; this stops the station.
    """


def utc_now() -> datetime:
    return datetime.now(UTC)


def replay_files(directory: Path) -> list[Path]:
    """The `*.png` files directly in `directory`, in name order. No recursion, no JPEG."""
    if not directory.is_dir():
        raise ReplayError(f"replay: {directory} is not a directory")
    files = sorted(
        (p for p in directory.iterdir() if p.suffix == ".png" and p.is_file()),
        key=lambda p: p.name,
    )
    if not files:
        raise ReplayError(f"replay: {directory} has no *.png frame")
    return files


@dataclass(frozen=True, slots=True)
class Recorded:
    """One `frames.csv` row as replay uses it; `problem` set when the row cannot be trusted."""

    camera: CameraReadback
    sha256: str
    problem: str | None = None


def _int_cell(text: str, column: str) -> int | None:
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{column} {text!r} is not an integer") from None


def _row(cells: dict[str, str]) -> Recorded:
    try:
        values = [_int_cell(cells[c], c) for c in _READBACK]
        verified_text = cells["verified"]
        if verified_text not in ("true", "false"):
            raise ValueError(f"verified {verified_text!r} is not true or false")
    except ValueError as exc:
        unverified = CameraReadback(None, None, None, None, verified=False)
        return Recorded(unverified, cells["sha256"], f"frames.csv row: {exc}")
    # A read-back is verified only if it says so and every value is there.
    verified = verified_text == "true" and all(v is not None for v in values)
    camera = CameraReadback(values[0], values[1], values[2], values[3], verified=verified)
    return Recorded(camera, cells["sha256"])


def read_frames_csv(folder: Path) -> dict[str, Recorded]:
    """`folder/frames.csv` by file name; empty when there is none. A file named by two rows is
    kept with a `problem`: its metadata is ambiguous and it is never replayed as recorded."""
    path = folder / FRAMES_FILE
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in _NEEDED if c not in (reader.fieldnames or [])]
            if missing:
                raise ReplayError(f"replay: {path} lacks column(s) {', '.join(missing)}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ReplayError(f"replay: {path} cannot be read: {exc}") from exc
    recorded: dict[str, Recorded] = {}
    for cells in rows:
        clean = {k: (v or "") for k, v in cells.items() if k is not None}
        name = clean["file"]
        if name in recorded:
            first = recorded[name]
            recorded[name] = Recorded(first.camera, first.sha256, f"{name} has two frames.csv rows")
            continue
        recorded[name] = _row(clean)
    return recorded


class ReplayTrigger:
    """`TriggerSource` over a fixed number of frames: one event per `poll()`, then exhausted."""

    def __init__(
        self, count: int, name: str, *, clock: Clock, wall: WallClock = utc_now
    ) -> None:
        self._count = count
        self._next = 1
        self._source_id = f"replay:{name}"
        self._clock = clock
        self._wall = wall

    def poll(self) -> TriggerEvent | None:
        if self._next > self._count:
            return None
        event = TriggerEvent(self._next, self._clock(), self._wall(), self._source_id)
        self._next += 1
        return event

    @property
    def exhausted(self) -> bool:
        return self._next > self._count

    def close(self) -> None:
        self._next = self._count + 1


class ReplaySource:
    """`FrameSource` that presents each file as a camera would, with its recorded metadata."""

    def __init__(self, config: StationConfig, files: Sequence[Path], *, clock: Clock) -> None:
        self._config = config
        self._files = tuple(Path(f) for f in files)
        self._clock = clock
        self._recorded: dict[Path, dict[str, Recorded]] = {}
        for folder in {f.parent for f in self._files}:
            self._recorded[folder] = read_frames_csv(folder)
        self._open = False

    @property
    def files(self) -> tuple[Path, ...]:
        return self._files

    def open(self) -> None:
        missing = [str(f) for f in self._files if not f.is_file()]
        if missing:
            raise ReplayError(f"replay: frame(s) missing: {', '.join(missing[:3])}")
        self._open = True

    def capture(self, event: TriggerEvent) -> Frame:
        if not self._open:
            raise CaptureError("capture() called before open()")
        if not 1 <= event.seq <= len(self._files):
            raise CaptureError(f"replay has no frame for seq {event.seq}")
        path = self._files[event.seq - 1]
        start = self._clock()
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CaptureError(f"{path.name}: cannot be read: {exc}") from exc
        recorded = self._recorded[path.parent].get(path.name)
        if recorded is not None:
            if recorded.problem is not None:
                raise CaptureError(f"{path.name}: {recorded.problem}")
            if hashlib.sha256(data).hexdigest() != recorded.sha256:
                raise CaptureError(f"{path.name}: SHA-256 differs from its frames.csv row")
            camera = recorded.camera
        else:
            camera = CameraReadback(None, None, None, None, verified=False)
        pixels = self._decode(path.name, data)
        grabbed_at = self._clock()
        return Frame(
            event=event,
            pixels=pixels,
            grabbed_at=grabbed_at,
            camera=camera,
            reference_patch=measure_patch(pixels, self._config.lighting.reference_patch.roi),
            capture_ms=(grabbed_at - start) * 1000.0,
        )

    def _decode(self, name: str, data: bytes) -> np.ndarray:
        camera = self._config.camera
        if not data.startswith(PNG_MAGIC):
            raise CaptureError(f"{name}: not a PNG")
        buffer = np.frombuffer(data, dtype=np.uint8)
        try:
            pixels = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:
            raise CaptureError(f"{name}: cannot be decoded: {exc}") from exc
        if pixels is None:
            raise CaptureError(f"{name}: cannot be decoded")
        if pixels.shape != (camera.height, camera.width, 3) or pixels.dtype != np.uint8:
            raise CaptureError(f"corrupt frame {name}: {pixels.shape} {pixels.dtype}")
        if int(pixels.max()) - int(pixels.min()) <= 1:
            raise CaptureError(f"corrupt frame {name}: flat")
        pixels.setflags(write=False)
        return pixels

    def preview(self) -> np.ndarray | None:
        return None  # replay has no live view (C9)

    def health(self) -> Health:
        return Health(self._open, "replaying" if self._open else "not open")

    def close(self) -> None:
        self._open = False


def replay_pair(
    config: StationConfig,
    files: Sequence[Path],
    name: str,
    *,
    clock: Clock,
    wall: WallClock = utc_now,
) -> tuple[ReplayTrigger, ReplaySource]:
    """The trigger and the source over the same ordered files (the Python API, for P08)."""
    if not files:
        raise ReplayError("replay: no frame to replay")
    source = ReplaySource(config, files, clock=clock)
    return ReplayTrigger(len(files), name, clock=clock, wall=wall), source
