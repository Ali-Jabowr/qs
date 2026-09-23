"""Builders for the P04 station tests: small configs, synthetic PNG frames, `frames.csv`,
a fake clock, a scripted `FrameSource` and a recording sink.

A helper module, not a conftest: the P04 test files import it directly. Frames are 320×240, so
a 200-event run stays fast; every file is written under the `tmp_path` given.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from qs.__main__ import build_parser
from qs.config import LoadedConfig, load_config
from qs.contracts import CameraReadback, Frame, Health, InspectionRecord, TriggerEvent
from qs.session import FRAMES_COLUMNS

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"
WIDTH, HEIGHT = 320, 240
PATCH_ROI = [280, 10, 30, 30]
PATCH_GREY = 158  # 158 / 255 = 0.6196, inside 0.62 ± 5 %
VERIFIED = CameraReadback(10000, 0, 4500, 30, verified=True)
UNVERIFIED = CameraReadback(None, None, None, None, verified=False)


def config_mapping(root: Path, *, inspector_type: str = "stub", **latency: Any) -> dict[str, Any]:
    """The example config, resized to 320×240, with the evidence under `root/evidence`."""
    data: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["camera"]["width"] = WIDTH
    data["camera"]["height"] = HEIGHT
    data["lighting"]["reference_patch"]["roi"] = list(PATCH_ROI)
    data["inspector"]["localize"]["roi"] = [20, 0, 240, 240]
    data["inspector"]["type"] = inspector_type
    data["evidence"]["root"] = str(root / "evidence")
    data["latency"].update(latency)
    return data


def write_config(root: Path, *, name: str = "station.yml", **changes: Any) -> Path:
    """A valid config file under `root`; `changes` go to `config_mapping`."""
    path = root / name
    path.write_text(
        yaml.safe_dump(config_mapping(root, **changes), allow_unicode=True), encoding="utf-8"
    )
    return path


def loaded_config(root: Path, **changes: Any) -> LoadedConfig:
    return load_config(write_config(root, **changes))


def textured(seed: int, *, patch_grey: int = PATCH_GREY) -> np.ndarray:
    """A 320×240 BGR frame of seeded low-contrast noise, with the grey card painted in."""
    rng = np.random.default_rng(seed)
    pixels = (40 + rng.integers(0, 60, (HEIGHT, WIDTH, 3))).astype(np.uint8)
    x, y, w, h = PATCH_ROI
    pixels[y : y + h, x : x + w] = patch_grey
    return pixels


def write_png(path: Path, pixels: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", pixels)
    assert ok
    data = buffer.tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_frames(directory: Path, count: int, *, first_seed: int = 0) -> list[Path]:
    """`count` textured PNGs named f0001.png … in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(1, count + 1):
        path = directory / f"f{index:04d}.png"
        write_png(path, textured(first_seed + index))
        paths.append(path)
    return paths


def write_frames_csv(directory: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """`frames.csv` in P05b's columns; missing cells are empty."""
    with (directory / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FRAMES_COLUMNS)
        for row in rows:
            writer.writerow("" if row.get(c) is None else str(row.get(c)) for c in FRAMES_COLUMNS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def station_args(config: Path, replay: Path | None) -> argparse.Namespace:
    argv = ["station", "--config", str(config)]
    if replay is not None:
        argv += ["--replay", str(replay)]
    return build_parser().parse_args(argv)


def rows(database: Path, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


class FakeClock:
    """A `perf_counter` stand-in that moves only when told, or by `step` on every read."""

    def __init__(self, start: float = 1000.0, step: float = 0.0) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StepWall:
    """A wall clock: aware UTC, one millisecond later on every read."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 9, 26, 9, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(milliseconds=1)
        return self.now


def event(seq: int, t0: float = 0.0) -> TriggerEvent:
    wall = datetime(2026, 9, 26, 9, 0, 0, tzinfo=UTC) + timedelta(seconds=seq)
    return TriggerEvent(seq, t0, wall, "test")


class CountingTrigger:
    """Wraps a trigger and counts its polls."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.polls = 0

    def poll(self) -> TriggerEvent | None:
        self.polls += 1
        return self.inner.poll()  # type: ignore[no-any-return]

    @property
    def exhausted(self) -> bool:
        return bool(self.inner.exhausted)

    def close(self) -> None:
        self.inner.close()


Step = Callable[[TriggerEvent], Frame] | BaseException | str


@dataclass
class ScriptedSource:
    """A `FrameSource` whose `capture` is scripted per `seq`: an exception to raise, a
    `(patch, camera)` override, or nothing — a textured, verified, in-band frame. `opens` is a
    list of outcomes for successive `open()` calls (None succeeds); `open_times` the clock at
    each call."""

    clock: Callable[[], float]
    captures: dict[int, object] = field(default_factory=dict)
    opens: list[BaseException | None] = field(default_factory=list)
    patch: float = PATCH_GREY / 255
    camera: CameraReadback = VERIFIED
    preview_pixels: np.ndarray | None = None
    captured: list[int] = field(default_factory=list)
    open_times: list[float] = field(default_factory=list)
    closes: int = 0

    def open(self) -> None:
        self.open_times.append(self.clock())
        outcome = self.opens.pop(0) if self.opens else None
        if outcome is not None:
            raise outcome

    def capture(self, event: TriggerEvent) -> Frame:
        self.captured.append(event.seq)
        step = self.captures.get(event.seq)
        if isinstance(step, BaseException):
            raise step
        patch, camera = self.patch, self.camera
        if isinstance(step, tuple):
            patch, camera = step
        return Frame(
            event=event,
            pixels=textured(event.seq),
            grabbed_at=self.clock(),
            camera=camera,
            reference_patch=patch,
            capture_ms=5.0,
        )

    def preview(self) -> np.ndarray | None:
        return self.preview_pixels

    def health(self) -> Health:
        return Health(True, "scripted")

    def close(self) -> None:
        self.closes += 1


class RecordingSink:
    """A `StationSink` in memory: keeps every record; `healthy(n)` scripts `health()` after
    `n` writes; `fail=True` makes `write` raise."""

    def __init__(
        self, healthy: Callable[[int], bool] = lambda n: True, *, fail: bool = False
    ) -> None:
        self.records: list[InspectionRecord] = []
        self.healthy = healthy
        self.fail = fail
        self.finished: tuple[datetime, dict[str, object]] | None = None
        self.closed = False

    def write(self, record: InspectionRecord) -> None:
        if self.fail:
            raise OSError("injected sink failure")
        self.records.append(record)

    def pending(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True

    def health(self) -> Health:
        ok = self.healthy(len(self.records))
        return Health(ok, "ok" if ok else "storage failing (injected)")

    def counters(self) -> dict[str, int]:
        n = len(self.records)
        return {"written": n, "held": 0, "images_shed": 0, "write_failures": 0,
                "rejected": 0, "lost": 0, "pending": 0}

    def finish_run(self, ended_utc: datetime, counters: Mapping[str, object]) -> None:
        self.finished = (ended_utc, dict(counters))
