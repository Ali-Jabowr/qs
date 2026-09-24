"""Builders for the station-with-a-screen tests (P09 #89–#91): a scripted `FakeScreen` on the
fake clock, a spying `ScreenAnnunciator`, a sink that keeps its records, and `Rig.run`, which
calls `qs station`'s `main` with every factory injected. No display is needed.

A helper module, not a conftest: the P09 station test files import it directly."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from conftest import FakeScreen
from qs.__main__ import build_parser
from qs.cmd.station import main
from qs.config import DisplayConfig, LoadedConfig, StationConfig
from qs.contracts import Frame, InspectionRecord, Verdict
from qs.inspector.stub import StubInspector
from qs.operator_view import Shown, Status
from qs.pipeline import StationSink
from qs.screen import KeyPress, ScreenLike
from qs.screen_annunciator import ScreenAnnunciator
from qs.sink import SqliteSink
from station_fixtures import FakeClock, StepWall, rows

GAP = 15  # empty key batches after a press: 15 × 30 ms clears the 300 ms rule with room
BAND = (5, 40)  # a band pixel the verdict word never covers
BANNER_AT = (2, 2)
LIVE_AT = (320, 456)  # the centre of the live box
CORNERS = ((0, 0), (1279, 0), (0, 799), (1279, 799))


def space() -> KeyPress:
    return KeyPress("space", 5000)


def presses(count: int, *, then: Sequence[Sequence[KeyPress]] = ()) -> list[list[KeyPress]]:
    """An empty first batch (a press before the first poll never counts), then `count`
    presses, each followed by `GAP` empty batches; then `then`."""
    script: list[list[KeyPress]] = [[]]
    for _ in range(count):
        script.append([space()])
        script.extend([] for _ in range(GAP))
    script.extend(list(batch) for batch in then)
    return script


def args(config: Path, replay: Path | None, *, screen: bool) -> argparse.Namespace:
    argv = ["station", "--config", str(config)]
    if replay is not None:
        argv += ["--replay", str(replay)]
    if screen:
        argv.append("--screen")
    return build_parser().parse_args(argv)


def stub(script: dict[int, str]) -> Callable[[LoadedConfig], StubInspector]:
    def make(loaded: LoadedConfig) -> StubInspector:
        return StubInspector(loaded.config_hash, script)

    return make


class Spy(ScreenAnnunciator):
    """The real `ScreenAnnunciator`, noting what it was given and what it presented: each
    status and live picture it refreshed with, each verdict and the `last_shown` it made,
    and the band colour, the four corners and the live box of every presented image."""

    def __init__(self, config: StationConfig, screen: ScreenLike) -> None:
        super().__init__(config, screen)
        self.screen = screen
        self.statuses: list[Status] = []
        self.lives: list[np.ndarray | None] = []
        self.timeline: list[tuple[str, int | None, tuple[int, ...]]] = []
        self.corners: list[set[tuple[int, ...]]] = []
        self.live_box: list[tuple[int, ...]] = []
        self.banner: list[tuple[int, ...]] = []
        self.verdicts: list[Verdict] = []
        self.shown: list[Shown | None] = []

    def _band(self) -> tuple[int, ...]:
        presented: list[Image.Image] = self.screen.presented  # type: ignore[attr-defined]
        image = presented[-1]
        self.corners.append({tuple(image.getpixel(c)) for c in CORNERS})  # type: ignore[arg-type]
        self.live_box.append(tuple(image.getpixel(LIVE_AT)))  # type: ignore[arg-type]
        self.banner.append(tuple(image.getpixel(BANNER_AT)))  # type: ignore[arg-type]
        return tuple(image.getpixel(BAND))  # type: ignore[arg-type]

    def show(self, frame: Frame | None, verdict: Verdict) -> None:
        try:
            super().show(frame, verdict)
        finally:
            self.verdicts.append(verdict)
            self.shown.append(self.last_shown)
            self.timeline.append(
                ("show", None if frame is None else frame.event.seq, self._band())
            )

    def refresh(self, status: Status, live: np.ndarray | None) -> None:
        self.statuses.append(status)
        self.lives.append(live)
        try:
            super().refresh(status, live)
        finally:
            self.timeline.append(("refresh", None, self._band()))


@dataclass
class Rig:
    """One `qs station` run with fakes: the clock, the sleep, the screens and the spies."""

    clock: FakeClock = field(default_factory=FakeClock)
    sleeps: list[float] = field(default_factory=list)
    screens: list[FakeScreen] = field(default_factory=list)
    spies: list[Spy] = field(default_factory=list)
    oversleep: float = 0.0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.clock.advance(seconds + self.oversleep)

    def screen_factory(self, **options: Any) -> Callable[[DisplayConfig, str, Any], FakeScreen]:
        def make(display: DisplayConfig, title: str, clock: Any) -> FakeScreen:
            assert clock is self.clock  # one clock for the screen, the trigger and the Pipeline
            screen = FakeScreen(clock=clock, keep=2, **options)
            self.screens.append(screen)
            return screen

        return make

    def spy(self, config: StationConfig, screen: ScreenLike) -> Spy:
        spy = Spy(config, screen)
        self.spies.append(spy)
        return spy

    def run(self, config: Path, replay: Path | None, **options: Any) -> int:
        """`--replay DIR --screen` when `replay` is given, live otherwise; `screen=False` with
        a `replay` is P04's headless replay."""
        screen_options = options.pop("screen_options", {})
        screen = options.pop("screen", replay is not None)
        return main(
            args(config, replay, screen=screen),
            make_screen=options.pop("make_screen", self.screen_factory(**screen_options)),
            make_screen_annunciator=self.spy,
            sleep=self.sleep,
            clock=self.clock,
            wall=options.pop("wall", StepWall()),
            **options,
        )


def closing_after(script: Sequence[Sequence[KeyPress]]) -> dict[str, Any]:
    """Screen options: play `script`, then close the window."""
    return {"script": script, "close_when": lambda calls: calls >= len(script)}


class RecordingSqlite:
    """The real sink, keeping every record handed to it."""

    def __init__(self, inner: StationSink) -> None:
        self.inner = inner
        self.written: list[InspectionRecord] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def write(self, record: InspectionRecord) -> None:
        self.written.append(record)
        self.inner.write(record)


def recording(sinks: list[RecordingSqlite]) -> Callable[..., StationSink]:
    def make(loaded: LoadedConfig, run_id: str, started: datetime) -> StationSink:
        sink = RecordingSqlite(SqliteSink.from_config(loaded, run_id, started))
        sinks.append(sink)
        return sink  # type: ignore[return-value]

    return make


def database(root: Path) -> Path:
    return root / "evidence" / "qs.db"


def run_row(root: Path) -> Any:
    [run] = rows(database(root), "SELECT * FROM run")
    return run
