"""The station runner: builds the implementations from the loaded config, runs the start-up
sequence, the main loop of docs/11 §3 and the stop (#68).

Start-up, each step failing closed: the config is loaded by the caller; `inspector.load()`;
then, and only then, the sink with a new run; then the camera. The main loop polls the trigger
and hands each event to the Pipeline until the trigger is exhausted or a stop is requested.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from qs.config import LoadedConfig
from qs.contracts import FrameSource, Inspector, TriggerSource
from qs.errors import ConfigError
from qs.inspector.stub import StubInspector
from qs.pipeline import (
    Clock,
    Pipeline,
    Shows,
    StationSink,
    WallClock,
    check_config,
    utc_now,
)

MakeSink = Callable[[LoadedConfig, str, datetime], StationSink]


class StartupRefused(Exception):
    """A start-up check failed: the station is STOPPED with exit code 2. One line."""


def run_id_for(started: datetime) -> str:
    """The run ID: the UTC start as `YYYYMMDDTHHMMSSffffffZ` — path-safe and sortable."""
    return started.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def build_inspector(loaded: LoadedConfig) -> Inspector:
    """The Inspector the config names: one `match` on `inspector.type` (docs/11 §8)."""
    match loaded.config.inspector.type:
        case "stub":
            return StubInspector(loaded.config_hash)
        case other:
            raise StartupRefused(
                f"inspector.type {other!r}: the AppearanceInspector is not wired into the "
                "station yet (#74)"
            )


class Station:
    """One run: start-up, the main loop and the stop, around one Pipeline."""

    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        trigger: TriggerSource,
        source: FrameSource,
        inspector: Inspector,
        annunciator: Shows,
        make_sink: MakeSink,
        clock: Clock = time.perf_counter,
        wall: WallClock = utc_now,
    ) -> None:
        self.loaded = loaded
        self.trigger = trigger
        self.source = source
        self.inspector = inspector
        self.annunciator = annunciator
        self._make_sink = make_sink
        self._clock = clock
        self._wall = wall
        self.run_id: str | None = None
        self.sink: StationSink | None = None
        self.pipeline: Pipeline | None = None

    def start(self) -> Pipeline:
        """Start-up in #68's order. `StartupRefused` for every refusal; nothing is written to
        the evidence before `inspector.load()` has succeeded."""
        config = self.loaded.config
        try:
            check_config(config)
        except ConfigError as exc:
            raise StartupRefused(f"config: {exc}") from exc
        try:
            self.inspector.load(config)
        except Exception as exc:
            raise StartupRefused(f"inspector: {exc}") from exc
        started = self._wall()
        self.run_id = run_id_for(started)
        self.sink = self._make_sink(self.loaded, self.run_id, started)
        self.pipeline = Pipeline(
            config=config,
            config_hash=self.loaded.config_hash,
            run_id=self.run_id,
            source=self.source,
            inspector=self.inspector,
            annunciator=self.annunciator,
            sink=self.sink,
            clock=self._clock,
            wall=self._wall,
        )
        try:
            self.pipeline.start()
        except Exception as exc:
            self.pipeline.stop()
            raise StartupRefused(f"camera: {exc}") from exc
        return self.pipeline

    def run(self) -> None:
        """The main loop of docs/11 §3: tick, poll, handle — until the trigger is exhausted,
        a stop is requested, or the station stopped itself."""
        pipeline = self.pipeline
        assert pipeline is not None, "start() first"
        while (
            not self.trigger.exhausted
            and not pipeline.stop_requested
            and not pipeline.stopped
        ):
            pipeline.tick()
            if pipeline.stopped:
                break
            event = self.trigger.poll()
            if event is not None:
                pipeline.handle(event)

    def stop(self) -> None:
        """No more polling; the run's counters and the sink's drain (Pipeline.stop)."""
        try:
            self.trigger.close()
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()
