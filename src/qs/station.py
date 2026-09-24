"""The station runner: builds the implementations from the loaded config, runs the start-up
sequence, the main loop of docs/11 §3 and the stop (#68).

Start-up, each step failing closed: the config is loaded by the caller; `inspector.load()`;
then, and only then, the sink with a new run; then the camera. The main loop polls the trigger
and hands each event to the Pipeline until the trigger is exhausted or a stop is requested.

`inspector.type: appearance` is P07's `AppearanceInspector` on the bundles under `models/`,
never in validation mode: the station runs only a bundle its config names, and never a
validation-only one (docs/17 §6, ADR-0016). Its scorer factory and CUDA probe are the real
ones — torch and Anomalib are imported only when a scorer is built — unless a caller passes
fakes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from qs.config import LoadedConfig
from qs.contracts import FrameSource, Inspector, TriggerSource
from qs.errors import ConfigError
from qs.inspector.appearance import AppearanceInspector
from qs.inspector.model import Scorer
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
from qs.sources.replay import ReplayTrigger

MakeSink = Callable[[LoadedConfig, str, datetime], StationSink]
MakeScorer = Callable[[str, np.ndarray], Scorer]

MODELS_ROOT = "models"  # the bundles, relative to the working directory, as `qs train` writes


class StartupRefused(Exception):
    """A start-up check failed: the station is STOPPED with exit code 2. One line."""


def run_id_for(started: datetime) -> str:
    """The run ID: the UTC start as `YYYYMMDDTHHMMSSffffffZ` — path-safe and sortable."""
    return started.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def real_scorer(device: str, bank: np.ndarray) -> Scorer:
    """The real PatchCore scorer on `device`; torch and Anomalib are imported here only."""
    from qs.inspector.patchcore import AnomalibPatchCore

    return AnomalibPatchCore(device).scorer(bank)


def real_cuda_available() -> bool:
    """Whether torch sees a CUDA device; torch is imported here only."""
    from qs.inspector.patchcore import cuda_available

    return cuda_available()


def build_inspector(
    loaded: LoadedConfig,
    *,
    make_scorer: MakeScorer = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
) -> Inspector:
    """The Inspector the config names: one `match` on `inspector.type` (docs/11 §8).

    `appearance` is never built in validation mode: a validation-only bundle is refused by
    its `load()`, and the station has no way to ask otherwise."""
    match loaded.config.inspector.type:
        case "stub":
            return StubInspector(loaded.config_hash)
        case "appearance":
            return AppearanceInspector(
                loaded.config_hash,
                models_root=MODELS_ROOT,
                make_scorer=make_scorer,
                cuda_available=cuda_available,
            )
        case other:
            raise StartupRefused(f"inspector.type {other!r}: no such Inspector")


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
        a stop is requested, or the station stopped itself.

        A replay's events are files, not units arriving in time, and it polls faster than any
        backoff: on wall time, a FAULT would outlast the run by a margin set by disk speed. So a
        replay ticks on the reconnect's own due time — one attempt per tick while in FAULT —
        and its verdicts depend on the frames alone (NFR-03). Live triggers tick on the clock."""
        pipeline = self.pipeline
        assert pipeline is not None, "start() first"
        replay = isinstance(self.trigger, ReplayTrigger)
        while (
            not self.trigger.exhausted
            and not pipeline.stop_requested
            and not pipeline.stopped
        ):
            pipeline.tick(pipeline.reconnect_due if replay else None)
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
