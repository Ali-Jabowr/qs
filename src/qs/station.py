"""The station runner: builds the implementations from the loaded config, runs the start-up
sequence, the main loop of docs/11 §3 and the stop (#68).

Start-up, each step failing closed: the config is loaded by the caller; `inspector.load()`;
then, and only then, the sink with a new run; then the camera. The main loop polls the trigger
and hands each event to the Pipeline until the trigger is exhausted or a stop is requested.

With a screen (P09, #89) the loop gains a screen step (docs/11 §3). Each iteration refreshes
the operator screen with the station's state, stops when the window was closed, polls the
trigger (which reads the window's keys), and then sleeps out the rest of a 30 ms iteration.
The Screen, the trigger and the Pipeline run on one clock (#88). Without a screen the loop is
P04's and never sleeps.

`inspector.type: appearance` is P07's `AppearanceInspector` on the bundles under `models/`,
never in validation mode: the station runs only a bundle its config names, and never a
validation-only one (docs/17 §6, ADR-0016). Its scorer factory and CUDA probe are the real
ones — torch and Anomalib are imported only when a scorer is built — unless a caller passes
fakes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

import numpy as np

from qs.config import LoadedConfig
from qs.contracts import FrameSource, Health, InspectionRecord, Inspector, TriggerSource
from qs.errors import ConfigError
from qs.inspector.appearance import AppearanceInspector
from qs.inspector.model import Scorer
from qs.inspector.stub import StubInspector
from qs.operator_view import Status
from qs.pipeline import (
    CAMERA_SETTINGS,
    Clock,
    Pipeline,
    Shows,
    State,
    StationSink,
    WallClock,
    check_config,
    utc_now,
)
from qs.screen import ScreenLike
from qs.sources.replay import ReplayTrigger
from qs.sources.v4l2 import to_readback

_log = logging.getLogger(__name__)

MakeSink = Callable[[LoadedConfig, str, datetime], StationSink]
MakeScorer = Callable[[str, np.ndarray], Scorer]

MODELS_ROOT = "models"  # the bundles, relative to the working directory, as `qs train` writes
ITERATION_S = 0.030  # the screen loop's shortest iteration (docs/11 §3)
REPLAY_FINISHED = "ПОВТОР ЗАВЕРШЁН — закройте окно"
STARTING = Status("starting", None, frozenset(), None)


def replay_caption(next_seq: int, frames: int) -> str:
    """The key-paced replay's caption: the frame the next accepted press takes."""
    return f"ПОВТОР: кадр {next_seq} из {frames}"


class Refreshes(Shows, Protocol):
    """The Annunciator of a station with a screen: it also presents the state between events."""

    def refresh(self, status: Status, live: np.ndarray | None) -> None: ...


@dataclass(frozen=True, slots=True)
class ScreenLoop:
    """What the main loop's screen step needs: the window, the annunciator that draws on it,
    the injected sleep, the trigger's `source_id` and, in key-paced replay, the number of
    frames for the caption.

    The evidence store has no column for an event's `source_id` (P03), so a run with a
    screen names it in `run.counters` as `source_id`: `replay:<DIR name>` or
    `keyboard:<key>`. A replayed run can then never pass for a live one (question #94)."""

    screen: ScreenLike
    annunciator: Refreshes
    sleep: Callable[[float], None] = time.sleep
    frames: int | None = None
    source_id: str | None = None


def station_status(pipeline: Pipeline, caption: str | None) -> Status:
    """The Pipeline's state as the screen shows it, by name: FAULT → `fault` with its reason,
    STARTING → `starting`, anything else → `ready`; the warnings by their names."""
    warnings = frozenset(pipeline.warnings)
    if pipeline.state is State.FAULT:
        return Status("fault", pipeline.fault_reason, warnings, caption)
    mode = "starting" if pipeline.state is State.STARTING else "ready"
    return Status(mode, None, warnings, caption)


class _WithRunnerCounters:
    """The sink, with the runner's own counters (`keys_discarded`, `screen_errors`) merged
    into `finish_run`, so they reach `run.counters` with no change to the Pipeline."""

    def __init__(self, sink: StationSink, extra: Callable[[], Mapping[str, object]]) -> None:
        self._sink = sink
        self._extra = extra

    def write(self, record: InspectionRecord) -> None:
        self._sink.write(record)

    def pending(self) -> int:
        return self._sink.pending()

    def close(self) -> None:
        self._sink.close()

    def health(self) -> Health:
        return self._sink.health()

    def counters(self) -> dict[str, int]:
        return self._sink.counters()

    def finish_run(self, ended_utc: datetime, counters: Mapping[str, object]) -> None:
        self._sink.finish_run(ended_utc, {**counters, **self._extra()})


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
        screen: ScreenLoop | None = None,
    ) -> None:
        self.loaded = loaded
        self.trigger = trigger
        self.source = source
        self.inspector = inspector
        self.annunciator = annunciator
        self._make_sink = make_sink
        self._clock = clock
        self._wall = wall
        self._screen = screen
        self.run_id: str | None = None
        self.sink: StationSink | None = None
        self.pipeline: Pipeline | None = None
        self.events = 0  # events the main loop handed to the Pipeline
        self.screen_errors = 0  # refreshes of the screen that raised
        self._unverifiable_open = False  # the camera's settings could not all be read at open

    def armed(self) -> bool:
        """Whether a key press may make an event: the Pipeline exists and is not STARTING."""
        return self.pipeline is not None and self.pipeline.state is not State.STARTING

    @property
    def keys_discarded(self) -> int:
        """Key presses the trigger discarded; 0 for a trigger without keys."""
        return int(getattr(self.trigger, "discarded", 0))

    def runner_counters(self) -> dict[str, object]:
        """The runner's counters, which go into `run.counters` beside the Pipeline's."""
        counters: dict[str, object] = {
            "keys_discarded": self.keys_discarded,
            "screen_errors": self.screen_errors,
        }
        if self._screen is not None and self._screen.source_id is not None:
            counters["source_id"] = self._screen.source_id
        return counters

    def start(self) -> Pipeline:
        """Start-up in #68's order. `StartupRefused` for every refusal; nothing is written to
        the evidence before `inspector.load()` has succeeded. With a screen, ЗАПУСК is
        presented first."""
        config = self.loaded.config
        if self._screen is not None:
            self._screen.annunciator.refresh(STARTING, None)
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
        if self._screen is not None:
            self.sink = _WithRunnerCounters(self.sink, self.runner_counters)
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
        if self._screen is not None:
            self._run_with_screen(pipeline, self._screen)
            return
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

    def _run_with_screen(self, pipeline: Pipeline, loop: ScreenLoop) -> None:
        """One iteration: tick; refresh the screen with the state and, while READY, the live
        picture; stop if the window was closed; else poll and handle; then sleep so that the
        iteration lasts at least 30 ms. An exhausted trigger is still polled, so that it
        reads the keys and discards and counts every press. The loop ticks on the clock, as a
        live station does, whatever its frames come from."""
        clock = self._clock
        while not pipeline.stop_requested and not pipeline.stopped:
            began = clock()
            pipeline.tick()
            if pipeline.stopped:
                break
            self._refresh(pipeline, loop)
            if loop.screen.closed:
                pipeline.request_stop()
                break
            event = self.trigger.poll()
            if event is not None:
                self.events += 1
                pipeline.handle(event)
            loop.sleep(max(0.0, ITERATION_S - (clock() - began)))

    def _caption(self, loop: ScreenLoop) -> str | None:
        if loop.frames is None:
            return None
        if self.trigger.exhausted:
            return REPLAY_FINISHED
        return replay_caption(self.events + 1, loop.frames)

    def _live(self, pipeline: Pipeline) -> np.ndarray | None:
        """The live picture while READY (FR-31): the source's preview, only ever drawn in the
        live box (C9). A preview that raises draws no picture this time."""
        if pipeline.state is not State.READY:
            return None
        try:
            return self.source.preview()
        except Exception:
            _log.warning("the live view could not be read", exc_info=True)
            return None

    def _unverifiable(self, pipeline: Pipeline) -> bool:
        """docs/16 §2 start-up step 3 (REVIEW-P04 n3): a camera whose settings could not all
        be read back at `open()` starts with `CAMERA_SETTINGS`, and keeps it for the run.

        The Pipeline raises it at the first unverified frame; the screen shows it from the
        moment the camera is open. Read from the source's `readings` when it keeps them
        (`WebcamSource`); a source without them adds nothing."""
        if not self._unverifiable_open and pipeline.state is State.READY:
            readings = getattr(self.source, "readings", ())
            if readings and not to_readback(tuple(readings)).verified:
                self._unverifiable_open = True
        return self._unverifiable_open

    def _refresh(self, pipeline: Pipeline, loop: ScreenLoop) -> None:
        """Present the state between events. A refresh that raises has already put plain
        amber up (#87). It is logged and counted, and the loop goes on: the next event is
        still decided and recorded, and no earlier verdict can come back up (I6)."""
        live = self._live(pipeline)
        status = station_status(pipeline, self._caption(loop))
        if self._unverifiable(pipeline):
            status = replace(status, warnings=status.warnings | {CAMERA_SETTINGS})
        try:
            loop.annunciator.refresh(status, live)
        except Exception:
            self.screen_errors += 1
            _log.exception("the screen could not be refreshed")

    def stop(self) -> None:
        """No more polling; the run's counters and the sink's drain (Pipeline.stop)."""
        try:
            self.trigger.close()
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()
