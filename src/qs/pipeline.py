"""The Pipeline: the only place that decides what an exception means (docs/11 §7).

One trigger goes in; one of PASS, DEFECT or NO_VERDICT with a reason comes out, is shown, and
is handed to the sink as exactly one record, which is also returned (C5, docs/16 §1). The
steps are the event table of #68; a verdict is accepted only through the guard of step 6, so a
label that disagrees with its own checks, a non-finite value or another config's verdict is
refused and never becomes a pass (C7, C8).

The Pipeline knows only the contracts. The trigger is polled by the station's main loop, which
hands each event to `handle()` (docs/11 §3).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

import numpy as np

from qs.config import StationConfig
from qs.contracts import (
    CHECK_APPEARANCE,
    CHECK_GEOMETRY,
    Frame,
    FrameSource,
    Health,
    InspectionRecord,
    Inspector,
    Label,
    Sink,
    TriggerEvent,
    Verdict,
    derive_label,
)
from qs.errors import (
    CameraLost,
    CameraSettingsDrift,
    CaptureError,
    LightingFault,
    PlacementFault,
)
from qs.inspector.heatmap import heatmap_png
from qs.store import encode_frame_jpeg

_log = logging.getLogger(__name__)

Clock = Callable[[], float]
WallClock = Callable[[], datetime]

# Station A decides with two checks (ADR-0008), for `appearance` and `stub` alike.
CONFIGURED_CHECKS = frozenset({CHECK_GEOMETRY, CHECK_APPEARANCE})


class State(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    INSPECTING = "INSPECTING"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


LIGHTING = "LIGHTING"
CAMERA_SETTINGS = "CAMERA_SETTINGS"


class Shows(Protocol):
    """The Annunciator as the Pipeline calls it: an event without a frame is shown without one.
    Every implementation that accepts `Frame | None` is also a P02 `Annunciator`."""

    def show(self, frame: Frame | None, verdict: Verdict) -> None: ...

    def close(self) -> None: ...


class StationSink(Sink, Protocol):
    """P02's `Sink` plus what `SqliteSink` provides beyond it (#59): the Pipeline reads
    `health()` and ends the run with `finish_run()`; the command reads `counters()`."""

    def health(self) -> Health: ...

    def counters(self) -> dict[str, int]: ...

    def finish_run(self, ended_utc: datetime, counters: Mapping[str, object]) -> None: ...


def utc_now() -> datetime:
    return datetime.now(UTC)


class _Refused(Exception):
    """A returned verdict the guard does not accept (C8)."""


@dataclass(slots=True)
class _Outcome:
    verdict: Verdict
    frame: Frame | None
    image_jpeg: bytes | None = None
    heatmap_png: bytes | None = None


def _finite_or_none(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


class Pipeline:
    """Trigger → capture → lighting → inspect → guard → encode → show → record."""

    def __init__(
        self,
        *,
        config: StationConfig,
        config_hash: str,
        run_id: str,
        source: FrameSource,
        inspector: Inspector,
        annunciator: Shows,
        sink: StationSink,
        clock: Clock = time.perf_counter,
        wall: WallClock = utc_now,
    ) -> None:
        self._config = config
        self._config_hash = config_hash
        self._run_id = run_id
        self._source = source
        self._inspector = inspector
        self._annunciator = annunciator
        self._sink = sink
        self._clock = clock
        self._wall = wall
        self._state = State.STARTING
        self._events = 0
        self._labels: dict[str, int] = {label.value: 0 for label in Label}
        self._reasons: dict[str, int] = {}
        self._annunciator_errors = 0
        self._sink_errors = 0
        self._lighting = False  # LIGHTING: the last patch read was out of band
        self._drift = False  # CAMERA_SETTINGS raised by drift, cleared by a verified frame
        self._unverifiable = False  # CAMERA_SETTINGS for the rest of the run (FR-13)

    # --- what the screen and the command read ---------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def warnings(self) -> frozenset[str]:
        """The active warnings (docs/16 §2): any number at once; they change what the screen
        shows, not what the station does."""
        active = set()
        if self._lighting:
            active.add(LIGHTING)
        if self._drift or self._unverifiable:
            active.add(CAMERA_SETTINGS)
        return frozenset(active)

    @property
    def counters(self) -> dict[str, object]:
        """The run's counters (docs/16 §6), JSON-serialisable; the sink adds its own."""
        return {
            "events": self._events,
            "labels": dict(self._labels),
            "no_verdict": dict(sorted(self._reasons.items())),
            "annunciator_errors": self._annunciator_errors,
            "sink_errors": self._sink_errors,
        }

    # --- start and stop ---------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera, then READY."""
        self._source.open()
        self._state = State.READY

    def stop(self) -> None:
        """The clean stop: `finish_run(ended_utc, counters)`, then `sink.close()` → STOPPED."""
        if self._state is State.STOPPED:
            return
        self._state = State.STOPPED
        try:
            self._sink.finish_run(self._wall(), self.counters)
        except Exception:
            _log.exception("the run's counters could not be queued")
        try:
            self._sink.close()
        except Exception:
            _log.exception("the sink did not close cleanly")
        try:
            self._source.close()
        except Exception:
            _log.exception("the source did not close cleanly")

    # --- one event ----------------------------------------------------------------------------

    def handle(self, event: TriggerEvent) -> InspectionRecord:
        """One trigger → exactly one record, shown and written, and returned (C5)."""
        self._state = State.INSPECTING
        outcome = self._inspect_event(event)
        self._state = State.READY
        return self._finish(event, outcome)

    def _inspect_event(self, event: TriggerEvent) -> _Outcome:
        """Steps 2–7 for one event in READY."""
        frame = self._capture(event)
        if not isinstance(frame, Frame):
            return frame
        # Step 3: the frame is accepted.
        if frame.camera.verified is True:
            self._drift = False
        else:
            self._unverifiable = True
        # Step 4: the reference patch (A1, #62).
        if not self._in_band(frame.reference_patch):
            self._lighting = True
            return self._no_verdict_with_frame("lighting", frame, 0.0)
        self._lighting = False
        # Step 5: inspect.
        inspect_start = self._clock()
        try:
            verdict = self._inspector.inspect(frame)
            inspect_ms = (self._clock() - inspect_start) * 1000.0
            self._guard(verdict)
        except PlacementFault as exc:
            inspect_ms = (self._clock() - inspect_start) * 1000.0
            _log.info("seq %d: placement: %s", event.seq, exc)
            return self._no_verdict_with_frame("placement", frame, inspect_ms)
        except Exception as exc:
            inspect_ms = (self._clock() - inspect_start) * 1000.0
            _log.warning("seq %d: inference failed: %s", event.seq, exc)
            return self._no_verdict_with_frame("inference", frame, inspect_ms)
        return self._encode(verdict, frame, inspect_ms)

    def _capture(self, event: TriggerEvent) -> Frame | _Outcome:
        """Step 2: a frame, or the NO_VERDICT its failure means. None of these has an image."""
        try:
            frame = self._source.capture(event)
        except CameraLost as exc:
            _log.warning("seq %d: camera lost: %s", event.seq, exc)
            return _Outcome(self._no_verdict("camera"), None)
        except CaptureError as exc:
            _log.warning("seq %d: capture failed: %s", event.seq, exc)
            return _Outcome(self._no_verdict("capture"), None)
        except CameraSettingsDrift as exc:
            _log.warning("seq %d: camera settings drift: %s", event.seq, exc)
            self._drift = True
            return _Outcome(self._no_verdict("camera_settings"), None)
        except LightingFault as exc:
            _log.warning("seq %d: lighting: %s", event.seq, exc)
            self._lighting = True
            return _Outcome(self._no_verdict("lighting"), None)
        except Exception as exc:  # a source that breaks its contract: no frame to trust
            _log.exception("seq %d: the source raised %s", event.seq, type(exc).__name__)
            return _Outcome(self._no_verdict("capture"), None)
        if not isinstance(frame, Frame) or frame.event != event:
            _log.warning("seq %d: the source returned a frame for another event", event.seq)
            return _Outcome(self._no_verdict("capture"), None)
        return frame

    def _in_band(self, patch: float | None) -> bool:
        """`|patch − expect| ≤ band × expect`, the boundary included; None is out of band."""
        reference = self._config.lighting.reference_patch
        if not _finite_or_none(patch) or patch is None:
            return False
        low = reference.expect * (1 - reference.band)
        high = reference.expect * (1 + reference.band)
        return low <= float(patch) <= high

    def _guard(self, verdict: object) -> None:
        """Step 6: accept a returned verdict only if nothing about it can hide a failed check."""
        if not isinstance(verdict, Verdict):
            raise _Refused(f"inspect() returned {type(verdict).__name__}, not a Verdict")
        if verdict.label is Label.NO_VERDICT:
            raise _Refused("an inspector returned NO_VERDICT instead of raising")
        try:
            derived = derive_label(verdict.checks, CONFIGURED_CHECKS)
        except ValueError as exc:
            raise _Refused(f"the checks do not make a verdict: {exc}") from exc
        if verdict.label is not derived:
            raise _Refused(f"label {verdict.label} disagrees with its checks ({derived})")
        for check in verdict.checks:
            for value in (check.value, check.limit, *(v for _, v in check.measures)):
                if not _finite_or_none(value):
                    raise _Refused(f"check {check.name!r} carries a non-finite value {value!r}")
        if verdict.config_hash != self._config_hash:
            raise _Refused("the verdict was made under another config")
        if verdict.threshold_id != self._config.threshold.id:
            raise _Refused("the verdict names another threshold")
        if verdict.model_id != self._inspector.model_id:
            raise _Refused("the verdict names another model")
        if not _finite_or_none(verdict.inspect_ms):
            raise _Refused("inspect_ms is not finite")
        heatmap = verdict.heatmap
        if heatmap is not None and (
            not isinstance(heatmap, np.ndarray)
            or heatmap.dtype.kind != "f"
            or not bool(np.isfinite(heatmap).all())
        ):
            raise _Refused("the heatmap is not a finite float array")

    def _encode(self, verdict: Verdict, frame: Frame, inspect_ms: float) -> _Outcome:
        """Step 7, on the pipeline thread (ADR-0012): the frame as JPEG q85, the heatmap as PNG."""
        try:
            image = encode_frame_jpeg(frame.pixels)
            heat = (
                None
                if verdict.heatmap is None
                else heatmap_png(verdict.heatmap, self._config.threshold.value)
            )
        except Exception as exc:
            _log.warning("seq %d: encoding failed: %s", frame.event.seq, exc)
            return self._no_verdict_with_frame("inference", frame, inspect_ms)
        return _Outcome(verdict, frame, image, heat)

    def _no_verdict(self, reason: str, inspect_ms: float = 0.0) -> Verdict:
        try:
            model_id = str(self._inspector.model_id)
        except Exception:
            model_id = "unknown"
        return Verdict(
            label=Label.NO_VERDICT,
            checks=(),
            heatmap=None,
            window_box=None,
            unit_box=None,
            model_id=model_id,
            threshold_id=self._config.threshold.id,
            config_hash=self._config_hash,
            inspect_ms=inspect_ms,
            reason=reason,
        )

    def _no_verdict_with_frame(self, reason: str, frame: Frame, inspect_ms: float) -> _Outcome:
        try:
            image: bytes | None = encode_frame_jpeg(frame.pixels)
        except Exception:
            image = None
        return _Outcome(self._no_verdict(reason, inspect_ms), frame, image)

    def _finish(self, event: TriggerEvent, outcome: _Outcome) -> InspectionRecord:
        """Steps 8–10: show, time, record."""
        verdict, frame = outcome.verdict, outcome.frame
        try:
            self._annunciator.show(frame, verdict)
        except Exception:
            self._annunciator_errors += 1
            _log.exception("seq %d: the annunciator failed; the record is still written", event.seq)
        total_ms = (self._clock() - event.t0) * 1000.0
        record = InspectionRecord(
            run_id=self._run_id,
            event=event,
            verdict=verdict,
            camera=None if frame is None else frame.camera,
            reference_patch=None if frame is None else frame.reference_patch,
            image_jpeg=outcome.image_jpeg,
            heatmap_png=outcome.heatmap_png,
            capture_ms=None if frame is None else frame.capture_ms,
            total_ms=total_ms,
        )
        self._events += 1
        self._labels[verdict.label.value] += 1
        if verdict.reason is not None:
            self._reasons[verdict.reason] = self._reasons.get(verdict.reason, 0) + 1
        try:
            self._sink.write(record)
        except Exception:
            self._sink_errors += 1
            _log.exception("seq %d: the sink raised (C3); counted", event.seq)
        return record
