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
from collections import deque
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
    CheckResult,
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
    ConfigError,
    LightingFault,
    PlacementFault,
)
from qs.inspector.heatmap import heatmap_png
from qs.sources.webcam import measure_patch
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
STORAGE = "STORAGE"
LATENCY = "LATENCY"
LATENCY_WINDOW = 50  # inspections (F8)
LATENCY_SHARE = 0.10  # LATENCY while more than this share of the window broke the budget

IN_A_ROW = 3  # capture or inference errors in a row → FAULT (docs/16 §2)
BACKOFF_S = (0.5, 1.0, 2.0, 4.0, 8.0)  # waits between reconnect attempts (F1) …
BACKOFF_THEN_S = 10.0  # … then every 10 s
RETRIED_FAULTS = frozenset({"camera", "capture"})  # FAULT(inference) is never retried (F7)
BREACH_ACTIONS = ("record_and_continue",)  # the only latency.breach_action known (F8)


def check_config(config: StationConfig) -> None:
    """What the Pipeline refuses to start with, beyond the loader: an unknown
    `latency.breach_action` (the loader is not changed). `ConfigError`, one line."""
    action = config.latency.breach_action
    if action not in BREACH_ACTIONS:
        raise ConfigError(
            f"latency.breach_action: {action!r} is not one of: {', '.join(BREACH_ACTIONS)}"
        )


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
        self._capture_streak = 0  # A2, #63: changed only by events that reach capture
        self._inference_streak = 0  # … and only by events that reach inspect
        self._fault_reason: str | None = None
        self._faults: dict[str, int] = {}
        self._reconnect_attempts = 0
        self._attempt = 0  # reconnect attempts in the current FAULT
        self._next_attempt = 0.0
        self._stop_requested = False
        self._stopped = False
        self._refusal: str | None = None
        self._final_state = State.STARTING.value
        self._storage_bad = False  # STORAGE: the sink's health() was not ok at the last read
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)  # total_ms, inspections
        self._breaches = 0
        self._frames_accepted = 0
        self._last_finished: float | None = None  # clock when the previous handle() returned
        self._reached_inspect = False  # this event reached step 5
        self._accepted = False  # this event's frame passed step 3

    # --- what the screen and the command read ---------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def fault_reason(self) -> str | None:
        """`camera`, `capture` or `inference` while in FAULT; else None."""
        return self._fault_reason

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def stopped(self) -> bool:
        """True once the stop sequence has run: the state is STOPPED for good."""
        return self._stopped

    @property
    def refusal(self) -> str | None:
        """Why the station stopped itself — a read-back mismatch on a reconnect — or None."""
        return self._refusal

    @property
    def final_state(self) -> str:
        """The state the run ended in, as `READY` or `FAULT(<reason>)`; set by `stop()`."""
        return self._final_state

    @property
    def warnings(self) -> frozenset[str]:
        """The active warnings (docs/16 §2): any number at once; they change what the screen
        shows, not what the station does."""
        active = set()
        if self._lighting:
            active.add(LIGHTING)
        if self._drift or self._unverifiable:
            active.add(CAMERA_SETTINGS)
        if self._storage_bad:
            active.add(STORAGE)
        if self._latency_high():
            active.add(LATENCY)
        return frozenset(active)

    def _latency_high(self) -> bool:
        """More than 10% of the last 50 inspections (all of them, while fewer) broke the budget."""
        if not self._latencies:
            return False
        budget = self._config.latency.budget_ms
        over = sum(1 for total in self._latencies if total > budget)
        return over > LATENCY_SHARE * len(self._latencies)

    def _percentile(self, q: float) -> float | None:
        if not self._latencies:
            return None
        return float(np.percentile(np.asarray(self._latencies, dtype=np.float64), q))

    @property
    def counters(self) -> dict[str, object]:
        """The run's counters (docs/16 §6), JSON-serialisable; the sink adds its own."""
        return {
            "events": self._events,
            "labels": dict(self._labels),
            "no_verdict": dict(sorted(self._reasons.items())),
            "faults": dict(sorted(self._faults.items())),
            "reconnect_attempts": self._reconnect_attempts,
            "annunciator_errors": self._annunciator_errors,
            "sink_errors": self._sink_errors,
            "latency": {
                "p50_ms": self._percentile(50),
                "p95_ms": self._percentile(95),
                "window": len(self._latencies),
                "budget_ms": self._config.latency.budget_ms,
                "breaches": self._breaches,
            },
            # False until a frame is accepted, and for good once one was unverified (FR-13).
            "camera_verified": self._frames_accepted > 0 and not self._unverifiable,
            "final_state": self._describe(),
        }

    def _describe(self) -> str:
        if self._state is State.FAULT:
            return f"FAULT({self._fault_reason})"
        if self._state is State.STOPPED:
            return self._final_state
        return self._state.value

    # --- start and stop ---------------------------------------------------------------------

    def start(self) -> None:
        """Start-up steps 3 and 4 of docs/16 §2 (the inspector is already loaded): the camera
        opens — not found is FAULT(camera), retrying; a value read back different from the one
        set (`CameraSettingsDrift`) or any other failure is raised, for the caller to stop the
        station — then the reference patch on the preview, then READY."""
        self._state = State.STARTING
        try:
            self._source.open()
        except CaptureError as exc:  # CameraLost too: not found, keep retrying
            _log.warning("camera not found at start: %s", exc)
            self._enter_fault("camera")
            return
        self._ready()

    def _ready(self) -> None:
        """Step 4, then READY: out of band → start with LIGHTING; no preview → skipped."""
        self._capture_streak = 0
        try:
            preview = self._source.preview()
            if preview is not None:
                roi = self._config.lighting.reference_patch.roi
                self._lighting = not self._in_band(measure_patch(preview, roi))
        except Exception as exc:  # a preview that cannot be measured is not a lit card
            _log.warning("the reference patch could not be read at start: %s", exc)
            self._lighting = True
        self._fault_reason = None
        self._state = State.READY

    def _enter_fault(self, reason: str) -> None:
        self._state = State.FAULT
        self._fault_reason = reason
        self._faults[reason] = self._faults.get(reason, 0) + 1
        self._attempt = 0
        self._next_attempt = self._clock() + BACKOFF_S[0]
        _log.error("FAULT(%s): the station is not checking", reason)

    def tick(self, now: float | None = None) -> None:
        """Once per main-loop tick: the sink's health is read, and a camera or capture fault
        retries on its backoff."""
        now = self._clock() if now is None else now
        self._read_storage()
        if (
            self._state is State.FAULT
            and self._fault_reason in RETRIED_FAULTS
            and not self._stop_requested
            and now >= self._next_attempt
        ):
            self._reconnect(now)

    def _reconnect(self, now: float) -> None:
        """`close()` then `open()`; success → STARTING → steps 3 and 4 → READY. A read-back
        mismatch stops the station, as at start-up."""
        self._reconnect_attempts += 1
        self._attempt += 1
        try:
            self._source.close()
        except Exception as exc:
            _log.warning("closing the camera before a reconnect failed: %s", exc)
        self._state = State.STARTING
        try:
            self._source.open()
        except CameraSettingsDrift as exc:
            self._refusal = f"camera: {exc}"
            _log.error("read-back mismatch on reconnect, stopping: %s", exc)
            self._state = State.FAULT
            self.stop()
            return
        except Exception as exc:
            wait = BACKOFF_S[self._attempt] if self._attempt < len(BACKOFF_S) else BACKOFF_THEN_S
            self._next_attempt = now + wait
            self._state = State.FAULT
            _log.warning("reconnect attempt %d failed: %s", self._attempt, exc)
            return
        _log.info("camera back after %d attempt(s)", self._attempt)
        self._ready()

    def request_stop(self) -> None:
        """Stop after the current event: the main loop polls no more (F13, SIGINT)."""
        self._stop_requested = True

    def stop(self) -> None:
        """The clean stop: `finish_run(ended_utc, counters)`, then `sink.close()` → STOPPED.
        The state it stopped from is kept as `final_state`. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self._final_state = self._describe()
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
        if self._state is State.STARTING or (
            self._last_finished is not None and event.t0 < self._last_finished
        ):
            # Step 0 (F9): a unit trigger while starting or inspecting is accounted for, not
            # checked. Nothing is captured, and no streak changes.
            return self._finish(event, _Outcome(self._no_verdict("busy"), None))
        if self._state is State.FAULT or self._state is State.STOPPED:
            # Step 1: not checking. Nothing is captured and there is no image.
            reason = self._fault_reason or "busy"
            return self._finish(event, _Outcome(self._no_verdict(reason), None))
        self._state = State.INSPECTING
        self._reached_inspect = False
        self._accepted = False
        try:
            outcome = self._inspect_event(event)
        except Exception as exc:
            # A collaborator broke its contract in a way no step above names. C5 still holds:
            # one NO_VERDICT record, without the frame, for nothing about it can be trusted.
            _log.error("seq %d: unexpected %s: %s", event.seq, type(exc).__name__, exc)
            outcome = self._unexpected()
        if self._state is State.INSPECTING:
            self._state = State.READY
        return self._finish(event, outcome)

    def _inspect_event(self, event: TriggerEvent) -> _Outcome:
        """Steps 2–7 for one event in READY."""
        frame = self._capture(event)
        if not isinstance(frame, Frame):
            return frame
        # Step 3: the frame is accepted — once its read-back can be read at all.
        verified = frame.camera.verified is True
        self._capture_streak = 0
        self._frames_accepted += 1
        self._accepted = True
        if verified:
            self._drift = False
        else:
            self._unverifiable = True
        # Step 4: the reference patch (A1, #62).
        if not self._in_band(frame.reference_patch):
            self._lighting = True
            return self._no_verdict_with_frame("lighting", frame, 0.0)
        self._lighting = False
        # Step 5: inspect.
        self._reached_inspect = True
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
            return self._inference_failed(frame, inspect_ms)
        return self._encode(verdict, frame, inspect_ms)

    def _inference_failed(self, frame: Frame, inspect_ms: float) -> _Outcome:
        """NO_VERDICT(`inference`) with the image; the third in a row is FAULT(inference)."""
        self._inference_streak += 1
        if self._inference_streak >= IN_A_ROW:
            self._enter_fault("inference")
        return self._no_verdict_with_frame("inference", frame, inspect_ms)

    def _unexpected(self) -> _Outcome:
        """An exception no step names: NO_VERDICT(`capture`) before step 3 accepted the frame,
        `inference` after; it counts toward its streak like any other of its kind."""
        if not self._accepted:
            return self._capture_failed()
        self._inference_streak += 1
        if self._inference_streak >= IN_A_ROW:
            self._enter_fault("inference")
        return _Outcome(self._no_verdict("inference"), None)

    def _capture_failed(self) -> _Outcome:
        """NO_VERDICT(`capture`) without an image; the third in a row is FAULT(capture)."""
        self._capture_streak += 1
        if self._capture_streak >= IN_A_ROW:
            self._enter_fault("capture")
        return _Outcome(self._no_verdict("capture"), None)

    def _capture(self, event: TriggerEvent) -> Frame | _Outcome:
        """Step 2: a frame, or the NO_VERDICT its failure means. None of these has an image."""
        try:
            frame = self._source.capture(event)
        except CameraLost as exc:
            _log.warning("seq %d: camera lost: %s", event.seq, exc)
            self._enter_fault("camera")
            return _Outcome(self._no_verdict("camera"), None)
        except CaptureError as exc:
            _log.warning("seq %d: capture failed: %s", event.seq, exc)
            return self._capture_failed()
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
            return self._capture_failed()
        if not isinstance(frame, Frame) or frame.event != event:
            _log.warning("seq %d: the source returned a frame for another event", event.seq)
            return self._capture_failed()
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
        if verdict.reason is not None:
            raise _Refused(f"a {verdict.label} verdict carries a reason ({verdict.reason!r})")
        # Read `checks` only as a tuple of CheckResult: a one-shot iterable would be used up
        # by derive_label and then read as empty below — a pass with a failed check behind it.
        if not isinstance(verdict.checks, tuple) or not all(
            isinstance(check, CheckResult) for check in verdict.checks
        ):
            raise _Refused(
                f"checks is a {type(verdict.checks).__name__}, not a tuple of CheckResult"
            )
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
            return self._inference_failed(frame, inspect_ms)
        self._inference_streak = 0  # a decided verdict, accepted and encoded
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
        if self._reached_inspect:  # an inspection: an event that reached step 5 (F8)
            self._reached_inspect = False
            self._latencies.append(total_ms)
            if total_ms > self._config.latency.budget_ms:
                self._breaches += 1
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
        self._read_storage()
        self._last_finished = self._clock()
        return record

    def _read_storage(self) -> None:
        """STORAGE while the sink's `health()` is not ok; a health() that raises is not ok."""
        try:
            self._storage_bad = self._sink.health().ok is not True
        except Exception:
            _log.exception("the sink's health() raised")
            self._storage_bad = True
