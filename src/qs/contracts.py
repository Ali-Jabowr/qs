from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from qs.config import StationConfig

CHECK_GEOMETRY = "geometry"
CHECK_APPEARANCE = "appearance"
CHECK_EMPTY_CUP = "empty_cup"
REASONS = frozenset(
    {
        "capture", "camera", "camera_settings", "lighting", "placement", "inference", "busy",
        "untracked",
    }
)


class Label(StrEnum):
    PASS = "pass"
    DEFECT = "defect"
    NO_VERDICT = "no_verdict"


@dataclass(frozen=True, slots=True)
class TriggerEvent:
    """`never_in_zone`: the cup trigger fired for a cup that left the view without ever being
    wholly in the zone; the unit is NO_VERDICT(`placement`) whatever the captured frame shows.
    `untracked`: the cup trigger saw warmth pass through its fixed window with no tracked cup
    there (ADR-0029 §7); the unit is NO_VERDICT(`untracked`) and the inspector is not asked."""

    seq: int
    t0: float
    wall: datetime
    source_id: str
    never_in_zone: bool = False
    untracked: bool = False


@dataclass(frozen=True, slots=True)
class CameraReadback:
    exposure_us: int | None
    gain: int | None
    wb_kelvin: int | None
    focus: int | None
    verified: bool


@dataclass(frozen=True, slots=True)
class Frame:
    event: TriggerEvent
    pixels: np.ndarray
    grabbed_at: float
    camera: CameraReadback
    reference_patch: float | None
    capture_ms: float


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    value: float | None
    limit: float | None
    measures: tuple[tuple[str, float], ...] = ()
    failed_on: str | None = None


@dataclass(frozen=True, slots=True)
class Verdict:
    label: Label
    checks: tuple[CheckResult, ...]
    heatmap: np.ndarray | None
    window_box: tuple[int, int, int, int] | None
    unit_box: tuple[int, int, int, int] | None
    model_id: str
    threshold_id: str
    config_hash: str
    inspect_ms: float
    reason: str | None = None
    evidence_index: int = 0  # which frame of a burst is the evidence; 0 for a single frame

    def __post_init__(self) -> None:
        if self.label is Label.NO_VERDICT and self.reason is None:
            raise ValueError("a NO_VERDICT must state its reason")
        if self.label is not Label.NO_VERDICT and not self.checks:
            raise ValueError(f"{self.label.value} with no checks is not a verdict (C8)")


@dataclass(frozen=True, slots=True)
class Health:
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class InspectionRecord:
    """What the Sink persists: one `inspection` row and its `inspection_check` rows. The image
    and heatmap arrive ALREADY ENCODED — the sink queue never holds raw pixels (ADR-0012)."""

    run_id: str
    event: TriggerEvent
    verdict: Verdict
    camera: CameraReadback | None
    reference_patch: float | None
    image_jpeg: bytes | None
    heatmap_png: bytes | None
    capture_ms: float | None
    total_ms: float


class TriggerSource(Protocol):
    """Polled once per main-loop tick; never blocks. At most one event per unit. A unit the
    station cannot inspect is never skipped silently: it still gets its seq and a NO_VERDICT
    record (C5)."""

    def poll(self) -> TriggerEvent | None: ...

    @property
    def exhausted(self) -> bool: ...

    def close(self) -> None: ...


class FrameSource(Protocol):
    """Turns an event into pixels. capture() returns the first frame whose timestamp is later
    than event.t0 plus one exposure time — never a buffered frame from before the trigger. It
    raises CaptureError on timeout or on a corrupt frame (wrong shape or dtype, or flat: every
    pixel within 1 grey level), and CameraSettingsDrift when a readable setting no longer
    matches the config. capture_burst(event, n) returns the n consecutive frames from that
    first one on, each later than event.t0 plus one exposure time, and raises like capture();
    capture_burst(event, 1) is capture(event). A source that cannot burst raises ValueError
    for n > 1. preview() feeds the live view only; a preview frame is never inspected."""

    def open(self) -> None: ...

    def capture(self, event: TriggerEvent) -> Frame: ...

    def capture_burst(self, event: TriggerEvent, count: int) -> tuple[Frame, ...]: ...

    def preview(self) -> np.ndarray | None: ...

    def health(self) -> Health: ...

    def close(self) -> None: ...


class Inspector(Protocol):
    """Decides. Must be deterministic given (frame, model, threshold): the same frame
    inspected twice returns the same score, or the evaluation harness is measuring noise.

    Must not do I/O. Must not mutate the frame. Must not read the clock for anything
    other than its own timing. Owns no state that changes between frames.

    inspect_burst(frames) judges one unit from the burst the source returned for its event;
    a burst of 1 is inspect(frame). An inspector that cannot judge a burst raises ValueError
    for more than one frame."""

    def load(self, config: StationConfig) -> None: ...

    def inspect(self, frame: Frame) -> Verdict: ...

    def inspect_burst(self, frames: Sequence[Frame]) -> Verdict: ...

    @property
    def model_id(self) -> str: ...


class Annunciator(Protocol):
    """Tells the operator. Must render NO_VERDICT visibly differently from PASS —
    see invariant I6. Must return within its slice of the latency budget."""

    def show(self, frame: Frame, verdict: Verdict) -> None: ...

    def close(self) -> None: ...


class Sink(Protocol):
    """Writes the record. MUST NOT raise into the Pipeline and MUST NOT block it:
    implementations queue and drain on a worker thread. A failing Sink degrades the
    station (see 16-failure-model.md); it never stalls the line and never loses a
    verdict the operator already acted on."""

    def write(self, record: InspectionRecord) -> None: ...

    def pending(self) -> int: ...

    def close(self) -> None: ...


def one_frame_burst(count: int) -> None:
    """The guard of a source that captures one frame per event: `ValueError` unless `count` is 1."""
    if count != 1:
        raise ValueError(f"this source captures one frame per event, not a burst of {count}")


def only_frame(frames: Sequence[Frame]) -> Frame:
    """The one frame of a burst of 1; `ValueError` for any other length, for an inspector that
    judges a single frame."""
    if len(frames) != 1:
        raise ValueError(f"this inspector judges one frame, not a burst of {len(frames)}")
    return frames[0]


def derive_label(checks: tuple[CheckResult, ...], configured: frozenset[str]) -> Label:
    """Derive a label from completed checks alone (C8). DEFECT if any check's `passed`
    is anything other than exactly True — a non-bool fails closed, never open; PASS only
    if every configured check ran and passed; never NO_VERDICT — a completed inspection
    with a configured check missing is a programming error, not a verdict."""
    if not configured:
        raise ValueError("no configured checks: a verdict needs at least one")
    names = [check.name for check in checks]
    if len(names) != len(set(names)):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise ValueError("two checks share a name: " + ", ".join(duplicated))
    unconfigured = sorted(set(names) - configured)
    if unconfigured:
        raise ValueError("check(s) not configured: " + ", ".join(unconfigured))
    if any(check.passed is not True for check in checks):
        return Label.DEFECT
    missing = sorted(configured - set(names))
    if missing:
        raise ValueError("configured check(s) did not run: " + ", ".join(missing))
    return Label.PASS
