"""The station's run report and its NFR-01 line (P09 #91). Pure: the run's facts and the
per-event timings in, the markdown text out. No I/O, no clock.

The report holds numbers and IDs only: no image, no path to one, and no name (NFR-07,
CLAUDE.md §6). It is written at the stop of every run with a screen; headless replay writes
none.

**Inspections.** An inspection is an event that reached P04's step 5: its record is PASS or
DEFECT, or NO_VERDICT for `placement` or `inference`. The other NO_VERDICT reasons
(`capture`, `lighting`, `camera`, `camera_settings`, `busy`) never reached the model.

**NFR-01** (≤ 1.0 s trigger to verdict on screen at p95, over ≥ 200 events, on the laptop's
GPU) is judged only for a live run with the `appearance` inspector on `torch-cuda`, with at
least 200 inspections, that never entered FAULT. Any other run says "not measured", with the
first reason that applies and, when there was an inspection, an indicative p95. A replay is
never NFR-01: it reads a file where the live station waits for a fresh frame (#85, A1). The
limit is NFR-01's 1000 ms, not `latency.budget_ms`. Percentiles are `numpy.percentile` with
its default (linear) method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from qs.contracts import InspectionRecord, Label

NFR01_LIMIT_MS = 1000.0
NFR01_MIN_INSPECTIONS = 200
NFR01_BACKEND = "torch-cuda"
INSPECTED_REASONS = frozenset({"placement", "inference"})
NONE = "—"


@dataclass(frozen=True, slots=True)
class EventTiming:
    """What the report keeps of one record: its label, reason and three timings."""

    label: str
    reason: str | None
    total_ms: float
    inspect_ms: float | None
    capture_ms: float | None

    @property
    def inspection(self) -> bool:
        """Whether the event reached the model (P04's step 5)."""
        if self.label in (Label.PASS.value, Label.DEFECT.value):
            return True
        return self.label == Label.NO_VERDICT.value and self.reason in INSPECTED_REASONS


def timing_of(record: InspectionRecord) -> EventTiming:
    """The report's view of one record the Pipeline returned."""
    return EventTiming(
        label=record.verdict.label.value,
        reason=record.verdict.reason,
        total_ms=float(record.total_ms),
        inspect_ms=float(record.verdict.inspect_ms),
        capture_ms=None if record.capture_ms is None else float(record.capture_ms),
    )


@dataclass(frozen=True, slots=True)
class RunFacts:
    """Everything the report says about the run beside the timings.

    `counters` are the Pipeline's (docs/16 §6), `sink` the sink's own, `keys_discarded` the
    trigger's. `mode` is `live, <device>` or `replay of <DIR name>`."""

    run_id: str
    live: bool
    mode: str
    invocation: str
    config_path: str
    config_hash: str
    model_id: str
    threshold_id: str
    inspector_type: str
    backend: str
    budget_ms: int
    counters: Mapping[str, object]
    sink: Mapping[str, int]
    keys_discarded: int

    @property
    def entered_fault(self) -> bool:
        faults = self.counters.get("faults", {})
        return bool(faults)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def nfr01_line(facts: RunFacts, timings: Sequence[EventTiming]) -> str:
    """`NFR-01: PASS …`, `NFR-01: FAIL …` or `NFR-01: not measured — <reason> …`."""
    totals = [t.total_ms for t in timings if t.inspection]
    n = len(totals)
    p95 = percentile(totals, 95) if totals else None
    reason: str | None = None
    if not facts.live:
        reason = "replay has no capture step"
    elif facts.inspector_type != "appearance":
        reason = f"{facts.inspector_type} inspector"
    elif facts.backend != NFR01_BACKEND:
        reason = f"backend {facts.backend}; NFR-01 is on the GPU"
    elif n < NFR01_MIN_INSPECTIONS:
        reason = f"{n} inspections, {NFR01_MIN_INSPECTIONS} needed"
    elif facts.entered_fault:
        reason = "the run entered FAULT"
    if reason is None:
        assert p95 is not None
        if p95 <= NFR01_LIMIT_MS:
            return f"NFR-01: PASS — p95 {p95:.0f} ms ≤ 1000 ms over {n} inspections"
        return f"NFR-01: FAIL — p95 {p95:.0f} ms > 1000 ms over {n} inspections"
    indicative = "" if p95 is None else f" (indicative p95 {p95:.0f} ms)"
    return f"NFR-01: not measured — {reason}{indicative}"


def _ms(value: float) -> str:
    return f"{value:.1f}"


def _row(name: str, values: Sequence[float], quantiles: Sequence[float], full: bool) -> str:
    cells = [name, str(len(values))]
    for q in quantiles:
        cells.append(_ms(percentile(values, q)) if values else NONE)
    if full:
        cells.append(_ms(max(values)) if values else NONE)
    return "| " + " | ".join(cells) + " |"


def _counts(mapping: Mapping[str, int]) -> str:
    return ", ".join(f"{key} {value}" for key, value in mapping.items()) or NONE


def render_report(facts: RunFacts, timings: Sequence[EventTiming]) -> str:
    """The run report as markdown, numbers and IDs only."""
    counters = facts.counters
    labels: Mapping[str, int] = counters["labels"]  # type: ignore[assignment]
    reasons: Mapping[str, int] = counters["no_verdict"]  # type: ignore[assignment]
    faults: Mapping[str, int] = counters["faults"]  # type: ignore[assignment]
    inspections = [t for t in timings if t.inspection]
    all_totals = [t.total_ms for t in timings]
    totals = [t.total_ms for t in inspections]
    inspects = [t.inspect_ms for t in inspections if t.inspect_ms is not None]
    captures = [t.capture_ms for t in timings if t.capture_ms is not None]
    breaches = sum(1 for total in all_totals if total > facts.budget_ms)
    sink = facts.sink
    lines = [
        f"# Station run {facts.run_id}",
        "",
        f"- Mode: {facts.mode}",
        f"- Invocation: `{facts.invocation}`",
        f"- Config: `{facts.config_path}` · config_hash `{facts.config_hash}`",
        f"- Model: `{facts.model_id}` · threshold `{facts.threshold_id}`",
        f"- Inspector: {facts.inspector_type} · backend {facts.backend}",
        f"- Camera verified: {'yes' if counters.get('camera_verified') else 'no'}",
        "",
        "## Counts",
        "",
        f"- Events: {counters['events']}",
        f"- Labels: pass {labels['pass']} · defect {labels['defect']}"
        f" · no_verdict {labels['no_verdict']}",
        f"- NO_VERDICT by reason: {_counts(reasons)}",
        f"- FAULT entries by reason: {_counts(faults)}",
        f"- Reconnect attempts: {counters['reconnect_attempts']}",
        f"- Keys discarded: {facts.keys_discarded}",
        f"- Evidence: written {sink['written']} · held {sink['held']}"
        f" · images shed {sink['images_shed']} · lost {sink['lost']}",
        "",
        "## Latency, ms",
        "",
        "| | n | p50 | p95 | p99 | max |",
        "|---|---|---|---|---|---|",
        _row("total_ms, all events", all_totals, (50, 95, 99), True),
        _row("total_ms, inspections", totals, (50, 95, 99), True),
        "",
        "| | n | p50 | p95 |",
        "|---|---|---|---|",
        _row("inspect_ms, inspections", inspects, (50, 95), False),
        _row("capture_ms, events with a frame", captures, (50, 95), False),
        "",
        f"Events over the budget ({facts.budget_ms} ms): {breaches}",
        "",
        "## NFR-01",
        "",
        nfr01_line(facts, timings),
        "",
    ]
    return "\n".join(lines)
