"""Scoring through the station's own path (docs/14 §4, docs/15 §7 step 3; ticket #81).

`score_frames` runs an explicit file list through P04's `Pipeline` in-process: the replay
pair (#69), the given loaded `Inspector`, `NullAnnunciator`, and an in-memory sink that keeps
every `InspectionRecord`. No evidence database is opened, nothing is written. This is the
only scoring path of `qs calibrate` and `qs evaluate`: a calibration score, an evaluation
score and a `qs station --replay` verdict on the same frame are the same number (#74).

A `FrameResult` carries what the evaluation reads: the file, its parsed name, the label, the
reason, the geometry `CheckResult`, the appearance score (the check's `value`), the model's
own `pred_score`, and the two timings. Nothing here localizes, scores, labels or maps a
failure itself — the Pipeline and the Inspector do.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qs.annunciator import NullAnnunciator
from qs.config import StationConfig
from qs.contracts import (
    CHECK_APPEARANCE,
    CHECK_GEOMETRY,
    CheckResult,
    Health,
    InspectionRecord,
    Inspector,
    Label,
)
from qs.names import FrameName, parse_name
from qs.pipeline import Pipeline, check_config
from qs.sources.replay import replay_pair

_CANDIDATE = "eval"


@dataclass(frozen=True, slots=True)
class FrameResult:
    """One file's outcome through the Pipeline: the record's decision fields, no images."""

    file: Path
    name: FrameName
    label: Label
    reason: str | None
    geometry: CheckResult | None
    score: float | None
    pred_score: float | None
    inspect_ms: float
    total_ms: float


class MemorySink:
    """A `StationSink` in memory: every record kept, in order; nothing is written anywhere."""

    def __init__(self) -> None:
        self.records: list[InspectionRecord] = []

    def write(self, record: InspectionRecord) -> None:
        self.records.append(record)

    def pending(self) -> int:
        return 0

    def close(self) -> None:
        return None

    def health(self) -> Health:
        return Health(True, "in memory")

    def counters(self) -> dict[str, int]:
        return {"written": len(self.records), "held": 0, "images_shed": 0,
                "write_failures": 0, "rejected": 0, "lost": 0, "pending": 0}

    def finish_run(self, ended_utc: datetime, counters: Mapping[str, object]) -> None:
        return None


def utc_now() -> datetime:
    return datetime.now(UTC)


def _of(record: InspectionRecord, files: Sequence[Path]) -> FrameResult:
    """The record's `FrameResult`: file `seq`, its parsed name, and the checks' numbers."""
    verdict = record.verdict
    geometry = next((c for c in verdict.checks if c.name == CHECK_GEOMETRY), None)
    appearance = next((c for c in verdict.checks if c.name == CHECK_APPEARANCE), None)
    pred = dict(appearance.measures).get("pred_score") if appearance else None
    return FrameResult(
        file=files[record.event.seq - 1],
        name=parse_name(files[record.event.seq - 1].name),
        label=verdict.label,
        reason=verdict.reason,
        geometry=geometry,
        score=None if appearance is None else appearance.value,
        pred_score=pred,
        inspect_ms=verdict.inspect_ms,
        total_ms=record.total_ms,
    )


def score_frames(
    config: StationConfig,
    config_hash: str,
    files: Sequence[Path],
    *,
    inspector: Inspector,
    name: str = _CANDIDATE,
) -> tuple[FrameResult, ...]:
    """Every file through the station's Pipeline, in order; the results, no images kept.

    `config` and `config_hash` must be the ones `inspector` was loaded with, so the guard
    accepts every verdict. Any frame the Pipeline refuses to inspect comes back as its
    `NO_VERDICT` result — deciding what that means is the caller's.
    """
    check_config(config)
    trigger, source = replay_pair(config, files, name, clock=time.perf_counter, wall=utc_now)
    sink = MemorySink()
    pipeline = Pipeline(
        config=config,
        config_hash=config_hash,
        run_id=name,
        source=source,
        inspector=inspector,
        annunciator=NullAnnunciator(),
        sink=sink,
    )
    pipeline.start()
    while not trigger.exhausted and not pipeline.stopped:
        pipeline.tick(pipeline.reconnect_due)
        if pipeline.stopped:
            break
        event = trigger.poll()
        if event is not None:
            pipeline.handle(event)
    pipeline.stop()
    if len(sink.records) != len(files):
        raise RuntimeError(
            f"the Pipeline wrote {len(sink.records)} record(s) for {len(files)} frame(s) — "
            "exactly one record per event is the contract (C5)"
        )
    return tuple(_of(record, files) for record in sink.records)
