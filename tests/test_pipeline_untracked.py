"""An untracked event takes one frame and never reaches the inspector (#134, ADR-0029 §7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from qs.annunciator import NullAnnunciator
from qs.config import load_config
from qs.contracts import Frame, Label, TriggerEvent, Verdict
from qs.pipeline import Pipeline
from station_fixtures import FakeClock, RecordingSink, ScriptedSource, event
from warmth_fixtures import warmth_config


class CountingSource(ScriptedSource):
    """Counts how many frames each kind of capture was asked for."""

    bursts: list[int] = []

    def capture_burst(self, event: TriggerEvent, count: int) -> tuple[Frame, ...]:
        self.bursts.append(count)
        return tuple(self.capture(event) for _ in range(count))


class NeverAsked:
    """An inspector that fails the test if it is asked for a verdict."""

    model_id = "none"
    asked = 0

    def load(self, config: object) -> None:
        return None

    def inspect(self, frame: Frame) -> Verdict:
        self.asked += 1
        raise AssertionError("the inspector must not be asked")

    def inspect_burst(self, frames: Sequence[Frame]) -> Verdict:
        self.asked += 1
        raise AssertionError("the inspector must not be asked")


def _pipeline(root: Path) -> tuple[Pipeline, CountingSource, NeverAsked, RecordingSink]:
    loaded = load_config(warmth_config(root))
    clock = FakeClock(start=0.0)
    source = CountingSource(clock)
    source.bursts = []
    inspector = NeverAsked()
    sink = RecordingSink()
    pipeline = Pipeline(
        config=loaded.config, config_hash=loaded.config_hash, run_id="R1", source=source,
        inspector=inspector, annunciator=NullAnnunciator(), sink=sink, clock=clock,
    )
    pipeline.start()
    return pipeline, source, inspector, sink


def test_an_untracked_event_captures_one_frame_and_asks_no_inspector(tmp_path: Path) -> None:
    pipeline, source, inspector, _ = _pipeline(tmp_path)

    record = pipeline.handle(replace(event(1), untracked=True))

    assert (record.verdict.label, record.verdict.reason) == (Label.NO_VERDICT, "untracked")
    assert (source.captured, source.bursts, inspector.asked) == ([1], [], 0)


def test_a_cup_event_on_the_same_station_still_captures_a_burst(tmp_path: Path) -> None:
    pipeline, source, _, _ = _pipeline(tmp_path)

    pipeline.handle(event(1))

    assert source.bursts == [3]
