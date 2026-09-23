"""The Pipeline's decided path and its C8/provenance guard (#69): a verdict whose label
disagrees with its checks, a non-finite value or another config's verdict never becomes a
pass. In memory: a scripted source, the stub inspector, a recording sink, a fake clock."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from qs.annunciator import NullAnnunciator
from qs.config import LoadedConfig
from qs.contracts import Frame, Label, Verdict
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline, State
from station_fixtures import FakeClock, RecordingSink, ScriptedSource, event, loaded_config


class Rehash:
    """An inspector whose verdicts name another config (or threshold, or model)."""

    def __init__(self, inner: StubInspector, **changes: str) -> None:
        self.inner = inner
        self.changes = changes

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def load(self, config: object) -> None:
        self.inner.load(config)  # type: ignore[arg-type]

    def inspect(self, frame: Frame) -> Verdict:
        return dataclasses.replace(self.inner.inspect(frame), **self.changes)


def _pipeline(
    loaded: LoadedConfig, inspector: object, sink: RecordingSink | None = None
) -> tuple[Pipeline, RecordingSink, NullAnnunciator]:
    clock = FakeClock(start=0.0)  # the events below are stamped t0 = 0
    sink = sink or RecordingSink()
    shown = NullAnnunciator()
    inspector.load(loaded.config)  # type: ignore[attr-defined]
    pipeline = Pipeline(
        config=loaded.config,
        config_hash=loaded.config_hash,
        run_id="R1",
        source=ScriptedSource(clock),
        inspector=inspector,  # type: ignore[arg-type]
        annunciator=shown,
        sink=sink,
        clock=clock,
    )
    pipeline.start()
    return pipeline, sink, shown


def test_a_pass_and_a_defect_are_recorded_with_their_images(tmp_path: Path) -> None:
    loaded = loaded_config(tmp_path)
    pipeline, sink, shown = _pipeline(
        loaded, StubInspector(loaded.config_hash, {2: "defect:appearance"})
    )
    first = pipeline.handle(event(1))
    second = pipeline.handle(event(2))
    assert [r.verdict.label for r in sink.records] == [Label.PASS, Label.DEFECT]
    assert sink.records == [first, second]
    assert first.image_jpeg is not None and first.heatmap_png is None
    assert second.image_jpeg is not None and second.heatmap_png is not None
    assert first.verdict.config_hash == loaded.config_hash
    assert pipeline.state is State.READY
    assert len(shown.calls) == 2


@pytest.mark.parametrize("outcome", ["mislabel", "nonfinite"])
def test_a_verdict_that_hides_a_failure_is_never_a_pass(tmp_path: Path, outcome: str) -> None:
    loaded = loaded_config(tmp_path)
    pipeline, sink, shown = _pipeline(loaded, StubInspector(loaded.config_hash, {1: outcome}))
    record = pipeline.handle(event(1))
    assert record.verdict.label is not Label.PASS
    assert shown.calls[0][1].label is not Label.PASS


@pytest.mark.parametrize(
    "changes",
    [{"config_hash": "f" * 64}, {"threshold_id": "cal-other"}, {"model_id": "other"}],
)
def test_a_verdict_with_other_provenance_is_not_accepted(
    tmp_path: Path, changes: dict[str, str]
) -> None:
    loaded = loaded_config(tmp_path)
    pipeline, sink, _ = _pipeline(loaded, Rehash(StubInspector(loaded.config_hash), **changes))
    record = pipeline.handle(event(1))
    assert record.verdict.label is not Label.PASS
    assert record.verdict.config_hash == loaded.config_hash
