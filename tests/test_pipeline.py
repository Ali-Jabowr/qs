"""The Pipeline's decided path and its C8/provenance guard (#69): a verdict whose label
disagrees with its checks, a non-finite value or another config's verdict never becomes a
pass. In memory: a scripted source, the stub inspector, a recording sink, a fake clock."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qs.annunciator import NullAnnunciator
from qs.config import LoadedConfig
from qs.contracts import CHECK_APPEARANCE, Frame, Label, Verdict
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline, State
from station_fixtures import (
    PATCH_GREY,
    FakeClock,
    RecordingSink,
    ScriptedSource,
    event,
    loaded_config,
)


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
    loaded: LoadedConfig,
    inspector: object,
    sink: RecordingSink | None = None,
    captures: dict[int, object] | None = None,
) -> tuple[Pipeline, RecordingSink, NullAnnunciator]:
    clock = FakeClock(start=0.0)  # the events below are stamped t0 = 0
    sink = sink or RecordingSink()
    shown = NullAnnunciator()
    inspector.load(loaded.config)  # type: ignore[attr-defined]
    pipeline = Pipeline(
        config=loaded.config,
        config_hash=loaded.config_hash,
        run_id="R1",
        source=ScriptedSource(clock, captures=captures or {}),
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


# --- REVIEW-P04 r1 ----------------------------------------------------------------------------


class Reshape:
    """An inspector that passes the stub's verdict through `change` before returning it."""

    def __init__(self, inner: StubInspector, change: Callable[[Verdict], Verdict]) -> None:
        self.inner = inner
        self.change = change

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def load(self, config: object) -> None:
        self.inner.load(config)  # type: ignore[arg-type]

    def inspect(self, frame: Frame) -> Verdict:
        return self.change(self.inner.inspect(frame))


def _one_shot(verdict: Verdict) -> Verdict:
    """A PASS whose checks are a generator: geometry passes, appearance FAILS (m2)."""
    geometry, appearance = verdict.checks
    failed = dataclasses.replace(appearance, passed=False, failed_on="score>threshold")
    checks: Any = (c for c in (geometry, failed))
    return dataclasses.replace(verdict, label=Label.PASS, checks=checks)


def _as_list(verdict: Verdict) -> Verdict:
    checks: Any = list(verdict.checks)
    return dataclasses.replace(verdict, checks=checks)


def _a_name_for_a_check(verdict: Verdict) -> Verdict:
    checks: Any = (verdict.checks[0], CHECK_APPEARANCE)
    return dataclasses.replace(verdict, checks=checks)


@pytest.mark.parametrize(
    "change",
    [_one_shot, _as_list, _a_name_for_a_check],
    ids=["generator-holding-a-failed-check", "list", "not-a-CheckResult"],
)
def test_checks_that_are_not_a_tuple_of_check_results_are_refused(
    tmp_path: Path, change: Callable[[Verdict], Verdict]
) -> None:
    """m2: the C8 guard reads `checks` only as a tuple of `CheckResult`; anything else is
    NO_VERDICT(`inference`), never a pass."""
    loaded = loaded_config(tmp_path)
    pipeline, sink, shown = _pipeline(loaded, Reshape(StubInspector(loaded.config_hash), change))
    record = pipeline.handle(event(1))
    assert record.verdict.label is Label.NO_VERDICT
    assert record.verdict.reason == "inference"
    assert shown.calls[0][1].label is Label.NO_VERDICT
    assert sink.records == [record]


@pytest.mark.parametrize("script", [{}, {1: "defect:appearance"}], ids=["pass", "defect"])
def test_a_decided_verdict_that_carries_a_reason_is_refused(
    tmp_path: Path, script: dict[int, str]
) -> None:
    """n1: a PASS or DEFECT with a `reason` is not a decided verdict: NO_VERDICT(`inference`)."""
    loaded = loaded_config(tmp_path)
    inspector = Reshape(
        StubInspector(loaded.config_hash, script),
        lambda v: dataclasses.replace(v, reason="lighting"),
    )
    pipeline, _, _ = _pipeline(loaded, inspector)
    record = pipeline.handle(event(1))
    assert record.verdict.label is Label.NO_VERDICT
    assert record.verdict.reason == "inference"
    assert pipeline.counters["no_verdict"] == {"inference": 1}


def test_a_frame_that_breaks_its_contract_is_one_no_verdict_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """m3: a source that returns a Frame with `camera=None` gives one NO_VERDICT(`capture`)
    record per trigger, the station leaves INSPECTING, and nothing escapes `handle()`. The next
    good frame is inspected; three in a row are FAULT(capture), as for any capture error."""
    loaded = loaded_config(tmp_path)
    patch = PATCH_GREY / 255
    captures: dict[int, object] = {
        1: (patch, None), 3: (patch, None), 4: (patch, "verified"), 5: (patch, None)
    }
    pipeline, sink, shown = _pipeline(loaded, StubInspector(loaded.config_hash), captures=captures)
    records = [pipeline.handle(event(seq)) for seq in (1, 2)]
    assert records[0].verdict.label is Label.NO_VERDICT
    assert records[0].verdict.reason == "capture" and records[0].image_jpeg is None
    assert records[1].verdict.label is Label.PASS
    assert pipeline.state is State.READY
    records += [pipeline.handle(event(seq)) for seq in (3, 4, 5)]
    assert [r.verdict.reason for r in records[2:]] == ["capture"] * 3
    assert pipeline.state is State.FAULT and pipeline.fault_reason == "capture"
    assert sink.records == records and len(shown.calls) == 5
    assert "Traceback" not in caplog.text
