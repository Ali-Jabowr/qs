"""What the station knows about itself beyond each verdict (#72): the `STORAGE` and `LATENCY`
warnings, the `busy` rule for unit triggers (F9), and the docs/16 §6 counters in the run record
and the summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig
from qs.contracts import Frame, Label, Verdict
from qs.errors import CaptureError
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline, State
from station_fixtures import (
    UNVERIFIED,
    FakeClock,
    RecordingSink,
    ScriptedSource,
    event,
    loaded_config,
    rows,
    station_args,
    textured,
    write_config,
    write_frames,
    write_png,
)


class Timed:
    """Wraps the stub: `inspect` takes `durations[seq]` seconds (default 0.1) on the fake clock."""

    def __init__(self, inner: StubInspector, clock: FakeClock, durations: dict[int, float]):
        self.inner, self.clock, self.durations = inner, clock, durations

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def load(self, config: object) -> None:
        self.inner.load(config)  # type: ignore[arg-type]

    def inspect(self, frame: Frame) -> Verdict:
        self.clock.advance(self.durations.get(frame.event.seq, 0.1))
        return self.inner.inspect(frame)


class Rig:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        sink: RecordingSink | None = None,
        durations: dict[int, float] | None = None,
        captures: dict[int, object] | None = None,
        start: bool = True,
    ) -> None:
        self.clock = FakeClock()
        self.source = ScriptedSource(self.clock, captures=captures or {})
        self.inspector = Timed(StubInspector(loaded.config_hash), self.clock, durations or {})
        self.inspector.load(loaded.config)
        self.sink = sink or RecordingSink()
        self.pipeline = Pipeline(
            config=loaded.config,
            config_hash=loaded.config_hash,
            run_id="R1",
            source=self.source,
            inspector=self.inspector,  # type: ignore[arg-type]
            annunciator=NullAnnunciator(),
            sink=self.sink,
            clock=self.clock,
        )
        self.seq = 0
        if start:
            self.pipeline.start()

    def trigger(self, t0: float | None = None) -> object:
        self.seq += 1
        return self.pipeline.handle(event(self.seq, self.clock() if t0 is None else t0))


@pytest.fixture
def loaded(tmp_path: Path) -> LoadedConfig:
    return loaded_config(tmp_path)


def test_storage_is_raised_while_the_sink_is_unhealthy(loaded: LoadedConfig) -> None:
    sink = RecordingSink(healthy=lambda written: not 10 <= written <= 20)
    rig = Rig(loaded, sink=sink)
    seen = []
    for _ in range(25):
        rig.trigger()
        seen.append("STORAGE" in rig.pipeline.warnings)
    assert seen == [False] * 9 + [True] * 11 + [False] * 5
    assert all(r.verdict.label is Label.PASS for r in sink.records)
    flaky = RecordingSink(healthy=lambda written: False)
    ticked = Rig(loaded, sink=flaky)
    assert "STORAGE" not in ticked.pipeline.warnings
    ticked.pipeline.tick()
    assert "STORAGE" in ticked.pipeline.warnings


def test_latency_is_raised_above_ten_percent_of_the_last_fifty(loaded: LoadedConfig) -> None:
    durations = {seq: 0.1 for seq in range(1, 46)}
    durations.update({seq: 1.5 for seq in range(46, 52)})
    rig = Rig(loaded, durations=durations)  # budget_ms 1000 in the example config
    for _ in range(50):
        rig.trigger()
    assert "LATENCY" not in rig.pipeline.warnings  # exactly 10 %
    rig.trigger()
    assert "LATENCY" in rig.pipeline.warnings  # 6 of the last 50
    cleared_after = None
    for n in range(1, 60):
        rig.trigger()
        if "LATENCY" not in rig.pipeline.warnings:
            cleared_after = n
            break
    assert cleared_after == 45  # the 45th fast one pushes the first slow one out
    assert rig.pipeline.counters["latency"]["breaches"] == 6  # type: ignore[index]
    breached = [r for r in rig.sink.records if r.total_ms > 1000]
    assert len(breached) == 6 and all(r.verdict.label is Label.PASS for r in breached)


def test_latency_percentiles_by_hand(loaded: LoadedConfig) -> None:
    few = Rig(loaded, durations={1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4})
    for _ in range(4):
        few.trigger()
    latency = few.pipeline.counters["latency"]
    assert latency["p50_ms"] == pytest.approx(250.0)  # type: ignore[index]
    assert latency["p95_ms"] == pytest.approx(385.0)  # type: ignore[index]
    many = Rig(loaded, durations={seq: seq / 100 for seq in range(1, 61)})
    for _ in range(60):
        many.trigger()
    latency = many.pipeline.counters["latency"]  # the last 50: 110 … 600 ms
    assert latency["p50_ms"] == pytest.approx(355.0)  # type: ignore[index]
    assert latency["p95_ms"] == pytest.approx(575.5)  # type: ignore[index]
    empty = Rig(loaded)
    assert empty.pipeline.counters["latency"]["p50_ms"] is None  # type: ignore[index]


def test_a_unit_trigger_during_an_inspection_is_busy(loaded: LoadedConfig) -> None:
    error = CaptureError("timeout")
    rig = Rig(loaded, captures={1: error, 2: error, 4: error})
    rig.trigger()
    during = rig.clock.now - 0.05  # fired while event 1 was being handled
    rig.trigger()
    busy = rig.trigger(t0=during)
    assert busy.verdict.reason == "busy"  # type: ignore[attr-defined]
    assert busy.image_jpeg is None and busy.camera is None  # type: ignore[attr-defined]
    assert rig.source.captured == [1, 2]
    assert rig.pipeline.state is State.READY  # two capture errors, busy changed no streak
    rig.trigger()
    assert rig.pipeline.fault_reason == "capture"
    assert [r.event.seq for r in rig.sink.records] == [1, 2, 3, 4]


def test_a_trigger_while_starting_is_busy(loaded: LoadedConfig) -> None:
    rig = Rig(loaded, start=False)
    assert rig.pipeline.state is State.STARTING
    record = rig.trigger()
    assert record.verdict.reason == "busy"  # type: ignore[attr-defined]
    assert rig.source.captured == []


def test_camera_verified_in_the_counters(loaded: LoadedConfig) -> None:
    rig = Rig(loaded)
    for _ in range(3):
        rig.trigger()
    assert rig.pipeline.counters["camera_verified"] is True
    unverified = Rig(loaded, captures={2: (0.62, UNVERIFIED)})
    for _ in range(3):
        unverified.trigger()
    assert unverified.pipeline.counters["camera_verified"] is False


def test_the_run_counters_in_the_database_and_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fast_sqlite(monkeypatch)
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 12)
    write_png(shoot / "f0006.png", textured(6, patch_grey=40))  # out of band
    script = {2: "defect:geometry", 3: "defect:appearance", 5: "defect:appearance",
              7: "placement:no unit", 8: "placement:edge", 10: "inference", 12: "crash"}
    code = main(
        station_args(config, shoot),
        make_inspector=lambda ld: StubInspector(ld.config_hash, script),
    )
    assert code == 0
    [run] = rows(tmp_path / "evidence" / "qs.db", "SELECT counters FROM run")
    counters = json.loads(run["counters"])
    assert counters["events"] == 12
    assert counters["labels"] == {"pass": 4, "defect": 3, "no_verdict": 5}
    assert counters["no_verdict"] == {"inference": 2, "lighting": 1, "placement": 2}
    assert (counters["sink"]["written"], counters["sink"]["pending"]) == (12, 0)
    assert set(counters["sink"]) >= {"written", "held", "images_shed", "rejected", "lost"}
    assert counters["camera_verified"] is False  # no frames.csv: never verified
    assert counters["latency"]["breaches"] == 0
    out = capsys.readouterr().out
    assert "events 12 · pass 4 · defect 3 · no_verdict 5" in out
    assert "no_verdict by reason: inference 2, lighting 1, placement 2" in out
    assert "records written 12" in out
    assert "camera verified no" in out
    assert np.isfinite(counters["latency"]["p95_ms"])
