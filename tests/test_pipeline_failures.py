"""Every per-unit failure ends in НЕТ РЕШЕНИЯ with the right reason, image and provenance, and
never in a pass (#70; docs/11 §7, docs/16 F2, F4, F5, F6, F7, F12, F15). Each failure is judged
on its own here; the FAULT transitions are #71."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import qs.pipeline
from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.config import LoadedConfig
from qs.contracts import Frame, Label, Verdict
from qs.errors import (
    CameraLost,
    CameraSettingsDrift,
    CaptureError,
    InferenceError,
    LightingFault,
)
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline, State
from qs.sink import SqliteSink
from station_fixtures import (
    UNVERIFIED,
    VERIFIED,
    FakeClock,
    RecordingSink,
    ScriptedSource,
    event,
    loaded_config,
    rows,
)

EXPECT, BAND = 0.62, 0.05


class Slow:
    """Wraps an inspector: `inspect` takes `seconds` on the fake clock before it answers."""

    def __init__(self, inner: StubInspector, clock: FakeClock, seconds: float) -> None:
        self.inner, self.clock, self.seconds = inner, clock, seconds

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def load(self, config: object) -> None:
        self.inner.load(config)  # type: ignore[arg-type]

    def inspect(self, frame: Frame) -> Verdict:
        self.clock.advance(self.seconds)
        return self.inner.inspect(frame)


class Rig:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        captures: dict[int, object] | None = None,
        script: dict[int, str] | None = None,
        sink: object | None = None,
        annunciator: NullAnnunciator | None = None,
        inspect_s: float = 0.0,
    ) -> None:
        self.clock = FakeClock()
        self.source = ScriptedSource(self.clock, captures=captures or {})
        self.inspector = Slow(StubInspector(loaded.config_hash, script), self.clock, inspect_s)
        self.inspector.load(loaded.config)
        self.shown = annunciator or NullAnnunciator()
        self.sink = sink if sink is not None else RecordingSink()
        self.pipeline = Pipeline(
            config=loaded.config,
            config_hash=loaded.config_hash,
            run_id="R1",
            source=self.source,
            inspector=self.inspector,  # type: ignore[arg-type]
            annunciator=self.shown,
            sink=self.sink,  # type: ignore[arg-type]
            clock=self.clock,
        )
        self.pipeline.start()

    def run(self, count: int) -> list[object]:
        return [self.pipeline.handle(event(seq, self.clock())) for seq in range(1, count + 1)]


@pytest.fixture
def loaded(tmp_path: Path) -> LoadedConfig:
    return loaded_config(tmp_path)


CAPTURE_ROWS = [
    (CaptureError("no frame newer than the trigger within 500 ms"), "capture"),
    (CaptureError("corrupt frame: flat"), "capture"),
    (CameraLost("/dev/video0: camera stopped delivering frames"), "camera"),
    (CameraSettingsDrift("exposure: set 10000, reads 9984"), "camera_settings"),
    (LightingFault("reference patch outside the frame"), "lighting"),
]


@pytest.mark.parametrize(("raised", "reason"), CAPTURE_ROWS)
def test_a_capture_failure_is_no_verdict_without_an_image(
    loaded: LoadedConfig, raised: Exception, reason: str
) -> None:
    rig = Rig(loaded, captures={1: raised})
    [record] = rig.run(1)
    assert rig.sink.records == [record]  # type: ignore[attr-defined]
    assert record.verdict.label is Label.NO_VERDICT  # type: ignore[attr-defined]
    assert record.verdict.reason == reason  # type: ignore[attr-defined]
    assert record.image_jpeg is None and record.camera is None  # type: ignore[attr-defined]
    assert record.capture_ms is None  # type: ignore[attr-defined]
    [(frame, shown)] = rig.shown.calls
    assert frame is None and shown.label is Label.NO_VERDICT


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [("placement:no unit", "placement"), ("inference", "inference"), ("crash", "inference"),
     ("nonfinite", "inference"), ("mislabel", "inference")],
)
def test_an_inspection_failure_is_no_verdict_with_the_image(
    loaded: LoadedConfig, outcome: str, reason: str
) -> None:
    rig = Rig(loaded, script={1: outcome}, inspect_s=0.05)
    [record] = rig.run(1)
    verdict = record.verdict  # type: ignore[attr-defined]
    assert (verdict.label, verdict.reason) == (Label.NO_VERDICT, reason)
    assert verdict.checks == ()
    assert record.image_jpeg is not None and record.camera == VERIFIED  # type: ignore[attr-defined]
    assert verdict.inspect_ms == pytest.approx(50.0)
    [(frame, shown)] = rig.shown.calls
    assert frame is not None and shown.label is Label.NO_VERDICT


def _patch_rig(loaded: LoadedConfig, patches: dict[int, float | None]) -> Rig:
    return Rig(loaded, captures={seq: (p, VERIFIED) for seq, p in patches.items()})


def test_the_lighting_band_includes_its_boundary(loaded: LoadedConfig) -> None:
    edge = EXPECT * (1 + BAND)
    above = edge + 1e-6
    rig = _patch_rig(loaded, {1: edge, 2: above, 3: EXPECT, 4: None, 5: EXPECT * (1 - BAND)})
    [first] = rig.run(1)
    assert first.verdict.label is Label.PASS  # type: ignore[attr-defined]
    second = rig.pipeline.handle(event(2, rig.clock()))
    assert (second.verdict.label, second.verdict.reason) == (Label.NO_VERDICT, "lighting")
    assert second.image_jpeg is not None and second.camera == VERIFIED
    assert "LIGHTING" in rig.pipeline.warnings
    assert rig.source.captured == [1, 2]
    third = rig.pipeline.handle(event(3, rig.clock()))
    assert third.verdict.label is Label.PASS
    assert "LIGHTING" not in rig.pipeline.warnings
    fourth = rig.pipeline.handle(event(4, rig.clock()))
    assert fourth.verdict.reason == "lighting"
    fifth = rig.pipeline.handle(event(5, rig.clock()))
    assert fifth.verdict.label is Label.PASS  # the lower edge is inside too


def test_the_inspector_is_never_called_out_of_band(loaded: LoadedConfig) -> None:
    calls: list[int] = []
    rig = _patch_rig(loaded, {1: 0.3})
    inner = rig.inspector.inspect

    def counting(frame: Frame) -> Verdict:
        calls.append(frame.event.seq)
        return inner(frame)

    rig.inspector.inspect = counting  # type: ignore[method-assign]
    rig.run(1)
    assert calls == []


def test_drift_then_an_unverifiable_camera(loaded: LoadedConfig) -> None:
    captures: dict[int, object] = {2: CameraSettingsDrift("gain: set 0, reads 12")}
    for seq in range(4, 7):
        captures[seq] = (EXPECT, UNVERIFIED)
    rig = Rig(loaded, captures=captures)
    labels, raised = [], []
    for seq in range(1, 7):
        record = rig.pipeline.handle(event(seq, rig.clock()))
        labels.append(record.verdict.reason or record.verdict.label.value)
        raised.append("CAMERA_SETTINGS" in rig.pipeline.warnings)
    assert labels == ["pass", "camera_settings", "pass", "pass", "pass", "pass"]
    assert raised == [False, True, False, True, True, True]
    assert [r.camera.verified for r in rig.sink.records[3:]] == [False] * 3  # type: ignore[attr-defined]


def test_a_heatmap_that_cannot_be_encoded_is_an_inference_failure(
    loaded: LoadedConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object) -> bytes:
        raise InferenceError("the heatmap could not be encoded as a PNG")

    monkeypatch.setattr(qs.pipeline, "heatmap_png", broken)
    rig = Rig(loaded, script={1: "defect:appearance"})
    [record] = rig.run(1)
    verdict = record.verdict  # type: ignore[attr-defined]
    assert (verdict.label, verdict.reason) == (Label.NO_VERDICT, "inference")
    assert record.image_jpeg is not None and record.heatmap_png is None  # type: ignore[attr-defined]
    assert verdict.heatmap is None


def test_a_failing_annunciator_and_a_failing_sink_are_counted(loaded: LoadedConfig) -> None:
    rig = Rig(loaded, annunciator=NullAnnunciator(fail=True))
    rig.run(10)
    assert len(rig.sink.records) == 10  # type: ignore[attr-defined]
    assert rig.pipeline.counters["annunciator_errors"] == 10
    failing = Rig(loaded, sink=RecordingSink(fail=True))
    [record] = failing.run(1)
    assert record.verdict.label is Label.PASS  # type: ignore[attr-defined]
    assert failing.pipeline.counters["sink_errors"] == 1


def test_every_pipeline_made_no_verdict_carries_its_provenance(loaded: LoadedConfig) -> None:
    captures: dict[int, object] = {1: CaptureError("timeout"), 2: CameraLost("gone"),
                                   3: CameraSettingsDrift("drift"), 4: (0.1, VERIFIED)}
    script = {5: "placement:edge", 6: "inference", 7: "crash", 8: "nonfinite", 9: "mislabel"}
    rig = Rig(loaded, captures=captures, script=script)
    records = rig.run(9)
    for record in records:
        verdict = record.verdict  # type: ignore[attr-defined]
        assert verdict.label is Label.NO_VERDICT
        assert verdict.checks == () and verdict.heatmap is None and verdict.unit_box is None
        assert verdict.model_id == "stub"
        assert verdict.threshold_id == loaded.config.threshold.id
        assert verdict.config_hash == loaded.config_hash
    assert [r.verdict.inspect_ms for r in records[:4]] == [0.0] * 4  # type: ignore[attr-defined]


def test_a_mixed_run_through_the_real_sink(
    loaded: LoadedConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast_sqlite(monkeypatch)
    root = tmp_path / "evidence"
    sink = SqliteSink(root, "R1", loaded.config.station_id, datetime.now(UTC),
                      loaded.config_hash, retry_s=0.01, stop_wait_s=1.0)
    captures: dict[int, object] = {3: CaptureError("timeout"), 6: CameraSettingsDrift("drift"),
                                   9: (0.1, VERIFIED), 12: LightingFault("roi"),
                                   20: CameraLost("gone")}
    script = {4: "defect:appearance", 7: "placement:no unit", 10: "inference", 13: "mislabel",
              17: "defect:geometry", 19: "nonfinite"}
    rig = Rig(loaded, captures=captures, script=script, sink=sink)
    rig.run(20)
    rig.pipeline.stop()
    found = rows(root / "qs.db", "SELECT * FROM inspection ORDER BY seq")
    assert [r["seq"] for r in found] == list(range(1, 21))
    for r in found:
        if r["reason"] in ("capture", "camera", "camera_settings") or r["seq"] == 12:
            assert r["image_status"] == "none"
    assert rows(
        root / "qs.db",
        "SELECT c.* FROM inspection_check c JOIN inspection i ON i.id = c.inspection_id"
        " WHERE i.label = 'no_verdict'",
    ) == []
    assert sum(r["label"] == "pass" for r in found) == 20 - len(captures) - len(script)
    assert sum(r["label"] == "defect" for r in found) == 2
    assert all(r["label"] != "pass" for r in found if r["seq"] in script or r["seq"] in captures)


def test_placement_and_lighting_change_counters_and_no_state(loaded: LoadedConfig) -> None:
    rig = Rig(loaded, captures={2: (0.2, VERIFIED)}, script={1: "placement:no unit"})
    rig.run(3)
    counters = rig.pipeline.counters
    assert counters["no_verdict"] == {"lighting": 1, "placement": 1}
    assert counters["labels"] == {"pass": 1, "defect": 0, "no_verdict": 2}
    assert counters["events"] == 3
    assert rig.pipeline.state is State.READY
