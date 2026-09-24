"""The station states of docs/16 §2 (#71): start-up fails closed, FAULT follows 3 in a row, the
camera reconnects with backoff on a fake clock, and the stop is clean."""

from __future__ import annotations

import json
import signal
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig
from qs.contracts import Frame, FrameSource, Label, Verdict
from qs.errors import CameraLost, CameraSettingsDrift, CaptureError, ConfigError
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline, State
from qs.sink import SqliteSink
from qs.sources.replay import replay_files, replay_pair
from qs.station import Station
from station_fixtures import (
    VERIFIED,
    CountingTrigger,
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
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


class Rig:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        captures: dict[int, object] | None = None,
        opens: list[BaseException | None] | None = None,
        script: dict[int, str] | None = None,
        sink: object | None = None,
    ) -> None:
        self.clock = FakeClock()
        self.source = ScriptedSource(self.clock, captures=captures or {}, opens=opens or [])
        self.inspector = StubInspector(loaded.config_hash, script)
        self.inspector.load(loaded.config)
        self.shown = NullAnnunciator()
        self.sink = sink if sink is not None else RecordingSink()
        self.pipeline = Pipeline(
            config=loaded.config,
            config_hash=loaded.config_hash,
            run_id="R1",
            source=self.source,
            inspector=self.inspector,
            annunciator=self.shown,
            sink=self.sink,  # type: ignore[arg-type]
            clock=self.clock,
        )
        self.seq = 0

    def trigger(self, count: int = 1) -> list[object]:
        out = []
        for _ in range(count):
            self.seq += 1
            out.append(self.pipeline.handle(event(self.seq, self.clock())))
        return out

    def reasons(self) -> list[str]:
        return [r.verdict.reason or r.verdict.label.value for r in self.sink.records]  # type: ignore[attr-defined]


@pytest.fixture
def loaded(tmp_path: Path) -> LoadedConfig:
    return loaded_config(tmp_path)


# --- start-up ---------------------------------------------------------------------------------


class Refusing(StubInspector):
    def load(self, config: object) -> None:
        raise ConfigError("model patchcore-mochi-v1-3f9a1c2b7d4e: memory_bank.pt hash mismatch")


def test_an_inspector_that_will_not_load_stops_before_any_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 2)
    code = main(
        station_args(config, tmp_path / "shoot"),
        make_inspector=lambda loaded: Refusing(loaded.config_hash),
    )
    out = capsys.readouterr().out
    assert code == 2
    assert out.count("\n") == 1 and "hash mismatch" in out
    assert not (tmp_path / "evidence" / "qs.db").exists()


class DriftOnOpen:
    """Wraps the replay source: `open()` reads back a clamped exposure."""

    def __init__(self, inner: FrameSource) -> None:
        self.inner = inner

    def open(self) -> None:
        raise CameraSettingsDrift("exposure: set 10000, reads 9984")

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def test_a_readback_mismatch_at_start_stops_with_the_mismatch_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 2)
    code = main(station_args(config, tmp_path / "shoot"), wrap_source=DriftOnOpen)
    out = capsys.readouterr().out
    assert code == 2
    assert "exposure: set 10000, reads 9984" in out
    assert rows(tmp_path / "evidence" / "qs.db", "SELECT * FROM inspection") == []


def test_an_unknown_breach_action_is_refused_at_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, breach_action="drop_the_verdict")
    write_frames(tmp_path / "shoot", 2)
    assert main(station_args(config, tmp_path / "shoot")) == 2
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "breach_action" in out
    assert not (tmp_path / "evidence").exists()


def test_an_out_of_band_preview_starts_with_lighting(loaded: LoadedConfig) -> None:
    rig = Rig(loaded)
    rig.source.preview_pixels = textured(1, patch_grey=40)
    rig.pipeline.start()
    assert rig.pipeline.state is State.READY
    assert "LIGHTING" in rig.pipeline.warnings


# --- FAULT(camera) and the reconnect ----------------------------------------------------------


def test_camera_lost_at_start_reconnects_on_the_backoff(
    loaded: LoadedConfig, tmp_path: Path
) -> None:
    sink = SqliteSink(tmp_path / "ev", "R1", "A-bench", datetime.now(UTC), loaded.config_hash,
                      retry_s=0.01, stop_wait_s=1.0)
    lost = CameraLost("/dev/video0: cannot open")
    rig = Rig(loaded, opens=[lost] * 8, sink=sink)  # the 9th open succeeds
    start = rig.clock.now
    rig.pipeline.start()
    assert (rig.pipeline.state, rig.pipeline.fault_reason) == (State.FAULT, "camera")
    rig.trigger(2)
    for _ in range(4 * 60):
        rig.clock.advance(0.25)
        rig.pipeline.tick(rig.clock.now)
        if rig.pipeline.state is State.READY:
            break
        rig.trigger()
    attempts = [round(t - start, 6) for t in rig.source.open_times[1:]]
    assert attempts == [0.5, 1.5, 3.5, 7.5, 15.5, 25.5, 35.5, 45.5]
    assert rig.pipeline.counters["reconnect_attempts"] == 8
    assert rig.source.captured == []
    [after] = rig.trigger()
    assert after.verdict.label is Label.PASS  # type: ignore[attr-defined]
    rig.pipeline.stop()
    found = rows(tmp_path / "ev" / "qs.db", "SELECT seq, reason, image_status FROM inspection")
    assert [r["seq"] for r in found] == list(range(1, rig.seq + 1))
    assert {(r["reason"], r["image_status"]) for r in found[:-1]} == {("camera", "none")}


def test_camera_lost_mid_run_is_a_fault(loaded: LoadedConfig) -> None:
    rig = Rig(loaded, captures={2: CameraLost("gone")})
    rig.pipeline.start()
    rig.trigger(4)
    assert rig.reasons() == ["pass", "camera", "camera", "camera"]
    assert rig.source.captured == [1, 2]
    assert rig.pipeline.counters["faults"] == {"camera": 1}


# --- FAULT(capture) ---------------------------------------------------------------------------


def test_three_capture_errors_in_a_row_are_a_fault_and_a_reopen_recovers(
    loaded: LoadedConfig,
) -> None:
    error = CaptureError("no frame newer than the trigger within 500 ms")
    captures: dict[int, object] = {1: error, 2: error, 4: error, 5: error, 6: error}
    rig = Rig(loaded, captures=captures)
    rig.pipeline.start()
    rig.trigger(5)
    assert rig.pipeline.state is State.READY
    rig.trigger()
    assert (rig.pipeline.state, rig.pipeline.fault_reason) == (State.FAULT, "capture")
    rig.trigger()
    assert rig.reasons()[6] == "capture" and rig.source.captured == [1, 2, 3, 4, 5, 6]
    rig.clock.advance(0.5)
    rig.pipeline.tick(rig.clock.now)
    assert rig.pipeline.state is State.READY and rig.source.closes >= 1
    [after] = rig.trigger()
    assert after.verdict.label is Label.PASS  # type: ignore[attr-defined]


# --- FAULT(inference) -------------------------------------------------------------------------


def test_three_inference_failures_are_a_fault_never_retried(loaded: LoadedConfig) -> None:
    rig = Rig(loaded, script={1: "inference", 2: "crash", 3: "nonfinite"})
    rig.pipeline.start()
    rig.trigger(2)
    assert rig.pipeline.state is State.READY
    rig.trigger()
    assert (rig.pipeline.state, rig.pipeline.fault_reason) == (State.FAULT, "inference")
    records = rig.trigger(5)
    assert all(r.verdict.reason == "inference" and r.image_jpeg is None for r in records)  # type: ignore[attr-defined]
    assert rig.source.captured == [1, 2, 3]
    opens = len(rig.source.open_times)
    for _ in range(240):
        rig.clock.advance(0.25)
        rig.pipeline.tick(rig.clock.now)
    assert len(rig.source.open_times) == opens
    assert rig.pipeline.state is State.FAULT


def test_only_inference_events_count_toward_the_inference_streak(loaded: LoadedConfig) -> None:
    script = {1: "inference", 2: "placement:no unit", 3: "inference", 5: "inference"}
    rig = Rig(loaded, captures={4: (0.1, VERIFIED)}, script=script)
    rig.pipeline.start()
    rig.trigger(4)
    assert rig.pipeline.state is State.READY
    rig.trigger()
    assert rig.reasons() == ["inference", "placement", "inference", "lighting", "inference"]
    assert (rig.pipeline.state, rig.pipeline.fault_reason) == (State.FAULT, "inference")


# --- a mismatch on reconnect, and the stop ----------------------------------------------------


def test_a_readback_mismatch_on_reconnect_stops_the_station(loaded: LoadedConfig) -> None:
    rig = Rig(loaded, captures={1: CameraLost("gone")},
              opens=[None, CameraSettingsDrift("exposure: set 10000, reads 9984")])
    rig.pipeline.start()
    rig.trigger()
    rig.clock.advance(0.5)
    rig.pipeline.tick(rig.clock.now)
    assert rig.pipeline.state is State.STOPPED
    assert rig.pipeline.refusal is not None and "reads 9984" in rig.pipeline.refusal
    assert rig.sink.finished is not None and rig.sink.closed  # type: ignore[attr-defined]


class StopAt(NullAnnunciator):
    def __init__(self, seq: int, stop: object) -> None:
        super().__init__()
        self.seq, self.stop = seq, stop

    def show(self, frame: Frame | None, verdict: Verdict) -> None:
        super().show(frame, verdict)
        if frame is not None and frame.event.seq == self.seq:
            self.stop()  # type: ignore[operator]


def test_request_stop_mid_replay_ends_the_run_cleanly(tmp_path: Path) -> None:
    loaded = loaded_config(tmp_path)
    files = replay_files(write_frames(tmp_path / "shoot", 8)[0].parent)
    clock = FakeClock(step=0.001)
    replay, source = replay_pair(loaded.config, files, "shoot", clock=clock)
    trigger = CountingTrigger(replay)
    holder: dict[str, Pipeline] = {}
    station = Station(
        loaded,
        trigger=trigger,
        source=source,
        inspector=StubInspector(loaded.config_hash, {2: "placement:edge"}),
        annunciator=StopAt(4, lambda: holder["p"].request_stop()),
        make_sink=lambda ld, run, at: SqliteSink.from_config(ld, run, at, retry_s=0.01),
        clock=clock,
    )
    holder["p"] = station.start()
    station.run()
    station.stop()
    assert trigger.polls == 4
    database = tmp_path / "evidence" / "qs.db"
    assert [r["seq"] for r in rows(database, "SELECT seq FROM inspection")] == [1, 2, 3, 4]
    [run] = rows(database, "SELECT * FROM run")
    assert run["ended_utc"] is not None
    counters = json.loads(run["counters"])
    assert counters["labels"] == {"pass": 3, "defect": 0, "no_verdict": 1}
    assert counters["no_verdict"] == {"placement": 1}
    assert counters["faults"] == {} and counters["reconnect_attempts"] == 0
    assert counters["sink"]["written"] == 4


def test_sigint_stops_the_command_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 6)
    before = signal.getsignal(signal.SIGINT)
    code = main(
        station_args(config, tmp_path / "shoot"),
        make_annunciator=lambda: StopAt(3, lambda: signal.raise_signal(signal.SIGINT)),
    )
    assert code == 0
    assert signal.getsignal(signal.SIGINT) is before
    database = tmp_path / "evidence" / "qs.db"
    assert [r["seq"] for r in rows(database, "SELECT seq FROM inspection")] == [1, 2, 3]
    assert rows(database, "SELECT ended_utc FROM run")[0]["ended_utc"] is not None
    assert "events 3" in capsys.readouterr().out


def test_a_replay_ending_in_fault_inference_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 6)
    code = main(
        station_args(config, tmp_path / "shoot"),
        make_inspector=lambda ld: StubInspector(ld.config_hash, {4: "inference", 5: "inference",
                                                                  6: "inference"}),
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "final state FAULT(inference)" in out
    assert "records written 6" in out
