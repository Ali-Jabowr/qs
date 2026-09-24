"""Fault injection — the 200-event runs behind NFR-04, NFR-05 and NFR-06 (#73, docs/16 §7).

Each test is one 200-event replay through `qs station`'s `main` with the real `SqliteSink`, and
ends with: records written = events = 200, `seq` 1–200 gapless in the database, no crash, and
every NO_VERDICT carrying its reason. A fault-injecting `FrameSource` wraps `ReplaySource` to
lose the camera; storage faults use P03's means (a `chmod` on the evidence root). The Pipeline
runs on a fake clock that moves 10 ms on every read, so the reconnect backoff is exact.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import qs.cmd.station
from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig
from qs.contracts import Frame, FrameSource, TriggerEvent, Verdict
from qs.errors import CameraLost
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline
from qs.sink import SqliteSink
from qs.station import Station
from station_fixtures import (
    HEIGHT,
    WIDTH,
    FakeClock,
    every_queued_record_held,
    rows,
    station_args,
    textured,
    wait_for,
    write_config,
    write_frames,
    write_frames_csv,
    write_png,
)

EVENTS = 200
needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores chmod: a read-only directory cannot be simulated"
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


class LosingSource:
    """Wraps `ReplaySource`: the camera is lost at event `lose_at`; after that every `open()`
    fails until attempt `reconnect_on`, which succeeds."""

    def __init__(self, inner: FrameSource, lose_at: int, reconnect_on: int) -> None:
        self.inner = inner
        self.lose_at = lose_at
        self.reconnect_on = reconnect_on
        self.lost = False
        self.attempts = 0

    def open(self) -> None:
        if self.lost:
            self.attempts += 1
            if self.attempts < self.reconnect_on:
                raise CameraLost(f"reconnect attempt {self.attempts}: no device")
            self.lost = False
        self.inner.open()

    def capture(self, event: TriggerEvent) -> Frame:
        if self.lost or event.seq == self.lose_at and self.attempts == 0:
            self.lost = True
            raise CameraLost("/dev/video0: camera stopped delivering frames (injected)")
        return self.inner.capture(event)

    def preview(self) -> np.ndarray | None:
        return self.inner.preview()

    def health(self) -> object:
        return self.inner.health()

    def close(self) -> None:
        self.inner.close()


@dataclass
class Spy:
    """What a run exposes to the test: the Pipeline, the sink, and per-event observations."""

    pipeline: Pipeline | None = None
    sink: SqliteSink | None = None
    storage: dict[int, bool] = field(default_factory=dict)  # STORAGE when seq was shown
    handle_s: list[float] = field(default_factory=list)  # wall seconds per handle()


class Hooks(NullAnnunciator):
    """Calls `hooks[seq]()` after showing `seq`, and records STORAGE as it was then."""

    def __init__(self, spy: Spy, hooks: dict[int, Callable[[], None]]) -> None:
        super().__init__()
        self.spy, self.hooks = spy, hooks

    def show(self, frame: Frame | None, verdict: Verdict) -> None:
        super().show(frame, verdict)
        seq = len(self.calls)
        assert self.spy.pipeline is not None
        self.spy.storage[seq] = "STORAGE" in self.spy.pipeline.warnings
        if seq in self.hooks:
            self.hooks[seq]()


def _spying_station(spy: Spy, between: Callable[[], None]) -> type[Station]:
    class Spying(Station):
        def start(self) -> Pipeline:
            pipeline = super().start()
            spy.pipeline = pipeline
            handle, tick = pipeline.handle, pipeline.tick

            def timed(event: TriggerEvent) -> object:
                begin = time.perf_counter()
                try:
                    return handle(event)
                finally:
                    spy.handle_s.append(time.perf_counter() - begin)

            def ticking(now: float | None = None) -> None:
                between()  # outside handle(): never counted as a verdict's time
                tick(now)

            pipeline.handle = timed  # type: ignore[method-assign]
            pipeline.tick = ticking  # type: ignore[method-assign]
            return pipeline

    return Spying


@dataclass
class Run:
    code: int
    out: str
    database: Path
    spy: Spy

    def inspections(self) -> list[dict[str, object]]:
        return [dict(r) for r in rows(self.database, "SELECT * FROM inspection ORDER BY id")]

    def counters(self) -> dict[str, object]:
        [run] = rows(self.database, "SELECT counters FROM run")
        return json.loads(run["counters"])  # type: ignore[no-any-return]

    def reasons(self) -> dict[int, str]:
        return {int(r["seq"]): str(r["reason"] or r["label"]) for r in self.inspections()}


def _replay(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    script: dict[int, str] | None = None,
    corrupt: dict[int, str] | None = None,
    lose_at: int | None = None,
    hooks: dict[int, Callable[[], None]] | None = None,
    between: Callable[[], None] = lambda: None,
    spy: Spy | None = None,
    clock: Callable[[], float] | None = None,
) -> Run:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    paths = write_frames(shoot, EVENTS)
    csv_rows = []
    for seq, kind in (corrupt or {}).items():
        path = paths[seq - 1]
        if kind == "flat":
            write_png(path, np.full((HEIGHT, WIDTH, 3), 90, np.uint8))
        elif kind == "shape":
            write_png(path, textured(seq)[:, : WIDTH - 8])
        else:  # "hash": the bytes on disk are not the ones recorded at the shoot
            csv_rows.append({"file": path.name, "verified": "true", "sha256": "0" * 64})
    if csv_rows:
        write_frames_csv(shoot, csv_rows)
    spy = spy if spy is not None else Spy()
    monkeypatch.setattr(qs.cmd.station, "Station", _spying_station(spy, between))

    def make_sink(loaded: LoadedConfig, run_id: str, started: object) -> SqliteSink:
        sink = SqliteSink.from_config(loaded, run_id, started, retry_s=0.02, stop_wait_s=5.0)  # type: ignore[arg-type]
        spy.sink = sink
        return sink

    def wrap(source: FrameSource) -> FrameSource:
        if lose_at is None:
            return source
        return LosingSource(source, lose_at, reconnect_on=3)  # type: ignore[return-value]

    code = main(
        station_args(config, shoot),
        make_inspector=lambda ld: StubInspector(ld.config_hash, script),
        make_sink=make_sink,  # type: ignore[arg-type]
        make_annunciator=lambda: Hooks(spy, hooks or {}),
        wrap_source=wrap,
        clock=FakeClock(step=0.01) if clock is None else clock,
    )
    run = Run(code, capsys.readouterr().out, tmp_path / "evidence" / "qs.db", spy)
    _every_event_has_one_record(run)
    return run


def _every_event_has_one_record(run: Run) -> None:
    """NFR-06 and C5: records written = events = 200, seq gapless, every NO_VERDICT has its
    reason and none is a pass (C7); NFR-04: the run completed."""
    found = run.inspections()
    assert sorted(int(r["seq"]) for r in found) == list(range(1, EVENTS + 1))  # type: ignore[call-overload]
    counters = run.counters()
    assert counters["events"] == EVENTS
    assert counters["sink"]["written"] == EVENTS  # type: ignore[index]
    assert counters["sink"]["lost"] == 0 and counters["sink"]["rejected"] == 0  # type: ignore[index]
    for r in found:
        if r["label"] == "no_verdict":
            assert r["reason"] is not None
    assert f"records written {EVENTS}" in run.out
    assert len(run.spy.handle_s) == EVENTS


def _stored_images_match(run: Run) -> None:
    root = run.database.parent
    for r in run.inspections():
        if r["image_status"] == "stored":
            data = (root / str(r["image_path"])).read_bytes()
            assert hashlib.sha256(data).hexdigest() == r["image_sha256"]


# --- the core cases ---------------------------------------------------------------------------


def test_camera_lost_at_event_50_reconnects_on_the_third_attempt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _replay(tmp_path, capsys, monkeypatch, lose_at=50)
    assert run.code == 0
    reasons = run.reasons()
    assert all(reasons[seq] == "pass" for seq in range(1, 50))
    back = min(seq for seq in range(51, EVENTS + 1) if reasons[seq] != "camera")
    assert all(reasons[seq] == "camera" for seq in range(50, back))
    assert all(reasons[seq] == "pass" for seq in range(back, EVENTS + 1))
    assert back < EVENTS  # READY again, and the rest were inspected
    for r in run.inspections():
        if r["reason"] == "camera":
            assert r["image_status"] == "none"
    counters = run.counters()
    assert counters["faults"] == {"camera": 1}
    assert counters["reconnect_attempts"] == 3
    assert counters["final_state"] == "READY"


@dataclass
class ReadOnly:
    """The evidence root read-only from `start()` until the tick after `restore()`, which is
    outside handle(). That tick, each wait bounded to 5 s, first waits until the sink has failed
    on every record queued so far, so each record of the outage has waited through a failed
    attempt (`held`) however the sink's worker thread happened to be scheduled — a real outage
    outlasts a retry, a 100-event one on the fake clock takes a few milliseconds of wall time;
    then makes the root writable; then waits until the backlog is written."""

    root: Path
    spy: Spy
    ending: bool = False
    done: bool = False

    def start(self) -> None:
        self.root.chmod(0o555)

    def restore(self) -> None:
        self.ending = True

    def between(self) -> None:
        if not self.ending or self.done:
            return
        sink = self.spy.sink
        assert sink is not None
        wait_for(lambda: every_queued_record_held(sink), "a failed attempt on every held record")
        self.root.chmod(0o755)
        wait_for(lambda: sink.pending() == 0, "the backlog written after the outage")
        self.done = True

    @contextmanager
    def always_restored(self) -> Iterator[ReadOnly]:
        try:
            yield self
        finally:
            if self.root.exists():
                self.root.chmod(0o755)


@needs_non_root
def test_a_read_only_evidence_directory_for_events_50_to_149(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = Spy()
    with ReadOnly(tmp_path / "evidence", spy).always_restored() as window:
        run = _replay(
            tmp_path, capsys, monkeypatch,
            hooks={49: window.start, 149: window.restore},
            between=window.between,
            spy=spy,
        )
    assert run.code == 0
    assert max(run.spy.handle_s) < 1.0  # NFR-05 within NFR-01's line: storage never delays
    assert any(run.spy.storage[seq] for seq in range(50, 150))
    assert not any(run.spy.storage[seq] for seq in range(1, 50))
    assert not run.spy.storage[EVENTS]  # cleared once the backlog was written
    found = run.inspections()
    assert [r["seq"] for r in found] == list(range(1, EVENTS + 1))  # held, then in seq order
    assert all(r["image_status"] == "stored" for r in found)
    _stored_images_match(run)
    sink = run.counters()["sink"]
    assert sink["held"] >= 100  # type: ignore[index]
    assert sink["images_shed"] == 0 and sink["lost"] == 0  # type: ignore[index]


def test_corrupt_frames_are_capture_and_three_in_a_row_a_fault(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupt = {20: "flat", 40: "shape", 60: "hash", 100: "flat", 101: "shape", 102: "hash"}
    run = _replay(tmp_path, capsys, monkeypatch, corrupt=corrupt)
    assert run.code == 0
    reasons = run.reasons()
    for seq in (20, 40, 60):
        assert reasons[seq] == "capture"
        assert reasons[seq + 1] == "pass"  # not in a row: still READY
    assert [reasons[seq] for seq in (100, 101, 102)] == ["capture"] * 3
    back = min(seq for seq in range(103, EVENTS + 1) if reasons[seq] == "pass")
    assert all(reasons[seq] == "capture" for seq in range(103, back))  # FAULT(capture)
    assert all(reasons[seq] == "pass" for seq in range(back, EVENTS + 1))  # reopened
    counters = run.counters()
    assert counters["faults"] == {"capture": 1}
    assert counters["reconnect_attempts"] == 1
    assert counters["final_state"] == "READY"


def test_three_bad_frames_on_the_real_clock_recover_the_same_way_every_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REVIEW-P04 m1: on the real clock a replay polls faster than the reconnect backoff.
    The reopen is due on the replay's own time, not after 0.5 s of wall time, so the frames
    after a FAULT(capture) get their verdicts and two runs agree record for record."""
    runs = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        runs.append(_replay(
            root, capsys, monkeypatch,
            corrupt={1: "flat", 2: "shape", 3: "hash"},
            clock=time.perf_counter,
        ))
    for run in runs:
        assert run.code == 0, run.out
        reasons = run.reasons()
        assert [reasons[seq] for seq in (1, 2, 3)] == ["capture"] * 3
        assert all(reasons[seq] == "pass" for seq in range(4, EVENTS + 1))
        counters = run.counters()
        assert counters["faults"] == {"capture": 1}
        assert counters["reconnect_attempts"] == 1
        assert counters["final_state"] == "READY"
    assert runs[0].reasons() == runs[1].reasons()


def test_non_finite_scores_are_a_fault_that_never_clears(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _replay(tmp_path, capsys, monkeypatch,
                  script={30: "nonfinite", 31: "nonfinite", 32: "nonfinite"})
    assert run.code == 1
    reasons = run.reasons()
    assert all(reasons[seq] == "inference" for seq in range(30, EVENTS + 1))
    for r in run.inspections():
        if int(r["seq"]) > 32:  # type: ignore[call-overload]
            assert r["image_status"] == "none"
        elif int(r["seq"]) >= 30:  # type: ignore[call-overload]
            assert r["image_status"] == "stored"
    assert run.counters()["final_state"] == "FAULT(inference)"
    assert "final state FAULT(inference)" in run.out


def test_placement_faults_every_tenth_event_are_not_a_fault(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = {seq: "placement:no unit" for seq in range(10, EVENTS + 1, 10)}
    run = _replay(tmp_path, capsys, monkeypatch, script=script)
    assert run.code == 0
    for r in run.inspections():
        expected = "placement" if int(r["seq"]) % 10 == 0 else None  # type: ignore[call-overload]
        assert r["reason"] == expected
        assert r["image_status"] == "stored"
    counters = run.counters()
    assert counters["faults"] == {}
    assert counters["no_verdict"] == {"placement": 20}


@needs_non_root
def test_all_of_it_in_one_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFR-04 as stated: one 200-event run with the faults interleaved — the camera lost at 50,
    the evidence read-only for 120–169, corrupt frames at 20, 40 and 180 — and no inference
    fault, so the run ends READY."""
    script = {seq: "placement:edge" for seq in (15, 35, 175)}
    script.update({seq: "defect:appearance" for seq in (5, 25, 190)})
    spy = Spy()
    with ReadOnly(tmp_path / "evidence", spy).always_restored() as window:
        run = _replay(
            tmp_path, capsys, monkeypatch,
            script=script,
            corrupt={20: "flat", 40: "shape", 180: "hash"},
            lose_at=50,
            hooks={119: window.start, 169: window.restore},
            between=window.between,
            spy=spy,
        )
    assert run.code == 0
    assert max(run.spy.handle_s) < 1.0
    counters = run.counters()
    found = run.inspections()
    by_label: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for r in found:
        by_label[str(r["label"])] = by_label.get(str(r["label"]), 0) + 1
        if r["reason"] is not None:
            by_reason[str(r["reason"])] = by_reason.get(str(r["reason"]), 0) + 1
    assert counters["labels"] == {"pass": 0, "defect": 0, "no_verdict": 0} | by_label
    assert counters["no_verdict"] == by_reason
    assert by_reason["capture"] == 3 and by_reason["placement"] == 3
    assert by_reason["camera"] >= 1
    assert counters["faults"] == {"camera": 1}
    assert counters["final_state"] == "READY"
    assert any(run.spy.storage[seq] for seq in range(120, 170))
    assert counters["sink"]["held"] >= 50  # type: ignore[index]
    _stored_images_match(run)
