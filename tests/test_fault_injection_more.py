"""Fault injection — the rest of docs/16 §7, each through the whole station.

Every fault here is already covered at a lower level (the sink, the store, the Pipeline); each
test repeats it through `qs station`'s `main` with the real `SqliteSink`, a fault-injecting
`FrameSource` around `ReplaySource` where the camera is at fault, and P03's means where storage
is (a `chmod` of the evidence root, a monkeypatched file write, a killed process). Every test
ends with: one record per event, `seq` gapless, no crash, and every NO_VERDICT with its reason.
The Pipeline runs on a fake clock that moves 10 ms on every read.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import qs.cmd.station
import qs.store
from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig
from qs.contracts import Frame, FrameSource, Inspector, TriggerEvent, TriggerSource, Verdict
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline
from qs.sink import SqliteSink
from qs.station import Station
from station_fixtures import FakeClock, rows, station_args, write_config, write_frames

needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores chmod: a read-only directory cannot be simulated"
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


# --- the harness ------------------------------------------------------------------------------


@dataclass
class Spy:
    """What a run exposes to the test: the Pipeline, the sink, and what each event saw."""

    pipeline: Pipeline | None = None
    sink: SqliteSink | None = None
    warnings: dict[int, frozenset[str]] = field(default_factory=dict)  # when seq was shown
    handle_s: list[float] = field(default_factory=list)  # wall seconds per handle()
    inspected: list[int] = field(default_factory=list)  # seqs that reached inspect()


class Hooks(NullAnnunciator):
    """Records the warnings as they were when each event was shown, then calls `hooks[seq]()`.
    Every event is shown once, in order, so the n-th call is `seq` n."""

    def __init__(self, spy: Spy, hooks: dict[int, Callable[[], None]]) -> None:
        super().__init__()
        self.spy, self.hooks = spy, hooks

    def show(self, frame: Frame | None, verdict: Verdict) -> None:
        super().show(frame, verdict)
        seq = len(self.calls)
        assert self.spy.pipeline is not None
        self.spy.warnings[seq] = self.spy.pipeline.warnings
        if seq in self.hooks:
            self.hooks[seq]()


class Watched:
    """The stub, recording every `seq` that reaches `inspect()`."""

    def __init__(self, inner: StubInspector, spy: Spy) -> None:
        self.inner, self.spy = inner, spy

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def load(self, config: Any) -> None:
        self.inner.load(config)

    def inspect(self, frame: Frame) -> Verdict:
        self.spy.inspected.append(frame.event.seq)
        return self.inner.inspect(frame)


def _spying_station(
    spy: Spy,
    between: Callable[[], None],
    wrap_trigger: Callable[[TriggerSource], TriggerSource] | None,
) -> type[Station]:
    class Spying(Station):
        def __init__(self, loaded: LoadedConfig, **parts: Any) -> None:
            if wrap_trigger is not None:
                parts["trigger"] = wrap_trigger(parts["trigger"])
            super().__init__(loaded, **parts)

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
    seconds: float

    def inspections(self) -> list[dict[str, Any]]:
        return [dict(r) for r in rows(self.database, "SELECT * FROM inspection ORDER BY id")]

    def run_row(self) -> dict[str, Any]:
        [run] = rows(self.database, "SELECT * FROM run")
        return dict(run)

    def counters(self) -> dict[str, Any]:
        return json.loads(self.run_row()["counters"])  # type: ignore[no-any-return]

    def by_seq(self) -> dict[int, dict[str, Any]]:
        return {int(r["seq"]): r for r in self.inspections()}

    def reasons(self) -> dict[int, str]:
        return {seq: str(r["reason"] or r["label"]) for seq, r in self.by_seq().items()}


def _station(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: int = 200,
    script: dict[int, str] | None = None,
    prepare: Callable[[list[Path]], None] = lambda paths: None,
    wrap_source: Callable[[FrameSource], FrameSource] | None = None,
    wrap_trigger: Callable[[TriggerSource], TriggerSource] | None = None,
    hooks: dict[int, Callable[[], None]] | None = None,
    between: Callable[[], None] = lambda: None,
    spy: Spy | None = None,
    stop_wait_s: float | None = 5.0,
    clock: FakeClock | None = None,
) -> Run:
    """One replay of `events` frames through `main`. `prepare` may rewrite the frames or add a
    `frames.csv`; `stop_wait_s=None` keeps the sink's own stop wait (P03's 10 s)."""
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    prepare(write_frames(shoot, events))
    spy = spy if spy is not None else Spy()
    monkeypatch.setattr(qs.cmd.station, "Station", _spying_station(spy, between, wrap_trigger))

    def make_sink(loaded: LoadedConfig, run_id: str, started: Any) -> SqliteSink:
        if stop_wait_s is None:
            sink = SqliteSink.from_config(loaded, run_id, started)
        else:
            sink = SqliteSink.from_config(
                loaded, run_id, started, retry_s=0.02, stop_wait_s=stop_wait_s
            )
        spy.sink = sink
        return sink

    def make_inspector(loaded: LoadedConfig) -> Inspector:
        return Watched(StubInspector(loaded.config_hash, script), spy)  # type: ignore[return-value]

    begin = time.monotonic()
    code = main(
        station_args(config, shoot),
        make_inspector=make_inspector,
        make_sink=make_sink,  # type: ignore[arg-type]
        make_annunciator=lambda: Hooks(spy, hooks or {}),
        wrap_source=wrap_source or (lambda source: source),
        clock=clock if clock is not None else FakeClock(step=0.01),
    )
    seconds = time.monotonic() - begin
    return Run(code, capsys.readouterr().out, tmp_path / "evidence" / "qs.db", spy, seconds)


def _one_record_per_event(run: Run, events: int) -> None:
    """C5 and NFR-06: records written = events, `seq` 1…events gapless, every NO_VERDICT with
    its reason and never a pass (C7); NFR-04: the run completed, with nothing lost."""
    found = run.inspections()
    assert [int(r["seq"]) for r in found] == list(range(1, events + 1))
    counters = run.counters()
    assert counters["events"] == events
    assert counters["sink"]["written"] == events
    assert counters["sink"]["lost"] == 0 and counters["sink"]["rejected"] == 0
    for r in found:
        assert (r["label"] == "no_verdict") == (r["reason"] is not None)
    assert f"records written {events} " in run.out
    assert len(run.spy.handle_s) == events


def _stored_files_match(run: Run) -> None:
    root = run.database.parent
    for r in run.inspections():
        if r["image_status"] == "stored":
            data = (root / str(r["image_path"])).read_bytes()
            assert hashlib.sha256(data).hexdigest() == r["image_sha256"]
        if r["heatmap_path"] is not None:
            assert (root / str(r["heatmap_path"])).is_file()


@dataclass
class Drain:
    """After `restore()`, the next tick waits (outside handle()) until the backlog is written,
    bounded to 5 s, so the events after the outage meet an empty queue."""

    spy: Spy
    restored: bool = False
    drained: bool = False

    def restore(self) -> None:
        self.restored = True

    def between(self) -> None:
        if self.restored and not self.drained:
            assert self.spy.sink is not None
            deadline = time.monotonic() + 5.0
            while self.spy.sink.pending() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.drained = True


# --- 1 · ENOSPC -------------------------------------------------------------------------------


def test_1_enospc_for_events_50_to_149_holds_every_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/16 §7, "Sink raises OSError(ENOSPC)": as the read-only case. The store's file write
    raises `ENOSPC` from event 50 to 149; every verdict is on time, `STORAGE` is raised and then
    cleared, every held row is written in `seq` order with its image, and nothing is shed."""
    full = threading.Event()
    original = qs.store.write_file

    def no_space(path: Path, data: bytes) -> None:
        if full.is_set():
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(path))
        original(path, data)

    monkeypatch.setattr(qs.store, "write_file", no_space)
    spy = Spy()
    drain = Drain(spy)
    details: list[str] = []

    def restore() -> None:
        assert spy.sink is not None
        details.append(spy.sink.health().detail)
        full.clear()
        drain.restore()

    run = _station(
        tmp_path, capsys, monkeypatch,
        hooks={49: full.set, 149: restore}, between=drain.between, spy=spy,
    )
    _one_record_per_event(run, 200)
    assert run.code == 0
    assert max(run.spy.handle_s) < 1.0  # NFR-05 within NFR-01's line: storage never delays
    assert not any("STORAGE" in run.spy.warnings[seq] for seq in range(1, 50))
    assert any("STORAGE" in run.spy.warnings[seq] for seq in range(50, 150))
    assert "STORAGE" not in run.spy.warnings[200]  # cleared once the backlog was written
    assert "No space left on device" in details[0]
    found = run.inspections()
    assert all(r["image_status"] == "stored" and r["label"] == "pass" for r in found)
    _stored_files_match(run)
    sink = run.counters()["sink"]
    assert sink["held"] >= 100 and sink["write_failures"] >= 1
    assert sink["images_shed"] == 0
    assert run.counters()["final_state"] == "READY"
