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
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import qs.cmd.station
import qs.store
from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig
from qs.contracts import (
    Frame,
    FrameSource,
    Health,
    Inspector,
    TriggerEvent,
    TriggerSource,
    Verdict,
)
from qs.errors import CameraSettingsDrift, CaptureError
from qs.inspector.stub import StubInspector
from qs.pipeline import Pipeline
from qs.sink import SqliteSink
from qs.station import Station
from qs.store import Store, verify
from station_fixtures import (
    UNVERIFIED,
    FakeClock,
    every_queued_record_held,
    loaded_config,
    rows,
    sha256_file,
    station_args,
    textured,
    wait_for,
    write_config,
    write_frames,
    write_frames_csv,
    write_png,
)

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
    """Ends a storage outage outside handle(). A hook calls `end()`, which only asks; the next
    tick then, each wait bounded to 5 s:

    1. waits until the sink has failed on every record queued so far, so each record of the
       outage has waited through a failed attempt (`held`), however the sink's worker thread
       happened to be scheduled — a real outage outlasts a retry; a 100-event one on the fake
       clock takes a few milliseconds of wall time;
    2. calls `restore`;
    3. waits until the backlog is written, so the events after the outage meet an empty queue.
    """

    spy: Spy
    restore: Callable[[], None]
    ending: bool = False
    done: bool = False

    def end(self) -> None:
        self.ending = True

    def between(self) -> None:
        if not self.ending or self.done:
            return
        sink = self.spy.sink
        assert sink is not None
        wait_for(lambda: every_queued_record_held(sink), "a failed attempt on every held record")
        self.restore()
        wait_for(lambda: sink.pending() == 0, "the backlog written after the outage")
        self.done = True


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
    details: list[str] = []

    def restore() -> None:
        assert spy.sink is not None
        details.append(spy.sink.health().detail)
        full.clear()

    drain = Drain(spy, restore)
    run = _station(
        tmp_path, capsys, monkeypatch,
        hooks={49: full.set, 149: drain.end}, between=drain.between, spy=spy,
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


# --- 2 · shedding at station level ------------------------------------------------------------


@contextmanager
def _read_only(path: Path) -> Iterator[Callable[[], None]]:
    """`path` created and read-only until the yielded restore() is called, or the block ends."""
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o555)
    restored = False

    def restore() -> None:
        nonlocal restored
        if not restored:
            path.chmod(0o755)
            restored = True

    try:
        yield restore
    finally:
        restore()


@needs_non_root
def test_2_images_are_shed_pass_first_then_no_verdict_then_defect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/16 §7 and F11, ADR-0012, #56: 300 events with the evidence root read-only from
    before event 1 to after event 260. Events 1–260 carry 20 PASS, 20 NO_VERDICT(`placement`)
    and 220 DEFECT images, so the 60 images past the high-water of 200 are every PASS, then
    every NO_VERDICT, then the 20 oldest DEFECT. Every row is still written; a shed row keeps
    its label and has `image_status = 'shed'` and no file."""
    passes = set(range(1, 261, 13))
    placements = set(range(2, 261, 13))
    assert len(passes) == len(placements) == 20
    script = {seq: "defect:appearance" for seq in range(1, 261)}
    script.update({seq: "placement:edge" for seq in placements})
    for seq in passes:
        del script[seq]  # unscripted: a pass
    defects = sorted(set(range(1, 261)) - passes - placements)
    spy = Spy()
    with _read_only(tmp_path / "evidence") as restore:
        drain = Drain(spy, restore)
        run = _station(
            tmp_path, capsys, monkeypatch,
            events=300, script=script,
            hooks={260: drain.end}, between=drain.between, spy=spy,
        )
    _one_record_per_event(run, 300)
    assert run.code == 0
    assert max(run.spy.handle_s) < 1.0  # shedding never delays a verdict
    assert all("STORAGE" in run.spy.warnings[seq] for seq in range(201, 261))  # shedding
    assert "STORAGE" not in run.spy.warnings[300]
    found = run.by_seq()
    shed = {seq for seq, r in found.items() if r["image_status"] == "shed"}
    assert shed == passes | placements | set(defects[:20])
    for seq in shed:
        r = found[seq]
        assert (r["image_path"], r["image_sha256"], r["heatmap_path"]) == (None, None, None)
        assert r["camera_verified"] == 0  # the row keeps what the frame said
    for seq in set(range(1, 301)) - shed:
        assert found[seq]["image_status"] == "stored"
    assert {found[seq]["label"] for seq in passes} == {"pass"}
    assert {found[seq]["reason"] for seq in placements} == {"placement"}
    assert {found[seq]["label"] for seq in defects} == {"defect"}
    assert all(found[seq]["heatmap_path"] is not None for seq in defects[20:])
    _stored_files_match(run)
    root = run.database.parent
    on_disk = {p.name for p in root.rglob("*") if p.is_file() and p.suffix in (".jpg", ".png")}
    assert len(on_disk) == (300 - 60) + (220 - 20)  # images, and the kept heatmaps
    sink = run.counters()["sink"]
    assert sink["images_shed"] == 60
    assert sink["held"] == 260  # every row of the outage waited through a failed attempt
    assert run.counters()["labels"] == {"pass": 60, "defect": 220, "no_verdict": 20}


# --- 3 · stop while storage is down -----------------------------------------------------------


@needs_non_root
def test_3_a_stop_while_storage_is_down_ends_in_10_s_and_counts_the_lost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/16 §7 and F13: the evidence root goes read-only after event 20 is written, and
    SIGINT stops the run after event 30. The sink keeps P03's own 10 s stop wait; the command
    returns once it has passed, prints the 10 rows lost, exits 1, and the run's `ended_utc`
    stays NULL. Every event is accounted for: 20 rows written and 10 lost."""
    root = tmp_path / "evidence"
    spy = Spy()
    stopped_at: list[float] = []

    def outage() -> None:  # at the show of event 21, before its own write
        assert spy.sink is not None
        deadline = time.monotonic() + 5.0
        while spy.sink.pending() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert spy.sink.pending() == 0
        root.chmod(0o555)

    def stop() -> None:
        stopped_at.append(time.monotonic())
        signal.raise_signal(signal.SIGINT)

    try:
        run = _station(
            tmp_path, capsys, monkeypatch,
            events=40, hooks={21: outage, 30: stop}, spy=spy, stop_wait_s=None,
        )
        returned = time.monotonic()
    finally:
        root.chmod(0o755)
    assert run.code == 1
    assert 10.0 <= returned - stopped_at[0] < 11.5  # P03's wait, not shortened, then out
    assert [int(r["seq"]) for r in run.inspections()] == list(range(1, 21))
    run_row = run.run_row()
    assert run_row["ended_utc"] is None and run_row["counters"] is None
    assert "events 30 · pass 30" in run.out  # nothing polled after the stop
    assert "records written 20 · " in run.out and " · lost 10" in run.out
    assert "10 record(s) LOST" in run.out
    assert len(run.spy.handle_s) == 30
    counters = _pipeline(run).counters
    assert counters["events"] == 30 and counters["final_state"] == "READY"
    assert spy.sink is not None
    sink = spy.sink.counters()
    assert (sink["written"], sink["lost"], sink["pending"], sink["rejected"]) == (20, 10, 0, 0)


# --- 4 · process killed mid-run ---------------------------------------------------------------

SRC = Path(__file__).resolve().parents[1] / "src"
KILL_AFTER = 100  # rows
KILL_FRAMES = 600  # the replay queues all of them in well under a second; the rows take seconds
SETUP_COMMITS = 2  # the schema and the run row are committed before the first record


def _commits(database: Path) -> int:
    """Write transactions committed to `database` so far: SQLite's file change counter, bytes
    24–27 of the header, big-endian, one more per write transaction in the store's rollback
    journal mode (https://www.sqlite.org/fileformat.html, §1.3.3). Read as plain bytes with no
    lock: the station's writer commits back to back and holds the database lock nearly all the
    time, so a reader of the database is starved until the writer goes idle, which is when
    the run has ended. 0 until the file has its header."""
    try:
        with database.open("rb") as handle:
            header = handle.read(28)
    except FileNotFoundError:
        return 0
    return int.from_bytes(header[24:28], "big") if len(header) == 28 else 0


def test_4_a_process_killed_mid_run_leaves_evidence_that_verifies(tmp_path: Path) -> None:
    """docs/16 §7, F14 and §4: `qs station --replay` in a child process, SIGKILLed once 100
    rows are committed. The database opens and is intact; its rows are `seq` 1…m with no gap;
    every row's files exist with their hashes; verification lists any orphan, which can only
    be the files of record m+1, written before its row could commit; and the run stays open
    (`ended_utc` NULL). At most the in-memory backlog is lost.

    The kill must land before the run ends by itself: the stub decides all 600 events in well
    under a second, and the child exits once its writer has drained them. So the child is
    frozen with SIGSTOP, its progress is read while it cannot move, and it is killed frozen —
    the same as a SIGKILL at the instant it stopped, at whatever point its writer was."""
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    [first] = write_frames(shoot, 1)
    for index in range(2, KILL_FRAMES + 1):
        os.link(first, shoot / f"f{index:04d}.png")  # the same pixels, no disk used
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), *filter(None, [env.get("PYTHONPATH")])])
    root = tmp_path / "evidence"
    database = root / "qs.db"
    with (tmp_path / "stderr.txt").open("wb") as stderr:
        child = subprocess.Popen(
            [sys.executable, "-m", "qs", "station", "--config", str(config),
             "--replay", str(shoot)],
            cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=stderr,
        )
        try:
            # One more commit than the rows needed: the last may be rolled back by the kill.
            wanted = SETUP_COMMITS + KILL_AFTER + 1
            deadline = time.monotonic() + 8.0
            while _commits(database) < wanted:
                assert child.poll() is None, "the station ended before it was killed"
                assert time.monotonic() < deadline, "100 rows were not written within 8 s"
                time.sleep(0.002)
            os.kill(child.pid, signal.SIGSTOP)
            stopped = os.waitid(os.P_PID, child.pid, os.WSTOPPED | os.WEXITED | os.WNOWAIT)
            assert stopped is not None and stopped.si_code == os.CLD_STOPPED, (
                "the station ended before it was frozen"
            )
            # Frozen: nothing more is committed. The run's end is committed after every row.
            assert _commits(database) < SETUP_COMMITS + KILL_FRAMES, (
                "the station had written every row before it was frozen"
            )
            child.send_signal(signal.SIGKILL)
        finally:
            child.kill()
            child.wait(timeout=5)
    assert child.returncode == -signal.SIGKILL, (tmp_path / "stderr.txt").read_text()

    conn = sqlite3.connect(database)  # a restart's open: rolls back any half-made commit
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        seqs = [r[0] for r in conn.execute("SELECT seq FROM inspection ORDER BY id")]
        [(run_id, ended, counters)] = conn.execute(
            "SELECT run_id, ended_utc, counters FROM run"
        ).fetchall()
    finally:
        conn.close()
    m = len(seqs)
    assert m >= KILL_AFTER and seqs == list(range(1, m + 1))
    assert (ended, counters) == (None, None)
    report = verify(root)
    assert report.missing == () and report.mismatched == () and report.seq_gaps == {}
    assert report.open_runs == (run_id,)
    for orphan in report.orphans:  # the next record's image, or its temporary file
        assert Path(orphan).name.startswith(f"{m + 1:06d}_"), report.orphans
    assert len(report.orphans) <= 1
    for r in rows(database, "SELECT * FROM inspection WHERE image_status = 'stored'"):
        data = (root / r["image_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r["image_sha256"]
    Store(root).close()  # the next run opens the same evidence


# --- 5 · settings -----------------------------------------------------------------------------


class Wrapped:
    """A `FrameSource` that passes everything to `inner`; subclasses change `capture`."""

    def __init__(self, inner: FrameSource) -> None:
        self.inner = inner

    def open(self) -> None:
        self.inner.open()

    def capture(self, event: TriggerEvent) -> Frame:
        return self.inner.capture(event)

    def preview(self) -> np.ndarray | None:
        return self.inner.preview()

    def health(self) -> Health:
        return self.inner.health()

    def close(self) -> None:
        self.inner.close()


class SettingsSource(Wrapped):
    """A readable exposure reads back changed at `drift_at` (`CameraSettingsDrift`), and from
    `unreadable_from` on no setting can be read back at all."""

    def __init__(self, inner: FrameSource, drift_at: int, unreadable_from: int) -> None:
        super().__init__(inner)
        self.drift_at, self.unreadable_from = drift_at, unreadable_from

    def capture(self, event: TriggerEvent) -> Frame:
        if event.seq == self.drift_at:
            raise CameraSettingsDrift("exposure_us: set 10000, reads 9984 (injected)")
        frame = self.inner.capture(event)
        if event.seq >= self.unreadable_from:
            return replace(frame, camera=UNVERIFIED)
        return frame


def _verified_frames_csv(paths: list[Path]) -> None:
    """A `frames.csv` recording every frame as shot with the config's settings, read back."""
    write_frames_csv(paths[0].parent, [
        {"file": p.name, "exposure_us": 10000, "gain": 0, "wb_kelvin": 4500, "focus": 30,
         "verified": "true", "sha256": sha256_file(p)}
        for p in paths
    ])


def _pipeline(run: Run) -> Pipeline:
    assert run.spy.pipeline is not None
    return run.spy.pipeline


def test_5_settings_drift_at_30_then_unreadable_from_60(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/16 §7 and F4: the drifted unit is NO_VERDICT(`camera_settings`) with no image and
    `CAMERA_SETTINGS` up until a verified frame; from the first unreadable read-back on, units
    are inspected as normal, recorded with `camera_verified = 0`, and the warning stays up."""
    run = _station(
        tmp_path, capsys, monkeypatch,
        prepare=_verified_frames_csv,
        wrap_source=lambda source: SettingsSource(source, drift_at=30, unreadable_from=60),
    )
    _one_record_per_event(run, 200)
    assert run.code == 0
    found = run.by_seq()
    drifted = found[30]
    assert (drifted["label"], drifted["reason"]) == ("no_verdict", "camera_settings")
    assert drifted["image_status"] == "none" and drifted["camera_verified"] is None
    for seq in [*range(1, 30), *range(31, 60)]:
        assert found[seq]["label"] == "pass"
        assert (found[seq]["camera_verified"], found[seq]["exposure_us"]) == (1, 10000)
    for seq in range(60, 201):
        assert found[seq]["label"] == "pass" and found[seq]["image_status"] == "stored"
        assert (found[seq]["camera_verified"], found[seq]["exposure_us"]) == (0, None)
    assert run.spy.inspected == [seq for seq in range(1, 201) if seq != 30]
    up = [seq for seq in range(1, 201) if "CAMERA_SETTINGS" in run.spy.warnings[seq]]
    assert up == [30, *range(60, 201)]
    assert "CAMERA_SETTINGS" in _pipeline(run).warnings  # still up at the stop
    counters = run.counters()
    assert counters["no_verdict"] == {"camera_settings": 1}
    assert counters["faults"] == {} and counters["camera_verified"] is False
    assert counters["final_state"] == "READY"
    assert "camera verified no" in run.out


# --- 6 · reference patch darkened -------------------------------------------------------------


def test_6_reference_patch_darkened_for_events_40_to_49(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/16 §7 and F5: ten NO_VERDICT(`lighting`) with their images, never inspected;
    `LIGHTING` raised at 40 and cleared by the in-band frame of event 50."""
    dark = 40  # 40 / 255 = 0.157, far below 0.62 ± 5 %

    def darken(paths: list[Path]) -> None:
        for seq in range(40, 50):
            write_png(paths[seq - 1], textured(seq, patch_grey=dark))

    run = _station(tmp_path, capsys, monkeypatch, prepare=darken)
    _one_record_per_event(run, 200)
    assert run.code == 0
    found = run.by_seq()
    for seq in range(40, 50):
        r = found[seq]
        assert (r["label"], r["reason"], r["image_status"]) == ("no_verdict", "lighting", "stored")
        assert r["reference_patch"] == pytest.approx(dark / 255, abs=0.01)
        assert r["inspect_ms"] == 0.0
    assert all(found[seq]["label"] == "pass" for seq in [*range(1, 40), *range(50, 201)])
    assert run.spy.inspected == [*range(1, 40), *range(50, 201)]
    lit = [seq for seq in range(1, 201) if "LIGHTING" in run.spy.warnings[seq]]
    assert lit == list(range(40, 50))  # raised by 40, cleared by 50
    [checks] = rows(
        run.database,
        "SELECT COUNT(*) AS n FROM inspection_check c JOIN inspection i"
        " ON c.inspection_id = i.id WHERE i.reason = 'lighting'",
    )
    assert checks["n"] == 0  # not inspected: no check was recorded
    _stored_files_match(run)
    counters = run.counters()
    assert counters["no_verdict"] == {"lighting": 10}
    assert counters["faults"] == {} and counters["final_state"] == "READY"


# --- 7 · unit trigger during an inspection ----------------------------------------------------


class Photoelectric:
    """A fake photoelectric unit trigger over `count` events, `seq` 1…count. An event in
    `inside` fired while the previous one was being inspected: its `t0` is 1 ms after the
    previous event's, which is before that event's `handle()` returned."""

    def __init__(self, count: int, inside: set[int], clock: Callable[[], float]) -> None:
        self.count, self.inside, self.clock = count, inside, clock
        self.next = 1
        self.last_t0 = 0.0

    def poll(self) -> TriggerEvent | None:
        if self.next > self.count:
            return None
        seq = self.next
        self.next += 1
        t0 = self.last_t0 + 0.001 if seq in self.inside else self.clock()
        self.last_t0 = t0
        wall = datetime(2026, 9, 26, 9, 0, 0, tzinfo=UTC) + timedelta(seconds=seq)
        return TriggerEvent(seq, t0, wall, "photoelectric:fake")

    @property
    def exhausted(self) -> bool:
        return self.next > self.count

    def close(self) -> None:
        self.next = self.count + 1


class Captures(Wrapped):
    """Records every `seq` the station asked to capture."""

    def __init__(self, inner: FrameSource, captured: list[int]) -> None:
        super().__init__(inner)
        self.captured = captured

    def capture(self, event: TriggerEvent) -> Frame:
        self.captured.append(event.seq)
        return self.inner.capture(event)


def test_7_twenty_unit_triggers_inside_an_inspection_are_busy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/16 §7 and F9: 200 units, and 20 more triggers that fire inside an inspection, give
    220 records with `seq` 1–220 gapless; the 20 are NO_VERDICT(`busy`) with no image, and are
    never captured or inspected."""
    inside = set(range(11, 221, 11))
    assert len(inside) == 20
    clock = FakeClock(step=0.01)
    captured: list[int] = []
    run = _station(
        tmp_path, capsys, monkeypatch,
        events=220,
        clock=clock,
        wrap_trigger=lambda replay: Photoelectric(220, inside, clock),
        wrap_source=lambda source: Captures(source, captured),
    )
    _one_record_per_event(run, 220)
    assert run.code == 0
    found = run.by_seq()
    busy = {seq for seq, r in found.items() if r["reason"] == "busy"}
    assert busy == inside
    for seq in busy:
        r = found[seq]
        assert (r["label"], r["image_status"], r["camera_verified"]) == ("no_verdict", "none", None)
        assert r["capture_ms"] is None
    assert all(found[seq]["label"] == "pass" for seq in set(found) - busy)
    others = [seq for seq in range(1, 221) if seq not in inside]
    assert captured == others and run.spy.inspected == others
    counters = run.counters()
    assert counters["labels"] == {"pass": 200, "defect": 0, "no_verdict": 20}
    assert counters["no_verdict"] == {"busy": 20}
    assert counters["faults"] == {} and counters["final_state"] == "READY"


# --- 8 · no frame after t0 --------------------------------------------------------------------


class StaleSource(Wrapped):
    """A camera whose frames, for the events in `stale`, are all stamped before `t0` plus one
    exposure. Following C4 it never returns one: it waits on the fake clock for a fresh frame
    until `timeout_ms` has passed, then raises `CaptureError`. At the events in `empty` its
    buffer holds nothing; at the others, three frames stamped too early."""

    def __init__(
        self, inner: FrameSource, clock: FakeClock, *, stale: set[int], empty: set[int],
        timeout_ms: int, exposure_us: int,
    ) -> None:
        super().__init__(inner)
        self.clock, self.stale, self.empty = clock, stale, empty
        self.timeout_s, self.exposure_s = timeout_ms / 1000, exposure_us / 1e6
        self.waited_ms: dict[int, float] = {}

    def capture(self, event: TriggerEvent) -> Frame:
        if event.seq not in self.stale:
            return self.inner.capture(event)
        not_before = event.t0 + self.exposure_s
        early = (event.t0 - 0.2, event.t0 - 0.001, event.t0 + self.exposure_s / 2)
        buffered = () if event.seq in self.empty else early
        assert not any(stamp > not_before for stamp in buffered)  # none is fresh
        deadline = event.t0 + self.timeout_s
        while (now := self.clock()) < deadline:
            self.clock.advance(0.005)  # waiting for the next frame
        self.waited_ms[event.seq] = (now - event.t0) * 1000
        raise CaptureError(
            f"no frame newer than the trigger within {self.timeout_s * 1000:.0f} ms"
        )


def test_8_no_frame_after_t0_and_a_newest_frame_older_than_t0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/16 §7, F2, F3 and C4: at 50 the camera delivers nothing after the trigger, and at
    120 and 121 its newest frames are older than `t0`. Each is NO_VERDICT(`capture`) with no
    image, recorded within the capture timeout + 50 ms of fake time, and never inspected; two
    in a row are not a fault, and the next unit is inspected."""
    clock = FakeClock(step=0.01)
    camera = loaded_config(tmp_path).config.camera
    timeout_ms = camera.capture_timeout_ms
    sources: list[StaleSource] = []

    def stale(source: FrameSource) -> FrameSource:
        sources.append(StaleSource(
            source, clock, stale={50, 120, 121}, empty={50},
            timeout_ms=timeout_ms, exposure_us=camera.exposure_us,
        ))
        return sources[0]

    run = _station(tmp_path, capsys, monkeypatch, clock=clock, wrap_source=stale)
    _one_record_per_event(run, 200)
    assert run.code == 0
    found = run.by_seq()
    for seq in (50, 120, 121):
        r = found[seq]
        assert (r["label"], r["reason"], r["image_status"]) == ("no_verdict", "capture", "none")
        assert r["capture_ms"] is None and r["camera_verified"] is None
        assert sources[0].waited_ms[seq] >= timeout_ms  # it waited the whole timeout
        assert timeout_ms <= r["total_ms"] <= timeout_ms + 50
    assert run.spy.inspected == [seq for seq in range(1, 201) if seq not in (50, 120, 121)]
    assert found[51]["label"] == "pass" and found[122]["label"] == "pass"
    counters = run.counters()
    assert counters["no_verdict"] == {"capture": 3}
    assert counters["faults"] == {} and counters["final_state"] == "READY"
