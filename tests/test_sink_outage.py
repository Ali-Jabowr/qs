"""`SqliteSink` under a storage outage: hold rows, shed images, stop in `stop_wait_s` (#60).

ADR-0012; docs/16 F11, F13, §4, §6 and §7. Faults are injected under `tmp_path`: a `chmod`, a
monkeypatched file write that raises `ENOSPC`, one that sleeps. No real full disk.
"""

from __future__ import annotations

import errno
import hashlib
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import qs.store
from evidence_records import (
    CONFIG_HASH,
    RUN,
    STARTED,
    STATION,
    fast_sqlite,
    files_under,
    make_record,
    mixed,
)
from qs.contracts import Label
from qs.sink import HIGH_WATER, SqliteSink
from qs.store import Store

RETRY_S = 0.02
STOP_WAIT_S = 0.5

needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores chmod: a read-only directory cannot be simulated"
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _sink(root: Path, stop_wait_s: float = STOP_WAIT_S) -> SqliteSink:
    return SqliteSink(
        root,
        run_id=RUN,
        station_id=STATION,
        started_utc=STARTED,
        config_hash=CONFIG_HASH,
        retry_s=RETRY_S,
        stop_wait_s=stop_wait_s,
    )


def _until(condition: Callable[[], bool], timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _rows(root: Path, sql: str = "SELECT * FROM inspection ORDER BY id") -> list[sqlite3.Row]:
    conn = sqlite3.connect(root / "qs.db")
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _exact(sink: SqliteSink, given: int) -> None:
    c = sink.counters()
    assert c["written"] + c["rejected"] + c["lost"] + c["pending"] == given, c


@contextmanager
def _read_only(path: Path) -> Iterator[Callable[[], None]]:
    """`path` read-only until the yielded restore() is called, or the block ends."""
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


def _timed_writes(sink: SqliteSink, records: object) -> float:
    slowest = 0.0
    for record in records:  # type: ignore[attr-defined]
        start = time.perf_counter()
        sink.write(record)
        slowest = max(slowest, time.perf_counter() - start)
    return slowest


def _stored_hashes_match(root: Path, rows: list[sqlite3.Row]) -> None:
    for r in rows:
        if r["image_status"] == "stored":
            data = (root / r["image_path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == r["image_sha256"]


@needs_non_root
def test_read_only_root_for_100_events(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    for record in mixed(1, 20):
        sink.write(record)
    _until(lambda: sink.pending() == 0)
    with _read_only(tmp_path) as restore:
        assert _timed_writes(sink, mixed(21, 100)) < 0.020
        _until(lambda: sink.counters()["write_failures"] >= 2)
        health = sink.health()
        assert not health.ok
        assert "held" in health.detail
        assert sink.pending() >= 100
        restore()
        _until(lambda: sink.pending() == 0)
    rows = _rows(tmp_path)
    assert [r["seq"] for r in rows] == list(range(1, 121))
    _stored_hashes_match(tmp_path, rows)
    c = sink.counters()
    assert c["written"] == 120 and c["held"] >= 100 and c["write_failures"] >= 1
    assert c["images_shed"] == 0 and c["lost"] == 0
    assert sink.health().ok
    sink.close()
    _exact(sink, 120)


def test_enospc_holds_every_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = threading.Event()
    full.set()
    original = qs.store.write_file

    def no_space(path: Path, data: bytes) -> None:
        if full.is_set():
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(path))
        original(path, data)

    monkeypatch.setattr(qs.store, "write_file", no_space)
    sink = _sink(tmp_path)
    assert _timed_writes(sink, mixed(1, 50)) < 0.020
    _until(lambda: sink.counters()["write_failures"] >= 2)
    health = sink.health()
    assert not health.ok and "No space left" in health.detail and "held" in health.detail
    full.clear()
    _until(lambda: sink.pending() == 0)
    sink.close()
    assert [r["seq"] for r in _rows(tmp_path)] == list(range(1, 51))
    c = sink.counters()
    assert c["written"] == 50 and c["held"] == 50 and c["write_failures"] >= 2
    assert sink.health().ok
    _exact(sink, 50)


@needs_non_root
def test_images_are_shed_pass_first_then_no_verdict_then_defect(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    with _read_only(root) as restore:
        sink = _sink(root)  # the run row is the stuck head; no record is mid-attempt
        _until(lambda: sink.counters()["write_failures"] >= 1)
        for seq in range(1, 151):
            sink.write(make_record(Label.PASS, seq))
        for seq in range(151, 211):
            sink.write(make_record(Label.NO_VERDICT, seq))
        for seq in range(211, 271):
            sink.write(make_record(Label.DEFECT, seq, heatmap=True))
        assert sink.counters()["images_shed"] == 270 - HIGH_WATER == 70
        assert not sink.health().ok and "high-water" in sink.health().detail
        for seq in range(271, 371):
            sink.write(make_record(Label.DEFECT, seq, heatmap=True))
        assert sink.counters()["images_shed"] == 170
        restore()
        _until(lambda: sink.pending() == 0)
    sink.close()

    rows = _rows(root)
    assert [r["seq"] for r in rows] == list(range(1, 371))
    shed = {r["seq"] for r in rows if r["image_status"] == "shed"}
    assert shed == set(range(1, 151)) | set(range(151, 171))  # every PASS, the 20 oldest NV
    for r in rows:
        if r["seq"] in shed:
            assert (r["image_path"], r["image_sha256"], r["heatmap_path"]) == (None, None, None)
            assert r["camera_verified"] == 1
        else:
            assert r["image_status"] == "stored"
    _stored_hashes_match(root, rows)
    kept = {r["image_path"] for r in rows if r["image_path"]} | {
        r["heatmap_path"] for r in rows if r["heatmap_path"]
    }
    assert set(files_under(root)) == kept  # no file for any shed image
    assert sink.health().ok
    _exact(sink, 370)


@needs_non_root
def test_below_the_high_water_nothing_is_shed(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    with _read_only(root) as restore:
        sink = _sink(root)
        for seq in range(1, HIGH_WATER + 1):
            sink.write(make_record(Label.PASS, seq))
        _until(lambda: sink.counters()["write_failures"] >= 10)
        assert sink.counters()["images_shed"] == 0
        health = sink.health()
        assert not health.ok
        assert "storage failing" in health.detail and "high-water" not in health.detail
        restore()
        _until(lambda: sink.pending() == 0)
    sink.close()
    rows = _rows(root)
    assert len(rows) == HIGH_WATER and {r["image_status"] for r in rows} == {"stored"}
    _exact(sink, HIGH_WATER)


def test_a_slow_disk_sheds_down_to_the_high_water(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, first = threading.Event(), threading.Event()
    original = qs.store.write_file
    calls = 0

    def slow(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            first.wait(5)  # the first record is mid-attempt while the backlog builds
        else:
            time.sleep(0.002)
        original(path, data)

    monkeypatch.setattr(qs.store, "write_file", slow)
    sink = _sink(tmp_path)
    sink.write(make_record(Label.PASS, 1))
    assert entered.wait(1)
    for seq in range(2, 251):
        sink.write(make_record(Label.PASS, seq))
    assert sink.counters()["images_shed"] == 50
    assert not sink.health().ok and "high-water" in sink.health().detail
    first.set()
    _until(lambda: sink.pending() == 0)
    assert sink.health().ok
    sink.close()
    rows = _rows(tmp_path)
    shed = [r["seq"] for r in rows if r["image_status"] == "shed"]
    assert shed == list(range(2, 52))  # the oldest PASS images, not the one mid-attempt
    assert sink.counters()["write_failures"] == 0
    _exact(sink, 250)


@needs_non_root
def test_a_stop_during_an_outage_ends_and_counts_the_lost(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    for seq in range(1, 6):
        sink.write(make_record(Label.PASS, seq))
    _until(lambda: sink.pending() == 0)
    with _read_only(tmp_path):
        for seq in range(6, 36):
            sink.write(make_record(Label.PASS, seq))
        sink.finish_run(STARTED, {"events": 35})
        _until(lambda: sink.counters()["write_failures"] >= 1)
        start = time.monotonic()
        sink.close()
        elapsed = time.monotonic() - start
    assert elapsed < STOP_WAIT_S + 0.5
    assert sink.counters()["lost"] == 30
    assert not sink.health().ok and "lost 30" in sink.health().detail
    (run,) = _rows(tmp_path, "SELECT * FROM run")
    assert run["ended_utc"] is None
    assert [r["seq"] for r in _rows(tmp_path)] == [1, 2, 3, 4, 5]
    sink.write(make_record(Label.PASS, 36))  # after close: lost, never raised
    _exact(sink, 36)


def test_a_stop_on_a_slow_disk_drains_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = qs.store.write_file

    def slow(path: Path, data: bytes) -> None:
        time.sleep(0.04)
        original(path, data)

    monkeypatch.setattr(qs.store, "write_file", slow)
    sink = _sink(tmp_path, stop_wait_s=0.3)
    for seq in range(1, 31):
        sink.write(make_record(Label.PASS, seq))
    start = time.monotonic()
    sink.close()
    assert time.monotonic() - start > 0.3  # longer than stop_wait_s, and still drained
    assert sink.counters()["lost"] == 0
    assert len(_rows(tmp_path)) == 30
    _exact(sink, 30)


@needs_non_root
def test_an_outage_from_the_start(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    with _read_only(parent) as restore:
        sink = _sink(parent / "evidence")  # cannot be created: no exception here
        _until(lambda: sink.counters()["write_failures"] >= 1)
        assert not sink.health().ok
        for record in mixed(1, 5):
            sink.write(record)
        restore()
        _until(lambda: sink.pending() == 0)
    sink.close()
    root = parent / "evidence"
    assert len(_rows(root, "SELECT * FROM run")) == 1
    assert [r["seq"] for r in _rows(root)] == [1, 2, 3, 4, 5]
    assert sink.health().ok
    _exact(sink, 5)


def test_an_unknown_schema_is_never_written_into(tmp_path: Path) -> None:
    Store(tmp_path).close()
    conn = sqlite3.connect(tmp_path / "qs.db")
    with conn:
        conn.execute("UPDATE schema_version SET version = 2")
    conn.close()
    before = hashlib.sha256((tmp_path / "qs.db").read_bytes()).hexdigest()

    sink = _sink(tmp_path, stop_wait_s=0.3)
    for record in mixed(1, 5):
        sink.write(record)
    _until(lambda: sink.counters()["write_failures"] >= 1)
    assert "schema_version 2" in sink.health().detail
    start = time.monotonic()
    sink.close()
    assert time.monotonic() - start < 0.3 + 0.5
    assert sink.counters()["lost"] == 5
    assert hashlib.sha256((tmp_path / "qs.db").read_bytes()).hexdigest() == before
    assert files_under(tmp_path) == []
    _exact(sink, 5)
