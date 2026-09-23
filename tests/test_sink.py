"""`SqliteSink` on healthy storage: queue, worker, never blocks, never raises, clean stop (#59).

docs/11 §3 (the `Sink` contract, C3, C5, C6) and §7, docs/12 §3, ADR-0012. Records are built
in memory; the database is read back with plain `sqlite3`, the files with `hashlib`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

import qs
import qs.store
from evidence_records import (
    CONFIG_HASH,
    RUN,
    STARTED,
    STATION,
    appearance,
    fast_sqlite,
    files_under,
    geometry,
    make_record,
    mixed,
)
from qs.config import load_config
from qs.contracts import InspectionRecord, Label
from qs.sink import SqliteSink
from qs.store import encode_frame_jpeg

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _sink(root: Path, run_id: str = RUN, **kwargs: float) -> SqliteSink:
    return SqliteSink(
        root,
        run_id=run_id,
        station_id=STATION,
        started_utc=STARTED,
        config_hash=CONFIG_HASH,
        **kwargs,
    )


def _rows(root: Path, sql: str = "SELECT * FROM inspection ORDER BY id") -> list[sqlite3.Row]:
    conn = sqlite3.connect(root / "qs.db")
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _exact(sink: SqliteSink, given: int) -> None:
    """docs/16 §6: every record given to write() is accounted for, exactly once."""
    c = sink.counters()
    assert c["written"] + c["rejected"] + c["lost"] + c["pending"] == given, c


@pytest.fixture
def held(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """The store's file write waits on an Event: the worker is held inside a write."""
    release = threading.Event()
    original: Callable[[Path, bytes], None] = qs.store.write_file

    def waiting(path: Path, data: bytes) -> None:
        release.wait(5)
        original(path, data)

    monkeypatch.setattr(qs.store, "write_file", waiting)
    yield release
    release.set()


def test_300_mixed_records_end_to_end(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    for record in mixed(1, 300):
        sink.write(record)
    sink.finish_run(STARTED + timedelta(minutes=5), {"events": 300, "pass": 75})
    sink.close()

    rows = _rows(tmp_path)
    assert [r["seq"] for r in rows] == list(range(1, 301))  # committed in seq order
    assert {r["label"] for r in rows} == {"pass", "defect", "no_verdict"}
    assert {r["image_status"] for r in rows} == {"stored", "none"}
    for r in rows:
        if r["image_status"] == "stored":
            data = (tmp_path / r["image_path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == r["image_sha256"]
    (run,) = _rows(tmp_path, "SELECT * FROM run")
    assert run["ended_utc"] == "2026-09-25T10:05:00.000000Z"
    counters = json.loads(run["counters"])
    assert (counters["events"], counters["pass"]) == (300, 75)
    assert counters["sink"]["written"] == 300
    assert sink.pending() == 0
    assert sink.health().ok
    assert sink.counters() == {
        "written": 300, "held": 0, "images_shed": 0, "write_failures": 0,
        "rejected": 0, "lost": 0, "pending": 0,
    }  # fmt: skip


def test_write_never_blocks_on_a_held_worker(tmp_path: Path, held: threading.Event) -> None:
    sink = _sink(tmp_path)
    slowest = 0.0
    for record in mixed(1, 300):
        start = time.perf_counter()
        sink.write(record)
        slowest = max(slowest, time.perf_counter() - start)
    assert slowest < 0.020
    assert sink.pending() == 300
    held.set()
    sink.close()
    assert [r["seq"] for r in _rows(tmp_path)] == list(range(1, 301))
    _exact(sink, 300)


def test_the_queue_holds_no_pixels(tmp_path: Path, held: threading.Event) -> None:
    sink = _sink(tmp_path)
    sink.write(make_record(Label.PASS, 1))  # the worker is held inside this one
    heat = np.random.default_rng(0).random((512, 512), dtype=np.float32)
    alive = weakref.ref(heat)
    record = make_record(Label.DEFECT, 2, heatmap=True, heatmap_array=heat)
    sink.write(record)
    del record, heat
    gc.collect()
    assert alive() is None
    assert sink.pending() == 2
    held.set()
    sink.close()
    assert len(_rows(tmp_path)) == 2


def test_an_invalid_record_is_rejected_and_the_rest_written(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    for seq in range(1, 12):
        if seq == 6:
            sink.write(make_record(Label.PASS, 6, checks=(geometry(False), appearance())))
        else:
            sink.write(make_record(Label.PASS, seq))
    sink.close()
    assert [r["seq"] for r in _rows(tmp_path)] == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    assert sink.counters()["rejected"] == 1
    health = sink.health()
    assert not health.ok
    assert "seq 6" in health.detail and "C8" in health.detail
    _exact(sink, 11)


def test_a_duplicate_seq_is_rejected_at_the_worker(tmp_path: Path) -> None:
    other = encode_frame_jpeg(np.full((48, 64, 3), 99, np.uint8))
    duplicate = replace(make_record(Label.PASS, 2), image_jpeg=other)
    sink = _sink(tmp_path)
    for record in (make_record(Label.PASS, 1), make_record(Label.PASS, 2), duplicate):
        sink.write(record)
    for seq in (3, 4):
        sink.write(make_record(Label.PASS, seq))
    sink.close()
    rows = _rows(tmp_path)
    assert [r["seq"] for r in rows] == [1, 2, 3, 4]
    assert sink.counters()["rejected"] == 1
    assert "seq 2" in sink.health().detail
    for r in rows:  # the duplicate never overwrote the first record's file
        assert hashlib.sha256((tmp_path / r["image_path"]).read_bytes()).hexdigest() == (
            r["image_sha256"]
        )
    assert len(files_under(tmp_path)) == 4
    _exact(sink, 5)


def test_write_never_raises(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    sink.write(None)  # type: ignore[arg-type]
    sink.write(make_record(Label.PASS, 1, run_id="another-run"))
    sink.write("not a record")  # type: ignore[arg-type]
    sink.write(make_record(Label.PASS, 2))
    sink.close()
    sink.write(make_record(Label.PASS, 3))
    c = sink.counters()
    assert (c["written"], c["rejected"], c["lost"]) == (1, 3, 1)
    _exact(sink, 5)


def test_close_is_idempotent_and_prompt(tmp_path: Path, held: threading.Event) -> None:
    sink = _sink(tmp_path)
    for record in mixed(1, 50):
        sink.write(record)
    assert sink.pending() == 50
    held.set()
    start = time.monotonic()
    sink.close()
    assert time.monotonic() - start < 2.0
    sink.close()
    assert sink.pending() == 0
    assert len(_rows(tmp_path)) == 50


def test_from_config_lands_in_the_configured_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load_config(EXAMPLE)
    monkeypatch.chdir(tmp_path)
    sink = SqliteSink.from_config(loaded, RUN, STARTED)
    sink.write(make_record(Label.PASS, 1))
    sink.close()
    assert (tmp_path / "evidence" / "qs.db").is_file()
    (run,) = _rows(tmp_path / "evidence", "SELECT * FROM run")
    assert run["config_hash"] == loaded.config_hash
    assert run["station_id"] == loaded.config.station_id
    assert run["qs_version"] == qs.__version__


@pytest.mark.parametrize("run_id", ["../x", "a/b", "", ".hidden", "a b"])
def test_a_path_unsafe_run_id_is_refused(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="path-safe"):
        _sink(tmp_path, run_id=run_id)


def test_a_path_unsafe_station_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path-safe"):
        SqliteSink(
            tmp_path, run_id=RUN, station_id="../A", started_utc=STARTED, config_hash=CONFIG_HASH
        )


def test_the_constructor_touches_no_disk(tmp_path: Path) -> None:
    root = tmp_path / "not" / "yet"
    sink = _sink(root)
    assert sink.pending() == 0
    sink.close()  # nothing to write but the run row
    assert (root / "qs.db").is_file()


def test_two_runs_share_one_database(tmp_path: Path) -> None:
    first = _sink(tmp_path, run_id="run-1")
    for record in mixed(1, 8, run_id="run-1"):
        first.write(record)
    first.close()
    second = _sink(tmp_path, run_id="run-2")
    for record in mixed(1, 5, run_id="run-2"):
        second.write(record)
    second.close()

    assert [r["run_id"] for r in _rows(tmp_path, "SELECT run_id FROM run ORDER BY run_id")] == [
        "run-1",
        "run-2",
    ]
    counts = _rows(tmp_path, "SELECT run_id, COUNT(*) AS n FROM inspection GROUP BY run_id")
    assert {r["run_id"]: r["n"] for r in counts} == {"run-1": 8, "run-2": 5}
    days = tmp_path / "images" / STATION / "2026" / "09" / "25"
    assert sorted(p.name for p in days.iterdir()) == ["run-1", "run-2"]
    for r in _rows(tmp_path):
        if r["image_path"] is not None:
            assert f"/{r['run_id']}/" in r["image_path"]


def test_finish_run_refuses_a_sink_key(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    with pytest.raises(ValueError, match="sink"):
        sink.finish_run(STARTED, {"sink": 1})
    sink.close()


def test_a_healthy_sink_reports_ok(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    records: list[InspectionRecord] = list(mixed(1, 4))
    for record in records:
        sink.write(record)
    sink.close()
    assert sink.health().ok
    _exact(sink, 4)
