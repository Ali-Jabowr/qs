"""Evidence verification: one query and a walk — orphans, hashes, gaps after a kill (#61).

docs/12 §3, docs/16 F14, §4 and §7 ("Process killed mid-run: database opens; every row's files
exist; orphans listed by verification"), docs/17 T2 (tamper-evident). Everything is under
`tmp_path`; a child Python process stands in for a kill.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

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
from qs.sink import SqliteSink
from qs.store import Store, inspection_row, run_row, verify

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"

needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores chmod: a read-only directory cannot be simulated"
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _clean_run(root: Path, count: int = 50) -> None:
    sink = SqliteSink(
        root, run_id=RUN, station_id=STATION, started_utc=STARTED, config_hash=CONFIG_HASH
    )
    for record in mixed(1, count):
        sink.write(record)
    sink.finish_run(STARTED + timedelta(minutes=1), {"events": count})
    sink.close()
    assert sink.counters()["written"] == count


def _db_sha(root: Path) -> str:
    return hashlib.sha256((root / "qs.db").read_bytes()).hexdigest()


def _stored(root: Path) -> list[str]:
    conn = sqlite3.connect(root / "qs.db")
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT image_path FROM inspection WHERE image_status = 'stored' ORDER BY seq"
            )
        ]
    finally:
        conn.close()


def test_a_clean_run_verifies(tmp_path: Path) -> None:
    _clean_run(tmp_path)
    report = verify(tmp_path)
    assert report.ok
    assert report.missing == () and report.mismatched == () and report.orphans == ()
    assert report.seq_gaps == {} and report.open_runs == ()


_CHILD = textwrap.dedent(
    """
    import os, sys, time
    from datetime import UTC, datetime
    from pathlib import Path

    sys.path[:0] = [sys.argv[2], sys.argv[3]]
    import qs.store
    from evidence_records import CONFIG_HASH, RUN, STARTED, STATION, make_record, mixed
    from qs.contracts import Label
    from qs.sink import SqliteSink

    original = qs.store.write_file

    def killed(path, data):
        original(path, data)
        if path.name.startswith("000011_"):
            os._exit(1)  # after record 11's image is written, before its row commits

    qs.store.write_file = killed
    sink = SqliteSink(
        Path(sys.argv[1]), run_id=RUN, station_id=STATION, started_utc=STARTED,
        config_hash=CONFIG_HASH,
    )
    for record in mixed(1, 10):
        sink.write(record)
    sink.write(make_record(Label.PASS, 11))
    time.sleep(5)
    os._exit(0)  # never reached: the worker kills the process first
    """
)


def test_a_process_killed_mid_record_leaves_one_orphan(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(root), str(SRC), str(TESTS)],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 1, child.stderr.decode()

    conn = sqlite3.connect(root / "qs.db")
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        rows = conn.execute(
            "SELECT seq, image_path, image_sha256 FROM inspection ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert [seq for seq, _, _ in rows] == list(range(1, 11))
    for _, path, sha in rows:
        if path is not None:
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha

    report = verify(root)
    eleventh = f"images/{STATION}/2026/09/25/{RUN}/000011_pass.jpg"
    assert report.orphans == (eleventh,)
    assert report.missing == () and report.mismatched == () and report.seq_gaps == {}
    assert report.open_runs == (RUN,)
    assert not report.ok


def test_tampering_is_evident(tmp_path: Path) -> None:
    _clean_run(tmp_path)
    flipped, deleted = _stored(tmp_path)[3], _stored(tmp_path)[7]
    data = bytearray((tmp_path / flipped).read_bytes())
    data[len(data) // 2] ^= 0xFF
    (tmp_path / flipped).write_bytes(bytes(data))
    (tmp_path / deleted).unlink()
    stray = f"images/{STATION}/2026/09/25/{RUN}/stray.jpg"
    (tmp_path / stray).write_bytes(b"\xff\xd8\xff not ours")

    report = verify(tmp_path)
    assert report.mismatched == (flipped,)
    assert report.missing == (deleted,)
    assert report.orphans == (stray,)
    assert not report.ok


def test_a_temp_file_left_by_a_crash_is_an_orphan(tmp_path: Path) -> None:
    _clean_run(tmp_path, 4)
    temp = f"heatmaps/{STATION}/2026/09/25/{RUN}/000099.png.tmp"
    (tmp_path / temp).write_bytes(b"half")
    assert verify(tmp_path).orphans == (temp,)


def test_a_missing_heatmap_is_reported(tmp_path: Path) -> None:
    _clean_run(tmp_path, 8)
    heatmaps = [p for p in files_under(tmp_path) if p.startswith("heatmaps/")]
    (tmp_path / heatmaps[0]).unlink()
    assert verify(tmp_path).missing == (heatmaps[0],)


def test_a_seq_gap_is_reported_per_run(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.begin_run(run_row(RUN, STATION, STARTED, CONFIG_HASH))
    for seq in (1, 2, 4, 5):
        store.write_record(inspection_row(make_record(Label.PASS, seq), RUN, STATION))
    store.end_run(RUN, STARTED, "{}")
    store.close()
    report = verify(tmp_path)
    assert report.seq_gaps == {RUN: [3]}
    assert not report.ok


def test_shed_and_none_rows_are_not_missing(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.begin_run(run_row(RUN, STATION, STARTED, CONFIG_HASH))
    shed = inspection_row(make_record(Label.PASS, 1), RUN, STATION)
    store.write_record(replace(shed, image_jpeg=None, heatmap_png=None, shed=True))
    store.write_record(
        inspection_row(make_record(Label.NO_VERDICT, 2, image=False), RUN, STATION)
    )
    store.end_run(RUN, STARTED, "{}")
    store.close()
    report = verify(tmp_path)
    assert report.missing == () and report.orphans == ()
    assert report.ok


def test_a_live_run_may_be_open(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.begin_run(run_row(RUN, STATION, STARTED, CONFIG_HASH))
    store.close()
    assert not verify(tmp_path).ok
    live = verify(tmp_path, live_runs={RUN})
    assert live.open_runs == (RUN,) and live.ok


@needs_non_root
def test_verify_modifies_nothing_and_runs_on_a_read_only_root(tmp_path: Path) -> None:
    _clean_run(tmp_path, 12)
    before = _db_sha(tmp_path)
    tmp_path.chmod(0o555)
    try:
        report = verify(tmp_path)
    finally:
        tmp_path.chmod(0o755)
    assert report.ok
    assert _db_sha(tmp_path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["heatmaps", "images", "qs.db"]


def test_a_missing_or_unreadable_database_names_its_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="qs.db"):
        verify(tmp_path)
    (tmp_path / "qs.db").write_bytes(b"not a database at all" * 100)
    with pytest.raises(ValueError, match=re.escape(str(tmp_path / "qs.db"))):
        verify(tmp_path)


def test_1000_rows_verify_in_under_2_s(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.begin_run(run_row(RUN, STATION, STARTED, CONFIG_HASH))
    for seq in range(1, 1001):
        store.write_record(inspection_row(make_record(Label.PASS, seq), RUN, STATION))
    store.end_run(RUN, STARTED, "{}")
    store.close()
    start = time.monotonic()
    report = verify(tmp_path)
    assert time.monotonic() - start < 2.0
    assert report.ok
