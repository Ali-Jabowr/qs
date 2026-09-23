"""The evidence store, synchronously: schema v1, one record files-first, validation (#58).

docs/12 §3 and §5, docs/16 §4, FR-41/42/43, C8. Everything is written under `tmp_path` and
read back with plain `sqlite3` and `hashlib`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import qs
import qs.store
from evidence_records import (
    CAMERA,
    CONFIG_HASH,
    RUN,
    STARTED,
    STATION,
    appearance,
    files_under,
    frame_jpeg,
    geometry,
    heatmap_png,
    make_record,
)
from qs.contracts import CheckResult, InspectionRecord, Label, Verdict
from qs.store import (
    RecordRejected,
    SchemaVersionError,
    Store,
    encode_frame_jpeg,
    inspection_row,
    run_row,
    validate_record,
)

TABLES = {"run", "inspection", "inspection_check", "override", "schema_version"}


def _db(root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(root / "qs.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _store(root: Path) -> Store:
    store = Store(root)
    store.begin_run(run_row(RUN, STATION, STARTED, CONFIG_HASH))
    return store


def _write(store: Store, record: InspectionRecord) -> None:
    store.write_record(inspection_row(record, RUN, STATION))


def _one(root: Path, seq: int) -> sqlite3.Row:
    with _db(root) as conn:
        row = conn.execute("SELECT * FROM inspection WHERE seq = ?", (seq,)).fetchone()
    assert row is not None
    return row


def _checks(root: Path, inspection_id: int) -> list[sqlite3.Row]:
    with _db(root) as conn:
        return conn.execute(
            "SELECT * FROM inspection_check WHERE inspection_id = ? ORDER BY rowid",
            (inspection_id,),
        ).fetchall()


# --- schema -------------------------------------------------------------------------------


def test_a_fresh_root_gets_schema_v1(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    Store(root).close()
    with _db(root) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == TABLES
        assert [tuple(r) for r in conn.execute("SELECT version FROM schema_version")] == [(1,)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        triggers = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
        assert len(triggers) == 6
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            )
        }
        assert indexes == {"idx_inspection_batch", "idx_inspection_label", "idx_check_failed"}


def test_reopening_is_idempotent(tmp_path: Path) -> None:
    Store(tmp_path).close()
    Store(tmp_path).close()
    with _db(tmp_path) as conn:
        assert [tuple(r) for r in conn.execute("SELECT version FROM schema_version")] == [(1,)]


def test_another_schema_version_is_refused_by_name(tmp_path: Path) -> None:
    Store(tmp_path).close()
    with _db(tmp_path) as conn:
        conn.execute("UPDATE schema_version SET version = 2")
    with pytest.raises(SchemaVersionError, match="schema_version 2"):
        Store(tmp_path)


def test_foreign_keys_are_on_in_the_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        _write(store, make_record(Label.PASS, 1))  # no run row yet
    store.close()


# --- the database refuses what the schema forbids ----------------------------------------------

_INSERT = (
    "INSERT INTO inspection (run_id, seq, batch_id, ts_utc, label, reason, model_id,"
    " threshold_id, config_hash, image_status, image_path, image_sha256, camera_verified)"
    " VALUES (:run, :seq, :run, '2026-09-25T10:00:00.000000Z', :label, :reason, 'm', 't', 'h',"
    " :status, :path, :sha, :verified)"
)
_GOOD = {
    "run": RUN,
    "seq": 1,
    "label": "pass",
    "reason": None,
    "status": "stored",
    "path": "images/x.jpg",
    "sha": "ab" * 32,
    "verified": 1,
}


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    """A root with one run, one inspection, one check and one override."""
    Store(tmp_path).begin_run(run_row(RUN, STATION, STARTED, CONFIG_HASH))
    with _db(tmp_path) as conn:
        conn.execute(_INSERT, _GOOD)
        conn.execute(
            "INSERT INTO inspection_check (inspection_id, name, passed) VALUES (1, 'geometry', 1)"
        )
        conn.execute(
            "INSERT INTO override (inspection_id, ts_utc, operator, new_label)"
            " VALUES (1, '2026-09-25T10:05:00.000000Z', 'op', 'defect')"
        )
    return tmp_path


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("UPDATE inspection SET label = 'defect'", "inspection is append-only"),
        ("DELETE FROM inspection", "inspection is append-only"),
        ("UPDATE inspection_check SET passed = 0", "inspection_check is append-only"),
        ("DELETE FROM inspection_check", "inspection_check is append-only"),
        ("UPDATE override SET new_label = 'pass'", "override is append-only"),
        ("DELETE FROM override", "override is append-only"),
    ],
)
def test_append_only_is_a_property_of_the_file(seeded: Path, statement: str, message: str) -> None:
    with _db(seeded) as conn, pytest.raises(sqlite3.IntegrityError, match=message):
        conn.execute(statement)


@pytest.mark.parametrize(
    "change",
    [
        {"label": "maybe", "seq": 2},
        {"label": "no_verdict", "reason": None, "seq": 2},
        {"sha": None, "seq": 2},
        {"label": "pass", "status": "none", "path": None, "sha": None, "seq": 2},
        {"status": "shed", "path": None, "sha": None, "verified": None, "seq": 2},
        {"seq": 1},  # a duplicate (run_id, seq)
        {"run": "no-such-run", "seq": 2},  # the FK
    ],
    ids=["label", "nv-reason", "stored-hash", "none-on-pass", "shed-camera", "dup-seq", "fk"],
)
def test_the_schema_refuses_a_bad_inspection(seeded: Path, change: dict[str, object]) -> None:
    with _db(seeded) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(_INSERT, {**_GOOD, **change})


# --- one record, files first --------------------------------------------------------------------


def test_a_pass_round_trips_column_by_column(tmp_path: Path) -> None:
    record = make_record(Label.PASS, 7)
    store = _store(tmp_path)
    _write(store, record)
    store.close()

    row = _one(tmp_path, 7)
    verdict, camera = record.verdict, record.camera
    assert camera is not None and verdict.unit_box is not None
    expected = {
        "run_id": RUN,
        "seq": 7,
        "batch_id": RUN,
        "ts_utc": "2026-09-25T10:00:07.000250Z",
        "label": "pass",
        "reason": None,
        "model_id": verdict.model_id,
        "threshold_id": verdict.threshold_id,
        "config_hash": verdict.config_hash,
        "image_status": "stored",
        "image_path": f"images/{STATION}/2026/09/25/{RUN}/000007_pass.jpg",
        "image_sha256": hashlib.sha256(frame_jpeg()).hexdigest(),
        "heatmap_path": None,
        "unit_x": verdict.unit_box[0],
        "unit_y": verdict.unit_box[1],
        "unit_w": verdict.unit_box[2],
        "unit_h": verdict.unit_box[3],
        "exposure_us": camera.exposure_us,
        "gain": camera.gain,
        "wb_kelvin": camera.wb_kelvin,
        "focus": camera.focus,
        "camera_verified": 1,
        "reference_patch": record.reference_patch,
        "capture_ms": record.capture_ms,
        "inspect_ms": verdict.inspect_ms,
        "total_ms": record.total_ms,
    }
    assert {key: row[key] for key in expected} == expected
    assert set(row.keys()) == set(expected) | {"id"}

    image = tmp_path / row["image_path"]
    assert image.name == "000007_pass.jpg"
    assert image.read_bytes() == frame_jpeg()
    assert hashlib.sha256(image.read_bytes()).hexdigest() == row["image_sha256"]

    checks = _checks(tmp_path, row["id"])
    assert [c["name"] for c in checks] == ["geometry", "appearance"]
    for stored, given in zip(checks, verdict.checks, strict=True):
        assert stored["passed"] == 1
        assert stored["value"] == given.value
        assert stored["limit_value"] == given.limit
        assert stored["failed_on"] is None
        assert list(json.loads(stored["measures"]).items()) == list(given.measures)


def test_a_defect_stores_its_heatmap_and_its_failure(tmp_path: Path) -> None:
    record = make_record(Label.DEFECT, 3, heatmap=True)
    store = _store(tmp_path)
    _write(store, record)
    store.close()

    row = _one(tmp_path, 3)
    assert row["label"] == "defect"
    assert row["image_path"].endswith("/000003_defect.jpg")
    assert row["heatmap_path"] == f"heatmaps/{STATION}/2026/09/25/{RUN}/000003.png"
    assert (tmp_path / row["heatmap_path"]).read_bytes() == heatmap_png()
    failed = _checks(tmp_path, row["id"])[1]
    assert (failed["name"], failed["passed"]) == ("appearance", 0)
    assert (failed["failed_on"], failed["value"], failed["limit_value"]) == (
        "score>threshold",
        0.93,
        0.62,
    )


def test_a_no_verdict_without_a_frame_writes_no_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write(store, make_record(Label.NO_VERDICT, 1, image=False, reason="camera"))
    _write(store, make_record(Label.NO_VERDICT, 2, reason="lighting"))
    store.close()

    none = _one(tmp_path, 1)
    assert (none["label"], none["reason"], none["image_status"]) == ("no_verdict", "camera", "none")
    for column in (
        "image_path", "image_sha256", "heatmap_path", "exposure_us", "gain", "wb_kelvin",
        "focus", "camera_verified", "unit_x", "reference_patch", "capture_ms",
    ):  # fmt: skip
        assert none[column] is None, column
    assert _checks(tmp_path, none["id"]) == []

    lit = _one(tmp_path, 2)
    assert (lit["reason"], lit["image_status"], lit["camera_verified"]) == ("lighting", "stored", 1)
    assert files_under(tmp_path) == [f"images/{STATION}/2026/09/25/{RUN}/000002_no_verdict.jpg"]


def test_the_timestamp_is_converted_to_utc(tmp_path: Path) -> None:
    moscow = timezone(timedelta(hours=3))
    wall = datetime(2026, 9, 26, 1, 30, 0, 5, tzinfo=moscow)
    store = _store(tmp_path)
    _write(store, make_record(Label.PASS, 1, wall=wall))
    store.close()
    row = _one(tmp_path, 1)
    assert row["ts_utc"] == "2026-09-25T22:30:00.000005Z"
    assert "/2026/09/25/" in row["image_path"]


def test_an_unverified_camera_is_stored_as_zero(tmp_path: Path) -> None:
    unverified = replace(CAMERA, verified=False, exposure_us=None)
    store = _store(tmp_path)
    _write(store, make_record(Label.PASS, 1, camera=unverified))
    store.close()
    row = _one(tmp_path, 1)
    assert (row["camera_verified"], row["exposure_us"], row["gain"]) == (0, None, 0)


# --- validation ------------------------------------------------------------------------------


def _without_checks(label: Label) -> InspectionRecord:
    record = make_record(label, 1)
    object.__setattr__(record.verdict, "checks", ())  # Verdict itself refuses this
    return record


def _with(record: InspectionRecord, **fields: object) -> InspectionRecord:
    return replace(record, **fields)  # type: ignore[arg-type]


_NAN = CheckResult("appearance", True, float("nan"), 0.62)
_NAN_MEASURE = replace(geometry(True), measures=(("area_px", float("inf")),))

REFUSED = [
    ("pass-failed", lambda: make_record(Label.PASS, 1, checks=(geometry(False), appearance())),
     "C8: a pass"),
    ("defect-all-passed",
     lambda: make_record(Label.DEFECT, 1, checks=(geometry(), appearance())), "C8: a defect"),
    ("pass-no-checks", lambda: _without_checks(Label.PASS), "C8: a pass with no checks"),
    ("defect-no-checks", lambda: _without_checks(Label.DEFECT), "C8: a defect with no checks"),
    ("pass-no-image", lambda: make_record(Label.PASS, 1, image=False, camera=CAMERA),
     "a pass must carry its image"),
    ("defect-no-image", lambda: make_record(Label.DEFECT, 1, image=False, camera=CAMERA),
     "a defect must carry its image"),
    ("nv-checks", lambda: make_record(Label.NO_VERDICT, 1, checks=(geometry(),)),
     "C8: a no_verdict carries no checks"),
    ("nv-reason", lambda: make_record(Label.NO_VERDICT, 1, reason="tired"), "reason 'tired'"),
    ("image-camera", lambda: make_record(Label.PASS, 1, camera=None), "an image needs its camera"),
    ("heatmap-image",
     lambda: make_record(Label.NO_VERDICT, 1, image=False, heatmap=True, camera=CAMERA),
     "a heatmap needs an image"),
    ("jpeg-magic", lambda: _with(make_record(Label.PASS, 1), image_jpeg=heatmap_png()),
     "not a JPEG"),
    ("png-magic",
     lambda: _with(make_record(Label.DEFECT, 1, heatmap=True), heatmap_png=frame_jpeg()),
     "not a PNG"),
    ("nan-value", lambda: make_record(Label.PASS, 1, checks=(geometry(), _NAN)),
     "check 'appearance' value is not finite"),
    ("nan-measure", lambda: make_record(Label.PASS, 1, checks=(_NAN_MEASURE, appearance())),
     "check 'geometry' measure 'area_px' is not finite"),
    ("dup-names", lambda: make_record(Label.PASS, 1, checks=(geometry(), geometry())),
     "two checks share the name 'geometry'"),
    ("naive-wall", lambda: make_record(Label.PASS, 1, wall=datetime(2026, 9, 25, 10)),
     "wall time is naive"),
    ("other-run", lambda: make_record(Label.PASS, 1, run_id="another-run"),
     "run_id 'another-run'"),
    ("negative-seq", lambda: make_record(Label.PASS, -1), "seq -1"),
]


@pytest.mark.parametrize(
    ("build", "message"), [r[1:] for r in REFUSED], ids=[r[0] for r in REFUSED]
)
def test_validation_refuses_by_rule(build: object, message: str) -> None:
    record = build()  # type: ignore[operator]
    with pytest.raises(RecordRejected, match=message) as refused:
        validate_record(record, RUN)
    assert isinstance(refused.value, ValueError)
    with pytest.raises(RecordRejected):
        inspection_row(record, RUN, STATION)


def test_validation_refuses_what_is_not_a_record() -> None:
    with pytest.raises(RecordRejected, match="not an InspectionRecord"):
        validate_record(None, RUN)  # type: ignore[arg-type]


def test_every_valid_shape_passes_validation() -> None:
    for record in (
        make_record(Label.PASS, 0),
        make_record(Label.DEFECT, 1, heatmap=True),
        make_record(Label.NO_VERDICT, 2),
        make_record(Label.NO_VERDICT, 3, image=False),
        make_record(Label.NO_VERDICT, 4, image=False, camera=CAMERA, reason="camera_settings"),
    ):
        validate_record(record, RUN)


def test_the_row_holds_no_array() -> None:
    heat = np.zeros((8, 8), np.float32)
    row = inspection_row(make_record(Label.DEFECT, 1, heatmap_array=heat), RUN, STATION)
    assert not any(isinstance(value, (np.ndarray, Verdict)) for value in vars_of(row))


def vars_of(row: object) -> list[object]:
    return [getattr(row, name) for name in row.__slots__]  # type: ignore[attr-defined]


# --- atomicity ---------------------------------------------------------------------------------


def test_a_failed_insert_leaves_the_files_and_a_retry_replaces_them(tmp_path: Path) -> None:
    store = Store(tmp_path)
    row = inspection_row(make_record(Label.DEFECT, 5, heatmap=True), RUN, STATION)
    with pytest.raises(sqlite3.IntegrityError):
        store.write_record(row)  # no run row: the FK refuses the insert
    written = files_under(tmp_path)
    assert len(written) == 2
    with _db(tmp_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM inspection").fetchone()[0] == 0

    store.begin_run(run_row(RUN, STATION, STARTED, CONFIG_HASH))
    store.write_record(row)
    store.close()
    assert files_under(tmp_path) == written  # the same two files, no temp, no duplicate
    stored = _one(tmp_path, 5)
    assert hashlib.sha256((tmp_path / stored["image_path"]).read_bytes()).hexdigest() == (
        stored["image_sha256"]
    )


def test_a_bad_check_row_rolls_back_the_whole_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    row = inspection_row(make_record(Label.PASS, 1), RUN, STATION)
    bad_check = replace(row.checks[1], passed=2)  # violates CHECK (passed IN (0, 1))
    with pytest.raises(sqlite3.IntegrityError):
        store.write_record(replace(row, checks=(row.checks[0], bad_check)))
    _write(store, make_record(Label.PASS, 2))  # the connection is still usable
    store.close()
    with _db(tmp_path) as conn:
        assert [tuple(r) for r in conn.execute("SELECT seq FROM inspection")] == [(2,)]
        assert conn.execute("SELECT COUNT(*) FROM inspection_check").fetchone()[0] == 2


# --- encoding and the run row ---------------------------------------------------------------


def test_encode_frame_jpeg_is_a_q85_jpeg_of_the_frame() -> None:
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    before = pixels.copy()
    data = encode_frame_jpeg(pixels)
    assert data[:3] == b"\xff\xd8\xff"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == pixels.shape
    assert np.array_equal(pixels, before)


def test_encode_frame_jpeg_raises_on_what_it_cannot_encode() -> None:
    with pytest.raises(ValueError):
        encode_frame_jpeg(np.zeros((0, 0, 3), np.uint8))


def test_end_run_sets_ended_and_counters(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_run(run_row("never-ended", STATION, STARTED, CONFIG_HASH))
    store.end_run(RUN, STARTED + timedelta(hours=1), json.dumps({"events": 3}))
    store.close()
    with _db(tmp_path) as conn:
        runs = {r["run_id"]: r for r in conn.execute("SELECT * FROM run")}
    assert runs[RUN]["ended_utc"] == "2026-09-25T11:00:00.000000Z"
    assert json.loads(runs[RUN]["counters"]) == {"events": 3}
    assert runs[RUN]["started_utc"] == "2026-09-25T10:00:00.000000Z"
    assert (runs[RUN]["station_id"], runs[RUN]["config_hash"]) == (STATION, CONFIG_HASH)
    assert runs[RUN]["qs_version"] == qs.__version__
    assert runs["never-ended"]["ended_utc"] is None
    assert runs["never-ended"]["counters"] is None


def test_run_row_refuses_a_naive_start() -> None:
    with pytest.raises(ValueError, match="naive"):
        run_row(RUN, STATION, datetime(2026, 9, 25), CONFIG_HASH)


def test_store_module_imports_no_network() -> None:
    source = Path(qs.store.__file__).read_text()
    for name in ("socket", "http", "urllib", "requests"):
        assert f"import {name}" not in source and f"from {name}" not in source


def test_timestamps_use_microseconds_and_z() -> None:
    assert qs.store.utc_text(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == (
        "2026-01-02T03:04:05.000000Z"
    )
