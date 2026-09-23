"""The evidence store: schema v1, one record written files-first, and record validation.

docs/12 §3 and §5, docs/16 §4, ADR-0012. SQLite from the standard library; images are files
under the evidence root with their SHA-256 in the row. `inspection`, `inspection_check` and
`override` are append-only by trigger, so FR-42 is a property of the file.

A `Store` owns one connection and is used by one thread at a time: `SqliteSink`'s worker, or a
tool or test directly. It raises; it does not decide. `OSError` and `sqlite3` errors other than
`IntegrityError` are storage failures for the caller to hold and retry (ADR-0012).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

import qs
from qs.contracts import REASONS, InspectionRecord, Label

SCHEMA_VERSION = 1
DB_NAME = "qs.db"
JPEG_QUALITY = 85  # ADR-0012
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PATH_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Applied to every connection the store opens. SQLite's default journal and `synchronous`.
PRAGMAS: tuple[str, ...] = ("PRAGMA foreign_keys = ON", "PRAGMA busy_timeout = 500")

# docs/12 §3 verbatim, then `override` from the ER diagram and `schema_version` (§5).
SCHEMA = """
CREATE TABLE run (
    run_id            TEXT    PRIMARY KEY,
    station_id        TEXT    NOT NULL,
    started_utc       TEXT    NOT NULL,
    ended_utc         TEXT,                    -- NULL if the process died
    config_hash       TEXT    NOT NULL,
    qs_version        TEXT    NOT NULL,
    counters          TEXT                     -- JSON, written at a clean stop
);

CREATE TABLE inspection (
    id                INTEGER PRIMARY KEY,
    run_id            TEXT    NOT NULL REFERENCES run(run_id),
    seq               INTEGER NOT NULL,
    batch_id          TEXT    NOT NULL,        -- the run_id on the bench (FR-43)
    ts_utc            TEXT    NOT NULL,

    label             TEXT    NOT NULL CHECK (label IN ('pass','defect','no_verdict')),
    reason            TEXT,

    model_id          TEXT    NOT NULL,
    threshold_id      TEXT    NOT NULL,
    config_hash       TEXT    NOT NULL,

    image_status      TEXT    NOT NULL CHECK (image_status IN ('stored','shed','none')),
    image_path        TEXT,                    -- set only when stored
    image_sha256      TEXT,
    heatmap_path      TEXT,
    unit_x INTEGER, unit_y INTEGER, unit_w INTEGER, unit_h INTEGER,

    exposure_us       INTEGER,
    gain              INTEGER,                 -- raw driver units
    wb_kelvin         INTEGER,
    focus             INTEGER,
    camera_verified   INTEGER CHECK (camera_verified IN (0, 1)),  -- NULL when no frame
    reference_patch   REAL,

    capture_ms        REAL,
    inspect_ms        REAL,
    total_ms          REAL,

    UNIQUE (run_id, seq),
    CHECK (label <> 'no_verdict' OR reason IS NOT NULL),
    CHECK ((image_status = 'stored') = (image_path IS NOT NULL AND image_sha256 IS NOT NULL)),
    CHECK (image_status <> 'none' OR label = 'no_verdict'),
    CHECK (image_status = 'none' OR camera_verified IS NOT NULL)
);

CREATE TABLE inspection_check (
    inspection_id     INTEGER NOT NULL REFERENCES inspection(id),
    name              TEXT    NOT NULL,        -- 'geometry' | 'appearance'
    passed            INTEGER NOT NULL CHECK (passed IN (0, 1)),
    value             REAL,                    -- appearance: the anomaly score
    limit_value       REAL,                    -- appearance: the threshold ("limit" is reserved)
    failed_on         TEXT,                    -- e.g. 'area_px<min', 'score>threshold'
    measures          TEXT    NOT NULL DEFAULT '{}',  -- JSON
    PRIMARY KEY (inspection_id, name)
);

CREATE TRIGGER inspection_append_only_u  BEFORE UPDATE ON inspection
    BEGIN SELECT RAISE(ABORT, 'inspection is append-only'); END;
CREATE TRIGGER inspection_append_only_d  BEFORE DELETE ON inspection
    BEGIN SELECT RAISE(ABORT, 'inspection is append-only'); END;
CREATE TRIGGER check_append_only_u       BEFORE UPDATE ON inspection_check
    BEGIN SELECT RAISE(ABORT, 'inspection_check is append-only'); END;
CREATE TRIGGER check_append_only_d       BEFORE DELETE ON inspection_check
    BEGIN SELECT RAISE(ABORT, 'inspection_check is append-only'); END;

CREATE INDEX idx_inspection_batch ON inspection(batch_id, ts_utc);
CREATE INDEX idx_inspection_label ON inspection(label, ts_utc);
CREATE INDEX idx_check_failed     ON inspection_check(name, passed);

CREATE TABLE override (
    id                INTEGER PRIMARY KEY,
    inspection_id     INTEGER NOT NULL REFERENCES inspection(id),
    ts_utc            TEXT    NOT NULL,
    operator          TEXT    NOT NULL,
    new_label         TEXT    NOT NULL CHECK (new_label IN ('pass','defect','no_verdict')),
    note              TEXT
);

CREATE TRIGGER override_append_only_u    BEFORE UPDATE ON override
    BEGIN SELECT RAISE(ABORT, 'override is append-only'); END;
CREATE TRIGGER override_append_only_d    BEFORE DELETE ON override
    BEGIN SELECT RAISE(ABORT, 'override is append-only'); END;

CREATE TABLE schema_version (
    version           INTEGER NOT NULL
);
"""


class SchemaVersionError(sqlite3.DatabaseError):
    """The database is not schema v1: it is never written into. A storage failure."""

    def __init__(self, path: Path, version: int | None) -> None:
        found = "no schema_version" if version is None else f"schema_version {version}"
        super().__init__(
            f"{path} has {found}; this station writes schema_version {SCHEMA_VERSION} only"
        )
        self.version = version


class RecordRejected(ValueError):
    """A record the store can never write (A1, #55). The message names the rule."""


def utc_text(moment: datetime) -> str:
    """An aware datetime as `YYYY-MM-DDTHH:MM:SS.ffffffZ`."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{moment!r} is naive: a timestamp must be timezone-aware")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def path_safe(name: str, what: str) -> str:
    """`name` if it is safe as one path component, else ValueError."""
    if not isinstance(name, str) or PATH_SAFE.fullmatch(name) is None:
        raise ValueError(f"{what} {name!r} is not path-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
    return name


def encode_frame_jpeg(pixels: np.ndarray) -> bytes:
    """A frame as JPEG q85 (ADR-0012), on the pipeline thread before queuing. Raises on
    failure; the frame is not modified."""
    try:
        ok, buffer = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    except cv2.error as exc:
        raise ValueError(f"the frame cannot be encoded as JPEG: {exc}") from exc
    if not ok:
        raise ValueError("the frame cannot be encoded as JPEG")
    return bytes(buffer.tobytes())


# --- rows -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunRow:
    run_id: str
    station_id: str
    started_utc: str
    config_hash: str
    qs_version: str


@dataclass(frozen=True, slots=True)
class CheckRow:
    name: str
    passed: int
    value: float | None
    limit_value: float | None
    failed_on: str | None
    measures: str  # a JSON object, in the tuple's order


@dataclass(frozen=True, slots=True)
class InspectionRow:
    """One record as column values plus its two encoded files — what the sink queues. It holds
    no reference to the Verdict or any array (ADR-0012, GATE-2 R15). `shed` marks a record
    whose image was dropped under a backlog: its row is written as `image_status 'shed'`."""

    station_id: str
    run_id: str
    seq: int
    ts_utc: str
    label: str
    reason: str | None
    model_id: str
    threshold_id: str
    config_hash: str
    unit_x: int | None
    unit_y: int | None
    unit_w: int | None
    unit_h: int | None
    exposure_us: int | None
    gain: int | None
    wb_kelvin: int | None
    focus: int | None
    camera_verified: int | None
    reference_patch: float | None
    capture_ms: float | None
    inspect_ms: float | None
    total_ms: float | None
    checks: tuple[CheckRow, ...]
    image_jpeg: bytes | None
    heatmap_png: bytes | None
    shed: bool = False

    @property
    def image_status(self) -> str:
        if self.image_jpeg is not None:
            return "stored"
        return "shed" if self.shed else "none"

    def relative_dir(self) -> str:
        """`<station_id>/<YYYY>/<MM>/<DD>/<run_id>`, the date taken from `ts_utc`."""
        return f"{self.station_id}/{self.ts_utc[0:4]}/{self.ts_utc[5:7]}/{self.ts_utc[8:10]}/" + (
            self.run_id
        )

    def image_relpath(self) -> str:
        return f"images/{self.relative_dir()}/{self.seq:06d}_{self.label}.jpg"

    def heatmap_relpath(self) -> str:
        return f"heatmaps/{self.relative_dir()}/{self.seq:06d}.png"


def run_row(run_id: str, station_id: str, started_utc: datetime, config_hash: str) -> RunRow:
    """The `run` row for a run's identity. ValueError for a path-unsafe ID or a naive time."""
    return RunRow(
        run_id=path_safe(run_id, "run_id"),
        station_id=path_safe(station_id, "station_id"),
        started_utc=utc_text(started_utc),
        config_hash=config_hash,
        qs_version=qs.__version__,
    )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def validate_record(record: InspectionRecord, run_id: str) -> None:
    """Every rule of the P03 spec's "Record validation"; RecordRejected names the one broken.
    C8 across the two tables, which SQLite cannot check, is asserted here (docs/12 §3)."""
    if not isinstance(record, InspectionRecord):
        raise RecordRejected(f"not an InspectionRecord: {type(record).__name__}")
    if record.run_id != run_id:
        raise RecordRejected(f"record run_id {record.run_id!r} is not this run's {run_id!r}")
    event, verdict = record.event, record.verdict
    if not isinstance(event.seq, int) or isinstance(event.seq, bool) or event.seq < 0:
        raise RecordRejected(f"seq {event.seq!r} is not an integer >= 0")
    if event.wall.tzinfo is None or event.wall.utcoffset() is None:
        raise RecordRejected("the event wall time is naive: it must be timezone-aware")
    label = verdict.label
    if not isinstance(label, Label):
        raise RecordRejected(f"label {label!r} is not one of pass, defect, no_verdict")
    checks = verdict.checks
    names = [check.name for check in checks]
    for name in names:
        if names.count(name) > 1:
            raise RecordRejected(f"two checks share the name {name!r}")
    if label is Label.NO_VERDICT:
        if checks:
            raise RecordRejected(
                "C8: a no_verdict carries no checks (partial results are discarded)"
            )
        if verdict.reason not in REASONS:
            raise RecordRejected(
                f"no_verdict reason {verdict.reason!r} is not one of {sorted(REASONS)}"
            )
    else:
        if not checks:
            raise RecordRejected(f"C8: a {label.value} with no checks is not a verdict")
        failed = [check.name for check in checks if check.passed is not True]
        if label is Label.PASS and failed:
            raise RecordRejected(f"C8: a pass whose check(s) did not pass: {', '.join(failed)}")
        if label is Label.DEFECT and not failed:
            raise RecordRejected("C8: a defect with every check passed")
        if record.image_jpeg is None:
            raise RecordRejected(f"a {label.value} must carry its image")
    for check in checks:
        for field, value in (("value", check.value), ("limit", check.limit)):
            if value is not None and not _finite(value):
                raise RecordRejected(f"check {check.name!r} {field} is not finite: {value!r}")
        measure_names = [name for name, _ in check.measures]
        for name, value in check.measures:
            if measure_names.count(name) > 1:
                raise RecordRejected(f"check {check.name!r} has two measures named {name!r}")
            if not _finite(value):
                raise RecordRejected(
                    f"check {check.name!r} measure {name!r} is not finite: {value!r}"
                )
    if record.image_jpeg is not None:
        if record.camera is None:
            raise RecordRejected("an image needs its camera read-back (camera_verified)")
        if not isinstance(record.image_jpeg, bytes) or not record.image_jpeg.startswith(
            JPEG_MAGIC
        ):
            raise RecordRejected("the image bytes are not a JPEG (FF D8 FF)")
    if record.heatmap_png is not None:
        if record.image_jpeg is None:
            raise RecordRejected("a heatmap needs an image")
        if not isinstance(record.heatmap_png, bytes) or not record.heatmap_png.startswith(
            PNG_MAGIC
        ):
            raise RecordRejected("the heatmap bytes are not a PNG")


def _real(value: float | None) -> float | None:
    return None if value is None else float(value)


def _integer(value: int | None) -> int | None:
    return None if value is None else int(value)


def inspection_row(record: InspectionRecord, run_id: str, station_id: str) -> InspectionRow:
    """The record validated, then as row values and bytes (the spec's "The row, from the
    record"). Raises RecordRejected; keeps no reference to the Verdict or an array."""
    validate_record(record, run_id)
    verdict, camera = record.verdict, record.camera
    unit = verdict.unit_box
    checks = tuple(
        CheckRow(
            name=str(check.name),
            passed=1 if check.passed is True else 0,
            value=_real(check.value),
            limit_value=_real(check.limit),
            failed_on=None if check.failed_on is None else str(check.failed_on),
            measures=json.dumps(
                {str(name): float(value) for name, value in check.measures}, allow_nan=False
            ),
        )
        for check in verdict.checks
    )
    return InspectionRow(
        station_id=path_safe(station_id, "station_id"),
        run_id=run_id,
        seq=int(record.event.seq),
        ts_utc=utc_text(record.event.wall),
        label=verdict.label.value,
        reason=verdict.reason,
        model_id=str(verdict.model_id),
        threshold_id=str(verdict.threshold_id),
        config_hash=str(verdict.config_hash),
        unit_x=None if unit is None else int(unit[0]),
        unit_y=None if unit is None else int(unit[1]),
        unit_w=None if unit is None else int(unit[2]),
        unit_h=None if unit is None else int(unit[3]),
        exposure_us=None if camera is None else _integer(camera.exposure_us),
        gain=None if camera is None else _integer(camera.gain),
        wb_kelvin=None if camera is None else _integer(camera.wb_kelvin),
        focus=None if camera is None else _integer(camera.focus),
        camera_verified=None if camera is None else int(camera.verified is True),
        reference_patch=_real(record.reference_patch),
        capture_ms=_real(record.capture_ms),
        inspect_ms=_real(verdict.inspect_ms),
        total_ms=_real(record.total_ms),
        checks=checks,
        image_jpeg=record.image_jpeg,
        heatmap_png=record.heatmap_png,
    )


# --- files and the database -----------------------------------------------------------------


def write_file(path: Path, data: bytes) -> None:
    """The store's single file write: to a temporary name beside `path`, then `os.replace`.
    A retry of the same record replaces its own half-written file. No fsync (docs/16 §8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


_INSERT_INSPECTION = """
INSERT INTO inspection (
    run_id, seq, batch_id, ts_utc, label, reason, model_id, threshold_id, config_hash,
    image_status, image_path, image_sha256, heatmap_path, unit_x, unit_y, unit_w, unit_h,
    exposure_us, gain, wb_kelvin, focus, camera_verified, reference_patch,
    capture_ms, inspect_ms, total_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_CHECK = """
INSERT INTO inspection_check (inspection_id, name, passed, value, limit_value, failed_on, measures)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class Store:
    """One evidence root: `<root>/qs.db` plus `images/` and `heatmaps/`. Opening creates the
    root and schema v1 if absent and refuses any other version (SchemaVersionError)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / DB_NAME
        conn = sqlite3.connect(path)
        try:
            for pragma in PRAGMAS:
                conn.execute(pragma)
            _ensure_schema(conn, path)
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def begin_run(self, run: RunRow) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO run (run_id, station_id, started_utc, config_hash, qs_version)"
                " VALUES (?, ?, ?, ?, ?)",
                (run.run_id, run.station_id, run.started_utc, run.config_hash, run.qs_version),
            )

    def end_run(self, run_id: str, ended_utc: datetime, counters_json: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE run SET ended_utc = ?, counters = ? WHERE run_id = ?",
                (utc_text(ended_utc), counters_json, run_id),
            )

    def write_record(self, row: InspectionRow) -> None:
        """Files first, then the `inspection` row and its `inspection_check` rows in one
        transaction (docs/16 §4). A failure leaves at most orphan files, never a row without
        its files. A `(run_id, seq)` already stored is refused before any file is written, so
        a duplicate never overwrites the first record's image."""
        if self._conn.execute(
            "SELECT 1 FROM inspection WHERE run_id = ? AND seq = ?", (row.run_id, row.seq)
        ).fetchone():
            raise sqlite3.IntegrityError(
                f"UNIQUE constraint failed: inspection.run_id, inspection.seq"
                f" ({row.run_id}, {row.seq}); no file was written"
            )
        image_path = image_sha256 = heatmap_path = None
        if row.image_jpeg is not None:
            image_path = row.image_relpath()
            write_file(self.root / image_path, row.image_jpeg)
            image_sha256 = hashlib.sha256(row.image_jpeg).hexdigest()
            if row.heatmap_png is not None:
                heatmap_path = row.heatmap_relpath()
                write_file(self.root / heatmap_path, row.heatmap_png)
        with self._conn:
            cursor = self._conn.execute(
                _INSERT_INSPECTION,
                (
                    row.run_id, row.seq, row.run_id, row.ts_utc, row.label, row.reason,
                    row.model_id, row.threshold_id, row.config_hash,
                    row.image_status, image_path, image_sha256, heatmap_path,
                    row.unit_x, row.unit_y, row.unit_w, row.unit_h,
                    row.exposure_us, row.gain, row.wb_kelvin, row.focus, row.camera_verified,
                    row.reference_patch, row.capture_ms, row.inspect_ms, row.total_ms,
                ),
            )  # fmt: skip
            inspection_id = cursor.lastrowid
            self._conn.executemany(
                _INSERT_CHECK,
                [
                    (inspection_id, c.name, c.passed, c.value, c.limit_value, c.failed_on,
                     c.measures)
                    for c in row.checks
                ],
            )  # fmt: skip


def _ensure_schema(conn: sqlite3.Connection, path: Path) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "schema_version" in tables:
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_version")]
        if versions != [SCHEMA_VERSION]:
            raise SchemaVersionError(path, versions[0] if len(versions) == 1 else None)
        return
    if tables:
        raise SchemaVersionError(path, None)  # a database this station did not create
    conn.executescript(
        "BEGIN;\n"
        + SCHEMA
        + f"\nINSERT INTO schema_version (version) VALUES ({SCHEMA_VERSION});\nCOMMIT;"
    )
