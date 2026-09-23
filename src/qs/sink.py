"""`SqliteSink`: the one `Sink` for the bench, the factory and the tests (docs/11 §3, §4).

`write()` validates the record, turns it into row values and encoded bytes, queues that and
returns: it never raises and never blocks (C3). One daemon worker owns the only database
connection and drains the queue FIFO, so rows are committed in `seq` order. The run row is its
first item. Errors are swallowed, counted and surfaced through `health()` (docs/11 §7).

When storage fails (ADR-0012) the verdict wins and the record outlives the image: the item at
the head is retried every `retry_s` and nothing behind it is attempted; rows are held without
limit; images are shed past `HIGH_WATER` queued, PASS first, then NO_VERDICT, then DEFECT
(A2, #56). A stop waits until `stop_wait_s` pass with no successful write (A3, #57), then
counts what is left as lost.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path

from qs.config import LoadedConfig
from qs.contracts import Health, InspectionRecord, Label
from qs.store import InspectionRow, RunRow, Store, inspection_row, run_row, utc_text

_log = logging.getLogger(__name__)

HIGH_WATER = 200  # queued records with images, ~110 MB (ADR-0012)
SHED_ORDER = (Label.PASS.value, Label.NO_VERDICT.value, Label.DEFECT.value)  # A2, #56
_ABANDON_JOIN_S = 0.25  # after a stop gives up, how long to wait for the worker to notice


class _Kind(Enum):
    RUN = "run"
    RECORD = "record"
    FINISH = "finish"


@dataclass(slots=True, eq=False)
class _Item:
    kind: _Kind
    row: InspectionRow | None = None
    ended_utc: datetime | None = None
    counters: dict[str, object] | None = None
    epoch: int = 0  # failed attempts before it was queued: it is held once more have failed


class SqliteSink:
    """The evidence sink. Construct it with the run's identity; it never touches the disk
    until its worker runs, and never raises for storage (ValueError only for a path-unsafe
    `run_id`/`station_id` or a naive start, which are programming errors)."""

    def __init__(
        self,
        root: Path | str,
        run_id: str,
        station_id: str,
        started_utc: datetime,
        config_hash: str,
        *,
        retry_s: float = 1.0,
        stop_wait_s: float = 10.0,
    ) -> None:
        self._run: RunRow = run_row(run_id, station_id, started_utc, config_hash)
        self._root = Path(root)
        self._retry_s = retry_s
        self._stop_wait_s = stop_wait_s
        self._cond = threading.Condition()
        self._queue: deque[_Item] = deque([_Item(_Kind.RUN)])
        self._records_queued = 0
        self._imaged: dict[str, deque[_Item]] = {label: deque() for label in SHED_ORDER}
        self._images_queued = 0  # queued records still carrying an image
        self._attempting: _Item | None = None
        self._closing = False
        self._abandoned = False
        self._last_success = 0.0  # time.monotonic() of the last successful write
        # counters (docs/16 §6)
        self._written = 0
        self._rejected = 0
        self._lost = 0
        self._write_failures = 0
        self._held_resolved = 0
        self._held_queued = 0
        self._images_shed = 0
        # health: degraded from a failed attempt or a shed until the backlog is written
        self._failing = False
        self._degraded = False
        self._shedding = False
        self._last_error: str | None = None
        self._last_rejection: str | None = None
        self._worker = threading.Thread(
            target=self._drain, name=f"qs-sink-{self._run.run_id}", daemon=True
        )
        self._worker.start()

    @classmethod
    def from_config(
        cls,
        loaded: LoadedConfig,
        run_id: str,
        started_utc: datetime,
        *,
        retry_s: float = 1.0,
        stop_wait_s: float = 10.0,
    ) -> SqliteSink:
        """C6: `evidence.root` resolved against the working directory, the station and the
        config hash from the loaded config."""
        return cls(
            Path(os.path.abspath(loaded.config.evidence.root)),
            run_id=run_id,
            station_id=loaded.config.station_id,
            started_utc=started_utc,
            config_hash=loaded.config_hash,
            retry_s=retry_s,
            stop_wait_s=stop_wait_s,
        )

    # --- the Sink contract --------------------------------------------------------------------

    def write(self, record: InspectionRecord) -> None:
        """Queue one record. Never raises, never blocks (C3); an invalid record is rejected:
        counted, logged with its seq and rule, and `health()` is not ok from then on (A1)."""
        try:
            with self._cond:
                if self._closing:
                    self._lost += 1
                    _log.error("sink closed: record seq %s not written", _seq_of(record))
                    return
            try:
                row = inspection_row(record, self._run.run_id, self._run.station_id)
            except Exception as exc:  # RecordRejected, or a malformed object
                with self._cond:
                    self._reject_locked(_seq_of(record), str(exc) or type(exc).__name__)
                return
            item = _Item(_Kind.RECORD, row=row)
            with self._cond:
                if self._closing:
                    self._lost += 1
                    _log.error("sink closed: record seq %s not written", row.seq)
                    return
                item.epoch = self._write_failures
                self._queue.append(item)
                self._records_queued += 1
                if row.image_jpeg is not None:
                    self._imaged[row.label].append(item)
                    self._images_queued += 1
                    if self._images_queued > HIGH_WATER:
                        self._shed_locked()
                self._cond.notify_all()
        except Exception:  # never into the Pipeline (docs/11 §7)
            _log.exception("sink write failed unexpectedly")
            with self._cond:
                self._rejected += 1

    def pending(self) -> int:
        """Records queued and not yet committed."""
        with self._cond:
            return self._records_queued

    def close(self) -> None:
        """Stop accepting records and drain. Returns once the queue is written, or once
        `stop_wait_s` pass with the queue non-empty and no successful write (F13, A3): what is
        left is counted as `lost` and `run.ended_utc` stays NULL. Idempotent."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
            start = time.monotonic()
            while self._queue and not self._abandoned:
                remaining = max(start, self._last_success) + self._stop_wait_s - time.monotonic()
                if remaining <= 0:
                    self._abandon_locked()
                    break
                self._cond.wait(remaining)
            abandoned = self._abandoned
        self._worker.join(_ABANDON_JOIN_S if abandoned else None)

    # --- beyond the protocol ------------------------------------------------------------------

    def finish_run(self, ended_utc: datetime, counters: Mapping[str, object]) -> None:
        """Queue the final `run` update: `ended_utc`, and `counters` as JSON with the sink's own
        under `"sink"`. ValueError for a naive time, a `"sink"` key, or counters not JSON."""
        utc_text(ended_utc)
        if "sink" in counters:
            raise ValueError("counters must not carry the key 'sink': the sink writes its own")
        caller = dict(counters)
        json.dumps(caller, allow_nan=False)
        with self._cond:
            if self._closing:
                _log.error("finish_run after close: run %s keeps ended_utc NULL", self._run.run_id)
                return
            self._queue.append(_Item(_Kind.FINISH, ended_utc=ended_utc, counters=caller))
            self._cond.notify_all()

    def counters(self) -> dict[str, int]:
        """The sink's docs/16 §6 counters. `written + rejected + lost + pending` is every record
        given to `write()`."""
        with self._cond:
            return {
                "written": self._written,
                "held": self._held_resolved + self._held_queued,
                "images_shed": self._images_shed,
                "write_failures": self._write_failures,
                "rejected": self._rejected,
                "lost": self._lost,
                "pending": self._records_queued,
            }

    def health(self) -> Health:
        """Not ok for the whole outage — from a failed attempt or a shed until the backlog is
        written — and for the rest of the run once a record was rejected or lost."""
        with self._cond:
            ok = not (self._failing or self._degraded or self._rejected or self._lost)
            if ok:
                return Health(True, f"evidence written: {self._written}")
            parts = [
                f"held {self._held_resolved + self._held_queued}, shed {self._images_shed},"
                f" rejected {self._rejected}, lost {self._lost}, pending {self._records_queued}"
            ]
            if self._failing:
                parts.append(f"storage failing: {self._last_error}")
            elif self._degraded and self._last_error and not self._shedding:
                parts.append(f"writing the backlog after: {self._last_error}")
            if self._shedding:
                parts.append(f"image backlog past the high-water ({HIGH_WATER}): images shed")
            if self._last_rejection:
                parts.append(self._last_rejection)
            return Health(False, "; ".join(parts))

    # --- queue bookkeeping, all under self._cond ------------------------------------------------

    def _reject_locked(self, seq: object, rule: str) -> None:
        message = f"rejected seq {seq}: {rule}"
        _log.error("evidence record %s", message)
        self._rejected += 1
        self._last_rejection = message

    def _shed_locked(self) -> None:
        """Drop images until HIGH_WATER remain: PASS, then NO_VERDICT, then DEFECT, oldest first
        within each. The record being attempted is not changed mid-attempt."""
        shed = 0
        while self._images_queued > HIGH_WATER:
            victim = self._victim_locked()
            if victim is None:
                break
            assert victim.row is not None
            victim.row = replace(victim.row, image_jpeg=None, heatmap_png=None, shed=True)
            self._images_queued -= 1
            self._images_shed += 1
            shed += 1
        if shed:
            if not self._shedding:
                _log.warning("evidence backlog past %d images: shedding images", HIGH_WATER)
            self._degraded = self._shedding = True

    def _victim_locked(self) -> _Item | None:
        for label in SHED_ORDER:
            candidates = self._imaged[label]
            for index, item in enumerate(candidates):
                if item is not self._attempting:
                    del candidates[index]
                    return item
        return None

    def _resolve_locked(self, item: _Item) -> None:
        """A record leaves the queue: counted as held if it waited through a failed attempt,
        and no longer a candidate for shedding."""
        self._records_queued -= 1
        if item.epoch < self._write_failures:
            self._held_resolved += 1
            self._held_queued -= 1
        assert item.row is not None
        if item.row.image_jpeg is not None:
            candidates = self._imaged[item.row.label]
            if candidates and candidates[0] is item:
                candidates.popleft()
            else:
                candidates.remove(item)
            self._images_queued -= 1

    def _abandon_locked(self) -> None:
        """The stop gave up: every record still queued is lost (F13)."""
        for item in self._queue:
            if item.kind is _Kind.RECORD:
                self._resolve_locked(item)
                self._lost += 1
        self._queue.clear()
        self._abandoned = True
        self._degraded = True
        _log.error(
            "evidence stop gave up after %.1f s without a successful write: %d record(s) lost",
            self._stop_wait_s,
            self._lost,
        )
        self._cond.notify_all()

    # --- the worker ---------------------------------------------------------------------------

    def _attempt(self, store: Store, item: _Item, row: InspectionRow | None) -> None:
        if item.kind is _Kind.RUN:
            try:
                store.begin_run(self._run)
            except sqlite3.IntegrityError as exc:  # never write into another run's rows
                raise sqlite3.OperationalError(
                    f"run {self._run.run_id} already exists in the database"
                ) from exc
        elif item.kind is _Kind.RECORD:
            assert row is not None
            store.write_record(row)
        else:
            assert item.ended_utc is not None and item.counters is not None
            counters = {**item.counters, "sink": self.counters()}
            store.end_run(self._run.run_id, item.ended_utc, json.dumps(counters))

    def _drain(self) -> None:
        store: Store | None = None
        try:
            while True:
                with self._cond:
                    while not self._queue and not self._closing and not self._abandoned:
                        self._cond.wait()
                    if self._abandoned or not self._queue:
                        return
                    item = self._queue[0]
                    self._attempting = item
                    row = item.row  # never changed mid-attempt: the shed skips this item
                failure: Exception | None = None
                rejection: Exception | None = None
                try:
                    if store is None:
                        store = Store(self._root)
                    self._attempt(store, item, row)
                except sqlite3.IntegrityError as exc:
                    if item.kind is _Kind.RECORD:
                        rejection = exc  # this record can never be written (A1)
                    else:
                        failure = exc
                except (OSError, sqlite3.Error) as exc:
                    failure = exc  # storage: hold it and retry (ADR-0012)
                except Exception as exc:
                    if item.kind is _Kind.RECORD:
                        rejection = exc
                    else:
                        failure = exc
                if failure is not None and store is not None:
                    store.close()
                    store = None
                with self._cond:
                    self._attempting = None
                    if self._abandoned:  # the stop already counted this record as lost
                        if item.kind is _Kind.RECORD and failure is None:
                            self._lost -= 1
                            self._record_outcome_locked(row, rejection)
                        return
                    if failure is None:
                        self._queue.popleft()
                        if item.kind is _Kind.RECORD:
                            self._resolve_locked(item)
                            self._record_outcome_locked(row, rejection)
                        self._failing = False
                        self._last_success = time.monotonic()
                        if not self._queue:
                            self._degraded = self._shedding = False
                    else:
                        self._write_failures += 1
                        self._held_queued = self._records_queued
                        self._failing = self._degraded = True
                        self._last_error = f"{type(failure).__name__}: {failure}"
                        _log.warning("evidence write failed, held: %s", self._last_error)
                    self._cond.notify_all()
                    if failure is not None:
                        self._cond.wait(self._retry_s)
        finally:
            if store is not None:
                store.close()

    def _record_outcome_locked(
        self, row: InspectionRow | None, rejection: Exception | None
    ) -> None:
        if rejection is None:
            self._written += 1
        else:
            assert row is not None
            self._reject_locked(row.seq, f"{type(rejection).__name__}: {rejection}")


def _seq_of(record: object) -> object:
    try:
        return record.event.seq  # type: ignore[attr-defined]
    except Exception:
        return "?"
