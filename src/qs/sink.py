"""`SqliteSink`: the one `Sink` for the bench, the factory and the tests (docs/11 §3, §4).

`write()` validates the record, turns it into row values and encoded bytes, queues that and
returns: it never raises and never blocks (C3). One daemon worker owns the only database
connection and drains the queue FIFO, so rows are committed in `seq` order. The run row is its
first item. Errors are swallowed, counted and surfaced through `health()` (docs/11 §7).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from qs.config import LoadedConfig
from qs.contracts import Health, InspectionRecord
from qs.store import InspectionRow, RunRow, Store, inspection_row, run_row, utc_text

_log = logging.getLogger(__name__)


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
        self._closing = False
        # counters (docs/16 §6)
        self._written = 0
        self._rejected = 0
        self._lost = 0
        self._write_failures = 0
        self._held_resolved = 0
        self._held_queued = 0
        self._images_shed = 0
        # health
        self._failing = False
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
                self._reject(_seq_of(record), str(exc) or type(exc).__name__)
                return
            item = _Item(_Kind.RECORD, row=row)
            with self._cond:
                if self._closing:
                    self._lost += 1
                    return
                item.epoch = self._write_failures
                self._queue.append(item)
                self._records_queued += 1
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
        """Stop accepting records, drain the queue, join the worker. Idempotent."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
            while self._queue:
                self._cond.wait()
        self._worker.join()

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
        with self._cond:
            ok = not self._failing and self._rejected == 0 and self._lost == 0
            if ok:
                return Health(True, f"evidence written: {self._written}")
            parts = [
                f"held {self._held_resolved + self._held_queued}, shed {self._images_shed},"
                f" rejected {self._rejected}, lost {self._lost}, pending {self._records_queued}"
            ]
            if self._failing and self._last_error:
                parts.append(f"storage failing: {self._last_error}")
            if self._last_rejection:
                parts.append(self._last_rejection)
            return Health(False, "; ".join(parts))

    # --- the worker ---------------------------------------------------------------------------

    def _reject(self, seq: object, rule: str) -> None:
        message = f"rejected seq {seq}: {rule}"
        _log.error("evidence record %s", message)
        with self._cond:
            self._rejected += 1
            self._last_rejection = message

    def _resolve_locked(self, item: _Item) -> None:
        """A record leaves the queue: count it as held if it waited through a failed attempt."""
        self._records_queued -= 1
        if item.epoch < self._write_failures:
            self._held_resolved += 1
            self._held_queued -= 1

    def _attempt(self, store: Store, item: _Item) -> None:
        if item.kind is _Kind.RUN:
            try:
                store.begin_run(self._run)
            except sqlite3.IntegrityError as exc:  # never write into another run's rows
                raise sqlite3.OperationalError(
                    f"run {self._run.run_id} already exists in the database"
                ) from exc
        elif item.kind is _Kind.RECORD:
            assert item.row is not None
            store.write_record(item.row)
        else:
            assert item.ended_utc is not None and item.counters is not None
            counters = {**item.counters, "sink": self.counters()}
            store.end_run(self._run.run_id, item.ended_utc, json.dumps(counters))

    def _drain(self) -> None:
        store: Store | None = None
        try:
            while True:
                with self._cond:
                    while not self._queue and not self._closing:
                        self._cond.wait()
                    if not self._queue:
                        return
                    item = self._queue[0]
                failure: Exception | None = None
                rejection: Exception | None = None
                try:
                    if store is None:
                        store = Store(self._root)
                    self._attempt(store, item)
                except sqlite3.IntegrityError as exc:
                    rejection = exc  # this record can never be written (A1)
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
                if rejection is not None:
                    assert item.row is not None
                    self._reject(item.row.seq, f"{type(rejection).__name__}: {rejection}")
                with self._cond:
                    if failure is None:
                        self._queue.popleft()
                        if item.kind is _Kind.RECORD:
                            self._resolve_locked(item)
                            if rejection is None:
                                self._written += 1
                        self._failing = False
                    else:
                        self._write_failures += 1
                        self._held_queued = self._records_queued
                        self._failing = True
                        self._last_error = f"{type(failure).__name__}: {failure}"
                        _log.warning("evidence write failed, held: %s", self._last_error)
                    self._cond.notify_all()
                    if failure is not None:
                        self._cond.wait(self._retry_s)
        finally:
            if store is not None:
                store.close()


def _seq_of(record: object) -> object:
    try:
        return record.event.seq  # type: ignore[attr-defined]
    except Exception:
        return "?"
