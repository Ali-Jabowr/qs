"""`qs station --config PATH --replay DIR`: a recorded session through the whole station (FR-61).

Headless and as fast as the disk allows: `ReplayTrigger` and `ReplaySource` in place of the
configured trigger and camera (in memory — the config file is never rewritten, and its hash is
still the file's), the Inspector the config names, `NullAnnunciator`, and `SqliteSink`. It prints
one summary and exits 0 when every event has exactly one written record and the run ended
READY, 1 when the run completed otherwise, and 2 when it could not start (or stopped itself on
a camera read-back mismatch). In P04 `--replay` is required: the live station is P09, and a
stub never faces a live unit. SIGINT is a stop request.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any

from qs.annunciator import NullAnnunciator
from qs.config import LoadedConfig, load_config
from qs.contracts import FrameSource, Inspector
from qs.errors import ConfigError
from qs.pipeline import Clock, Pipeline, Shows, StationSink, WallClock, utc_now
from qs.sink import SqliteSink
from qs.sources.replay import ReplayError, replay_files, replay_pair
from qs.station import StartupRefused, Station, build_inspector
from qs.store import DB_NAME


def add_parser(commands: Any) -> None:
    """Register `station` on the `qs` subcommands."""
    station = commands.add_parser(
        "station", help="replay a directory of frames through the whole station (FR-61)"
    )
    station.add_argument("--config", required=True, metavar="PATH", help="the station config")
    station.add_argument(
        "--replay", metavar="DIR", default=None,
        help="the *.png frames directly in DIR, in name order (required until P09)",
    )


def default_sink(loaded: LoadedConfig, run_id: str, started: datetime) -> StationSink:
    return SqliteSink.from_config(loaded, run_id, started)


def _identity(source: FrameSource) -> FrameSource:
    return source


def main(
    args: argparse.Namespace,
    *,
    make_inspector: Callable[[LoadedConfig], Inspector] = build_inspector,
    make_sink: Callable[[LoadedConfig, str, datetime], StationSink] = default_sink,
    make_annunciator: Callable[[], Shows] = NullAnnunciator,
    wrap_source: Callable[[FrameSource], FrameSource] = _identity,
    clock: Clock = time.perf_counter,
    wall: WallClock = utc_now,
) -> int:
    """Run the station over `--replay`; 0, 1 or 2 as the module says."""
    try:
        loaded = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    config = loaded.config
    if args.replay is None:
        if config.inspector.type == "stub":
            print("inspector.type stub runs only with --replay: a stub never faces a live unit")
        else:
            print("--replay DIR is required: the live station is P09")
        return 2
    directory = Path(args.replay)
    try:
        files = replay_files(directory)
        trigger, source = replay_pair(
            config, files, directory.resolve().name, clock=clock, wall=wall
        )
        inspector = make_inspector(loaded)
    except (ReplayError, ConfigError, StartupRefused) as exc:
        print(str(exc))
        return 2
    station = Station(
        loaded,
        trigger=trigger,
        source=wrap_source(source),
        inspector=inspector,
        annunciator=make_annunciator(),
        make_sink=make_sink,
        clock=clock,
        wall=wall,
    )
    try:
        pipeline = station.start()
    except StartupRefused as exc:
        print(str(exc))
        return 2

    def interrupted(signum: int, frame: FrameType | None) -> None:
        pipeline.request_stop()

    previous = signal.signal(signal.SIGINT, interrupted)
    try:
        station.run()
    finally:
        try:
            station.stop()
        finally:
            signal.signal(signal.SIGINT, previous)
    assert station.sink is not None and station.run_id is not None
    complete = _summary(pipeline, station.sink, station.run_id, loaded)
    if pipeline.refusal is not None:
        print(pipeline.refusal)
        return 2
    return 0 if complete and pipeline.final_state == "READY" else 1


def _summary(pipeline: Pipeline, sink: StationSink, run_id: str, loaded: LoadedConfig) -> bool:
    """Print the run's summary; whether every event has exactly one written record."""
    counters = pipeline.counters
    written = sink.counters()
    labels: dict[str, int] = counters["labels"]  # type: ignore[assignment]
    reasons: dict[str, int] = counters["no_verdict"]  # type: ignore[assignment]
    faults: dict[str, int] = counters["faults"]  # type: ignore[assignment]
    events = int(counters["events"])  # type: ignore[call-overload]
    database = Path(os.path.abspath(loaded.config.evidence.root)) / DB_NAME
    print(f"run {run_id}")
    print(f"evidence {database}")
    print(
        f"events {events} · pass {labels['pass']} · defect {labels['defect']}"
        f" · no_verdict {labels['no_verdict']}"
    )
    if reasons:
        print("no_verdict by reason: " + ", ".join(f"{r} {n}" for r, n in reasons.items()))
    if faults:
        print(
            "faults: " + ", ".join(f"{r} {n}" for r, n in faults.items())
            + f" · reconnect attempts {counters['reconnect_attempts']}"
        )
    latency: dict[str, Any] = counters["latency"]  # type: ignore[assignment]
    p50, p95 = latency["p50_ms"], latency["p95_ms"]
    print(
        "latency "
        + ("—" if p50 is None else f"p50 {p50:.0f} ms · p95 {p95:.0f} ms")
        + f" · breaches {latency['breaches']} (budget {latency['budget_ms']} ms)"
        + f" · camera verified {'yes' if counters['camera_verified'] else 'no'}"
    )
    print(f"final state {pipeline.final_state}")
    print(
        f"records written {written['written']} · held {written['held']}"
        f" · images shed {written['images_shed']} · rejected {written['rejected']}"
        f" · lost {written['lost']}"
    )
    if written["lost"]:
        print(f"{written['lost']} record(s) LOST: storage did not come back before the stop")
    return (
        written["written"] == events
        and written["lost"] == 0
        and written["rejected"] == 0
        and written["pending"] == 0
        and counters["sink_errors"] == 0
    )
