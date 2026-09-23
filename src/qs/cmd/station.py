"""`qs station --config PATH --replay DIR`: a recorded session through the whole station (FR-61).

Headless and as fast as the disk allows: `ReplayTrigger` and `ReplaySource` in place of the
configured trigger and camera (in memory — the config file is never rewritten, and its hash is
still the file's), the Inspector the config names, `NullAnnunciator`, and `SqliteSink`. It prints
one summary and exits 0 when every event has exactly one written record and the run ended
READY, 1 when the run completed otherwise, and 2 when it could not start. In P04 `--replay` is
required: the live station is P09, and a stub never faces a live unit.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from qs.annunciator import NullAnnunciator
from qs.config import LoadedConfig, load_config
from qs.contracts import Inspector
from qs.errors import ConfigError
from qs.pipeline import Clock, Shows, StationSink, WallClock, utc_now
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


def main(
    args: argparse.Namespace,
    *,
    make_inspector: Callable[[LoadedConfig], Inspector] = build_inspector,
    make_sink: Callable[[LoadedConfig, str, datetime], StationSink] = default_sink,
    make_annunciator: Callable[[], Shows] = NullAnnunciator,
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
        source=source,
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
    try:
        station.run()
    finally:
        station.stop()
    assert station.sink is not None and station.run_id is not None
    counters = pipeline.counters
    sink = station.sink.counters()
    labels: dict[str, int] = counters["labels"]  # type: ignore[assignment]
    reasons: dict[str, int] = counters["no_verdict"]  # type: ignore[assignment]
    events = int(counters["events"])  # type: ignore[call-overload]
    database = Path(os.path.abspath(config.evidence.root)) / DB_NAME
    print(f"run {station.run_id}")
    print(f"evidence {database}")
    print(
        f"events {events} · pass {labels['pass']} · defect {labels['defect']}"
        f" · no_verdict {labels['no_verdict']}"
    )
    if reasons:
        print("no_verdict by reason: " + ", ".join(f"{r} {n}" for r, n in reasons.items()))
    print(
        f"records written {sink['written']} · held {sink['held']}"
        f" · images shed {sink['images_shed']} · rejected {sink['rejected']}"
        f" · lost {sink['lost']}"
    )
    complete = (
        sink["written"] == events
        and sink["lost"] == 0
        and sink["rejected"] == 0
        and sink["pending"] == 0
        and counters["sink_errors"] == 0
    )
    return 0 if complete else 1
