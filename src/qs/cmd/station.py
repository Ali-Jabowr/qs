"""`qs station --config PATH --replay DIR [--screen]`: the whole station (FR-61).

**Headless replay** (`--replay DIR`, P04) runs as fast as the disk allows: `ReplayTrigger` and
`ReplaySource` in place of the configured trigger and camera (in memory — the config file is
never rewritten, and its hash is still the file's), the Inspector the config names,
`NullAnnunciator`, and `SqliteSink`.

**Key-paced replay on the screen** (`--replay DIR --screen`, P09 #89) is what the station runs
on until the webcam arrives (#85, assumption A1). It opens the operator window (ADR-0018),
arms `KeyboardTrigger` on the config's key, and shows every event through
`ScreenAnnunciator`. Each accepted press takes the next frame of `DIR`, and its record says
`replay:<DIR name>`. After the last frame the window stays open and says so; closing it ends
the run.

Both print one summary and exit 0 when every event has exactly one written record and the run
ended READY, 1 when the run completed otherwise, and 2 when it could not start (or stopped
itself on a camera read-back mismatch). `--replay` is still required: the live station comes
with #90, and a stub never faces a live unit. SIGINT is a stop request.

There is no validation mode: `inspector.type: appearance` loads only the bundle the config
names, verified against its `model_id`, and refuses a validation-only one (docs/17 §6).
`make_scorer` and `cuda_available` are the model seams P07 defines — the real PatchCore by
default, P07's numpy fakes in the tests.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
import tkinter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any

from qs.annunciator import NullAnnunciator
from qs.config import DisplayConfig, LoadedConfig, StationConfig, load_config
from qs.contracts import FrameSource, Inspector, TriggerSource
from qs.errors import ConfigError
from qs.keyboard import KeyboardTrigger
from qs.pipeline import Clock, Pipeline, Shows, StationSink, WallClock, utc_now
from qs.screen import Screen, ScreenLike
from qs.screen_annunciator import ScreenAnnunciator
from qs.sink import SqliteSink
from qs.sources.replay import ReplayError, ReplaySource, replay_files, replay_pair
from qs.station import (
    MakeScorer,
    Refreshes,
    ScreenLoop,
    StartupRefused,
    Station,
    build_inspector,
    real_cuda_available,
    real_scorer,
)
from qs.store import DB_NAME

MakeScreen = Callable[[DisplayConfig, str, Clock], ScreenLike]
MakeScreenAnnunciator = Callable[[StationConfig, ScreenLike], Refreshes]

NO_DISPLAY = (
    "no display: the station's screen needs a desktop session (ADR-0018); "
    "headless: qs station --config C --replay DIR"
)


def add_parser(commands: Any) -> None:
    """Register `station` on the `qs` subcommands."""
    station = commands.add_parser(
        "station", help="run the whole station over a directory of frames (FR-61)"
    )
    station.add_argument("--config", required=True, metavar="PATH", help="the station config")
    station.add_argument(
        "--replay", metavar="DIR", default=None,
        help="the *.png frames directly in DIR, in name order (required until the webcam)",
    )
    station.add_argument(
        "--screen", action="store_true",
        help="with --replay: the operator window; each SPACE press takes the next frame",
    )


def default_sink(loaded: LoadedConfig, run_id: str, started: datetime) -> StationSink:
    return SqliteSink.from_config(loaded, run_id, started)


def default_screen(display: DisplayConfig, title: str, clock: Clock) -> ScreenLike:
    """The Tk window on the station's clock, so its key stamps and the trigger agree (#88)."""
    return Screen(display, title, clock=clock)


def _identity(source: FrameSource) -> FrameSource:
    return source


def main(
    args: argparse.Namespace,
    *,
    make_inspector: Callable[[LoadedConfig], Inspector] | None = None,
    make_scorer: MakeScorer = real_scorer,
    cuda_available: Callable[[], bool] = real_cuda_available,
    make_sink: Callable[[LoadedConfig, str, datetime], StationSink] = default_sink,
    make_annunciator: Callable[[], Shows] = NullAnnunciator,
    wrap_source: Callable[[FrameSource], FrameSource] = _identity,
    make_screen: MakeScreen = default_screen,
    make_screen_annunciator: MakeScreenAnnunciator = ScreenAnnunciator,
    sleep: Callable[[float], None] = time.sleep,
    clock: Clock = time.perf_counter,
    wall: WallClock = utc_now,
) -> int:
    """Run the station; 0, 1 or 2 as the module says. `make_inspector` replaces the Inspector
    the config names; by default it is built from the config with `make_scorer` and
    `cuda_available`. `make_screen`, `make_screen_annunciator` and `sleep` are the screen
    mode's seams."""
    try:
        loaded = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    config = loaded.config
    screen_mode = bool(getattr(args, "screen", False))
    if args.replay is None:
        if screen_mode:
            print("--screen goes with --replay DIR: qs station --config C --replay DIR --screen")
        elif config.inspector.type == "stub":
            print("inspector.type stub runs only with --replay: a stub never faces a live unit")
        else:
            print("--replay DIR is required: the live station is P09")
        return 2
    directory = Path(args.replay)
    name = directory.resolve().name
    try:
        files = replay_files(directory)
        trigger: TriggerSource
        source: FrameSource
        if screen_mode:
            source = ReplaySource(config, files, clock=clock)
        else:
            trigger, source = replay_pair(config, files, name, clock=clock, wall=wall)
        if make_inspector is None:
            inspector = build_inspector(
                loaded, make_scorer=make_scorer, cuda_available=cuda_available
            )
        else:
            inspector = make_inspector(loaded)
    except (ReplayError, ConfigError, StartupRefused) as exc:
        print(str(exc))
        return 2
    loop: ScreenLoop | None = None
    annunciator: Shows
    if screen_mode:
        try:
            screen = make_screen(config.display, f"QS — {config.station_id}", clock)
        except tkinter.TclError:
            print(NO_DISPLAY)
            return 2
        screen_annunciator = make_screen_annunciator(config, screen)
        source_id = f"replay:{name}"
        loop = ScreenLoop(screen, screen_annunciator, sleep, len(files), source_id)
        annunciator = screen_annunciator
        # Late-bound: armed() is read at each poll, once the station below exists.
        trigger = KeyboardTrigger(
            config, screen, clock=clock, wall_clock=wall,
            armed=lambda: station.armed(), source_id=source_id, limit=len(files),
        )
    else:
        annunciator = make_annunciator()
    station = Station(
        loaded,
        trigger=trigger,
        source=wrap_source(source),
        inspector=inspector,
        annunciator=annunciator,
        make_sink=make_sink,
        clock=clock,
        wall=wall,
        screen=loop,
    )
    try:
        pipeline = station.start()
    except StartupRefused as exc:
        if loop is not None:
            loop.annunciator.close()
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
            if loop is not None:
                loop.annunciator.close()
    assert station.sink is not None and station.run_id is not None
    complete = _summary(
        pipeline, station.sink, station.run_id, loaded,
        keys_discarded=station.keys_discarded if loop is not None else None,
    )
    if pipeline.refusal is not None:
        print(pipeline.refusal)
        return 2
    return 0 if complete and pipeline.final_state == "READY" else 1


def _summary(
    pipeline: Pipeline,
    sink: StationSink,
    run_id: str,
    loaded: LoadedConfig,
    *,
    keys_discarded: int | None = None,
) -> bool:
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
    if keys_discarded is not None:
        print(f"keys discarded {keys_discarded}")
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
