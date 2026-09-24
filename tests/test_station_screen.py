"""`qs station --config C --replay DIR --screen` (#89): the station on the operator screen,
key-paced over recorded frames, until the webcam arrives (#85, A1).

Through `main(args, *, factories)` with a `FakeScreen` on the fake clock, P04's synthetic
PNGs, the stub inspector and the real `SqliteSink` on `tmp_path`. No display is needed."""

from __future__ import annotations

import json
import tkinter
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from evidence_records import fast_sqlite
from qs.__main__ import build_parser
from qs.config import DisplayConfig, LoadedConfig, StationConfig
from qs.contracts import Label
from qs.errors import ConfigError
from qs.inspector.stub import StubInspector
from qs.operator_view import AMBER, BANNER, GREEN, RED, status_line
from qs.pipeline import StationSink
from qs.screen import KeyPress, ScreenLike
from qs.sink import SqliteSink
from screen_station import (
    BANNER_AT,
    GAP,
    RecordingSqlite,
    Rig,
    closing_after,
    presses,
    recording,
    space,
    stub,
)
from screen_station import database as _db
from screen_station import run_row as _run_row
from station_fixtures import FakeClock, rows, textured, write_config, write_frames, write_png


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


# AC1 ------------------------------------------------------------------------------------


def test_five_presses_make_five_replay_records_and_a_clean_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 5)
    rig = Rig()
    sinks: list[RecordingSqlite] = []
    code = rig.run(
        config, shoot, screen_options=closing_after(presses(5)), make_sink=recording(sinks)
    )
    assert code == 0
    records = rows(_db(tmp_path), "SELECT * FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == [1, 2, 3, 4, 5]
    # qs.db has no source_id column (#94): each record carries it, and run.counters names it.
    assert [r.event.source_id for r in sinks[0].written] == ["replay:shoot"] * 5
    run = _run_row(tmp_path)
    assert run["ended_utc"] is not None
    counters = json.loads(run["counters"])
    assert counters["keys_discarded"] == 0
    assert counters["source_id"] == "replay:shoot"
    assert rig.screens[0].close_calls == 1
    out = capsys.readouterr().out
    assert "events 5 · pass 5" in out and "keys discarded 0" in out


# AC2 ------------------------------------------------------------------------------------


def test_the_band_takes_each_verdicts_colour_and_keeps_it_between_events(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 5)
    rig = Rig()
    code = rig.run(
        config, shoot, screen_options=closing_after(presses(5)),
        make_inspector=stub({3: "defect:appearance", 4: "placement:no unit"}),
    )
    assert code == 0
    timeline = rig.spies[0].timeline
    shows = {seq: band for kind, seq, band in timeline if kind == "show"}
    assert shows[1] == GREEN and shows[3] == RED and shows[4] == AMBER and shows[5] == GREEN
    # Every refresh keeps the last verdict's colour until the next one.
    last: tuple[int, ...] | None = None
    for kind, _, band in timeline:
        if kind == "show":
            last = band
        elif last is not None:
            assert band == last


# AC3 ------------------------------------------------------------------------------------


def test_presses_made_while_an_event_is_handled_are_discarded_and_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 5)
    script: list[list[KeyPress]] = [[], [space()], [space(), space()]]  # 2 during event 1
    script.extend([] for _ in range(GAP))
    script.append([space()])
    script.append([space()])  # 1 during event 2
    script.extend([] for _ in range(GAP))
    rig = Rig()
    assert rig.run(config, shoot, screen_options=closing_after(script)) == 0
    records = rows(_db(tmp_path), "SELECT * FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == [1, 2]  # records == events (C5)
    assert json.loads(_run_row(tmp_path)["counters"])["keys_discarded"] == 3
    assert "keys discarded 3" in capsys.readouterr().out


# AC4 ------------------------------------------------------------------------------------


def test_after_the_last_frame_the_window_stays_and_presses_are_discarded(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 5)
    script = presses(5, then=[[space()], [], [space(), space()], []])
    rig = Rig()
    assert rig.run(config, shoot, screen_options=closing_after(script)) == 0
    assert len(rows(_db(tmp_path), "SELECT * FROM inspection")) == 5
    assert json.loads(_run_row(tmp_path)["counters"])["keys_discarded"] == 3
    spy = rig.spies[0]
    final = spy.statuses[-1]
    assert status_line(final, spy._config.display.labels) == "ПОВТОР ЗАВЕРШЁН — закройте окно"
    captions = [s.caption for s in spy.statuses]
    assert "ПОВТОР: кадр 1 из 5" in captions and "ПОВТОР: кадр 5 из 5" in captions


# AC5 ------------------------------------------------------------------------------------


def test_closing_the_window_after_two_events_stops_cleanly(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 5)
    rig = Rig()
    assert rig.run(config, shoot, screen_options=closing_after(presses(2))) == 0
    assert [r["seq"] for r in rows(_db(tmp_path), "SELECT seq FROM inspection")] == [1, 2]
    assert _run_row(tmp_path)["ended_utc"] is not None


class LosingSink:
    """The real sink, reporting one row lost at its close, as P03 does after an outage."""

    def __init__(self, inner: StationSink) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def counters(self) -> dict[str, int]:
        counters = dict(self.inner.counters())
        counters["lost"] += 1
        return counters


def test_rows_lost_at_the_stop_are_printed_and_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 5)

    def make_sink(loaded: LoadedConfig, run_id: str, started: datetime) -> StationSink:
        return LosingSink(SqliteSink.from_config(loaded, run_id, started))  # type: ignore[return-value]

    rig = Rig()
    code = rig.run(config, shoot, screen_options=closing_after(presses(2)), make_sink=make_sink)
    assert code == 1
    assert "1 record(s) LOST" in capsys.readouterr().out


# AC6 ------------------------------------------------------------------------------------


def test_a_held_key_never_makes_a_busy_record(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 20)
    rig = Rig(oversleep=0.07)  # a 100 ms iteration: the held key repeats at the loop's pace
    code = rig.run(
        config, shoot,
        screen_options={"held": "space", "close_when": lambda calls: calls >= 20 * 5 + 10},
    )
    assert code == 0
    records = rows(_db(tmp_path), "SELECT * FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == list(range(1, 21))
    assert all(r["reason"] != "busy" for r in records)
    assert {r["label"] for r in records} == {"pass"}


# AC7 ------------------------------------------------------------------------------------


def test_three_inference_failures_put_up_the_fault_screen_and_it_stays(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 5)
    rig = Rig()
    code = rig.run(
        config, shoot, screen_options=closing_after(presses(5)),
        make_inspector=stub({1: "inference", 2: "inference", 3: "inference"}),
    )
    assert code == 1  # the run ended in FAULT (P04)
    records = rows(_db(tmp_path), "SELECT * FROM inspection ORDER BY seq")
    assert [(r["seq"], r["reason"]) for r in records] == [(n, "inference") for n in range(1, 6)]
    spy = rig.spies[0]
    [third] = [i for i, (kind, seq, _) in enumerate(spy.timeline) if kind == "show" and seq == 3]
    first_refresh = next(
        i for i in range(third + 1, len(spy.timeline)) if spy.timeline[i][0] == "refresh"
    )
    assert first_refresh + 1 < len(spy.timeline)
    for corners in spy.corners[first_refresh:]:  # the FAULT screen, and only it, from then on
        assert corners == {AMBER}
    faults = [s for s in spy.statuses if s.mode == "fault"]
    assert faults and {s.fault_reason for s in faults} == {"inference"}
    assert all(s.mode == "fault" for s in spy.statuses[spy.statuses.index(faults[0]):])


# AC8 ------------------------------------------------------------------------------------


def test_a_frame_out_of_the_lighting_band_is_amber_and_raises_the_banner(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    write_png(shoot / "f0001.png", textured(1, patch_grey=100))  # 0.39, far below 0.62 ± 5 %
    rig = Rig()
    assert rig.run(config, shoot, screen_options=closing_after(presses(1))) == 0
    [record] = rows(_db(tmp_path), "SELECT * FROM inspection")
    assert (record["label"], record["reason"]) == ("no_verdict", "lighting")
    spy = rig.spies[0]
    [(_, _, band)] = [entry for entry in spy.timeline if entry[0] == "show"]
    assert band == AMBER
    assert "LIGHTING" in spy.statuses[-1].warnings
    assert rig.screens[0].presented[-1].getpixel(BANNER_AT) == BANNER


# AC9 ------------------------------------------------------------------------------------


def test_refusals_exit_2_with_one_line_and_no_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 2)

    def no_display(display: DisplayConfig, title: str, clock: Any) -> ScreenLike:
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    capsys.readouterr()
    cases: list[tuple[Path | None, dict[str, Any], str]] = [
        (None, {"screen": True}, "--screen goes with --replay"),
        (shoot, {"make_screen": no_display}, "ADR-0018"),
        (tmp_path / "missing", {}, "not a directory"),
    ]
    for replay, options, words in cases:
        assert Rig().run(config, replay, **options) == 2
        out = capsys.readouterr().out
        assert out.count("\n") == 1 and words in out, out
        assert "Traceback" not in out
    assert not (tmp_path / "evidence").exists()


# AC10 -----------------------------------------------------------------------------------


def test_each_screen_iteration_sleeps_out_the_rest_of_30_ms(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 2)
    rig = Rig(clock=FakeClock(step=0.001))  # every clock read takes 1 ms
    assert rig.run(config, shoot, screen_options=closing_after(presses(2))) == 0
    screen = rig.screens[0]
    assert len(rig.sleeps) == screen.keys_calls  # one sleep per polling iteration
    assert all(0.0 <= s < 0.030 for s in rig.sleeps)
    assert max(rig.sleeps) > 0.020  # an idle iteration is mostly sleep

    other = tmp_path / "slow"
    other.mkdir()
    slow = Rig(clock=FakeClock(step=0.020))  # an iteration longer than 30 ms sleeps 0
    assert slow.run(write_config(other), shoot, screen_options=closing_after(presses(2))) == 0
    assert slow.sleeps and all(s == 0.0 for s in slow.sleeps)


def test_headless_replay_never_sleeps(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 3)
    rig = Rig()
    assert rig.run(config, shoot, screen=False) == 0
    assert rig.sleeps == [] and rig.screens == []
    assert len(rows(_db(tmp_path), "SELECT * FROM inspection")) == 3


# AC11 -----------------------------------------------------------------------------------


class FailingLoad(StubInspector):
    def load(self, config: StationConfig) -> None:
        raise ConfigError("inspector.model: no such bundle")


def test_a_failed_start_up_shows_zapusk_then_closes_the_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 2)
    rig = Rig()
    code = rig.run(
        config, shoot, make_inspector=lambda loaded: FailingLoad(loaded.config_hash)
    )
    assert code == 2
    spy = rig.spies[0]
    assert [s.mode for s in spy.statuses] == ["starting"]
    assert status_line(spy.statuses[0], spy._config.display.labels) == "ЗАПУСК"
    assert rig.screens[0].presents == 1 and rig.screens[0].close_calls == 1
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "no such bundle" in out
    assert not (tmp_path / "evidence").exists()


def test_the_help_lists_screen() -> None:
    parser = build_parser()
    station = parser._subparsers._group_actions[0].choices["station"]  # type: ignore[union-attr]
    assert "--screen" in station.format_help()


def test_a_verdict_is_never_shown_green_unless_it_passed(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 4)
    rig = Rig()
    rig.run(
        config, shoot, screen_options=closing_after(presses(4)),
        make_inspector=stub({1: "placement:no unit", 2: "crash", 3: "defect:geometry"}),
    )
    spy = rig.spies[0]
    for verdict, (_, _, band) in zip(
        spy.verdicts, [e for e in spy.timeline if e[0] == "show"], strict=True
    ):
        assert (band == GREEN) == (verdict.label is Label.PASS)
