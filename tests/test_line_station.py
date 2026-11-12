"""The whole line station on saved frames: drawn cups moving across the view, through the cup
trigger, the line replay and a real `Pipeline`, asserting on records. The clock is fake and moves
one frame per main-loop turn. Every value here proves the station on these drawings only, never
on real footage.

In these frames a rim of about 105 px is wholly in the zone (x 20–260, y 30–210) for a centre of
x 73–207, when it rides at y 120. A cup one step before the zone is at x 50, and one step past it
at x 210, with steps of 40 px.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig, load_config
from qs.contracts import Frame, InspectionRecord, Label, Verdict
from qs.inspector.line import LineInspector
from qs.pipeline import Pipeline
from qs.sources.line_replay import line_replay_pair
from qs.sources.replay import replay_files
from station_fixtures import (
    FakeClock,
    RecordingSink,
    StepWall,
    config_mapping,
    rows,
    station_args,
    write_png,
)
from test_cup import drawn_frame

FPS = 8.0
PATH_X = (10, 50, 90, 130, 170, 210, 250, 290, 330)
BLANK_AFTER = 4  # enough frames of steel for a cup gone from view to fire and be captured
OFF_THE_ZONE_Y = 65  # a rim riding here sticks out of the top of the zone all the way across


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _config_file(root: Path, *, line: bool = True, direction: str = "right") -> Path:
    data = config_mapping(root, inspector_type="line" if line else "stub")
    data["trigger"]["type"] = "cup"
    if line:
        data["inspector"]["checks"] = ["empty_cup"]
        data["camera"]["type"] = "replay"
        # A burst of one: each cup is judged on the one frame captured after its event.
        data["inspector"]["line"] = {
            "zone": [20, 30, 240, 180],
            "rim_size_px": [105, 105],
            "direction": direction,
            "fps": FPS,
            "warmth_min": 5.0,
            "burst": 1,
            "burst_min_frames": 1,
            "warmth_spread_max": 40.0,
            "max_speed_px_s": 320.0,
        }
    path = root / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _crossing(contents: str, *, y: int = 120) -> list[np.ndarray]:
    """One cup carried across the view, then steel. Each frame has its own noise."""
    return [drawn_frame(contents, centre=(x, y), seed=x) for x in PATH_X] + _steel(BLANK_AFTER)


def _steel(count: int) -> list[np.ndarray]:
    return [drawn_frame("none", seed=100 + index) for index in range(count)]


def _at(contents: str, xs: Sequence[int]) -> list[np.ndarray]:
    return [drawn_frame(contents, centre=(x, 120), seed=x) for x in xs]


def _held(contents: str, x: int, count: int) -> list[np.ndarray]:
    """A cup standing still: only each frame's noise differs."""
    return [drawn_frame(contents, centre=(x, 120), seed=200 + i) for i in range(count)]


def _pair_crossing(xs: Sequence[int]) -> list[np.ndarray]:
    """A full cup at each x with an empty one 120 px behind it, the view split between them."""
    frames = []
    for x in xs:
        pixels = drawn_frame("pale", centre=(x, 120), seed=x)
        split = x - 60
        behind = drawn_frame("empty", centre=(x - 120, 120), seed=x)
        pixels[:, :split] = behind[:, :split]
        frames.append(pixels)
    return frames


def _write_shoot(root: Path, frames: Sequence[np.ndarray]) -> Path:
    shoot = root / "shoot"
    shoot.mkdir()
    for index, pixels in enumerate(frames):
        write_png(shoot / f"f{index:04d}.png", pixels)
    return shoot


class _Inspected(LineInspector):
    """The real inspector, keeping what it was given."""

    def __init__(self, config_hash: str) -> None:
        super().__init__(config_hash)
        self.seen: list[Frame] = []

    def inspect_burst(self, frames: Sequence[Frame]) -> Verdict:
        self.seen.extend(frames)
        return super().inspect_burst(frames)


class _Station:
    """A `Pipeline` over the frames, driven as the main loop drives it, a frame per turn."""

    def __init__(
        self, root: Path, frames: Sequence[np.ndarray], *, direction: str = "right"
    ) -> None:
        shoot = _write_shoot(root, frames)
        self.count = len(frames)
        self.loaded: LoadedConfig = load_config(_config_file(root, direction=direction))
        self.clock = FakeClock()
        self.trigger, self.source = line_replay_pair(
            self.loaded.config, replay_files(shoot), "shoot", clock=self.clock, wall=StepWall()
        )
        self.inspector = _Inspected(self.loaded.config_hash)
        self.inspector.load(self.loaded.config)
        self.sink = RecordingSink()
        self.pipeline = Pipeline(
            config=self.loaded.config,
            config_hash=self.loaded.config_hash,
            run_id="run",
            source=self.source,
            inspector=self.inspector,
            annunciator=NullAnnunciator(),
            sink=self.sink,
            clock=self.clock,
            wall=StepWall(),
        )

    def run(self) -> list[InspectionRecord]:
        self.pipeline.start()
        for _ in range(2 * self.count + 8):
            event = self.trigger.poll()
            if event is not None:
                self.pipeline.handle(event)
            self.clock.advance(1 / FPS)
        assert self.trigger.exhausted
        return self.sink.records


def _labels(records: list[InspectionRecord]) -> list[Label]:
    return [record.verdict.label for record in records]


def _reasons(records: list[InspectionRecord]) -> list[str | None]:
    return [record.verdict.reason for record in records]


def _seqs(records: list[InspectionRecord]) -> list[int]:
    return [record.event.seq for record in records]


def test_moving_cups_give_one_record_each_in_order(tmp_path: Path) -> None:
    frames = _crossing("empty") + _crossing("pale") + _crossing("pink") + _crossing("empty")
    records = _Station(tmp_path, frames).run()
    assert _seqs(records) == [1, 2, 3, 4]
    assert _labels(records) == [Label.DEFECT, Label.PASS, Label.PASS, Label.DEFECT]
    assert _reasons(records) == [None, None, None, None]


def test_two_cups_in_view_at_once_give_one_record_each(tmp_path: Path) -> None:
    """Cups 120 px apart: the one ahead has a full cup, the one behind is empty."""
    frames = _pair_crossing((130, 170, 210, 250, 290, 330, 370, 410))
    records = _Station(tmp_path, frames + _steel(BLANK_AFTER)).run()
    assert _labels(records) == [Label.PASS, Label.DEFECT]


@pytest.mark.parametrize(
    ("direction", "centres"),
    [
        ("left", [(x, 120) for x in (330, 290, 250, 210, 170, 130, 90, 50, 10)]),
        ("down", [(130, y) for y in (-30, 10, 50, 90, 130, 170, 210, 250, 290)]),
        ("up", [(130, y) for y in (290, 250, 210, 170, 130, 90, 50, 10, -30)]),
    ],
    ids=["left", "down", "up"],
)
def test_a_cup_travelling_any_configured_way_gives_one_record(
    tmp_path: Path, direction: str, centres: list[tuple[int, int]]
) -> None:
    frames = [drawn_frame("pale", centre=c, seed=i) for i, c in enumerate(centres)]
    records = _Station(tmp_path, frames + _steel(BLANK_AFTER), direction=direction).run()
    assert _labels(records) == [Label.PASS]


def test_a_gap_between_cups_gives_no_record(tmp_path: Path) -> None:
    frames = _steel(6) + _crossing("pale") + _steel(10) + _crossing("pale") + _steel(6)
    records = _Station(tmp_path, frames).run()
    assert _labels(records) == [Label.PASS, Label.PASS]


def test_a_chain_of_nothing_but_gap_gives_no_record(tmp_path: Path) -> None:
    assert _Station(tmp_path, _steel(12)).run() == []


def test_a_stopped_chain_gives_no_further_verdict_for_the_same_cup(tmp_path: Path) -> None:
    moving_in = _at("pale", (10, 50, 90, 130))
    stopped = _held("pale", 130, 40)
    moving_on = _at("pale", (170, 210, 250, 290, 330))
    records = _Station(tmp_path, moving_in + stopped + moving_on + _steel(BLANK_AFTER)).run()
    assert _labels(records) == [Label.PASS]


def test_a_cup_seen_but_never_wholly_in_the_zone_gives_one_placement(tmp_path: Path) -> None:
    records = _Station(tmp_path, _crossing("pale", y=OFF_THE_ZONE_Y)).run()
    assert _labels(records) == [Label.NO_VERDICT]
    assert _reasons(records) == ["placement"]


def test_a_cup_never_seen_gives_no_event_and_no_record(tmp_path: Path) -> None:
    """A rim too small for `rim_size_px` is a cup the finder cannot see, between two it can."""
    unseen = [drawn_frame("pale", centre=(x, 120), axes=(20, 20), seed=x) for x in PATH_X]
    station = _Station(tmp_path, _crossing("pale") + unseen + _crossing("empty"))
    records = station.run()
    assert _seqs(records) == [1, 2]
    assert _labels(records) == [Label.PASS, Label.DEFECT]


def test_a_cup_on_the_last_frame_has_a_record_saying_there_was_no_frame_to_judge(
    tmp_path: Path,
) -> None:
    records = _Station(tmp_path, _at("pale", (10, 50, 90))).run()
    assert _labels(records) == [Label.NO_VERDICT]
    assert _reasons(records) == ["capture"]


def test_the_verdict_comes_from_the_captured_frame_not_the_preview(tmp_path: Path) -> None:
    """The cup is empty on the frame the trigger fires on and full on the next one."""
    frames = (
        _at("empty", (50, 90)) + _at("pale", (130,)) + _at("empty", (170, 210, 250, 290, 330))
        + _steel(BLANK_AFTER)
    )
    assert _labels(_Station(tmp_path, frames).run()) == [Label.PASS]


def test_the_stored_image_is_the_captured_frame_not_the_preview(tmp_path: Path) -> None:
    frames = _crossing("pale")
    [record] = _Station(tmp_path, frames).run()
    assert record.image_jpeg is not None
    stored = cv2.imdecode(np.frombuffer(record.image_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert record.verdict.unit_box is not None
    x, _, width, _ = record.verdict.unit_box
    assert x + width // 2 == pytest.approx(130, abs=3)
    assert float(np.abs(stored.astype(np.int16) - frames[3].astype(np.int16)).mean()) < 3.0


def test_only_captured_frames_are_inspected(tmp_path: Path) -> None:
    station = _Station(tmp_path, _crossing("pale") + _crossing("empty"))
    records = station.run()
    assert len(records) == 2
    assert [frame.event.seq for frame in station.inspector.seen] == [1, 2]
    assert station.inspector.seen[0].event == records[0].event
    assert station.inspector.seen[0].grabbed_at == 1000.375


def test_the_command_scores_a_drawn_recording_with_a_record_per_cup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shoot = _write_shoot(
        tmp_path,
        _steel(3) + _crossing("empty") + _crossing("pale")
        + _crossing("pale", y=OFF_THE_ZONE_Y) + _steel(BLANK_AFTER),
    )
    config = _config_file(tmp_path)

    code = main(station_args(config, shoot), clock=FakeClock(step=1 / 256))

    out = capsys.readouterr().out
    assert code == 0
    assert "events 3 · pass 1 · defect 1 · no_verdict 1" in out
    assert "no_verdict by reason: placement 1" in out
    records = rows(tmp_path / "evidence" / "qs.db", "SELECT * FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert [r["label"] for r in records] == ["defect", "pass", "no_verdict"]
    assert [r["model_id"] for r in records] == ["none", "none", "none"]


def test_the_command_refuses_a_cup_trigger_with_no_zone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config_file(tmp_path, line=False)
    assert main(station_args(config, tmp_path)) == 2
    assert "trigger.type: cup needs inspector.line" in capsys.readouterr().out


def test_the_command_has_no_live_cup_trigger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(station_args(_config_file(tmp_path), None)) == 2
    assert "trigger.type 'cup' has no live trigger" in capsys.readouterr().out


def test_the_command_refuses_a_cup_trigger_on_the_screen_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = station_args(_config_file(tmp_path), tmp_path)
    args.screen = True
    assert main(args) == 2
    assert "--screen has no cup trigger" in capsys.readouterr().out
