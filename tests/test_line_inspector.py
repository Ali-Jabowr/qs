"""The line inspector on drawn frames: a burst of three frames gives one verdict, through
`inspect_burst`, through a whole Pipeline, and through `qs station --replay`. Every value here
proves the check on these drawings only, never on real footage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from evidence_records import fast_sqlite
from qs.cmd.station import main
from qs.config import LoadedConfig, load_config
from qs.contracts import Frame, Label, Verdict
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.line import LineInspector
from station_fixtures import (
    PATCH_GREY,
    VERIFIED,
    config_mapping,
    event,
    rows,
    station_args,
    write_png,
)
from test_cup import GLARE_APART, drawn_frame
from test_line_station import BLANK_AFTER, OFF_THE_ZONE_Y, PATH_X, _Station, _steel

OFF_THE_ZONE_BELOW = 175  # a rim riding here sticks out of the bottom of the zone
HASH = "c" * 64
LINE_EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-line.example.yml"


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _line_config_file(root: Path) -> Path:
    """A valid line config for the drawn 320×240 frames, with the evidence under `root`."""
    data = config_mapping(root, inspector_type="line")
    data["camera"]["type"] = "replay"
    data["inspector"]["checks"] = ["empty_cup"]
    data["trigger"]["type"] = "cup"
    # A burst of three, so the zone is long enough for the cup to be wholly in it on all three.
    data["inspector"]["line"] = {
        "zone": [20, 30, 255, 180],
        "rim_size_px": [105, 105],
        "direction": "right",
        "fps": 8.0,
        "warmth_min": 5.0,
        "burst": 3,
        "warmth_spread_max": 40.0,
        "max_speed_px_s": 320.0,
    }
    path = root / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _loaded(root: Path) -> LoadedConfig:
    return load_config(_line_config_file(root))


def _inspector(root: Path) -> LineInspector:
    inspector = LineInspector(HASH)
    inspector.load(_loaded(root).config)
    return inspector


def _frame(pixels: np.ndarray) -> Frame:
    return Frame(
        event=event(1),
        pixels=pixels,
        grabbed_at=0.0,
        camera=VERIFIED,
        reference_patch=PATCH_GREY / 255,
        capture_ms=5.0,
    )


def _crossing(contents: str, *, y: int = 120, **drawing: Any) -> list[np.ndarray]:
    """One cup carried across the view at a height `y`, then steel."""
    path = [drawn_frame(contents, centre=(x, y), seed=x, **drawing) for x in PATH_X]
    return path + _steel(BLANK_AFTER)


def _burst(pixels: np.ndarray) -> tuple[Frame, Frame, Frame]:
    """The same cup on three frames: a burst in which nothing moves."""
    return (_frame(pixels), _frame(pixels), _frame(pixels))


def _inspected(root: Path, pixels: np.ndarray) -> Verdict:
    return _inspector(root).inspect_burst(_burst(pixels))


def test_the_verdict_outlines_the_cup_it_judged(tmp_path: Path) -> None:
    verdict = _inspected(tmp_path, drawn_frame("empty"))
    assert verdict.unit_box == pytest.approx((78, 68, 105, 105), abs=4)
    assert verdict.model_id == "none"
    assert verdict.config_hash == HASH
    assert verdict.heatmap is None


@pytest.mark.parametrize(
    "pixels",
    [drawn_frame("none"), drawn_frame("none", glare_on_steel=GLARE_APART)],
    ids=["steel", "glare"],
)
def test_no_cup_is_a_placement_fault(tmp_path: Path, pixels: np.ndarray) -> None:
    with pytest.raises(PlacementFault, match="^the cup was found on 0 of 3 frames"):
        _inspected(tmp_path, pixels)


@pytest.mark.parametrize("contents", ["empty", "pale", "pink"])
def test_a_cup_partly_out_of_the_zone_is_a_placement_fault(
    tmp_path: Path, contents: str
) -> None:
    with pytest.raises(PlacementFault, match="^the cup was found on 0 of 3 frames"):
        _inspected(tmp_path, drawn_frame(contents, centre=(235, 120)))


def test_the_same_frame_gives_the_same_verdict_and_stays_unwritten(tmp_path: Path) -> None:
    pixels = drawn_frame("pale", glare_inside=5)
    pixels.setflags(write=False)
    inspector = _inspector(tmp_path)
    first = inspector.inspect_burst(_burst(pixels))
    second = inspector.inspect_burst(_burst(pixels))
    assert first.checks == second.checks


def test_inspecting_before_load_is_an_inference_error() -> None:
    with pytest.raises(InferenceError, match="not loaded"):
        LineInspector(HASH).inspect(_frame(drawn_frame("pale")))


def test_loading_a_config_without_the_line_block_is_refused(tmp_path: Path) -> None:
    data = config_mapping(tmp_path)
    path = tmp_path / "bench.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="inspector.line"):
        LineInspector(HASH).load(load_config(path).config)


def test_a_pipeline_record_lists_empty_cup_and_only_that(tmp_path: Path) -> None:
    first, second = _Station(tmp_path, _crossing("empty") + _crossing("pale")).run()
    assert first.verdict.label is Label.DEFECT
    assert second.verdict.label is Label.PASS
    assert [c.name for c in first.verdict.checks] == ["empty_cup"]
    assert [c.name for c in second.verdict.checks] == ["empty_cup"]
    assert first.verdict.model_id == "none"


def test_station_replay_runs_end_to_end_on_a_line_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _line_config_file(tmp_path)
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    frames = (
        _crossing("empty", glare_on_steel=GLARE_APART)
        + _crossing("pale", glare_inside=5)
        + _crossing("pink")
        + _crossing("pale", y=OFF_THE_ZONE_Y)
        + _crossing("pale", y=OFF_THE_ZONE_BELOW)
        + _steel(BLANK_AFTER)  # a burst for the last cup to be captured from
    )
    for index, pixels in enumerate(frames):
        write_png(shoot / f"f{index:03d}.png", pixels)

    assert main(station_args(config, shoot)) == 0

    out = capsys.readouterr().out
    assert "events 5 · pass 2 · defect 1 · no_verdict 2" in out
    assert "no_verdict by reason: placement 2" in out
    database = tmp_path / "evidence" / "qs.db"
    records = rows(database, "SELECT * FROM inspection ORDER BY seq")
    assert [r["label"] for r in records] == ["defect", "pass", "pass", "no_verdict", "no_verdict"]
    assert [r["reason"] for r in records] == [None, None, None, "placement", "placement"]
    assert [r["model_id"] for r in records] == ["none"] * 5
    checks = rows(
        database,
        "SELECT i.seq, c.name, c.passed, c.limit_value, c.failed_on FROM inspection_check c"
        " JOIN inspection i ON i.id = c.inspection_id ORDER BY i.seq",
    )
    assert [tuple(c) for c in checks] == [
        (1, "empty_cup", 0, 5.0, "warmth<min"),
        (2, "empty_cup", 1, 5.0, None),
        (3, "empty_cup", 1, 5.0, None),
    ]


def test_the_line_example_is_refused_without_replay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(station_args(LINE_EXAMPLE, None)) == 2
    assert "trigger.type 'cup' has no live trigger" in capsys.readouterr().out
