"""The line inspector on drawn frames: one frame gives one verdict, through `inspect`, through a
whole Pipeline, and through `qs station --replay`. Every value here proves the check on these
drawings only, never on real footage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig, load_config
from qs.contracts import Frame, Label, Verdict
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.line import LineInspector
from qs.pipeline import Pipeline
from qs.sources.replay import replay_files, replay_pair
from station_fixtures import (
    PATCH_GREY,
    VERIFIED,
    FakeClock,
    RecordingSink,
    StepWall,
    config_mapping,
    event,
    rows,
    station_args,
    write_png,
)
from test_cup import GLARE_ACROSS, GLARE_APART, drawn_frame

HASH = "c" * 64
BIG_GLARE_INSIDE = ([(106, 98), (154, 98), (154, 142), (106, 142)],)
LINE_EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-line.example.yml"


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _line_config_file(root: Path) -> Path:
    """A valid line config for the drawn 320×240 frames, with the evidence under `root`."""
    data = config_mapping(root, inspector_type="line")
    data["inspector"]["checks"] = ["empty_cup"]
    data["inspector"]["line"] = {
        "zone": [20, 0, 240, 240],
        "rim_axis_px": [50, 140],
        "blob_share_min": 0.3,
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


def _inspected(root: Path, pixels: np.ndarray) -> Verdict:
    return _inspector(root).inspect(_frame(pixels))


@pytest.mark.parametrize(
    "pixels",
    [
        drawn_frame("pale"),
        drawn_frame("pink"),
        drawn_frame("pale", axes=(50, 34), angle=25.0),
        drawn_frame("pink", glare_on_steel=GLARE_APART),
        drawn_frame("pale", glare_on_steel=GLARE_ACROSS),
        drawn_frame("pale", glare_inside=5),
        drawn_frame("pink", glare_inside=5),
    ],
    ids=["pale", "pink", "pale_ellipse", "pink_glare_steel", "pale_glare_rim", "pale_glare_in",
         "pink_glare_in"],
)
def test_a_full_cup_passes_on_the_empty_cup_check(tmp_path: Path, pixels: np.ndarray) -> None:
    verdict = _inspected(tmp_path, pixels)
    assert verdict.label is Label.PASS
    [check] = verdict.checks
    assert check.name == "empty_cup"
    assert check.passed is True
    assert check.limit == 0.3
    assert check.value is not None and check.value >= 0.5


@pytest.mark.parametrize(
    "pixels",
    [
        drawn_frame("empty"),
        drawn_frame("empty", axes=(50, 34), angle=25.0),
        drawn_frame("empty", glare_on_steel=GLARE_APART),
        drawn_frame("empty", glare_on_steel=GLARE_ACROSS),
        drawn_frame("empty", glare_inside=5),
        drawn_frame("empty", glare_on_steel=BIG_GLARE_INSIDE),
    ],
    ids=["empty", "ellipse", "glare_steel", "glare_rim", "glare_in", "big_glare_in"],
)
def test_an_empty_cup_is_a_defect_on_the_empty_cup_check(
    tmp_path: Path, pixels: np.ndarray
) -> None:
    verdict = _inspected(tmp_path, pixels)
    assert verdict.label is Label.DEFECT
    [check] = verdict.checks
    assert check.name == "empty_cup"
    assert check.passed is False
    assert check.limit == 0.3
    assert check.value is not None and check.value <= 0.1
    assert check.failed_on == "blob_share<min"


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
    with pytest.raises(PlacementFault, match="no cup wholly in the zone"):
        _inspected(tmp_path, pixels)


@pytest.mark.parametrize("contents", ["empty", "pale", "pink"])
def test_a_cup_partly_out_of_the_zone_is_a_placement_fault(
    tmp_path: Path, contents: str
) -> None:
    with pytest.raises(PlacementFault, match="no cup wholly in the zone"):
        _inspected(tmp_path, drawn_frame(contents, centre=(235, 120)))


def test_the_same_frame_gives_the_same_verdict_and_stays_unwritten(tmp_path: Path) -> None:
    pixels = drawn_frame("pale", glare_inside=5)
    pixels.setflags(write=False)
    inspector = _inspector(tmp_path)
    first = inspector.inspect(_frame(pixels))
    second = inspector.inspect(_frame(pixels))
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
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    write_png(shoot / "f1.png", drawn_frame("empty"))
    write_png(shoot / "f2.png", drawn_frame("pale"))
    loaded = _loaded(tmp_path)
    clock = FakeClock(step=0.001)
    trigger, source = replay_pair(
        loaded.config, replay_files(shoot), "shoot", clock=clock, wall=StepWall()
    )
    inspector = LineInspector(loaded.config_hash)
    inspector.load(loaded.config)
    sink = RecordingSink()
    pipeline = Pipeline(
        config=loaded.config,
        config_hash=loaded.config_hash,
        run_id="run",
        source=source,
        inspector=inspector,
        annunciator=NullAnnunciator(),
        sink=sink,
        clock=clock,
        wall=StepWall(),
    )
    pipeline.start()
    first = pipeline.handle(trigger.poll())  # type: ignore[arg-type]
    second = pipeline.handle(trigger.poll())  # type: ignore[arg-type]
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
    write_png(shoot / "f1_empty.png", drawn_frame("empty", glare_on_steel=GLARE_APART))
    write_png(shoot / "f2_pale.png", drawn_frame("pale", glare_inside=5))
    write_png(shoot / "f3_pink.png", drawn_frame("pink"))
    write_png(shoot / "f4_none.png", drawn_frame("none"))
    write_png(shoot / "f5_partly.png", drawn_frame("pale", centre=(235, 120)))

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
        (1, "empty_cup", 0, 0.3, "blob_share<min"),
        (2, "empty_cup", 1, 0.3, None),
        (3, "empty_cup", 1, 0.3, None),
    ]


def test_the_line_example_is_refused_without_replay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(station_args(LINE_EXAMPLE, None)) == 2
    assert "trigger.type 'cup' has no live trigger" in capsys.readouterr().out
