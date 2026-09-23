"""`qs station --config C --replay DIR`: one written record per image, a summary, an exit code
(#69). Through `main(args, *, factories)`, P05's command pattern; the real `SqliteSink` on
`tmp_path`, the stub inspector, synthetic PNGs."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_records import fast_sqlite
from qs.annunciator import NullAnnunciator
from qs.cmd.station import main
from qs.config import LoadedConfig, load_config
from qs.contracts import Label
from qs.inspector.stub import StubInspector
from qs.sources.webcam import measure_patch
from station_fixtures import (
    PATCH_ROI,
    StepWall,
    rows,
    sha256_file,
    station_args,
    textured,
    write_config,
    write_frames,
    write_frames_csv,
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_sqlite(monkeypatch)


def _stub(script: dict[int, str]):  # type: ignore[no-untyped-def]
    def make(loaded: LoadedConfig) -> StubInspector:
        return StubInspector(loaded.config_hash, script)

    return make


def _db(tmp_path: Path) -> Path:
    return tmp_path / "evidence" / "qs.db"


def test_five_frames_become_five_passing_records(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 5)
    assert main(station_args(config, tmp_path / "shoot")) == 0
    loaded = load_config(config)
    database = _db(tmp_path)
    [run] = rows(database, "SELECT * FROM run")
    assert run["ended_utc"] is not None
    records = rows(database, "SELECT * FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == [1, 2, 3, 4, 5]
    for r in records:
        assert r["label"] == "pass"
        assert r["model_id"] == "stub"
        assert r["threshold_id"] == loaded.config.threshold.id
        assert r["config_hash"] == loaded.config_hash
        assert r["image_status"] == "stored"
        image = tmp_path / "evidence" / r["image_path"]
        assert sha256_file(image) == r["image_sha256"]
        checks = rows(
            database, "SELECT * FROM inspection_check WHERE inspection_id = ?", (r["id"],)
        )
        assert sorted(c["name"] for c in checks) == ["appearance", "geometry"]
        assert all(c["passed"] == 1 for c in checks)


def test_a_scripted_appearance_defect_has_its_heatmap(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 5)
    code = main(
        station_args(config, tmp_path / "shoot"), make_inspector=_stub({3: "defect:appearance"})
    )
    assert code == 0
    records = rows(_db(tmp_path), "SELECT * FROM inspection ORDER BY seq")
    assert [r["label"] for r in records] == ["pass", "pass", "defect", "pass", "pass"]
    defect = records[2]
    assert defect["heatmap_path"] is not None
    assert (tmp_path / "evidence" / defect["heatmap_path"]).read_bytes().startswith(b"\x89PNG")
    [appearance] = rows(
        _db(tmp_path),
        "SELECT * FROM inspection_check WHERE inspection_id = ? AND name = 'appearance'",
        (defect["id"],),
    )
    assert appearance["passed"] == 0
    assert appearance["failed_on"] == "score>threshold"
    assert all(r["heatmap_path"] is None for r in records if r["seq"] != 3)


def _outcomes(database: Path, run_id: str) -> list[tuple[object, ...]]:
    return [
        (r["seq"], r["label"], r["image_sha256"], r["value"], r["passed"])
        for r in rows(
            database,
            "SELECT i.seq, i.label, i.image_sha256, c.value, c.passed FROM inspection i"
            " JOIN inspection_check c ON c.inspection_id = i.id WHERE i.run_id = ?"
            " ORDER BY i.seq, c.name",
            (run_id,),
        )
    ]


def test_replay_is_in_name_order_and_repeatable(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 4)
    (shoot / "f0002.png").rename(shoot / "a_first.png")  # sorts before every f….png
    script = _stub({1: "defect:geometry", 3: "defect:appearance"})
    wall = StepWall()
    assert main(station_args(config, shoot), make_inspector=script, wall=wall) == 0
    assert main(station_args(config, shoot), make_inspector=script, wall=wall) == 0
    runs = [r["run_id"] for r in rows(_db(tmp_path), "SELECT run_id FROM run ORDER BY run_id")]
    assert len(runs) == 2 and runs[0] != runs[1]
    first, second = (_outcomes(_db(tmp_path), run) for run in runs)
    assert first == second
    # seq 1 is a_first.png: its stored JPEG is the one made from that file's pixels
    [seq1] = rows(_db(tmp_path), "SELECT * FROM inspection WHERE run_id = ? AND seq = 1",
                  (runs[0],))
    assert seq1["label"] == "defect"
    assert seq1["reference_patch"] == pytest.approx(
        measure_patch(textured(2), tuple(PATCH_ROI))  # type: ignore[arg-type]
    )


def test_frames_csv_gives_the_camera_values_and_its_absence_none(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    recorded = tmp_path / "recorded"
    bare = tmp_path / "bare"
    paths = write_frames(recorded, 2)
    write_frames(bare, 2)
    write_frames_csv(recorded, [
        {"file": p.name, "exposure_us": 10000, "gain": 3, "wb_kelvin": 4500, "focus": 30,
         "verified": "true", "reference_patch": 0.99, "sha256": sha256_file(p),
         "status": "saved"}
        for p in paths
    ])
    wall = StepWall()
    assert main(station_args(config, recorded), wall=wall) == 0
    assert main(station_args(config, bare), wall=wall) == 0
    runs = [r["run_id"] for r in rows(_db(tmp_path), "SELECT run_id FROM run ORDER BY run_id")]
    with_csv = rows(_db(tmp_path), "SELECT * FROM inspection WHERE run_id = ?", (runs[0],))
    without = rows(_db(tmp_path), "SELECT * FROM inspection WHERE run_id = ?", (runs[1],))
    for r in with_csv:
        assert (r["exposure_us"], r["gain"], r["wb_kelvin"], r["focus"]) == (10000, 3, 4500, 30)
        assert r["camera_verified"] == 1
    for r in without:
        assert (r["exposure_us"], r["gain"], r["wb_kelvin"], r["focus"]) == (None,) * 4
        assert r["camera_verified"] == 0
    for r in (*with_csv, *without):  # measured from the pixels, never from frames.csv
        expected = measure_patch(textured(r["seq"]), tuple(PATCH_ROI))  # type: ignore[arg-type]
        assert r["reference_patch"] == pytest.approx(expected)


def test_refusals_exit_2_with_one_line_and_no_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = write_config(tmp_path)
    shoot = tmp_path / "shoot"
    write_frames(shoot, 2)
    empty = tmp_path / "empty"
    empty.mkdir()
    bad = tmp_path / "bad.yml"
    bad.write_text(good.read_text(encoding="utf-8").replace("version: 1", "version: one"))
    appearance = write_config(tmp_path, name="appearance.yml", inspector_type="appearance")
    cases = [
        (bad, shoot, "config:"),
        (good, tmp_path / "missing", "not a directory"),
        (good, empty, "no *.png"),
        (good, None, "stub runs only with --replay"),
        (appearance, None, "--replay DIR is required"),
    ]
    capsys.readouterr()
    for config, replay, words in cases:
        assert main(station_args(config, replay)) == 2
        out = capsys.readouterr().out
        assert out.count("\n") == 1 and words in out, out
        assert "Traceback" not in out
    assert not (tmp_path / "evidence").exists()


def test_summary_and_one_annunciator_call_per_event(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 6)
    shown = NullAnnunciator()
    code = main(
        station_args(config, tmp_path / "shoot"),
        make_inspector=_stub({2: "defect:geometry", 5: "placement:no unit"}),
        make_annunciator=lambda: shown,
    )
    assert code == 0
    out = capsys.readouterr().out
    [run] = rows(_db(tmp_path), "SELECT run_id FROM run")
    assert f"run {run['run_id']}" in out
    assert str(_db(tmp_path)) in out
    assert "events 6 · pass 4 · defect 1 · no_verdict 1" in out
    assert "records written 6" in out and "lost 0" in out
    assert [verdict.label for _, verdict in shown.calls] == [
        Label.PASS, Label.DEFECT, Label.PASS, Label.PASS, Label.NO_VERDICT, Label.PASS
    ]
    assert [frame.event.seq for frame, _ in shown.calls if frame is not None] == [1, 2, 3, 4, 5, 6]
