"""`qs unit-thresholds` and `qs unit-report` (#177).

The rig is the one of the unit-scores tests (footage with calibrate C-1, train T-1 and test E-1,
whose one labelled full pass is at frame 5). The scoring of the test clip is a stand-in that
records its calls, so a test can say the test clip was never opened.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from lane_rig import fake_git
from qs.__main__ import build_parser
from qs.cmd import unit_report, unit_thresholds
from qs.lane.split import file_sha256
from qs.lane.unit_limits import load_limits
from qs.lane.unit_scores import ScoreRow, write_scores
from test_lane_unit_scores import make_rig


def _calibrate_scores(path: Path, n: int = 40) -> None:
    write_scores([ScoreRow("C-1.mkv", i, i, "full", i, "", float(1 + i), 70.0, 150.0, 120.0, None)
                  for i in range(n)], path)


def _thresholds(tmp_path: Path, paths: dict[str, Path], n: int = 40) -> tuple[int, Path]:
    scores, out = tmp_path / "scores.csv", tmp_path / "unit-thresholds.yml"
    _calibrate_scores(scores, n)
    args = build_parser().parse_args([
        "unit-thresholds", "--scores", str(scores), "--lane", str(paths["lane"]),
        "--unit", str(paths["unit"]), "--out", str(out), "--models", str(tmp_path / "models")])
    return unit_thresholds.main(args), out


class Scoring:
    """The stand-in for `score_block`: one row for E-1's full pass; it records its calls."""

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []

    def __call__(self, labels: Any, footage: Path, videos: Sequence[str],
                 *rest: Any) -> list[ScoreRow]:
        self.calls.append(list(videos))
        return [ScoreRow("E-1.mkv", 5, 1, "full", 5, "", 1.0, 70.0, 150.0, 120.0, None)]


def _report(
    tmp_path: Path, paths: dict[str, Path], thresholds: Path, scoring: Scoring,
    *extra: str, dirty: str = "",
) -> int:
    args = build_parser().parse_args([
        "unit-report", "--thresholds", str(thresholds), "--lane", str(paths["lane"]),
        "--unit", str(paths["unit"]), "--labels", str(paths["labels"]),
        "--footage", str(paths["footage"]), "--split", str(paths["split"]),
        "--models", str(tmp_path / "models"), *extra])
    return unit_report.main(args, git=fake_git(dirty=dirty), score=scoring)


# --- qs unit-thresholds -----------------------------------------------------------------


def test_the_limits_are_written_with_the_hashes_of_the_inputs(tmp_path: Path) -> None:
    paths = make_rig(tmp_path)
    code, out = _thresholds(tmp_path, paths)
    assert code == 0
    written = load_limits(out)
    assert written.lane_sha256 == file_sha256(paths["lane"])
    assert written.unit_sha256 == file_sha256(paths["unit"])
    assert written.states["full"].colour_spread_max == 40.0


def test_an_existing_output_file_is_refused_and_left_alone(tmp_path: Path) -> None:
    paths = make_rig(tmp_path)
    out = tmp_path / "unit-thresholds.yml"
    out.write_text("keep me")
    assert _thresholds(tmp_path, paths)[0] == 2
    assert out.read_text() == "keep me"


def test_a_state_with_29_rows_is_named_and_a_run_with_no_limits_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_rig(tmp_path)
    assert _thresholds(tmp_path, paths, n=29)[0] == 1
    assert "full: 29 usable rows, fewer than 30; no limits" in capsys.readouterr().out
    assert not (tmp_path / "unit-thresholds.yml").exists()


# --- qs unit-report ---------------------------------------------------------------------


def test_the_table_of_the_test_clip_is_printed_and_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_rig(tmp_path)
    thresholds = _thresholds(tmp_path, paths)[1]
    scoring = Scoring()
    assert _report(tmp_path, paths, thresholds, scoring, "--out", str(tmp_path / "t.md")) == 0
    out = capsys.readouterr().out
    assert "| good units flagged | 0 of 1 | 0 of 1 | 0 of 1 |" in out
    assert "RESCORED" not in out
    assert scoring.calls == [["E-1.mkv"]]
    assert (tmp_path / "t.md").read_text().strip() == out.strip()


def test_without_a_thresholds_file_no_test_clip_is_opened(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_rig(tmp_path)
    scoring = Scoring()
    assert _report(tmp_path, paths, tmp_path / "missing.yml", scoring) == 2
    assert scoring.calls == []
    assert "unit-thresholds" in capsys.readouterr().out


@pytest.mark.parametrize("changed", ["lane", "unit"])
def test_a_file_changed_after_the_thresholds_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], changed: str
) -> None:
    paths = make_rig(tmp_path)
    thresholds = _thresholds(tmp_path, paths)[1]
    paths[changed].write_bytes(paths[changed].read_bytes() + b"\n# edited\n")
    scoring = Scoring()
    assert _report(tmp_path, paths, thresholds, scoring) == 2
    assert scoring.calls == []
    assert f"{changed}.yml" in capsys.readouterr().out


def test_a_dirty_tree_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = make_rig(tmp_path)
    thresholds = _thresholds(tmp_path, paths)[1]
    scoring = Scoring()
    assert _report(tmp_path, paths, thresholds, scoring, dirty=" M src/qs/x.py") == 2
    assert scoring.calls == []
    assert "uncommitted" in capsys.readouterr().out


def test_a_test_clip_changed_after_the_split_is_refused(tmp_path: Path) -> None:
    paths = make_rig(tmp_path)
    thresholds = _thresholds(tmp_path, paths)[1]
    (paths["footage"] / "E-1.mkv").write_bytes(b"changed")
    scoring = Scoring()
    assert _report(tmp_path, paths, thresholds, scoring) == 2
    assert scoring.calls == []


def test_the_second_run_says_rescored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_rig(tmp_path)
    thresholds = _thresholds(tmp_path, paths)[1]
    assert _report(tmp_path, paths, thresholds, Scoring()) == 0
    capsys.readouterr()
    assert _report(tmp_path, paths, thresholds, Scoring()) == 0
    assert capsys.readouterr().out.splitlines()[0] == "RESCORED (run 2)"


def test_a_defect_in_a_non_test_clip_is_refused_before_scoring(tmp_path: Path) -> None:
    paths = make_rig(tmp_path)
    thresholds = _thresholds(tmp_path, paths)[1]
    defects = tmp_path / "defects.yml"
    defects.write_text("version: 1\ndefects:\n"
                       "  - {video: T-1.mkv, frame: 5, kind: patch, unit: D01}\n")
    scoring = Scoring()
    assert _report(tmp_path, paths, thresholds, scoring, "--defects", str(defects)) == 2
    assert scoring.calls == []


def test_a_defect_of_the_test_clip_is_counted_as_a_unit_not_caught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_rig(tmp_path)
    thresholds = _thresholds(tmp_path, paths)[1]
    defects = tmp_path / "defects.yml"
    defects.write_text("version: 1\ndefects:\n"
                       "  - {video: E-1.mkv, frame: 5, kind: patch, unit: D01}\n")
    assert _report(tmp_path, paths, thresholds, Scoring(), "--defects", str(defects)) == 0
    out = capsys.readouterr().out
    assert "| defective units caught | 0 of 1 | 0 of 1 | 0 of 1 |" in out
    assert "- D01 (patch), 1 passes: colour spread 0 of 1" in out
