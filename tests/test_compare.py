"""`qs compare`: the test block scored once, behind the freeze (#160).

E-1 holds eight labelled passes (full@100, empty@200, empty@300, full@400, full@500, gap@600,
empty@700, gap@800). The drawn probabilities give method C these runs, at threshold 0.5:
100 ГОДЕН · 200 БРАК · 300 (3 frames, P(empty) 0.3) ГОДЕН · 400 (2 frames) НЕТ РЕШЕНИЯ ·
500 БРАК · 600 no run · 700 no run · 800 ГОДЕН.
So empty: 1 caught of 3 (1 ГОДЕН, 1 no verdict); full: 1 БРАК, 1 ГОДЕН, 1 НЕТ РЕШЕНИЯ;
gap: 1 ГОДЕН, 1 no verdict. Bounds worked out with scipy's beta quantiles.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from lane_rig import HEAD, Rig, clock, fake_git, make_rig, run_freeze
from qs.__main__ import build_parser
from qs.cmd import compare
from qs.contracts import Label
from qs.eval.line_score import Judged
from qs.labels import Cup

Row = tuple[float, float, float]
F: Row = (0.9, 0.05, 0.05)
E: Row = (0.05, 0.9, 0.05)
N: Row = (0.05, 0.05, 0.9)
M: Row = (0.65, 0.3, 0.05)  # a cup frame with P(empty) 0.3


def _probs(length: int, spans: dict[tuple[int, int], Row]) -> list[Row]:
    rows = [N] * length
    for (first, last), row in spans.items():
        rows[first : last + 1] = [row] * (last - first + 1)
    return rows


E1_PROBS = _probs(1200, {(98, 102): F, (198, 202): E, (298, 300): M, (399, 400): F,
                         (498, 502): E, (799, 803): F})

C_ROW = "| C | 1 of 3 | 0.0170 | 1 of 3 | 0.8646 | 1 of 3 | 0.8646 | 1 of 2 | 0.9747 |"

B_SCORES = {100: 0.1, 200: 0.35, 300: 0.9, 400: 0.2, 500: 0.41, 700: 0.45}


class Reader:
    """The frame-probability reader; it records which clips it was asked to read."""

    def __init__(self) -> None:
        self.read: list[str] = []

    def __call__(self, video: Path) -> list[Row]:
        self.read.append(video.name)
        return E1_PROBS


def _run(rig: Rig, reader: Reader, *extra: str, head: str = HEAD, **inject: object) -> int:
    args = build_parser().parse_args(
        ["compare", "--freeze", str(rig.freeze), "--labels", str(rig.labels),
         "--footage", str(rig.footage), "--models", str(rig.models), *extra])
    return compare.main(args, git=fake_git(head), clock=clock, probs_of=reader,
                        **inject)  # type: ignore[arg-type]


def _frozen(tmp_path: Path, threshold: float = 0.5, *extra: str) -> Rig:
    rig = make_rig(tmp_path, threshold)
    assert run_freeze(rig, *extra) == 0
    return rig


def test_the_table_of_the_drawn_test_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path)
    assert _run(rig, Reader()) == 0
    out = capsys.readouterr().out
    assert C_ROW in out
    assert "| full | 1 | 1 | 1 | 0 |" in out
    assert "| empty | 1 | 1 | 0 | 1 |" in out
    assert "| gap | 1 | 0 | 0 | 1 |" in out


def test_only_the_test_clips_are_read(tmp_path: Path) -> None:
    rig = _frozen(tmp_path)
    reader = Reader()
    assert _run(rig, reader) == 0
    assert reader.read == ["E-1.mkv"]


def test_the_frozen_threshold_decides_defect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path, threshold=0.25)
    assert _run(rig, Reader()) == 0
    assert "| C | 2 of 3 |" in capsys.readouterr().out


def test_without_a_freeze_file_no_clip_is_opened(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = make_rig(tmp_path)
    reader = Reader()
    assert _run(rig, reader) == 2
    assert reader.read == []
    assert "freeze" in capsys.readouterr().out


@pytest.mark.parametrize("changed", ["threshold.yml", "lane.yml", "split.yml", "model.pt"])
def test_a_file_changed_after_the_freeze_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], changed: str
) -> None:
    rig = _frozen(tmp_path)
    target = {"threshold.yml": rig.threshold, "lane.yml": rig.lane, "split.yml": rig.split,
              "model.pt": rig.model_file}[changed]
    target.write_bytes(target.read_bytes() + b"\n# edited\n")
    reader = Reader()
    assert _run(rig, reader) == 2
    assert reader.read == []
    assert changed in capsys.readouterr().out


def test_a_different_commit_is_refused_naming_both(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path)
    reader = Reader()
    assert _run(rig, reader, head="b" * 40) == 2
    out = capsys.readouterr().out
    assert reader.read == []
    assert "aaaaaaaa" in out
    assert "bbbbbbbb" in out


def test_a_test_clip_changed_since_the_split_is_refused(tmp_path: Path) -> None:
    rig = _frozen(tmp_path)
    (rig.footage / "E-1.mkv").write_bytes(b"swapped")
    reader = Reader()
    assert _run(rig, reader) == 2
    assert reader.read == []


def test_the_first_scoring_is_not_marked_and_the_next_ones_are(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path)
    _run(rig, Reader())
    first = capsys.readouterr().out.splitlines()[0]
    _run(rig, Reader())
    second = capsys.readouterr().out.splitlines()[0]
    _run(rig, Reader())
    third = capsys.readouterr().out.splitlines()[0]
    assert (first.startswith("RESCORED"), second, third) == (
        False, "RESCORED (run 2)", "RESCORED (run 3)")


def test_each_scoring_is_recorded_beside_the_freeze_file(tmp_path: Path) -> None:
    rig = _frozen(tmp_path)
    _run(rig, Reader())
    _run(rig, Reader())
    record = json.loads((tmp_path / "freeze.yml.scored.json").read_text())
    assert [sorted(r) for r in record["runs"]] == [["scored_utc", "table_sha256"]] * 2
    assert record["runs"][0]["scored_utc"] == "2026-11-16T12:00:00+00:00"
    assert record["runs"][0]["table_sha256"] == record["runs"][1]["table_sha256"]


def test_a_refused_scoring_is_not_recorded(tmp_path: Path) -> None:
    rig = _frozen(tmp_path)
    _run(rig, Reader(), head="b" * 40)
    assert not (tmp_path / "freeze.yml.scored.json").exists()


def test_the_table_goes_to_out_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path)
    _run(rig, Reader(), "--out", str(tmp_path / "table.md"))
    assert (tmp_path / "table.md").read_text().rstrip("\n") == capsys.readouterr().out.rstrip("\n")


def test_a_scoring_with_no_labelled_pass_in_a_test_clip_is_refused(tmp_path: Path) -> None:
    rig = _frozen(tmp_path)
    rig.labels.write_text(json.dumps({"version": 1, "source": "L", "frames": {}, "cups": [
        {"video": "T-1.mkv", "order": 1, "state": "full", "frame": 50}]}))
    reader = Reader()
    assert _run(rig, reader) == 2
    assert reader.read == []


def test_method_a_is_added_with_its_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path)
    seen: list[str] = []

    def judge(config: Path, video: Path) -> tuple[list[Judged], list[int]]:
        seen.append(f"{config.name} {video.name}")
        return ([Judged(100, Label.PASS), Judged(200, Label.DEFECT), Judged(300, Label.DEFECT),
                 Judged(400, Label.PASS), Judged(500, Label.PASS),
                 Judged(700, Label.DEFECT)], [])

    assert _run(rig, Reader(), "--a-config", "a.yml", judge_a=judge) == 0
    out = capsys.readouterr().out
    assert seen == ["a.yml E-1.mkv"]
    assert "| A | 3 of 3 | 0.3684 | 0 of 3 | 0.6316 |" in out
    assert "| 0 of 2 | 0.7764 |" in out
    assert C_ROW in out


def test_method_b_needs_the_freeze_to_name_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path)
    _run(rig, Reader())
    assert "judge only" not in capsys.readouterr().out


def test_method_b_judges_the_labelled_full_and_empty_passes_at_its_frozen_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = _frozen(tmp_path, 0.5, "--patchcore", "patchcore-mochi-v1-0123456789ab",
                  "--patchcore-threshold", "0.4")
    asked: list[tuple[str, list[int]]] = []

    def b_scores(video: Path, cups: Sequence[Cup]) -> list[float]:
        asked.append((video.name, [cup.frame for cup in cups]))
        return [B_SCORES[cup.frame] for cup in cups]

    assert _run(rig, Reader(), score_b=b_scores) == 0
    out = capsys.readouterr().out
    assert asked == [("E-1.mkv", [100, 200, 300, 400, 500, 700])]
    # above 0.4: empty@300 and @700 -> 2 of 3 caught; full@500 -> 1 of 3 flagged
    assert "| B (judge only) | 2 of 3 | 0.1354 | 1 of 3 | 0.8646 | 0 of 3 | 0.6316 |" in out
    assert "gaps not scored" in out
