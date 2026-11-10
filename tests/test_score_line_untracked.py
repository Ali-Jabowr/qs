"""`qs score-line` counts untracked records as their own line (#134).

The video, at 8 fps, frame by frame (a crossing is nine steps and four frames of steel):

    0–2    steel
    3–15   a warm disc crosses, no rim drawn       NO_VERDICT(untracked), judged at 10
    16–28  a pink cup crosses                      judged at 19: ГОДЕН
    29–31  steel

One pink cup is labelled, at frame 19. The untracked record belongs to no cup, so it is neither
a pair nor a record paired with no cup.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qs.__main__ import run
from untracked_fixtures import VIDEO, config, cup_crossing, disc_crossing, steel, write_video

UNTRACKED_LINE = "- untracked passages (НЕТ РЕШЕНИЯ, no cup tracked): 1"


def _score(root: Path, frames: dict[str, str] | None = None, **line: object) -> None:
    footage = root / "footage"
    footage.mkdir()
    write_video(footage / VIDEO, steel(3) + disc_crossing() + cup_crossing("pink") + steel(3))
    document = {
        "version": 1,
        "source": "line",
        "frames": {frame_id: {"label": label, "flavour": None}
                   for frame_id, label in (frames or {}).items()},
        "cups": [{"video": VIDEO, "order": 1, "state": "full_pink", "frame": 19}],
    }
    labels = root / "top.json"
    labels.write_text(json.dumps(document), encoding="utf-8")
    code = run(["score-line", "--config", str(config(root, **line)), "--footage", str(footage),
                "--labels", f"top={labels}", "--window", "5"])
    assert code == 0


def test_the_untracked_record_is_counted_on_its_own_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _score(tmp_path)

    out = capsys.readouterr().out
    assert UNTRACKED_LINE in out


def test_the_untracked_record_is_not_taken_for_a_cup_or_an_unpaired_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _score(tmp_path)

    out = capsys.readouterr().out
    assert "- full cups seen but not judged: 0 of 1 = 0.0%, upper bound 95.0%" in out
    assert "- false БРАК on full cups: 0 of 1 = 0.0%, upper bound 95.0%" in out
    assert "- records paired with no cup: 0" in out


def test_a_config_without_the_window_prints_no_untracked_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _score(tmp_path, untracked_window=None)

    assert "untracked" not in capsys.readouterr().out


def test_an_untracked_record_on_a_hand_frame_is_counted_on_its_own_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _score(tmp_path, {f"{VIDEO}#10": "hand"})

    out = capsys.readouterr().out
    assert "- untracked passages on hand frames: 1" in out
    assert "- hand frames labelled: 1; judged: 0 (БРАК 0, ГОДЕН 0, НЕТ РЕШЕНИЯ 0)" in out


def test_no_untracked_record_on_a_hand_frame_counts_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _score(tmp_path, {f"{VIDEO}#30": "hand"})

    assert "- untracked passages on hand frames: 0" in capsys.readouterr().out
