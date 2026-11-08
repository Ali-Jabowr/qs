"""`qs score-line` counts a hand frame in a video that has no counted cup (review L1, spec #4).

The footage of `test_score_line` is copied to a second file, `other.avi`, that no cup names. A
hand is labelled there on frame 6, where the station judges the first (empty) cup: БРАК. A second
hand is labelled on frame 30 of `line.avi`, which no record judges.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from qs.__main__ import run
from test_score_line import CUPS, VIDEO, _config, _footage, _labels

OTHER = "other.avi"


def test_a_hand_in_a_video_with_no_counted_cup_is_counted_and_judged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    footage = _footage(tmp_path)
    shutil.copyfile(footage / VIDEO, footage / OTHER)
    frames = {f"{OTHER}#6": "hand", f"{VIDEO}#30": "hand"}
    labels = _labels(tmp_path, "top.json", CUPS, frames)

    code = run(["score-line", "--config", str(_config(tmp_path)), "--footage", str(footage),
                "--window", "5", "--labels", f"top={labels}"])

    assert code == 0
    assert ("- hand frames labelled: 2; judged: 1 (БРАК 1, ГОДЕН 0, НЕТ РЕШЕНИЯ 0)"
            in capsys.readouterr().out)
