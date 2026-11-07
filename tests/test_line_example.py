"""The shipped line example config is one `qs score-line` accepts (#105, review L1 spec #7).

The video is three frames of plain steel at the example's own frame size, and the labels count one
cup on it, so the run has one never-seen cup to report.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.__main__ import run

LINE_EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-line.example.yml"
VIDEO = "line.avi"


def _footage(root: Path) -> Path:
    footage = root / "footage"
    footage.mkdir()
    writer = cv2.VideoWriter(
        str(footage / VIDEO), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (1920, 1080)
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((1080, 1920, 3), 90, dtype=np.uint8))
    writer.release()
    return footage


def _labels(root: Path) -> Path:
    path = root / "top.json"
    path.write_text(json.dumps({
        "version": 1,
        "source": "line",
        "frames": {},
        "cups": [{"video": VIDEO, "order": 1, "state": "empty", "frame": 1}],
    }), encoding="utf-8")
    return path


def test_score_line_accepts_the_line_example_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(["score-line", "--config", str(LINE_EXAMPLE), "--footage", str(_footage(tmp_path)),
                "--labels", f"top={_labels(tmp_path)}"])
    assert code == 0, capsys.readouterr().out
