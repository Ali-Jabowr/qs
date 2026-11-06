"""`qs score-line` on a drawn video and hand-made labels (#105). The video is lossless, so every
frame the station reads is the frame drawn. Every expected count is worked by hand from the
frame plan below; the bounds are the exact one-sided 95% Clopper-Pearson values for those counts.

The drawn video, at 8 fps, frame by frame. In these drawings a rim is wholly in the zone from x 90,
the third step of a crossing, so a crossing that starts at frame s fires at s + 2 and judges the
next frame, s + 3:

    0–2    steel
    3–15   an empty cup crosses                      judged at 6:  БРАК
    16–28  a full (pale) cup crosses                 judged at 19: ГОДЕН
    29–41  a full cup crosses, never wholly in zone  judged when lost: НЕТ РЕШЕНИЯ (placement)
    42–54  an empty cup crosses                      judged at 45: БРАК
    55–74  steel
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from qs.__main__ import run
from station_fixtures import config_mapping
from test_cup import drawn_frame

FPS = 8.0
PATH_X = (10, 50, 90, 130, 170, 210, 250, 290, 330)
OFF_THE_ZONE_Y = 65
VIDEO = "line.avi"


def _crossing(contents: str, *, y: int = 120) -> list[np.ndarray]:
    return [drawn_frame(contents, centre=(x, y), seed=x) for x in PATH_X] + _steel(4)


def _steel(count: int) -> list[np.ndarray]:
    return [drawn_frame("none", seed=100 + index) for index in range(count)]


def _write_video(path: Path, frames: Sequence[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (320, 240))
    assert writer.isOpened()
    for pixels in frames:
        writer.write(pixels)
    writer.release()


def _footage(root: Path) -> Path:
    footage = root / "footage"
    footage.mkdir()
    _write_video(
        footage / VIDEO,
        _steel(3) + _crossing("empty") + _crossing("pale")
        + _crossing("pale", y=OFF_THE_ZONE_Y) + _crossing("empty") + _steel(20),
    )
    return footage


def _config(root: Path) -> Path:
    data = config_mapping(root, inspector_type="line")
    data["trigger"]["type"] = "cup"
    data["inspector"]["checks"] = ["empty_cup"]
    data["inspector"]["line"] = {
        "zone": [20, 30, 240, 180],
        "rim_axis_px": [50, 140],
        "blob_share_min": 0.3,
        "direction": "right",
        "fps": FPS,
    }
    path = root / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _labels(root: Path, name: str, cups: Sequence[tuple[str, int]],
            frames: dict[str, str] | None = None) -> Path:
    document = {
        "version": 1,
        "source": "line",
        "frames": {
            frame_id: {"label": label, "flavour": None}
            for frame_id, label in (frames or {}).items()
        },
        "cups": [
            {"video": VIDEO, "order": order, "state": state, "frame": frame}
            for order, (state, frame) in enumerate(cups, start=1)
        ],
    }
    path = root / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# Each cup is labelled a frame or two from where it is judged, as a person pressing a key would.
# The fourth crossing is drawn empty but labelled full and pink: to the scorer it is a full cup
# the station called БРАК. The two cups at 62 and 70 cross where the video shows only steel, so
# the station never sees them.
CUPS = [("empty", 7), ("full_white", 18), ("full", 39), ("full_pink", 46),
        ("full", 62), ("empty", 70)]


def _run(root: Path, *labels: str, window: int = 5) -> None:
    args = ["score-line", "--config", str(_config(root)), "--footage", str(_footage(root)),
            "--window", str(window)]
    for item in labels:
        args += ["--labels", item]
    assert run(args) == 0


def test_the_counts_for_one_position_are_the_hand_worked_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = _labels(tmp_path, "top.json", CUPS)

    _run(tmp_path, f"top={labels}")

    out = capsys.readouterr().out
    assert "empty cups caught: 1 of 2 = 50.0%, lower bound 2.5%" in out
    assert "empty cups never seen: 1 of 2 = 50.0%, upper bound 97.5%" in out
    assert "full cups never seen: 1 of 4 = 25.0%, upper bound 75.1%" in out
    assert "empty cups seen but not judged: 0 of 2 = 0.0%, upper bound 77.6%" in out
    assert "full cups seen but not judged: 1 of 4 = 25.0%, upper bound 75.1%" in out
    assert "false БРАК on full cups: 1 of 4 = 25.0%, upper bound 75.1%" in out


def test_hand_frames_are_counted_with_the_verdicts_of_those_judged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A hand is labelled on frame 6, the frame the first cup is judged on, and on frame 30, which
    # no record judges. A unit label on frame 19 is not a hand.
    frames = {f"{VIDEO}#6": "hand", f"{VIDEO}#30": "hand", f"{VIDEO}#19": "unit"}
    labels = _labels(tmp_path, "top.json", CUPS, frames)

    _run(tmp_path, f"top={labels}")

    out = capsys.readouterr().out
    assert "- hand frames labelled: 2; judged: 1 (БРАК 1, ГОДЕН 0, НЕТ РЕШЕНИЯ 0)" in out


def test_the_output_is_stamped_indicative_states_the_camera_limits_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, footage = _config(tmp_path), _footage(tmp_path)
    labels = _labels(tmp_path, "top.json", CUPS)
    before = sorted(tmp_path.rglob("*"))

    code = run(["score-line", "--config", str(config), "--footage", str(footage),
                "--labels", f"top={labels}"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("# INDICATIVE — line footage scored against hand-counted cups\n")
    assert "not Gate A and not a client claim" in out
    assert "The camera sees the top of each cup only" in out
    assert sorted(tmp_path.rglob("*")) == before


def test_each_camera_position_is_scored_on_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The side labels count only the first two cups: their span, 2–23, holds the records at 6 and
    # 19, and none of the others.
    top = _labels(tmp_path, "top.json", CUPS)
    side = _labels(tmp_path, "side.json", [("empty", 7), ("full", 18)])

    _run(tmp_path, f"top={top}", f"side={side}")

    out = capsys.readouterr().out
    assert out.count("## position ") == 2
    top_part, side_part = out.split("## position side")
    assert "## position top" in top_part
    assert "- empty cups caught: 1 of 2 = 50.0%, lower bound 2.5%" in top_part
    assert "- empty cups caught: 1 of 1 = 100.0%, lower bound 5.0%" in side_part
    assert "- false БРАК on full cups: 0 of 1 = 0.0%, upper bound 95.0%" in side_part
    assert "- records paired with no cup: 0" in side_part
