"""`qs unit-cut`: overlay pictures of the cut-out on the crop of every full pass (#174).

The drawn clip is 300 x 260 at 10 fps, 120 frames, each the drawn crop of `test_lane_unit`
placed in the lane window [10, 10, 276, 230]. `T-1.mkv` has three passes: full@50, empty@82,
gap@114.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from qs.__main__ import build_parser, run
from qs.lane.split import Split, file_sha256, write_split
from test_lane_unit import GOOD, PINK, draw_crop

FRAMES = 120
PASSES = [("full", 50), ("empty", 82), ("gap", 114)]


def _write_clip(path: Path, empty_from: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (300, 260))
    assert writer.isOpened()
    for n in range(FRAMES):
        frame = np.full((260, 300, 3), 100, dtype=np.uint8)
        frame[10:240, 10:286] = draw_crop(None if n >= empty_from else PINK)
        writer.write(frame)
    writer.release()


def _setup(
    root: Path, passes: list[tuple[str, int]] = PASSES, empty_from: int = FRAMES
) -> dict[str, Path]:
    footage = root / "footage"
    footage.mkdir()
    _write_clip(footage / "T-1.mkv", empty_from)
    cups = [{"video": "T-1.mkv", "order": i + 1, "state": s, "frame": f}
            for i, (s, f) in enumerate(passes)]
    labels = root / "L.json"
    labels.write_text(json.dumps({"version": 1, "source": "L", "frames": {}, "cups": cups}))
    lane = root / "lane.yml"
    lane.write_text(yaml.safe_dump({"version": 1, "window": [10, 10, 276, 230], "fps": 10}))
    unit = root / "unit.yml"
    unit.write_text(GOOD)
    split = root / "split.yml"
    write_split(Split({"T-1.mkv": ("train", file_sha256(footage / "T-1.mkv"))}), split)
    return {"footage": footage, "labels": labels, "lane": lane, "unit": unit, "split": split}


def _argv(root: Path, paths: dict[str, Path], block: str = "train", *extra: str) -> list[str]:
    return ["unit-cut", "--lane", str(paths["lane"]), "--unit", str(paths["unit"]),
            "--labels", str(paths["labels"]), "--footage", str(paths["footage"]),
            "--split", str(paths["split"]), "--block", block, "--out", str(root / "out"), *extra]


def test_one_full_pass_gives_one_overlay_and_is_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    assert run(_argv(tmp_path, paths)) == 0
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["T-1.mkv-50.png"]
    assert capsys.readouterr().out.splitlines() == ["cut 1 of 1"]
    overlay = cv2.imread(str(tmp_path / "out" / "T-1.mkv-50.png"))
    assert overlay.shape == (230, 276, 3)
    assert (overlay == (0, 255, 0)).all(axis=2).any()


def test_a_pass_with_no_unit_is_counted_by_its_fault_and_gets_no_overlay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path, [("full", 50), ("full_pink", 110)], empty_from=100)
    assert run(_argv(tmp_path, paths)) == 0
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["T-1.mkv-50.png"]
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "cut 1 of 2"
    assert out[1] in {"no unit: 1", "unit size out of range: 1"}


def test_limit_keeps_the_first_and_last_of_evenly_spaced_passes(tmp_path: Path) -> None:
    paths = _setup(tmp_path, [("full", 20), ("full", 50), ("full_white", 80)])
    assert run(_argv(tmp_path, paths, "train", "--limit", "2")) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "T-1.mkv-20.png", "T-1.mkv-80.png"]


def test_the_test_block_is_not_offered(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    with pytest.raises(SystemExit):
        build_parser().parse_args(_argv(tmp_path, paths, "test"))


def test_a_clip_that_is_not_in_the_block_is_not_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    assert run(_argv(tmp_path, paths, "calibrate")) == 1
    assert capsys.readouterr().out.strip() == "refused: no full pass in the calibrate clips"


def test_a_changed_clip_is_refused_before_any_frame_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    with (paths["footage"] / "T-1.mkv").open("ab") as clip:
        clip.write(b"x")
    assert run(_argv(tmp_path, paths)) == 2
    assert "changed since the split" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_a_bad_unit_file_stops_the_run_naming_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    paths["unit"].write_text(GOOD.replace("reach_px: 70\n", ""))
    assert run(_argv(tmp_path, paths)) == 2
    assert "reach_px" in capsys.readouterr().out
